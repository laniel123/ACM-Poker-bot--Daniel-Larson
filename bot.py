# === ACM.Dev — PredatorBot v2 (Exploit-leaning, legal-raise safe) ===
# Interface: def bet(state) -> int
# Uses your helpers.py definitions exactly as requested.

from __future__ import annotations
from typing import List, Tuple
import numpy as np

# ---- helpers API (from your provided helpers.py) ----
from helpers import (
    get_player_list, amount_to_call, get_my_pots, get_best_hand_from,
    fold, check, call, all_in, min_raise, is_valid_bet, get_round_name,
    my_stack, opp_stacks, legal_actions, total_pot, parse_card, deck_remaining
)

# --------------------------
# Card parsing & evaluator
# --------------------------

_SUITS = {"s","h","d","c"}
_RANKS = "23456789tjqka"
_VAL = {"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"t":10,"j":11,"q":12,"k":13,"a":14}

def _to_tuple(cs: str) -> Tuple[int,str]:
    return (_VAL[cs[0].lower()], cs[1].lower())

def _is_straight(vals_sorted_desc: List[int]) -> Tuple[bool, int]:
    seen = []
    last = None
    for v in vals_sorted_desc:
        if v != last:
            seen.append(v)
            last = v
    if 14 in seen:
        seen.append(1)
    cnt = 1
    best_high = 0
    for i in range(1, len(seen)):
        if seen[i-1] - 1 == seen[i]:
            cnt += 1
            if cnt >= 5:
                best_high = max(best_high, seen[i-4])
        else:
            cnt = 1
    return (best_high > 0, best_high)

def _hand_rank_7(cards: List[Tuple[int,str]]) -> Tuple[int, List[int]]:
    vals = [v for v,_ in cards]
    suits = [s for _,s in cards]
    vals_sorted = sorted(vals, reverse=True)

    counts = {}
    for v in vals:
        counts[v] = counts.get(v,0)+1
    suit_map = {}
    for v,s in cards:
        suit_map.setdefault(s, []).append(v)

    flush_suit = None
    flush_vals = None
    for s, vs in suit_map.items():
        if len(vs) >= 5:
            flush_suit = s
            flush_vals = sorted(vs, reverse=True)
            break

    is_stra, stra_high = _is_straight(vals_sorted)

    if flush_suit is not None:
        fs_vals = sorted(suit_map[flush_suit], reverse=True)
        is_sf, sf_high = _is_straight(fs_vals)
        if is_sf:
            return (8, [sf_high])

    by_count = {}
    for v,c in counts.items():
        by_count.setdefault(c, []).append(v)
    for c in by_count:
        by_count[c].sort(reverse=True)

    if 4 in by_count:
        quad = by_count[4][0]
        kickers = [x for x in vals_sorted if x != quad]
        return (7, [quad, kickers[0]])

    trips_list = by_count.get(3, [])
    pairs_list = by_count.get(2, [])

    if trips_list:
        trips = trips_list[0]
        rest_trips = trips_list[1:] if len(trips_list)>1 else []
        if rest_trips:
            return (6, [trips, rest_trips[0]])
        if pairs_list:
            return (6, [trips, pairs_list[0]])

    if flush_suit is not None:
        return (5, flush_vals[:5])

    if is_stra:
        return (4, [stra_high])

    if trips_list:
        trips = trips_list[0]
        kickers = [x for x in vals_sorted if x != trips][:2]
        return (3, [trips]+kickers)

    if len(pairs_list) >= 2:
        p1, p2 = pairs_list[:2]
        hp, lp = max(p1,p2), min(p1,p2)
        kicker = [x for x in vals_sorted if x!=hp and x!=lp][0]
        return (2, [hp, lp, kicker])

    if len(pairs_list) == 1:
        p = pairs_list[0]
        kickers = [x for x in vals_sorted if x != p][:3]
        return (1, [p]+kickers)

    return (0, vals_sorted[:5])

def _cmp_rank(a: Tuple[int,List[int]], b: Tuple[int,List[int]]) -> int:
    if a[0] != b[0]:
        return 1 if a[0] > b[0] else -1
    la, lb = a[1], b[1]
    for x,y in zip(la,lb):
        if x != y:
            return 1 if x>y else -1
    if len(la)!=len(lb):
        return 1 if len(la)>len(lb) else -1
    return 0

# --------------------------
# Equity via Monte Carlo
# --------------------------

def _estimate_equity(hole: List[str], board: List[str], num_opps: int, iterations: int = 280, rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    our_hole = [_to_tuple(c) for c in hole]
    board_parsed = [_to_tuple(c) for c in board]
    known = set(hole + board)

    # build deck minus known
    deck = []
    for r in _RANKS:
        for s in _SUITS:
            cs = r+s
            if cs in known:
                continue
            deck.append((_VAL[r], s))
    deck_arr = np.array(deck, dtype=object)

    need_board = max(0, 5 - len(board_parsed))
    wins = ties = total = 0

    for _ in range(iterations):
        draw_cnt = 2*num_opps + need_board
        if draw_cnt > len(deck_arr):
            break
        idx = rng.choice(len(deck_arr), size=draw_cnt, replace=False)
        draw = deck_arr[idx]

        # opponents
        opps = []
        p = 0
        for _o in range(num_opps):
            opps.append([draw[p], draw[p+1]])
            p += 2
        sim_board = list(board_parsed)
        for _ in range(need_board):
            sim_board.append(draw[p]); p += 1

        our7 = our_hole + sim_board
        our_rank = _hand_rank_7(our7)

        better = False
        equal_cnt = 0
        for oh in opps:
            their7 = list(oh) + sim_board
            rr = _hand_rank_7(their7)
            c = _cmp_rank(our_rank, rr)
            if c < 0:
                better = True
                equal_cnt = 0
                break
            elif c == 0:
                equal_cnt += 1
        if better:
            pass
        else:
            if equal_cnt > 0:
                ties += 1
            else:
                wins += 1
        total += 1

    if total == 0: return 0.0
    return (wins + 0.5*ties) / total

# --------------------------
# Preflop tiers (tighter & clearer)
# --------------------------

def _preflop_tier(hole: List[str]) -> str:
    a, b = hole
    va, sa = _to_tuple(a)
    vb, sb = _to_tuple(b)
    hi, lo = max(va, vb), min(va, vb)
    suited = (sa == sb)
    gap = hi - lo
    # pairs
    if va == vb:
        if hi >= 13: return 'premium'     # AA, KK
        if hi >= 11: return 'strong'      # QQ, JJ
        if hi >= 8:  return 'good'        # TT-88
        return 'setmine'                   # 77-22

    # AK/AQ/AJ
    if hi == 14 and lo >= 11:
        return 'strong' if suited or lo==13 else 'good'  # AK, AQ, AJ

    # broadways
    if hi >= 13 and lo >= 10:
        return 'good' if suited else 'ok'  # KQ/KJ/QJ

    # Ax suited down to A5s
    if hi == 14 and suited and lo >= 5:
        return 'ok'

    # suited connectors 98s-54s
    if suited and gap <= 4 and hi >= 9 and lo >= 4:
        return 'ok'

    return 'trash'

# --------------------------
# Sizing helpers (legalize any target)
# --------------------------

def _pot_now(state) -> int:
    pot_val = total_pot(state)
    # safety if engine hasn’t pushed pot yet: derive from bets
    try:
        pot_val = max(pot_val, sum(max(0,b) for b in state.bet_money))
    except Exception:
        pass
    return max(pot_val, 0)

def _legal_bet_closest(state, target_amount: int) -> int:
    """
    We return an action amount that the engine accepts, nearest to target.
    - If no bet yet, min_raise(state) acts like a minimum open.
    - If facing a bet, min_raise(state) is the minimum raise *amount*.
    - If raising is impossible (short), fall back to call or all-in if valid.
    """
    acts = legal_actions(state)  # already filtered by is_valid_bet
    if not acts:  # fail-safe
        return check()

    # If target equals a listed action, great
    if target_amount in acts:
        return target_amount

    # If we wanted to bet/raise but only call/check/all_in exist, pick best EV-ish:
    # choose the maximum valid <= target; else the minimum valid >= target.
    lowers = [a for a in acts if a >= 0 and a <= target_amount]
    uppers = [a for a in acts if a >= 0 and a > target_amount]

    if lowers:
        return max(lowers)
    if uppers:
        return min(uppers)

    # if only -1 or 0 exist
    if 0 in acts:
        return 0
    if -1 in acts:
        return -1
    # last resort
    return acts[0]

# --------------------------
# Main decision policy
# --------------------------

def _decide(state) -> int:
    idx = state.index_to_action
    bb = state.big_blind
    players = get_player_list(state)
    n = len(players)

    to_call = amount_to_call(state)
    can_check_now = (to_call == 0)
    stack = my_stack(state)

    hole = list(state.player_cards)[:2]
    board = list(state.community_cards)
    preflop = (len(board) == 0)
    pot = _pot_now(state)
    rng = np.random.default_rng()

    # always fold if we literally cannot call
    if to_call > stack:
        return fold()

    # ---------- PRE-FLOP ----------
    if preflop:
        tier = _preflop_tier(hole)
        live_opp = sum(1 for b in state.bet_money if b >= 0) - 1

        # facing action
        if not can_check_now:
            # Pot odds gate for non-premiums
            if tier in ('premium','strong'):
                return call(state)
            if tier in ('good','ok','setmine'):
                cheap = (to_call <= max(bb, pot // 10))
                return call(state) if cheap else fold()
            return fold()

        # no bet yet → open-raise by tier
        if tier == 'premium':
            # 3.0-3.5bb open
            mult = 3.5 if rng.random()<0.35 else 3.0
            target = int(max(bb, round(mult*bb)))
            return _legal_bet_closest(state, target)
        if tier == 'strong':
            mult = 3.0 if rng.random()<0.5 else 2.5
            target = int(max(bb, round(mult*bb)))
            return _legal_bet_closest(state, target)
        if tier == 'good':
            mult = 2.5 if rng.random()<0.7 else 2.0
            target = int(max(bb, round(mult*bb)))
            # some chance to just check to pot control multiway
            if rng.random() < 0.15:
                if is_valid_bet(state, 0): return check()
            return _legal_bet_closest(state, target)
        if tier in ('ok','setmine'):
            # mostly check; occasional steal
            if rng.random() < 0.18:
                target = int(max(bb, round(2.0*bb)))
                return _legal_bet_closest(state, target)
            return check()
        return check()

    # ---------- POST-FLOP ----------
    # fast equity estimate vs remaining opponents
    live_opp = max(1, sum(1 for b in state.bet_money if b >= 0) - 1)
    base_iters = 260
    iters = int(max(120, base_iters - 40*max(0, live_opp-1)))
    eq = _estimate_equity(hole, board, live_opp, iterations=iters, rng=rng)

    # simple texture flags
    parsed_board = [_to_tuple(c) for c in board]
    suit_counts = {}
    uniq_vals = set()
    for v,s in parsed_board:
        suit_counts[s] = suit_counts.get(s,0)+1
        uniq_vals.add(v)
    flushed = any(c>=4 for c in suit_counts.values())
    connected = (len(uniq_vals)>=4 and (max(uniq_vals)-min(uniq_vals) <= 6))

    # FACING A BET → use pot-odds gate + semi-bluff sprinkle
    if not can_check_now:
        eff_pot = pot + to_call
        pot_odds = (to_call/eff_pot) if eff_pot>0 else 1.0
        # Call when equity covers pot odds + small buffer
        if eq >= pot_odds + 0.025:
            return call(state)
        # Occasionally convert to raise as a semi-bluff on scary/connected boards if price small
        semi = (not flushed) and connected and (rng.random() < 0.08) and (to_call <= stack)
        if semi:
            # raise to ~ 2.5x current bet size (approx via pot fraction)
            target = int(max(bb, round(0.65 * max(pot, 4*bb))))
            action = _legal_bet_closest(state, target)
            if action > 0:
                return action
        return fold()

    # We can bet (no one bet yet this street)
    # Value thresholds and bluff freq adapted to board texture
    if eq >= 0.75:
        # big value ~80-100% pot
        target = int(max(3*bb, round(0.9 * max(pot, 4*bb))))
        return _legal_bet_closest(state, target)
    if eq >= 0.6:
        # medium value/protection ~60-70% pot
        target = int(max(2*bb, round(0.65 * max(pot, 4*bb))))
        return _legal_bet_closest(state, target)

    # bluff / c-bet logic: on coordinated but not flushed boards, stab small sometimes
    bluff_ok = (not flushed) and connected
    if bluff_ok and np.random.random() < 0.22:
        target = int(max(bb, round(0.45 * max(pot, 3*bb))))
        action = _legal_bet_closest(state, target)
        if action > 0:
            return action

    # otherwise, take the free card
    return check()

# --------------------------
# Public entrypoint
# --------------------------

def bet(state) -> int:
    try:
        return int(_decide(state))
    except Exception:
        # fail-safe
        return fold()