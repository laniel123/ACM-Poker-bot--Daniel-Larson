# ACM.Dev Mann vs Machine — Balanced Exploit Bot (single file)
# - Conservative→Balanced TAG with position & stack awareness
# - Short-stack push/fold (~Nash-ish) + late-position steals
# - Postflop Monte Carlo equity with pot-odds gating & c-bets
# - Min-raise safe, integer actions, no external libs beyond numpy
# - No prints, no global I/O. Works in engines that pass a dict GameState.

from __future__ import annotations
from typing import List, Tuple
import numpy as np

# --------------------------
# Helpers over GameState-like dict
# --------------------------

def active_player_indexes(bet_money: List[int]) -> List[int]:
    return [i for i, b in enumerate(bet_money) if b >= 0]

def current_max_bet(bet_money: List[int]) -> int:
    act = [b for b in bet_money if b >= 0]
    return max(act) if act else 0

def amount_to_call(state) -> int:
    mx = current_max_bet(state["bet_money"])
    my = state["bet_money"][state["index_to_action"]]
    return max(0, mx - max(0, my))

def can_check(state) -> bool:
    return amount_to_call(state) == 0

def my_stack(state) -> int:
    return int(state["held_money"][state["index_to_action"]])

def live_opponents(state) -> int:
    return max(0, len(active_player_indexes(state["bet_money"])) - 1)

def estimated_pot_value(state) -> int:
    pots = state.get("pots", [])
    try:
        ps = sum(int(p.get("value", 0)) for p in pots)
    except Exception:
        ps = 0
    # Some engines don't finalize pot into pots[] until showdown → fallback:
    return max(ps, sum(max(0, b) for b in state["bet_money"]))

def my_position(state) -> int:
    # Rough relative seat index (later number -> later position)
    return (int(state["index_to_action"]) - int(state["index_of_small_blind"])) % len(state["players"])

def min_raise(state) -> int:
    """
    Minimum *amount to put in this action* to complete a legal raise
    relative to my current committed chips this street.
    """
    bets = state["bet_money"]
    current_bet = current_max_bet(bets)
    my_idx = state["index_to_action"]
    my_bet = max(0, int(bets[my_idx]))
    # Highest lower bet among active players:
    lowers = [b for b in bets if 0 <= b < current_bet]
    if lowers:
        prev_max = max(lowers)
        last_raise = current_bet - prev_max
    else:
        last_raise = int(state["big_blind"])
    min_raise_size = max(last_raise, int(state["big_blind"]))
    target_total = current_bet + min_raise_size
    return max(0, target_total - my_bet)

def legalize_amount(state, amount: int) -> int:
    """Clamp to a legal action in {-1, 0, call, min_raise..all_in}."""
    if amount == -1:
        return -1
    to_call = amount_to_call(state)
    my_idx = state["index_to_action"]
    my_bet = max(0, int(state["bet_money"][my_idx]))
    my_hold = int(state["held_money"][my_idx])
    current_b = current_max_bet(state["bet_money"])

    if amount == 0:
        return 0 if to_call == 0 else to_call

    if amount < 0:
        return -1

    # Interpret "amount" as the chips we add now (engine convention)
    add = int(amount)
    total_post = my_bet + add

    # All-in allowed
    if add >= my_hold:
        return my_hold

    # Calls must reach current bet
    if total_post < current_b:
        return max(0, to_call)

    # Raise case: total_post > current_b
    # Must be >= min-raise
    need_min = min_raise(state)
    if add < need_min:
        # If we intended a raise but it's too small, default to call
        return max(0, to_call)
    return add

# --------------------------
# Cards & Hand Evaluator (5/7)
# --------------------------

_RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'t':10,'j':11,'q':12,'k':13,'a':14}
_SUITS = ('s','h','d','c')
_VAL_ORDER = '23456789tjqka'

def parse_card(cs: str) -> Tuple[int,str]:
    r, s = cs[0].lower(), cs[1].lower()
    return _RANK_MAP[r], s

def _is_straight(sorted_desc_vals: List[int]) -> Tuple[bool,int]:
    seen = []
    last = None
    for v in sorted_desc_vals:
        if v != last:
            seen.append(v)
            last = v
    if 14 in seen:
        seen.append(1)
    run = 1
    best = 0
    for i in range(1, len(seen)):
        if seen[i-1] - 1 == seen[i]:
            run += 1
            if run >= 5:
                best = max(best, seen[i-4])
        else:
            run = 1
    return (best > 0, best)

def hand_rank_7(cards: List[Tuple[int,str]]) -> Tuple[int,List[int]]:
    vals = [v for v,_ in cards]
    suits = [s for _,s in cards]
    vals_sorted = sorted(vals, reverse=True)

    # suit buckets
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

    # straight flags
    is_stra, stra_high = _is_straight(vals_sorted)
    if flush_suit is not None:
        is_sf, sf_high = _is_straight(sorted(suit_map[flush_suit], reverse=True))
        if is_sf:
            return (8, [sf_high])

    # counts
    cnt = {}
    for v in vals:
        cnt[v] = cnt.get(v,0)+1
    by_count = {}
    for v,c in cnt.items():
        by_count.setdefault(c, []).append(v)
    for c in by_count:
        by_count[c].sort(reverse=True)

    if 4 in by_count:
        quad = by_count[4][0]
        kick = [x for x in vals_sorted if x != quad][0]
        return (7, [quad, kick])

    trips = by_count.get(3, [])
    pairs = by_count.get(2, [])
    if trips:
        t = trips[0]
        if len(trips) > 1:
            return (6, [t, trips[1]])
        if pairs:
            return (6, [t, pairs[0]])

    if flush_suit is not None:
        return (5, flush_vals[:5])

    if is_stra:
        return (4, [stra_high])

    if trips:
        t = trips[0]
        kicks = [x for x in vals_sorted if x != t][:2]
        return (3, [t] + kicks)

    if len(pairs) >= 2:
        p1, p2 = pairs[:2]
        hp, lp = max(p1,p2), min(p1,p2)
        kick = [x for x in vals_sorted if x != hp and x != lp][0]
        return (2, [hp, lp, kick])

    if len(pairs) == 1:
        p = pairs[0]
        kicks = [x for x in vals_sorted if x != p][:3]
        return (1, [p] + kicks)

    return (0, vals_sorted[:5])

def compare_rank(a: Tuple[int,List[int]], b: Tuple[int,List[int]]) -> int:
    if a[0] != b[0]:
        return 1 if a[0] > b[0] else -1
    la, lb = a[1], b[1]
    for x, y in zip(la, lb):
        if x != y:
            return 1 if x > y else -1
    if len(la) != len(lb):
        return 1 if len(la) > len(lb) else -1
    return 0

# --------------------------
# Monte Carlo Equity
# --------------------------

def deck_minus(known: List[str]) -> List[Tuple[int,str]]:
    known_set = set(c.lower() for c in known)
    deck = []
    for r in _VAL_ORDER:
        for s in _SUITS:
            cs = f"{r}{s}"
            if cs in known_set:
                continue
            deck.append(parse_card(cs))
    return deck

def estimate_equity(hole: List[str], board: List[str], num_opps: int, iters: int, rng: np.random.Generator) -> float:
    known = list(hole) + list(board)
    deck = deck_minus(known)
    if not deck:
        return 0.0
    our_hole = [parse_card(c) for c in hole]
    board_parsed = [parse_card(c) for c in board]
    need_board = max(0, 5 - len(board_parsed))

    wins = ties = total = 0
    deck_arr = np.array(deck, dtype=object)

    for _ in range(iters):
        draw_need = 2*num_opps + need_board
        if draw_need > len(deck_arr):
            break
        idx = rng.choice(len(deck_arr), size=draw_need, replace=False)
        draw = deck_arr[idx]
        ptr = 0
        opps = []
        for _o in range(num_opps):
            opps.append([draw[ptr], draw[ptr+1]]); ptr += 2
        sim_board = list(board_parsed)
        for _ in range(need_board):
            sim_board.append(draw[ptr]); ptr += 1

        our7 = our_hole + sim_board
        our_rank = hand_rank_7(our7)

        better = False
        equal = 0
        for oh in opps:
            their = list(oh) + sim_board
            cmpv = compare_rank(our_rank, hand_rank_7(their))
            if cmpv < 0:
                better = True; equal = 0; break
            elif cmpv == 0:
                equal += 1
        if not better:
            if equal > 0:
                ties += 1
            else:
                wins += 1
        total += 1

    if total == 0:
        return 0.0
    return (wins + 0.5*ties) / total

# --------------------------
# Preflop tiers & push/fold
# --------------------------

def preflop_tier(hole: List[str]) -> str:
    a, b = hole
    va, sa = parse_card(a)
    vb, sb = parse_card(b)
    vhi, vlo = max(va, vb), min(va, vb)
    suited = (sa == sb)
    gap = vhi - vlo

    # Pairs
    if va == vb:
        if vhi >= 13: return 'premium'   # AA, KK
        if vhi >= 11: return 'strong'    # QQ, JJ
        if vhi >= 8:  return 'medium'    # TT-88
        return 'medium'                  # 77-22 set-mine

    # Broadways & Ax
    if vhi >= 13 and vlo >= 10:             # KQ/KJ/QJ+
        return 'strong' if suited else 'medium'
    if vhi == 14:                            # Ax
        if suited and vlo >= 10:  return 'strong'   # ATs+
        if vlo >= 13:             return 'strong'   # AKo
        if suited and vlo >= 6:   return 'medium'   # A9s-A6s
        if vlo >= 10:             return 'medium'   # AQo-AJo-ATo
        return 'trash'

    # Suited connectors/gappers
    if suited and gap <= 3 and vhi >= 10 and vlo >= 6:
        return 'medium'
    if suited and gap <= 2 and vhi >= 9 and vlo >= 5:
        return 'medium'
    return 'trash'

def pushfold_should_shove(hole: List[str], eff_bb: float, late_pos: bool) -> bool:
    """
    Simple ~Nash-ish shove thresholds:
      - <=5BB: 22+, A2+, K9s+, QTs+, JTs, KTo+, QJo (late expands a bit)
      - <=8BB: 22+, A2s+, A7o+, KTs+, QTs+, JTs, KQo (late expands: add K9o, T9s)
    """
    a, b = hole
    va, sa = parse_card(a)
    vb, sb = parse_card(b)
    suited = (sa == sb)
    vhi, vlo = max(va, vb), min(va, vb)
    pair = (va == vb)

    def is_ax(): return (vhi == 14)
    def is_kx(): return (vhi == 13)
    def offs():  return not suited

    if eff_bb <= 5.5:
        if pair: return True
        if is_ax(): return True
        if is_kx() and ( (suited and vlo >= 9) or (offs() and vlo >= 10) ):
            return True
        # QTs+, JTs
        if suited and ((vhi==12 and vlo>=10) or (vhi==11 and vlo==10)):
            return True
        if late_pos and suited and vhi>=10 and (vhi - vlo)<=3:
            return True
        return False

    if eff_bb <= 8.5:
        if pair: return True
        if is_ax():
            if suited: return True
            return vlo >= 7  # A7o+
        if vhi==13:  # Kx
            if suited and vlo>=10: return True
        if suited and ((vhi==12 and vlo>=10) or (vhi==11 and vlo==10)):  # QTs+, JTs
            return True
        if (vhi==13 and vlo==12 and offs()):  # KQo
            return True
        if late_pos:
            if (vhi==13 and vlo>=9 and offs()):  # K9o+
                return True
            if suited and vhi>=10 and (vhi - vlo)<=3:  # T9s, 98s, etc.
                return True
        return False

    return False

# --------------------------
# Core decision policy
# --------------------------

def decide_action(state) -> int:
    idx = int(state["index_to_action"])
    bb = int(state["big_blind"])
    players: List[str] = state["players"]
    bet_money: List[int] = [int(x) for x in state["bet_money"]]
    held_money: List[int] = [int(x) for x in state["held_money"]]
    hole = list(state["player_cards"])[:2]
    board = list(state.get("community_cards", []))
    n = len(players)

    # Per-turn RNG seeded from our private info to avoid synchronized behavior
    seed_basis = hash((tuple(hole), tuple(board), idx, int(sum(bet_money))))
    rng = np.random.default_rng(np.int64(seed_basis & 0xFFFFFFFF))

    # Context
    to_call = amount_to_call(state)
    can_chk = (to_call == 0)
    live_opp = live_opponents(state)
    pos = my_position(state)
    pot = estimated_pot_value(state)
    my_bank = my_stack(state)

    # If cannot afford call, fold
    if to_call > my_bank:
        return -1

    preflop = (len(board) == 0)

    # Effective stack in BB (vs biggest live opp)
    opp_max = 0
    for i in range(n):
        if i == idx: continue
        if bet_money[i] >= 0:
            opp_max = max(opp_max, held_money[i])
    eff_stack = min(my_bank, opp_max) if opp_max > 0 else my_bank
    eff_bb = eff_stack / max(1, bb)
    late_pos = (pos >= n-2)  # last two seats are "late"

    # -------- PRE-FLOP --------
    if preflop:
        tier = preflop_tier(hole)
        unopened = (current_max_bet(bet_money) == 0)

        # Short-stack push/fold
        if eff_bb <= 8.5:
            if pushfold_should_shove(hole, eff_bb, late_pos):
                # Shove if unopened or versus raise if call is non-trivial
                # Otherwise take the free check when allowed (in BB)
                if unopened and can_chk:
                    # allow a steal attempt: small open to 2.5-3bb
                    open_mult = 3.0 if rng.random() < 0.6 else 2.5
                    amt = int(max(bb, round(open_mult * bb)))
                    return legalize_amount(state, min(amt, my_bank))
                # prefer shove over flatting short
                return legalize_amount(state, my_bank)
            else:
                # If short and weak: fold to raises, check BB when free
                return 0 if can_chk else -1

        # Normal stacks:
        if unopened and can_chk:
            # Opening ranges by tier + position
            if tier == 'premium':
                open_mult = 3.0 if rng.random() < 0.6 else 2.5
                amt = int(max(bb, round(open_mult * bb)))
                return legalize_amount(state, min(amt, my_bank))
            if tier == 'strong':
                open_mult = 2.5 if rng.random() < 0.7 else 2.0
                amt = int(max(bb, round(open_mult * bb)))
                # late-position steal widen
                if late_pos and rng.random() < 0.25:
                    amt = int(max(bb, round(3.0 * bb)))
                return legalize_amount(state, min(amt, my_bank))
            if tier == 'medium':
                # LP steal sometimes
                if late_pos and rng.random() < 0.30:
                    amt = int(max(bb, round(2.0 * bb)))
                    return legalize_amount(state, min(amt, my_bank))
                return 0
            return 0

        # Facing raise preflop:
        if not can_chk:
            if tier in ('premium','strong'):
                # Mostly call; occasional re-raise when deep and late
                if eff_bb >= 25 and late_pos and rng.random() < 0.20:
                    # min-raise (safe) or small 3-bet ~3x open
                    r_amt = max(min_raise(state), int(3.0 * bb))
                    r_amt = min(r_amt, my_bank)
                    return legalize_amount(state, r_amt)
                return legalize_amount(state, to_call)
            if tier == 'medium':
                # Peel if price is cheap (<= ~12.5% pot or <= 2.5bb)
                cheap = (to_call <= max(bb*2.5, max(bb, pot//8)))
                return legalize_amount(state, to_call) if cheap else -1
            return -1

    # -------- POST-FLOP+ --------
    # Fast equity estimate
    base_iters = 260
    iters = int(max(140, base_iters - 40 * max(0, live_opp - 1)))
    equity = estimate_equity(hole, board, max(1, live_opp), iters, rng)

    # Texture flags
    parsed_board = [parse_card(c) for c in board]
    suit_counts = {}
    for _, s in parsed_board:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    flushed = any(c >= 4 for c in suit_counts.values())
    uniq = sorted(set(v for v,_ in parsed_board), reverse=True)
    straighty = (len(uniq) >= 4 and (uniq[0] - uniq[-1] <= 6))

    # Facing a bet → pot-odds gate
    if not can_chk:
        eff_pot = pot + to_call
        pot_odds = (to_call / eff_pot) if eff_pot > 0 else 1.0

        # Strong: raise sometimes, else call
        if equity >= max(0.70, pot_odds + 0.08):
            # Mix in raises but keep it legal and small (min-raise) to avoid rule traps
            if rng.random() < 0.25:
                r_amt = min(my_bank, max(min_raise(state), int(0.6 * max(pot, 3*bb))))
                return legalize_amount(state, r_amt)
            return legalize_amount(state, to_call)

        # Medium: call if equity beats price with small buffer
        if equity >= pot_odds + 0.03:
            return legalize_amount(state, to_call)

        # Rare semi-bluff if board is dynamic and price small
        if (not flushed) and straighty and rng.random() < 0.07 and to_call <= my_bank//12:
            r_amt = min(my_bank, max(min_raise(state), int(0.5 * max(pot, 3*bb))))
            return legalize_amount(state, r_amt)

        # Otherwise fold
        return -1

    # We can bet (no prior bet on this street)
    if equity >= 0.72:
        # Value bet ~70-90% pot (fallback to ~3bb if pot small)
        target = int(max(3*bb, round(0.8 * max(pot, 4*bb))))
        return legalize_amount(state, min(target, my_bank))
    elif equity >= 0.55:
        target = int(max(2*bb, round(0.6 * max(pot, 4*bb))))
        return legalize_amount(state, min(target, my_bank))
    else:
        # Small c-bet on bluffable boards
        if (not flushed) and straighty and rng.random() < 0.18:
            target = int(max(bb, round(0.4 * max(pot, 3*bb))))
            return legalize_amount(state, min(target, my_bank))
        return 0

# --------------------------
# Public entrypoint
# --------------------------

def bet(state) -> int:
    try:
        return int(decide_action(state))
    except Exception:
        # Never crash: safest fallback is fold
        return -1