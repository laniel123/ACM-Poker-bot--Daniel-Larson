# === ACM.Dev Mann vs Machine — Poker Bot V2 (Conservative → Balanced TAG) ===
# Features:
# - Late-position blind steals (30–40% range) when folded to us
# - Isolation raises vs limpers
# - Short-stack push/fold mode (< 15 BB) using simple tiers
# - Pot-odds/postflop Monte Carlo equity (fast)
# - Texture-aware c-bets & semi-bluffs
# - Strict legality: fold=-1, check/call=0 or call_amount, raises use min-raise()
#
# Interface: def bet(state) -> int
# Libraries allowed: numpy (and stdlib typing)
#
# Notes:
# - No persistent memory across turns/hands (tournament constraint)
# - Opponent “modeling” is in-hand: limpers, openers, action in front of us
# - Works with dict-like GameState as in the starter kit

from typing import List, Tuple
import numpy as np

# --------------------------
# Small helpers over state
# --------------------------

def _players(state) -> List[str]:
    return list(state["players"])

def _idx(state) -> int:
    return int(state["index_to_action"])

def _bb(state) -> int:
    return int(state["big_blind"])

def _sb(state) -> int:
    return int(state["small_blind"])

def _bet_money(state) -> List[int]:
    return [int(x) for x in state["bet_money"]]

def _held_money(state) -> List[int]:
    return [int(x) for x in state["held_money"]]

def _pots_total(state) -> int:
    pots = state.get("pots", [])
    try:
        return int(sum(int(p.get("value", 0)) for p in pots))
    except Exception:
        return 0

def _community(state) -> List[str]:
    return list(state.get("community_cards", []))

def _hole(state) -> List[str]:
    # exactly our two cards
    return list(state["player_cards"])[:2]

def _active_indexes(bet_money: List[int]) -> List[int]:
    return [i for i, b in enumerate(bet_money) if b >= 0]

def _current_max_bet(bet_money: List[int]) -> int:
    act = [b for b in bet_money if b >= 0]
    return max(act) if act else 0

def _call_amount(state) -> int:
    bm = _bet_money(state)
    me = _idx(state)
    return max(0, _current_max_bet(bm) - max(0, bm[me]))

def _can_check(state) -> bool:
    return _call_amount(state) == 0

def _my_stack(state) -> int:
    return _held_money(state)[_idx(state)]

def _num_live_opponents(state) -> int:
    bm = _bet_money(state)
    me = _idx(state)
    return max(0, len(_active_indexes(bm)) - (1 if bm[me] >= 0 else 0))

def _round_name(state) -> str:
    n = len(_community(state))
    if n == 0: return "Pre-Flop"
    if n == 3: return "Flop"
    if n == 4: return "Turn"
    if n == 5: return "River"
    return "Unknown"

def _min_raise_amount(state) -> int:
    """
    Return the chip amount THIS ACTION must put in to make a legal min-raise.
    (Not the final total; the delta we return to simulator.)
    """
    bm = _bet_money(state)
    me = _idx(state)
    current_bet = _current_max_bet(bm)
    my_bet = max(0, bm[me])

    # previous highest strictly below current bet, ignoring folds (-1)
    lower = [b for b in bm if 0 <= b < current_bet]
    if lower:
        prev = max(lower)
        last_raise = current_bet - prev
    else:
        last_raise = _bb(state)

    min_raise_size = max(last_raise, _bb(state))
    target_total = current_bet + min_raise_size
    return max(0, target_total - my_bet)

def _legal_actions(state) -> List[int]:
    """
    Legal encoded actions for this framework:
      -1 fold
       0 check (if no bet) OR call (only if to_call == 0; we use to_call explicitly otherwise)
      call_amount (when facing bet)
      min_raise_amount (if we can afford and > 0)
      all_in (our whole stack)
    We deduplicate and ensure feasibility with simple checks.
    """
    acts = []
    me = _idx(state)
    bm = _bet_money(state)
    my_bet = max(0, bm[me])
    held = _my_stack(state)
    current_bet = _current_max_bet(bm)
    to_call = _call_amount(state)

    # fold is always legal
    acts.append(-1)

    # check
    if to_call == 0:
        acts.append(0)

    # call
    if to_call > 0 and to_call <= held:
        acts.append(to_call)

    # min-raise
    mr = _min_raise_amount(state)
    if mr > 0 and mr <= held:
        acts.append(mr)

    # all-in
    if held > 0:
        acts.append(held)

    # dedupe while keeping order
    out = []
    seen = set()
    for a in acts:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out

# --------------------------
# Cards & Hand Evaluator (5-of-7)
# --------------------------

_VAL_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'t':10,'j':11,'q':12,'k':13,'a':14}
_SUITS = set(['s','h','d','c'])

def parse_card(cs: str) -> Tuple[int,str]:
    r = cs[0].lower()
    s = cs[1].lower()
    return _VAL_MAP[r], s

def _is_straight(vals_sorted_desc: List[int]) -> Tuple[bool, int]:
    # unique descending
    seen = []
    last = None
    for v in vals_sorted_desc:
        if v != last:
            seen.append(v)
            last = v
    # wheel
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

def hand_rank_7(cards: List[Tuple[int,str]]) -> Tuple[int, List[int]]:
    """
    Rank best 5 from up to 7 cards.
    Category:
      8 SF, 7 Quads, 6 FH, 5 Flush, 4 Straight, 3 Trips, 2 TwoPair, 1 Pair, 0 High
    """
    vals = [v for v,_ in cards]
    suits = [s for _,s in cards]
    vals_sorted = sorted(vals, reverse=True)

    # counts
    cnt = {}
    for v in vals:
        cnt[v] = cnt.get(v,0) + 1
    by_count = {}
    for v,c in cnt.items():
        by_count.setdefault(c, []).append(v)
    for c in by_count:
        by_count[c].sort(reverse=True)

    # flush?
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

    # straight?
    is_str, str_hi = _is_straight(vals_sorted)

    # straight flush?
    if flush_suit is not None:
        fs_vals = sorted(suit_map[flush_suit], reverse=True)
        is_sf, sf_hi = _is_straight(fs_vals)
        if is_sf:
            return (8, [sf_hi])

    # quads
    if 4 in by_count:
        q = by_count[4][0]
        kick = [x for x in vals_sorted if x != q][0]
        return (7, [q, kick])

    # full house
    trips = by_count.get(3, [])
    pairs = by_count.get(2, [])
    if trips:
        t = trips[0]
        rem_trips = trips[1:] if len(trips)>1 else []
        if rem_trips:
            return (6, [t, rem_trips[0]])
        if pairs:
            return (6, [t, pairs[0]])

    # flush
    if flush_suit is not None:
        return (5, flush_vals[:5])

    # straight
    if is_str:
        return (4, [str_hi])

    # trips
    if trips:
        t = trips[0]
        kicks = [x for x in vals_sorted if x != t][:2]
        return (3, [t] + kicks)

    # two pair
    if len(pairs) >= 2:
        p1, p2 = pairs[:2]
        hp, lp = max(p1, p2), min(p1, p2)
        kick = [x for x in vals_sorted if x!=hp and x!=lp][0]
        return (2, [hp, lp, kick])

    # pair
    if len(pairs) == 1:
        p = pairs[0]
        kicks = [x for x in vals_sorted if x != p][:3]
        return (1, [p] + kicks)

    # high
    return (0, vals_sorted[:5])

# --------------------------
# Equity Estimation (MC)
# --------------------------

def deck_minus_strings(exclude: List[str]) -> List[Tuple[int,str]]:
    ex = set(exclude)
    deck = []
    for r in _VAL_MAP.keys():
        for s in _SUITS:
            cs = r + s
            if cs in ex:
                continue
            deck.append(parse_card(cs))
    return deck

def estimate_equity(hole: List[str], board: List[str], opps: int, iterations: int, rng: np.random.Generator) -> float:
    known = list(hole) + list(board)
    deck = deck_minus_strings(known)
    deck_arr = np.array(deck, dtype=object)
    our_hole = [parse_card(c) for c in hole]
    board_parsed = [parse_card(c) for c in board]
    need_board = max(0, 5 - len(board_parsed))

    wins = 0
    ties = 0
    total = 0

    for _ in range(iterations):
        draw_cnt = 2*opps + need_board
        if draw_cnt > len(deck_arr):
            break
        idx = rng.choice(len(deck_arr), size=draw_cnt, replace=False)
        draw = deck_arr[idx]

        ptr = 0
        opp_hands = []
        for _o in range(opps):
            opp_hands.append([draw[ptr], draw[ptr+1]])
            ptr += 2

        sim_board = list(board_parsed)
        for _ in range(need_board):
            sim_board.append(draw[ptr]); ptr += 1

        our7 = our_hole + sim_board
        our_rank = hand_rank_7(our7)

        beat = False
        equal = 0
        for oh in opp_hands:
            their = list(oh) + sim_board
            their_rank = hand_rank_7(their)
            # compare
            if our_rank[0] != their_rank[0]:
                if our_rank[0] < their_rank[0]:
                    beat = True
                    equal = 0
                    break
            else:
                # tie-breakers
                a = our_rank[1]
                b = their_rank[1]
                res = 0
                for x,y in zip(a,b):
                    if x != y:
                        res = 1 if x>y else -1
                        break
                if res < 0:
                    beat = True
                    equal = 0
                    break
                elif res == 0:
                    equal += 1
        if beat:
            pass
        else:
            if equal > 0:
                ties += 1
            else:
                wins += 1
        total += 1

    if total == 0:
        return 0.0
    return (wins + 0.5*ties) / total

# --------------------------
# Ranges & Policy
# --------------------------

def _preflop_tier(hole: List[str]) -> str:
    """
    'premium' | 'strong' | 'medium' | 'spec' | 'trash'
    A hair looser than V1 to enable steals/isolations.
    """
    a, b = hole
    va, sa = parse_card(a)
    vb, sb = parse_card(b)
    vhi, vlo = max(va, vb), min(va, vb)
    suited = (sa == sb)
    gap = vhi - vlo

    # Pairs
    if va == vb:
        if vhi >= 13: return 'premium'    # AA, KK
        if vhi >= 11: return 'strong'     # QQ, JJ
        if vhi >= 7:  return 'medium'     # TT..77
        return 'spec'                      # 66..22 (set-mine / cheap steal support)

    # Broadways
    if vhi >= 13 and vlo >= 10:           # KQ, KJ/QJ, etc.
        return 'strong' if suited else 'medium'

    # Ax
    if vhi == 14:
        if suited and vlo >= 10:  # ATs+
            return 'strong'
        if suited and vlo >= 6:   # A9s-A6s
            return 'medium'
        if vlo >= 13:             # AK off
            return 'strong'
        if vlo >= 10:             # AQo/AJo/Ato
            return 'medium'
        return 'spec' if suited else 'trash'

    # Suited connectors/gappers
    if suited and gap <= 4 and vhi >= 10 and vlo >= 5:  # T9s..A5s-ish
        return 'medium'
    if suited and gap <= 3 and vhi >= 9 and vlo >= 4:   # 98s..T6s-ish
        return 'spec'

    # Kx/Qx suited small
    if suited and vhi >= 12 and vlo >= 7:
        return 'spec'

    return 'trash'

def _position_late(state) -> bool:
    # crude: closer to end of array is later
    n = len(_players(state))
    sb_idx = int(state["index_of_small_blind"])
    me = _idx(state)
    # relative position distance from SB
    pos = (me - sb_idx) % max(1, n)
    # last two seats ~ late
    return pos >= n-2

def _players_to_act_before_me_folded(state) -> bool:
    """
    True if everyone before us this street has either checked (no bet yet) or folded,
    i.e., we face no bet and can open the action.
    """
    bm = _bet_money(state)
    me = _idx(state)
    # If there is any bet already, we aren't opening
    if _current_max_bet(bm) > 0:
        return False
    # Not perfect ordering info, but if max is 0 and it's on us, action so far is checks/folds.
    return True

def _count_limpers(state) -> int:
    bm = _bet_money(state)
    # Limpers are active players with bet == big blind (preflop) when there was no raise.
    # We can't perfectly deduce preflop stage from state alone; use community==[].
    if len(_community(state)) != 0:
        return 0
    bb = _bb(state)
    mx = _current_max_bet(bm)
    if mx > bb:
        return 0
    # treat bets >0 as limp/posted blind; exclude folded
    return sum(1 for b in bm if b == bb)

# --------------------------
# Decision Core
# --------------------------

def decide_action(state: dict) -> int:
    rng = np.random.default_rng()
    round_name = _round_name(state)
    hole = _hole(state)
    board = _community(state)
    bb = _bb(state)
    stack = _my_stack(state)
    to_call = _call_amount(state)
    can_check = _can_check(state)
    pot_guess = max(_pots_total(state), sum(max(0,b) for b in _bet_money(state)))
    live_opp = max(1, _num_live_opponents(state))  # use >=1 for equity sim
    legal = _legal_actions(state)

    # ---------- Short-stack mode ----------
    bb_count = stack / max(1, bb)
    if bb_count < 15 and round_name == "Pre-Flop":
        tier = _preflop_tier(hole)
        # Simple shove chart:
        # premium/strong → shove
        # medium → shove if <= 10bb; else open if can
        # spec → shove only <= 7bb, else fold/check
        # trash → fold/check
        if to_call > stack:
            return -1
        if tier in ("premium", "strong"):
            return stack if stack in legal else (to_call if to_call in legal else -1)
        if tier == "medium":
            if bb_count <= 10:
                return stack if stack in legal else (to_call if to_call in legal else -1)
            # else open small if allowed
            if can_check and _players_to_act_before_me_folded(state):
                mr = _min_raise_amount(state)
                open_amt = max(bb*2, int(2.2*bb))
                # If framework treats "bet" as amount-to-put, ensure it's >= min-raise if raising from 0
                open_amt = max(open_amt, mr if mr>0 else open_amt)
                if open_amt <= stack and open_amt in legal:
                    return open_amt
                if to_call in legal:
                    return to_call
            return 0 if 0 in legal else (to_call if to_call in legal else -1)
        if tier == "spec":
            if bb_count <= 7:
                return stack if stack in legal else (-1 if -1 in legal else 0)
            return 0 if 0 in legal else (-1 if -1 in legal else to_call)
        # trash
        return 0 if 0 in legal else (-1 if -1 in legal else to_call)

    # ---------- Preflop standard mode ----------
    if round_name == "Pre-Flop":
        tier = _preflop_tier(hole)
        limpers = _count_limpers(state)
        # Facing a bet?
        if not can_check:
            # Tight call threshold by tier
            if tier in ("premium", "strong"):
                return to_call if to_call in legal else (-1 if -1 in legal else 0)
            if tier == "medium":
                cheap = to_call <= max(bb, pot_guess // 10)
                return (to_call if cheap and to_call in legal else -1)
            # spec/trash fold
            return -1

        # We can open (folded to us / no bet yet this street)
        if _players_to_act_before_me_folded(state):
            late = _position_late(state)
            if limpers > 0:
                # Isolation raise vs limpers with playable hands
                if tier in ("premium","strong","medium"):
                    mr = _min_raise_amount(state)
                    # iso size ~ (3bb + 1bb per limper)
                    base = int(3*bb + limpers*1.2*bb)
                    open_amt = max(base, mr if mr>0 else base)
                    open_amt = min(open_amt, stack)
                    return open_amt if open_amt in legal else (0 if 0 in legal else -1)
                # spec sometimes iso in late
                if tier == "spec" and late and rng.random() < 0.25:
                    mr = _min_raise_amount(state)
                    base = int(3*bb + limpers*1.0*bb)
                    open_amt = max(base, mr if mr>0 else base)
                    open_amt = min(open_amt, stack)
                    return open_amt if open_amt in legal else (0 if 0 in legal else -1)
                # otherwise just check
                return 0 if 0 in legal else -1
            else:
                # Blind steal in late position with ~35% range
                if late:
                    if tier in ("premium","strong","medium"):
                        mr = _min_raise_amount(state)
                        base = int(2.5*bb if rng.random() < 0.6 else 3.0*bb)
                        open_amt = max(base, mr if mr>0 else base)
                        open_amt = min(open_amt, stack)
                        return open_amt if open_amt in legal else (0 if 0 in legal else -1)
                    if tier == "spec" and rng.random() < 0.35:
                        mr = _min_raise_amount(state)
                        base = int(2.2*bb)
                        open_amt = max(base, mr if mr>0 else base)
                        open_amt = min(open_amt, stack)
                        return open_amt if open_amt in legal else (0 if 0 in legal else -1)
                    return 0 if 0 in legal else -1
                else:
                    # Early/mid: open tighter
                    if tier == "premium":
                        mr = _min_raise_amount(state)
                        base = int(3.0*bb)
                        open_amt = max(base, mr if mr>0 else base)
                        open_amt = min(open_amt, stack)
                        return open_amt if open_amt in legal else (0 if 0 in legal else -1)
                    if tier == "strong":
                        mr = _min_raise_amount(state)
                        base = int(2.5*bb)
                        open_amt = max(base, mr if mr>0 else base)
                        open_amt = min(open_amt, stack)
                        return open_amt if open_amt in legal else (0 if 0 in legal else -1)
                    if tier == "medium" and rng.random() < 0.25:
                        mr = _min_raise_amount(state)
                        base = int(2.2*bb)
                        open_amt = max(base, mr if mr>0 else base)
                        open_amt = min(open_amt, stack)
                        return open_amt if open_amt in legal else (0 if 0 in legal else -1)
                    return 0 if 0 in legal else -1

        # default
        return 0 if 0 in legal else (-1 if -1 in legal else to_call)

    # ---------- Postflop ----------
    # Equity estimate
    base_iters = 220
    iters = int(max(120, base_iters - 40 * max(0, live_opp - 1)))
    equity = estimate_equity(hole, board, opps=live_opp, iterations=iters, rng=rng)

    # Texture flags
    parsed_board = [parse_card(c) for c in board]
    suits = {}
    for _,s in parsed_board:
        suits[s] = suits.get(s,0) + 1
    flushy = any(c >= 4 for c in suits.values())
    uniq_vals = sorted(set(v for v,_ in parsed_board), reverse=True)
    straighty = (len(uniq_vals) >= 4 and (uniq_vals[0]-uniq_vals[-1] <= 6))

    # Facing bet → pot odds gate
    if not can_check:
        to_call = _call_amount(state)
        eff_pot = pot_guess + to_call
        pot_odds = (to_call / eff_pot) if eff_pot > 0 else 1.0
        # Call if equity clear of pot odds
        if equity >= pot_odds + 0.035:
            return to_call if to_call in legal else (-1 if -1 in legal else 0)
        # Rare semi-bluff raise when board is straighty and not flushy and call is small
        if (not flushy) and straighty and rng.random() < 0.10:
            mr = _min_raise_amount(state)
            # raise small-ish if legal
            if mr in legal and mr <= _my_stack(state):
                return mr
        # Otherwise fold
        return -1

    # No bet yet → we can bet
    # Value bet tiers by equity
    if equity >= 0.72:
        # strong value: ~70-90% pot
        target = int(max(2*bb, round(0.8 * max(pot_guess, 4*bb))))
        # ensure >= min-raise if needed
        mr = _min_raise_amount(state)
        if mr > 0:
            target = max(target, mr)
        target = min(target, _my_stack(state))
        if target in legal and target > 0:
            return target
        return 0 if 0 in legal else -1

    if equity >= 0.56:
        target = int(max(bb, round(0.55 * max(pot_guess, 4*bb))))
        mr = _min_raise_amount(state)
        if mr > 0:
            target = max(target, mr)
        target = min(target, _my_stack(state))
        if target in legal and target > 0:
            return target
        return 0 if 0 in legal else -1

    # Bluffs/c-bets: prefer dry/straighty boards without 4+ flush
    if (not flushy) and straighty and rng.random() < 0.22:
        target = int(max(bb, round(0.4 * max(pot_guess, 3*bb))))
        mr = _min_raise_amount(state)
        if mr > 0:
            target = max(target, mr)
        target = min(target, _my_stack(state))
        if target in legal and target > 0:
            return target

    # Take the free card
    return 0 if 0 in legal else -1

# --------------------------
# Public entrypoint
# --------------------------

def bet(state) -> int:
    try:
        return int(decide_action(state))
    except Exception:
        # hard fail-safe: never crash the tournament engine
        return -1