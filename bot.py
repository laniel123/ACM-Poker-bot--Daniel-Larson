# === ACM.Dev Mann vs Machine — One-File Poker Bot (Hybrid Strategy D, Per-Hand Profiling, Option B Snap) ===
# Compliant: one file, Python 3.11 + numpy only, entrypoint bet(state)->int, ≤5s per turn.
# Uses only the helper-style APIs defined below (amount_to_call, get_best_hand_from, legal_actions, etc.).
# No I/O, no disallowed imports, no cross-hand persistence.

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
import numpy as np

if TYPE_CHECKING:
    # Types imported for static checking only (the engine defines them)
    from bot import GameState, Pot

# -------------------------------------------------------------------------
# Helper Functions (merged into this single file as requested)
# -------------------------------------------------------------------------

def get_player_list(state: GameState) -> list[str]:
    return state.players

def amount_to_call(state: GameState) -> int:
    current_bet = max(state.bet_money) if state.bet_money else 0
    player_bet = state.bet_money[state.index_to_action] if state.bet_money else 0
    return max(0, current_bet - max(0, player_bet))

def get_my_pots(state: GameState) -> list[Pot]:
    my_index = state.index_to_action
    my_id = state.players[my_index]
    my_pots = []
    for pot in state.pots:
        try:
            if my_id in pot.players:
                my_pots.append(pot)
        except Exception:
            # if Pot is dict-like
            if my_id in pot.get("players", []):
                my_pots.append(pot)
    return my_pots

# --- Card parsing + mapping ---

_RANK_TO_INT = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                't': 10, 'j': 11, 'q': 12, 'k': 13, 'a': 14}
_INT_TO_RANK = {v: k for k, v in _RANK_TO_INT.items()}
_VALID_SUITS = {'s','h','d','c'}

def parse_card(card: str) -> tuple[int, str]:
    if len(card) != 2:
        raise ValueError(f"Invalid card string: {card}")
    r = card[0].lower()
    s = card[1].lower()
    if r not in _RANK_TO_INT:
        raise ValueError(f"Invalid rank: {r}")
    if s not in _VALID_SUITS:
        raise ValueError(f"Invalid suit: {s}")
    return (_RANK_TO_INT[r], s)

def _is_straight_from_unique_desc(ranks_desc_unique: list[int]) -> int:
    # returns high-card of straight if any, else 0; supports wheel A-2-3-4-5
    if not ranks_desc_unique:
        return 0
    ranks_set = set(ranks_desc_unique)
    # Try normal straights
    mx = max(ranks_set)
    mn = min(ranks_set)
    for high in range(mx, 4, -1):
        if all((high - i) in ranks_set for i in range(5)):
            return high
    # Wheel
    if {14, 5, 4, 3, 2}.issubset(ranks_set):
        return 5
    return 0

def _evaluate_five(cards5: list[str]) -> tuple[int, list[int]]:
    """
    Evaluate exactly 5 cards.
    Returns (category, kickers) with 0..8 categories like standard:
      8: Straight Flush
      7: Four of a Kind
      6: Full House
      5: Flush
      4: Straight
      3: Three of a Kind
      2: Two Pair
      1: One Pair
      0: High Card
    Tiebreakers follow standard descending values.
    """
    parsed = [parse_card(c) for c in cards5]
    ranks = sorted((r for r, _ in parsed), reverse=True)
    suits = [s for _, s in parsed]

    from collections import Counter
    cnt = Counter(ranks)
    counts = sorted(((count, rank) for rank, count in cnt.items()), reverse=True)
    is_flush = len(set(suits)) == 1
    unique_desc = sorted(set(ranks), reverse=True)
    straight_high = _is_straight_from_unique_desc(unique_desc)

    # Straight flush
    if is_flush and straight_high:
        return (8, [straight_high])
    # Four of a kind
    if counts[0][0] == 4:
        four_rank = counts[0][1]
        kicker = max(r for r in ranks if r != four_rank)
        return (7, [four_rank, kicker])
    # Full house
    if counts[0][0] == 3 and len(counts) > 1 and counts[1][0] >= 2:
        three_rank = counts[0][1]
        pair_rank = counts[1][1]
        return (6, [three_rank, pair_rank])
    # Flush
    if is_flush:
        return (5, ranks)
    # Straight
    if straight_high:
        return (4, [straight_high])
    # Trips
    if counts[0][0] == 3:
        three_rank = counts[0][1]
        kickers = sorted((r for r in ranks if r != three_rank), reverse=True)
        return (3, [three_rank] + kickers)
    # Two pair
    if counts[0][0] == 2 and len(counts) > 1 and counts[1][0] == 2:
        high_pair = max(counts[0][1], counts[1][1])
        low_pair = min(counts[0][1], counts[1][1])
        kicker = max(r for r in ranks if r != high_pair and r != low_pair)
        return (2, [high_pair, low_pair, kicker])
    # One pair
    if counts[0][0] == 2:
        pair_rank = counts[0][1]
        kickers = sorted((r for r in ranks if r != pair_rank), reverse=True)
        return (1, [pair_rank] + kickers)
    # High card
    return (0, ranks)

def get_best_hand_from(hand: list[str], community: list[str]) -> tuple[int, list[str]]:
    """
    Returns (category, best_five_cards_as_str).
    """
    if not isinstance(hand, list) or not isinstance(community, list):
        raise TypeError("hand and community must be lists of card strings")
    if len(hand) > 2:
        raise ValueError("hand should contain at most 2 cards")

    from itertools import combinations
    cards = list(hand) + list(community)
    best = (-1, [])
    best_five = None
    for combo in combinations(cards, 5):
        val = _evaluate_five(list(combo))
        if val > best:
            best = val
            best_five = combo
    if best[0] < 0 or best_five is None:
        return (-1, [])
    return (best[0], list(best_five))

def fold() -> int:
    return -1

def check() -> int:
    return 0

def call(state: GameState) -> int:
    return amount_to_call(state)

def all_in(state: GameState) -> int:
    my_index = state.index_to_action
    return max(0, int(state.held_money[my_index]))

def min_raise(state: GameState) -> int:
    current_bet = max(state.bet_money) if state.bet_money else 0
    my_index = state.index_to_action
    my_bet = state.bet_money[my_index] if state.bet_money else 0
    # lower active bets (exclude folds)
    lower_bets = [b for b in state.bet_money if 0 <= b < current_bet]
    if lower_bets:
        prev_max = max(lower_bets)
        previous_raise = current_bet - prev_max
    else:
        previous_raise = state.big_blind
    min_raise_size = max(previous_raise, state.big_blind)
    target_total_bet = current_bet + min_raise_size
    return max(0, target_total_bet - my_bet)

def is_valid_bet(state: GameState, amount: int) -> bool:
    my_index = state.index_to_action
    if amount == -1:
        return True
    to_call = amount_to_call(state)
    current_bet = max(state.bet_money) if state.bet_money else 0
    my_bet = state.bet_money[my_index] if state.bet_money else 0
    my_held = state.held_money[my_index] if state.held_money else 0

    if amount == 0:
        return to_call == 0
    elif amount < 0:
        return False
    else:
        total_bet = my_bet + amount
        if total_bet < current_bet:
            return False
        if total_bet > my_bet + my_held:
            return False
        if total_bet > current_bet:
            # it's a raise; must meet min raise requirement
            if amount < min_raise(state):
                return False
        return True

def get_round_name(state: GameState) -> str:
    community_len = len(state.community_cards)
    if community_len == 0:
        return "Pre-Flop"
    elif community_len == 3:
        return "Flop"
    elif community_len == 4:
        return "Turn"
    elif community_len == 5:
        return "River"
    else:
        return "Unknown Round"

def my_stack(state: GameState) -> int:
    my_index = state.index_to_action
    return int(state.held_money[my_index])

def opp_stacks(state: GameState) -> dict[int, int]:
    my_index = state.index_to_action
    return {i: int(state.held_money[i]) for i in range(len(state.players)) if i != my_index}

def legal_actions(state: GameState) -> list[int]:
    actions = []
    for amount in [-1, 0, amount_to_call(state), min_raise(state), all_in(state)]:
        if is_valid_bet(state, amount) and amount not in actions:
            actions.append(amount)
    return actions

def total_pot(state: GameState) -> int:
    try:
        return int(sum(int(p.value) for p in state.pots))
    except Exception:
        return int(sum(int(p.get("value", 0)) for p in state.pots))

def _normalize_card_code(cs: str) -> str:
    # to lowercase 'rank' + 'suit' (e.g., 'As'/'AS'/'as' -> 'as')
    return cs[0].lower() + cs[1].lower()

def deck_remaining(state: GameState) -> list[tuple[int, str]]:
    """
    Robust deck builder that tolerates mixed-case input from engine.
    Returns list of (rank:int, suit:str) for undealt cards.
    """
    all_ranks = ['2','3','4','5','6','7','8','9','t','j','q','k','a']
    all_suits = ['s','h','d','c']
    full_deck = {r+s for r in all_ranks for s in all_suits}

    dealt = set()
    for c in state.player_cards:
        if c:
            dealt.add(_normalize_card_code(c))
    for c in state.community_cards:
        if c:
            dealt.add(_normalize_card_code(c))

    remaining = full_deck - dealt
    out: list[tuple[int,str]] = []
    for cs in remaining:
        r, s = parse_card(cs)
        out.append((r, s))
    return out

# -------------------------------------------------------------------------
# Small utilities for policy
# -------------------------------------------------------------------------

def _active_player_indexes(bet_money: List[int]) -> List[int]:
    return [i for i, b in enumerate(bet_money) if b >= 0]

def _current_max_bet(bet_money: List[int]) -> int:
    act = [b for b in bet_money if b >= 0]
    return max(act) if act else 0

def _num_live_opponents(state: GameState) -> int:
    me = state.index_to_action
    return max(0, len(_active_player_indexes(state.bet_money)) - (1 if state.bet_money[me] >= 0 else 0))

def _position_index(state: GameState) -> int:
    # distance from SB; later seats have larger values (rough heuristic)
    n = len(state.players)
    return (state.index_to_action - state.index_of_small_blind) % max(1, n)

def _board_texture_flags(board: List[str]) -> tuple[bool,bool,bool]:
    """Return (flushy, straighty, paired)."""
    if not board:
        return (False, False, False)
    parsed = [parse_card(c) for c in board]
    # flushy?
    suit_counts = {}
    for _, s in parsed:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    flushy = any(c >= 3 for c in suit_counts.values())
    # straighty?
    uniq = sorted(set(v for v,_ in parsed), reverse=True)
    straighty = (len(uniq) >= 4 and (uniq[0] - uniq[-1] <= 6))
    # paired?
    val_counts = {}
    for v,_ in parsed:
        val_counts[v] = val_counts.get(v, 0) + 1
    paired = any(c >= 2 for c in val_counts.values())
    return (flushy, straighty, paired)

def _per_hand_aggression(state: GameState) -> float:
    mx = _current_max_bet(state.bet_money)
    me = state.index_to_action
    opps = [i for i,b in enumerate(state.bet_money) if i != me and b >= 0]
    if not opps:
        return 0.0
    hi = sum(1 for i in opps if state.bet_money[i] >= max(1, mx // 2))
    return hi / len(opps)

def _per_hand_passivity(state: GameState) -> float:
    me = state.index_to_action
    opps = [i for i,b in enumerate(state.bet_money) if i != me and b >= 0]
    if not opps:
        return 0.0
    mx = _current_max_bet(state.bet_money)
    checks_or_behind = 0
    for i in opps:
        b = state.bet_money[i]
        if b == 0 or (mx > 0 and b < mx):
            checks_or_behind += 1
    return checks_or_behind / len(opps)

# -------------------------------------------------------------------------
# Equity: Monte Carlo using deck_remaining + get_best_hand_from
# -------------------------------------------------------------------------

def _card_tuple_to_str(rank: int, suit: str) -> str:
    return _INT_TO_RANK[rank] + suit

def estimate_equity_mc(state: GameState, iterations: int = 280) -> float:
    """
    Estimate our equity vs live opponents using Monte Carlo.
    Uses helpers:
      - deck_remaining(state)  -> remaining cards as (rank, suit)
      - get_best_hand_from()   -> best 5-card selection
    """
    rng = np.random.default_rng()

    # Known cards (normalize to lower-case)
    hole = list(state.player_cards[:2])
    board = list(state.community_cards)
    known_set = {_normalize_card_code(c) for c in hole if c} | {_normalize_card_code(c) for c in board if c}

    # Build deck from helper (tuples) -> strings 'rs'
    rem_tuples = deck_remaining(state)
    deck = np.array([_card_tuple_to_str(r, s) for (r, s) in rem_tuples], dtype=object)

    live_opp = _num_live_opponents(state)
    if live_opp <= 0:
        # heads-up vs nobody (shouldn't happen), count as full equity
        return 1.0

    # how many community to draw
    need_board = max(0, 5 - len(board))

    wins = 0
    ties = 0
    total = 0

    for _ in range(iterations):
        draw_cnt = 2 * live_opp + need_board
        if draw_cnt > len(deck):
            break
        idx = rng.choice(len(deck), size=draw_cnt, replace=False)
        draw = deck[idx]

        # Opponents' hands
        opp_hands = []
        ptr = 0
        for _i in range(live_opp):
            opp_hands.append([draw[ptr], draw[ptr+1]])
            ptr += 2

        # Fill board
        sim_board = list(board)
        for _j in range(need_board):
            sim_board.append(draw[ptr]); ptr += 1

        # Our best 5 and category
        our_cat, our5 = get_best_hand_from(hole, sim_board)
        our_val = _evaluate_five(our5)

        better = False
        equal = 0
        for oh in opp_hands:
            cat, best5 = get_best_hand_from(oh, sim_board)
            val = _evaluate_five(best5)
            if val > our_val:
                better = True
                equal = 0
                break
            elif val == our_val:
                equal += 1

        if better:
            pass
        else:
            if equal > 0:
                ties += 1
            else:
                wins += 1
        total += 1

    if total == 0:
        return 0.0
    return (wins + 0.5 * ties) / total

# -------------------------------------------------------------------------
# Preflop tiers (position-aware) — conservative → balanced
# -------------------------------------------------------------------------

def _preflop_tier(hole: list[str], pos: int, n_players: int) -> str:
    """
    Returns 'premium' | 'strong' | 'medium' | 'trash'.
    Slightly looser in late position.
    """
    if len(hole) < 2:
        return 'trash'
    a, b = hole[0], hole[1]
    va, sa = parse_card(a)
    vb, sb = parse_card(b)
    vhi, vlo = max(va, vb), min(va, vb)
    suited = (sa == sb)
    gap = vhi - vlo
    late = (pos >= n_players - 2)

    # Pairs
    if va == vb:
        if vhi >= 13:   # AA, KK
            return 'premium'
        if vhi >= 11:   # QQ, JJ
            return 'strong'
        if vhi >= 8:    # TT, 99, 88
            return 'medium'
        return 'medium' if late else 'trash'

    # Broadways / Ax
    if vhi >= 13 and vlo >= 10:  # KQ/KJ/QJ
        return 'strong' if suited or late else 'medium'
    if vhi == 14:  # Ax
        if suited and vlo >= 10:        # ATs+
            return 'strong'
        if vlo >= 13:                    # AK offsuit
            return 'strong'
        if suited and (vlo >= 6 or late):
            return 'medium'
        if vlo >= 10:
            return 'medium' if late else 'trash'
        return 'trash' if not late else 'medium'

    # Suited connectors/gappers (late)
    if suited and gap <= 3 and vhi >= 9 and (vlo >= 5 or late):
        return 'medium'
    if suited and late and gap <= 4 and vhi >= 8 and vlo >= 5:
        return 'medium'

    return 'trash'

# -------------------------------------------------------------------------
# Option-B bet snapping (pick the largest legal action ≤ target)
# -------------------------------------------------------------------------

def _snap_to_legal_option_b(state: GameState, target: int) -> int:
    """
    Given a desired 'target' chip amount for THIS action, return the largest legal action
    that does not exceed 'target'. If none exist ≤ target, return 0 (check) if legal,
    else fold().
    """
    acts = legal_actions(state)
    # exact match first
    if target in acts:
        return target
    # collect numeric actions ≥ 0 and ≤ target
    leq = [a for a in acts if a >= 0 and a <= target]
    if leq:
        return max(leq)
    # if we can check, do it
    if 0 in acts:
        return 0
    # otherwise fold safely
    return -1

# -------------------------------------------------------------------------
# Core Decision Policy (Hybrid Strategy D)
# -------------------------------------------------------------------------

def _decide(state: GameState) -> int:
    idx = state.index_to_action
    bb = int(state.big_blind)
    n_players = len(state.players)
    hole = list(state.player_cards[:2])
    board = list(state.community_cards)
    pot_val = max(total_pot(state), sum(max(0, b) for b in state.bet_money))
    pos = _position_index(state)
    live_opp = _num_live_opponents(state)
    call_amt = amount_to_call(state)
    can_check = (call_amt == 0)
    rng = np.random.default_rng()

    # Per-street opponent reads + board texture
    opp_aggr = _per_hand_aggression(state)
    opp_pass = _per_hand_passivity(state)
    flushy, straighty, paired = _board_texture_flags(board)

    # Safety: if we can't afford call, fold
    if call_amt > my_stack(state):
        return fold()

    preflop = (len(board) == 0)

    # ---------------- PRE-FLOP ----------------
    if preflop:
        tier = _preflop_tier(hole, pos, n_players)

        if not can_check:
            if tier in ('premium', 'strong'):
                return call(state)
            if tier == 'medium':
                cheap = call_amt <= max(bb, pot_val // 8)
                late_bonus = (pos >= n_players - 2) and (call_amt <= 3*bb)
                return call(state) if (cheap or late_bonus) else fold()
            return fold()

        # We can open
        if tier == 'premium':
            base = 3.2 if rng.random() < 0.5 else 2.8
            target = int(max(bb, round(base * bb)))
            return _snap_to_legal_option_b(state, target)
        elif tier == 'strong':
            base = 2.7 if rng.random() < 0.6 else 2.3
            if pos >= n_players - 2 and rng.random() < 0.25:
                base += 0.4
            target = int(max(bb, round(base * bb)))
            return _snap_to_legal_option_b(state, target)
        elif tier == 'medium':
            open_freq = 0.12 + 0.10 * (1.0 if pos >= n_players - 2 else 0.0)
            if rng.random() < open_freq:
                target = int(max(bb, round(2.0 * bb)))
                return _snap_to_legal_option_b(state, target)
            return check()
        else:
            return check()

    # --------------- POST-FLOP & LATER ---------------
    # Monte Carlo equity vs live opponents
    base_iters = 300
    iters = int(max(140, base_iters - 40 * max(0, live_opp - 1)))
    equity = estimate_equity_mc(state, iterations=iters)

    if not can_check:
        # facing a bet
        eff_pot = pot_val + call_amt
        pot_odds = (call_amt / eff_pot) if eff_pot > 0 else 1.0
        board_scary = (flushy and straighty) or (paired and (flushy or straighty))
        slack = 0.035 + (0.02 if board_scary else 0.0) - (0.01 * min(1.0, opp_pass))

        if equity >= pot_odds + slack:
            return call(state)
        else:
            # rare semi-continue when cheap + fold equity (passive opps)
            semi_window = (not paired) and (flushy or straighty)
            if semi_window and call_amt <= 2*bb and rng.random() < (0.06 + 0.04*max(0.0, opp_pass - 0.5)):
                return call(state)
            return fold()

    # We can bet
    pot_proxy = max(pot_val, 4*bb)
    value_hi = 0.72 - 0.03 * min(1.0, opp_pass)     # bet thinner vs passive
    value_mid = 0.56 - 0.02 * min(1.0, opp_pass)

    if equity >= value_hi:
        pct = 0.85 if not (paired or flushy) else 0.72
        if opp_pass > 0.6:
            pct = max(pct, 0.9)
        target = int(max(3*bb, round(pct * pot_proxy)))
        return _snap_to_legal_option_b(state, target)

    if equity >= value_mid:
        pct = 0.55 + (0.05 if pos >= n_players - 2 else 0.0)
        target = int(max(2*bb, round(pct * pot_proxy)))
        return _snap_to_legal_option_b(state, target)

    # Bluff / c-bet
    bluffable = (not paired) and (flushy ^ straighty)  # exactly one pressure vector
    base_bluff = 0.10 + 0.08 * max(0.0, opp_pass - 0.5) - 0.05 * max(0.0, opp_aggr - 0.3)
    if bluffable and rng.random() < base_bluff:
        target = int(max(bb, round(0.4 * pot_proxy)))
        return _snap_to_legal_option_b(state, target)

    return check()

# -------------------------------------------------------------------------
# Public entrypoint required by the tournament engine
# -------------------------------------------------------------------------

def bet(state) -> int:
    try:
        return int(_decide(state))
    except Exception:
        # Fail-safe to never crash the engine
        return -1