#!/usr/bin/env python3
"""
ACM Poker Bot Local Simulator — Fixed & Side-Pot Correct

- Texas Hold'em 2+ players
- Proper blinds, betting rounds, min-raise rules, all-ins
- Side-pot creation & correct distribution
- No negative stacks; chips are conserved
- Bot interface: bet(state) -> int  (amount *to add this action*)
    - -1 : fold
    -  0 : check (only if to_call == 0), else illegal -> fold
    - >0 : add this many chips now (capped by stack). If total < to_call -> illegal -> fold.
            If total == to_call -> call.
            If total > to_call -> raise. Must meet min-raise unless all-in; otherwise illegal -> fold.
"""

from __future__ import annotations
import argparse
import importlib.util
import random
from dataclasses import dataclass
from itertools import combinations
from collections import Counter
from typing import List, Optional

# ================================================================
# === DATA CLASSES (ACM-like) ====================================
# ================================================================

@dataclass
class Pot:
    value: int
    players: list[str]  # eligible winners (ids)


@dataclass
class GameState:
    index_to_action: int
    index_of_small_blind: int
    players: list[str]              # bot ids (strings)
    player_cards: list[str]         # current player's 2 cards, like ['as','kd']
    held_money: list[int]           # stacks per seat
    bet_money: list[int]            # current-street contributions; -1 for folded
    community_cards: list[str]      # 0..5 cards
    pots: list[Pot]                 # coarse view of current pot(s)
    small_blind: int
    big_blind: int


# ================================================================
# === BOT LOADER =================================================
# ================================================================

def load_bot(path: str):
    spec = importlib.util.spec_from_file_location("bot_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "bet"):
        raise ValueError(f"{path} must define bet(state)->int")
    return mod.bet


# ================================================================
# === CARDS & HAND EVAL ==========================================
# ================================================================

RANKS = "23456789tjqka"
SUITS = "shdc"

def fresh_deck() -> list[str]:
    return [r+s for r in RANKS for s in SUITS]

def deal_cards(deck: list[str], n: int) -> list[str]:
    out = deck[:n]
    del deck[:n]
    return out

def parse_rank(card: str) -> int:
    return RANKS.index(card[0])

def straight_high_value(unique_desc: list[int]) -> int:
    """Return high rank index (0..12) of straight, 0 means no straight.
       Note: we treat A-2-3-4-5 (wheel) as high=3 (the '5')."""
    # Try any 5-length run in rank indices
    sset = set(unique_desc)
    for hi in unique_desc:
        chain = [hi - i for i in range(5)]
        if all(v in sset for v in chain):
            return hi
    # Wheel: A(12), 3,2,1,0 => needs {12,3,2,1,0}
    if {12, 3, 2, 1, 0}.issubset(sset):
        return 3
    return -1

def evaluate_five(cards5: list[str]) -> tuple[int, list[int]]:
    """Return (category, kickers) — higher is better.
       category: 0..8 (High→StraightFlush). Kickers are rank indices (0..12)."""
    ranks = sorted((parse_rank(c) for c in cards5), reverse=True)
    suits = [c[1] for c in cards5]
    cnt = Counter(ranks)
    counts = sorted(((count, r) for r, count in cnt.items()), reverse=True)
    is_flush = len(set(suits)) == 1
    unique_desc = sorted(set(ranks), reverse=True)
    s_hi = straight_high_value(unique_desc)

    # Straight Flush
    if is_flush and s_hi >= 0:
        return (8, [s_hi])

    # Four of a Kind
    if counts[0][0] == 4:
        four = counts[0][1]
        kicker = max(r for r in ranks if r != four)
        return (7, [four, kicker])

    # Full House
    if counts[0][0] == 3 and len(counts) > 1 and counts[1][0] >= 2:
        return (6, [counts[0][1], counts[1][1]])

    # Flush
    if is_flush:
        return (5, ranks)

    # Straight
    if s_hi >= 0:
        return (4, [s_hi])

    # Trips
    if counts[0][0] == 3:
        three = counts[0][1]
        kickers = [r for r in ranks if r != three]
        return (3, [three] + kickers)

    # Two Pair
    if counts[0][0] == 2 and len(counts) > 1 and counts[1][0] == 2:
        hp = max(counts[0][1], counts[1][1])
        lp = min(counts[0][1], counts[1][1])
        kicker = max(r for r in ranks if r != hp and r != lp)
        return (2, [hp, lp, kicker])

    # One Pair
    if counts[0][0] == 2:
        p = counts[0][1]
        kickers = [r for r in ranks if r != p]
        return (1, [p] + kickers)

    # High Card
    return (0, ranks)

def best_of_seven(hole: list[str], community: list[str]) -> tuple[int, list[int]]:
    """Pick best 5-card hand value from 7 cards."""
    best: Optional[tuple[int, list[int]]] = None
    for combo in combinations(hole + community, 5):
        val = evaluate_five(list(combo))
        if best is None or val > best:
            best = val
    assert best is not None
    return best


# ================================================================
# === TABLE / BETTING ENGINE =====================================
# ================================================================

class SeatState:
    __slots__ = ("cards", "stack", "folded", "all_in", "invested_total", "street_bet")
    def __init__(self, cards: list[str], stack: int):
        self.cards = cards
        self.stack = stack
        self.folded = False
        self.all_in = False
        self.invested_total = 0  # total this hand
        self.street_bet = 0      # this street


def min_raise_size(bb: int, current_bet: int, last_bet_before_current: int) -> int:
    """If no raise yet: use bb. Else size = current_bet - last_bet_before_current, but at least bb."""
    base = current_bet - last_bet_before_current if last_bet_before_current >= 0 else bb
    return max(bb, base)


def action_order(start: int, n: int) -> List[int]:
    return [(start + i) % n for i in range(n)]


def current_max_bet(seats: list[SeatState]) -> int:
    return max((s.street_bet for s in seats if not s.folded), default=0)


def any_to_act_open(seats: list[SeatState], have_acted: set[int]) -> bool:
    """Betting continues if some nonfolded, not all-in seat hasn't matched the max bet OR a raise was made."""
    mx = current_max_bet(seats)
    for i, s in enumerate(seats):
        if s.folded or s.all_in:
            continue
        if s.street_bet != mx:
            return True
    # If all matched and no more raises pending, done
    return False


def only_one_live(seats: list[SeatState]) -> Optional[int]:
    alive = [i for i, s in enumerate(seats) if not s.folded]
    return alive[0] if len(alive) == 1 else None


def build_pots_from_contributions(seats: list[SeatState], player_ids: list[str]) -> list[Pot]:
    """
    Build side pots from seats[i].invested_total.
    Folded players still contribute chips but are *not* eligible to win.
    """
    contrib = {i: s.invested_total for i, s in enumerate(seats)}
    # Distinct positive tiers
    tiers = sorted(set(v for v in contrib.values() if v > 0))
    pots: list[Pot] = []
    prev = 0
    for level in tiers:
        slice_amt = level - prev
        if slice_amt <= 0:
            continue
        value = sum(min(contrib[i], slice_amt) for i in range(len(seats)))
        elig = [player_ids[i] for i, s in enumerate(seats) if (not s.folded) and contrib[i] >= level]
        if value > 0 and len(elig) > 0:
            pots.append(Pot(value=value, players=elig))
        prev = level
    return pots


def payout_showdown(seats: list[SeatState], player_ids: list[str], community: list[str]) -> None:
    """Create side pots and distribute to winners (split by ties)."""
    pots = build_pots_from_contributions(seats, player_ids)

    # Precompute hand values for live players
    hand_val: dict[int, tuple[int, list[int]]] = {}
    for i, s in enumerate(seats):
        if not s.folded:
            hand_val[i] = best_of_seven(s.cards, community)

    for pot in pots:
        # Eligible seat indices:
        elig_idx = [player_ids.index(pid) for pid in pot.players]
        # Find best among these
        best = None
        winners: list[int] = []
        for i in elig_idx:
            val = hand_val[i]
            if best is None or val > best:
                best = val
                winners = [i]
            elif val == best:
                winners.append(i)
        # Split chips
        share = pot.value // len(winners)
        remainder = pot.value - share * len(winners)
        for j, w in enumerate(winners):
            seats[w].stack += share + (1 if j < remainder else 0)


def run_betting_round(
    seats: list[SeatState],
    player_ids: list[str],
    dealer_idx: int,
    sb: int,
    bb: int,
    community: list[str],
) -> bool:
    """
    Run a single betting round. Returns True if hand should continue to next street,
    False if hand ended early (everyone but one folded).
    Implements min-raise logic and all-ins. Resets seats[i].street_bet when done.
    """
    n = len(seats)
    # Determine first to act:
    # Preflop: first to act is seat after BB
    # Postflop: first to act is seat after dealer
    preflop = (len(community) == 0)
    if preflop:
        start = (dealer_idx + 3) % n  # UTG (after BB)
    else:
        start = (dealer_idx + 1) % n  # first after dealer

    # Track raise size baseline
    # We track the last bet amount strictly below current max to compute min-raise size
    last_lower_than_max = -1  # if no raise yet, use bb as baseline

    # Betting loop
    acted = set()
    order = action_order(start, n)

    def to_call(i: int) -> int:
        mx = current_max_bet(seats)
        return max(0, mx - seats[i].street_bet)

    while True:
        # If only one live, end hand
        lone = only_one_live(seats)
        if lone is not None:
            return False  # hand ended early

        # If everyone matched and either all acted or no raises possible, break
        if not any_to_act_open(seats, acted):
            break

        for i in order:
            s = seats[i]
            if s.folded or s.all_in:
                acted.add(i)
                continue

            # If round is already closed and this seat matched, skip
            mx = current_max_bet(seats)
            need = mx - s.street_bet
            if need == 0 and i in acted:
                continue

            # Build lightweight pots view (for bots)
            # Use cumulative invested_total so far (single abstract pot for UI)
            pots_view = [Pot(value=sum(ss.invested_total for ss in seats),
                             players=[pid for j, pid in enumerate(player_ids) if not seats[j].folded])]
            bet_money_view = [(-1 if ss.folded else ss.street_bet) for ss in seats]
            held_view = [ss.stack for ss in seats]

            state = GameState(
                index_to_action=i,
                index_of_small_blind=(dealer_idx + 1) % n,
                players=player_ids,
                player_cards=seats[i].cards,
                held_money=held_view,
                bet_money=bet_money_view,
                community_cards=community[:],
                pots=pots_view,
                small_blind=sb,
                big_blind=bb,
            )

            # Ask bot
            try:
                action = int(load_bot.cache[player_ids[i]])(state)  # type: ignore[attr-defined]
            except Exception:
                # If something goes wrong, fold to be safe
                action = -1

            # Normalize/cap action
            if action is None:
                action = -1

            # Process action
            cur_bet = s.street_bet
            mx = current_max_bet(seats)
            need = mx - cur_bet

            if action == -1:
                # Fold
                s.folded = True
                acted.add(i)
            elif action == 0:
                # Check only if need == 0, else illegal -> fold
                if need == 0:
                    acted.add(i)
                else:
                    s.folded = True
                    acted.add(i)
            else:
                # add amount this action
                add = max(0, action)
                if add > s.stack:
                    add = s.stack  # all-in cap

                new_total = cur_bet + add

                # Must at least CALL if facing a bet; otherwise illegal -> fold
                if new_total < mx:
                    s.folded = True
                    acted.add(i)
                    continue

                # Deduct chips & update contributions
                s.stack -= add
                s.street_bet = new_total
                s.invested_total += add
                if s.stack == 0:
                    s.all_in = True

                if new_total == mx:
                    # This is a CALL (or check if need==0 which wouldn't be here). Acted.
                    acted.add(i)
                else:
                    # This is a RAISE. Compute min-raise.
                    raise_inc = new_total - mx

                    # If no previous lower-than-max bet this street, use bb baseline
                    # Otherwise use difference between current max and the highest lower-than-max
                    # For accuracy, reconstruct previous lower-than-max now:
                    lower = [ss.street_bet for ss in seats if (not ss.folded) and not ss.all_in and ss.street_bet < mx]
                    prev_lower = max(lower) if lower else -1
                    mr = min_raise_size(bb, mx, prev_lower)

                    # If this is an all-in that doesn't meet min-raise, it's allowed but does not re-open action.
                    # If not all-in and raise_inc < min-raise -> illegal → fold & revert chips.
                    if (not s.all_in) and raise_inc < mr:
                        # revert
                        s.stack += add
                        s.street_bet = cur_bet
                        s.invested_total -= add
                        # illegal -> fold
                        s.folded = True
                        acted.add(i)
                        continue

                    # Legal raise:
                    # Re-open betting for others who are live and not all-in
                    mx = new_total
                    acted = {i}  # this player acted; others must respond unless they're all-in/folded
                    order = action_order((i + 1) % n, n)

        # Loop until closed
        if not any_to_act_open(seats, acted):
            break

    # Reset street bets for next street; keep invested_total
    for s in seats:
        if not s.folded:
            # Move street_bet into invested_total already done; zero street_bet
            pass
    # Move ALL street_bets back to 0 (they are already accounted in invested_total)
    for s in seats:
        s.street_bet = 0

    # Continue to next street
    return True


# Attach a small cache so we don't reload the same module every action
load_bot.cache = {}  # type: ignore[attr-defined]


def play_hand(
    bet_funcs: list,
    ids: list[str],
    stacks: list[int],
    dealer_idx: int,
    sb: int,
    bb: int,
) -> None:
    """
    Play a single hand, mutating `stacks` in place.
    """
    n = len(ids)
    # Initialize seats
    deck = fresh_deck()
    random.shuffle(deck)
    seats = [SeatState(deal_cards(deck, 2), stacks[i]) for i in range(n)]

    # Post blinds (SB at dealer+1, BB at dealer+2)
    sb_i = (dealer_idx + 1) % n
    bb_i = (dealer_idx + 2) % n

    # Post SB
    sb_pay = min(sb, seats[sb_i].stack)
    seats[sb_i].stack -= sb_pay
    seats[sb_i].street_bet += sb_pay
    seats[sb_i].invested_total += sb_pay
    if seats[sb_i].stack == 0:
        seats[sb_i].all_in = True

    # Post BB
    bb_pay = min(bb, seats[bb_i].stack)
    seats[bb_i].stack -= bb_pay
    seats[bb_i].street_bet += bb_pay
    seats[bb_i].invested_total += bb_pay
    if seats[bb_i].stack == 0:
        seats[bb_i].all_in = True

    community: list[str] = []

    # Make bot id -> function cache
    for pid, func in zip(ids, bet_funcs):
        load_bot.cache[pid] = func  # type: ignore[attr-defined]

    # === Preflop betting ===
    cont = run_betting_round(seats, ids, dealer_idx, sb, bb, community)
    lone = only_one_live(seats)
    if not cont or lone is not None:
        # Award pot to lone player or showdown not needed
        if lone is not None:
            # total pot:
            total = sum(s.invested_total for s in seats)
            seats[lone].stack += total
        # Update outer stacks
        for i in range(n):
            stacks[i] = seats[i].stack
        return

    # === Flop ===
    community += deal_cards(deck, 3)
    cont = run_betting_round(seats, ids, dealer_idx, sb, bb, community)
    lone = only_one_live(seats)
    if not cont or lone is not None:
        if lone is not None:
            seats[lone].stack += sum(s.invested_total for s in seats)
        for i in range(n):
            stacks[i] = seats[i].stack
        return

    # === Turn ===
    community += deal_cards(deck, 1)
    cont = run_betting_round(seats, ids, dealer_idx, sb, bb, community)
    lone = only_one_live(seats)
    if not cont or lone is not None:
        if lone is not None:
            seats[lone].stack += sum(s.invested_total for s in seats)
        for i in range(n):
            stacks[i] = seats[i].stack
        return

    # === River ===
    community += deal_cards(deck, 1)
    cont = run_betting_round(seats, ids, dealer_idx, sb, bb, community)

    # === Showdown ===
    payout_showdown(seats, ids, community)

    # Push back to outer stacks
    for i in range(n):
        stacks[i] = seats[i].stack


# ================================================================
# === CLI RUNNER =================================================
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bots", nargs="+", required=True, help="Paths to bot files, e.g. bot.py bot.py bot.py")
    p.add_argument("--hands", type=int, default=2000)
    p.add_argument("--stack", type=int, default=2000)
    p.add_argument("--sb", type=int, default=5)
    p.add_argument("--bb", type=int, default=10)
    args = p.parse_args()

    # Load once
    bet_funcs = [load_bot(path) for path in args.bots]
    ids = [f"bot{i}" for i in range(len(bet_funcs))]
    stacks = [args.stack for _ in bet_funcs]

    dealer = 0
    for h in range(args.hands):
        play_hand(bet_funcs, ids, stacks, dealer, args.sb, args.bb)
        dealer = (dealer + 1) % len(bet_funcs)
        if h % 50 == 0:
            print(f"Hand {h} → {stacks}")

    print("\n=== FINAL STACKS ===")
    for path, s in zip(args.bots, stacks):
        print(f"{path}: {s}")


if __name__ == "__main__":
    main()