# --- fishbot.py ---
# Loose-passive calling station. Calls too much, bets too little.

import random

def bet(state):
    idx = state["index_to_action"]
    bet_money = state["bet_money"]
    current_bet = max(bet_money)
    my_bet = bet_money[idx]
    to_call = max(0, current_bet - my_bet)
    held = state["held_money"][idx]
    bb = state["big_blind"]

    # Fish behavior:
    # – Calls ~80% of bets if affordable
    # – Almost never raises
    # – Occasionally donks a small bet when checked to

    # If facing a bet
    if to_call > 0:
        if to_call <= held and random.random() < 0.80:
            return to_call   # calling station
        return -1            # fold

    # No bet → check or tiny donk
    if random.random() < 0.15:
        return min(bb, held)
    return 0