# --- randbot.py ---
# A pure random-action bot. Very weak. Good baseline test opponent.

import random

def bet(state):
    # Legal actions: fold (-1), check/call (0 or call_amount), min_raise, all-in
    # We reimplement minimal helpers here to avoid imports.
    bet_money = state["bet_money"]
    idx = state["index_to_action"]
    held = state["held_money"][idx]

    current_bet = max(bet_money)
    my_bet = bet_money[idx]
    to_call = max(0, current_bet - my_bet)

    # Compute min raise (ACM rules)
    lower = [b for b in bet_money if 0 <= b < current_bet]
    if lower:
        prev = max(lower)
        last_raise = current_bet - prev
    else:
        last_raise = state["big_blind"]
    min_raise_size = max(last_raise, state["big_blind"])
    min_raise_amount = max(0, (current_bet + min_raise_size) - my_bet)

    actions = [-1]  # fold always allowed

    if to_call == 0:
        actions.append(0)  # check
    if to_call > 0:
        actions.append(to_call)

    if min_raise_amount <= held and min_raise_amount > 0:
        actions.append(min_raise_amount)

    actions.append(held)  # all-in

    return random.choice(actions)