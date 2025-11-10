# --- nitbot.py ---
# Extremely tight bot. Folds almost everything. Only plays strong hands.

import random

def rank(card):
    r = card[0].lower()
    order = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,
             '8':8,'9':9,'t':10,'j':11,'q':12,'k':13,'a':14}
    return order[r]

def strong_prehands(hole):
    # Only AA, KK, QQ, JJ, AKs, AKo
    if len(hole) < 2:
        return False
    a, b = hole
    ra = rank(a)
    rb = rank(b)
    if a[1] == b[1]:
        suited = True
    else:
        suited = False

    if ra == rb and ra >= 11:  # JJ, QQ, KK, AA
        return True
    if {ra, rb} == {14, 13}:   # AK
        return True
    return False

def bet(state):
    idx = state["index_to_action"]
    bet_money = state["bet_money"]
    current_bet = max(bet_money)
    my_bet = bet_money[idx]
    to_call = max(0, current_bet - my_bet)
    held = state["held_money"][idx]

    hole = state["player_cards"]

    # Nit: fold everything except very strong prehands
    if not strong_prehands(hole):
        return 0 if to_call == 0 else -1

    # If strong hand:
    if to_call == 0:
        # Small value bet
        return min(state["big_blind"]*3, held)
    else:
        # Call or small raise
        return to_call