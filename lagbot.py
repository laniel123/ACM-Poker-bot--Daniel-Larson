# --- lagbot.py ---
# Loose-aggressive: opens wide, bets often, steals blinds.

import random

def bet(state):
    idx = state["index_to_action"]
    bet_money = state["bet_money"]
    current_bet = max(bet_money)
    my_bet = bet_money[idx]
    to_call = max(0, current_bet - my_bet)
    held = state["held_money"][idx]
    bb = state["big_blind"]

    hole = state["player_cards"]

    # Loose open: play ~40% of hands
    ranks = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,
             '9':9,'t':10,'j':11,'q':12,'k':13,'a':14}
    hi = max(ranks[hole[0][0].lower()], ranks[hole[1][0].lower()])
    lo = min(ranks[hole[0][0].lower()], ranks[hole[1][0].lower()])
    suited = (hole[0][1] == hole[1][1])
    gap = hi - lo

    playable = (
        hi >= 12 or                                     # broadway-heavy
        (suited and hi >= 10 and lo >= 6) or            # suited connectors
        random.random() < 0.2                           # occasional random VPIP
    )

    if not playable:
        return 0 if to_call == 0 else -1

    # If facing raise → LAG bot calls wide
    if to_call > 0:
        # ~20% chance to reraise as bluff
        if random.random() < 0.20 and (my_bet + held) >= (current_bet + bb*3):
            return min(bb*3, held)
        else:
            return min(to_call, held)

    # Not facing raise → open aggressively
    size = bb * random.choice([2,3,4])
    return min(size, held)