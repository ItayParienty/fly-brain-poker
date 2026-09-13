"""Heads-up (2 player) Texas Hold'em engine, simplified for AI agents.

Simplifications, and why:
  - Only 3 actions: FOLD / CALL / RAISE. A raise is always a fixed size
    (one big blind), so an agent only has to decide *whether* to put money
    in, not how much. That keeps the decision space small enough for a tiny
    evolved brain to actually learn something.
  - Max 3 raises per street, so two stubborn agents can't raise forever.
  - No stack sizes / all-ins. With the caps above the pot is bounded anyway,
    and a hand's result is just the net chip transfer between the players.
  - Folding when it costs nothing to stay in is auto-converted to a check,
    since folding for free is never right and would just waste evolution's
    time.
"""

from cards import Deck
from hand_eval import best_hand_score, describe

FOLD, CALL, RAISE = "FOLD", "CALL", "RAISE"
ACTIONS = [FOLD, CALL, RAISE]

STREETS = ["preflop", "flop", "turn", "river"]
CARDS_PER_STREET = {"flop": 3, "turn": 1, "river": 1}


class Observation:
    """Everything a player is allowed to know when it's their turn."""

    def __init__(self, hole, community, street, pot, to_call,
                 raises_this_street, is_button):
        self.hole = hole
        self.community = community
        self.street = street
        self.pot = pot
        self.to_call = to_call
        self.raises_this_street = raises_this_street
        self.is_button = is_button

    def __repr__(self):
        return (f"<{self.street} hole={self.hole} board={self.community} "
                f"pot={self.pot} to_call={self.to_call}>")


def _betting_round(players, holes, community, street, committed, first_to_act,
                   street_committed, raise_size, max_raises, button, log):
    current_bet = max(street_committed)
    raises = 0
    acted = set()
    to_act = first_to_act

    while True:
        to_call = current_bet - street_committed[to_act]

        obs = Observation(
            hole=holes[to_act],
            community=list(community),
            street=street,
            pot=sum(committed),
            to_call=to_call,
            raises_this_street=raises,
            is_button=(to_act == button),
        )
        action = players[to_act].act(obs)

        if action == FOLD and to_call == 0:
            action = CALL
        if action == RAISE and raises >= max_raises:
            action = CALL

        if action == FOLD:
            if log is not None:
                log.append(f"  {street}: P{to_act} folds")
            return to_act

        if action == CALL:
            paid = to_call
            street_committed[to_act] += paid
            committed[to_act] += paid
            acted.add(to_act)
            if log is not None:
                log.append(f"  {street}: P{to_act} {'checks' if paid == 0 else f'calls {paid}'}")
        else:  # RAISE
            current_bet += raise_size
            paid = current_bet - street_committed[to_act]
            street_committed[to_act] += paid
            committed[to_act] += paid
            raises += 1
            acted = {to_act}
            if log is not None:
                log.append(f"  {street}: P{to_act} raises to {current_bet}")

        if len(acted) == 2:
            return None
        to_act = 1 - to_act


def play_hand(players, seed=None, button=0, small_blind=1, big_blind=2,
              raise_size=2, max_raises=3, log=None):
    """Plays one hand. Returns [net_chips_p0, net_chips_p1] (sums to zero)."""

    deck = Deck(seed)
    holes = [deck.deal(2), deck.deal(2)]
    community = []

    sb, bb = button, 1 - button
    committed = [0, 0]
    committed[sb] = small_blind
    committed[bb] = big_blind

    if log is not None:
        log.append(f"P0 hole={holes[0]}  P1 hole={holes[1]}  (button=P{button})")

    folder = None
    for street in STREETS:
        if street in CARDS_PER_STREET:
            community += deck.deal(CARDS_PER_STREET[street])
            if log is not None:
                log.append(f"  -- {street}: {community} --")

        if street == "preflop":
            street_committed = [0, 0]
            street_committed[sb] = small_blind
            street_committed[bb] = big_blind
            first = sb
        else:
            street_committed = [0, 0]
            first = bb

        folder = _betting_round(
            players, holes, community, street, committed, first,
            street_committed, raise_size, max_raises, button, log)
        if folder is not None:
            break

    pot = sum(committed)

    if folder is not None:
        winner = 1 - folder
    else:
        scores = [best_hand_score(holes[i] + community) for i in (0, 1)]
        if log is not None:
            log.append(f"  showdown: P0 {describe(scores[0])} vs P1 {describe(scores[1])}")
        if scores[0] > scores[1]:
            winner = 0
        elif scores[1] > scores[0]:
            winner = 1
        else:
            winner = None  # split pot

    if winner is None:
        deltas = [0, 0]
    else:
        deltas = [-committed[0], -committed[1]]
        deltas[winner] += pot

    if log is not None:
        log.append(f"  pot={pot} result={deltas}")
    return deltas


class AlwaysCall:
    def act(self, obs):
        return CALL


class AlwaysRaise:
    def act(self, obs):
        return RAISE


if __name__ == "__main__":
    import console_utf8  # noqa: F401

    log = []
    deltas = play_hand([AlwaysCall(), AlwaysRaise()], seed=7, log=log)
    print("\n".join(log))
    print("\nתוצאה נטו:", deltas)
