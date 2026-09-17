"""Blackjack, reduced to the decision that matters: hit or stand.

Why blackjack, after poker did not work: the fly's plasticity rule learns a
stimulus -> outcome pairing by repetition, and it failed at poker because a
poker situation almost never recurs - five continuous features and a reacting
opponent - so the noise in any one hand's outcome never averages out.
Blackjack is the opposite case. The state is your total and the dealer's up
card, "16 against a 10" comes up dozens of times in a few hundred hands, there
is no opponent, and for every state there is one correct action. If the rule
can learn anything from a card game, it should be this.

No doubling or splitting, so the fly has exactly two actions. Dealer stands on
soft 17. A natural pays 3:2.

`BASIC_STRATEGY` is the known optimal hit/stand table. It is never shown to
the fly; it is the answer key the fly is scored against afterwards.
"""

from cards import Deck

HIT, STAND = "HIT", "STAND"
ACTIONS = [HIT, STAND]


def card_value(card):
    if card.rank == "A":
        return 11
    if card.rank in "TJQK":
        return 10
    return int(card.rank)


def hand_total(cards):
    """Returns (total, is_soft). Aces count 11 unless that busts, in which
    case they drop to 1 one at a time. A hand is soft while an ace still
    counts as 11 - meaning a hit cannot bust it."""

    total = sum(card_value(c) for c in cards)
    aces = sum(1 for c in cards if c.rank == "A")
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total, aces > 0


class Observation:
    def __init__(self, total, is_soft, dealer_up):
        self.total = total
        self.is_soft = is_soft
        self.dealer_up = dealer_up  # 2..11, ace is 11

    def __repr__(self):
        kind = "soft" if self.is_soft else "hard"
        return f"<{kind} {self.total} vs dealer {self.dealer_up}>"


def play_hand(player, seed=None, log=None):
    """Plays one hand. Returns the player's result in bets: +1 / -1 / 0,
    or +1.5 for a natural blackjack."""

    deck = Deck(seed)
    player_cards = deck.deal(2)
    dealer_cards = deck.deal(2)
    dealer_up = card_value(dealer_cards[0])

    if log is not None:
        log.append(f"player {player_cards}  dealer shows {dealer_cards[0]}")

    total, soft = hand_total(player_cards)
    dealer_total, _ = hand_total(dealer_cards)
    if total == 21:
        return 0.0 if dealer_total == 21 else 1.5

    while True:
        # a player holding 21 is not offered a card - table rule, as in a
        # casino. Without it the fly is asked, and it sometimes says yes.
        if total >= 21:
            break
        action = player.act(Observation(total, soft, dealer_up))
        if log is not None:
            log.append(f"  {action} on {total}{' soft' if soft else ''}")
        if action == STAND:
            break
        player_cards += deck.deal(1)
        total, soft = hand_total(player_cards)
        if total > 21:
            if log is not None:
                log.append(f"  bust with {total}")
            return -1.0

    while True:
        dealer_total, dealer_soft = hand_total(dealer_cards)
        if dealer_total >= 17:
            break
        dealer_cards += deck.deal(1)

    if log is not None:
        log.append(f"  dealer {dealer_cards} = {dealer_total}")

    if dealer_total > 21 or total > dealer_total:
        return 1.0
    if total < dealer_total:
        return -1.0
    return 0.0


def basic_strategy(obs):
    """Optimal hit/stand without doubling or splitting."""

    d = obs.dealer_up
    if obs.is_soft:
        if obs.total >= 19:
            return STAND
        if obs.total == 18:
            return STAND if d <= 8 else HIT
        return HIT
    if obs.total >= 17:
        return STAND
    if obs.total >= 13:
        return STAND if d <= 6 else HIT
    if obs.total == 12:
        return STAND if 4 <= d <= 6 else HIT
    return HIT


BASIC_STRATEGY = basic_strategy


class BasicStrategyPlayer:
    def act(self, obs):
        return basic_strategy(obs)


class AlwaysStand:
    def act(self, obs):
        return STAND


class HitTo17:
    """Mimics the dealer's own rule."""

    def act(self, obs):
        return HIT if obs.total < 17 else STAND


def house_edge(player, n_hands=20000, seed=0):
    return sum(play_hand(player, seed=seed + i) for i in range(n_hands)) / n_hands


if __name__ == "__main__":
    import console_utf8  # noqa: F401

    log = []
    result = play_hand(BasicStrategyPlayer(), seed=3, log=log)
    print("\n".join(log))
    print(f"תוצאה: {result:+.1f}\n")

    print("תוחלת ליד לאורך 20,000 ידיים:")
    for name, player in [("אסטרטגיה בסיסית", BasicStrategyPlayer()),
                         ("מחקה את הדילר", HitTo17()),
                         ("תמיד עומד", AlwaysStand())]:
        print(f"  {name:<18} {house_edge(player):+.4f}")
