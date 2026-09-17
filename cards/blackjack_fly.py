"""The mushroom body circuit playing blackjack.

Same brain, same encoder, same plasticity rule as the poker fly. Only the
game and its three features change:

    total     - the player's hand total, 4..21
    soft      - whether an ace is still counting as 11
    dealer    - the dealer's visible card, 2..11

and two actions instead of three. HIT is mapped to the cholinergic MBONs,
STAND to the glutamatergic ones, following the same approach / avoid reading
of their valence used for poker.

The score is agreement with the basic strategy table across the grid of every
hard and soft total against every dealer card - i.e. what fraction of the
answer key the fly has found. This is a much sharper measure than return per
hand, because the return is dominated by the deal and only weakly by the
decision.
"""

import numpy as np

from cards.blackjack import (ACTIONS, HIT, STAND, BasicStrategyPlayer, Observation,
                       basic_strategy, play_hand)
from flybrain.connectome import load_circuit
from cards.fly import Fly

ACTION_POOLS = {HIT: "ACH", STAND: "GLUT"}

FEATURE_BUDGET = {"total": 28, "dealer": 20, "soft": 12}


def featurize(obs):
    return {
        "total": (obs.total - 4) / 17.0,
        "dealer": (obs.dealer_up - 2) / 9.0,
        "soft": 1.0 if obs.is_soft else 0.0,
    }


def make_fly(circuit, name="fly", calibrate=True):
    return Fly(circuit, name=name, calibrate=calibrate,
               action_pools=ACTION_POOLS, featurize=featurize, budget=FEATURE_BUDGET)


def decision_grid():
    """Every state a player can be asked to act in."""
    states = []
    for dealer in range(2, 12):
        for total in range(5, 21):        # 21 is never asked (table rule)
            states.append(Observation(total, False, dealer))
        for total in range(13, 21):
            states.append(Observation(total, True, dealer))
    return states


def agreement(player, states=None):
    """Fraction of states where the player matches basic strategy, every
    state counted once. Blind to how often a state actually comes up."""
    states = states or decision_grid()
    matches = sum(player.act(s) == basic_strategy(s) for s in states)
    return matches / len(states)


_state_frequency = None


def state_frequency(n_hands=6000, seed=777):
    """How often each decision state arises in real play under basic
    strategy. Cached; used to weight agreement by what actually matters."""

    global _state_frequency
    if _state_frequency is not None:
        return _state_frequency

    counts = {}

    class Counter:
        def act(self, obs):
            key = (obs.total, obs.is_soft, obs.dealer_up)
            counts[key] = counts.get(key, 0) + 1
            return basic_strategy(obs)

    for i in range(n_hands):
        play_hand(Counter(), seed=seed + i)
    total = sum(counts.values())
    _state_frequency = {k: v / total for k, v in counts.items()}
    return _state_frequency


def weighted_agreement(player):
    """Agreement with basic strategy, weighted by how often each state is
    actually faced. A wrong answer on "hard 16 vs 10" costs far more here than
    one on "soft 13 vs 4", which is the right way round."""

    freq = state_frequency()
    score = 0.0
    for (total, soft, dealer), weight in freq.items():
        obs = Observation(total, soft, dealer)
        if player.act(obs) == basic_strategy(obs):
            score += weight
    return score


def catastrophic_rate(player):
    """Share of hard 17-20 states, weighted by frequency, where the player
    hits - a decision that busts most of the time and has no upside."""

    freq = state_frequency()
    hit_weight = total_weight = 0.0
    for (total, soft, dealer), weight in freq.items():
        if not soft and total >= 17:
            total_weight += weight
            if player.act(Observation(total, soft, dealer)) == HIT:
                hit_weight += weight
    return hit_weight / total_weight if total_weight else 0.0


def strategy_table(player):
    """Renders the player's policy as the familiar hit/stand chart, with a
    mark on every cell that disagrees with basic strategy."""

    dealers = list(range(2, 12))
    header = "        " + " ".join(f"{'A' if d == 11 else d:>2}" for d in dealers)
    lines = [header]
    for soft, totals, label in [(False, range(20, 4, -1), "hard"),
                                (True, range(20, 12, -1), "soft")]:
        lines.append(f"  {label}")
        for total in totals:
            cells = []
            for d in dealers:
                obs = Observation(total, soft, d)
                mine, right = player.act(obs), basic_strategy(obs)
                cell = "H" if mine == HIT else "S"
                cells.append(f"{cell:>2}" if mine == right else f"{cell.lower() + '*':>2}")
            lines.append(f"  {total:>4}  " + " ".join(cells))
    return "\n".join(lines)


def return_per_hand(player, n_hands=4000, seed=50000):
    return sum(play_hand(player, seed=seed + i) for i in range(n_hands)) / n_hands


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401

    circuit = load_circuit()
    fly = make_fly(circuit)
    print("זבוב לא מאומן:")
    print(f"  התאמה לאסטרטגיה הבסיסית: {agreement(fly):.1%}")
    print(f"  תוחלת ליד:                {return_per_hand(fly, 2000):+.4f}"
          f"   (אסטרטגיה בסיסית: {return_per_hand(BasicStrategyPlayer(), 2000):+.4f})")
    print()
    print(strategy_table(fly))
    print("\n  אות קטנה + כוכבית = לא תואם לאסטרטגיה הבסיסית")
