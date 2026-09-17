"""Measures how an untrained fly plays, before any learning happens.

This is the control condition. A fly brain has never been under any pressure
to play poker, so the expectation is that it plays badly and that its choices
barely track how good its cards are. Everything the learning stage claims
later has to be measured against these numbers.
"""

import numpy as np

from flybrain.connectome import load_circuit
from cards.fly import Fly, features
from cards.game import ACTIONS, AlwaysCall, play_hand


class Recorder:
    """Wraps a player and logs (hand strength, action) for every decision."""

    def __init__(self, player):
        self.player = player
        self.log = []

    def act(self, obs):
        action = self.player.act(obs)
        self.log.append((features(obs)["hand_strength"], action))
        return action


def action_distribution(log):
    counts = {a: 0 for a in ACTIONS}
    for _, action in log:
        counts[action] += 1
    total = max(len(log), 1)
    return {a: counts[a] / total for a in ACTIONS}


def strength_split(log, threshold=0.5):
    """How the player acts with weak vs strong holdings."""
    weak = [a for s, a in log if s < threshold]
    strong = [a for s, a in log if s >= threshold]
    return (action_distribution([(0, a) for a in weak]),
            action_distribution([(0, a) for a in strong]))


def run(n_hands=200, seed=0):
    circuit = load_circuit()
    fly = Recorder(Fly(circuit))
    opponent = AlwaysCall()

    chips = 0
    for i in range(n_hands):
        deltas = play_hand([fly, opponent], seed=seed + i, button=i % 2)
        chips += deltas[0]

    print(f"{n_hands} ידיים מול שחקן שתמיד משווה")
    print(f"רווח ממוצע ליד: {chips / n_hands:+.3f} צ'יפים\n")

    dist = action_distribution(fly.log)
    print("התפלגות פעולות:")
    for action in ACTIONS:
        print(f"  {action:<6} {dist[action]:6.1%}")

    weak, strong = strength_split(fly.log)
    print("\nהאם הזבוב מבחין בין ידיים חלשות לחזקות?")
    print(f"{'':<8}{'יד חלשה':>10}{'יד חזקה':>10}")
    for action in ACTIONS:
        print(f"  {action:<6} {weak[action]:>9.1%} {strong[action]:>9.1%}")

    gap = abs(strong[ACTIONS[2]] - weak[ACTIONS[2]])
    print(f"\nפער ההעלאות בין חזק לחלש: {gap:.1%}")
    print("(0% = מתעלם לגמרי מהקלפים. זו נקודת הפתיחה שהלמידה צריכה לשפר)")
    return chips / n_hands


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    run()
