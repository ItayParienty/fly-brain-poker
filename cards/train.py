"""Runs a fly through repeated hands and reports whether it learns anything.

Every hand ends with a call to reward(), which is the fly's food: winning
chips is sugar, losing them is not. Nothing else is supervised - the fly is
never told which action was correct, only whether the hand as a whole went
well, exactly as a fly is only ever told whether the odour it approached
turned out to be sweet.
"""

import numpy as np

from cards.baseline import Recorder, action_distribution, strength_split
from flybrain.connectome import load_circuit
from cards.fly import Fly
from cards.game import ACTIONS, AlwaysCall, play_hand
from flybrain.learning import LearningFly


def evaluate(player, opponent, n_hands=120, seed=9000):
    """Plays a fixed set of hands without learning, to measure current skill."""

    recorder = Recorder(player)
    chips = 0
    for i in range(n_hands):
        deltas = play_hand([recorder, opponent], seed=seed + i, button=i % 2)
        chips += deltas[0]
    return chips / n_hands, recorder.log


def report(label, chips_per_hand, log):
    dist = action_distribution(log)
    weak, strong = strength_split(log)
    print(f"{label:<12} {chips_per_hand:+7.3f}  "
          f"FOLD {dist['FOLD']:5.1%}  CALL {dist['CALL']:5.1%}  RAISE {dist['RAISE']:5.1%}"
          f"   |  העלאה: חלש {weak['RAISE']:5.1%} / חזק {strong['RAISE']:5.1%}")


def train(n_hands=600, learning_rate=0.02, seed=0, eval_every=200):
    circuit = load_circuit()
    learner = LearningFly(Fly(circuit, name="learner"), learning_rate=learning_rate)
    opponent = AlwaysCall()

    print(f"{'':<12} {'רווח/יד':>7}  {'התפלגות פעולות':^34}")
    before, log = evaluate(learner.fly, opponent)
    report("לפני אימון", before, log)

    for i in range(n_hands):
        deltas = play_hand([learner, opponent], seed=seed + i, button=i % 2)
        learner.reward(deltas[0])

        if (i + 1) % eval_every == 0:
            score, log = evaluate(learner.fly, opponent)
            report(f"אחרי {i + 1:>4}", score, log)

    after, log = evaluate(learner.fly, opponent)
    print()
    print(f"סינפסות שהוחלשו: {learner.total_depression():.1%} מהעוצמה המקורית")
    print(f"שיפור: {after - before:+.3f} צ'יפים ליד")
    return before, after


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    train()
