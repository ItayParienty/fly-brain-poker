"""Control experiments: is the improvement actually learning?

A number going up is not evidence on its own. Any process that keeps modifying
a circuit changes its behaviour, and some of those changes pay off by luck.
Three groups play identical hands:

  LEARNING  - dopamine reports the real result of the hand
  SHUFFLED  - dopamine fires with the same magnitudes, but the sign is drawn
              at random and has nothing to do with what happened
  FROZEN    - no plasticity at all

Two things are measured, and they are not the same thing:

  chips/hand       - how much money it makes. Against a weak opponent this can
                     rise for reasons that have nothing to do with the cards.
  discrimination   - how much more often it raises with a strong hand than a
                     weak one. This is the quantity that can only improve if
                     the circuit has actually associated card strength with an
                     action, so it is the real test.
"""

import numpy as np

from baseline import Recorder, strength_split
from connectome import load_circuit
from fly import Fly
from game import play_hand
from learning import LearningFly
from opponents import AlwaysCall, TightOpponent
from train import evaluate


def discrimination(log):
    """Raise rate with strong hands minus raise rate with weak hands."""
    weak, strong = strength_split(log)
    return strong["RAISE"] - weak["RAISE"]


def run_group(label, opponent, n_hands, reward_mode, seed=0, rng_seed=7):
    circuit = load_circuit()
    learner = LearningFly(Fly(circuit, name=label))
    rng = np.random.default_rng(rng_seed)

    chips_before, log_before = evaluate(learner.fly, opponent)
    for i in range(n_hands):
        deltas = play_hand([learner, opponent], seed=seed + i, button=i % 2)
        if reward_mode == "real":
            learner.reward(deltas[0])
        elif reward_mode == "shuffled":
            learner.reward(abs(deltas[0]) * rng.choice([-1, 1]))
        else:
            learner.pending.clear()
    chips_after, log_after = evaluate(learner.fly, opponent)

    print(f"  {label:<10} {chips_before:+7.3f} -> {chips_after:+7.3f}"
          f"   {discrimination(log_before):+7.1%} -> {discrimination(log_after):+7.1%}")
    return chips_after - chips_before, discrimination(log_after) - discrimination(log_before)


def main(n_hands=600, opponent=None, opponent_name="יריב הדוק"):
    opponent = opponent or TightOpponent()
    print(f"שלוש קבוצות, {n_hands} ידיים מול {opponent_name}, אותן חלוקות\n")
    print(f"  {'קבוצה':<10} {'צ׳יפים ליד':>19}       {'אבחנה בין ידיים':>17}")
    print("  " + "-" * 62)

    results = {}
    for label, mode in [("למידה", "real"), ("אקראי", "shuffled"), ("מוקפא", "frozen")]:
        results[label] = run_group(label, opponent, n_hands, mode)

    real_chips, real_disc = results["למידה"]
    fake_chips, fake_disc = results["אקראי"]

    print()
    print(f"תגמול אמיתי מול אקראי:  צ׳יפים {real_chips - fake_chips:+.3f}"
          f"   |   אבחנה {real_disc - fake_disc:+.1%}")
    if real_disc > fake_disc + 0.03:
        print("הזבוב שקיבל תגמול אמיתי מבחין בין ידיים טוב יותר. זו למידה.")
    else:
        print("האבחנה לא השתפרה מעבר לרעש - השינוי אינו למידה של הקלפים.")


if __name__ == "__main__":
    import console_utf8  # noqa: F401
    main()
