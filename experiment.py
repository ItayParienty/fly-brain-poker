"""The controlled experiment: does dopamine-gated plasticity teach the fly?

Repeats the three-group comparison over several independent flies, because a
single run cannot separate a small effect from noise. Each repeat uses a
different sequence of hands, and every group within a repeat sees exactly the
same sequence.

Reported measure is discrimination - how much more often the fly raises with a
strong hand than a weak one. Chips per hand is reported too but is the weaker
measure: against any given opponent there are ways to win chips that have
nothing to do with reading your cards.
"""

import numpy as np

from connectome import load_circuit
from control import discrimination
from fly import Fly
from game import play_hand
from learning import LearningFly
from opponents import TightOpponent
from train import evaluate

GROUPS = ["frozen", "shuffled", "real"]
LABELS = {"real": "תגמול אמיתי", "shuffled": "תגמול אקראי", "frozen": "ללא למידה"}


def run_one(circuit, opponent, group, n_hands, hand_seed, learning_rate, rng_seed):
    learner = LearningFly(Fly(circuit), learning_rate=learning_rate)
    rng = np.random.default_rng(rng_seed)

    for i in range(n_hands):
        deltas = play_hand([learner, opponent], seed=hand_seed + i, button=i % 2)
        if group == "real":
            learner.reward(deltas[0])
        elif group == "shuffled":
            learner.reward(abs(deltas[0]) * rng.choice([-1, 1]))
        else:
            learner.pending.clear()

    chips, log = evaluate(learner.fly, opponent)
    return chips, discrimination(log)


def main(n_hands=1200, repeats=4, learning_rate=0.002):
    circuit = load_circuit()
    opponent = TightOpponent()

    results = {group: {"chips": [], "disc": []} for group in GROUPS}
    print(f"{repeats} חזרות, {n_hands} ידיים לכל זבוב, קצב למידה {learning_rate}\n")
    print(f"  {'חזרה':<6}" + "".join(f"{LABELS[g]:>16}" for g in GROUPS))
    print("  " + "-" * 56)

    for repeat in range(repeats):
        row = []
        for group in GROUPS:
            chips, disc = run_one(circuit, opponent, group, n_hands,
                                  hand_seed=repeat * 10000,
                                  learning_rate=learning_rate,
                                  rng_seed=100 + repeat)
            results[group]["chips"].append(chips)
            results[group]["disc"].append(disc)
            row.append(f"{disc:>15.1%}")
        print(f"  {repeat + 1:<6}" + "".join(row))

    print("\n  אבחנה בין ידיים (ממוצע וסטיית תקן):")
    for group in GROUPS:
        values = np.array(results[group]["disc"])
        print(f"    {LABELS[group]:<14} {values.mean():+7.1%}  ±{values.std():.1%}"
              f"   |  צ׳יפים {np.mean(results[group]['chips']):+.3f}")

    real = np.array(results["real"]["disc"])
    shuffled = np.array(results["shuffled"]["disc"])
    frozen = np.array(results["frozen"]["disc"])

    gain_vs_shuffled = real.mean() - shuffled.mean()
    gain_vs_frozen = real.mean() - frozen.mean()
    spread = max(real.std(), shuffled.std(), 1e-9)

    print(f"\n  אמיתי מול אקראי: {gain_vs_shuffled:+.1%}"
          f"   |   אמיתי מול מוקפא: {gain_vs_frozen:+.1%}"
          f"   |   רעש בין חזרות: ±{spread:.1%}")

    if gain_vs_shuffled > 2 * spread and gain_vs_frozen > 0:
        print("  ההפרש גדול מהרעש: התגמול האמיתי לימד את הזבוב לקרוא את הקלפים.")
    else:
        print("  ההפרש אינו גדול מהרעש - אין ראיה ללמידה של הקלפים.")

    return results


if __name__ == "__main__":
    import console_utf8  # noqa: F401
    main()
