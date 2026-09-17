"""The controlled experiment, on blackjack.

Same three groups as for poker - no plasticity, randomly signed reward, real
reward - so the two games can be compared directly.

Three measures, because they answer different questions:

  weighted agreement  - share of the basic strategy table the fly has found,
                        weighting each state by how often it actually comes up
  catastrophic rate   - how often it hits on a hard 17-21, the one mistake that
                        is nearly always punished and should be the easiest to
                        unlearn
  return per hand     - the bottom line; basic strategy manages about -0.03

Real reward has to beat randomly signed reward by more than the run-to-run
spread for the learning to count.
"""

import numpy as np

from blackjack import play_hand
from blackjack_fly import (catastrophic_rate, make_fly, return_per_hand,
                           strategy_table, weighted_agreement)
from connectome import load_circuit
from learning import LearningFly

GROUPS = ["frozen", "shuffled", "real"]
LABELS = {"frozen": "ללא למידה", "shuffled": "תגמול אקראי", "real": "תגמול אמיתי"}


def measure(fly):
    return {
        "agree": weighted_agreement(fly),
        "catastrophe": catastrophic_rate(fly),
        "ret": return_per_hand(fly, 1500),
    }


def run_one(circuit, group, n_hands, hand_seed, learning_rate, rng_seed):
    learner = LearningFly(make_fly(circuit), learning_rate=learning_rate,
                          reward_scale=1.0)
    rng = np.random.default_rng(rng_seed)

    for i in range(n_hands):
        result = play_hand(learner, seed=hand_seed + i)
        if group == "real":
            learner.reward(result)
        elif group == "shuffled":
            learner.reward(abs(result) * rng.choice([-1, 1]))
        else:
            learner.pending.clear()
    return learner


def main(n_hands=3000, repeats=3, learning_rate=0.01, show_table=True):
    circuit = load_circuit()

    print(f"{repeats} חזרות, {n_hands} ידיים, קצב למידה {learning_rate}\n")
    results = {g: [] for g in GROUPS}
    best = None
    for repeat in range(repeats):
        for group in GROUPS:
            learner = run_one(circuit, group, n_hands, repeat * 10000,
                              learning_rate, 100 + repeat)
            m = measure(learner.fly)
            results[group].append(m)
            if group == "real" and (best is None or m["ret"] > best[0]):
                best = (m["ret"], learner.fly)
        print(f"  חזרה {repeat + 1} הסתיימה")

    print(f"\n  {'קבוצה':<14}{'התאמה משוקללת':>16}{'שגיאות קטסטרופליות':>22}{'תוחלת ליד':>14}")
    print("  " + "-" * 66)
    summary = {}
    for g in GROUPS:
        a = np.array([m["agree"] for m in results[g]])
        c = np.array([m["catastrophe"] for m in results[g]])
        r = np.array([m["ret"] for m in results[g]])
        summary[g] = (a, c, r)
        print(f"  {LABELS[g]:<14}{a.mean():>9.1%} ±{a.std():.1%}"
              f"{c.mean():>15.1%} ±{c.std():.1%}"
              f"{r.mean():>+11.3f} ±{r.std():.3f}")

    ra, rc, rr = summary["real"]
    sa, sc, sr = summary["shuffled"]
    print(f"\n  אמיתי מול אקראי:  התאמה {ra.mean() - sa.mean():+.1%}"
          f"   |   קטסטרופות {rc.mean() - sc.mean():+.1%}"
          f"   |   תוחלת {rr.mean() - sr.mean():+.3f}")

    verdicts = []
    if ra.mean() - sa.mean() > 2 * max(ra.std(), sa.std(), 1e-9):
        verdicts.append("ההתאמה לטבלה עלתה מעבר לרעש")
    if sc.mean() - rc.mean() > 2 * max(rc.std(), sc.std(), 1e-9):
        verdicts.append("השגיאות הקטסטרופליות ירדו מעבר לרעש")
    if rr.mean() - sr.mean() > 2 * max(rr.std(), sr.std(), 1e-9):
        verdicts.append("התוחלת ליד עלתה מעבר לרעש")
    print("  " + (" | ".join(verdicts) if verdicts else "אין הפרש מעבר לרעש"))

    if show_table and best is not None:
        print(f"\nהטבלה של הזבוב הטוב ביותר (תוחלת {best[0]:+.3f}):")
        print(strategy_table(best[1]))
    return results


if __name__ == "__main__":
    import console_utf8  # noqa: F401
    main()
