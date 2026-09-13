"""The textbook fly experiment, run on the simulated circuit.

Poker turned out to be a poor teacher: training with the real outcome of a
hand produced no more discrimination than training with a randomly signed
one. That leaves two possibilities, and they need separating before anything
else is worth trying.

  1. the plasticity implementation is broken, or
  2. the implementation is fine and poker is simply not the kind of problem
     this rule can solve.

So the rule is given the task it evolved for. One odour is paired with
punishment, another is left alone, and the question is whether the punished
odour's drive onto the approach pathway falls while the other one's does not.
This is the classic aversive conditioning protocol (Tully & Quinn 1985), and
a working mushroom body must pass it.

Nothing about poker is involved here.
"""

import numpy as np

from connectome import load_circuit
from fly import ACTION_POOLS, Fly
from game import RAISE
from learning import Experience, LearningFly


def make_odour(circuit, n_neurons, rng, n_active=60):
    chosen = rng.choice(circuit.indices_of("olfactory"), size=n_active, replace=False)
    current = np.zeros((n_neurons, 1), dtype=np.float32)
    current[chosen] = 1.0
    return current


def approach_drive(learner, odour):
    """How strongly this odour drives the approach pathway."""
    rates = learner.fly._pool_rates(odour)
    return rates[RAISE]


def pair_with_punishment(learner, odour, rate):
    """One training trial: the odour is presented and punishment arrives, so
    dopamine depresses the approach pathway for whichever Kenyon cells the
    odour just activated."""

    learner.fly._pool_rates(odour)
    active_kc = learner.fly.last_counts[learner.kc_indices] > 0
    learner.synapses.depress(learner.fly.net.weights, RAISE, active_kc, rate)


def main(n_trials=12, learning_rate=0.05, repeats=3):
    circuit = load_circuit()
    print(f"התניה קלאסית: ריח אחד מזווג לעונש, ריח שני לא. {n_trials} חזרות אימון.\n")
    print(f"  {'ניסוי':<7}{'ריח מעונש':>22}{'ריח ביקורת':>22}")
    print(f"  {'':<7}{'לפני':>10}{'אחרי':>11}{'לפני':>11}{'אחרי':>11}")
    print("  " + "-" * 51)

    punished_drop, control_drop = [], []
    for repeat in range(repeats):
        rng = np.random.default_rng(repeat)
        learner = LearningFly(Fly(circuit, calibrate=False), learning_rate=learning_rate,
                              recovery=0.0)
        punished = make_odour(circuit, learner.fly.net.n_neurons, rng)
        control = make_odour(circuit, learner.fly.net.n_neurons, rng)

        before_p = approach_drive(learner, punished)
        before_c = approach_drive(learner, control)

        for _ in range(n_trials):
            pair_with_punishment(learner, punished, learning_rate)

        after_p = approach_drive(learner, punished)
        after_c = approach_drive(learner, control)

        punished_drop.append((before_p - after_p) / max(before_p, 1e-9))
        control_drop.append((before_c - after_c) / max(before_c, 1e-9))
        print(f"  {repeat + 1:<7}{before_p:>10.2f}{after_p:>11.2f}"
              f"{before_c:>11.2f}{after_c:>11.2f}")

    print(f"\n  ירידה בריח המעונש : {np.mean(punished_drop):+.1%}")
    print(f"  ירידה בריח הביקורת: {np.mean(control_drop):+.1%}")

    if np.mean(punished_drop) > np.mean(control_drop) + 0.05:
        print("\n  הזבוב למד להימנע מהריח המעונש, והריח השני כמעט לא הושפע.")
        print("  מנגנון הלמידה תקין - הבעיה בפוקר היא באופי המשימה, לא במימוש.")
    else:
        print("\n  גם הריח המעונש לא נחלש יותר מהביקורת - המימוש עצמו שבור.")


if __name__ == "__main__":
    import console_utf8  # noqa: F401
    main()
