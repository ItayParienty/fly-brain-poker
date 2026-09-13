"""Checks that the simulated circuit behaves like a mushroom body should.

Three properties have to hold before any learning can work on top of it:

  1. SPARSENESS   - a stimulus should activate only a small share of Kenyon
                    cells (5-10% in living flies).
  2. SEPARATION   - two different stimuli should activate mostly different
                    Kenyon cells, otherwise the circuit cannot tell the
                    situations apart and there is nothing to associate.
  3. RELIABILITY  - the same stimulus twice should activate the same Kenyon
                    cells. If the code drifts, an association learned on one
                    trial is meaningless on the next.

Also measures the contribution of APL, the giant inhibitory neuron, by
running the circuit with it silenced.
"""

import numpy as np

from connectome import load_circuit
from lif import SpikingNetwork

STEPS = 60


def kc_pattern(net, circuit, stimulus):
    net.reset()
    counts = net.run(stimulus, steps=STEPS)
    return counts[circuit.indices_of("Kenyon_Cell"), 0] > 0


def odour(circuit, n_neurons, rng, n_active=60):
    chosen = rng.choice(circuit.indices_of("olfactory"), size=n_active, replace=False)
    current = np.zeros((n_neurons, 1), dtype=np.float32)
    current[chosen] = 1.0
    return current


def jaccard(a, b):
    union = (a | b).sum()
    return (a & b).sum() / union if union else 1.0


def main():
    circuit = load_circuit()
    net = SpikingNetwork(circuit.weights)
    rng = np.random.default_rng(1)

    stimuli = [odour(circuit, net.n_neurons, rng) for _ in range(8)]
    patterns = [kc_pattern(net, circuit, s) for s in stimuli]

    sparsity = np.mean([p.mean() for p in patterns])
    print(f"1. דלילות    : {sparsity:.1%} מתאי הקניון פעילים  (יעד ביולוגי: 5-10%)")

    overlaps = [jaccard(patterns[i], patterns[j])
                for i in range(len(patterns)) for j in range(i + 1, len(patterns))]
    print(f"2. הפרדה     : חפיפה ממוצעת בין גירויים שונים {np.mean(overlaps):.1%}  (נמוך = טוב)")

    repeats = [jaccard(kc_pattern(net, circuit, stimuli[0]), patterns[0]) for _ in range(3)]
    print(f"3. יציבות    : חפיפה של אותו גירוי עם עצמו {np.mean(repeats):.1%}  (יעד: 100%)")

    # silence APL and re-measure sparseness
    apl = [i for i in circuit.indices_of("MBIN")
           if circuit.weights[circuit.indices_of("Kenyon_Cell"), i].nnz > 100]
    muted = circuit.weights.tolil(copy=True)
    for i in apl:
        muted[:, i] = 0
    net_no_apl = SpikingNetwork(muted.tocsr())
    without = np.mean([kc_pattern(net_no_apl, circuit, s).mean() for s in stimuli[:4]])
    print(f"\nAPL מושתק   : {without:.1%} מתאי הקניון פעילים  "
          f"(לעומת {sparsity:.1%} עם APL)")


if __name__ == "__main__":
    import console_utf8  # noqa: F401
    main()
