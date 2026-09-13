"""Finds the synaptic weight scale that puts the circuit in a biological regime.

The connectome measures how MANY synapses join two neurons, but not how
strongly each one pushes. `weight_scale` converts synapse counts into
membrane current and is the one number we have to determine ourselves.

The target is taken from experiment: in a real fly, any given odour drives
roughly 5-10% of Kenyon cells to spike (Turner et al. 2008; Honegger et al.
2011). That sparse code is what lets the mushroom body tell stimuli apart.
Too much drive and every odour lights up every cell (all situations look
identical); too little and nothing fires at all.
"""

import numpy as np

from connectome import load_circuit
from lif import SpikingNetwork

TARGET_SPARSITY = (0.05, 0.10)


def make_stimulus(circuit, n_neurons_total, rng, n_active=60, strength=1.0,
                  n_agents=1):
    """Builds an input current that activates a random subset of olfactory
    neurons - the network's equivalent of 'a smell'."""

    olfactory = circuit.indices_of("olfactory")
    chosen = rng.choice(olfactory, size=n_active, replace=False)
    current = np.zeros((n_neurons_total, n_agents), dtype=np.float32)
    current[chosen] = strength
    return current


def measure(circuit, weight_scale, steps=60, n_odours=5, seed=0):
    """Returns the mean fraction of Kenyon cells that spike for a stimulus."""

    rng = np.random.default_rng(seed)
    net = SpikingNetwork(circuit.weights * weight_scale)
    kc = circuit.indices_of("Kenyon_Cell")
    mbon = circuit.indices_of("MBON")

    kc_fractions, mbon_fractions, patterns = [], [], []
    for _ in range(n_odours):
        net.reset()
        stimulus = make_stimulus(circuit, net.n_neurons, rng)
        counts = net.run(stimulus, steps=steps)
        active = counts[:, 0] > 0
        kc_fractions.append(active[kc].mean())
        mbon_fractions.append(active[mbon].mean())
        patterns.append(active[kc])

    # how distinguishable are two different odours? 0 = identical, 1 = disjoint
    overlaps = []
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            a, b = patterns[i], patterns[j]
            union = (a | b).sum()
            overlaps.append((a & b).sum() / union if union else 1.0)

    return np.mean(kc_fractions), np.mean(mbon_fractions), np.mean(overlaps)


if __name__ == "__main__":
    import console_utf8  # noqa: F401

    circuit = load_circuit(weight_scale=1.0)  # raw synapse counts; the sweep scales them
    print("סורק ערכי weight_scale...\n")
    print(f"{'scale':>9} {'תאי קניון פעילים':>18} {'MBON פעילים':>14} {'חפיפה בין ריחות':>17}")
    print("-" * 62)

    for scale in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        kc, mbon, overlap = measure(circuit, scale)
        flag = "  <-- ביולוגי" if TARGET_SPARSITY[0] <= kc <= TARGET_SPARSITY[1] else ""
        print(f"{scale:>9} {kc:>17.1%} {mbon:>13.1%} {overlap:>16.1%}{flag}")
