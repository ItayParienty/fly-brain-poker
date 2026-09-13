"""Builds a runnable network out of the FlyWire connectome tables.

The subcircuit used here is the fly's mushroom body learning pathway, which
is the part of the brain that actually forms associations between a stimulus
and a reward:

    olfactory  ->  ALPN  ->  Kenyon cells  ->  MBON
                                   ^
                                   |  dopaminergic neurons (DAN) report
                                   |  reward and reshape the KC->MBON weights

Two things the connectome gives us directly, and one it does not:

  - syn_count  : how many synapses form a connection. Used as the relative
                 strength of that connection. This is measured data.
  - nt_type    : the neurotransmitter, which tells us the SIGN of the
                 connection (excitatory or inhibitory). This is a per-neuron
                 machine-learning prediction in FlyWire, not a measurement,
                 but it is far better than guessing.
  - It does NOT give absolute synaptic strength in physical units. No
    connectome does. `weight_scale` below is therefore a free parameter that
    has to be calibrated until the network fires in a sensible regime - see
    calibrate.py. Every connectome simulation faces this and it is the main
    reason results are qualitative rather than predictive.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "circuit_cache.npz"

# Synapse counts converted to membrane current. Determined empirically by
# calibrate.py: this is the value at which a stimulus activates 5-10% of
# Kenyon cells, the sparsity measured in living flies. See calibrate.py.
DEFAULT_WEIGHT_SCALE = 0.026

# The mushroom body learning pathway, in signal-flow order.
#
# MBIN is included because it contains APL - a single giant GABAergic neuron
# per hemisphere (root ids ...624547622 and ...613583001, labelled "APL-RHS"
# and "APL-LHS" by FlyWire annotators) that makes ~96,000 inhibitory synapses
# onto Kenyon cells. APL is what keeps Kenyon cell activity sparse: without
# it every stimulus drives every Kenyon cell, all stimuli look identical to
# the downstream MBONs, and no association can be learned.
CIRCUIT_CLASSES = ["olfactory", "ALPN", "Kenyon_Cell", "MBIN", "MBON", "DAN"]

# Sign of fast synaptic transmission.
# GLUT is listed as inhibitory: in Drosophila, glutamate most often acts on
# GluCl chloride channels, unlike in vertebrates where it is excitatory.
SYNAPSE_SIGN = {"ACH": +1.0, "GABA": -1.0, "GLUT": -1.0}

# Neuromodulators. These do not drive spiking directly on a millisecond
# timescale, so they are excluded from the fast weight matrix and handled
# separately - dopamine is the learning signal, not an input current.
MODULATORS = {"DA", "SER", "OCT"}


class Circuit:
    """A connectome subcircuit, ready to simulate."""

    def __init__(self, neuron_ids, neuron_classes, weights, modulatory,
                 neuron_nt=None):
        self.neuron_ids = neuron_ids          # FlyWire root_id per index
        self.neuron_classes = neuron_classes  # class name per index
        self.weights = weights                # signed sparse matrix [post, pre]
        self.modulatory = modulatory          # DA/SER/OCT edges, same layout
        self.neuron_nt = neuron_nt            # each neuron's own transmitter

    @property
    def n_neurons(self):
        return len(self.neuron_ids)

    def indices_of(self, class_name):
        """Row/column indices of every neuron in a given class."""
        return np.flatnonzero(self.neuron_classes == class_name)

    def indices_of_nt(self, class_name, nt_type):
        """Neurons of a class that release a given neurotransmitter."""
        return np.flatnonzero((self.neuron_classes == class_name)
                              & (self.neuron_nt == nt_type))

    def summary(self):
        lines = [f"{self.n_neurons:,} neurons, {self.weights.nnz:,} fast connections"]
        for name in CIRCUIT_CLASSES:
            lines.append(f"  {name:<12} {len(self.indices_of(name)):>6,}")
        excitatory = (self.weights.data > 0).sum()
        lines.append(f"  excitatory {excitatory:,} / inhibitory {self.weights.nnz - excitatory:,}")
        return "\n".join(lines)


def build_circuit(classes=None, weight_scale=1.0, data_dir=DATA_DIR):
    """Reads the FlyWire CSVs and assembles the signed connection matrix."""

    classes = classes or CIRCUIT_CLASSES

    annotations = pd.read_csv(data_dir / "classification.csv.gz")
    annotations = annotations[annotations["class"].isin(classes)]

    neuron_ids = annotations.root_id.to_numpy()
    neuron_classes = annotations["class"].to_numpy()
    index_of = {root_id: i for i, root_id in enumerate(neuron_ids)}

    transmitters = pd.read_csv(data_dir / "neurons.csv.gz")[["root_id", "nt_type"]]
    nt_of = dict(zip(transmitters.root_id, transmitters.nt_type))
    neuron_nt = np.array([str(nt_of.get(rid, "UNK")) for rid in neuron_ids])

    connections = pd.read_csv(data_dir / "connections.csv.gz")
    inside = (connections.pre_root_id.isin(index_of)
              & connections.post_root_id.isin(index_of))
    connections = connections[inside]

    pre = connections.pre_root_id.map(index_of).to_numpy()
    post = connections.post_root_id.map(index_of).to_numpy()
    counts = connections.syn_count.to_numpy(dtype=np.float32)
    nt = connections.nt_type.to_numpy()

    is_modulator = np.isin(nt, list(MODULATORS))
    signs = np.array([SYNAPSE_SIGN.get(t, 0.0) for t in nt], dtype=np.float32)

    shape = (len(neuron_ids), len(neuron_ids))
    fast = sparse.csr_matrix(
        (counts[~is_modulator] * signs[~is_modulator] * weight_scale,
         (post[~is_modulator], pre[~is_modulator])), shape=shape)
    modulatory = sparse.csr_matrix(
        (counts[is_modulator], (post[is_modulator], pre[is_modulator])), shape=shape)

    return Circuit(neuron_ids, neuron_classes, fast, modulatory, neuron_nt)


def load_circuit(weight_scale=DEFAULT_WEIGHT_SCALE, rebuild=False, data_dir=DATA_DIR):
    """Same as build_circuit, but caches the parsed result (the CSV parse
    takes ~30s, the cached load takes well under a second)."""

    cache = data_dir / CACHE_PATH.name
    if cache.exists() and not rebuild:
        blob = np.load(cache, allow_pickle=True)
        shape = tuple(blob["shape"])
        fast = sparse.csr_matrix(
            (blob["fast_data"], blob["fast_indices"], blob["fast_indptr"]), shape=shape)
        modulatory = sparse.csr_matrix(
            (blob["mod_data"], blob["mod_indices"], blob["mod_indptr"]), shape=shape)
        fast.data = fast.data * weight_scale
        return Circuit(blob["neuron_ids"], blob["neuron_classes"], fast, modulatory,
                       blob["neuron_nt"])

    circuit = build_circuit(weight_scale=1.0, data_dir=data_dir)
    np.savez_compressed(
        cache,
        neuron_ids=circuit.neuron_ids,
        neuron_classes=circuit.neuron_classes,
        neuron_nt=circuit.neuron_nt,
        shape=np.array(circuit.weights.shape),
        fast_data=circuit.weights.data,
        fast_indices=circuit.weights.indices,
        fast_indptr=circuit.weights.indptr,
        mod_data=circuit.modulatory.data,
        mod_indices=circuit.modulatory.indices,
        mod_indptr=circuit.modulatory.indptr,
    )
    circuit.weights.data = circuit.weights.data * weight_scale
    return circuit


if __name__ == "__main__":
    import console_utf8  # noqa: F401

    circuit = load_circuit(rebuild=True)
    print(circuit.summary())
