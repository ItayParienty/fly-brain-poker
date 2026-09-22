"""The FlyWire optic lobe with flyvis's cell-type parameters and dynamics.

flyvis (Lappalainen et al. 2024) simulates each optic-lobe neuron as a
passive membrane with a rectified graded output:

    tau_i dV_i/dt = -V_i + bias_i + sum_j w_ij relu(V_j) + input_i
    w_ij = sign(type_j -> type_i) * N_ij * strength(type_j -> type_i)

with a time constant and resting potential per cell type and a strength
per pair of types, all fitted on optic flow.  This module applies exactly
those numbers (flyvis_params.json) to FlyWire's wiring: N_ij is FlyWire's
synapse count, and the type names are FlyWire's, matched to flyvis's.

Where FlyWire has a type flyvis does not (about 20,000 of the 47,000
neurons - the LC projection neurons, Dm/Pm interneurons, and so on), the
time constant is flyvis's floor (20 ms), the synaptic strength is flyvis's
own initial rule (0.01 divided by the pair's mean synapse count, so every
type pair starts with the same total weight), and the resting potential
is fitted to a blank screen (Eye.calibrate) as before.

    circuit, dyn = flyvis_circuit()          # weights in flyvis units, plus tau/bias per neuron
    net = FlyvisNetwork(circuit.weights, dyn)
"""
import hashlib

import numpy as np
import torch
from scipy import sparse

from flybrain import flyvis_params
from flybrain.connectome import DATA_DIR
from flybrain.vision import load_visual_circuit

# FlyWire names -> flyvis names (CT1 is handled by split_ct1 below)
ALIASES = {"R1-6": ["R1", "R2", "R3", "R4", "R5", "R6"], "Am1": ["Am"]}
FLOOR_TAU = 0.02          # flyvis clamps time constants at its integration step
INIT_SCALE = 0.01         # flyvis's initial strength rule: scale / <N> per type pair


def _names(t):
    return [t] + ALIASES.get(t, [])


MEDULLA_PARTNERS = ("Mi", "Tm3", "T4", "C2", "C3", "L", "Dm", "Pm", "R", "TmY")   # CT1's medulla compartment talks to these


def split_ct1(c):
    """Replace the single CT1 neuron by one unit per column and compartment.

    CT1 is one giant cell per hemisphere, but it is electrotonically
    compartmentalised: each column's branch integrates that column's inputs
    and feeds that column's T4 (in the medulla) or T5 (in the lobula) on its
    own.  flyvis models it as two columnar types, CT1(M10) and CT1(Lo1), and
    its parameters assume that.  As one summed unit it takes input from all
    ~800 columns at once, saturates, and inhibits every T4 the same way.
    """
    types = c.neuron_classes.astype(str)
    ct1 = np.flatnonzero(types == "CT1")
    if len(ct1) != 1:
        return c
    old = int(ct1[0])
    n, ncol = c.n_neurons, c.n_columns
    n_new = n - 1 + 2 * ncol
    keep = np.array([i for i in range(n) if i != old])
    remap = -np.ones(n, dtype=np.int64); remap[keep] = np.arange(n - 1)
    m10 = lambda k: n - 1 + k                # CT1(M10) compartment of column k
    lo1 = lambda k: n - 1 + ncol + k         # CT1(Lo1) compartment of column k
    w = c.weights.tocoo()
    rows, cols, data = [], [], []
    for r, col, v in zip(w.row, w.col, w.data):
        if r != old and col != old:
            rows.append(remap[r]); cols.append(remap[col]); data.append(v); continue
        partner = col if r == old else r
        k = c.column[partner]
        if k < 0:
            continue                          # a non-columnar partner: no compartment to route to
        comp = m10(k) if types[partner].startswith(MEDULLA_PARTNERS) else lo1(k)
        if r == old:                          # partner -> CT1
            rows.append(comp); cols.append(remap[col]); data.append(v)
        else:                                 # CT1 -> partner
            rows.append(remap[r]); cols.append(comp); data.append(v)
    weights = sparse.csr_matrix((np.array(data, dtype=np.float32), (rows, cols)), shape=(n_new, n_new))
    ids = np.concatenate([c.neuron_ids[keep], np.full(2 * ncol, c.neuron_ids[old])])
    classes = np.concatenate([types[keep], ["CT1(M10)"] * ncol, ["CT1(Lo1)"] * ncol]).astype(str)
    nt = np.concatenate([c.neuron_nt[keep], [c.neuron_nt[old]] * (2 * ncol)])
    column = np.concatenate([c.column[keep], np.arange(ncol), np.arange(ncol)]).astype(np.int32)
    mod = c.modulatory.tocsr()[keep][:, keep]
    mod = sparse.csr_matrix((mod.data, mod.indices, mod.indptr), shape=(n - 1, n - 1))
    mod.resize((n_new, n_new))
    from flybrain.vision import VisualCircuit
    return VisualCircuit(ids, classes, weights, mod, nt, column, c.columns_xy, c.side)


def flyvis_circuit(side="right"):
    """FlyWire's optic lobe with flyvis weights, time constants and biases.

    Returns the circuit (weights replaced, in flyvis units) and a dict with
    per-neuron `tau` (s), `bias` (NaN where flyvis has no type), and the
    boolean `known` mask of neurons whose type flyvis has."""
    c = split_ct1(load_visual_circuit(side, weight_scale=1.0))
    P = flyvis_params.load()
    types = c.neuron_classes.astype(str)

    tau = np.full(c.n_neurons, FLOOR_TAU, dtype=np.float32)
    bias = np.full(c.n_neurons, np.nan, dtype=np.float32)
    for t in np.unique(types):
        for name in _names(t):
            if name in P["time_const"]:
                idx = types == t
                tau[idx] = max(P["time_const"][name], FLOOR_TAU)
                bias[idx] = P["bias"][name]
                break
    known = ~np.isnan(bias)

    # weights: flyvis strength and sign where the pair exists, the init rule elsewhere
    import pandas as pd
    w = c.weights.tocoo()
    df = pd.DataFrame({"pre": types[w.col], "post": types[w.row], "n": np.abs(w.data).astype(np.float32),
                       "sign": np.sign(w.data).astype(np.float32), "tgt": w.row})
    mean_n = df.groupby(["pre", "post"]).n.transform("mean").to_numpy()
    # flyvis's rule, scale / <N>, gives a columnar cell ~0.01 per input type; a
    # wide-field cell with hundreds of partners of one type would get hundreds
    # of times that, so the total per (source type, target cell) is capped
    per_cell = df.groupby(["pre", "tgt"]).n.transform("sum").to_numpy()
    strength = np.minimum(INIT_SCALE / np.maximum(mean_n, 1e-6), 10 * INIT_SCALE / np.maximum(per_cell, 1e-6)).astype(np.float32)
    strength[mean_n <= 0] = 0.0
    sign = df["sign"].to_numpy().copy()
    covered = np.zeros(len(df), dtype=bool)
    pairs = df[["pre", "post"]].drop_duplicates()
    lookup = {}
    for s_, t_ in zip(pairs.pre, pairs.post):
        for sn in _names(s_):
            for tn in _names(t_):
                if f"{sn}>{tn}" in P["syn_strength"]:
                    lookup[(s_, t_)] = (P["syn_strength"][f"{sn}>{tn}"], P["sign"][f"{sn}>{tn}"]); break
            if (s_, t_) in lookup: break
    if lookup:
        keys = list(zip(df.pre, df.post))
        hit = np.array([k in lookup for k in keys])
        vals = np.array([lookup[k] for k, h in zip(keys, hit) if h], dtype=np.float32)
        strength[hit] = vals[:, 0]; sign[hit] = vals[:, 1]; covered = hit
        # flyvis's strengths were fitted with flyvis's synapse totals; FlyWire's
        # totals per pair differ (L3->Mi9 is 5x larger, Mi1->T4a 0.7x).  Each
        # pair's total weight per target cell is matched to flyvis's, and
        # FlyWire decides how that total is spread over the cell's partners.
        fw_total = df.groupby(["pre", "post"]).apply(lambda g: g.groupby("tgt").n.sum().mean(), include_groups=False)
        ratio = np.ones(len(df), dtype=np.float32)
        for (s_, t_), (_, _) in lookup.items():
            fv_total = None
            for sn in _names(s_):
                for tn in _names(t_):
                    if f"{sn}>{tn}" in P["pair_total_per_cell"]:
                        fv_total = P["pair_total_per_cell"][f"{sn}>{tn}"]; break
                if fv_total is not None: break
            if fv_total and fw_total.get((s_, t_), 0) > 0:
                ratio[(df.pre.values == s_) & (df.post.values == t_)] = fv_total / fw_total[(s_, t_)]
        strength[hit] *= ratio[hit]
    weights = sparse.csr_matrix((sign * df.n.to_numpy() * strength, (w.row, w.col)), shape=w.shape)
    weights.eliminate_zeros()
    c.weights = weights
    return c, dict(tau=tau, bias=bias, known=known, pairs_covered=float(covered.mean()))


class FlyvisNetwork:
    """flyvis's dynamics on the GPU, several flies as columns.

        V += (dt / max(tau, dt)) * (-V + bias + W relu(V) + input)
    """
    threshold = 1.0

    def __init__(self, weights, dyn, n_agents=1, dt=0.01, cap=10.0, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        w = weights.tocsr().astype(np.float32)
        self.weights = torch.sparse_csr_tensor(
            torch.from_numpy(w.indptr.astype(np.int64)), torch.from_numpy(w.indices.astype(np.int64)),
            torch.from_numpy(w.data), size=w.shape, device=self.device)
        self.n_neurons, self.n_agents, self.dt, self.cap = w.shape[0], n_agents, dt, cap
        tau = np.maximum(np.asarray(dyn["tau"], dtype=np.float32), dt)
        self.k = torch.as_tensor(dt / tau, device=self.device)[:, None]
        self.leak = float(np.median(dt / tau))                  # for Eye.calibrate's step size
        self.bias = torch.as_tensor(np.nan_to_num(dyn["bias"], nan=0.0), device=self.device)
        self.reset()

    def reset(self):
        self.potential = self.bias[:, None].expand(-1, self.n_agents).clone()
        self.rate = self.potential.clamp(0.0, self.cap)

    def to_device(self, x):
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x, dtype=np.float32))
        x = x.to(self.device)
        return x[:, None].expand(-1, self.n_agents) if x.dim() == 1 else x

    def step(self, external_current=None):
        drive = torch.sparse.mm(self.weights, self.rate) + self.bias[:, None]
        if external_current is not None:
            drive = drive + self.to_device(external_current)
        self.potential = self.potential + self.k * (drive - self.potential)
        self.rate = self.potential.clamp(0.0, self.cap)
        return self.rate

    def run(self, external_current=None, steps=50):
        ext = self.to_device(external_current)
        total = torch.zeros((self.n_neurons, self.n_agents), device=self.device)
        for _ in range(steps):
            total += self.step(ext)
        return total / steps


def make_eye(side="right", dt=0.0125, rounds=60, verbose=False, rebuild=False):
    """The complete eye: circuit, screen mapping, and the network with resting
    potentials - flyvis's for the types it has, fitted to a blank screen for
    the rest (towards the median resting output of the flyvis types).

    Building it takes a minute; the result is cached in data/ and rebuilt
    when flyvis_params.json changes (or with rebuild=True)."""
    from flybrain.vision import Eye, VisualCircuit
    stamp = hashlib.sha1(flyvis_params.OUT.read_bytes()).hexdigest()[:10]
    cache = DATA_DIR / f"flyvis_eye_{side}_{int(round(dt * 1e4))}_{rounds}_{stamp}.npz"
    if cache.exists() and not rebuild:
        b = np.load(cache, allow_pickle=False)
        w = sparse.csr_matrix((b["w_data"], b["w_indices"], b["w_indptr"]), shape=tuple(b["w_shape"]))
        m = sparse.csr_matrix((b["m_data"], b["m_indices"], b["m_indptr"]), shape=tuple(b["w_shape"]))
        c = VisualCircuit(b["ids"], b["classes"], w, m, b["nt"], b["column"], b["columns_xy"], side)
        dyn = dict(tau=b["tau"], bias=b["flyvis_bias"], known=b["known"], pairs_covered=float(b["pairs_covered"]))
        net = FlyvisNetwork(c.weights, dyn, dt=dt)
        net.bias = torch.as_tensor(b["bias"], device=net.device)
        eye = Eye(c, strength=1.0); eye.bias[:] = 0.0; eye.dyn = dyn
        return c, eye, net
    c, dyn = flyvis_circuit(side)
    net = FlyvisNetwork(c.weights, dyn, dt=dt)
    eye = Eye(c, strength=1.0)
    eye.bias[:] = 0.0                                    # resting potentials live in the network
    known = torch.as_tensor(dyn["known"], device=net.device)
    blank = np.full((eye.height, eye.width, 3), 128, dtype=np.uint8)
    cur = torch.as_tensor(eye.currents(blank), device=net.device)[:, None]
    for r in range(rounds):
        net.reset()
        for _ in range(40): net.step(cur)
        a = torch.zeros(net.n_neurons, device=net.device)
        for _ in range(20): a += net.step(cur)[:, 0]
        a /= 20
        net.bias = torch.where(known, net.bias, net.bias + 0.3 * (a[known].median() - a))
        if verbose and r % 20 == 0:
            print(f"  round {r}: unknown types median {float(a[~known].median()):.3f} vs target {float(a[known].median()):.3f}")
    eye.dyn = dyn
    w, m = c.weights.tocsr(), c.modulatory.tocsr()
    np.savez_compressed(cache, ids=c.neuron_ids, classes=c.neuron_classes.astype(str), nt=c.neuron_nt.astype(str),
                        column=c.column, columns_xy=c.columns_xy, w_shape=np.array(w.shape),
                        w_data=w.data, w_indices=w.indices, w_indptr=w.indptr,
                        m_data=m.data, m_indices=m.indices, m_indptr=m.indptr,
                        tau=dyn["tau"], flyvis_bias=dyn["bias"], known=dyn["known"], pairs_covered=dyn["pairs_covered"],
                        bias=net.bias.cpu().numpy())
    return c, eye, net


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    c, dyn = flyvis_circuit()
    print(c.summary())
    print(f"flyvis parameters cover {dyn['known'].sum():,} neurons ({dyn['known'].mean():.0%}) and "
          f"{dyn['pairs_covered']:.0%} of connections; time constants > 20 ms on {(dyn['tau'] > 0.021).sum():,} neurons")
    print("weight magnitudes: median %.4f, 99th pct %.3f" % (np.median(abs(c.weights.data)), np.percentile(abs(c.weights.data), 99)))
