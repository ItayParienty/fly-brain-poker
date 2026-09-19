"""Cell-type parameters of the optic lobe from the flyvis ensemble.

Lappalainen et al. (2024, Nature, "Connectome-constrained networks predict
neural activity across the fly visual system") built a model of the optic
lobe from a hemibrain-derived connectome and fitted, per cell type, the
membrane time constant and resting potential, and per pair of cell types
the synaptic strength - by training on optic flow.  Their 50 trained models
are published with the `flyvis` package.

The connectome does not contain time constants; without them the wiring
cannot compute motion direction (verify_vision.py, before this file).  So
this takes the flyvis values and applies them to FlyWire's neurons by cell
type.  The wiring stays FlyWire's; only these scalars are borrowed, and
they were never fitted to anything here.

Which model: the ensemble's best by validation loss (model 000; the
directories are sorted by it).  The 50 models disagree on which cell types
are slow - that is a finding of the paper - so a median across them would
mix incompatible solutions; one coherent set is used instead, and the
per-type median is kept alongside for reference.

    python -m flybrain.flyvis_params      # writes flybrain/flyvis_params.json
"""
import glob
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("flyvis_params.json")


def extract():
    import flyvis
    import torch
    root = Path(flyvis.__file__).parent
    spec = json.load(open(root / "connectome" / "fib25-fib19_v2.2.json"))
    types = [n["name"] for n in spec["nodes"]]
    pairs = []
    for e in spec["edges"]:
        k = (e["src"], e["tar"])
        # the lattice never materialises this self-connection, so it has no parameter
        if k not in pairs and k != ("Lawf1", "Lawf1"):
            pairs.append(k)
    sign_spec = {(e["src"], e["tar"]): e["alpha"] for e in spec["edges"]}
    tcs, biases, strengths, signs = [], [], [], []
    models = sorted(glob.glob(str(root / "data" / "results" / "flow" / "0000" / "0*")))
    for m in models:
        ck = sorted(glob.glob(m + "/chkpts/*"))
        if not ck:
            continue
        sd = torch.load(ck[-1], map_location="cpu", weights_only=False)["network"]
        tcs.append(sd["nodes_time_const"].numpy()); biases.append(sd["nodes_bias"].numpy())
        strengths.append(sd["edges_syn_strength"].numpy()); signs.append(sd["edges_sign"].numpy())
    tcs, biases, strengths, signs = map(np.array, (tcs, biases, strengths, signs))
    assert tcs.shape[1] == len(types) and strengths.shape[1] == len(pairs), (tcs.shape, strengths.shape, len(types), len(pairs))
    b = 0                                        # model 000: lowest validation loss of the ensemble
    # the lattice model's own synapse totals per pair: on average, how many
    # synapses of a source type one target cell receives.  FlyWire's counts
    # differ pair by pair, and flyvis's strengths only make sense with their
    # own totals, so flyvis_eye rescales each pair to match.
    totals = lattice_pair_totals(root)
    params = {
        "source": "flyvis 1.2.0, ensemble flow/0000, model 000 (best of %d by validation loss)" % len(tcs),
        "pair_total_per_cell": totals,
        "dt": 0.02,
        "time_const": {t: float(tcs[b, i]) for i, t in enumerate(types)},
        "bias": {t: float(biases[b, i]) for i, t in enumerate(types)},
        "syn_strength": {f"{s}>{t}": float(strengths[b, i]) for i, (s, t) in enumerate(pairs)},
        "sign": {f"{s}>{t}": float(signs[b, i]) for i, (s, t) in enumerate(pairs)},
        "time_const_median": {t: float(np.median(tcs[:, i])) for i, t in enumerate(types)},
    }
    OUT.write_text(json.dumps(params, indent=1), encoding="utf-8")
    return params, len(tcs)


def lattice_pair_totals(root):
    """Mean over target cells of the summed synapse count from each source type,
    read from flyvis's materialised lattice connectome (built on first use)."""
    import h5py
    import pandas as pd
    # datamate's h5 writer trips over Windows file locking; close before unlinking
    import datamate.io as dio, datamate.directory as ddir
    def _write_h5(path, val):
        val = np.asarray(val); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): path.unlink()
        with h5py.File(path, libver="latest", mode="w") as f: f["data"] = val
    dio._write_h5 = _write_h5; ddir._write_h5 = _write_h5
    import flyvis
    ct = flyvis.ConnectomeFromAvgFilters(file="fib25-fib19_v2.2.json", extent=15, n_syn_fill=1)
    dec = lambda v: np.array([x.decode() if isinstance(x, bytes) else x for x in v])
    df = pd.DataFrame({"pre": dec(ct.edges.source_type[:]), "post": dec(ct.edges.target_type[:]),
                       "n": np.asarray(ct.edges.n_syn[:], dtype=float), "tgt": np.asarray(ct.edges.target_index[:])})
    tot = df.groupby(["pre", "post", "tgt"]).n.sum().groupby(["pre", "post"]).mean()
    return {f"{a}>{b}": float(v) for (a, b), v in tot.items()}


def load():
    return json.loads(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    params, n = extract()
    tc = params["time_const"]
    slow = sorted(tc.items(), key=lambda kv: -kv[1])[:12]
    print(f"{n} models; {len(tc)} cell types, {len(params['syn_strength'])} type pairs -> {OUT.name}")
    print("slowest types (s):", ", ".join(f"{t} {v:.3f}" for t, v in slow))
    print("time constants at the floor (0.02 s):", sum(v < 0.021 for v in tc.values()), "of", len(tc))
