"""Does the measured optic lobe behave like an eye?  Five checks.

The optic lobe is simulated as graded potentials (graded.py) - its neurons
mostly do not spike - with two fitted numbers: weight_scale, how much
current one synapse carries, and a resting current per neuron fitted to a
blank screen (Eye.calibrate).  Everything else is the wiring.

  1. calibration: the weight scale at which brightness reaches the medulla
     without the lobe saturating
  2. brightness: photoreceptors up with light, lamina down (histamine
     inverts the sign), and then the split the fly is known for: Mi1 (ON)
     up with light, Tm1 (OFF) down
  3. object: a dark disc drives the lamina and OFF cells whose receptive
     field it covers, not the others
  4. position: a linear readout recovers where the disc is from the
     population - the property the game needs
  5. direction: T4a/b/c/d are named for the four directions they prefer;
     does a grating moving each way pick out a different one?

    python -m flybrain.verify_vision
"""
import numpy as np
import torch

from flybrain.graded import GradedNetwork
from flybrain.vision import Eye, load_visual_circuit

STEPS_PER_FRAME = 5          # brain steps per screen frame
WEIGHT_SCALE = 0.003         # current per synapse (check 1)
STRENGTH = 0.15 / 0.85       # photoreceptor output equals the brightness it sees
LAYERS = ["R1-6", "L1", "L2", "L3", "Mi1", "Tm1", "Tm3", "T4a", "T5a", "LC11", "LC18", "LC12", "LC17", "LC4", "LPLC2", "LC6"]
OBJECT_TYPES = ["L1", "Mi1", "Tm3", "T4a", "T5a", "LC11", "LC18", "LC12", "LC17", "LC4", "LPLC2", "LC6"]


def grey(v=0.5):
    return np.full((480, 640, 3), int(v * 255), dtype=np.uint8)


def disc(frame, x, y, r=25, colour=(20, 20, 20)):
    ys, xs = np.mgrid[0:480, 0:640]
    frame = frame.copy(); frame[(xs - x) ** 2 + (ys - y) ** 2 <= r * r] = colour
    return frame


def grating(phase, period=80, vertical=True):
    n = 640 if vertical else 480
    stripe = ((((np.arange(n) + phase) // (period // 2)) % 2) * 200 + 30).astype(np.uint8)
    if vertical:
        return np.repeat(np.repeat(stripe[None, :, None], 480, 0), 3, 2)
    return np.repeat(np.repeat(stripe[:, None, None], 640, 1), 3, 2)


def counts_per_neuron(eye, net, frames, warm=20):
    """Mean output per neuron per step, after `warm` frames of the first frame."""
    net.reset()
    for _ in range(warm):
        for _ in range(STEPS_PER_FRAME): net.step(eye.currents(frames[0]))
    counts = torch.zeros(net.n_neurons, device=net.device)
    for f in frames:
        cur = eye.currents(f)
        for _ in range(STEPS_PER_FRAME): counts += net.step(cur)[:, 0]
    return counts.cpu().numpy() / (len(frames) * STEPS_PER_FRAME)


def by_layer(circuit, counts, layers=LAYERS):
    return {name: counts[circuit.indices_of(name)].mean() for name in layers if len(circuit.indices_of(name))}


def table(rows, cols, width=24):
    print(f"{'':{width}s}" + "".join(f"{c:>7s}" for c in cols))
    for name, r in rows:
        print(f"{name:{width}s}" + "".join(f"{r.get(c, 0):7.2f}" for c in cols))


def make_eye(scale=WEIGHT_SCALE):
    c = load_visual_circuit(weight_scale=scale); eye = Eye(c, strength=STRENGTH); net = GradedNetwork(c.weights)
    eye.calibrate(net, target=0.1, rounds=60, gain=0.3)
    return c, eye, net


def kernel_ridge(Xtr, Ytr, Xte, lam=100.0):
    """Ridge regression in its dual form: fine with 25,000 features and 200 samples."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    alpha = np.linalg.solve(Ztr @ Ztr.T + lam * np.eye(len(Ztr)), Ytr - Ytr.mean(0))
    return (Zte @ Ztr.T) @ alpha + Ytr.mean(0)


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    base = load_visual_circuit()
    print(base.summary(), "\n")

    print("1. weight_scale: output on a static mid-grey screen after calibration (mean per step, by cell type)")
    rows = []
    for scale in (0.001, 0.003, 0.01):
        c, eye, net = make_eye(scale)
        r = counts_per_neuron(eye, net, [grey()] * 12)
        rows.append((f"weight_scale {scale}", by_layer(c, r)))
    table(rows, LAYERS)

    c, eye, net = make_eye()
    print(f"\n2. brightness (weight_scale {WEIGHT_SCALE}): lamina against the photoreceptors, then ON against OFF")
    rows = [(f"screen {v:.2f}", by_layer(c, counts_per_neuron(eye, net, [grey(v)] * 12))) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
    table(rows, ["R1-6", "L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9"])

    print("\n3. a still dark disc (r=30) at the centre: cells whose receptive field is inside it / outside (beyond 120 px)")
    r_disc = counts_per_neuron(eye, net, [disc(grey(), 300, 240, r=30)] * 12)
    r_blank = counts_per_neuron(eye, net, [grey()] * 12)
    d = np.hypot(eye.rf_screen[:, 0] - 300, eye.rf_screen[:, 1] - 240)
    print(f"{'':10s}" + "".join(f"{t:>13s}" for t in OBJECT_TYPES))
    for name, r in (("disc", r_disc), ("blank", r_blank)):
        line = f"{name:10s}"
        for t in OBJECT_TYPES:
            idx = c.indices_of(t); dd = d[idx]; i, o = r[idx][dd < 30], r[idx][dd > 120]
            line += f"   {i.mean():4.2f} / {o.mean():4.2f}" if len(i) >= 2 and len(o) else f"   {'n<2':>11s}"
        print(line)

    print("\n4. position: a dark disc (r=25) at 240 random places; ridge regression from the population to (x, y)")
    rng = np.random.default_rng(0); X, Y = [], []
    for _ in range(240):
        x, y = rng.uniform(40, 430), rng.uniform(40, 440)
        X.append(counts_per_neuron(eye, net, [disc(grey(), x, y, r=25)] * 8, warm=8)); Y.append((x, y))
    X, Y = np.array(X), np.array(Y); tr, te = slice(0, 180), slice(180, 240)
    for name, idx in [("L1 + L2", np.concatenate([c.indices_of("L1"), c.indices_of("L2")])),
                      ("OFF path: L1, L2, Tm1, Tm2, T2, T3", np.concatenate([c.indices_of(t) for t in ("L1", "L2", "Tm1", "Tm2", "T2", "T3")])),
                      ("every non-photoreceptor", np.flatnonzero(~np.isin(c.neuron_classes, ["R1-6", "R7", "R8"])))]:
        Xs = X[:, idx]; keep = Xs[tr].std(0) > 1e-6; Xs = Xs[:, keep]
        err = np.hypot(*(kernel_ridge(Xs[tr], Y[tr], Xs[te]) - Y[te]).T)
        print(f"   {name:36s} median error {np.median(err):5.1f} px   ({keep.sum():,} cells that vary; chance ~180 px)")

    print("\n5. direction: a grating moving right / left / down / up - T4 subtypes should split by direction")
    rows = []
    for name, frames in (("right", [grating(-i * 8) for i in range(40)]), ("left", [grating(i * 8) for i in range(40)]),
                         ("down", [grating(-i * 8, vertical=False) for i in range(40)]), ("up", [grating(i * 8, vertical=False) for i in range(40)])):
        rows.append((f"grating {name}", by_layer(c, counts_per_neuron(eye, net, frames, warm=10), ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"])))
    table(rows, ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"])
