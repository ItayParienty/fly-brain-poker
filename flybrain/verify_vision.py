"""Does the measured optic lobe behave like an eye?  Six checks.

The eye is FlyWire's right optic lobe with flyvis's cell-type parameters
and graded dynamics (flyvis_eye.py).  Nothing here is fitted to these
tests; they ask what the wiring does with those numbers.

  1. resting state: output of each cell type on a blank grey screen
  2. brightness: photoreceptors up with light, lamina down (histamine
     inverts the sign), then the split the fly is known for: Mi1 and Tm3
     (ON) up with light, Tm1 (OFF) down
  3. object: a dark disc changes the cells whose receptive field it covers
     and not the others - lamina and OFF cells up, ON cells down
  4. position: a linear readout recovers where the disc is from the
     population - the property the game needs
  5. direction: T4a-d and T5a-d are named for the four directions they
     prefer.  An edge moves across the eye's own hexagonal lattice at the
     same speed in twelve directions; does each subtype prefer one?
  6. looming: LPLC2 and LC4 respond to expansion in the fly; does an
     expanding disc drive them more than a shrinking or a static one?

    python -m flybrain.verify_vision
"""
import numpy as np
import torch

from flybrain.flyvis_eye import make_eye
from flybrain.vision import SPECTRAL

STEPS_PER_FRAME = 2          # brain steps of 12.5 ms per 25 ms screen frame
LAYERS = ["R1-6", "L1", "L2", "L3", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm9", "T4a", "T5a", "LC11", "LC18", "LC4", "LPLC2"]
OBJECT_TYPES = ["L1", "L2", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9", "T4a", "T5a", "LC11", "LC18", "LC4", "LPLC2"]
MOTION_TYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def grey(v=0.5):
    return np.full((480, 640, 3), int(v * 255), dtype=np.uint8)


def disc(frame, x, y, r=25, colour=(20, 20, 20)):
    ys, xs = np.mgrid[0:480, 0:640]
    frame = frame.copy(); frame[(xs - x) ** 2 + (ys - y) ** 2 <= r * r] = colour
    return frame


def outputs(eye, net, frames, warm=8):
    """Mean output per neuron over `frames`, after `warm` frames of the first one."""
    net.reset()
    for _ in range(warm * STEPS_PER_FRAME):
        net.step(eye.currents(frames[0]))
    total = torch.zeros(net.n_neurons, device=net.device)
    for f in frames:
        cur = eye.currents(f)
        for _ in range(STEPS_PER_FRAME):
            total += net.step(cur)[:, 0]
    return total.cpu().numpy() / (len(frames) * STEPS_PER_FRAME)


def by_layer(circuit, values, layers=LAYERS):
    return {name: values[circuit.indices_of(name)].mean() for name in layers if len(circuit.indices_of(name))}


def table(rows, cols, width=18):
    print(f"{'':{width}s}" + "".join(f"{c:>7s}" for c in cols))
    for name, r in rows:
        print(f"{name:{width}s}" + "".join(f"{r.get(c, 0):7.2f}" for c in cols))


def kernel_ridge(Xtr, Ytr, Xte, lam=100.0):
    """Ridge regression in its dual form: fine with 36,000 features and 200 samples."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    alpha = np.linalg.solve(Ztr @ Ztr.T + lam * np.eye(len(Ztr)), Ytr - Ytr.mean(0))
    return (Zte @ Ztr.T) @ alpha + Ytr.mean(0)


def lattice_currents(circuit, eye, net, lum):
    """Photoreceptor drive from a luminance (0..1) per column, bypassing the screen."""
    cur = np.zeros(circuit.n_neurons, dtype=np.float32)
    for kind, (idx, cols) in eye.receptors.items():
        cur[idx] = lum[cols] * SPECTRAL[kind].sum()
    return torch.as_tensor(cur, device=net.device)[:, None]


def edge_on_lattice(circuit, eye, net, angle_deg, cols_per_s, on=True):
    """An edge sweeping across the hexagonal lattice at a fixed speed in columns/s."""
    xy = circuit.columns_xy - circuit.columns_xy.mean(0)
    ang = np.radians(angle_deg); proj = xy[:, 0] * np.cos(ang) + xy[:, 1] * np.sin(ang)
    per_frame = cols_per_s / 40.0
    n = int((proj.max() - proj.min() + 4) / per_frame) + 10
    return [lattice_currents(circuit, eye, net, np.where(proj < proj.min() - 2 + per_frame * i, 0.9 if on else 0.1, 0.5).astype(np.float32))
            for i in range(n)]


def peak_by_type(net, idxs, frames, blank_cur, warm=10):
    net.reset()
    for _ in range(warm * STEPS_PER_FRAME):
        net.step(blank_cur)
    peaks = {t: -1e9 for t in idxs}
    for cur in frames:
        for _ in range(STEPS_PER_FRAME):
            r = net.step(cur)[:, 0]
            for t, idx in idxs.items():
                peaks[t] = max(peaks[t], float(r[idx].mean()))
    return peaks


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    c, eye, net = make_eye()
    print(c.summary())
    print(f"flyvis parameters on {eye.dyn['known'].mean():.0%} of neurons; the rest fitted to a blank screen\n")

    print("1. resting output on a blank grey screen, by cell type")
    blank = outputs(eye, net, [grey()] * 8)
    table([("blank", by_layer(c, blank))], LAYERS)

    print("\n2. brightness: lamina against the photoreceptors, then ON against OFF")
    rows = [(f"screen {v:.2f}", by_layer(c, outputs(eye, net, [grey(v)] * 8))) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
    table(rows, ["R1-6", "L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9"])

    print("\n3. a still dark disc (r=30) at the centre: cells whose receptive field is inside it / outside (beyond 120 px)")
    r_disc = outputs(eye, net, [disc(grey(), 300, 240, r=30)] * 8)
    d = np.hypot(eye.rf_screen[:, 0] - 300, eye.rf_screen[:, 1] - 240)
    print(f"{'':10s}" + "".join(f"{t:>13s}" for t in OBJECT_TYPES))
    for name, r in (("disc", r_disc), ("blank", blank)):
        line = f"{name:10s}"
        for t in OBJECT_TYPES:
            idx = c.indices_of(t); dd = d[idx]; i, o = r[idx][dd < 30], r[idx][dd > 120]
            line += f"   {i.mean():4.2f} / {o.mean():4.2f}" if len(i) >= 2 and len(o) else f"   {'n<2':>11s}"
        print(line)

    print("\n4. position: a dark disc (r=25) at 240 random places; ridge regression from the population to (x, y)")
    rng = np.random.default_rng(0); X, Y = [], []
    for _ in range(240):
        x, y = rng.uniform(40, 430), rng.uniform(40, 440)
        X.append(outputs(eye, net, [disc(grey(), x, y, r=25)] * 6, warm=6)); Y.append((x, y))
    X, Y = np.array(X), np.array(Y); tr, te = slice(0, 180), slice(180, 240)
    for name, idx in [("L1 + L2", np.concatenate([c.indices_of("L1"), c.indices_of("L2")])),
                      ("every non-photoreceptor", np.flatnonzero(~np.isin(c.neuron_classes, ["R1-6", "R7", "R8"])))]:
        Xs = X[:, idx]; keep = Xs[tr].std(0) > 1e-6; Xs = Xs[:, keep]
        err = np.hypot(*(kernel_ridge(Xs[tr], Y[tr], Xs[te]) - Y[te]).T)
        print(f"   {name:26s} median error {np.median(err):5.1f} px   ({keep.sum():,} cells that vary; chance ~180 px)")

    print("\n5. direction: an edge crossing the lattice at 15 columns/s in 12 directions; peak response above rest per direction")
    print("   (T4 gets a bright edge, T5 a dark one; a subtype with a direction should show one peak well above the others)")
    idxs = {t: torch.as_tensor(c.indices_of(t), device=net.device) for t in MOTION_TYPES}
    blank_cur = lattice_currents(c, eye, net, np.full(c.n_columns, 0.5, dtype=np.float32))
    rest = peak_by_type(net, idxs, [blank_cur] * 4, blank_cur)
    angles = list(range(0, 360, 30))
    peaks = {t: [] for t in MOTION_TYPES}
    for ang in angles:
        for on in (True, False):
            pk = peak_by_type(net, idxs, edge_on_lattice(c, eye, net, ang, 15, on), blank_cur)
            for t in MOTION_TYPES:
                if t.startswith("T4") == on:
                    peaks[t].append(pk[t] - rest[t])
    print(f"{'':6s}" + "".join(f"{a:>7d}" for a in angles) + "   preferred   DSI")
    for t in MOTION_TYPES:
        pk = np.array(peaks[t])
        vec = (pk[:, None] * np.stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))], 1)).sum(0)
        pref = (np.degrees(np.arctan2(vec[1], vec[0])) + 360) % 360
        dsi = (pk.max() - pk.min()) / (abs(pk.max()) + abs(pk.min()) + 1e-9)
        print(f"{t:6s}" + "".join(f"{v:+7.3f}" for v in pk) + f"   {pref:6.0f} deg  {dsi:.2f}")

    print("\n6. looming: a dark disc expanding from r=6 to r=90 over 30 frames, the same shrinking, and static discs")
    d_rf = np.hypot(eye.rf_screen[:, 0] - 300, eye.rf_screen[:, 1] - 240)
    rs = np.linspace(6, 90, 30)
    conds = {"expanding": [disc(grey(), 300, 240, r=int(r)) for r in rs], "shrinking": [disc(grey(), 300, 240, r=int(r)) for r in rs[::-1]],
             "static big": [disc(grey(), 300, 240, r=90)] * 30, "static small": [disc(grey(), 300, 240, r=6)] * 30}
    loom_types = ["LPLC2", "LC4", "LC6", "LPLC1", "LC11", "LC18", "T4a", "T5a"]
    print(f"{'':14s}" + "".join(f"{t:>8s}" for t in loom_types) + "    (cells with a receptive field within 60 px of the disc)")
    for name, frames in conds.items():
        r = outputs(eye, net, frames, warm=6)
        line = f"{name:14s}"
        for t in loom_types:
            idx = c.indices_of(t); near = r[idx][d_rf[idx] < 60]
            line += f"{near.mean():8.3f}" if len(near) else f"{'-':>8s}"
        print(line)
