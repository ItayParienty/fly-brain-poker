"""The fly's visual system, as the connectome measured it, wired to a screen.

One optic lobe of FlyWire v783 (the right one, by default): every neuron
FlyWire's optic-lobe atlas assigns to that side - photoreceptors, lamina,
medulla, lobula, lobula plate, and the projection neurons that carry the
result into the central brain.  47,291 neurons, 845,712 connections.

What makes it an eye rather than a pile of neurons is `column_assignment`:
the atlas places each columnar neuron in one of the 796 columns of the
hexagonal lattice behind the ommatidia, with hex coordinates.  So we know,
for each L1 or T4 or Tm3, which patch of the visual field it looks at.
That map is measured; nothing about it is ours.

Light enters where it does in the fly: the photoreceptors.  R1-6 (broad
spectrum, no red) get the brightness of their column's patch of screen;
R7 is ultraviolet, which a monitor does not emit, so it stays dark; R8
(blue/green) gets the blue-green part.  Photoreceptors release histamine,
which inhibits the lamina cells - FlyWire's transmitter predictor has no
histamine class and calls them cholinergic, so their sign is overridden
here to inhibitory.  That is the one place this file corrects the data.

    circuit = load_visual_circuit()          # cached after the first build
    eye = Eye(circuit)                       # screen -> photoreceptor currents
    current = eye.currents(frame_rgb_uint8)  # (n_neurons,) numpy array
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from flybrain.connectome import Circuit, DATA_DIR, MODULATORS, SYNAPSE_SIGN

PHOTORECEPTORS = ("R1-6", "R7", "R8")
LAMINA_TARGETS = ("L1", "L2", "L3")             # how an R1-6 finds its column: it synapses onto these

# Sensitivity of each photoreceptor class to a monitor's R, G, B primaries,
# from the fly's known spectral sensitivities: R1-6 peak in the blue-green
# with no red sensitivity to speak of; R8 is blue (pale) or green (yellow);
# R7 is UV only.  Rows sum to 1 for R1-6 and R8.
SPECTRAL = {
    "R1-6": np.array([0.05, 0.60, 0.35], dtype=np.float32),
    "R8":   np.array([0.00, 0.50, 0.50], dtype=np.float32),
    "R7":   np.array([0.00, 0.00, 0.00], dtype=np.float32),
}


class VisualCircuit(Circuit):
    """A Circuit whose classes are cell types, plus the column each neuron looks through."""

    def __init__(self, neuron_ids, neuron_classes, weights, modulatory, neuron_nt,
                 column, columns_xy, side):
        super().__init__(neuron_ids, neuron_classes, weights, modulatory, neuron_nt)
        self.column = column            # column index per neuron, -1 if not columnar
        self.columns_xy = columns_xy    # (n_columns, 2) cartesian hex-lattice positions
        self.side = side

    @property
    def n_columns(self):
        return len(self.columns_xy)

    def receptive_fields(self, hops=3):
        """Lattice position each neuron looks at: its own column, or the
        synapse-weighted mean of its columnar inputs' columns (repeated for
        neurons whose inputs are themselves non-columnar). NaN if unknown."""
        if getattr(self, "_rf", None) is not None:
            return self._rf
        rf = np.full((self.n_neurons, 2), np.nan, dtype=np.float32)
        has = self.column >= 0
        rf[has] = self.columns_xy[self.column[has]]
        w = abs(self.weights).tocsr()
        for _ in range(hops):
            known = ~np.isnan(rf[:, 0])
            wk = w[:, known]
            total = np.asarray(wk.sum(axis=1)).ravel()
            sx = wk @ rf[known, 0]; sy = wk @ rf[known, 1]
            fill = (~known) & (total > 0)
            rf[fill, 0] = sx[fill] / total[fill]; rf[fill, 1] = sy[fill] / total[fill]
        self._rf = rf
        return rf

    def summary(self):
        lines = [f"{self.side} optic lobe: {self.n_neurons:,} neurons, {self.weights.nnz:,} fast connections, "
                 f"{self.n_columns} columns"]
        for name in ("R1-6", "R7", "R8", "L1", "L2", "L3", "Mi1", "Tm3", "T4a", "T5a", "LC4", "LPLC2", "LC6", "LC11"):
            idx = self.indices_of(name)
            if len(idx):
                lines.append(f"  {name:<6} {len(idx):>5,}  ({(self.column[idx] >= 0).sum():,} with a column)")
        excitatory = (self.weights.data > 0).sum()
        lines.append(f"  excitatory {excitatory:,} / inhibitory {self.weights.nnz - excitatory:,}")
        return "\n".join(lines)


def hex_to_xy(p, q):
    """Axial hex coordinates -> cartesian, unit spacing between neighbours."""
    return np.stack([p + q / 2.0, q * np.sqrt(3) / 2.0], axis=1)


def build_visual_circuit(side="right", data_dir=DATA_DIR):
    types = pd.read_csv(data_dir / "visual_neuron_types.csv.gz")
    types = types[types.side == side]
    neuron_ids = types.root_id.to_numpy()
    neuron_classes = types.type.to_numpy().astype(str)
    index_of = {rid: i for i, rid in enumerate(neuron_ids)}
    n = len(neuron_ids)

    transmitters = pd.read_csv(data_dir / "neurons.csv.gz")[["root_id", "nt_type"]]
    nt_of = dict(zip(transmitters.root_id, transmitters.nt_type))
    neuron_nt = np.array([str(nt_of.get(rid, "UNK")) for rid in neuron_ids])
    is_photoreceptor = np.isin(neuron_classes, PHOTORECEPTORS)
    neuron_nt[is_photoreceptor] = "HIST"

    # ---- columns: the atlas gives one to every columnar neuron except R1-6
    assign = pd.read_csv(data_dir / "column_assignment.csv.gz")
    assign = assign[assign.hemisphere == side]
    cols = assign.drop_duplicates("column_id").sort_values("column_id")
    col_index = {cid: i for i, cid in enumerate(cols.column_id)}
    columns_xy = hex_to_xy(cols.p.to_numpy(float), cols.q.to_numpy(float)).astype(np.float32)
    column = np.full(n, -1, dtype=np.int32)
    for rid, cid in zip(assign.root_id, assign.column_id):
        i = index_of.get(rid)
        if i is not None:
            column[i] = col_index[cid]

    connections = pd.read_csv(data_dir / "connections.csv.gz")
    inside = connections.pre_root_id.isin(index_of) & connections.post_root_id.isin(index_of)
    connections = connections[inside]
    pre = connections.pre_root_id.map(index_of).to_numpy()
    post = connections.post_root_id.map(index_of).to_numpy()
    counts = connections.syn_count.to_numpy(dtype=np.float32)
    nt = connections.nt_type.to_numpy().astype(str)

    # R1-6 have no column in the atlas; each one synapses onto the L1/L2/L3
    # of its own cartridge, so it inherits the column of its strongest target
    r16 = np.flatnonzero(neuron_classes == "R1-6")
    lamina = np.isin(neuron_classes, LAMINA_TARGETS)
    best = {}
    for a, b, c in zip(pre, post, counts):
        if neuron_classes[a] == "R1-6" and lamina[b] and column[b] >= 0:
            if c > best.get(a, (0, -1))[0]:
                best[a] = (c, column[b])
    for a, (_, col) in best.items():
        column[a] = col

    # ---- signed weights; photoreceptor output is histaminergic = inhibitory
    nt[is_photoreceptor[pre]] = "HIST"
    sign_table = dict(SYNAPSE_SIGN, HIST=-1.0)
    is_modulator = np.isin(nt, list(MODULATORS))
    signs = np.array([sign_table.get(t, 0.0) for t in nt], dtype=np.float32)
    shape = (n, n)
    fast = sparse.csr_matrix((counts[~is_modulator] * signs[~is_modulator],
                              (post[~is_modulator], pre[~is_modulator])), shape=shape)
    modulatory = sparse.csr_matrix((counts[is_modulator], (post[is_modulator], pre[is_modulator])), shape=shape)
    return VisualCircuit(neuron_ids, neuron_classes, fast, modulatory, neuron_nt, column, columns_xy, side)


def load_visual_circuit(side="right", weight_scale=1.0, rebuild=False, data_dir=DATA_DIR):
    cache = Path(data_dir) / f"visual_{side}_cache.npz"
    if cache.exists() and not rebuild:
        blob = np.load(cache, allow_pickle=True)
        shape = tuple(blob["shape"])
        fast = sparse.csr_matrix((blob["fast_data"] * weight_scale, blob["fast_indices"], blob["fast_indptr"]), shape=shape)
        modulatory = sparse.csr_matrix((blob["mod_data"], blob["mod_indices"], blob["mod_indptr"]), shape=shape)
        return VisualCircuit(blob["neuron_ids"], blob["neuron_classes"], fast, modulatory, blob["neuron_nt"],
                             blob["column"], blob["columns_xy"], str(blob["side"]))
    c = build_visual_circuit(side, data_dir)
    np.savez_compressed(cache, neuron_ids=c.neuron_ids, neuron_classes=c.neuron_classes, neuron_nt=c.neuron_nt,
                        shape=np.array(c.weights.shape), fast_data=c.weights.data, fast_indices=c.weights.indices,
                        fast_indptr=c.weights.indptr, mod_data=c.modulatory.data, mod_indices=c.modulatory.indices,
                        mod_indptr=c.modulatory.indptr, column=c.column, columns_xy=c.columns_xy, side=c.side)
    c.weights.data *= weight_scale
    return c


class Eye:
    """Turns a screen image into currents for the photoreceptors of each column.

    The screen is fitted to the eye: the lattice of columns is rotated so its
    long axis runs along the screen's long axis, then scaled to cover it, and
    each column averages the pixels of its own patch.  Adjacent columns see
    adjacent patches, which is all the downstream motion and looming
    circuits need from the mapping.
    """

    def __init__(self, circuit, width=640, height=480, strength=1.0):
        self.circuit, self.width, self.height, self.strength = circuit, width, height, strength
        xy = circuit.columns_xy - circuit.columns_xy.mean(0)
        # principal axis of the lattice -> the screen's x axis
        u, s, vt = np.linalg.svd(xy, full_matrices=False)
        xy = xy @ vt.T
        lo, hi = xy.min(0), xy.max(0)
        self.spacing = min(width / (hi[0] - lo[0] + 1), height / (hi[1] - lo[1] + 1))
        # scale to fill the screen; a patch is one lattice spacing across
        sx, sy = width / (hi[0] - lo[0] + 1), height / (hi[1] - lo[1] + 1)
        px = (xy[:, 0] - lo[0] + 0.5) * sx
        py = (xy[:, 1] - lo[1] + 0.5) * sy
        self.patch = (sx, sy)
        self.centres = np.stack([px, py], axis=1)                    # screen position per column
        # pixel -> column lookup by nearest centre, done once
        ys, xs = np.mgrid[0:height, 0:width]
        grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
        from scipy.spatial import cKDTree
        _, owner = cKDTree(self.centres).query(grid)
        self.owner = owner.reshape(height, width)
        self.counts = np.bincount(self.owner.ravel(), minlength=circuit.n_columns).astype(np.float32)
        # photoreceptors grouped by class, with their columns
        self.receptors = {}
        for kind in PHOTORECEPTORS:
            idx = circuit.indices_of(kind)
            idx = idx[circuit.column[idx] >= 0]
            self.receptors[kind] = (idx, circuit.column[idx])
        self.bias = self.dark_current()
        # every neuron's receptive field, on the screen
        self._fit = (vt, lo, sx, sy)
        rf = circuit.receptive_fields() - circuit.columns_xy.mean(0)
        rf = rf @ vt.T
        self.rf_screen = np.stack([(rf[:, 0] - lo[0] + 0.5) * sx, (rf[:, 1] - lo[1] + 0.5) * sy], axis=1)

    def dark_current(self, receptor_rate=0.25):
        """Tonic drive for the photoreceptors' targets.

        Photoreceptors inhibit the lamina, so in the fly a lamina cell is
        depolarised in the dark and light silences it.  A leaky
        integrate-and-fire cell with no excitatory input never fires, so the
        inhibition would be invisible.  Each direct target therefore gets a
        constant current equal to the inhibition it would receive under a
        fully lit screen (its photoreceptor synapses times the rate a lit
        photoreceptor fires at), which puts it exactly where the fly's is:
        active in the dark, silenced in proportion to the light.
        """
        w = self.circuit.weights.tocsc()
        idx = np.concatenate([i for i, _ in self.receptors.values()])
        from_receptors = w[:, idx]
        inhibition = -np.asarray(from_receptors.minimum(0).sum(axis=1)).ravel()
        return (inhibition * receptor_rate).astype(np.float32)

    def column_colours(self, frame):
        """Mean RGB (0-1) of the screen patch each column looks at. frame: (H, W, 3) uint8."""
        flat = frame.reshape(-1, 3).astype(np.float32) / 255.0
        sums = np.zeros((self.circuit.n_columns, 3), dtype=np.float32)
        for c in range(3):
            sums[:, c] = np.bincount(self.owner.ravel(), weights=flat[:, c], minlength=self.circuit.n_columns)
        return sums / np.maximum(self.counts, 1)[:, None]

    def currents(self, frame):
        """External current per neuron (zero except photoreceptors)."""
        return self.currents_from(self.column_colours(frame))

    def currents_from(self, colours):
        """The same, from the mean colour each column sees (n_columns, 3), 0..1."""
        current = self.bias.copy()
        for kind, (idx, cols) in self.receptors.items():
            current[idx] = (colours[cols] @ SPECTRAL[kind]) * self.strength
        return current

    def calibrate(self, net, target=0.02, rounds=40, steps=60, gain=0.6, blank=0.5, verbose=False):
        """Fit a resting current per neuron so that, on a blank grey screen,
        every non-photoreceptor neuron fires at `target` of steps.

        The optic lobe works by graded potentials around a resting level:
        Mi1 sees light because L1 stops inhibiting it, T4 sees motion because
        inhibition arrives a moment late.  A leaky integrate-and-fire neuron
        with no resting drive cannot be released from anything, so without
        this every disinhibition in the wiring is invisible.  The published
        connectome models of this system fit a resting potential per cell
        type; here it is one number per neuron, fitted to a blank screen and
        nothing else.  It is the second and last free parameter after
        weight_scale.
        """
        import torch
        frame = np.full((self.height, self.width, 3), int(blank * 255), dtype=np.uint8)
        light = torch.as_tensor(self.currents(frame) - self.bias, device=net.device)   # photoreceptor drive only
        bias = torch.as_tensor(self.bias.copy(), device=net.device)
        adjustable = torch.ones(self.circuit.n_neurons, dtype=torch.bool, device=net.device)
        for idx, _ in self.receptors.values():
            adjustable[torch.as_tensor(idx, device=net.device)] = False
        for r in range(rounds):
            net.reset()
            cur = (light + bias)[:, None]
            for _ in range(steps // 2): net.step(cur)                 # settle
            counts = torch.zeros(self.circuit.n_neurons, device=net.device)
            for _ in range(steps): counts += net.step(cur)[:, 0]
            rate = counts / steps
            # move the resting current towards the target rate; a neuron
            # needs about threshold*leak of current per step to fire at all
            bias = torch.where(adjustable, bias + gain * net.threshold * net.leak * (target - rate) / max(target, 1e-3), bias)
            if verbose and (r % 10 == 0 or r == rounds - 1):
                a = rate[adjustable]
                print(f"  round {r:2d}: mean rate {a.mean():.3f}, silent {(a == 0).float().mean():.2f}, "
                      f"above 5x target {(a > 5 * target).float().mean():.3f}")
        self.bias = bias.cpu().numpy().astype(np.float32)
        return self.bias

    def image(self, frame):
        """What the fly sees: the frame as its columns sample it (for display)."""
        colours = self.column_colours(frame)
        return (colours[self.owner] * 255).astype(np.uint8)


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    import time
    t0 = time.perf_counter()
    c = load_visual_circuit()
    print(c.summary())
    print(f"loaded in {time.perf_counter() - t0:.1f}s")
    eye = Eye(c)
    print(f"eye: {c.n_columns} columns, patch {eye.patch[0]:.1f} x {eye.patch[1]:.1f} px")
    # where do the photoreceptors sit on the screen?  a quick picture
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (640, 480), (20, 20, 20)); d = ImageDraw.Draw(im)
    for (x, y) in eye.centres:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(90, 200, 120))
    im.save(DATA_DIR / "eye_columns.png"); print("wrote data/eye_columns.png")
