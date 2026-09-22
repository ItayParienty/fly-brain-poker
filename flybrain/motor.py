"""The fly's hands: a pointer, a press and a wing beat, read from the eye.

The eye (flyvis_eye.make_eye) turns a screen into the activity of 48,882
optic-lobe neurons.  This module turns that activity into what a mouse
needs and nothing more: where the pointer is, and whether it presses.
Every read-out is a sum over cell types with one weight per type, of each
neuron's output relative to its output at rest (on a blank grey screen).
No weight belongs to a place on the screen: where things are comes only
from the eye's own map of the visual field, so the fly can only point at
something its optic lobe responds to.

  gaze    a salience map over the 796 columns: each type's activity in each
          column (normalised), weighted, summed, and averaged with the six
          neighbouring columns.  The pointer jumps to the peak - a saccade,
          one step of 12.5 ms, however far across the screen.
  press   the proboscis: the same kind of sum over the fixated column and its
          neighbours, i.e. over what the fly is looking at.  Above zero, it
          presses (and cannot press again for PRESS_REFRACTORY steps).
  wings   a sum over the whole eye.  Above zero, the wings beat - which
          starts the next round.

Two rules shape the gaze.  It stays where it is unless another column
beats it by `hold` (flies fixate, then saccade).  And a spot the fly has
just pressed loses `habituation` of its salience, recovering over 1.5 s,
so the gaze moves on instead of pressing the same button forever.

    motor = Motor(circuit, eye, n_agents, rest=resting_output(net, eye))
    motor.set(params)                    # weights per fly, see random_params()
    gaze, press, wings = motor.step(net.rate)
"""
import numpy as np
import torch
from scipy import sparse

from flybrain.connectome import DATA_DIR

EXCLUDE = ("R1-6", "R7", "R8")      # photoreceptors are the input, not the brain's reading of it
MIN_CELLS = 20                      # types with fewer cells than this are too sparse to map
PRESS_REFRACTORY = 4                # steps (50 ms) between presses
HABITUATION_TAU = 1.5               # a pressed spot's lost salience recovers with this time constant (s)


def resting_output(net, eye, grey=128, settle=80, average=40):
    """Each neuron's output on a blank grey screen: the level the read-outs measure from."""
    net.reset()
    cur = torch.as_tensor(eye.currents(np.full((eye.height, eye.width, 3), grey, np.uint8)), device=net.device)[:, None]
    for _ in range(settle): net.step(cur)
    rest = torch.zeros(net.n_neurons, device=net.device)
    for _ in range(average): rest += net.step(cur)[:, 0]
    net.reset()
    return rest / average


class Motor:
    def __init__(self, circuit, eye, n_agents=1, dt=0.0125, rest=None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_agents, self.dt = n_agents, dt
        self.rest = torch.zeros(circuit.n_neurons, device=self.device) if rest is None else torch.as_tensor(rest, device=self.device)
        types = circuit.neuron_classes.astype(str)
        rf = eye.rf_screen
        ok = ~np.isnan(rf[:, 0]) & ~np.isin(types, EXCLUDE)
        names, counts = np.unique(types[ok], return_counts=True)
        self.types = [t for t, n in zip(names, counts) if n >= MIN_CELLS]
        T, K = len(self.types), circuit.n_columns
        self.n_types, self.n_columns = T, K
        # each neuron reports to the column whose screen patch holds its receptive-field centre
        centres = eye.centres
        from scipy.spatial import cKDTree
        tree = cKDTree(centres)
        rows, cols = [], []
        for ti, t in enumerate(self.types):
            idx = np.flatnonzero((types == t) & ok)
            _, k = tree.query(rf[idx])
            rows.append(ti * K + k); cols.append(idx)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        per_bin = np.bincount(rows, minlength=T * K).astype(np.float32)
        a = sparse.csr_matrix((1.0 / per_bin[rows], (rows, cols)), shape=(T * K, circuit.n_neurons))
        self.pool = _torch_csr(a, self.device)                 # (T*K, N): mean activity per type and column
        self.present = torch.as_tensor((per_bin > 0).reshape(T, K), device=self.device)
        # the hexagonal neighbourhood: a column and the (up to) six around it
        xy = circuit.columns_xy
        pairs = cKDTree(xy).query_pairs(1.1, output_type="ndarray")
        adj = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(K, K))
        adj = (adj + adj.T + sparse.eye(K)).tocsr()
        adj = sparse.diags(1.0 / np.asarray(adj.sum(1)).ravel()) @ adj
        self.near = torch.as_tensor(adj.toarray(), dtype=torch.float32, device=self.device)   # (K, K), rows sum to 1
        self.centres = centres
        self.mean = torch.zeros(T, device=self.device)
        self.scale = torch.ones(T, device=self.device)
        self.params = None
        self.reset()

    # ------------------------------------------------ normalisation
    def maps(self, rate):
        """(T, K, A): each type's activity in each column, normalised by that type's
        typical level and spread; zero where a type has no cell."""
        m = torch.sparse.mm(self.pool, rate - self.rest[:, None]).view(self.n_types, self.n_columns, -1)
        m = (m - self.mean[:, None, None]) / self.scale[:, None, None]
        return m * self.present[:, :, None]

    def fit_normalisation(self, rates):
        """Mean and spread of each type's column activity over a set of network
        states (e.g. an eye watching the game), so one unit of weight means the
        same for every type."""
        s = torch.zeros(self.n_types, device=self.device); s2 = torch.zeros_like(s); n = torch.zeros_like(s)
        for r in rates:
            m = torch.sparse.mm(self.pool, r - self.rest[:, None]).view(self.n_types, self.n_columns, -1) * self.present[:, :, None]
            s += m.sum((1, 2)); s2 += (m ** 2).sum((1, 2)); n += self.present.sum(1) * m.shape[2]
        self.mean = s / n
        self.scale = (s2 / n - self.mean ** 2).clamp_min(1e-8).sqrt()

    def save_normalisation(self, path=DATA_DIR / "motor_norm.npz"):
        np.savez(path, types=np.array(self.types), mean=self.mean.cpu().numpy(), scale=self.scale.cpu().numpy())

    def load_normalisation(self, path=DATA_DIR / "motor_norm.npz"):
        b = np.load(path)
        assert list(b["types"]) == self.types, "normalisation was fitted for other cell types"
        self.mean = torch.as_tensor(b["mean"], device=self.device)
        self.scale = torch.as_tensor(b["scale"], device=self.device)

    # ------------------------------------------------ weights
    def set(self, params):
        """params: dict of arrays with a leading fly axis -
        gaze (A, T), press (A, T), press_bias (A,), wings (A, T), wings_bias (A,),
        and optionally hold (A,) and habituation (A,)."""
        params = dict(dict(hold=np.full(self.n_agents, 0.25), habituation=np.full(self.n_agents, 4.0)), **params)
        self.params = {k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device) for k, v in params.items()}

    def random_params(self, rng, scale=0.3):
        A, T = self.n_agents, self.n_types
        return dict(gaze=rng.normal(0, scale, (A, T)), press=rng.normal(0, scale, (A, T)), press_bias=np.full(A, -1.0),
                    wings=rng.normal(0, scale, (A, T)), wings_bias=np.full(A, -1.0))

    # ------------------------------------------------ running
    def reset(self):
        A, K = self.n_agents, self.n_columns
        self.fixation = torch.full((A,), K // 2, dtype=torch.long, device=self.device)
        self.habituation = torch.zeros((K, A), device=self.device)
        self.refractory = torch.zeros(A, dtype=torch.long, device=self.device)
        self.salience = torch.zeros((K, A), device=self.device)

    def step(self, rate):
        """One brain step. rate: (N, A). Returns numpy (gaze column, press, wings) per fly."""
        p, ar = self.params, torch.arange(self.n_agents, device=self.device)
        m = self.maps(rate)                                                    # (T, K, A)
        sal = self.near @ torch.einsum("tka,at->ka", m, p["gaze"])             # (K, A)
        sal = sal - self.habituation
        best = sal.argmax(0)
        move = sal[best, ar] > sal[self.fixation, ar] + p["hold"]
        self.fixation = torch.where(move, best, self.fixation)
        fovea = self.near[self.fixation]                                       # (A, K)
        look = torch.einsum("ak,tka->at", fovea, m)                            # (A, T): what is being looked at
        press = ((look * p["press"]).sum(1) + p["press_bias"] > 0) & (self.refractory == 0)
        wings = (m.mean(1).T * p["wings"]).sum(1) + p["wings_bias"] > 0
        self.refractory = torch.where(press, torch.full_like(self.refractory, PRESS_REFRACTORY), (self.refractory - 1).clamp_min(0))
        self.habituation = self.habituation * float(np.exp(-self.dt / HABITUATION_TAU)) + p["habituation"][None, :] * fovea.T * press[None, :]
        self.salience = sal
        return self.fixation.cpu().numpy(), press.cpu().numpy(), wings.cpu().numpy()

    def pointer(self, column):
        """Screen position of a column: where the pointer goes when the fly looks there."""
        return self.centres[column]


def _torch_csr(a, device):
    a = a.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(torch.from_numpy(a.indptr.astype(np.int64)), torch.from_numpy(a.indices.astype(np.int64)),
                                   torch.from_numpy(a.data), size=a.shape, device=device)
