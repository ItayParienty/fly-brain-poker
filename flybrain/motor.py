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

Three rules shape the gaze.  It stays where it is unless another column
beats it by `hold` (flies fixate, then saccade).  A spot the fly has just
pressed loses `habituation` of its salience, and a spot it keeps looking at
loses `fatigue` per step; both recover over 1.5 s.  So the gaze moves on,
instead of pressing the same button forever or staring at one spot.

The gaze and the press can read the eye differently in each state of the
hand (n_states; for the game: holding nothing, holding a tower, a tower
selected - bloons/ui.py MODES), so what the fly looks for can depend on
what it is doing - as a fly's visual responses change with its behaviour
(walking or flying, for one).  The state comes with each step; a weight
still never belongs to a place on the screen.

    motor = Motor(circuit, eye, n_agents, rest=resting_output(net, eye))
    motor.set(params)                    # weights per fly, see random_params()
    gaze, press, wings = motor.step(net.rate, state)
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
    def __init__(self, circuit, eye, n_agents=1, dt=0.0125, rest=None, device=None, n_states=3):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_agents, self.dt, self.n_states = n_agents, dt, n_states
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
        # the same pooling with the type folded out, for the read-outs: (K, N), and each neuron's type
        self._column_pool = _torch_csr(sparse.csr_matrix((1.0 / per_bin[rows], (rows % K, cols)), shape=(K, circuit.n_neurons)), self.device)
        type_of = np.full(circuit.n_neurons, T, dtype=np.int64); type_of[cols] = rows // K
        self._type_of = torch.as_tensor(type_of, device=self.device)
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
        self._refold()

    def save_normalisation(self, path=DATA_DIR / "motor_norm.npz"):
        np.savez(path, types=np.array(self.types), mean=self.mean.cpu().numpy(), scale=self.scale.cpu().numpy())

    def load_normalisation(self, path=DATA_DIR / "motor_norm.npz"):
        b = np.load(path)
        assert list(b["types"]) == self.types, "normalisation was fitted for other cell types"
        self.mean = torch.as_tensor(b["mean"], device=self.device)
        self.scale = torch.as_tensor(b["scale"], device=self.device)
        self._refold()

    def _refold(self):
        if self.params is not None:                         # the weights carry the normalisation: redo them
            self.set({k: v.cpu().numpy() for k, v in self.params.items()})

    # ------------------------------------------------ weights
    def set(self, params):
        """params: dict of arrays with a leading fly axis -
        gaze (A, S, T), press (A, S, T), press_bias (A, S): one set per state, or
        (A, T), (A, T), (A,): the same in every state; wings (A, T), wings_bias (A,),
        and optionally hold (A,), habituation (A,) and fatigue (A,)."""
        A, S = self.n_agents, self.n_states
        params = dict(dict(hold=np.full(A, 0.25), habituation=np.full(A, 4.0), fatigue=np.zeros(A)), **params)
        params = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
        for k in ("gaze", "press"):
            if params[k].ndim == 2:
                params[k] = np.repeat(params[k][:, None], S, 1)
        if params["press_bias"].ndim == 1:
            params["press_bias"] = np.repeat(params["press_bias"][:, None], S, 1)
        self.params = {k: torch.as_tensor(v, device=self.device) for k, v in params.items()}
        self._folded = False

    def set_state(self, state):
        """Each fly's state, (A,) ints: which of its weight sets the gaze and press use."""
        state = np.asarray(state, dtype=np.int64)
        if not np.array_equal(state, self._state):
            self._state = state.copy()
            self.state = torch.as_tensor(state, device=self.device)
            self._folded = False

    def _fold(self):
        """Each fly's weights for its current state, with the normalisation folded in:
        per neuron w / scale, and a per-column offset."""
        A, ar = self.n_agents, torch.arange(self.n_agents, device=self.device)
        w = torch.stack([self.params["gaze"][ar, self.state], self.params["press"][ar, self.state], self.params["wings"]])  # (3, A, T)
        per_type = torch.cat([w / self.scale, torch.zeros_like(w[:, :, :1])], 2)            # a zero for unread neurons
        self._neuron_w = per_type.permute(2, 0, 1).reshape(self.n_types + 1, 3 * A)[self._type_of]   # (N, 3A)
        self._offset = self.present.T.float() @ (w * (self.mean / self.scale)).permute(2, 0, 1).reshape(self.n_types, 3 * A)
        self._folded = True

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
        self._state = np.zeros(A, np.int64)
        self.state = torch.zeros(A, dtype=torch.long, device=self.device)
        self._folded = False

    def sums(self, rate):
        """The three weighted sums in every column, (3, K, A): gaze, press and wings -
        sum over types of weight x maps(rate), computed without building the maps."""
        A = self.n_agents
        if not self._folded:
            self._fold()
        x = (rate - self.rest[:, None]).repeat(1, 3) * self._neuron_w                     # (N, 3A)
        return (torch.sparse.mm(self._column_pool, x) - self._offset).view(-1, 3, A).permute(1, 0, 2)

    def step(self, rate, state=None):
        """One brain step. rate: (N, A); state: (A,) each fly's state, if it changed.
        Returns numpy (gaze column, press, wings) per fly."""
        if state is not None:
            self.set_state(state)
        p, ar = self.params, torch.arange(self.n_agents, device=self.device)
        gaze_sum, press_sum, wings_sum = self.sums(rate)                        # each (K, A)
        sal = self.near @ gaze_sum - self.habituation
        best = sal.argmax(0)
        move = sal[best, ar] > sal[self.fixation, ar] + p["hold"]
        self.fixation = torch.where(move, best, self.fixation)
        fovea = self.near[self.fixation]                                       # (A, K): what is being looked at
        press = ((fovea * press_sum.T).sum(1) + p["press_bias"][ar, self.state] > 0) & (self.refractory == 0)
        wings = wings_sum.mean(0) + p["wings_bias"] > 0
        self.refractory = torch.where(press, torch.full_like(self.refractory, PRESS_REFRACTORY), (self.refractory - 1).clamp_min(0))
        self.habituation = (self.habituation * float(np.exp(-self.dt / HABITUATION_TAU))
                            + fovea.T * (p["habituation"] * press + p["fatigue"])[None, :])
        self.salience = sal
        out = torch.stack([self.fixation, press.long(), wings.long()]).cpu().numpy()        # one trip to the CPU
        return out[0], out[1].astype(bool), out[2].astype(bool)

    def pointer(self, column):
        """Screen position of a column: where the pointer goes when the fly looks there."""
        return self.centres[column]


def _torch_csr(a, device):
    a = a.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(torch.from_numpy(a.indptr.astype(np.int64)), torch.from_numpy(a.indices.astype(np.int64)),
                                   torch.from_numpy(a.data), size=a.shape, device=device)
