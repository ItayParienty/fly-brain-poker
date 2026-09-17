"""A poker player whose decisions come out of the fly connectome.

Two translations are needed, and both are our design choices rather than
anything the connectome dictates:

INPUT - the table state is presented to the circuit as if it were a smell.
    Olfactory neurons are split into one band per feature, and a feature's
    value lights a localised bump of neurons inside its band, the way a
    sensory map encodes intensity with tuning curves. Nearby values share
    neurons (so the fly generalises across similar spots) while distant
    values do not (so it can tell them apart).

OUTPUT - MBONs are grouped by the transmitter they release, which is what
    carries their valence in the fly literature: cholinergic MBONs tend to
    drive approach, glutamatergic ones avoidance, GABAergic ones suppression.
    Mapping those three onto raise / fold / call is an interpretation, not a
    measurement. Each group's mean firing rate is a vote, and the loudest
    group acts.

What the fly is GIVEN is perception: how strong its hand is, what the pot
odds are. What it has to LEARN is what to do about it.
"""

from pathlib import Path

import numpy as np

from cards.deck import Deck
from cards.game import FOLD, CALL, RAISE
from cards.hand_eval import best_hand_score
from flybrain.lif import SpikingNetwork

# MBON transmitter -> poker action. See module docstring.
ACTION_POOLS = {RAISE: "ACH", FOLD: "GLUT", CALL: "GABA"}

# Feature -> how many olfactory neurons it drives. The totals are held near
# the ~60 the circuit was calibrated with, but the budget is not split evenly:
# a feature that owns more neurons shapes more of the Kenyon cell pattern, and
# therefore more of what can be learned about it. Hand strength gets the
# largest share because two situations that differ only in position should
# still look similar, while two that differ in card strength should not.
FEATURE_BUDGET = {
    "hand_strength": 24,
    "pot_odds": 14,
    "street": 8,
    "position": 6,
    "aggression": 8,
}
FEATURES = list(FEATURE_BUDGET)

STREET_INDEX = {"preflop": 0.0, "flop": 1 / 3, "turn": 2 / 3, "river": 1.0}


def _score_to_int(score):
    """Packs a hand score tuple into one sortable integer, so scores can be
    compared against a precomputed distribution with a binary search."""

    packed = int(score[0])
    padded = list(score[1:]) + [0] * (6 - len(score))
    for tiebreak in padded[:5]:
        packed = packed * 15 + int(tiebreak)
    return packed


class HandStrength:
    """Converts a hand into the share of random hands it beats.

    A raw category number is a poor measure: two pair is a genuinely good
    holding but sits at 2 out of 8, and since made hands cluster at the bottom
    of that range most of the scale goes unused - which both misleads the
    evaluation and wastes half of the neurons the encoder assigns to card
    strength. Ranking against an empirical distribution of real hands spreads
    the values out and gives them a meaning: 0.7 means this beats 70% of what
    an opponent could be holding.
    """

    def __init__(self, n_samples=20000, seed=12345, cache_path=None):
        self.cache_path = cache_path or (Path(__file__).resolve().parents[1] / "data" / "hand_cdf.npy")
        if self.cache_path.exists():
            self.distribution = np.load(self.cache_path)
        else:
            self.distribution = self._sample(n_samples, seed)
            self.cache_path.parent.mkdir(exist_ok=True)
            np.save(self.cache_path, self.distribution)

    @staticmethod
    def _sample(n_samples, seed):
        packed = np.empty(n_samples, dtype=np.int64)
        for i in range(n_samples):
            deck = Deck(seed=seed + i)
            packed[i] = _score_to_int(best_hand_score(deck.deal(2) + deck.deal(5)))
        packed.sort()
        return packed

    def postflop(self, hole, community):
        rank = np.searchsorted(self.distribution, _score_to_int(
            best_hand_score(hole + community)))
        return float(rank / len(self.distribution))

    @staticmethod
    def preflop(hole):
        """Two cards, before any board. Scaled to the same meaning as the
        postflop measure: roughly the share of random holdings it beats, which
        for real starting hands spans about 0.35 (seven-deuce) to 0.85 (aces)."""

        high, low = sorted((c.rank_value for c in hole), reverse=True)
        suited = hole[0].suit == hole[1].suit

        if high == low:
            raw = 0.62 + 0.38 * (high - 2) / 12
        else:
            raw = 0.30 * (high - 2) / 12 + 0.12 * (low - 2) / 12
            if suited:
                raw += 0.05
            if high - low == 1:
                raw += 0.03
            raw += 0.30
        return float(np.clip(raw, 0.0, 1.0))

    def __call__(self, hole, community):
        if len(community) < 3:
            return self.preflop(hole)
        return self.postflop(hole, community)


_strength = None


def hand_strength(hole, community):
    global _strength
    if _strength is None:
        _strength = HandStrength()
    return _strength(hole, community)


def features(obs):
    """Turns an Observation into named values in 0..1."""

    to_call, pot = obs.to_call, max(obs.pot, 1)
    return {
        "hand_strength": hand_strength(obs.hole, obs.community),
        "pot_odds": to_call / (pot + to_call) if to_call else 0.0,
        "street": STREET_INDEX[obs.street],
        "position": 1.0 if obs.is_button else 0.0,
        "aggression": min(obs.raises_this_street / 3.0, 1.0),
    }


class OdourEncoder:
    """Maps feature values onto currents into olfactory neurons.

    Each feature owns a band of olfactory neurons. Its value selects the
    `per_feature` neurons sitting closest to that value along the band, and
    only those are driven. The hard cutoff matters: a soft Gaussian leaks
    current into its tails, and with five features that is enough extra drive
    to saturate the Kenyon cells and collapse the sparse code - at which
    point every poker situation looks alike. Total drive is held at roughly
    the level the circuit was calibrated against.
    """

    def __init__(self, circuit, n_neurons, budget=None, strength=1.0,
                 slots_per_active=2.5):
        self.olfactory = circuit.indices_of("olfactory")
        self.n_neurons = n_neurons
        self.budget = budget or FEATURE_BUDGET
        self.features = list(self.budget)
        self.strength = strength

        # A band is divided into a small number of slots rather than being used
        # neuron by neuron, and this resolution is what decides whether the fly
        # can generalise at all.
        #
        # With one slot per neuron, a band of ~900 neurons and 24 active ones
        # means any change in the feature larger than about 2.6% selects a
        # completely disjoint set: hand strength 0.1 and 0.2 then look exactly
        # as different as 0.1 and 0.9, so "a strong hand" can never form as a
        # category and every value has to be memorised on its own.
        #
        # Keeping only ~2.5 slots per active neuron makes the active set span a
        # useful fraction of the range, so nearby values share most of their
        # neurons and distant ones share none - a tuning curve, which is what
        # sensory neurons actually have.
        shares = np.array([self.budget[name] for name in self.features], dtype=float)
        edges = np.cumsum(shares / shares.sum() * len(self.olfactory)).astype(int)
        bands = np.split(self.olfactory, edges[:-1])

        # The slot axis is padded by half a window on each side. Without the
        # padding, the K nearest slots to any value near an edge are simply
        # the K slots at that edge - so hand totals 18, 19, 20 and 21 all
        # selected the identical set and the fly could not tell them apart,
        # which is exactly the region where it kept hitting.
        self.bands, self.positions = [], []
        for band, name in zip(bands, self.features):
            k = self.budget[name]
            spacing = 1.0 / (k * slots_per_active)
            half = k * spacing / 2
            positions = np.arange(-half, 1.0 + half + spacing / 2, spacing)
            n_slots = min(len(positions), len(band))
            chosen = np.linspace(0, len(band) - 1, n_slots).astype(int)
            self.bands.append(band[chosen])
            self.positions.append(np.linspace(-half, 1.0 + half, n_slots))

    def __call__(self, feature_values, n_agents=1):
        current = np.zeros((self.n_neurons, n_agents), dtype=np.float32)
        for band, positions, name in zip(self.bands, self.positions, self.features):
            distance = np.abs(positions - feature_values[name])
            k = min(self.budget[name], len(band))
            nearest = np.argpartition(distance, k - 1)[:k]
            current[band[nearest], :] = self.strength
        return current


class Fly:
    """One fly, playing poker with its own copy of the mushroom body."""

    def __init__(self, circuit, steps=60, name="fly", calibrate=True,
                 action_pools=None, featurize=None, budget=None):
        self.circuit = circuit
        self.steps = steps
        self.name = name
        self.featurize = featurize or features
        # each fly owns its synapses - flies that learn must diverge from
        # each other rather than share one brain
        self.net = SpikingNetwork(circuit.weights.copy())
        self.encoder = OdourEncoder(circuit, self.net.n_neurons, budget=budget)
        self.pools = {action: circuit.indices_of_nt("MBON", nt)
                      for action, nt in (action_pools or ACTION_POOLS).items()}
        self.last_votes = None
        self.last_counts = None
        # when True, each decision also stores a step-by-step record of which
        # Kenyon cells fired and how the MBON pools' votes built up over time,
        # for visualisation. Costs a Python loop per step, so off by default.
        self.record_trace = False
        self.last_trace = None
        self._kc = circuit.indices_of("Kenyon_Cell")
        self.baseline = {action: (0.0, 1.0) for action in self.pools}
        if calibrate:
            self._calibrate_baseline()

    def _calibrate_baseline(self, n_samples=24, seed=0):
        """Records how hard each MBON pool fires across a spread of ordinary
        situations.

        The three pools are different sizes and different cell types, so their
        resting rates differ by more than any hand ever shifts them - compared
        raw, one pool would always win and the cards would be irrelevant. What
        carries meaning in the mushroom body is the shift away from a pool's
        own balance point, so each pool is scored against its own spread."""

        rng = np.random.default_rng(seed)
        samples = {action: [] for action in self.pools}
        for _ in range(n_samples):
            values = {name: float(rng.random()) for name in self.encoder.features}
            rates = self._pool_rates(self.encoder(values))
            for action, rate in rates.items():
                samples[action].append(rate)

        self.baseline = {
            action: (float(np.mean(values)), float(np.std(values)) or 1.0)
            for action, values in samples.items()
        }

    def _pool_rates(self, current):
        self.net.reset()
        if self.record_trace:
            counts = np.zeros(self.net.n_neurons, dtype=np.int32)
            frames, votes = [], []
            for _ in range(self.steps):
                fired = self.net.step(current)[:, 0]
                counts += fired
                frames.append(np.flatnonzero(fired[self._kc]).tolist())
                votes.append({action: float(counts[idx].mean()) if len(idx) else 0.0
                              for action, idx in self.pools.items()})
            self.last_trace = {"kc_frames": frames, "votes": votes}
        else:
            counts = self.net.run(current, steps=self.steps)[:, 0]
        # kept so a learning rule can see which cells were active when the
        # decision was made - the synapses it may later modify are exactly
        # the ones leaving these cells
        self.last_counts = counts
        return {action: float(counts[idx].mean()) if len(idx) else 0.0
                for action, idx in self.pools.items()}

    def votes(self, obs):
        """How far each action pool is driven above or below its own norm."""

        rates = self._pool_rates(self.encoder(self.featurize(obs)))
        return {action: (rate - self.baseline[action][0]) / self.baseline[action][1]
                for action, rate in rates.items()}

    def act(self, obs):
        self.last_votes = self.votes(obs)
        return max(self.last_votes, key=self.last_votes.get)
