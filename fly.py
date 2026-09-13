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

import numpy as np

from game import FOLD, CALL, RAISE
from hand_eval import best_hand_score
from lif import SpikingNetwork

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


def preflop_strength(hole):
    """Crude but monotonic ranking of a two-card starting hand, 0..1."""

    a, b = sorted((c.rank_value for c in hole), reverse=True)
    suited = hole[0].suit == hole[1].suit

    if a == b:                      # pocket pair: 0.55 (twos) .. 1.0 (aces)
        return 0.55 + 0.45 * (a - 2) / 12

    score = 0.5 * (a - 2) / 12 + 0.2 * (b - 2) / 12
    if suited:
        score += 0.08
    if a - b == 1:                  # connected
        score += 0.05
    return float(np.clip(score, 0.0, 0.54))


def hand_strength(hole, community):
    """0..1 estimate of how good the hand is right now."""

    if len(community) < 3:
        return preflop_strength(hole)

    score = best_hand_score(hole + community)
    category = score[0]                      # 0 high card .. 8 straight flush
    kicker = (score[1] - 2) / 12 if len(score) > 1 else 0.0
    return float(np.clip((category + kicker) / 9.0, 0.0, 1.0))


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

    def __init__(self, circuit, n_neurons, budget=None, strength=1.0):
        self.olfactory = circuit.indices_of("olfactory")
        self.n_neurons = n_neurons
        self.budget = budget or FEATURE_BUDGET
        self.strength = strength

        # each feature owns a slice of olfactory neurons proportional to its budget
        shares = np.array([self.budget[name] for name in FEATURES], dtype=float)
        edges = np.cumsum(shares / shares.sum() * len(self.olfactory)).astype(int)
        self.bands = np.split(self.olfactory, edges[:-1])
        self.positions = [np.linspace(0.0, 1.0, len(band)) for band in self.bands]

    def __call__(self, feature_values, n_agents=1):
        current = np.zeros((self.n_neurons, n_agents), dtype=np.float32)
        for band, positions, name in zip(self.bands, self.positions, FEATURES):
            distance = np.abs(positions - feature_values[name])
            k = min(self.budget[name], len(band))
            nearest = np.argpartition(distance, k - 1)[:k]
            current[band[nearest], :] = self.strength
        return current


class Fly:
    """One fly, playing poker with its own copy of the mushroom body."""

    def __init__(self, circuit, steps=60, name="fly", calibrate=True):
        self.circuit = circuit
        self.steps = steps
        self.name = name
        self.net = SpikingNetwork(circuit.weights)
        self.encoder = OdourEncoder(circuit, self.net.n_neurons)
        self.pools = {action: circuit.indices_of_nt("MBON", nt)
                      for action, nt in ACTION_POOLS.items()}
        self.last_votes = None
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
            values = {name: float(rng.random()) for name in FEATURES}
            rates = self._pool_rates(self.encoder(values))
            for action, rate in rates.items():
                samples[action].append(rate)

        self.baseline = {
            action: (float(np.mean(values)), float(np.std(values)) or 1.0)
            for action, values in samples.items()
        }

    def _pool_rates(self, current):
        self.net.reset()
        counts = self.net.run(current, steps=self.steps)[:, 0]
        return {action: float(counts[idx].mean()) if len(idx) else 0.0
                for action, idx in self.pools.items()}

    def votes(self, obs):
        """How far each action pool is driven above or below its own norm."""

        rates = self._pool_rates(self.encoder(features(obs)))
        return {action: (rate - self.baseline[action][0]) / self.baseline[action][1]
                for action, rate in rates.items()}

    def act(self, obs):
        self.last_votes = self.votes(obs)
        return max(self.last_votes, key=self.last_votes.get)
