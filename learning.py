"""Dopamine-gated plasticity at the Kenyon cell to MBON synapse.

This is the one place in the circuit that changes, and it is the same place
that changes in a real fly. The rule, from the Drosophila learning
literature (Hige et al. 2015; Cohn et al. 2015; Aso & Rubin 2016):

    a Kenyon cell synapse onto an MBON is DEPRESSED when that Kenyon cell was
    active and a dopaminergic neuron fired onto the same compartment.

Three consequences worth keeping in mind, because they are properties of the
biology rather than choices we made:

  - Learning only ever weakens. Nothing is strengthened. A behaviour becomes
    more likely because its competitors got quieter.
  - Only the Kenyon cells that were actually active are affected. Since a
    situation activates ~7% of them, the update is naturally specific to the
    situation the fly was just in, and leaves the rest of its experience alone.
  - The outcome arrives seconds after the decision, so the synapse has to stay
    tagged in the meantime. Real flies do this with a synaptic eligibility
    trace, which is what `Experience` below stands in for.

Reward and punishment are carried by different dopaminergic populations in
the fly, with opposite effects on behaviour. Here a losing hand depresses the
pathway to the action that was taken, and a winning hand depresses the
pathways to the actions that were not - both push the balance the same way a
real reward or punishment signal would.
"""

from collections import namedtuple

import numpy as np

# one decision, kept until the hand pays out
Experience = namedtuple("Experience", "active_kc action")


class PlasticSynapses:
    """Locates the KC->MBON block inside a CSR weight matrix.

    The simulation needs the weights in sparse form for speed, so rather than
    rebuilding the matrix on every update this records, once, where each
    KC->MBON connection lives in the `data` array. Updating a synapse is then
    a direct write into that array.
    """

    def __init__(self, weights, kc_indices, pools):
        self.n_kc = len(kc_indices)
        kc_position = -np.ones(weights.shape[1], dtype=np.int32)
        kc_position[kc_indices] = np.arange(self.n_kc)

        self.blocks = {}
        for action, mbon_indices in pools.items():
            data_positions, kc_columns = [], []
            for row in mbon_indices:
                start, end = weights.indptr[row], weights.indptr[row + 1]
                columns = weights.indices[start:end]
                from_kc = kc_position[columns] >= 0
                data_positions.append(np.arange(start, end)[from_kc])
                kc_columns.append(kc_position[columns[from_kc]])
            self.blocks[action] = (np.concatenate(data_positions),
                                   np.concatenate(kc_columns))

    def n_synapses(self, action):
        return len(self.blocks[action][0])

    def depress(self, weights, action, active_kc, rate):
        """Weakens synapses from the given active Kenyon cells onto one pool."""

        positions, kc_columns = self.blocks[action]
        affected = active_kc[kc_columns]
        weights.data[positions[affected]] *= (1.0 - rate)
        return int(affected.sum())


class LearningFly:
    """Wraps a Fly so that the outcome of each hand reshapes its synapses."""

    def __init__(self, fly, learning_rate=0.02, floor=0.05, recovery=0.01,
                 trace_decay=0.8, reward_scale=10.0, expectation_rate=0.05):
        self.fly = fly
        self.learning_rate = learning_rate
        self.reward_scale = reward_scale  # outcome size that means "full dopamine"

        # Dopamine reports the outcome relative to what was expected, not the
        # raw outcome. This matters in any game you mostly lose: blackjack is
        # lost on 48% of hands even when played perfectly, so a rule that
        # punishes whatever was done on every loss punishes whichever action
        # is used most, correct or not - and flips between them instead of
        # converging. Scoring the outcome against a running average of recent
        # outcomes turns a routine loss into a near-zero event and a win into a
        # strong signal, which is the reward-prediction-error picture of
        # dopamine (Schultz 1997) and is also seen in fly DANs (Felsenberg et
        # al. 2017). expectation_rate=0 recovers the raw-sign rule.
        self.expectation_rate = expectation_rate
        self.expected_outcome = 0.0

        self.floor = floor  # synapses are weakened, never erased or reversed

        # Depression-only learning saturates: every hand weakens something and
        # nothing ever comes back, so after a few hundred hands the circuit
        # falls quiet and the fly folds everything. Flies avoid this by
        # forgetting - depressed synapses recover unless the experience keeps
        # being repeated (Berry et al. 2012; Shuai et al. 2015). Recovery is
        # what turns the rule from a one-way ratchet into a moving balance,
        # and it is why a fly's memory fades rather than accumulating forever.
        self.recovery = recovery

        # How fast a synaptic tag fades between decisions. A hand contains
        # several decisions taken on different streets, with different cards on
        # the table, and only the last of them is close in time to the payout.
        # Crediting them all equally - as a flat list does - blames early,
        # perfectly good decisions for a result they did not cause, and the
        # association never forms. A real eligibility trace decays, so the most
        # recent decision carries most of the credit.
        self.trace_decay = trace_decay

        self.kc_indices = fly.circuit.indices_of("Kenyon_Cell")
        self.synapses = PlasticSynapses(fly.net.weights, self.kc_indices, fly.pools)
        self.initial_weights = fly.net.weights.data.copy()
        self.pending = []
        self.hands_played = 0

    @property
    def name(self):
        return self.fly.name

    def act(self, obs):
        action = self.fly.act(obs)
        active_kc = self.fly.last_counts[self.kc_indices] > 0
        self.pending.append(Experience(active_kc, action))
        return action

    def reward(self, chips):
        """Called once a hand is settled. `chips` is the net result for this
        fly: positive is food, negative is a loss."""

        if not self.pending:
            return

        surprise = chips - self.expected_outcome
        self.expected_outcome += self.expectation_rate * (chips - self.expected_outcome)
        if surprise == 0:
            self.pending.clear()
            return

        # a bigger surprise is a stronger dopamine signal, but with a ceiling
        magnitude = min(abs(surprise) / self.reward_scale, 1.0)
        rate = self.learning_rate * magnitude

        last = len(self.pending) - 1
        for age, experience in enumerate(self.pending):
            tag = self.trace_decay ** (last - age)
            if surprise < 0:
                targets = [experience.action]
            else:
                targets = [a for a in self.fly.pools if a != experience.action]
            for action in targets:
                self.synapses.depress(self.fly.net.weights, action,
                                      experience.active_kc, rate * tag)

        self._recover()
        self._apply_floor()
        self.pending.clear()
        self.hands_played += 1

    def _recover(self):
        """Drifts every plastic synapse back towards the strength it started
        with. Only the synapses a situation keeps re-activating stay depressed."""

        data = self.fly.net.weights.data
        data += (self.initial_weights - data) * self.recovery

    def _apply_floor(self):
        """Keeps depression from driving a synapse through zero, which would
        flip an excitatory connection into an inhibitory one - something
        depression does not do in a real synapse."""

        data = self.fly.net.weights.data
        limit = self.initial_weights * self.floor
        shrunk = np.abs(data) < np.abs(limit)
        data[shrunk] = limit[shrunk]

    def total_depression(self):
        """How far the plastic synapses have moved from their starting point."""
        current, initial = self.fly.net.weights.data, self.initial_weights
        changed = current != initial
        if not changed.any():
            return 0.0
        return float(1.0 - (current[changed] / initial[changed]).mean())
