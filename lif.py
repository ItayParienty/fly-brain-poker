"""Vectorised leaky integrate-and-fire simulation of a connectome circuit.

Same neuron model as neuron.py, which is written one-neuron-at-a-time for
readability. This version runs the whole population at once:

  - the membrane potentials of all N neurons are one array
  - one sparse matrix-vector product delivers every spike to every target
  - several independent flies are simply extra COLUMNS of that array, so a
    six-seat poker table costs barely more than a single fly

Per timestep, in order:
    1. neurons in their refractory period are held at zero
    2. incoming current arrives (spikes from the previous step + external input)
    3. the membrane leaks a fixed fraction
    4. anything at or above threshold spikes, resets, and goes refractory
"""

import numpy as np


class SpikingNetwork:
    def __init__(self, weights, n_agents=1, threshold=1.0, leak=0.15,
                 refractory_steps=3, dtype=np.float32):
        self.weights = weights.astype(dtype)
        self.n_neurons = weights.shape[0]
        self.n_agents = n_agents
        self.threshold = threshold
        self.leak = leak
        self.refractory_steps = refractory_steps
        self.dtype = dtype
        self.reset()

    def reset(self):
        shape = (self.n_neurons, self.n_agents)
        self.potential = np.zeros(shape, dtype=self.dtype)
        self.refractory_left = np.zeros(shape, dtype=np.int8)
        self.spikes = np.zeros(shape, dtype=self.dtype)

    def step(self, external_current=None):
        """Advances one timestep. `external_current` is either None or an
        array shaped (n_neurons, n_agents) - this is where the outside world
        (for us: the poker table) is injected into the sensory neurons."""

        incoming = self.weights @ self.spikes
        if external_current is not None:
            incoming = incoming + external_current

        resting = self.refractory_left > 0
        self.potential[resting] = 0.0
        np.subtract(self.refractory_left, 1, out=self.refractory_left,
                    where=resting)

        self.potential += incoming
        self.potential -= self.leak * self.potential
        self.potential[resting] = 0.0

        fired = self.potential >= self.threshold
        self.potential[fired] = 0.0
        self.refractory_left[fired] = self.refractory_steps

        self.spikes = fired.astype(self.dtype)
        return fired

    def run(self, external_current=None, steps=50, record=None):
        """Runs `steps` timesteps and returns the spike count per neuron.

        `record`: optional array of neuron indices whose full spike train is
        kept, for plotting. Returns (spike_counts, recorded_trains)."""

        counts = np.zeros((self.n_neurons, self.n_agents), dtype=np.int32)
        trains = [] if record is not None else None

        for _ in range(steps):
            fired = self.step(external_current)
            counts += fired
            if trains is not None:
                trains.append(fired[record].copy())

        if trains is not None:
            return counts, np.array(trains)
        return counts
