"""Leaky integrate-and-fire (LIF) neuron - the standard toy model used to
simulate spiking activity on top of a connectome graph."""

import numpy as np


class LIFNeuron:
    def __init__(self, name, threshold=1.0, leak=0.15, refractory_steps=3):
        self.name = name
        self.threshold = threshold
        self.leak = leak
        self.refractory_steps = refractory_steps

        self.potential = 0.0
        self.refractory_left = 0
        self.spiked = False

    def step(self, input_current):
        self.spiked = False

        if self.refractory_left > 0:
            self.refractory_left -= 1
            self.potential = 0.0
            return

        self.potential += input_current
        self.potential -= self.leak * self.potential

        if self.potential >= self.threshold:
            self.spiked = True
            self.potential = 0.0
            self.refractory_left = self.refractory_steps
