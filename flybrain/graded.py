"""A graded-potential network for the optic lobe, on the GPU.

Most neurons of the fly's optic lobe do not spike.  Photoreceptors, lamina
and medulla cells signal with continuous membrane potentials, and the
transmitter they release follows that potential smoothly.  Simulating them
as integrate-and-fire units adds a coarse quantisation the fly does not
have: at the low rates the calibration settles on, a neuron's output is a
handful of spikes and the position of a small object is lost in the
counting noise (verify_vision.py, before this file existed).

So the eye is graded and the brain spikes.  Same weights, same leak:

    v  <-  (1 - leak) * (v + W r + I + b)
    r  =   clip(v, 0, 1)                    what the neuron releases

`b` is the resting current fitted by Eye.calibrate; `I` the photoreceptor
drive.  Several flies are extra columns of `v`, as in lif.py.
"""
import numpy as np
import torch


class GradedNetwork:
    threshold = 1.0                      # so Eye.calibrate can treat it like TorchNetwork

    def __init__(self, weights, n_agents=1, leak=0.15, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        w = weights.tocsr().astype(np.float32)
        self.weights = torch.sparse_csr_tensor(
            torch.from_numpy(w.indptr.astype(np.int64)), torch.from_numpy(w.indices.astype(np.int64)),
            torch.from_numpy(w.data), size=w.shape, device=self.device)
        self.n_neurons, self.n_agents, self.leak = w.shape[0], n_agents, leak
        self.reset()

    def reset(self):
        self.potential = torch.zeros((self.n_neurons, self.n_agents), device=self.device)
        self.rate = torch.zeros_like(self.potential)

    def to_device(self, external_current):
        if external_current is None:
            return None
        if not torch.is_tensor(external_current):
            external_current = torch.as_tensor(np.asarray(external_current, dtype=np.float32))
        external_current = external_current.to(self.device)
        if external_current.dim() == 1:
            external_current = external_current[:, None].expand(-1, self.n_agents)
        return external_current

    def step(self, external_current=None):
        incoming = torch.sparse.mm(self.weights, self.rate)
        if external_current is not None:
            incoming = incoming + self.to_device(external_current)
        self.potential = (self.potential + incoming) * (1.0 - self.leak)
        self.rate = self.potential.clamp(0.0, 1.0)
        return self.rate

    def run(self, external_current=None, steps=50):
        """Mean output per neuron over `steps` steps of a constant input."""
        ext = self.to_device(external_current)
        total = torch.zeros((self.n_neurons, self.n_agents), device=self.device)
        for _ in range(steps):
            total += self.step(ext)
        return total / steps
