"""The same leaky integrate-and-fire network as lif.py, on the GPU.

Identical neuron model and update order - refractory hold, incoming
current, leak, threshold - so results agree with the numpy version to
floating-point order.  The difference is scale: the mushroom body was
8,574 neurons and ran fine on the CPU; one optic lobe is 47,291 neurons
and 775,000 connections, and the fly has to see every game frame.

    net = TorchNetwork(circuit.weights, n_agents=16)
    fired = net.step(external_current)        # torch tensor (n_neurons, n_agents), bool
"""
import numpy as np
import torch


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


class TorchNetwork:
    def __init__(self, weights, n_agents=1, threshold=1.0, leak=0.15, refractory_steps=3, device=None):
        self.device = torch.device(device or device_name())
        w = weights.tocsr().astype(np.float32)
        self.weights = torch.sparse_csr_tensor(
            torch.from_numpy(w.indptr.astype(np.int64)), torch.from_numpy(w.indices.astype(np.int64)),
            torch.from_numpy(w.data), size=w.shape, device=self.device)
        self.n_neurons, self.n_agents = w.shape[0], n_agents
        self.threshold, self.leak, self.refractory_steps = threshold, leak, refractory_steps
        self.reset()

    def reset(self):
        shape = (self.n_neurons, self.n_agents)
        self.potential = torch.zeros(shape, device=self.device)
        self.refractory_left = torch.zeros(shape, dtype=torch.int8, device=self.device)
        self.spikes = torch.zeros(shape, device=self.device)

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
        incoming = torch.sparse.mm(self.weights, self.spikes)
        if external_current is not None:
            incoming = incoming + self.to_device(external_current)
        resting = self.refractory_left > 0
        self.potential.masked_fill_(resting, 0.0)
        self.refractory_left.sub_(resting.to(torch.int8))
        self.potential.add_(incoming)
        self.potential.mul_(1.0 - self.leak)
        self.potential.masked_fill_(resting, 0.0)
        fired = self.potential >= self.threshold
        self.potential.masked_fill_(fired, 0.0)
        self.refractory_left.masked_fill_(fired, self.refractory_steps)
        self.spikes = fired.to(torch.float32)
        return fired

    def run(self, external_current=None, steps=50):
        """Spike count per neuron over `steps` steps of a constant input."""
        ext = self.to_device(external_current)
        counts = torch.zeros((self.n_neurons, self.n_agents), dtype=torch.int32, device=self.device)
        for _ in range(steps):
            counts += self.step(ext)
        return counts


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    import time
    from flybrain.connectome import load_circuit
    from flybrain.lif import SpikingNetwork
    # agreement with the numpy version on the mushroom body, then speed on the optic lobe
    c = load_circuit()
    rng = np.random.default_rng(0)
    ext = np.zeros((c.n_neurons, 1), dtype=np.float32)
    olf = c.indices_of("olfactory"); ext[rng.choice(olf, 200, replace=False), 0] = 1.0
    a = SpikingNetwork(c.weights).run(ext, steps=60)
    b = TorchNetwork(c.weights).run(ext, steps=60).cpu().numpy()
    print(f"mushroom body: numpy {a.sum()} spikes, torch {b.sum()} spikes, identical: {np.array_equal(a, b)}")
    from flybrain.vision import load_visual_circuit
    v = load_visual_circuit(weight_scale=0.026)
    for agents in (1, 8, 32):
        net = TorchNetwork(v.weights, n_agents=agents)
        ext = torch.rand((v.n_neurons, agents), device=net.device) * 0.5
        net.step(ext); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100): net.step(ext)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / 100
        print(f"optic lobe, {agents:2d} flies: {dt*1000:.2f} ms per step  ({dt*1000/agents:.2f} ms per fly-step)")
