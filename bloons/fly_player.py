"""The closed loop: screen -> eye -> brain -> pointer and press -> game -> screen.

Each fly has its own game, mouse and retina; all of them share one GPU
network with a column per fly.  Every game frame (25 ms):

  1. the retina gives the mean colour each of the eye's 796 columns sees
  2. the photoreceptors get it as current, and the optic lobe runs two steps
     of 12.5 ms
  3. after each step the motor read-out moves the pointer to where the fly
     looks and, if the proboscis extends, presses - through the same Mouse
     a person would use; a wing beat presses Start Round
  4. the game advances one frame

Nothing else reaches the game, and nothing about the game reaches the fly
except through the screen.

    c, eye, net = make_eye()
    motor = Motor(c, eye, n_agents=3, rest=resting_output(net, eye)); motor.set(params)
    arena = Arena(c, eye, net, motor, seeds=[1, 2, 3])
    for _ in range(4000): arena.frame()
"""
import numpy as np
import torch

from bloons import rules as R
from bloons.game import Game
from bloons.screen import Retina, render
from bloons.ui import Mouse
from flybrain.flyvis_eye import FlyvisNetwork
from flybrain.vision import SPECTRAL

STEPS_PER_FRAME = 2


class Arena:
    def __init__(self, circuit, eye, net, motor, seeds, games=None):
        self.circuit, self.eye, self.motor = circuit, eye, motor
        self.n = len(seeds)
        assert motor.n_agents == self.n
        self.games = games or [Game(seed=s) for s in seeds]
        self.mice = [Mouse(g) for g in self.games]
        self.retinas = [Retina(eye.owner, circuit.n_columns) for _ in range(self.n)]
        self.net = FlyvisNetwork(circuit.weights, eye.dyn, n_agents=self.n, dt=net.dt, device=net.device)
        self.net.bias = net.bias.clone(); self.net.reset()
        dev = self.net.device
        idx, cols, spec = [], [], []
        for kind, (i, c) in eye.receptors.items():
            idx.append(i); cols.append(c); spec.append(np.repeat(SPECTRAL[kind][None], len(i), 0) * eye.strength)
        self.r_idx = torch.as_tensor(np.concatenate(idx), device=dev)
        self.r_col = torch.as_tensor(np.concatenate(cols), device=dev)
        self.r_spec = torch.as_tensor(np.concatenate(spec), device=dev)            # (receptors, 3)
        self.frame_no = 0
        self.log = [[] for _ in range(self.n)]                                   # (frame, step, event) per fly
        self.control = np.ones(self.n, dtype=bool)                              # False: the fly only watches

    def colours(self):
        return np.stack([r.colours(g, m) for r, g, m in zip(self.retinas, self.games, self.mice)])   # (A, K, 3)

    def currents(self, colours):
        """Photoreceptor currents on the GPU from the column colours of every fly."""
        col = torch.as_tensor(colours, device=self.net.device)                  # (A, K, 3)
        drive = (col[:, self.r_col, :] * self.r_spec[None]).sum(2)              # (A, receptors)
        cur = torch.zeros((self.net.n_neurons, self.n), device=self.net.device)
        cur[self.r_idx] = drive.T
        return cur

    def frame(self, colours=None):
        """One game frame for every fly. Returns the column colours the flies saw."""
        colours = self.colours() if colours is None else colours
        cur = self.currents(colours)
        self.pointers = np.zeros((STEPS_PER_FRAME, self.n, 2))                 # where each fly looked after each step
        for s in range(STEPS_PER_FRAME):
            rate = self.net.step(cur)
            gaze, press, wings = self.motor.step(rate)
            self.pointers[s] = self.motor.pointer(gaze)
            for a in np.flatnonzero(self.control):
                m, g = self.mice[a], self.games[a]
                m.move(*self.motor.pointer(gaze[a]))
                if press[a]:
                    self.log[a].append((self.frame_no, s, m.press()))
                if wings[a] and not g.in_round and not g.over and not g.won:
                    g.start_round(); self.log[a].append((self.frame_no, s, "wings: start"))
        for g in self.games:
            g.step()
        self.frame_no += 1
        return colours


# ---------------------------------------------------------------- pictures
def eye_view(eye, colours, salience=None, gaze=None, scale=0.5):
    """What the fly sees - each column's patch filled with its mean colour -
    beside its salience map, as one RGB array."""
    view = (colours[eye.owner] * 255).astype(np.uint8)
    panels = [view]
    if salience is not None:
        s = salience - salience.min(); s = s / max(s.max(), 1e-6)
        heat = np.stack([s, s ** 2, 0.2 + 0.0 * s], axis=1)[eye.owner]
        panels.append((heat * 255).astype(np.uint8))
    out = np.concatenate(panels, axis=1)
    if gaze is not None:
        x, y = eye.centres[gaze]
        for dx in range(len(panels)):
            cx = int(x) + dx * eye.width
            out[max(int(y) - 1, 0):int(y) + 2, max(cx - 9, 0):cx + 10] = (255, 255, 255)
            out[max(int(y) - 9, 0):int(y) + 10, max(cx - 1, 0):cx + 2] = (255, 255, 255)
    if scale != 1:
        from PIL import Image
        im = Image.fromarray(out)
        out = np.asarray(im.resize((int(im.width * scale), int(im.height * scale)), Image.NEAREST))
    return out


class Video:
    """Frames to an MP4 through ffmpeg."""

    def __init__(self, path, size, fps=R.FPS):
        import subprocess, shutil
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("ffmpeg not found")
        self.p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p",
                                   "-vcodec", "libx264", "-crf", "23", str(path)], stdin=subprocess.PIPE)

    def add(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())

    def close(self):
        self.p.stdin.close(); self.p.wait()


def composite(game, mouse, eye, colours, salience, gaze):
    """The screen with the fly's pointer, over what the fly sees and where it wants to look."""
    top = np.asarray(render(game, mouse))
    bottom = eye_view(eye, colours, salience, gaze, scale=0.5)
    return np.concatenate([top, bottom], axis=0)
