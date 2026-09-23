"""Step 4: teach the fly to play - by evolution, with the brain frozen.

The connectome, its dynamics and the eye are fixed.  What evolves is the
read-out (flybrain/motor.py): one weight per cell type for the gaze, one
for the proboscis, one for the wings, two biases, and the three constants
of the gaze (hold, habituation, fatigue) - 647 numbers in all.  None of
them belongs to a place on the screen.

Evolution strategy (OpenAI-ES): each generation, 32 random directions in
the space of read-outs are tried both ways round - 64 flies - plus the
current read-out itself, all on the same game (same random bloon offsets,
so the comparison is fair).  Flies are ranked by how far they got, and the
read-out moves towards the directions whose flies did better.

How far a fly got: rounds finished, plus the fraction of the current
round's layers it popped before it lost; a win is 50 plus a bonus for lives
left.  A fly's game ends when it loses, wins, presses Restart, or sits for
IDLE_SECONDS between rounds without starting the next.

The games run in worker processes (the screen, the mouse, the rules); the
brains run on the GPU, in two halves of the population taking turns, so that
one half's brains think while the other half's games move.

    python -m bloons.train --generations 300          # resumes from data/train/ if present
"""
import argparse
import json
import multiprocessing as mp
import time
from multiprocessing import shared_memory

import numpy as np

from bloons import rules as R

STEPS_PER_FRAME = 2
IDLE_SECONDS = 10
MAX_FRAMES = 100_000
OUT = None                    # set in main: data/train


# ---------------------------------------------------------------- the games, in worker processes
EVENTS = ["pick", "place", "select", "upgrade", "sell", "broke", "red ring", "blocked", "deselect", "panel", "start",
          "restart", "wings"]


def _progress(g, pops0):
    """How far a game got: rounds finished + the part of the current round popped."""
    if g.won:
        return R.ROUNDS + g.lives / R.START_LIVES
    frac = 0.0
    if g.in_round or g.over:
        rbe = R.round_rbe(g.round_no + 1)
        frac = min((g.stats["pops"] - pops0) / max(rbe, 1), 1.0)
    return g.round_no + 0.99 * frac


def worker(conn, lo, hi, owner, n_columns, colours_name, actions_name, n_total, n_columns_total):
    from bloons.game import Game
    from bloons.screen import Retina
    from bloons.ui import Mouse
    n = hi - lo
    shm_c = shared_memory.SharedMemory(name=colours_name)
    shm_a = shared_memory.SharedMemory(name=actions_name)
    colours = np.ndarray((n_total, n_columns_total, 3), dtype=np.float32, buffer=shm_c.buf)
    actions = np.ndarray((n_total, STEPS_PER_FRAME, 4), dtype=np.float32, buffer=shm_a.buf)
    from bloons.fly_player import RETINA_STRIDE
    retinas = [Retina(owner, n_columns, RETINA_STRIDE) for _ in range(n)]
    games = mice = None
    while True:
        cmd, arg = conn.recv()
        if cmd == "stop":
            break
        if cmd == "reset":
            games = [Game(seed=arg) for _ in range(n)]
            mice = [Mouse(g) for g in games]
            done = np.zeros(n, bool); fitness = np.zeros(n); idle = np.zeros(n, int); pops0 = np.zeros(n)
            counts = np.zeros((n, len(EVENTS)), int); frames = np.zeros(n, int)
            for i in range(n):
                colours[lo + i] = retinas[i].colours(games[i], mice[i])
            conn.send(None)
            continue
        # one frame for the flies in [arg[0], arg[1]): both brain steps' actions, in order, then the rules
        for i in np.flatnonzero(~done):
            if not arg[0] <= lo + i < arg[1]:
                continue
            frames[i] += 1
            g, m = games[i], mice[i]
            for s in range(STEPS_PER_FRAME):
                x, y, press, wings = actions[lo + i, s]
                m.move(x, y)
                if press:
                    before = _progress(g, pops0[i])
                    ev = m.press().split(" ")[0]
                    ev = {"red": "red ring", "upgrade": "upgrade"}.get(ev, ev)
                    counts[i, EVENTS.index(ev)] += 1
                    if ev == "restart":
                        done[i] = True; fitness[i] = before
                        break
                    if ev == "start":
                        pops0[i] = g.stats["pops"]
                if wings and not g.in_round and not g.over and not g.won:
                    g.start_round(); pops0[i] = g.stats["pops"]; counts[i, EVENTS.index("wings")] += 1
            if done[i]:
                continue
            g.step()
            idle[i] = 0 if g.in_round else idle[i] + 1
            fitness[i] = _progress(g, pops0[i])
            if g.over or g.won or idle[i] > IDLE_SECONDS * R.FPS or frames[i] >= MAX_FRAMES:
                done[i] = True
            else:
                colours[lo + i] = retinas[i].colours(g, m)
        status = [(bool(done[i]), float(fitness[i]), games[i].round_no, games[i].lives, len(games[i].towers),
                   counts[i].tolist()) for i in range(n)]
        conn.send(status)
    shm_c.close(); shm_a.close()


class Games:
    """The flies' games, spread over worker processes, talking through shared memory."""

    def __init__(self, n, owner, n_columns, workers=7):
        self.n, self.K = n, n_columns
        self.shm_c = shared_memory.SharedMemory(create=True, size=n * n_columns * 3 * 4)
        self.shm_a = shared_memory.SharedMemory(create=True, size=n * STEPS_PER_FRAME * 4 * 4)
        self.colours = np.ndarray((n, n_columns, 3), dtype=np.float32, buffer=self.shm_c.buf)
        self.actions = np.ndarray((n, STEPS_PER_FRAME, 4), dtype=np.float32, buffer=self.shm_a.buf)
        bounds = np.linspace(0, n, workers + 1).astype(int)
        self.conns, self.procs = [], []
        ctx = mp.get_context("spawn")
        for lo, hi in zip(bounds, bounds[1:]):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, lo, hi, owner, n_columns, self.shm_c.name, self.shm_a.name, n, n_columns),
                            daemon=True)
            p.start(); self.conns.append(a); self.procs.append(p)

    def reset(self, seed):
        for c in self.conns: c.send(("reset", seed))
        for c in self.conns: c.recv()

    def send(self, flies):
        """Start one frame for the flies in the range (lo, hi); recv() waits for it."""
        for c in self.conns: c.send(("frame", flies))

    def recv(self):
        """Every fly's status after the last frame sent."""
        out = []
        for c in self.conns: out += c.recv()
        return out

    def close(self):
        for c in self.conns: c.send(("stop", None))
        for p in self.procs: p.join(timeout=5)
        self.shm_c.close(); self.shm_c.unlink(); self.shm_a.close(); self.shm_a.unlink()


# ---------------------------------------------------------------- the read-out as one vector
def layout(T):
    return [("gaze", T), ("press", T), ("wings", T), ("press_bias", 1), ("wings_bias", 1),
            ("hold", 1), ("habituation", 1), ("fatigue", 1)]


def unpack(theta, T):
    """(A, d) -> the Motor's params. The gaze constants are kept positive."""
    out, i = {}, 0
    for name, size in layout(T):
        v = theta[:, i:i + size]; i += size
        out[name] = v if size > 1 else v[:, 0]
    out["hold"] = np.abs(out["hold"]); out["habituation"] = np.abs(out["habituation"]) * 4
    out["fatigue"] = np.abs(out["fatigue"]) * 0.05
    return out


def initial(T, rng):
    theta = np.concatenate([rng.normal(0, 0.05, 3 * T), [-0.5, 0.5, 0.1, 1.0, 0.2]]).astype(np.float32)
    return theta


class Adam:
    def __init__(self, d, lr, b1=0.9, b2=0.999):
        self.m, self.v, self.t, self.lr, self.b1, self.b2 = np.zeros(d), np.zeros(d), 0, lr, b1, b2

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        return self.lr * (self.m / (1 - self.b1 ** self.t)) / (np.sqrt(self.v / (1 - self.b2 ** self.t)) + 1e-8)


def centred_ranks(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


# ---------------------------------------------------------------- one generation
def play(games, halves, photoreceptors, seed, max_frames=MAX_FRAMES):
    """Every fly plays one game with its read-out; returns the final status per fly.

    halves: [(lo, hi, net, motor), ...] - two groups of flies whose turns
    interleave: while one group's games advance a frame in the workers, the
    GPU works out the other group's next actions."""
    games.reset(seed)
    for _, _, net, motor in halves:
        net.reset(); motor.reset()

    def act(h):
        lo, hi, net, motor = halves[h]
        cur = photoreceptors.currents(games.colours[lo:hi])
        for s in range(STEPS_PER_FRAME):
            gaze, press, wings = motor.step(net.step(cur))
            xy = motor.centres[gaze]
            games.actions[lo:hi, s, 0] = xy[:, 0]; games.actions[lo:hi, s, 1] = xy[:, 1]
            games.actions[lo:hi, s, 2] = press; games.actions[lo:hi, s, 3] = wings

    act(0); games.send(halves[0][:2]); act(1)
    turn, frames = 0, 0
    while True:
        status = games.recv()                                # group `turn` has played a frame
        frames += turn == 0
        if all(st[0] for st in status) or frames >= max_frames:
            return status, frames
        games.send(halves[1 - turn][:2])
        act(turn)
        turn = 1 - turn


def main():
    global OUT
    from flybrain import console_utf8  # noqa: F401
    from flybrain.connectome import DATA_DIR
    from flybrain.flyvis_eye import make_eye, FlyvisNetwork
    from flybrain.motor import Motor, resting_output
    from bloons.fly_player import Photoreceptors, fit_normalisation, norm_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--run", default="linear")
    ap.add_argument("--start-from", help="a run whose current read-out this one starts from")
    args = ap.parse_args()
    OUT = DATA_DIR / "train" / args.run
    OUT.mkdir(parents=True, exist_ok=True)

    c, eye, net1 = make_eye()
    A = 2 * args.pairs + 1                                   # the last fly plays the current read-out itself
    rest = resting_output(net1, eye)
    halves = []
    for lo, hi in ((0, A // 2), (A // 2, A)):
        net = FlyvisNetwork(c.weights, eye.dyn, n_agents=hi - lo, dt=net1.dt, device=net1.device)
        net.bias = net1.bias.clone()
        motor = Motor(c, eye, n_agents=hi - lo, rest=rest)
        if not norm_path().exists():
            print("fitting the read-outs' normalisation to this look ...", flush=True)
            fit_normalisation(c, eye, net1, motor)
        motor.load_normalisation(norm_path())
        halves.append((lo, hi, net, motor))
    photoreceptors = Photoreceptors(eye, net1.n_neurons, net1.device)
    T = motor.n_types
    rng = np.random.default_rng(0)

    state_file = OUT / "state.npz"
    if state_file.exists():
        st = np.load(state_file)
        theta, gen0 = st["theta"], int(st["generation"]) + 1
        opt = Adam(len(theta), args.lr); opt.m, opt.v, opt.t = st["m"], st["v"], int(st["t"])
        rng = np.random.default_rng(gen0)
        print(f"resuming {OUT.name} at generation {gen0}")
    elif args.start_from:
        theta, gen0 = np.load(DATA_DIR / "train" / args.start_from / "state.npz")["theta"], 0
        opt = Adam(len(theta), args.lr)
        print(f"starting from {args.start_from}'s current read-out")
    else:
        theta, gen0 = initial(T, rng), 0
        opt = Adam(len(theta), args.lr)
    history_file = OUT / "history.jsonl"
    print(f"{A} flies per generation, {len(theta)} numbers in the read-out, {args.workers} game workers")

    games = Games(A, eye.owner, c.n_columns, workers=args.workers)
    try:
        for gen in range(gen0, gen0 + args.generations):
            t0 = time.perf_counter()
            eps = rng.normal(0, 1, (args.pairs, len(theta))).astype(np.float32)
            pop = np.concatenate([theta + args.sigma * eps, theta - args.sigma * eps, theta[None]])
            for lo, hi, _, motor in halves:
                motor.set(unpack(pop[lo:hi], T))
            status, frames = play(games, halves, photoreceptors, seed=1000 + gen)
            fit = np.array([s[1] for s in status])
            u = centred_ranks(fit[:-1])
            grad = ((u[:args.pairs] - u[args.pairs:])[:, None] * eps).sum(0) / (2 * args.pairs * args.sigma)
            theta = (theta + opt.step(grad - 0.002 * theta)).astype(np.float32)
            counts = np.array([s[5] for s in status])
            best = int(np.argmax(fit))
            rec = dict(generation=gen, seconds=round(time.perf_counter() - t0, 1), frames=frames,
                       mean=round(float(fit[:-1].mean()), 3), best=round(float(fit.max()), 3), current=round(float(fit[-1]), 3),
                       best_round=status[best][2], best_lives=status[best][3], best_towers=status[best][4],
                       towers=round(float(np.mean([s[4] for s in status])), 2),
                       events={e: round(float(v), 1) for e, v in zip(EVENTS, counts.mean(0))})
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            np.savez(state_file, theta=theta, generation=gen, m=opt.m, v=opt.v, t=opt.t)
            np.save(OUT / f"best_{gen:04d}.npy", pop[best])
            ev = rec["events"]
            print(f"gen {gen:4d}  {rec['seconds']:6.1f}s {frames:6d} frames | progress mean {rec['mean']:6.2f} "
                  f"best {rec['best']:6.2f} current {rec['current']:6.2f} | towers {rec['towers']:5.2f} "
                  f"(best fly: round {rec['best_round']}, {rec['best_towers']} towers) | "
                  f"picks {ev['pick']:.1f} places {ev['place']:.1f} sells {ev['sell']:.1f} restarts {ev['restart']:.2f}",
                  flush=True)
    finally:
        games.close()


if __name__ == "__main__":
    main()
