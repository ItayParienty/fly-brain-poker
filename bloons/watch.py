"""Step 3 check: does the loop work, and is it fast?

The fly does not play here - a scripted player builds the towers and starts
the rounds.  The fly only watches, through its eye, and its gaze drives the
pointer.  To give it something to look at, its gaze read-out is taught one
thing: what a bloon looks like in its optic lobe.  That is a ridge
regression from the cell types' activity in each column to "a bloon is in
this column", fitted while the fly watched one game and tested on another.
The brain is not changed; only one weight per cell type is fitted.

Then, on the new game:
  * aim: how often the pointer is on a bloon, against a random spot
  * speed: game frames per second through the whole loop
  * reaction time: a bloon appears on the track - when does each layer of
    the eye respond at its column, and does the pointer go to it
and a video (data/fly_watch.mp4): the screen with the fly's pointer, and
under it what the fly sees and where it wants to look.

    python -m bloons.watch
"""
import time

import numpy as np
import torch

from bloons import original as O
from bloons import rules as R
from bloons.bot import ScriptedPlayer
from bloons.fly_player import Arena, Video, composite, norm_path, STEPS_PER_FRAME
from bloons.game import Game
from flybrain.connectome import DATA_DIR
from flybrain.flyvis_eye import make_eye
from flybrain.motor import Motor, resting_output

ON_TARGET = 30          # px: the pointer is "on" a bloon within this distance of its centre


def bloon_columns(eye, game, n_columns):
    """1 for every column whose patch holds a bloon's centre (visible bloons only)."""
    y = np.zeros(n_columns, dtype=np.float32)
    for b in visible(game):
        x, yy = b.pos
        y[eye.owner[int(yy), int(x)]] = 1.0
    return y


def visible(game):
    return [b for b in game.bloons if 0 <= b.pos[0] < R.PANEL_X and 0 <= b.pos[1] < R.HEIGHT and not b.popped]


def watch(arena, bot, frames, on_frame=None, jitter=None, rng=None, pause=80):
    """The bot plays arena's game(s) while the flies watch, waiting `pause` frames between
    rounds as a person would; on_frame(arena, colours) after each frame."""
    waited = [0] * len(arena.games)
    for f in range(frames):
        for i, g in enumerate(arena.games):
            if not g.in_round and not g.over and not g.won:
                waited[i] += 1
                if waited[i] >= pause:
                    bot.act(g); g.start_round(); waited[i] = 0
            elif g.frame % 40 == 0:
                bot.act(g)
        if jitter is not None and f % jitter == 0:            # a pointer that wanders, while no read-out moves it
            for m in arena.mice:
                m.move(rng.uniform(0, R.WIDTH), rng.uniform(0, R.HEIGHT))
        colours = arena.frame()
        if on_frame is not None:
            on_frame(arena, colours)


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    rng = np.random.default_rng(0)
    c, eye, net = make_eye()
    motor = Motor(c, eye, n_agents=1, rest=resting_output(net, eye))
    T, K = motor.n_types, motor.n_columns
    print(f"read-out: {T} cell types x {K} columns")
    off = dict(gaze=np.zeros((1, T)), press=np.zeros((1, T)), press_bias=np.full(1, -1.0),
               wings=np.zeros((1, T)), wings_bias=np.full(1, -1.0), hold=np.full(1, 0.05), habituation=np.zeros(1))
    motor.set(off)

    def arena_for(seed, played_rounds=0):
        g = Game(seed=seed); bot = ScriptedPlayer()
        for _ in range(played_rounds):
            bot.act(g); g.start_round()
            while g.in_round: g.step(); bot.act(g) if g.frame % 40 == 0 else None
        a = Arena(c, eye, net, motor, seeds=[seed], games=[g]); a.control[:] = False   # the fly only watches
        return a, bot

    # ---- 1. the eye watches a game: first the normalisation, then the regression
    t0 = time.perf_counter()
    rates = []
    a, bot = arena_for(seed=1)
    watch(a, bot, 2400, on_frame=lambda ar, col: rates.append(ar.net.rate.clone()) if ar.frame_no % 8 == 0 else None,
          jitter=20, rng=rng)
    motor.fit_normalisation(rates); motor.save_normalisation(norm_path()); del rates
    print(f"normalisation from {2400 // 8} frames of rounds 1-{a.games[0].current_round} ({time.perf_counter() - t0:.0f}s)")

    XtX = torch.zeros((T, T), device=motor.device, dtype=torch.float64); Xty = torch.zeros(T, device=motor.device, dtype=torch.float64)
    n_samples = [0]

    def collect(ar, colours):                               # every frame, bloons or not: "not here" counts too
        y = torch.as_tensor(bloon_columns(eye, ar.games[0], K), device=motor.device, dtype=torch.float64)
        X = torch.einsum("kj,tj->kt", motor.near, motor.maps(ar.net.rate)[:, :, 0]).double()     # what the gaze sum sees
        XtX.add_(X.T @ X); Xty.add_(X.T @ y); n_samples[0] += K
    a, bot = arena_for(seed=1)
    watch(a, bot, 5000, on_frame=collect, jitter=20, rng=rng)
    lam = 1e-3 * torch.trace(XtX) / T
    w = torch.linalg.solve(XtX + lam * torch.eye(T, device=motor.device, dtype=torch.float64), Xty).float().cpu().numpy()
    top = np.argsort(-np.abs(w))[:8]
    print(f"gaze weights fitted on {n_samples[0]:,} column samples; strongest: " +
          ", ".join(f"{motor.types[i]} {w[i]:+.2f}" for i in top))

    # the proboscis stays in: pressing gets its job in the next step, playing
    params = dict(off, gaze=w[None])
    motor.set(params); motor.reset()
    np.savez(DATA_DIR / "watch_readout.npz", **params)

    # ---- 2. a new game, the fly's gaze on the pointer: aim, video, speed
    a, bot = arena_for(seed=7, played_rounds=4)
    a.control[:] = True
    hits, chance = [], []
    video = Video(DATA_DIR / "fly_watch.mp4", (640, 480 + 240), fps=R.FPS)
    busy = 0.0; frames = 0; waited = 0
    for f in range(8000):
        g = a.games[0]
        if g.over:
            break
        if not g.in_round:
            waited += 1
            if waited >= 80:
                bot.act(g); g.start_round(); waited = 0
        elif g.frame % 40 == 0:
            bot.act(g)
        shown = visible(g)                                   # what this frame's screen shows
        t0 = time.perf_counter()
        colours = a.frame(); frames += 1
        busy += time.perf_counter() - t0
        px, py = a.mice[0].x, a.mice[0].y                    # the pointer after this frame's two steps
        now = visible(g)
        if shown and now:
            hits.append(any(np.hypot(b.pos[0] - px, b.pos[1] - py) < ON_TARGET for b in now))
            k = rng.integers(K)                              # chance: a random column of the map
            while motor.pointer(k)[0] >= R.PANEL_X: k = rng.integers(K)
            cx, cy = motor.pointer(k)
            chance.append(any(np.hypot(b.pos[0] - cx, b.pos[1] - cy) < ON_TARGET for b in now))
        if 1600 <= f < 2800:                                 # 30 s of video
            video.add(composite(g, a.mice[0], eye, colours[0], motor.salience[:, 0].cpu().numpy(), motor.fixation[0].item()))
    video.close()
    print(f"\nwatched rounds 5-{g.current_round} of a new game (seed 7), {frames} frames")
    print(f"aim: pointer on a bloon in {np.mean(hits):.0%} of frames with bloons on screen (a random spot on the map: {np.mean(chance):.0%})")
    print(f"speed: {frames / busy:.0f} frames/s through the loop without the video ({frames / busy / R.FPS:.1f}x real time)")

    # ---- 3. reaction time: between rounds, with nothing else moving and the message box
    # closed, a bloon appears somewhere on the track and floats along it as bloons do.
    # When does each layer of the eye - and the gaze sum - react at the bloon's column
    # (half of its largest change in the first 150 ms), and does the pointer go there?
    while g.in_round: a.frame()
    g.message_frame = 0                                      # its text is a distractor of its own
    LAYERS = [t for t in ("L1", "L2", "Mi1", "Tm3", "Tm1", "Tm9", "T4a", "T5a", "LC11", "LC17") if t in motor.types]
    looked, onsets = [], {t: [] for t in LAYERS + ["gaze sum"]}
    ms = 1000 / R.FPS / STEPS_PER_FRAME
    for trial in range(60):
        for _ in range(40): a.frame()                        # a second of nothing new
        while True:
            b = g._spawn("Red", int(rng.integers(len(O.TWEENS["Red"]))), -10 + int(rng.integers(20)), 200 + int(rng.integers(20)))
            if 40 < b.pos[0] < 430 and 40 < b.pos[1] < 380 and b.frame < len(O.TWEENS["Red"]) - 60: break
            g.bloons.remove(b)
        k0 = eye.owner[int(b.pos[1]), int(b.pos[0])]
        m0 = motor.maps(a.net.rate)[:, k0, 0].clone(); s0 = float(motor.salience[k0, 0])
        trace = {t: [] for t in onsets}
        seen = None
        for fr in range(24):
            cur = a.currents(a.colours())
            for st in range(STEPS_PER_FRAME):                # the arena's frame, unrolled to look at every step
                rate = a.net.step(cur)
                gaze, _, _ = motor.step(rate)
                m = a.mice[0]; m.move(*motor.pointer(gaze[0]))
                mm = motor.maps(rate)[:, k0, 0]
                for t in LAYERS: trace[t].append(abs(float(mm[motor.types.index(t)] - m0[motor.types.index(t)])))
                trace["gaze sum"].append(float(motor.salience[k0, 0]) - s0)
                if seen is None and np.hypot(m.x - b.pos[0], m.y - b.pos[1]) < ON_TARGET:
                    seen = (fr * STEPS_PER_FRAME + st + 1) * ms
            b.frame += 1
            a.frame_no += 1
        g.bloons.remove(b)
        looked.append(seen)
        for t, v in trace.items():
            v = np.array(v[:12])
            if v.max() > 0:
                onsets[t].append((int(np.argmax(v >= 0.5 * v.max())) + 1) * ms)
    print("reaction: a bloon appears on the track; half of each layer's response at its column after (median of 60):")
    print("          " + ", ".join(f"{t} {np.median(v):.0f} ms" for t, v in onsets.items()))
    got = np.array([t for t in looked if t is not None])
    print(f"          the pointer went to it within 600 ms in {len(got)}/{len(looked)}"
          + (f", median {np.median(got):.0f} ms after it appeared" if len(got) else ""))
