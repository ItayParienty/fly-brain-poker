"""Does the optic lobe help find bloons - or would the raw pixels do as well?

The fly watches a game (the scripted player builds and plays) and every
other frame we record two descriptions of each of the 796 columns:

  optic lobe     the 214 cell types' activity in that column (normalised,
                 averaged with the neighbours) - what the gaze read-out uses
  photoreceptors R1-6 and R8 in that column and the 18 around it: the raw
                 image patch, before any of the brain's processing

and train the same read-out on each to answer "is a bloon in this column?",
on one game, then test on another: the fraction of frames in which the
column it scores highest is on a bloon.  Two read-outs: one weight per
feature (linear), and a small network shared by every column (32 hidden
units) - the same function applied everywhere, so it knows nothing about
places either.

    python -m bloons.eye_vs_pixels
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

from bloons.bot import ScriptedPlayer
from bloons.fly_player import Arena
from bloons.game import Game
from bloons.watch import watch, bloon_columns, visible
from flybrain.flyvis_eye import make_eye
from flybrain.motor import Motor, resting_output


def patch_index(columns_xy):
    """(K, 19): each column, then the 18 around it in two rings of the hex lattice,
    in a fixed order (the column itself where the lattice ends)."""
    offsets = [(np.cos(a) * r, np.sin(a) * r)
               for r, n, turn in ((1.0, 6, 0), (np.sqrt(3), 6, np.pi / 6), (2.0, 6, 0))
               for a in np.arange(n) * np.pi / 3 + turn]
    tree, K = cKDTree(columns_xy), len(columns_xy)
    idx = np.zeros((K, 1 + len(offsets)), dtype=np.int64); idx[:, 0] = np.arange(K)
    for j, off in enumerate(offsets):
        d, i = tree.query(columns_xy + np.array(off))
        idx[:, j + 1] = np.where(d < 0.3, i, np.arange(K))
    return idx


def train(X, Y, hidden, device, epochs=6, seed=0):
    """A read-out applied to every column: linear, or one hidden layer. Returns a scoring function."""
    torch.manual_seed(seed)
    d = X.shape[2]
    mu, sd = X.float().mean((0, 1)), X.float().std((0, 1)) + 1e-6
    layers = [torch.nn.Linear(d, 1)] if hidden == 0 else [torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)]
    model = torch.nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    Xf, Yf = X.reshape(-1, d), Y.reshape(-1)
    pos_weight = (Yf.numel() - Yf.sum()) / Yf.sum()
    for _ in range(epochs):
        perm = torch.randperm(len(Xf), device=device)
        for i in range(0, len(Xf), 65536):
            b = perm[i:i + 65536]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model((Xf[b].float() - mu) / sd).squeeze(1), Yf[b], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()

    def score(Xt):
        with torch.no_grad():
            return torch.cat([model((Xt[i:i + 200].float() - mu) / sd).squeeze(2) for i in range(0, len(Xt), 200)])
    return score


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    c, eye, net = make_eye()
    motor = Motor(c, eye, n_agents=1, rest=resting_output(net, eye)); motor.load_normalisation()
    T, K, dev = motor.n_types, motor.n_columns, motor.device
    motor.set(dict(gaze=np.zeros((1, T)), press=np.zeros((1, T)), press_bias=np.full(1, -1.0),
                   wings=np.zeros((1, T)), wings_bias=np.full(1, -1.0)))
    rng = np.random.default_rng(0)
    patch = torch.as_tensor(patch_index(c.columns_xy), device=dev)
    receptors = {k: (torch.as_tensor(i, device=dev), torch.as_tensor(cc, device=dev)) for k, (i, cc) in eye.receptors.items()}

    def record(seed, frames):
        g, bot = Game(seed=seed), ScriptedPlayer()
        for _ in range(2):
            bot.act(g); g.start_round()
            while g.in_round: g.step(); bot.act(g) if g.frame % 40 == 0 else None
        a = Arena(c, eye, net, motor, seeds=[seed], games=[g]); a.control[:] = False
        lobe, pixels, ys, where = [], [], [], []

        def on(ar, colours):
            if ar.frame_no % 2:
                return
            lobe.append(torch.einsum("kj,tj->kt", motor.near, motor.maps(ar.net.rate)[:, :, 0]).half())
            rate = ar.net.rate[:, 0] - motor.rest
            per_col = []
            for kind in ("R1-6", "R8"):
                idx, col = receptors[kind]
                v = torch.zeros(K, device=dev).index_reduce_(0, col, rate[idx], "mean", include_self=False)
                per_col.append(v[patch])
            pixels.append(torch.cat(per_col, 1).half())
            ys.append(torch.as_tensor(bloon_columns(eye, ar.games[0], K), device=dev))
            where.append(np.array([b.pos for b in visible(ar.games[0])]).reshape(-1, 2))
        watch(a, bot, frames, on_frame=on, jitter=20, rng=rng)
        return torch.stack(lobe), torch.stack(pixels), torch.stack(ys), where

    train_lobe, train_pix, train_y, _ = record(1, 6000)
    test_lobe, test_pix, _, test_where = record(7, 3000)
    print(f"{len(train_y)} training frames (game 1), {len(test_where)} test frames (game 7)\n")

    def aim(scores):
        best, hits = scores.argmax(1).cpu().numpy(), []
        for f, pos in enumerate(test_where):
            if len(pos):
                gx, gy = eye.centres[best[f]]
                hits.append(np.hypot(pos[:, 0] - gx, pos[:, 1] - gy).min() < 30)
        return np.mean(hits)

    print(f"{'':40s}{'linear':>10s}{'network':>10s}")
    for name, X, Xt in ((f"optic lobe ({T} cell types)", train_lobe, test_lobe),
                        ("photoreceptors (19-column patch)", train_pix, test_pix)):
        res = [aim(train(X, train_y, h, dev)(Xt)) for h in (0, 32)]
        print(f"{name:40s}{res[0]:10.0%}{res[1]:10.0%}")
    print("\n(fraction of test frames in which the top-scoring column is within 30 px of a bloon)")
