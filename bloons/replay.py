"""Watch a trained fly play: one game, as a video, with what it saw and did.

    python -m bloons.replay data/train/linear/best_0039.npy            # a saved read-out
    python -m bloons.replay data/train/linear --current --seed 5       # the run's current read-out

Writes data/replay_<name>.mp4 (the screen, with what the fly sees and its
salience map underneath) and prints every press, as the game saw it.
"""
import argparse
from pathlib import Path

import numpy as np

from bloons import rules as R
from bloons.fly_player import Arena, Video, composite
from bloons.game import Game
from bloons.train import IDLE_SECONDS, unpack
from flybrain.connectome import DATA_DIR
from flybrain.flyvis_eye import make_eye
from flybrain.motor import Motor, resting_output


def main():
    from flybrain import console_utf8  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--current", action="store_true", help="path is a run folder: take its current read-out")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--minutes", type=float, default=3.0, help="at most this much game time on video")
    args = ap.parse_args()
    path = Path(args.path)
    theta = np.load(path / "state.npz")["theta"] if args.current else np.load(path)
    name = (path.name + "_current") if args.current else path.stem

    c, eye, net = make_eye()
    motor = Motor(c, eye, n_agents=1, rest=resting_output(net, eye)); motor.load_normalisation()
    motor.set(unpack(theta[None], motor.n_types))
    game = Game(seed=args.seed)
    arena = Arena(c, eye, net, motor, seeds=[args.seed], games=[game])
    out = DATA_DIR / f"replay_{name}.mp4"
    video = Video(out, (640, 480 + 240))
    idle, logged = 0, 0
    for f in range(int(args.minutes * 60 * R.FPS)):
        colours = arena.frame()
        video.add(composite(game, arena.mice[0], eye, colours[0], motor.salience[:, 0].cpu().numpy(), motor.fixation[0].item()))
        for fr, step, event in arena.log[0][logged:]:
            if event not in ("panel", "deselect"):
                print(f"  {fr / R.FPS:6.2f}s  round {game.current_round:2d}  money {game.money:4d}  {event}")
        logged = len(arena.log[0])
        idle = 0 if game.in_round else idle + 1
        if game.over or game.won or idle > IDLE_SECONDS * R.FPS or any(e == "restart" for _, _, e in arena.log[0]):
            break
    video.close()
    print(f"{'won' if game.won else 'reached round ' + str(game.current_round)} with {game.lives} lives, "
          f"{len(game.towers)} towers; video: {out}")


if __name__ == "__main__":
    main()
