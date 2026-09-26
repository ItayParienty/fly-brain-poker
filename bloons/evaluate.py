"""How good is a read-out over many games, rather than the one it was scored on?

In training every fly of a generation plays the same game, and one game says
little: the brains run on the GPU in batches, the sums come out in a different
order in a batch of 33 than for one fly alone, and in a game of tens of
thousands of steps a difference in the seventh digit is enough to send the gaze
elsewhere once - after which the game goes its own way.  (The first fly to win,
in generation 16 of the states run, lost in round 40 when replayed alone on the
same game.)  So a read-out is judged here on many games at once: one copy of
the fly per game, each with its own seed, through the training's workers.

    python -m bloons.evaluate data/first_winner_states_gen16.npy          # a saved read-out
    python -m bloons.evaluate data/train/states --current --games 32      # a run's current read-out

Seeds start at 5000 (training used 1000 + generation).  Prints how each game
ended and the share of games won.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from bloons import rules as R
from bloons.train import EVENTS, Games, play, setup, unpack


def main():
    from flybrain import console_utf8  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--current", action="store_true", help="path is a run folder: take its current read-out")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--first-seed", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    path = Path(args.path)
    theta = np.load(path / "state.npz")["theta"] if args.current else np.load(path)

    A = args.games
    c, eye, halves, photoreceptors = setup(A)
    T = halves[0][3].n_types
    for lo, hi, _, motor in halves:
        motor.set(unpack(np.repeat(theta[None], hi - lo, 0), T))
    seeds = list(range(args.first_seed, args.first_seed + A))
    games = Games(A, eye.owner, c.n_columns, workers=args.workers)
    t0 = time.perf_counter()
    try:
        status, frames = play(games, halves, photoreceptors, seed=seeds)
    finally:
        games.close()

    print(f"{path.name}{' (current)' if args.current else ''}: {len(theta)} numbers, {A} games, "
          f"{time.perf_counter() - t0:.0f} s\n")
    print(f"{'seed':>6}  {'result':<16}{'lives':>6}{'towers':>7}{'upgrades':>9}")
    progress = []
    for seed, (done, fit, round_no, lives, towers, counts) in zip(seeds, status):
        result = "won" if fit >= R.ROUNDS else f"lost in round {round_no + 1}"
        print(f"{seed:>6}  {result:<16}{lives:>6}{towers:>7}{counts[EVENTS.index('upgrade')]:>9}")
        progress.append(fit)
    progress = np.array(progress)
    won = int((progress >= R.ROUNDS).sum())
    print(f"\nwon {won} of {A} ({won / A:.0%}); rounds finished: median {np.median(np.minimum(progress, R.ROUNDS)):.1f}, "
          f"mean {np.minimum(progress, R.ROUNDS).mean():.1f}")


if __name__ == "__main__":
    main()
