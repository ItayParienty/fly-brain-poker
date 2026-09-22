"""A scripted BTD1 player: proves the clone can be won, and by how much.

It plays the way the community guide says to - Dart Towers on corners that
cover the most track, Piercing Darts, Tack Towers where the track folds
back on itself, a Super Monkey late - following a fixed build order and
placing each tower on the free spot that covers the most track.

    python -m bloons.bot            # one game, round-by-round
"""
import functools

import numpy as np

from bloons import original as O
from bloons import rules as R
from bloons.game import Game, overlap, shifted
from bloons.track import PATH_LENGTH, point_at

# track sampled every 5 px, for measuring how much of it a spot covers
_SAMPLES = [point_at(s) for s in range(0, int(PATH_LENGTH), 5)]
# candidate spots on the grass, every 6 px
_GRID = [(x, y) for x in range(12, R.PANEL_X - 12, 6) for y in range(12, R.HEIGHT - 12, 6)]


def coverage(x, y, rng):
    """Pixels of track within `rng` of (x, y)."""
    r2 = rng * rng
    return 5 * sum(1 for px, py in _SAMPLES if (px - x) ** 2 + (py - y) ** 2 <= r2)


@functools.lru_cache(maxsize=None)
def _ranked(kind, rng):
    """Grid spots clear of the track, most track covered first (ties in grid order).
    The track never changes, so this is worked out once per tower kind and range."""
    grid, samples = np.array(_GRID, dtype=float), np.array(_SAMPLES, dtype=float)
    d2 = ((grid[:, None, :] - samples[None, :, :]) ** 2).sum(-1)
    cover = 5 * (d2 <= rng * rng).sum(1)
    clear = [i for i, (x, y) in enumerate(_GRID)
             if not any(overlap(shifted(O.TOWER_BOX[kind], x, y), blk) for blk in O.PATH_BLOCKS)]
    return [(_GRID[i], int(cover[i])) for i in sorted(clear, key=lambda i: (-cover[i], i))]


def best_spot(game, kind, rng=None):
    """The free spot covering the most track: the first of the ranked spots that is not taken."""
    for where, c in _ranked(kind, rng or R.TOWERS[kind]["range"]):
        if game.can_place(kind, *where):
            return where, c
    return None, -1


# what to buy, in order.  ("Dart",) places a tower; ("up", "Dart", 0) buys
# upgrade 0 on every Dart that lacks it.
BUILD_ORDER = [
    ("Dart",), ("Dart",), ("Dart",),
    ("up", "Dart", 0),
    ("Tack",), ("Tack",),
    ("Dart",), ("Dart",), ("up", "Dart", 0),
    ("Tack",), ("Tack",), ("up", "Tack", 0),
    ("Dart",), ("Dart",), ("Dart",), ("up", "Dart", 0),
    ("Super",), ("up", "Super", 0),
    ("Tack",), ("Tack",), ("up", "Tack", 0), ("up", "Tack", 1),
    ("Super",), ("up", "Super", 0),
    ("Bomb",), ("Bomb",), ("up", "Bomb", 0),
    ("Super",), ("up", "Super", 0),
    ("Super",), ("up", "Super", 0),
]


class ScriptedPlayer:
    def __init__(self, build_order=BUILD_ORDER):
        self.order = list(build_order)
        self.step_no = 0

    def act(self, game):
        """Buy as far down the build order as the money allows."""
        while self.step_no < len(self.order):
            item = self.order[self.step_no]
            if item[0] == "up":
                _, kind, which = item
                pending = [t for t in game.towers if t.kind == kind and not t.upgrades[which]]
                if not pending:
                    self.step_no += 1; continue
                if game.money < R.UPGRADES[kind][which][1]:
                    return
                game.upgrade(pending[0], which)
            else:
                kind = item[0]
                if game.money < R.TOWERS[kind]["cost"]:
                    return
                where, _ = best_spot(game, kind)
                if where is None:
                    self.step_no += 1; continue
                game.place(kind, *where)
                self.step_no += 1


def play(player=None, verbose=False, act_every=40):
    """Play a whole game; the player acts between rounds and every `act_every` frames."""
    game, player = Game(), player or ScriptedPlayer()
    while not game.over and not game.won:
        player.act(game)
        game.start_round()
        while game.in_round and not game.over:
            game.step()
            if game.frame % act_every == 0:
                player.act(game)
        if verbose:
            towers = {}
            for t in game.towers: towers[t.kind] = towers.get(t.kind, 0) + 1
            print(f"round {game.round_no:2d}  lives {game.lives:3d}  money {game.money:5d}  "
                  + " ".join(f"{k}:{n}" for k, n in towers.items()))
    return game


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    import time
    t0 = time.perf_counter()
    g = play(verbose=True)
    print(f"\n{'WON' if g.won else 'LOST in round ' + str(g.current_round)}: lives {g.lives}, "
          f"pops {g.stats['pops']}, leaked {g.stats['leaks']} lives, {g.frame} frames, "
          f"{time.perf_counter() - t0:.1f}s")
