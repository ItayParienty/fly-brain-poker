"""Draws a Game as the 640x480 screen the fly looks at.

With the original game's art extracted (bloons/extract_art.py -> data/btd1_art/)
the screen is the original's: its map, monkeys, bloons, darts, panel and
buttons, drawn in the original's order and animated as it animates them - a
bloon's pose, a monkey's throwing arm, the Start Round button's glow, the
message box fading in and out (bloons/look_art.py).  Without it, simple
drawings of our own stand in (bloons/look_drawn.py).  BTD_LOOK=drawn picks
the drawings even when the art is there.

A frame is a background plus three layers of pictures, each (sprite, x, y):

  world   bloons, darts, towers, the held tower - over the map and the
          panel's backing, as in the original's depths
  boxes   the panel's buttons and labels, the upgrade panel, tower info,
          Start Round, the message box, the banner: over the world; they
          change only now and then
  top     the panel's numbers and the pointer

Pictures have soft edges (alpha), so a pixel is the world blended over the
background, the boxes blended over that, and the top over everything.  The
one list of pictures gives two things:

  * `Painter().picture(game, mouse)` / `render(game, mouse)` - the whole screen
  * `Retina(owner, n).colours(game, mouse)` - only what each of the eye's
    columns sees, the mean colour of its patch: the same pixels, but only
    those the world and top layers touch are worked out each frame

    frame = render(game, mouse)              # PIL.Image, RGB, 640x480
"""
import math
import os
from collections import OrderedDict

import numba
import numpy as np
from PIL import Image

from bloons import art
from bloons import rules as R

if art.available() and os.environ.get("BTD_LOOK", "art") != "drawn":
    from bloons import look_art as LOOK
    LOOK_NAME = "art"
else:
    from bloons import look_drawn as LOOK
    LOOK_NAME = "drawn"


# ---------------------------------------------------------------- sprites
class Sprite:
    """A picture with an anchor, pasted so that the anchor lands on (x, y).
    Keeps the pixels that are not fully transparent, with their opacity."""
    __slots__ = ("ax", "ay", "ys", "xs", "rgb", "a")

    def __init__(self, rgba, ox, oy):
        rgba = np.asarray(rgba)
        self.ax, self.ay = int(round(ox)), int(round(oy))
        self.ys, self.xs = np.nonzero(rgba[:, :, 3])
        self.rgb = rgba[self.ys, self.xs, :3].astype(np.float32) / 255.0
        self.a = rgba[self.ys, self.xs, 3].astype(np.float32) / 255.0


_CACHE = {}


def sprite(name, *args):
    """LOOK.<name>(*args) as a Sprite, made once."""
    key = (name,) + args
    s = _CACHE.get(key)
    if s is None:
        s = _CACHE[key] = Sprite(*getattr(LOOK, name)(*args))
    return s


def _rot(deg):
    return int(round(deg / 360.0 * LOOK.ROT_STEPS)) % LOOK.ROT_STEPS


def _pose(bloon):
    """The frame a bloon stopped on when it appeared (random(50) in the original;
    here fixed by its id, so the game's own random numbers are left alone)."""
    return (bloon.id * 7919) % LOOK.POSES


# ---------------------------------------------------------------- what is on screen
def draw_list(game, mouse=None):
    """Every picture on screen this frame, as (world, boxes, top) lists of (sprite, x, y)."""
    world, boxes, top = [], [], []
    sel = game.selected if game.selected in game.towers else None
    for b in game.bloons:                                        # bloonholder
        x, y = b.tween
        frame = LOOK.pop_frame(b.kind) + min(b.pop_timer, 3) if b.popped else _pose(b)
        freeze = min(b.time_frozen, 29) // 4 * 4 + 1 if b.frozen else 0
        world.append((sprite("bloon", b.kind, frame, freeze), int(round(x)), int(round(y))))
    for p in game.bullets:                                       # bulletholder
        x, y = int(round(p.x)), int(round(p.y))
        if p.kind == "Tack":
            for i, on in enumerate(p.tacks):
                if on:
                    world.append((sprite("tack", i + 1, min(p.age, 6)), x, y))
        elif p.kind == "Bomb":
            frame = min(p.hit_age + 1, 14) if p.hit else 0
            world.append((sprite("bomb", frame, _rot(math.degrees(p.angle) + 90), round(p.scale, 2)), x, y))
        elif p.kind == "Ice":
            world.append((sprite("ice", min(p.age, 10), round(p.scale, 2)), x, y))
        else:
            world.append((sprite("dart", p.kind, _rot(math.degrees(p.angle) + 90)), x, y))
    towers = ([sel] if sel is not None else []) + [t for t in game.towers if t is not sel]
    for t in towers:                                             # towerholder: the selected one at the bottom
        if t is sel:
            world.append((sprite("ring", t.kind, False), int(round(t.x)), int(round(t.y))))
        arm = t.since_shot if t.shots and t.since_shot < LOOK.ARM_FRAMES[t.kind] else 0   # the arm swings after a shot
        world.append((sprite("tower", t.kind, _rot(t.rotation), arm), int(round(t.x)), int(round(t.y))))
    if mouse is not None and mouse.tool is not None:             # towerplace
        world.append((sprite("ring", mouse.tool, not mouse.placeable), mouse.x, mouse.y))
        world.append((sprite("ghost", mouse.tool), mouse.x, mouse.y))

    if sel is not None:                                          # toweroptions (depth 161, under the buttons)
        aff = tuple(game.money >= cost for _, cost, _ in R.UPGRADES[sel.kind])
        boxes.append((sprite("options", sel.kind, tuple(sel.upgrades), aff, sel.sell_value, int(sel.range), sel.attack_rate), 0, 0))
    boxes.append((sprite("panel_front"), 0, 0))
    hover = mouse.hover if mouse is not None else None
    if hover is not None:
        boxes.append((sprite("button_over", hover), 0, 0))
    if not game.in_round and not game.over and not game.won:
        boxes.append((sprite("start_button", game.tick % LOOK.START_FRAMES), 0, 0))
    if game.end_tick is not None:
        boxes.append((sprite("banner", game.won, min(game.tick - game.end_tick, LOOK.BANNER_FRAMES - 1)), 0, 0))
    if game.message and game.message_frame:                      # fading in, showing (all alike), fading out
        f = game.message_frame
        boxes.append((sprite("message", game.message, f if f < 15 or f > 220 else 20), 0, 0))
    if hover is not None:
        boxes.append((sprite("towerinfo", hover), 0, 0))

    for field, value in (("level_txt", max(game.current_round, 1)), ("money_txt", game.money), ("lives_txt", max(game.lives, 0))):
        top.append((sprite("number", field, value), 0, 0))
    if mouse is not None:
        top.append((sprite("pointer"), mouse.x, mouse.y))
    return world, boxes, top


# ---------------------------------------------------------------- the boxes, kept per state
_BG = LOOK.background().astype(np.float32) / 255.0
_STATES = OrderedDict()
KEEP = 16


def _boxes(boxes):
    """(base = boxes over the background, premultiplied box colour, box opacity) for a set of boxes."""
    key = tuple((id(s), x, y) for s, x, y in boxes)
    state = _STATES.get(key)
    if state is not None:
        _STATES.move_to_end(key)
        return state
    pm = np.zeros(_BG.shape, np.float32)
    a = np.zeros(_BG.shape[:2], np.float32)
    for s, x, y in boxes:
        ys, xs = s.ys + (y - s.ay), s.xs + (x - s.ax)
        on = (ys >= 0) & (ys < R.HEIGHT) & (xs >= 0) & (xs < R.WIDTH)
        ys, xs, sa = ys[on], xs[on], s.a[on]
        pm[ys, xs] = pm[ys, xs] * (1 - sa[:, None]) + s.rgb[on] * sa[:, None]
        a[ys, xs] = a[ys, xs] * (1 - sa) + sa
    base = pm + (1 - a[:, :, None]) * _BG
    state = _STATES[key] = (base, pm, a)
    if len(_STATES) > KEEP:
        _STATES.popitem(last=False)
    return state


def _blend(img, items):
    for s, x, y in items:
        ys, xs = s.ys + (y - s.ay), s.xs + (x - s.ax)
        on = (ys >= 0) & (ys < R.HEIGHT) & (xs >= 0) & (xs < R.WIDTH)
        ys, xs, sa = ys[on], xs[on], s.a[on][:, None]
        img[ys, xs] = img[ys, xs] * (1 - sa) + s.rgb[on] * sa


class Painter:
    """The whole screen as a uint8 array."""

    def picture(self, game, mouse=None):
        world, boxes, top = draw_list(game, mouse)
        base, pm, a = _boxes(boxes)
        img = _BG.copy()
        _blend(img, world)
        img = pm + (1 - a[:, :, None]) * img
        _blend(img, top)
        return np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)


def render(game, mouse=None):
    """The whole screen as a PIL image."""
    return Image.fromarray(Painter().picture(game, mouse))


# ---------------------------------------------------------------- what the eye's columns see
# every sprite the retina has pasted, in one flat store the compiled loop can index
_STORE = dict(ys=np.zeros(1 << 16, np.int32), xs=np.zeros(1 << 16, np.int32), rgb=np.zeros((1 << 16, 3), np.float32),
              a=np.zeros(1 << 16, np.float32), used=0, where={})


def _stored(s, stride=1, py=0, px=0):
    """(start, length) in the store of a sprite's pixels that land on the retina's
    sampling grid when the sprite is pasted at an offset with (y, x) % stride == (py, px),
    in grid coordinates; added the first time."""
    key = (id(s), stride, py, px)
    at = _STORE["where"].get(key)
    if at is None:
        if stride == 1:
            ys, xs, rgb, a = s.ys, s.xs, s.rgb, s.a
        else:
            keep = ((s.ys + py) % stride == 0) & ((s.xs + px) % stride == 0)
            ys, xs, rgb, a = (s.ys[keep] + py) // stride, (s.xs[keep] + px) // stride, s.rgb[keep], s.a[keep]
        n, used = len(ys), _STORE["used"]
        while used + n > len(_STORE["ys"]):                    # grow by doubling
            for k in ("ys", "xs", "rgb", "a"):
                v = _STORE[k]; _STORE[k] = np.concatenate([v, np.zeros_like(v)])
        _STORE["ys"][used:used + n], _STORE["xs"][used:used + n] = ys, xs
        _STORE["rgb"][used:used + n], _STORE["a"][used:used + n] = rgb, a
        _STORE["used"] += n
        at = _STORE["where"][key] = (used, n)
    return at


@numba.njit(cache=True)
def _column_sums(start, count, oy, ox, is_world, s_ys, s_xs, s_rgb, s_a, bg, base, pm, box_a, world, canvas, owner,
                 w_stamp, c_stamp, a_stamp, frame_id, sums):
    """1. the world's pixels, blended over the bare background; 2. the boxes over them;
    3. the top layer over everything; 4. each changed pixel's difference from the
    base (boxes over background) added to its column, once."""
    h, w = owner.shape
    n = start.shape[0]
    for k in range(n):
        if not is_world[k]:
            continue
        for i in range(start[k], start[k] + count[k]):
            y, x = s_ys[i] + oy[k], s_xs[i] + ox[k]
            if 0 <= y < h and 0 <= x < w:
                if w_stamp[y, x] != frame_id:
                    w_stamp[y, x] = frame_id
                    for j in range(3):
                        world[y, x, j] = bg[y, x, j]
                a = s_a[i]
                for j in range(3):
                    world[y, x, j] = world[y, x, j] * (1 - a) + s_rgb[i, j] * a
    for k in range(n):
        if not is_world[k]:
            continue
        for i in range(start[k], start[k] + count[k]):
            y, x = s_ys[i] + oy[k], s_xs[i] + ox[k]
            if 0 <= y < h and 0 <= x < w and c_stamp[y, x] != frame_id:
                c_stamp[y, x] = frame_id
                for j in range(3):
                    canvas[y, x, j] = pm[y, x, j] + (1 - box_a[y, x]) * world[y, x, j]
    for k in range(n):
        if is_world[k]:
            continue
        for i in range(start[k], start[k] + count[k]):
            y, x = s_ys[i] + oy[k], s_xs[i] + ox[k]
            if 0 <= y < h and 0 <= x < w:
                if c_stamp[y, x] != frame_id:
                    c_stamp[y, x] = frame_id
                    for j in range(3):
                        canvas[y, x, j] = base[y, x, j]
                a = s_a[i]
                for j in range(3):
                    canvas[y, x, j] = canvas[y, x, j] * (1 - a) + s_rgb[i, j] * a
    for k in range(n):
        for i in range(start[k], start[k] + count[k]):
            y, x = s_ys[i] + oy[k], s_xs[i] + ox[k]
            if 0 <= y < h and 0 <= x < w and c_stamp[y, x] == frame_id and a_stamp[y, x] != frame_id:
                a_stamp[y, x] = frame_id
                c = owner[y, x]
                for j in range(3):
                    sums[c, j] += canvas[y, x, j] - base[y, x, j]


class Retina:
    """What each of the eye's columns sees: the mean colour of its patch of screen.

    `owner` is (480, 640), the column each pixel belongs to (Eye.owner).  With
    stride 1 the result equals Eye.column_colours(Painter().picture(game, mouse));
    with stride 2 each column averages every other pixel in each direction -
    about a hundred of its four hundred - which is four times less work and
    differs from the full average by a fraction of a percent (bloons/screen.py
    __main__ measures it)."""

    def __init__(self, owner, n_columns, stride=1):
        self.n, self.stride = n_columns, stride
        self.owner = np.ascontiguousarray(owner[::stride, ::stride], dtype=np.int64)
        self.counts = np.maximum(np.bincount(self.owner.ravel(), minlength=n_columns), 1).astype(np.float64)[:, None]
        self.world = np.zeros(self.owner.shape + (3,), np.float32)
        self.canvas = np.zeros(self.owner.shape + (3,), np.float32)
        self.stamps = [np.zeros(self.owner.shape, np.int64) for _ in range(3)]
        self.bg = np.ascontiguousarray(_BG[::stride, ::stride])
        self.frame_id = 0
        self._states = {}

    def _state(self, boxes):
        """The boxes' arrays on this retina's grid, and the base's column sums."""
        key = tuple((id(s), x, y) for s, x, y in boxes)
        st = self._states.get(key)
        if st is None:
            base, pm, a = _boxes(boxes)
            k = self.stride
            base, pm, a = (np.ascontiguousarray(v[::k, ::k]) for v in (base, pm, a))
            own = self.owner.ravel()
            sums = np.stack([np.bincount(own, weights=base[:, :, c].ravel(), minlength=self.n) for c in range(3)], axis=1)
            st = self._states[key] = (base, pm, a, sums)
            if len(self._states) > KEEP:
                self._states.pop(next(iter(self._states)))
        return st

    def colours(self, game, mouse=None):
        world, boxes, top = draw_list(game, mouse)
        base, pm, box_a, base_sums = self._state(boxes)
        k = self.stride
        items = world + top
        oy = [y - s.ay for s, _, y in items]; ox = [x - s.ax for s, x, _ in items]
        where = [_stored(s, k, dy % k, dx % k) for (s, _, _), dy, dx in zip(items, oy, ox)]
        start = np.array([a for a, _ in where], dtype=np.int64); count = np.array([n for _, n in where], dtype=np.int64)
        oy = np.array(oy, dtype=np.int64) // k; ox = np.array(ox, dtype=np.int64) // k
        is_world = np.zeros(len(items), dtype=np.bool_); is_world[:len(world)] = True
        sums = base_sums.copy()
        self.frame_id += 1
        _column_sums(start, count, oy, ox, is_world, _STORE["ys"], _STORE["xs"], _STORE["rgb"], _STORE["a"],
                     self.bg, base, pm, box_a, self.world, self.canvas, self.owner, *self.stamps, self.frame_id, sums)
        return (sums / self.counts).astype(np.float32)


if __name__ == "__main__":
    from bloons.bot import ScriptedPlayer
    from bloons.game import Game
    from bloons.ui import Mouse
    import time
    g, bot = Game(), ScriptedPlayer()
    for _ in range(12):
        bot.act(g); g.start_round()
        while g.in_round: g.step(); bot.act(g) if g.frame % 40 == 0 else None
    bot.act(g); g.start_round()
    for _ in range(700): g.step()
    g.selected = g.towers[0]
    m = Mouse(g); m.move(300, 250); m.tool = "Tack"
    t0 = time.perf_counter(); im = render(g, m); dt = time.perf_counter() - t0
    im.save("data/bloons_frame.png")
    print(f"look: {LOOK_NAME}; rendered in {dt*1000:.1f} ms -> data/bloons_frame.png  (round {g.current_round}, {len(g.bloons)} bloons)")
