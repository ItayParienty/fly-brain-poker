"""Draws a Game as the 640x480 screen the fly looks at.

Our own drawing, laid out like the original (rules.py has the measured
geometry): the map on the left, the panel on the right with Round / Money /
Lives, the five tower buttons, Start Round, the upgrade panel for a
selected tower, the tower-info box while the pointer is over a button, the
message box, and the pointer itself with the tower it is holding.

A frame is a fixed background plus sprites.  Every bloon, tower, dart,
button and line of text is drawn once, with PIL, into a small image and
cached; a frame is those images pasted at their places, in order.  The one
list of pastes gives two things:

  * `render(game, mouse)` - the whole picture, for people and videos
  * `Retina(owner, n).colours(game, mouse)` - only what each of the eye's
    columns sees: the mean colour of its patch.  It pastes into a canvas
    too, but reads back only the pixels the sprites covered and adds their
    difference from the background to the background's own column means.
    The same pixels as `render`, at a small fraction of the cost - which
    matters, because the fly sees all 40 frames of every second.

    frame = render(game, mouse)              # PIL.Image, RGB, 640x480
"""
import math

import numba
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bloons import rules as R
from bloons.track import point_at

GRASS, GRASS_DARK = (30, 168, 20), (24, 140, 16)
STONE, STONE_EDGE = (133, 133, 133), (96, 96, 96)
PANEL_BG, PANEL_EDGE = (150, 198, 150), (110, 160, 110)
TEXT = (36, 90, 36)
BLOON_COLOURS = {
    "Red": (220, 30, 30), "Blue": (40, 80, 230), "Green": (40, 180, 50),
    "Yellow": (240, 210, 30), "Black": (30, 30, 30), "White": (245, 245, 245),
}
TOWER_COLOURS = {"Dart": (150, 95, 40), "Tack": (245, 150, 190), "Ice": (150, 220, 255),
                 "Bomb": (35, 35, 35), "Super": (230, 40, 40)}
TOWER_NAMES = {"Dart": "Dart Tower", "Tack": "Tack Tower", "Ice": "Ice Tower", "Bomb": "Bomb Tower", "Super": "Super Monkey"}
TOWER_INFO = {                                    # [C] ShowTowerInfo
    "Dart": "Shoots a single dart. Can upgrade to piercing darts and long range darts",
    "Tack": "Shoots volley of tacks in 8 directions. Can upgrade its shoot speed and its range.",
    "Bomb": "Launches a bomb that explodes on impact. Can upgrade to bigger bombs and longer range.",
    "Ice": "Freezes nearby bloons. Frozen bloons are immune to darts and tacks, but bombs will destroy them. "
           "Can upgrade to increased freeze time, and larger freeze radius.",
    "Super": "Super monkey shoots a continuous stream of darts and can mow down even the fastest and most stubborn bloons.",
}
ANGLES = 32                                        # a tower's or dart's heading is drawn in 32 steps

try:
    _FONT = ImageFont.truetype("arial.ttf", 17)
    _FONT_SMALL = ImageFont.truetype("arial.ttf", 12)
    _FONT_BIG = ImageFont.truetype("arial.ttf", 22)
except OSError:                                   # no Arial: PIL's built-in font
    _FONT = _FONT_SMALL = _FONT_BIG = ImageFont.load_default()
DIGIT_W = int(round(_FONT_BIG.getlength("0")))


# ---------------------------------------------------------------- the background
def _background():
    """The map, and the parts of the panel that never change. Drawn once."""
    im = Image.new("RGB", (R.WIDTH, R.HEIGHT), GRASS)
    d = ImageDraw.Draw(im)
    for y in range(0, R.HEIGHT, 8):                  # a little texture so the grass is not one flat value
        for x in range(0, R.PANEL_X, 8):
            if (x * 7 + y * 13) % 5 == 0:
                d.rectangle([x, y, x + 3, y + 3], fill=GRASS_DARK)
    w = R.PATH_HALF_WIDTH
    pts = R.WAYPOINTS
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):        # stone segments, edged
        d.rectangle([min(x0, x1) - w - 2, min(y0, y1) - w - 2, max(x0, x1) + w + 2, max(y0, y1) + w + 2], fill=STONE_EDGE)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d.rectangle([min(x0, x1) - w, min(y0, y1) - w, max(x0, x1) + w, max(y0, y1) + w], fill=STONE)
    s = 0.0
    while s < R.PATH_LENGTH:                            # tile seams every 40 px of track
        x, y = point_at(s)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=STONE_EDGE)
        s += 40
    # the panel's backing; what is written and drawn on it is _panel_front(), which the
    # original puts above the towers, bloons and darts, and the backing below them
    d.rectangle([R.PANEL_X, 0, R.WIDTH, R.HEIGHT], fill=(60, 60, 60))
    d.rounded_rectangle(R.PANEL, radius=8, fill=PANEL_BG, outline=PANEL_EDGE, width=3)
    return im


def _panel_front():
    """Labels, tower buttons and the two buttons under the panel, on a clear sheet
    (drawn over the panel's colour, so their edges blend as they would on it)."""
    im = Image.new("RGBA", (R.WIDTH, R.HEIGHT), PANEL_BG + (0,))
    d = ImageDraw.Draw(im)
    for label, y in R.TEXT_ROWS.items():
        d.text((R.PANEL[0] + 10, y - 10), f"{label}:", fill=TEXT, font=_FONT_BIG)
    d.text((R.PANEL[0] + 10, R.BUILD_LABEL_Y - 10), "Build Towers", fill=TEXT, font=_FONT_BIG)
    d.line([R.PANEL[0] + 10, R.BUILD_LABEL_Y + 11, R.PANEL[2] - 10, R.BUILD_LABEL_Y + 11], fill=TEXT, width=2)
    for kind, x in zip(R.TOWER_ORDER, R.TOWER_BUTTON_X):  # always in colour, as in the original
        r = R.TOWER_BUTTON_R
        d.ellipse([x - r, R.TOWER_BUTTON_Y - r, x + r, R.TOWER_BUTTON_Y + r], fill=TOWER_COLOURS[kind], outline=(40, 40, 40), width=2)
    for box, text in ((R.MORE_GAMES_BUTTON, "More games"), (R.RESTART_BUTTON, "Restart")):
        d.rounded_rectangle(box, radius=9, fill=(200, 215, 200), outline=(40, 40, 40))
        tw = d.textlength(text, font=_FONT_SMALL)
        d.text(((box[0] + box[2]) / 2 - tw / 2, box[1] + 3), text, fill=(20, 20, 20), font=_FONT_SMALL)
    return Sprite(im, 0, 0)


_BG = _background()


# ---------------------------------------------------------------- sprites
class Sprite:
    """A small picture with an anchor: pasted so that the anchor lands on (x, y).
    Only its opaque pixels are kept - every sprite here is solid where it is drawn."""
    __slots__ = ("ax", "ay", "w", "h", "ys", "xs", "rgb", "rgb01")

    def __init__(self, im, ax, ay):
        a = np.asarray(im.convert("RGBA"))
        self.ax, self.ay, self.h, self.w = ax, ay, a.shape[0], a.shape[1]
        self.ys, self.xs = np.nonzero(a[:, :, 3] > 127)
        self.rgb = a[self.ys, self.xs, :3].astype(np.float32)
        self.rgb01 = self.rgb / 255.0                   # as the eye takes it


_CACHE = {}


def sprite(key):
    s = _CACHE.get(key)
    if s is None:
        s = _CACHE[key] = _DRAW[key[0]](*key[1:])
    return s


def _canvas(w, h):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def _bloon(kind, state):
    im, d = _canvas(24, 30)
    c = BLOON_COLOURS[kind]
    if state == "frozen": c = tuple(int(v * 0.5 + 120) for v in c)
    if state == "popped": c = tuple(int(v * 0.6 + 100) for v in c)
    x, y = 12, 14
    d.ellipse([x - 10, y - 13, x + 10, y + 11], fill=c, outline=(20, 20, 20))
    d.polygon([(x, y + 11), (x - 3, y + 15), (x + 3, y + 15)], fill=c)
    d.ellipse([x - 6, y - 9, x - 2, y - 4], fill=tuple(min(255, v + 90) for v in c))
    return Sprite(im, x, y)


def _tower(kind, angle_step, upgrades):
    im, d = _canvas(40, 40)
    x, y, c = 20, 20, TOWER_COLOURS[kind]
    angle = angle_step * 2 * math.pi / ANGLES
    if kind == "Tack":
        d.regular_polygon((x, y, 13), 8, fill=c, outline=(120, 60, 90))
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255))
    elif kind == "Ice":
        d.regular_polygon((x, y, 13), 6, fill=c, outline=(80, 140, 200))
    elif kind == "Bomb":
        d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=c, outline=(90, 90, 90))
        d.line([x, y, x + 14 * math.cos(angle), y + 14 * math.sin(angle)], fill=(90, 90, 90), width=5)
    else:                                                # a monkey: body, head, facing its target
        d.ellipse([x - 12, y - 10, x + 12, y + 10], fill=c, outline=(80, 50, 20))
        hx, hy = x + 9 * math.cos(angle), y + 9 * math.sin(angle)
        d.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], fill=(230, 190, 120) if kind == "Dart" else (40, 60, 200))
    for i, up in enumerate(upgrades):                    # little pips for bought upgrades
        if up: d.ellipse([x - 12 + i * 8, y + 11, x - 7 + i * 8, y + 16], fill=(255, 230, 60))
    return Sprite(im, x, y)


def _ring(r, colour, width):
    im, d = _canvas(2 * r + 4, 2 * r + 4)
    d.ellipse([2, 2, 2 * r + 2, 2 * r + 2], fill=None, outline=colour, width=width)
    return Sprite(im, r + 2, r + 2)


def _dart(angle_step):
    im, d = _canvas(24, 24)
    a = angle_step * 2 * math.pi / ANGLES
    d.line([12, 12, 12 - math.cos(a) * 10, 12 - math.sin(a) * 10], fill=(60, 60, 60), width=3)
    return Sprite(im, 12, 12)


def _dot(r, colour):
    im, d = _canvas(2 * r + 2, 2 * r + 2)
    d.ellipse([0, 0, 2 * r, 2 * r], fill=colour)
    return Sprite(im, r, r)


def _box_outline(w, h, colour, oval):
    im, d = _canvas(w + 1, h + 1)
    (d.ellipse if oval else d.rectangle)([0, 0, w, h], fill=None, outline=colour, width=3 if oval else 2)
    return Sprite(im, 0, 0)


def _pointer():
    im, d = _canvas(14, 21)
    d.polygon([(0, 0), (12, 10), (5, 11), (8, 18), (5, 19), (2, 12), (0, 16)], fill=(255, 255, 255), outline=(0, 0, 0))
    return Sprite(im, 0, 0)


def _text_in(im, d, box, lines, font, colour, align="centre", top=None, step=14):
    y = box[1] + 8 if top is None else top
    for line in lines:
        tw = d.textlength(line, font=font)
        x = (box[0] + box[2]) / 2 - tw / 2 if align == "centre" else box[0] + 8
        d.text((x, y), line, fill=colour, font=font); y += step


def _wrap(d, text, font, width):
    lines, cur = [], ""
    for w in text.split():
        if d.textlength((cur + " " + w).strip(), font=font) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def _digit(ch):
    """One digit of the panel's numbers (Arial's digits are all the same width)."""
    im = Image.new("L", (DIGIT_W + 2, 28), 0)
    ImageDraw.Draw(im).text((0, 0), ch, fill=255, font=_FONT_BIG)
    rgba = Image.new("RGBA", im.size, TEXT + (0,)); rgba.putalpha(im)
    return Sprite(rgba, 0, 0)


def _options(kind, upgrades, affordable, sell_value, rng):
    """The upgrade panel of a selected tower (toweroptions)."""
    x0, y0, x1, y1 = R.PANEL[0] + 4, R.UPGRADE_TITLE_Y - 12, R.PANEL[2] - 4, R.SELL_BUTTON[3] + 2
    im, d = _canvas(x1 - x0, y1 - y0)
    sh = lambda b: (b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0)
    d.rectangle([0, 0, x1 - x0, y1 - y0], fill=PANEL_BG)
    d.text((6, R.UPGRADE_TITLE_Y - 9 - y0), TOWER_NAMES[kind], fill=TEXT, font=_FONT)
    d.text((10, R.UPGRADE_SPEED_Y - 7 - y0), "Speed:", fill=TEXT, font=_FONT_SMALL)
    d.text((86, R.UPGRADE_SPEED_Y - 7 - y0), R.TOWERS[kind].get("speed", ""), fill=TEXT, font=_FONT_SMALL)
    d.text((10, R.UPGRADE_RANGE_Y - 7 - y0), "Range:", fill=TEXT, font=_FONT_SMALL)
    d.text((86, R.UPGRADE_RANGE_Y - 7 - y0), str(rng), fill=TEXT, font=_FONT_SMALL)
    for i, (name, cost, _) in enumerate(R.UPGRADES[kind]):
        box = sh(R.UPGRADE_BUTTONS[i]); bought = upgrades[i]
        fill = (110, 110, 110) if bought else ((70, 150, 70) if affordable[i] else (180, 70, 60))
        d.rounded_rectangle(box, radius=5, fill=fill, outline=(60, 60, 60))
        tail = ["Bought"] if bought else (["Buy for:", str(cost)] if affordable[i] else ["Can't Afford", str(cost)])
        _text_in(im, d, box, name.split() + tail, _FONT_SMALL, (255, 255, 255))
    box = sh(R.SELL_BUTTON)
    d.rounded_rectangle(box, radius=6, fill=(190, 60, 50), outline=(60, 60, 60))
    _text_in(im, d, box, [f"Sell for: {sell_value}"], _FONT_SMALL, (255, 255, 255), top=box[1] + 7)
    return Sprite(im, -x0, -y0)


def _towerinfo(kind):
    x0, y0, x1, y1 = R.TOWERINFO_BOX
    im, d = _canvas(x1 - x0 + 1, y1 - y0 + 1)
    d.rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=6, fill=(235, 245, 235), outline=(60, 90, 60), width=2)
    d.text((8, 6), TOWER_NAMES[kind], fill=TEXT, font=_FONT)
    d.text((8, 28), f"Cost: {R.TOWERS[kind]['cost']}   Speed: {R.TOWERS[kind]['speed']}", fill=TEXT, font=_FONT_SMALL)
    lines = _wrap(d, TOWER_INFO[kind], _FONT_SMALL, x1 - x0 - 16)
    _text_in(im, d, (0, 0, x1 - x0, 0), lines[:7], _FONT_SMALL, (20, 20, 20), align="left", top=48, step=15)
    return Sprite(im, -x0, -y0)


def _start():
    x0, y0, x1, y1 = R.START_BUTTON
    im, d = _canvas(x1 - x0 + 1, y1 - y0 + 1)
    d.rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=6, fill=(90, 170, 90), outline=(60, 60, 60))
    tw = d.textlength("Start Round", font=_FONT)
    d.text(((x1 - x0) / 2 - tw / 2, (y1 - y0) / 2 - 9), "Start Round", fill=(255, 255, 255), font=_FONT)
    return Sprite(im, -x0, -y0)


def _message(text):
    x0, y0, x1, y1 = R.HINT_BOX
    im, d = _canvas(x1 - x0 + 1, y1 - y0 + 1)
    d.rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=8, fill=(255, 255, 255), outline=(60, 60, 60))
    lines = _wrap(d, text, _FONT_SMALL, x1 - x0 - 16)
    _text_in(im, d, (0, 0, x1 - x0, 0), lines[:4], _FONT_SMALL, (20, 20, 20), align="left", top=8, step=16)
    return Sprite(im, -x0, -y0)


def _banner(text):
    im, d = _canvas(241, 61)
    d.rounded_rectangle([0, 0, 240, 60], radius=10, fill=(255, 255, 255), outline=(60, 60, 60))
    tw = d.textlength(text, font=_FONT_BIG)
    d.text((120 - tw / 2, 18), text, fill=(20, 20, 20), font=_FONT_BIG)
    return Sprite(im, -200, -200)


_DRAW = dict(panel=_panel_front, bloon=_bloon, tower=_tower, ring=_ring, dart=_dart, dot=_dot, box=_box_outline, pointer=_pointer,
             digit=_digit, options=_options, towerinfo=_towerinfo, start=_start, message=_message, banner=_banner)


def _step(angle):
    return int(round(angle / (2 * math.pi) * ANGLES)) % ANGLES


# ---------------------------------------------------------------- what is on screen
def draw_list(game, mouse=None):
    """Every sprite on screen this frame, as three layers of (sprite, x, y), each bottom to top:

      world   bloons, darts, towers, rings and the held tower - in the original's
              order, over the map and the panel's backing
      boxes   the panel's labels and buttons, the upgrade panel, tower info,
              Start Round, the message box, the win/lose banner: they cover the
              world, and change only now and then
      top     the panel's numbers and the pointer, over everything
    """
    world, boxes, top = [], [], []
    sel = game.selected if game.selected in game.towers else None
    for b in game.bloons:
        x, y = b.pos
        state = "popped" if b.popped else ("frozen" if b.frozen else "normal")
        world.append((sprite(("bloon", b.kind, state)), int(round(x)), int(round(y))))
    for p in game.bullets:
        for hb, _ in p.boxes():
            if p.kind == "Bomb" and p.hit:
                world.append((sprite(("box", int(hb[2] - hb[0]), int(hb[3] - hb[1]), (255, 140, 30), True)), int(round(hb[0])), int(round(hb[1]))))
            elif p.kind == "Bomb":
                world.append((sprite(("dot", 5, (30, 30, 30))), int(round(p.x)), int(round(p.y))))
            elif p.kind == "Ice":
                world.append((sprite(("box", int(hb[2] - hb[0]), int(hb[3] - hb[1]), (170, 230, 255), False)), int(round(hb[0])), int(round(hb[1]))))
            elif p.kind == "Tack":
                world.append((sprite(("dot", 2, (240, 240, 240))), int(round((hb[0] + hb[2]) / 2)), int(round((hb[1] + hb[3]) / 2))))
            else:
                world.append((sprite(("dart", _step(p.angle))), int(round(p.x)), int(round(p.y))))
    if sel is not None:                                          # the selected tower goes to the bottom, ring showing
        world.append((sprite(("ring", int(sel.range), (255, 255, 255), 2)), sel.x, sel.y))
    for t in ([sel] if sel is not None else []) + [t for t in game.towers if t is not sel]:
        world.append((sprite(("tower", t.kind, _step(t.angle), tuple(t.upgrades))), int(round(t.x)), int(round(t.y))))
    if mouse is not None and mouse.tool is not None:             # the tower being held, and its ring
        red = not mouse.placeable
        world.append((sprite(("ring", R.TOWERS[mouse.tool]["range"], (230, 30, 30) if red else (255, 255, 255), 2)), mouse.x, mouse.y))
        world.append((sprite(("tower", mouse.tool, 0, (False,) * len(R.UPGRADES[mouse.tool]))), mouse.x, mouse.y))

    boxes.append((sprite(("panel",)), 0, 0))
    if sel is not None:
        aff = tuple(game.money >= cost for _, cost, _ in R.UPGRADES[sel.kind])
        boxes.append((sprite(("options", sel.kind, tuple(sel.upgrades), aff, sel.sell_value, int(sel.range))), 0, 0))
    if mouse is not None and mouse.hover is not None:
        boxes.append((sprite(("towerinfo", mouse.hover)), 0, 0))
    if not game.in_round and not game.over and not game.won:
        boxes.append((sprite(("start",)), 0, 0))
    if game.message:
        boxes.append((sprite(("message", game.message)), 0, 0))
    if game.over or game.won:
        boxes.append((sprite(("banner", "You Win!" if game.won else "Game Over")), 0, 0))

    for label, value in (("Round", max(game.current_round, 1)), ("Money", game.money), ("Lives", game.lives)):
        text = str(value)                                        # right-aligned, digit by digit
        x0, y0 = R.PANEL[2] - 10 - DIGIT_W * len(text), R.TEXT_ROWS[label] - 10
        for i, ch in enumerate(text):
            top.append((sprite(("digit", ch)), x0 + i * DIGIT_W, y0))
    if mouse is not None:
        top.append((sprite(("pointer",)), mouse.x, mouse.y))
    return world, boxes, top


# ---------------------------------------------------------------- two ways to look at it
_BG_ARR = np.asarray(_BG).astype(np.float32)


def _paste(img, items):
    """Paste sprites into an (H, W, 3) image, in order."""
    for s, x, y in items:
        ys, xs = s.ys + (y - s.ay), s.xs + (x - s.ax)
        on = (ys >= 0) & (ys < R.HEIGHT) & (xs >= 0) & (xs < R.WIDTH)
        img[ys[on], xs[on]] = s.rgb01[on]


def render(game, mouse=None):
    """The whole screen as a PIL image."""
    world, boxes, top = draw_list(game, mouse)
    img = _BG_ARR / 255.0
    _paste(img, world)
    _paste(img, boxes); _paste(img, top)
    return Image.fromarray(np.rint(img * 255.0).astype(np.uint8))


# every sprite the retina has pasted, in one flat store the compiled loop can index
_STORE = dict(ys=np.zeros(1 << 16, np.int32), xs=np.zeros(1 << 16, np.int32),
              rgb=np.zeros((1 << 16, 3), np.float32), used=0, where={})


def _stored(s):
    """(start, length) of a sprite's pixels in the store, adding it the first time."""
    at = _STORE["where"].get(id(s))
    if at is None:
        n, used = len(s.ys), _STORE["used"]
        while used + n > len(_STORE["ys"]):                    # grow by doubling
            for k in ("ys", "xs", "rgb"):
                a = _STORE[k]; _STORE[k] = np.concatenate([a, np.zeros_like(a)])
        _STORE["ys"][used:used + n], _STORE["xs"][used:used + n], _STORE["rgb"][used:used + n] = s.ys, s.xs, s.rgb01
        _STORE["used"] += n
        at = _STORE["where"][id(s)] = (used, n)
    return at


@numba.njit(cache=True)
def _column_sums(start, count, oy, ox, cut, s_ys, s_xs, s_rgb, canvas, base, covered, owner, written, counted, frame_id, sums):
    """Paste sprite k's pixels at (oy[k], ox[k]) in order into a scratch canvas -
    world pixels (cut[k]) only where no box covers them - then add each pasted
    pixel's change from the base picture to its column, once."""
    h, w = owner.shape
    for k in range(start.shape[0]):
        for i in range(start[k], start[k] + count[k]):
            y, x = s_ys[i] + oy[k], s_xs[i] + ox[k]
            if 0 <= y < h and 0 <= x < w and not (cut[k] and covered[y, x]):
                canvas[y, x, 0] = s_rgb[i, 0]; canvas[y, x, 1] = s_rgb[i, 1]; canvas[y, x, 2] = s_rgb[i, 2]
                written[y, x] = frame_id
    for k in range(start.shape[0]):
        for i in range(start[k], start[k] + count[k]):
            y, x = s_ys[i] + oy[k], s_xs[i] + ox[k]
            if 0 <= y < h and 0 <= x < w and written[y, x] == frame_id and counted[y, x] != frame_id:
                counted[y, x] = frame_id
                c = owner[y, x]
                for j in range(3):
                    sums[c, j] += canvas[y, x, j] - base[y, x, j]


class Retina:
    """What each of the eye's columns sees: the mean colour of its patch of screen.

    `owner` is (480, 640), the column each pixel belongs to (Eye.owner).
    The result equals Eye.column_colours(np.asarray(render(game, mouse))).

    The boxes change rarely and cycle through a few states (a button hovered
    or not, the message box up or not), so for each state the picture of
    background + boxes (the base), its column sums and the mask of what the
    boxes cover are kept.  Every frame only the world and the top layer are
    pasted, into a scratch canvas, and their difference from the base added."""

    KEEP = 12                                               # box states remembered

    def __init__(self, owner, n_columns):
        from collections import OrderedDict
        self.n = n_columns
        self.owner = np.ascontiguousarray(owner, dtype=np.int64)
        self.counts = np.maximum(np.bincount(self.owner.ravel(), minlength=n_columns), 1).astype(np.float64)[:, None]
        self.canvas = np.zeros(owner.shape + (3,), dtype=np.float32)
        self.written = np.zeros(owner.shape, dtype=np.int64)
        self.counted = np.zeros(owner.shape, dtype=np.int64)
        self.frame_id = 0
        self.bg = _BG_ARR / 255.0
        self.bg_sums = np.stack([np.bincount(self.owner.ravel(), weights=self.bg[:, :, c].ravel(), minlength=self.n)
                                 for c in range(3)], axis=1)
        self._states = OrderedDict()

    def _boxes(self, boxes):
        key = tuple((id(s), x, y) for s, x, y in boxes)
        state = self._states.get(key)
        if state is not None:
            self._states.move_to_end(key)
            return state
        base = self.bg.copy()
        covered = np.zeros(self.owner.shape, dtype=np.bool_)
        for s, x, y in boxes:
            ys, xs = s.ys + (y - s.ay), s.xs + (x - s.ax)
            on = (ys >= 0) & (ys < R.HEIGHT) & (xs >= 0) & (xs < R.WIDTH)
            base[ys[on], xs[on]] = s.rgb01[on]; covered[ys[on], xs[on]] = True
        px = np.flatnonzero(covered)
        delta = (base - self.bg).reshape(-1, 3)[px]
        own = self.owner.ravel()[px]
        sums = self.bg_sums + np.stack([np.bincount(own, weights=delta[:, c], minlength=self.n) for c in range(3)], axis=1)
        state = self._states[key] = (base, covered, sums)
        if len(self._states) > self.KEEP:
            self._states.popitem(last=False)
        return state

    def colours(self, game, mouse=None):
        world, boxes, top = draw_list(game, mouse)
        base, covered, base_sums = self._boxes(boxes)
        items = world + top
        where = [_stored(s) for s, _, _ in items]
        start = np.array([a for a, _ in where], dtype=np.int64); count = np.array([n for _, n in where], dtype=np.int64)
        oy = np.array([y - s.ay for s, _, y in items], dtype=np.int64); ox = np.array([x - s.ax for s, x, _ in items], dtype=np.int64)
        cut = np.zeros(len(items), dtype=np.bool_); cut[:len(world)] = True
        sums = base_sums.copy()
        self.frame_id += 1
        _column_sums(start, count, oy, ox, cut, _STORE["ys"], _STORE["xs"], _STORE["rgb"],
                     self.canvas, base, covered, self.owner, self.written, self.counted, self.frame_id, sums)
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
    print(f"rendered in {dt*1000:.1f} ms -> data/bloons_frame.png  (round {g.current_round}, {len(g.bloons)} bloons)")
