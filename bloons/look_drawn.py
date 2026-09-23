"""How things look without the original art: simple drawings of our own.

The fallback for anyone without the game's SWF (see bloons/extract_art.py):
the same objects, in the same places, drawn with PIL - circles for monkeys,
ovals for bloons.  Same functions as look_art; each returns a straight RGBA
uint8 picture and its origin.
"""
import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bloons import rules as R
from bloons.track import point_at

GRASS, GRASS_DARK = (30, 168, 20), (24, 140, 16)
STONE, STONE_EDGE = (133, 133, 133), (96, 96, 96)
PANEL_BG, PANEL_EDGE = (150, 198, 150), (110, 160, 110)
TEXT = (36, 90, 36)
BLOON_COLOURS = {"Red": (220, 30, 30), "Blue": (40, 80, 230), "Green": (40, 180, 50),
                 "Yellow": (240, 210, 30), "Black": (30, 30, 30), "White": (245, 245, 245)}
TOWER_COLOURS = {"Dart": (150, 95, 40), "Tack": (245, 150, 190), "Ice": (150, 220, 255), "Bomb": (35, 35, 35), "Super": (230, 40, 40)}
TOWER_NAMES = {"Dart": "Dart Tower", "Tack": "Tack Tower", "Ice": "Ice Tower", "Bomb": "Bomb Tower", "Super": "Super Monkey"}
ROT_STEPS = 64
POSES = 1                                   # bloons have one pose here
ARM_FRAMES = {"Dart": 1, "Tack": 1, "Ice": 1, "Bomb": 1, "Super": 1}
START_FRAMES, MESSAGE_FRAMES, BANNER_FRAMES = 1, 234, 47

try:
    _FONT = ImageFont.truetype("arial.ttf", 17)
    _FONT_SMALL = ImageFont.truetype("arial.ttf", 12)
    _FONT_BIG = ImageFont.truetype("arial.ttf", 22)
except OSError:                                   # no Arial: PIL's built-in font
    _FONT = _FONT_SMALL = _FONT_BIG = ImageFont.load_default()


def _canvas(w, h):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    return im, ImageDraw.Draw(im)


def _pic(im, ox, oy):
    return np.asarray(im.convert("RGBA")).copy(), ox, oy


@functools.lru_cache(maxsize=None)
def background():
    im = Image.new("RGB", (R.WIDTH, R.HEIGHT), GRASS)
    d = ImageDraw.Draw(im)
    for y in range(0, R.HEIGHT, 8):
        for x in range(0, R.PANEL_X, 8):
            if (x * 7 + y * 13) % 5 == 0:
                d.rectangle([x, y, x + 3, y + 3], fill=GRASS_DARK)
    w = R.PATH_HALF_WIDTH
    for (x0, y0), (x1, y1) in zip(R.WAYPOINTS, R.WAYPOINTS[1:]):
        d.rectangle([min(x0, x1) - w - 2, min(y0, y1) - w - 2, max(x0, x1) + w + 2, max(y0, y1) + w + 2], fill=STONE_EDGE)
    for (x0, y0), (x1, y1) in zip(R.WAYPOINTS, R.WAYPOINTS[1:]):
        d.rectangle([min(x0, x1) - w, min(y0, y1) - w, max(x0, x1) + w, max(y0, y1) + w], fill=STONE)
    s = 0.0
    while s < R.PATH_LENGTH:
        x, y = point_at(s); d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=STONE_EDGE); s += 40
    d.rectangle([R.PANEL_X, 0, R.WIDTH, R.HEIGHT], fill=(60, 60, 60))
    d.rounded_rectangle(R.PANEL, radius=8, fill=PANEL_BG, outline=PANEL_EDGE, width=3)
    return np.asarray(im).copy()


@functools.lru_cache(maxsize=None)
def panel_front():
    im = Image.new("RGBA", (R.WIDTH, R.HEIGHT), PANEL_BG + (0,))
    d = ImageDraw.Draw(im)
    for label, y in R.TEXT_ROWS.items():
        d.text((R.PANEL[0] + 10, y - 10), f"{label}:", fill=TEXT, font=_FONT_BIG)
    d.text((R.PANEL[0] + 10, R.BUILD_LABEL_Y - 10), "Build Towers", fill=TEXT, font=_FONT_BIG)
    d.line([R.PANEL[0] + 10, R.BUILD_LABEL_Y + 11, R.PANEL[2] - 10, R.BUILD_LABEL_Y + 11], fill=TEXT, width=2)
    for kind, x in zip(R.TOWER_ORDER, R.TOWER_BUTTON_X):
        r = R.TOWER_BUTTON_R
        d.ellipse([x - r, R.TOWER_BUTTON_Y - r, x + r, R.TOWER_BUTTON_Y + r], fill=TOWER_COLOURS[kind], outline=(40, 40, 40), width=2)
    for box, text in ((R.MORE_GAMES_BUTTON, "More games"), (R.RESTART_BUTTON, "Restart")):
        d.rounded_rectangle(box, radius=9, fill=(200, 215, 200), outline=(40, 40, 40))
        tw = d.textlength(text, font=_FONT_SMALL)
        d.text(((box[0] + box[2]) / 2 - tw / 2, box[1] + 3), text, fill=(20, 20, 20), font=_FONT_SMALL)
    return _pic(im, 0, 0)


@functools.lru_cache(maxsize=None)
def tower(kind, rot, arm=0):
    im, d = _canvas(40, 40)
    x, y, c = 20, 20, TOWER_COLOURS[kind]
    angle = math.radians(rot * 360 / ROT_STEPS - 90)
    if kind == "Tack":
        d.regular_polygon((x, y, 13), 8, fill=c, outline=(120, 60, 90)); d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255))
    elif kind == "Ice":
        d.regular_polygon((x, y, 13), 6, fill=c, outline=(80, 140, 200))
    elif kind == "Bomb":
        d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=c, outline=(90, 90, 90))
        d.line([x, y, x + 14 * math.cos(angle), y + 14 * math.sin(angle)], fill=(90, 90, 90), width=5)
    else:
        d.ellipse([x - 12, y - 10, x + 12, y + 10], fill=c, outline=(80, 50, 20))
        hx, hy = x + 9 * math.cos(angle), y + 9 * math.sin(angle)
        d.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], fill=(230, 190, 120) if kind == "Dart" else (40, 60, 200))
    return _pic(im, x, y)


@functools.lru_cache(maxsize=None)
def ring(kind, red=False):
    r = R.TOWERS[kind]["range"]
    im, d = _canvas(2 * r + 4, 2 * r + 4)
    d.ellipse([2, 2, 2 * r + 2, 2 * r + 2], fill=None, outline=(230, 30, 30) if red else (255, 255, 255), width=2)
    return _pic(im, r + 2, r + 2)


def ghost(kind):
    return tower(kind, 0)


@functools.lru_cache(maxsize=None)
def bloon(kind, frame=0, freeze=0):
    """frame: 0 = floating, 1 = popping (drawn paler)."""
    im, d = _canvas(24, 30)
    c = BLOON_COLOURS[kind]
    if freeze: c = tuple(int(v * 0.5 + 120) for v in c)
    if frame: c = tuple(int(v * 0.6 + 100) for v in c)
    x, y = 12, 14
    d.ellipse([x - 10, y - 13, x + 10, y + 11], fill=c, outline=(20, 20, 20))
    d.polygon([(x, y + 11), (x - 3, y + 15), (x + 3, y + 15)], fill=c)
    d.ellipse([x - 6, y - 9, x - 2, y - 4], fill=tuple(min(255, v + 90) for v in c))
    return _pic(im, x, y)


def pop_frame(kind):
    return 1


@functools.lru_cache(maxsize=None)
def dart(kind, rot):
    im, d = _canvas(24, 24)
    a = math.radians(rot * 360 / ROT_STEPS - 90)
    d.line([12, 12, 12 - math.cos(a) * 10, 12 - math.sin(a) * 10], fill=(60, 60, 60), width=3)
    return _pic(im, 12, 12)


@functools.lru_cache(maxsize=None)
def tack(i, age):
    from bloons import original as O
    reach = O.TACK_REACH[min(age, len(O.TACK_REACH) - 1)]
    a = math.radians([-90, -45, 0, 45, 90, 135, 180, 225][i - 1])
    x, y = 70 + math.cos(a) * reach, 70 + math.sin(a) * reach
    im, d = _canvas(140, 140)
    d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(240, 240, 240))
    return _pic(im, 70, 70)


@functools.lru_cache(maxsize=None)
def bomb(frame, rot, scale):
    from bloons import original as O
    if frame == 0:
        im, d = _canvas(12, 12); d.ellipse([0, 0, 10, 10], fill=(30, 30, 30)); return _pic(im, 5, 5)
    x0, y0, x1, y1 = [v * scale for v in O.BLAST_BOX]
    im, d = _canvas(int(x1 - x0) + 2, int(y1 - y0) + 2)
    d.ellipse([0, 0, x1 - x0, y1 - y0], fill=None, outline=(255, 140, 30), width=3)
    return _pic(im, -x0, -y0)


@functools.lru_cache(maxsize=None)
def ice(frame, scale):
    from bloons import original as O
    x0, y0, x1, y1 = [v * scale for v in O.ICE_BOX]
    im, d = _canvas(int(x1 - x0) + 2, int(y1 - y0) + 2)
    d.rectangle([0, 0, x1 - x0, y1 - y0], fill=None, outline=(170, 230, 255), width=2)
    return _pic(im, -x0, -y0)


def _text_in(d, box, lines, font, colour, align="centre", top=None, step=14):
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


@functools.lru_cache(maxsize=None)
def button_over(kind):
    return np.zeros((1, 1, 4), np.uint8), 0, 0


@functools.lru_cache(maxsize=None)
def start_button(frame=0):
    x0, y0, x1, y1 = R.START_BUTTON
    im, d = _canvas(x1 - x0 + 1, y1 - y0 + 1)
    d.rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=6, fill=(90, 170, 90), outline=(60, 60, 60))
    tw = d.textlength("Start Round", font=_FONT)
    d.text(((x1 - x0) / 2 - tw / 2, (y1 - y0) / 2 - 9), "Start Round", fill=(255, 255, 255), font=_FONT)
    return _pic(im, -x0, -y0)


@functools.lru_cache(maxsize=4096)
def number(field, value):
    label = {"level_txt": "Round", "money_txt": "Money", "lives_txt": "Lives"}[field]
    text = str(value)
    im = Image.new("RGBA", (120, 30), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    tw = d.textlength(text, font=_FONT_BIG)
    d.text((110 - tw, 0), text, fill=TEXT, font=_FONT_BIG)
    return _pic(im, -(R.PANEL[2] - 120), -(R.TEXT_ROWS[label] - 10))


@functools.lru_cache(maxsize=256)
def message(text, frame=20):
    x0, y0, x1, y1 = R.HINT_BOX
    im, d = _canvas(x1 - x0 + 1, y1 - y0 + 1)
    d.rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=8, fill=(255, 255, 255), outline=(60, 60, 60))
    _text_in(d, (0, 0, x1 - x0, 0), _wrap(d, text, _FONT_SMALL, x1 - x0 - 16)[:4], _FONT_SMALL, (20, 20, 20), align="left", top=8, step=16)
    return _pic(im, -x0, -y0)


@functools.lru_cache(maxsize=16)
def towerinfo(kind):
    x0, y0, x1, y1 = R.TOWERINFO_BOX
    im, d = _canvas(x1 - x0 + 1, y1 - y0 + 1)
    d.rounded_rectangle([0, 0, x1 - x0, y1 - y0], radius=6, fill=(235, 245, 235), outline=(60, 90, 60), width=2)
    d.text((8, 6), TOWER_NAMES[kind], fill=TEXT, font=_FONT)
    d.text((8, 28), f"Cost: {R.TOWERS[kind]['cost']}   Speed: {R.TOWERS[kind]['speed']}", fill=TEXT, font=_FONT_SMALL)
    return _pic(im, -x0, -y0)


@functools.lru_cache(maxsize=512)
def options(kind, upgrades, affordable, sell_value, rng, rate):
    x0, y0, x1, y1 = R.PANEL[0] + 4, R.UPGRADE_TITLE_Y - 12, R.PANEL[2] - 4, R.SELL_BUTTON[3] + 2
    im, d = _canvas(x1 - x0, y1 - y0)
    sh = lambda b: (b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0)
    d.rectangle([0, 0, x1 - x0, y1 - y0], fill=PANEL_BG)
    d.text((6, R.UPGRADE_TITLE_Y - 9 - y0), TOWER_NAMES[kind], fill=TEXT, font=_FONT)
    d.text((10, R.UPGRADE_RANGE_Y - 7 - y0), f"Range: {rng}", fill=TEXT, font=_FONT_SMALL)
    for i, (name, cost, _) in enumerate(R.UPGRADES[kind]):
        box = sh(R.UPGRADE_BUTTONS[i]); bought = upgrades[i]
        fill = (110, 110, 110) if bought else ((70, 150, 70) if affordable[i] else (180, 70, 60))
        d.rounded_rectangle(box, radius=5, fill=fill, outline=(60, 60, 60))
        tail = ["Bought"] if bought else (["Buy for:", str(cost)] if affordable[i] else ["Can't Afford", str(cost)])
        _text_in(d, box, name.split() + tail, _FONT_SMALL, (255, 255, 255))
    box = sh(R.SELL_BUTTON)
    d.rounded_rectangle(box, radius=6, fill=(190, 60, 50), outline=(60, 60, 60))
    _text_in(d, box, [f"Sell for: {sell_value}"], _FONT_SMALL, (255, 255, 255), top=box[1] + 7)
    return _pic(im, -x0, -y0)


@functools.lru_cache(maxsize=4)
def banner(won, frame=46):
    im, d = _canvas(241, 61)
    d.rounded_rectangle([0, 0, 240, 60], radius=10, fill=(255, 255, 255), outline=(60, 60, 60))
    text = "You Win!" if won else "Game Over"
    d.text((120 - d.textlength(text, font=_FONT_BIG) / 2, 18), text, fill=(20, 20, 20), font=_FONT_BIG)
    return _pic(im, -200, -200)


@functools.lru_cache(maxsize=None)
def pointer():
    im, d = _canvas(14, 21)
    d.polygon([(0, 0), (12, 10), (5, 11), (8, 18), (5, 19), (2, 12), (0, 16)], fill=(255, 255, 255), outline=(0, 0, 0))
    return _pic(im, 0, 0)
