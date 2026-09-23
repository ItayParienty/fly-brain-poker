"""How things look with the original game's art (bloons/art.py over data/btd1_art/).

Each function returns a picture - straight RGBA, uint8, at the game's
resolution - and the point in it that is the object's origin.  Pictures
are made on first use and kept.  The main timeline decides what is where:
the map and the panel's backing lie under the bloons, darts and towers,
the panel's buttons and labels over them, as in the original's depths.
"""
import functools
import math

import numpy as np

from bloons import rules as R
from bloons.art import IDENTITY, Library, compose, rotation, shrink, trim
from bloons.look_drawn import pointer  # noqa: F401  (the system's arrow, not part of the game)

LIB = Library()
MAIN = LIB.label(0, "main")
TIMELINE = LIB.display_list(0, MAIN)
DEPTH = {p["name"]: d for d, p in TIMELINE.items() if p.get("name")}
TOWER_CHAR = {"Dart": "towerdart", "Tack": "towertack", "Ice": "towerice", "Bomb": "towerbomb", "Super": "towersuper"}
TOOL = {"Dart": "dart", "Tack": "tack", "Ice": "ice", "Bomb": "bomb", "Super": "super"}
BLOON_CLIP = {"Red": "plain1", "Blue": "plain2", "Green": "plain3", "Yellow": "plain4", "Black": "plain5", "White": "plain6"}
RANGE = {"Dart": 100, "Tack": 70, "Ice": 60, "Bomb": 120, "Super": 140}      # BloonsTD.RANGE_*: the rings never grow
NOT_THE_GAME = (510, 537, 548)            # the title menu and intro (hidden once a game starts), and the stage mask
BUTTON_DEPTHS = {"Dart": 348, "Ice": 353, "Tack": 356, "Bomb": 387, "Super": 461}
ROT_STEPS = 64                             # a tower's or bullet's heading is drawn in 64 steps
POSES = 50                                 # a bloon stops on one of the first 50 frames of its clip (Bloon: random(50))
ARM = {"Dart": ("towerdart", 82), "Super": ("towersuper", 74), "Bomb": ("towerbomb", 54)}
START_FRAMES = 38                          # the Start Round button's glow loops over 38 frames
MESSAGE_FRAMES = 234                       # the message box: fades in, waits, fades out (output's timeline)
BANNER_FRAMES = 47
SPEED = lambda rate: ("hypersonic" if rate < 4 else "very fast" if rate < 24 else "fast" if rate < 40 else
                      "medium" if rate < 60 else "slow" if rate < 100 else "very slow")   # Tower.GetSpeedRating
INFO = {                                   # BloonsTD.ShowTowerInfo
    "Dart": ("Dart Tower", "Fast", "Shoots a single dart. Can upgrade to piercing darts and long range darts"),
    "Tack": ("Tack Tower", "Medium", "Shoots volley of tacks in 8 directions. Can upgrade its shoot speed and its range."),
    "Bomb": ("Bomb Tower", "Medium", "Launches a bomb that explodes on impact. Can upgrade to bigger bombs and longer range."),
    "Ice": ("Ice Tower", "Slow", "Freezes nearby bloons. Frozen bloons are immune to darts and tacks, but bombs will destroy them. "
                                 "Can upgrade to increased freeze time, and larger freeze radius."),
    "Super": ("Super Monkey", "Hypersonic", "Super monkey shoots a continuous stream of darts and can mow down even the fastest and most stubborn bloons."),
}


def char(name):
    return LIB.names[name]


def _stage(depths, set=None):
    """Part of the main timeline, as a full-screen picture."""
    z = LIB.zoom
    canvas = np.zeros((R.HEIGHT * z, R.WIDTH * z, 4), np.float32)
    keep = {d for d in TIMELINE if depths[0] <= d <= depths[1] and d not in NOT_THE_GAME}
    for d in sorted(keep):
        p = TIMELINE[d]
        name = p.get("name", "")
        o = (set or {}).get(name, {}) if name else {}
        if "char" not in p or "mask" in p or name.startswith("pathhit") or o.get("visible") is False:
            continue
        m = compose((z, 0, 0, z, 0, 0), (p["matrix"]["sx"], p["matrix"]["r0"], p["matrix"]["r1"], p["matrix"]["sy"],
                                        p["matrix"]["tx"], p["matrix"]["ty"]))
        LIB.draw(canvas, p["char"], o.get("frame", 0), m, (tuple(p["cx"]),) if "cx" in p else (),
                 {k: v for k, v in o.items() if isinstance(v, dict)}, o.get("state", "up"), o.get("text"))
    return shrink(canvas, z)


@functools.lru_cache(maxsize=None)
def background():
    """The map and the panel's backing: everything under the bloons."""
    return _stage((0, DEPTH["bloonholder"] - 1))[:, :, :3]


@functools.lru_cache(maxsize=None)
def panel_front():
    """The panel's buttons, labels, the logo and the buttons under the panel."""
    dynamic = ("toweroptions", "startrnd_btn", "losepanel", "winpanel", "output", "level_txt", "money_txt", "lives_txt", "towerinfo")
    return _stage((DEPTH["towerplace"] + 1, 600), {n: dict(visible=False) for n in dynamic}), 0, 0


def _placed(name_or_depth, set=None, frame=0, state="up", text=None):
    """One main-timeline item, drawn where the timeline puts it, trimmed."""
    d = DEPTH[name_or_depth] if isinstance(name_or_depth, str) else name_or_depth
    p = TIMELINE[d]
    z = LIB.zoom
    canvas = np.zeros((R.HEIGHT * z, R.WIDTH * z, 4), np.float32)
    m = compose((z, 0, 0, z, 0, 0), (p["matrix"]["sx"], p["matrix"]["r0"], p["matrix"]["r1"], p["matrix"]["sy"], p["matrix"]["tx"], p["matrix"]["ty"]))
    LIB.draw(canvas, p["char"], frame, m, (tuple(p["cx"]),) if "cx" in p else (), set, state, text)
    return trim(shrink(canvas, z), 0, 0)


@functools.lru_cache(maxsize=None)
def button_over(kind):
    return _placed(BUTTON_DEPTHS[kind], state="over")


@functools.lru_cache(maxsize=None)
def start_button(frame=0):
    return _placed("startrnd_btn", frame=frame % START_FRAMES)


@functools.lru_cache(maxsize=4096)
def number(field, value):
    return _placed(field, text=str(value))


@functools.lru_cache(maxsize=128)
def message(text, frame=20):
    """The message box (output) on a frame of its timeline: 1-14 fade in, 221-233 fade out."""
    return _placed("output", frame=frame, set={"inner": {"output_txt": dict(text=text)}})


@functools.lru_cache(maxsize=16)
def towerinfo(kind):
    name, speed, info = INFO[kind]
    return _placed("towerinfo", set=dict(towername_txt=dict(text=name), towercost_txt=dict(text=str(R.TOWERS[kind]["cost"])),
                                         towerspeed_txt=dict(text=speed), towerinfo_txt=dict(text=info)))


@functools.lru_cache(maxsize=64)
def options(kind, upgrades, affordable, sell_value, rng, rate):
    """The upgrade panel for a selected tower (toweroptions and its Refresh())."""
    t = TOOL[kind]
    ups = {}
    for i, name in enumerate(("upgrade1", "upgrade2")):
        if i >= len(R.UPGRADES[kind]):
            ups[name] = dict(visible=False); continue
        ups[name] = dict(frame=LIB.label(262, f"{t}{i + 1}"), cost_txt=dict(text=str(R.UPGRADES[kind][i][1])),
                         cantafford=dict(visible=not affordable[i]), hasbought=dict(visible=upgrades[i]))
    return _placed("toweroptions", set=dict(ups, towername_txt=dict(text=INFO[kind][0]), towerspeed_txt=dict(text=SPEED(rate)),
                                            towerrange_txt=dict(text=str(rng)), sellfor_txt=dict(text=str(sell_value))))


@functools.lru_cache(maxsize=None)
def banner(won, frame=46):
    return _placed("winpanel" if won else "losepanel", frame=min(frame, BANNER_FRAMES - 1))


# ---------------------------------------------------------------- things on the map
def _object(sid, frame=0, m=IDENTITY, set=None, box=(-90, -90, 90, 90)):
    img, ox, oy = LIB.picture(sid, frame, m, set, box=box)
    return trim(img, ox, oy)


ARM_FRAMES = {k: len(LIB.timelines[a]) if k in ARM else 1 for k, a in [(k, ARM.get(k, (0, 0))[1]) for k in TOWER_CHAR]}


@functools.lru_cache(maxsize=None)
def tower(kind, rot, arm=0):
    """rot: the tower's _rotation in steps of 360/64 (0 = as drawn, facing up);
    arm: the frame of its throwing arm (it swings after every shot)."""
    return _object(char(TOWER_CHAR[kind]), 0, rotation(rot * 360.0 / ROT_STEPS), dict(inner=dict(arm=dict(frame=arm))))


@functools.lru_cache(maxsize=None)
def ring(kind, red=False):
    """The range disc: a selected tower's (white) or the one under a held tower (white, or red
    where it cannot go) - radiusmc, at the 40% opacity its parents give it."""
    scale = 2 * RANGE[kind] / 100.0                     # radiusmc._width = 2 * range; the disc is 100 px
    r = RANGE[kind] + 4
    z = LIB.zoom
    canvas = np.zeros((2 * r * z, 2 * r * z, 4), np.float32)
    LIB.draw(canvas, 49, 1 if red else 0, (scale * z, 0, 0, scale * z, r * z, r * z), ((1.0, 1.0, 1.0, 0.4, 0, 0, 0, 0),),
             dict(cantplace=dict(visible=red)))
    return trim(shrink(canvas, z), r, r)


@functools.lru_cache(maxsize=None)
def ghost(kind):
    """The tower held under the pointer (towerplace's 'tower' child for this tool), without its disc."""
    frame = LIB.label(215, TOOL[kind])
    return _object(215, frame, set=dict(radiusmc=dict(visible=False)))


@functools.lru_cache(maxsize=None)
def bloon(kind, frame, freeze=0):
    """A bloon's inner clip on a frame: 0-49 are the poses a bloon keeps (random(50)),
    `pop` + n the pop.  freeze: the ice's frame (0 = none, 29 = all the way in)."""
    sid = LIB.display_list(char(BLOON_CLIP[kind]), 0)[min(LIB.display_list(char(BLOON_CLIP[kind]), 0))]["char"]
    return _object(sid, frame, set=dict(inner=dict(freeze=dict(frame=freeze))), box=(-60, -80, 60, 60))


def pop_frame(kind):
    sid = LIB.display_list(char(BLOON_CLIP[kind]), 0)[min(LIB.display_list(char(BLOON_CLIP[kind]), 0))]["char"]
    return LIB.label(sid, "pop")


@functools.lru_cache(maxsize=None)
def dart(kind, rot):
    sid = char("super" if kind == "Super" else "dart")
    return _object(sid, 0, rotation(rot * 360.0 / ROT_STEPS), dict(inner=dict(dx=8)), box=(-40, -40, 40, 40))


@functools.lru_cache(maxsize=None)
def tack(i, age):
    """Tack i (1-8) of a volley, `age` frames after it was fired."""
    only = {f"tack{j}": dict(visible=(j == i), frame=age) for j in range(1, 9)}
    return _object(char("tack"), 0, IDENTITY, only, box=(-80, -80, 80, 80))


@functools.lru_cache(maxsize=None)
def bomb(frame, rot, scale):
    return _object(char("bomb"), frame, compose((scale, 0, 0, scale, 0, 0), rotation(rot * 360.0 / ROT_STEPS)), box=(-100, -100, 100, 100))


@functools.lru_cache(maxsize=None)
def ice(frame, scale):
    return _object(char("ice"), frame, (scale, 0, 0, scale, 0, 0), box=(-110, -110, 110, 110))
