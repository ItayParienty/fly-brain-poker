"""Every number of Bloons Tower Defense 1 (Ninja Kiwi, 2007), in one place.

Sources, so each value can be checked:
  [C] the original's ActionScript, decompiled from the 2007 SWF (archive.org,
      item bloons_tower_defense_1) with JPEXS: costs, ranges, attack rates,
      pierce, bullet speeds and lifespans, upgrade effects, the round table,
      spawn timing, money, lives, the end-of-round bonus.
  [G] the same SWF's geometry, pulled out by extract_original.py into
      original.py: bloon paths, hit boxes, placement blocks.
  [S] screenshots of the original: the panel layout.
  [W] bloonswiki.com/Bloons_Tower_Defense_(game), where it agrees with [C].

Everything the game logic uses is [C] or [G]; the panel layout is measured.
"""
import math

FPS = 40                     # [C] SWF header; every duration below is in frames
WIDTH, HEIGHT = 640, 480     # [C]

START_MONEY = 650            # [C] STARTING_MONEY
START_LIVES = 40             # [C] MAX_LIVES
ROUNDS = 50                  # [C]
SELL_FRACTION = 0.8          # [C] SELL_RATE, floored
CASH_PER_LAYER = 1           # [C] PoppedOne: money + 1
def round_bonus(round_no):   # [C] EndLevel: 101 - curLevel
    return 101 - round_no


def spawn_interval(round_no):
    """[C] StartLevel: frames between bloons is 15 - round, and once that
    drops under 5 it is ceil(5 - round / 20).  A bloon is released when a
    counter exceeds the interval, so the gap is interval + 1 frames."""
    interval = 15 - round_no
    if interval < 5:
        interval = math.ceil(5 - round_no / 20)
    return interval

# ---------------------------------------------------------------- the track
# Centreline of "the maze": enters on the left, 13 bends, exits at the top.
# Used for drawing and for the bot's coverage estimates; the bloons themselves
# follow the original's frame-by-frame tweens in original.py.  [G]
WAYPOINTS = [(0, 228), (86, 228), (86, 108), (192, 108), (192, 333), (65, 333),
             (65, 420), (417, 420), (417, 308), (310, 308), (310, 186), (423, 186),
             (423, 69), (271, 69), (271, 0)]
PATH_HALF_WIDTH = 24         # the stone path is ~48 px wide
PATH_LENGTH = 1895.0         # sum of segment lengths above

# ---------------------------------------------------------------- bloons  [C]
# rank = the original's number for the colour; a hit takes one layer off
# and the rank below comes out (two yellows out of a black or white).
# RBE = "red bloon equivalent" = layers = cash for popping it all = lives
# lost if it escapes (Escaped: lives -= rank).
RANK = {"Red": 1, "Blue": 2, "Green": 3, "Yellow": 4, "Black": 5, "White": 6}
BLOONS = {
    "Red":    dict(rbe=1, children=[],                   frames=930),
    "Blue":   dict(rbe=2, children=["Red"],              frames=779),
    "Green":  dict(rbe=3, children=["Blue"],             frames=544),
    "Yellow": dict(rbe=4, children=["Green"],            frames=300),
    "Black":  dict(rbe=9, children=["Yellow", "Yellow"], frames=524),
    "White":  dict(rbe=9, children=["Yellow", "Yellow"], frames=407),
}
for _b in BLOONS.values():
    _b["speed"] = PATH_LENGTH / _b["frames"]          # px per frame, for reference
IMMUNE_TO_BOMBS = {"Black"}     # [C] Pop: a bomb on rank 5 does nothing
IMMUNE_TO_FREEZE = {"White"}    # [C] ice freezes rank < 6 only

# ---------------------------------------------------------------- towers  [C]
# cooldown = attackRate: a tower shoots when its counter *exceeds* it, so a
# Dart fires every 30 frames.  range in px (squared distance to the target
# point).  pierce = pierceMax.  proj_speed = shootPower px/frame, proj_life =
# the bullet's lifespan in frames.  scale = bulletScale, the hit box factor.
TOWERS = {
    "Dart":  dict(cost=250,  range=100, cooldown=29,  pierce=1,  proj_speed=20, proj_life=7,  scale=1.0, speed="fast"),
    "Tack":  dict(cost=400,  range=70,  cooldown=55,  pierce=1,  proj_speed=15, proj_life=5,  scale=1.0, speed="medium"),
    "Ice":   dict(cost=850,  range=60,  cooldown=100, pierce=20, proj_speed=0,  proj_life=10, scale=0.8, freeze=50, speed="very slow"),
    "Bomb":  dict(cost=900,  range=120, cooldown=55,  pierce=20, proj_speed=11, proj_life=18, scale=1.0, speed="medium"),
    "Super": dict(cost=4000, range=140, cooldown=2,   pierce=1,  proj_speed=20, proj_life=20, scale=1.0, speed="hypersonic"),
}
TOWER_ORDER = ["Dart", "Tack", "Ice", "Bomb", "Super"]   # left to right in the panel

# (name, cost, effect)   [C] GetUpgrade
UPGRADES = {
    "Dart":  [("Piercing Darts", 210, dict(pierce=+1)), ("Long Range Darts", 100, dict(range=+25))],
    "Tack":  [("Faster Shooting", 250, dict(cooldown=-15)), ("Extra Range Tacks", 150, dict(range=+10, scale=1.3))],
    "Ice":   [("Long Freeze Time", 450, dict(freeze=+20)), ("Wide Freeze Radius", 300, dict(range=+15, scale=1.0))],
    "Bomb":  [("Bigger Bombs", 650, dict(scale=1.5)), ("Extra Range Bombs", 250, dict(range=+20))],
    "Super": [("Epic Range", 2400, dict(range=+100))],
}

# ---------------------------------------------------------------- the screen  [S]
# Measured on the original screenshots (scaled to 640x480).  These matter
# because the fly will look at this layout and click on it.
PANEL_X = 470                                   # the UI covers the right of the map
PANEL = (478, 8, 634, 453)                      # inner light-green box
TEXT_ROWS = {"Round": 22, "Money": 46, "Lives": 70}
BUILD_LABEL_Y = 107
# [G] the buttons' hit areas, measured on the original's hit-state shapes
TOWER_BUTTON_Y = 136.5
TOWER_BUTTON_X = [494.5, 524, 553.5, 583, 612.5]    # Dart, Tack, Ice, Bomb, Super
TOWER_BUTTON_R = 13.75
START_BUTTON = (480, 400, 625, 450)             # "Start Round", shown between rounds
# the upgrade panel replaces "Build Towers" while a tower is selected
UPGRADE_TITLE_Y, UPGRADE_SPEED_Y, UPGRADE_RANGE_Y = 164, 180, 199
UPGRADE_BUTTONS = [(484, 220, 552, 363), (554, 220, 623, 363)]   # [G]
SELL_BUTTON = (484, 367, 622, 391)                                # [G]
HINT_BOX = (30, 393, 440, 470)                  # the between-rounds message
# [G] placements on the main timeline: the tower-info box shown while the
# pointer is over a tower button, and the two buttons under the panel
TOWERINFO_BOX = (482, 159, 630, 300)
RESTART_BUTTON = (571, 459, 629, 479)           # td.Init(): the whole game starts again
MORE_GAMES_BUTTON = (475, 459, 564, 479)        # opens ninjakiwi.com; does nothing here
MAP_RECT = (0, 0, PANEL_X, HEIGHT)

# ---------------------------------------------------------------- rounds  [C] BuildLevels
# (round, [(bloon, count), ...] in release order, the game's own hint text)
ROUND_TABLE = [
    (1, [("Red", 12)], "That was too easy, press 'Start Round' to play the next round."),
    (2, [("Red", 25)], "Still super easy, you didn't miss any did you?"),
    (3, [("Red", 12), ("Blue", 2), ("Red", 12), ("Blue", 3)], "Blue bloons move faster and have red bloons inside them."),
    (4, [("Red", 5), ("Blue", 12), ("Red", 5), ("Blue", 12)], "Remember to spend your money."),
    (5, [("Red", 15), ("Blue", 10), ("Red", 15), ("Blue", 15)], "Green bloons move even faster and have blue bloons inside them!"),
    (6, [("Green", 10), ("Green", 5)], "Lots of blue ones coming up. Hope you're ready..."),
    (7, [("Blue", 75)], "The dart tower piercing upgrade allows darts to pop up to 2 bloons each."),
    (8, [("Red", 20), ("Blue", 30), ("Red", 30), ("Blue", 20), ("Red", 20), ("Blue", 20)], "Ice towers work best with bomb towers nearby."),
    (9, [("Blue", 25), ("Green", 15), ("Blue", 25)], "Are you ready for a whole bunch of greens?"),
    (10, [("Green", 35)], "Yellow bloons are - you guessed it, even bigger and even faster and have greens inside them."),
    (11, [("Yellow", 15)], "Tower upgrades are usually a better option than just adding more towers."),
    (12, [("Blue", 25), ("Green", 25), ("Yellow", 3)], "Tower defense is about what towers you use and where you put them."),
    (13, [("Blue", 40), ("Red", 40), ("Green", 28), ("Blue", 35)], "The super monkey tower is not a joke, he really kicks ass!"),
    (14, [("Yellow", 28)], "You lose one life for every bloon that escapes. So a blue bloon costs you two lives, a green three lives etc."),
    (15, [("Green", 30), ("Blue", 30), ("Green", 30)], "I've tried and you can't pass the game using only tack towers. You can slow the game down a lot though."),
    (16, [("Blue", 20), ("Green", 30), ("Blue", 30), ("Green", 20), ("Blue", 20), ("Green", 25)], "Relax a bit, there are no yellow bloons in the next level."),
    (17, [("Blue", 70), ("Green", 45), ("Blue", 70)], "Have you played [[Bloons (game)|Bloons]]?"),
    (18, [("Blue", 30), ("Yellow", 27), ("Green", 25)], "A whole bunch of greens coming up."),
    (19, [("Green", 90)], "Too easy. Let's step this up a bit."),
    (20, [("Yellow", 16), ("Green", 12), ("Yellow", 15), ("Green", 12), ("Yellow", 17)], "Place your towers so that they can be shooting at something for a long time, corners are good."),
    (21, [("Yellow", 15), ("Blue", 10), ("Yellow", 20), ("Green", 15), ("Green", 70)], "Ready for 45 straight yellows?"),
    (22, [("Yellow", 45)], "Yellows, greens, then more yellows - that should take care of you..."),
    (23, [("Yellow", 30), ("Green", 35), ("Yellow", 34)], "Did you know that the Greek national anthem has 136 verses?"),
    (24, [("Green", 30), ("Yellow", 42), ("Green", 20), ("Blue", 30)], "93% of American teenage girls say shopping is their favourite activity."),
    (25, [("Yellow", 25), ("Green", 30), ("Yellow", 28), ("Green", 40)], "Tack towers are really useful for thinning out the crowds - get the speed upgrade for extra effectiveness."),
    (26, [("Yellow", 85)], "Black bloons are nasty - they are small but contain 2 yellows inside them. Oh did I mention they are IMMUNE TO BOMBS?!"),
    (27, [("Black", 20)], "When you sell a tower, you get 80% of what you paid for it, including all the upgrade money you spent."),
    (28, [("Yellow", 55), ("Green", 45)], "Lots and lots of yellows - more than a hundred even, followed by a bunch of black bloons."),
    (29, [("Yellow", 100), ("Yellow", 25), ("Black", 19)], "Next is a cash round - pop hundreds and hundreds of greens to top up your money. If you leak any I'll wince."),
    (30, [("Green", 250)], "Monkeys aren't so good at shooting at things moving to their left. Something about being right handed I guess."),
    (31, [("Black", 27), ("Green", 55), ("Blue", 10)], "The good thing about black bloons is that they move slower than yellows."),
    (32, [("Yellow", 20), ("Green", 25), ("Black", 23)], "A - lot of yellows."),
    (33, [("Yellow", 150)], "10% of people lose their temper every day. If they play counterstrike its more like 90% I think..."),
    (34, [("Black", 25), ("Green", 35), ("Yellow", 35)], "You will probably need to use every tower type to finish the game."),
    (35, [("Yellow", 25), ("Green", 85), ("Yellow", 85)], "You can improve frame rate a bit by having no towers selected during the round."),
    (36, [("Black", 17), ("Yellow", 115), ("Black", 18)], "Just black bloons coming up. Lots of 'em"),
    (37, [("Black", 59)], "Just around the corner there are a throng of yellow bloons waiting to have a go..."),
    (38, [("Yellow", 220)], "Just for fun, there are some of each colour bloon in the next level. Enjoy popping those easy reds for a change."),
    (39, [("Red", 50), ("Blue", 50), ("Green", 50), ("Yellow", 50), ("Black", 40)], "80 blacks bloons. Enjoy.{{sic"),
    (40, [("Black", 80)], "Its important to not lose early lives, because the levels are only getting harder."),
    (41, [("White", 20), ("Black", 20), ("White", 20)], "White bloons are IMMUNE TO FREEZING - and they also have 2 yellows inside them."),
    (42, [("Yellow", 50), ("Black", 30), ("White", 30)], "How many monkeys does it take to make a super monkey? That stuff will keep you up at night."),
    (43, [("Yellow", 150), ("Black", 60), ("White", 40)], "Lots and lots and lots of black bloons. More than ever."),
    (44, [("Black", 120)], "Lots and lots and lots of WHITE bloons. More than ever."),
    (45, [("White", 120)], "Lots of white, then black, then yellow bloons. The next level is going to hurt."),
    (46, [("White", 60), ("Black", 60), ("Yellow", 59)], "You still playing? I'm impressed, I couldn't get this far without cheating."),
    (47, [("Black", 70), ("Yellow", 79), ("White", 40)], "You got any super monkeys yet? Are they really worth all that money?"),
    (48, [("Yellow", 70), ("White", 80), ("Black", 80)], "Its ok if you don't pass this level. Really it is. Just hit 'try again'. Its the effort that counts."),
    (49, [("White", 70), ("Yellow", 99), ("Black", 80)], "This is the last level. There are TONS AND TONS of black AND white bloons coming. Hope you have lots of lives left..."),
    (50, [("White", 10), ("Black", 10), ("White", 10), ("Black", 10), ("White", 10), ("Black", 10), ("White", 10), ("Black", 10), ("White", 10), ("Black", 10), ("White", 10), ("Black", 10), ("White", 10), ("Black", 10), ("White", 10), ("Black", 10), ("White", 10), ("Black", 10), ("White", 10), ("Black", 9)], ""),
]
assert len(ROUND_TABLE) == ROUNDS and [r[0] for r in ROUND_TABLE] == list(range(1, ROUNDS + 1))


def round_rbe(round_no):
    """Total layers in a round = the cash it pays for popping everything."""
    return sum(BLOONS[b]["rbe"] * n for b, n in ROUND_TABLE[round_no - 1][1])


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    length = sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(WAYPOINTS, WAYPOINTS[1:]))
    print(f"track: {len(WAYPOINTS) - 2} bends, {length:.0f} px")
    for name, b in BLOONS.items():
        print(f"  {name:7s} rbe {b['rbe']}  {b['speed']:.2f} px/frame  -> {b['children'] or '-'}")
    print("round   bloons   cash+bonus")
    for r in (1, 10, 25, 40, 50):
        n = sum(k for _, k in ROUND_TABLE[r - 1][1])
        print(f"  {r:3d}   {n:5d}    {round_rbe(r):5d} + {round_bonus(r)}")
    print("total cash over 50 rounds:", sum(round_rbe(r) + round_bonus(r) for r in range(1, 51)) + START_MONEY)
