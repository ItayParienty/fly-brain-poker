"""A headless, frame-exact clone of Bloons Tower Defense 1.

Mechanics follow the original's ActionScript (decompiled from the 2007 SWF)
and its geometry (bloons/original.py, extracted from the same file):

  * a bloon is a clip playing along the track one frame at a time; its
    position on frame f is the original motion tween's, plus a random
    (0..19, 0..19) px spawn offset, exactly as the original does it
  * a tower shoots when its counter exceeds its attack rate and there is a
    target: the unfrozen bloon (any bloon, for bombs) inside its range whose
    clip has progressed furthest
  * every collision is an axis-aligned box overlap, as Flash's hitTest is
  * a bloon can be hit at most once per frame; a dart loses a pierce per
    hit and vanishes at its pierce limit; a hit on a frozen bloon "clinks"
    and costs the pierce anyway; bombs explode into a box that keeps
    popping for 14 frames; ice is a box around the tower for 10 frames
  * a popped bloon keeps moving for 4 frames of pop animation before it
    pays $1 and releases its children at the same progress
  * bloons enter every (16 - round) frames, down to 3 by round 41

No drawing here: `Game.step()` advances one frame (1/40 s) and the state is
plain objects that a renderer or a bot can read.

    g = Game(seed=1)
    g.place("Dart", 139, 157)
    g.start_round()
    while g.in_round: g.step()
"""
import math
import random

from bloons import rules as R
from bloons import original as O

POP_DELAY = 4                      # frames from a hit to the pop paying out
END_ROUND_WAIT = 21                # frames of empty track before a round ends
DART_TIP = 6.0                     # the original shifts a dart's hit box this far forward
TACK_ANGLES = [-90, -45, 0, 45, 90, 135, 180, 225]     # tack1..tack8, screen degrees


def overlap(a, b):
    """Axis-aligned boxes (x0, y0, x1, y1) intersect."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def shifted(box, x, y, s=1.0):
    return (box[0] * s + x, box[1] * s + y, box[2] * s + x, box[3] * s + y)


# ---------------------------------------------------------------- pieces
class Bloon:
    __slots__ = ("id", "kind", "frame", "ox", "oy", "frozen", "time_frozen", "freezer", "popped", "pop_timer")

    def __init__(self, id, kind, frame, ox, oy):
        self.id, self.kind, self.frame, self.ox, self.oy = id, kind, frame, ox, oy
        self.frozen, self.time_frozen, self.freezer = False, 0, None
        self.popped, self.pop_timer = False, 0

    @property
    def rank(self):
        return R.RANK[self.kind]

    @property
    def total_frames(self):
        return len(O.TWEENS[self.kind])

    @property
    def progress(self):
        return (self.frame + 1) / self.total_frames          # _currentframe / _totalframes

    @property
    def tween(self):
        tx, ty = O.TWEENS[self.kind][min(max(self.frame, 0), self.total_frames - 1)]
        return self.ox + tx, self.oy + ty

    @property
    def target_point(self):
        """The point the original measures range to."""
        x, y = self.tween
        return x - 10, y - 15

    @property
    def box(self):
        x, y = self.tween
        return shifted(O.BLOON_BOX[self.kind], x, y)

    @property
    def pos(self):
        """Centre of the hit box: where to draw it."""
        b = self.box
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


class Tower:
    __slots__ = ("id", "kind", "x", "y", "attack_rate", "range", "pierce_max", "bullet_scale", "freeze_len",
                 "shoot_power", "upgrades", "spent", "since_shot", "angle", "pops")

    def __init__(self, id, kind, x, y):
        st = R.TOWERS[kind]
        self.id, self.kind, self.x, self.y = id, kind, x, y
        self.attack_rate, self.range, self.pierce_max = st["cooldown"], st["range"], st["pierce"]
        self.bullet_scale, self.freeze_len, self.shoot_power = st.get("scale", 1.0), st.get("freeze", 0), st.get("proj_speed", 0)
        self.upgrades = [False] * len(R.UPGRADES[kind])
        self.spent = st["cost"]
        self.since_shot, self.angle, self.pops = 0, 0.0, 0

    @property
    def sell_value(self):
        return int(math.floor(self.spent * R.SELL_FRACTION))

    @property
    def stats(self):                    # for the renderer / bot
        return dict(range=self.range, cooldown=self.attack_rate, pierce=self.pierce_max)

    @property
    def box(self):
        return shifted(O.TOWER_BOX[self.kind], self.x, self.y)


class Bullet:
    __slots__ = ("kind", "x", "y", "vx", "vy", "age", "lifespan", "pierce_max", "pierce_count", "tower", "scale",
                 "hit", "tacks", "checked", "angle")

    def __init__(self, tower, target_point):
        self.kind, self.tower = tower.kind, tower
        self.x, self.y = tower.x, tower.y
        self.vx = self.vy = 0.0
        if target_point is not None:
            dx, dy = target_point[0] - tower.x, target_point[1] - tower.y
            d = math.hypot(dx, dy) or 1.0
            self.vx, self.vy = dx / d * tower.shoot_power, dy / d * tower.shoot_power
            self.angle = math.atan2(dy, dx)
        else:
            self.angle = 0.0
        self.age, self.lifespan = 0, R.TOWERS[tower.kind]["proj_life"]
        self.pierce_max, self.pierce_count = tower.pierce_max, 0
        self.scale = tower.bullet_scale
        self.hit = False                          # a bomb that has gone off
        self.tacks = [True] * 8 if tower.kind == "Tack" else None
        # the original flips a "checked" flag on each tack per bloon it tests,
        # and skips the test when the flag is up - so every tack is only
        # tested against every other bloon.  Kept, because that is the game.
        self.checked = [False] * 8 if tower.kind == "Tack" else None

    def tack_box(self, i):
        reach = O.TACK_REACH[min(self.age, len(O.TACK_REACH) - 1)]
        a = math.radians(TACK_ANGLES[i])
        px, py = self.x + math.cos(a) * reach, self.y + math.sin(a) * reach
        return (px - 6, py - 6, px + 6, py + 6)

    def boxes(self):
        """Hit boxes this frame, as (box, tack_index) pairs; tack_index is None except for tacks."""
        if self.kind == "Tack":
            return [(self.tack_box(i), i) for i, on in enumerate(self.tacks) if on]
        if self.kind == "Ice":
            return [(shifted(O.ICE_BOX, self.x, self.y, self.scale), None)]
        if self.kind == "Bomb":
            if self.hit:
                return [(shifted(O.BLAST_BOX, self.x, self.y, self.scale), None)]
            return [(shifted(O.BOMB_BOX, self.x, self.y), None)]
        # dart / super: a small box at the tip, along the flight direction
        cx = self.x + math.cos(self.angle) * DART_TIP
        cy = self.y + math.sin(self.angle) * DART_TIP
        return [((cx - 5, cy - 5, cx + 5, cy + 5), None)]


# ---------------------------------------------------------------- the game
class Game:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.money, self.lives = R.START_MONEY, R.START_LIVES
        self.round_no = 0                       # rounds completed
        self.in_round = False
        self.frame = 0                          # frames since the game began
        self.bloons, self.towers, self.bullets = [], [], []
        self.hint = "Press Start Round to begin."
        self.selected = None                    # a Tower, for the upgrade panel
        self._next_id = 0
        self.stats = dict(pops=0, leaks=0)
        # round state
        self._queue, self._counter, self._interval, self._no_more, self._end_wait = [], 0, 0, True, 0

    # ------------------------------------------------ status
    @property
    def over(self):
        return self.lives <= 0

    @property
    def won(self):
        return self.round_no >= R.ROUNDS and not self.in_round and not self.over

    @property
    def current_round(self):
        return self.round_no + 1 if self.in_round else self.round_no

    # ------------------------------------------------ building
    def can_place(self, kind, x, y):
        box = shifted(O.TOWER_BOX[kind], x, y)
        if any(overlap(box, blk) for blk in O.PATH_BLOCKS):
            return False
        return not any(overlap(box, t.box) for t in self.towers)

    def place(self, kind, x, y):
        cost = R.TOWERS[kind]["cost"]
        if self.money < cost or not self.can_place(kind, x, y):
            return None
        self.money -= cost
        t = Tower(self._new_id(), kind, x, y); self.towers.append(t)
        self.selected = t
        return t

    def upgrade(self, tower, which):
        name, cost, effect = R.UPGRADES[tower.kind][which]
        if tower.upgrades[which] or self.money < cost:
            return False
        self.money -= cost; tower.spent += cost; tower.upgrades[which] = True
        for key, val in effect.items():
            if key == "scale": tower.bullet_scale = val
            elif key == "range": tower.range += val
            elif key == "pierce": tower.pierce_max += val
            elif key == "cooldown": tower.attack_rate += val
            elif key == "freeze": tower.freeze_len += val
        return True

    def sell(self, tower):
        if tower not in self.towers:
            return False
        self.money += tower.sell_value; self.towers.remove(tower)
        if self.selected is tower: self.selected = None
        return True

    # ------------------------------------------------ rounds
    def start_round(self):
        if self.in_round or self.over or self.round_no >= R.ROUNDS:
            return False
        level = self.round_no + 1
        self._queue = [kind for kind, n in R.ROUND_TABLE[self.round_no][1] for _ in range(n)]
        self._interval = R.spawn_interval(level)
        self._counter, self._no_more, self._end_wait = 0, False, 0
        self.in_round = True
        return True

    def _finish_round(self):
        self.in_round = False
        self.bullets.clear(); self.bloons.clear()
        self.round_no += 1
        if self.round_no < R.ROUNDS:
            bonus = R.round_bonus(self.round_no)
            self.money += bonus
            hint = R.ROUND_TABLE[self.round_no][2]        # the hint shown after round n is the next round's
            self.hint = f"Round {self.round_no} passed. {bonus} money awarded. {hint}".strip()

    def _spawn(self, kind, frame=0, ox=None, oy=None):
        if ox is None:
            ox, oy = -10 + self.rng.randrange(20), 200 + self.rng.randrange(20)
        b = Bloon(self._new_id(), kind, frame, ox, oy); self.bloons.append(b)
        return b

    # ------------------------------------------------ one frame
    def step(self):
        """Advance one frame. Returns (pops, leaks) that happened in it."""
        if self.over or not self.in_round:
            return 0, 0
        pops = leaks = 0
        self.frame += 1

        # 1. clips advance: bloons move (unless frozen), pop animations run out, escapes
        for b in list(self.bloons):
            if not b.frozen:
                b.frame += 1
                if b.frame >= O.EXIT_FRAME[b.kind] - 1:
                    self.bloons.remove(b)
                    self.lives -= b.rank; leaks += b.rank
                    continue
            if b.popped:
                b.pop_timer += 1
                if b.pop_timer >= POP_DELAY:
                    pops += self._remove_popped(b)
        if self.lives <= 0:
            self.in_round = False; self.stats["leaks"] += leaks
            return pops, leaks

        # 2. every bloon looks for a bullet touching it (one hit per bloon per frame)
        for b in self.bloons:
            if b.popped:
                continue
            if b.frozen:
                b.time_frozen += 1
                if b.time_frozen > b.freezer.freeze_len or b.time_frozen > 100:
                    b.frozen = False
            box = b.box
            for p in list(self.bullets):
                if p.kind == "Tack":
                    struck = None
                    for i in range(8):
                        if p.tacks[i] and not p.checked[i]:
                            p.checked[i] = True
                            if overlap(box, p.tack_box(i)):
                                struck = i; break
                        elif p.checked[i]:
                            p.checked[i] = False
                    if struck is None:
                        continue
                    p.pierce_count += 1
                    if p.pierce_count >= p.pierce_max:
                        p.tacks[struck] = False
                    if not b.frozen:
                        pops += self._pop(b, p.tower, blew_up=False)
                    else:
                        p.pierce_count += 5
                    break
                if not any(overlap(box, hb) for hb, _ in p.boxes()):
                    continue
                p.pierce_count += 1
                if p.pierce_count >= p.pierce_max:
                    self.bullets.remove(p)
                if p.kind == "Bomb" and not p.hit:
                    p.hit = True; p.vx = p.vy = 0.0
                if p.kind == "Ice":
                    if not b.frozen and b.kind not in R.IMMUNE_TO_FREEZE:
                        b.frozen, b.time_frozen, b.freezer = True, 0, p.tower
                elif not b.frozen or p.kind == "Bomb":
                    pops += self._pop(b, p.tower, blew_up=(p.kind == "Bomb"))
                break

        # 3. release the next bloon
        if not self._no_more:
            self._counter += 1
            if self._counter > self._interval:
                self._counter = 0
                self._spawn(self._queue.pop(0))
                if not self._queue:
                    self._no_more = True
        elif not self.bloons:
            self._end_wait += 1
            if self._end_wait >= END_ROUND_WAIT:
                self._finish_round()
                self.stats["pops"] += pops; self.stats["leaks"] += leaks
                return pops, leaks

        # 4. towers
        for t in self.towers:
            t.since_shot += 1
            if t.since_shot > t.attack_rate:
                target = self._target(t)
                if target is not None:
                    t.since_shot = 0
                    aim = None if t.kind in ("Tack", "Ice") else target.target_point
                    if aim is not None:
                        t.angle = math.atan2(aim[1] - t.y, aim[0] - t.x)
                    self.bullets.append(Bullet(t, aim))

        # 5. bullets age and fly
        for p in list(self.bullets):
            p.age += 1
            if p.age > p.lifespan:
                self.bullets.remove(p); continue
            p.x += p.vx; p.y += p.vy

        self.stats["pops"] += pops; self.stats["leaks"] += leaks
        return pops, leaks

    def _target(self, t):
        best, best_progress, r2 = None, 0.0, t.range * t.range
        for b in self.bloons:                   # in spawn order, like the original's bloon0..N
            if b.popped or (b.frozen and t.kind != "Bomb"):
                continue
            px, py = b.target_point
            if (px - t.x) ** 2 + (py - t.y) ** 2 < r2 and b.progress > best_progress:
                best, best_progress = b, b.progress
        return best

    # ------------------------------------------------ popping
    def _pop(self, b, tower, blew_up):
        """A hit lands on an unpopped bloon (or a bomb on a frozen one)."""
        if blew_up:
            if b.kind in R.IMMUNE_TO_BOMBS:
                return 0
            b.popped = True
            if tower is not None: tower.pops += 1
            return self._remove_popped(b)          # bombs skip the animation
        b.popped = True; b.pop_timer = 0
        if tower is not None: tower.pops += 1
        return 0

    def _remove_popped(self, b):
        """End of the pop: pay $1 and release the children at the same progress."""
        if b not in self.bloons:
            return 0
        self.bloons.remove(b)
        self.money += R.CASH_PER_LAYER
        children = R.BLOONS[b.kind]["children"]
        if len(children) == 1:
            self._spawn(children[0], round(b.progress * len(O.TWEENS[children[0]])) - 1, b.ox, b.oy)
        elif len(children) == 2:
            n = len(O.TWEENS[children[0]])
            self._spawn(children[0], round((b.frame + 1 + 5) / b.total_frames * n) - 1, b.ox, b.oy)
            self._spawn(children[1], round((b.frame + 1 - 5) / b.total_frames * n) - 1, -10 + self.rng.randrange(20), 200 + self.rng.randrange(20))
        return 1

    def _new_id(self):
        self._next_id += 1
        return self._next_id


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    import time
    # one dart tower where we tested the original: it leaked 1 bloon in round 1 and 8 in round 2
    for seed in range(3):
        g = Game(seed=seed)
        assert g.place("Dart", 139, 157)
        out = []
        for _ in range(2):
            lives = g.lives; g.start_round()
            while g.in_round: g.step()
            out.append(lives - g.lives)
        print(f"seed {seed}: leaks per round {out}   (original: [1, 8])")
    g = Game(); g.place("Dart", 139, 157); g.place("Dart", 150, 290)
    t0 = time.perf_counter(); frames = 0
    for _ in range(3):
        g.start_round()
        while g.in_round: g.step(); frames += 1
        print(f"round {g.round_no}: money {g.money} lives {g.lives}  |  {g.hint[:60]}")
    dt = time.perf_counter() - t0
    print(f"{frames} frames in {dt:.2f}s = {frames/dt:.0f} frames/s ({frames/dt/R.FPS:.0f}x real time)")
