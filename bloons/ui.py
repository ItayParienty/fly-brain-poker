"""The mouse, handled the way the original game handles it.

Everything a player does in BTD1 is a mouse press somewhere (BloonsTD.as,
Tower.as and the buttons' own scripts, decompiled).  A person, a bot and
the fly all play through this one class: move the pointer, press.  What a
press does depends on what is under the pointer, top to bottom:

  the panel      a tower button   pick that tower up if the money is there,
                                  else the message box says "not enough money."
                 Start Round      only between rounds
                 upgrade 1 / 2    only with a tower selected; a "Can't Afford"
                                  or "Bought" cover over the button swallows
                                  the press
                 Sell             only with a tower selected
                 Restart          the whole game starts again
  a tower        selects it (and drops a tower being held)
  the map        holding a tower: put it down here, unless its range ring is
                 red (on the track, or on another tower); else deselect

While a tower is held it follows the pointer with its range ring.  With
the pointer over a tower button, the tower-info box shows.

    mouse = Mouse(game)
    mouse.move(495, 138); mouse.press()          # pick up a Dart tower
    mouse.move(139, 157); mouse.press()          # put it down
"""
from bloons import rules as R


def inside(box, x, y):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


class Mouse:
    def __init__(self, game):
        self.game = game
        self.x, self.y = R.PANEL_X // 2, R.HEIGHT // 2
        self.tool = None                    # the tower kind being held (currentTool)
        self.presses = 0

    def move(self, x, y):
        """Pointer positions are whole pixels, as Flash's _xmouse is."""
        self.x = int(round(min(max(x, 0), R.WIDTH - 1)))
        self.y = int(round(min(max(y, 0), R.HEIGHT - 1)))

    # ------------------------------------------------ what is under the pointer
    @property
    def hover(self):
        """The tower button under the pointer, if any: its info box is showing."""
        for kind, bx in zip(R.TOWER_ORDER, R.TOWER_BUTTON_X):
            if (self.x - bx) ** 2 + (self.y - R.TOWER_BUTTON_Y) ** 2 <= R.TOWER_BUTTON_R ** 2:
                return kind
        return None

    @property
    def placeable(self):
        """The held tower's ring is not red here."""
        return self.tool is not None and self.game.can_place(self.tool, self.x, self.y)

    def tower_at(self, x, y):
        for t in reversed(self.game.towers):          # the newest is on top
            b = t.box
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                return t
        return None

    def _deselect(self):
        self.tool = None
        self.game.selected = None

    # ------------------------------------------------ a press
    def press(self):
        """Press and release where the pointer is. Returns what happened, for logs."""
        g, x, y = self.game, self.x, self.y
        self.presses += 1
        if x >= R.PANEL_X:
            kind = self.hover
            if kind is not None:                                  # SetCurrentTool
                if R.TOWERS[kind]["cost"] > g.money:
                    g.output("not enough money.")
                    return "broke"
                self._deselect(); self.tool = kind
                return "pick " + kind
            if inside(R.START_BUTTON, x, y) and not g.in_round and not g.over and not g.won:
                g.start_round()
                return "start"
            t = g.selected
            if t is not None and t in g.towers:
                for i, box in enumerate(R.UPGRADE_BUTTONS[:len(R.UPGRADES[t.kind])]):
                    if inside(box, x, y):
                        if t.upgrades[i] or g.money < R.UPGRADES[t.kind][i][1]:
                            return "blocked"                      # the cover takes the press
                        g.upgrade(t, i)
                        return f"upgrade {t.kind} {i + 1}"
                if inside(R.SELL_BUTTON, x, y):
                    g.sell(t); self._deselect()
                    return "sell"
            if inside(R.RESTART_BUTTON, x, y):
                g.reset(); self.tool = None
                return "restart"
            return "panel"
        t = self.tower_at(x, y)
        if t is not None:                                         # Tower.Press -> SelectTower
            self.tool = None; g.selected = t
            return "select"
        if self.tool is not None:                                 # OnClick
            if not self.placeable:
                return "red ring"
            if R.TOWERS[self.tool]["cost"] > g.money:
                g.output("not enough money.")
                return "broke"
            kind = self.tool
            g.place(kind, x, y); self.tool = None                 # CreateNewTower selects it
            return "place " + kind
        self._deselect()
        return "deselect"


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    from bloons.game import Game
    g = Game(); m = Mouse(g)
    for (x, y) in [(495, 138), (139, 157), (553, 138), (613, 138), (495, 138), (235, 200), (139, 157),
                   (518, 270), (518, 270), (555, 420)]:
        m.move(x, y); print(f"press at ({x:3d},{y:3d}) -> {m.press():14s} money {g.money:4d}  holding {m.tool}  message {g.message!r}")
