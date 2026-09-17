"""Draws a Game as the 640x480 screen the fly will look at.

Our own drawing, laid out like the original (rules.py has the measured
geometry): the map on the left, the panel on the right with Round / Money /
Lives, the five tower buttons, Start Round, the upgrade panel for a selected
tower, and the hint box between rounds.

    frame = render(game)          # PIL.Image, RGB, 640x480
"""
import math

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

try:
    _FONT = ImageFont.truetype("arial.ttf", 17)
    _FONT_SMALL = ImageFont.truetype("arial.ttf", 12)
    _FONT_BIG = ImageFont.truetype("arial.ttf", 22)
except OSError:                                   # no Arial: PIL's built-in font
    _FONT = _FONT_SMALL = _FONT_BIG = ImageFont.load_default()


def _background():
    """The map: grass with the stone track along the waypoints. Drawn once."""
    im = Image.new("RGB", (R.WIDTH, R.HEIGHT), GRASS)
    d = ImageDraw.Draw(im)
    # a little texture so the grass is not one flat value
    for y in range(0, R.HEIGHT, 8):
        for x in range(0, R.PANEL_X, 8):
            if (x * 7 + y * 13) % 5 == 0:
                d.rectangle([x, y, x + 3, y + 3], fill=GRASS_DARK)
    w = R.PATH_HALF_WIDTH
    pts = R.WAYPOINTS
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):        # stone segments, edged
        d.rectangle([min(x0, x1) - w - 2, min(y0, y1) - w - 2, max(x0, x1) + w + 2, max(y0, y1) + w + 2], fill=STONE_EDGE)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        d.rectangle([min(x0, x1) - w, min(y0, y1) - w, max(x0, x1) + w, max(y0, y1) + w], fill=STONE)
    # tile seams every 40 px of track
    s = 0.0
    while s < R.PATH_LENGTH:
        x, y = point_at(s)
        d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=STONE_EDGE)
        s += 40
    return im


_BG = _background()


def _button(d, box, text, fill, font=_FONT):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=6, fill=fill, outline=(60, 60, 60))
    tw = d.textlength(text, font=font)
    d.text(((x0 + x1) / 2 - tw / 2, (y0 + y1) / 2 - 9), text, fill=(255, 255, 255), font=font)


def _tower(d, t, selected=False):
    c = TOWER_COLOURS[t.kind]
    if selected:
        r = t.range
        d.ellipse([t.x - r, t.y - r, t.x + r, t.y + r], fill=None, outline=(255, 255, 255), width=2)
    if t.kind == "Tack":
        d.regular_polygon((t.x, t.y, 13), 8, fill=c, outline=(120, 60, 90))
        d.ellipse([t.x - 5, t.y - 5, t.x + 5, t.y + 5], fill=(255, 255, 255))
    elif t.kind == "Ice":
        d.regular_polygon((t.x, t.y, 13), 6, fill=c, outline=(80, 140, 200))
    elif t.kind == "Bomb":
        d.ellipse([t.x - 12, t.y - 12, t.x + 12, t.y + 12], fill=c, outline=(90, 90, 90))
        d.line([t.x, t.y, t.x + 14 * math.cos(t.angle), t.y + 14 * math.sin(t.angle)], fill=(90, 90, 90), width=5)
    else:                                                # a monkey: body, head, facing its target
        d.ellipse([t.x - 12, t.y - 10, t.x + 12, t.y + 10], fill=c, outline=(80, 50, 20))
        hx, hy = t.x + 9 * math.cos(t.angle), t.y + 9 * math.sin(t.angle)
        d.ellipse([hx - 7, hy - 7, hx + 7, hy + 7], fill=(230, 190, 120) if t.kind == "Dart" else (40, 60, 200))
    for i, up in enumerate(t.upgrades):                  # little pips for bought upgrades
        if up: d.ellipse([t.x - 12 + i * 8, t.y + 11, t.x - 7 + i * 8, t.y + 16], fill=(255, 230, 60))


def _bloon(d, b, frame):
    x, y = b.pos
    c = BLOON_COLOURS[b.kind]
    if b.frozen: c = tuple(int(v * 0.5 + 120) for v in c)
    if b.popped: c = tuple(int(v * 0.6 + 100) for v in c)
    d.ellipse([x - 10, y - 13, x + 10, y + 11], fill=c, outline=(20, 20, 20))
    d.polygon([(x, y + 11), (x - 3, y + 15), (x + 3, y + 15)], fill=c)
    d.ellipse([x - 6, y - 9, x - 2, y - 4], fill=tuple(min(255, v + 90) for v in c))


def render(game, cursor=None):
    im = _BG.copy()
    d = ImageDraw.Draw(im)
    for t in game.towers:
        _tower(d, t, selected=(t is game.selected))
    for b in game.bloons:
        _bloon(d, b, game.frame)
    for p in game.bullets:
        for hb, _ in p.boxes():
            if p.kind == "Bomb" and p.hit:
                d.ellipse(hb, fill=None, outline=(255, 140, 30), width=3)
            elif p.kind == "Bomb":
                d.ellipse([p.x - 5, p.y - 5, p.x + 5, p.y + 5], fill=(30, 30, 30))
            elif p.kind == "Ice":
                d.rectangle(hb, fill=None, outline=(170, 230, 255), width=2)
            elif p.kind == "Tack":
                cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
                d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(240, 240, 240))
            else:
                d.line([p.x, p.y, p.x - math.cos(p.angle) * 10, p.y - math.sin(p.angle) * 10], fill=(60, 60, 60), width=3)

    # ---- the panel
    d.rectangle([R.PANEL_X, 0, R.WIDTH, R.HEIGHT], fill=(60, 60, 60))
    d.rounded_rectangle(R.PANEL, radius=8, fill=PANEL_BG, outline=PANEL_EDGE, width=3)
    for label, y in R.TEXT_ROWS.items():
        value = {"Round": game.current_round, "Money": game.money, "Lives": game.lives}[label]
        d.text((R.PANEL[0] + 10, y - 10), f"{label}:", fill=TEXT, font=_FONT_BIG)
        vw = d.textlength(str(value), font=_FONT_BIG)
        d.text((R.PANEL[2] - 10 - vw, y - 10), str(value), fill=TEXT, font=_FONT_BIG)
    d.text((R.PANEL[0] + 10, R.BUILD_LABEL_Y - 10), "Build Towers", fill=TEXT, font=_FONT_BIG)
    d.line([R.PANEL[0] + 10, R.BUILD_LABEL_Y + 11, R.PANEL[2] - 10, R.BUILD_LABEL_Y + 11], fill=TEXT, width=2)
    for kind, x in zip(R.TOWER_ORDER, R.TOWER_BUTTON_X):
        r = R.TOWER_BUTTON_R
        affordable = game.money >= R.TOWERS[kind]["cost"]
        d.ellipse([x - r, R.TOWER_BUTTON_Y - r, x + r, R.TOWER_BUTTON_Y + r],
                  fill=TOWER_COLOURS[kind] if affordable else (120, 120, 120), outline=(40, 40, 40), width=2)

    t = game.selected
    if t is not None and t in game.towers:
        d.text((R.PANEL[0] + 10, R.UPGRADE_TITLE_Y - 9), f"{t.kind} Tower", fill=TEXT, font=_FONT)
        d.text((R.PANEL[0] + 14, R.UPGRADE_SPEED_Y - 7), "Speed:", fill=TEXT, font=_FONT_SMALL)
        d.text((R.PANEL[0] + 90, R.UPGRADE_SPEED_Y - 7), R.TOWERS[t.kind].get("speed", ""), fill=TEXT, font=_FONT_SMALL)
        d.text((R.PANEL[0] + 14, R.UPGRADE_RANGE_Y - 7), "Range:", fill=TEXT, font=_FONT_SMALL)
        d.text((R.PANEL[0] + 90, R.UPGRADE_RANGE_Y - 7), str(t.range), fill=TEXT, font=_FONT_SMALL)
        for i, (name, cost, _) in enumerate(R.UPGRADES[t.kind]):
            box = R.UPGRADE_BUTTONS[i]
            bought = t.upgrades[i]
            fill = (110, 110, 110) if bought else ((70, 150, 70) if game.money >= cost else (180, 70, 60))
            d.rounded_rectangle(box, radius=5, fill=fill, outline=(60, 60, 60))
            y = box[1] + 8
            for word in name.split():
                tw = d.textlength(word, font=_FONT_SMALL)
                d.text(((box[0] + box[2]) / 2 - tw / 2, y), word, fill=(255, 255, 255), font=_FONT_SMALL); y += 14
            tail = "Bought" if bought else ("Buy for:" if game.money >= cost else "Can't Afford")
            for line in (tail, "" if bought else str(cost)):
                tw = d.textlength(line, font=_FONT_SMALL)
                d.text(((box[0] + box[2]) / 2 - tw / 2, y + 6), line, fill=(255, 255, 255), font=_FONT_SMALL); y += 14
        _button(d, R.SELL_BUTTON, f"Sell for: {t.sell_value}", (190, 60, 50), _FONT_SMALL)

    if not game.in_round and not game.over and not game.won:
        _button(d, R.START_BUTTON, "Start Round", (90, 170, 90))
        # the hint box
        d.rounded_rectangle(R.HINT_BOX, radius=8, fill=(255, 255, 255), outline=(60, 60, 60))
        words, lines, cur = game.hint.split(), [], ""
        for w in words:
            if d.textlength(cur + " " + w, font=_FONT_SMALL) > R.HINT_BOX[2] - R.HINT_BOX[0] - 16:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for i, line in enumerate(lines[:4]):
            d.text((R.HINT_BOX[0] + 8, R.HINT_BOX[1] + 8 + i * 16), line, fill=(20, 20, 20), font=_FONT_SMALL)
    if game.over or game.won:
        msg = "You Win!" if game.won else "Game Over"
        tw = d.textlength(msg, font=_FONT_BIG)
        d.rounded_rectangle([200, 200, 440, 260], radius=10, fill=(255, 255, 255), outline=(60, 60, 60))
        d.text((320 - tw / 2, 218), msg, fill=(20, 20, 20), font=_FONT_BIG)

    if cursor is not None:                                # the mouse pointer
        x, y = cursor
        d.polygon([(x, y), (x + 12, y + 10), (x + 5, y + 11), (x + 8, y + 18), (x + 5, y + 19), (x + 2, y + 12), (x, y + 16)],
                  fill=(255, 255, 255), outline=(0, 0, 0))
    return im


if __name__ == "__main__":
    from bloons.bot import ScriptedPlayer, best_spot
    from bloons.game import Game
    import time
    g, bot = Game(), ScriptedPlayer()
    for _ in range(12):
        bot.act(g); g.start_round()
        while g.in_round: g.step(); bot.act(g) if g.frame % 40 == 0 else None
    bot.act(g); g.start_round()
    for _ in range(700): g.step()
    g.selected = g.towers[0]
    t0 = time.perf_counter(); im = render(g, cursor=(300, 250)); dt = time.perf_counter() - t0
    im.save("data/bloons_frame.png")
    print(f"rendered in {dt*1000:.1f} ms -> data/bloons_frame.png  (round {g.current_round}, {len(g.bloons)} bloons)")
