"""Pull the geometry of Bloons Tower Defense 1 out of the original SWF.

    python -m bloons.extract_original path/to/bloons_tower_defense_1.swf

writes bloons/original.py: the six bloon motion tweens (every bloon in the
original is a keyframed animation along the track, one clip per colour), the
frame at which each clip reports an escape, the bounding boxes hit tests use,
and the 31 rectangles that block tower placement.  The SWF itself is not
part of this repository; it is on archive.org (item bloons_tower_defense_1).

Only the SWF tag types needed are parsed: DefineShape (bounds), DefineSprite
with PlaceObject2/3 (per-frame matrices, names) and ExportAssets (names).
"""
import struct
import sys
import zlib
from pathlib import Path


class Bits:
    def __init__(self, data, pos=0):
        self.data, self.pos, self.bit = data, pos, 0

    def read(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.data[self.pos] >> (7 - self.bit)) & 1)
            self.bit += 1
            if self.bit == 8:
                self.bit = 0; self.pos += 1
        return v

    def sread(self, n):
        v = self.read(n)
        return v - (1 << n) if n and v & (1 << (n - 1)) else v

    def align(self):
        if self.bit:
            self.bit = 0; self.pos += 1


def read_rect(b):
    n = b.read(5); r = [b.sread(n) / 20 for _ in range(4)]; b.align()
    return r                                     # xmin, xmax, ymin, ymax in px


def read_matrix(b):
    m = dict(sx=1.0, sy=1.0, r0=0.0, r1=0.0, tx=0.0, ty=0.0)
    if b.read(1):
        n = b.read(5); m["sx"] = b.sread(n) / 65536; m["sy"] = b.sread(n) / 65536
    if b.read(1):
        n = b.read(5); m["r0"] = b.sread(n) / 65536; m["r1"] = b.sread(n) / 65536
    n = b.read(5); m["tx"] = b.sread(n) / 20; m["ty"] = b.sread(n) / 20
    b.align()
    return m


def read_string(data, pos):
    end = data.index(b"\0", pos)
    return data[pos:end].decode("latin-1"), end + 1


def tags(data, pos, end):
    while pos < end:
        h = struct.unpack_from("<H", data, pos)[0]; pos += 2
        code, length = h >> 6, h & 0x3F
        if length == 0x3F:
            length = struct.unpack_from("<I", data, pos)[0]; pos += 4
        yield code, data[pos:pos + length]
        pos += length


def place_object(body, v3):
    flags = body[0]; pos = 1; flags2 = 0
    if v3:
        flags2 = body[1]; pos = 2
    depth = struct.unpack_from("<H", body, pos)[0]; pos += 2
    out = dict(depth=depth)
    if v3 and flags2 & 8:
        _, pos = read_string(body, pos)
    if flags & 2:
        out["char"] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    if flags & 4:
        b = Bits(body, pos); out["matrix"] = read_matrix(b); pos = b.pos
    if flags & 8:
        b = Bits(body, pos); add, mul, n = b.read(1), b.read(1), b.read(4)
        for _ in range((4 if mul else 0) + (4 if add else 0)):
            b.read(n)
        b.align(); pos = b.pos
    if flags & 16:
        pos += 2
    if flags & 32:
        out["name"], pos = read_string(body, pos)
    return out


def parse(data):
    shapes, sprites, names, main = {}, {}, {}, []

    def walk(seq, frames):
        cur = []
        for code, body in seq:
            if code == 1:
                frames.append(cur); cur = []
            elif code in (26, 70):
                cur.append(place_object(body, code == 70))
            elif code in (2, 22, 32, 83):
                sid = struct.unpack_from("<H", body, 0)[0]; shapes[sid] = read_rect(Bits(body, 2))
            elif code == 39:
                sid = struct.unpack_from("<H", body, 0)[0]
                fr = []; walk(tags(body, 4, len(body)), fr); sprites[sid] = fr
            elif code == 56:
                n = struct.unpack_from("<H", body, 0)[0]; pos = 2
                for _ in range(n):
                    cid = struct.unpack_from("<H", body, pos)[0]; pos += 2
                    nm, pos = read_string(body, pos); names[nm] = cid
        if cur:
            frames.append(cur)

    body = zlib.decompress(data[8:]) if data[:3] == b"CWS" else data[8:]
    b = Bits(body); read_rect(b)
    walk(tags(body, b.pos + 4, len(body)), main)
    return shapes, sprites, names, main


def transform_box(box, m):
    xs, ys = [], []
    for x, y in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
        xs.append(x * m["sx"] + y * m["r1"] + m["tx"]); ys.append(x * m["r0"] + y * m["sy"] + m["ty"])
    return [min(xs), min(ys), max(xs), max(ys)]


def box_of(sid, shapes, sprites):
    """Bounding box of a shape or of everything a sprite places on its first frame."""
    if sid in shapes:
        x0, x1, y0, y1 = shapes[sid]; return [x0, y0, x1, y1]
    box = None
    for p in sprites.get(sid, [[]])[0]:
        if "char" not in p:
            continue
        b = box_of(p["char"], shapes, sprites)
        if b is None:
            continue
        b = transform_box(b, p.get("matrix", dict(sx=1, sy=1, r0=0, r1=0, tx=0, ty=0)))
        box = b if box is None else [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
    return box


def tween(frames):
    """(tx, ty) of the sprite's single child on every frame; frames that only
    show the previous position repeat it."""
    out, cur = [], None
    for fr in frames:
        for p in fr:
            if "matrix" in p:
                cur = (p["matrix"]["tx"], p["matrix"]["ty"])
        out.append(cur)
    return out


KINDS = [("Red", "plain1"), ("Blue", "plain2"), ("Green", "plain3"), ("Yellow", "plain4"), ("Black", "plain5"), ("White", "plain6")]
# frame (1-based) whose script calls GotToEnd, read from the decompiled clips
EXIT_FRAME = {"Red": 916, "Blue": 768, "Green": 533, "Yellow": 295, "Black": 519, "White": 402}


def main():
    swf = Path(sys.argv[1])
    shapes, sprites, names, timeline = parse(swf.read_bytes())
    r = lambda v: round(v, 2)
    lines = ['"""Geometry of the original game, generated by extract_original.py. Do not edit."""', ""]
    lines.append("# per-frame position of each bloon clip's `inner` child, relative to the clip")
    lines.append("TWEENS = {")
    for kind, sym in KINDS:
        pts = tween(sprites[names[sym]])
        assert all(p is not None for p in pts), kind
        lines.append(f'    "{kind}": [' + ", ".join(f"({r(x)}, {r(y)})" for x, y in pts) + "],")
    lines.append("}")
    lines.append(f"EXIT_FRAME = {EXIT_FRAME}")
    lines.append("# hit box of each bloon's `inner`, relative to the tween position")
    lines.append("BLOON_BOX = {")
    for kind, sym in KINDS:
        inner = sprites[names[sym]][0][0]["char"]
        lines.append(f'    "{kind}": {[r(v) for v in box_of(inner, shapes, sprites)]},')
    lines.append("}")
    lines.append("# footprint used for placement, relative to the tower position")
    lines.append("TOWER_BOX = {")
    for kind, sym in (("Dart", "towerdart"), ("Tack", "towertack"), ("Ice", "towerice"), ("Bomb", "towerbomb"), ("Super", "towersuper")):
        for p in sprites[names[sym]][0]:
            if p.get("name") == "hitbit":
                lines.append(f'    "{kind}": {[r(v) for v in transform_box(box_of(p["char"], shapes, sprites), p["matrix"])]},')
    lines.append("}")
    blocks = []
    for fr in timeline:
        for p in fr:
            if p.get("name", "").startswith("pathhit"):
                blocks.append([r(v) for v in transform_box(box_of(p["char"], shapes, sprites), p["matrix"])])
    lines.append("# rectangles a tower footprint may not overlap: the track, the panel, the margins")
    lines.append(f"PATH_BLOCKS = {blocks}")
    # bullets: the standard hit box (shape inside sprite 'hitbit'), the bomb blast, the ice field
    dart = [p for p in sprites[names["dart"]][0] if p.get("name") == "hitbit"][0]
    lines.append(f'DART_BOX = {[r(v) for v in transform_box(box_of(dart["char"], shapes, sprites), dart["matrix"])]}  # +6 px along the dart')
    bomb = sprites[names["bomb"]]
    b0 = [p for p in bomb[0] if p.get("name") == "hitbit"][0]
    b1 = [p for p in bomb[1] if p["depth"] == b0["depth"]][0]
    lines.append(f'BOMB_BOX = {[r(v) for v in transform_box(box_of(b0["char"], shapes, sprites), b0["matrix"])]}')
    lines.append(f'BLAST_BOX = {[r(v) for v in transform_box(box_of(b0["char"], shapes, sprites), b1["matrix"])]}  # at bullet scale 100')
    lines.append(f"BLAST_FRAMES = {len(bomb) - 1}")
    ice = [p for p in sprites[names["ice"]][0] if p.get("name") == "hitbit"][0]
    lines.append(f'ICE_BOX = {[r(v) for v in transform_box(box_of(ice["char"], shapes, sprites), ice["matrix"])]}  # at bullet scale 100')
    tack = tween(sprites[95]) if 95 in sprites else None
    sub = [p for p in sprites[names["tack"]][0] if p.get("name") == "tack1"][0]
    tk = [p for p in sprites[sub["char"]][0] if "char" in p][0]
    lines.append("# a tack volley: the 8 tacks sit around the tower and fly out; distance per frame of age")
    lines.append(f'TACK_REACH = {[r(-t[1]) for t in tween(sprites[sub["char"]]) if t]}')
    lines.append(f'TACK_BOX = {[r(v) for v in box_of(tk["char"], shapes, sprites)]}')
    out = Path(__file__).with_name("original.py")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out.name, f"({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
