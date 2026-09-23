"""Pull the original game's pictures out of its SWF, for the screen the fly sees.

    python -m bloons.extract_art data/btd1.swf --ffdec path/to/ffdec-cli.jar

The art is Ninja Kiwi's and stays out of this repository: this writes it to
data/btd1_art/ (git-ignored), from a copy of the game you supply.  Without it
bloons/screen.py falls back to its own simple drawings.

What is extracted is the SWF's own building blocks, not finished pictures:

  * every shape and button, rasterised by JPEXS (ffdec) at twice the game's
    resolution, with the point that is the shape's origin
  * every static text, rasterised from ffdec's SVG (resvg)
  * the embedded fonts, for the text fields the game fills in while it runs
  * every sprite's timeline - which child sits at which depth, with which
    matrix and colour transform, on every frame - its frame labels, and the
    layout of every text field
  * the few sprites that tween a morph shape (a monkey's throwing arm, the
    Start Round button's pulse), rasterised whole, frame by frame

bloons/art.py puts these together the way Flash does, for whatever state
the game is in: a tower turned towards its target, a bloon on frame 31 of
its wobble, the range ring grown to 2 x 140 px.
"""
import argparse
import io
import json
import re
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

from PIL import Image

from bloons.extract_original import Bits, read_matrix, read_rect, read_string, tags
from flybrain.connectome import DATA_DIR

OUT = DATA_DIR / "btd1_art"
ZOOM = 2


# ---------------------------------------------------------------- the SWF's structure
def place_object(body, v3):
    flags = body[0]; pos = 1; flags2 = 0
    if v3:
        flags2 = body[1]; pos = 2
    depth = struct.unpack_from("<H", body, pos)[0]; pos += 2
    out = dict(depth=depth, move=bool(flags & 1))
    if v3 and flags2 & 8:
        _, pos = read_string(body, pos)
    if flags & 2:
        out["char"] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    if flags & 4:
        b = Bits(body, pos); out["matrix"] = read_matrix(b); pos = b.pos
    if flags & 8:
        b = Bits(body, pos); add, mul, n = b.read(1), b.read(1), b.read(4)
        m = [b.sread(n) / 256 for _ in range(4)] if mul else [1.0, 1.0, 1.0, 1.0]
        a = [b.sread(n) for _ in range(4)] if add else [0, 0, 0, 0]
        b.align(); pos = b.pos
        out["cx"] = m + a                                   # multiply r g b a, then add r g b a (0..255)
    if flags & 16:
        out["ratio"] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    if flags & 32:
        out["name"], pos = read_string(body, pos)
    if flags & 64:
        out["mask"] = struct.unpack_from("<H", body, pos)[0]; pos += 2    # clips depths up to this; not drawn
    return out


def edit_text(body):
    cid = struct.unpack_from("<H", body, 0)[0]
    b = Bits(body, 2); x0, x1, y0, y1 = read_rect(b); pos = b.pos
    f1, f2 = body[pos], body[pos + 1]; pos += 2
    bit = lambda f, i: (f >> (7 - i)) & 1
    out = dict(bounds=[x0, y0, x1, y1], multiline=bit(f1, 2), wrap=bit(f1, 1), html=bit(f2, 6))
    if bit(f1, 7):
        out["font"] = struct.unpack_from("<H", body, pos)[0]; pos += 2
    if bit(f2, 0):
        _, pos = read_string(body, pos)
    if bit(f1, 7) or bit(f2, 0):
        out["size"] = struct.unpack_from("<H", body, pos)[0] / 20; pos += 2
    if bit(f1, 5):
        out["color"] = list(body[pos:pos + 4]); pos += 4
    if bit(f1, 6):
        pos += 2
    if bit(f2, 2):
        out["align"] = ["left", "right", "center", "justify"][body[pos]]; pos += 1
        out["left"], out["right"], out["indent"], out["leading"] = [v / 20 for v in struct.unpack_from("<HHhh", body, pos)]
        pos += 8
    out["var"], pos = read_string(body, pos)
    if bit(f1, 0):
        out["text"], pos = read_string(body, pos)
    return cid, out


def parse(data):
    """Timelines (frames of display-list changes), frame labels, text fields,
    exported names and which ids are shapes, buttons, texts and fonts."""
    kinds, timelines, labels, fields, names, buttons = {}, {}, {}, {}, {}, {}

    def walk(seq, sid):
        frames, cur, lab = [], [], {}
        for code, body in seq:
            if code == 1:
                frames.append(cur); cur = []
            elif code in (26, 70):
                cur.append(place_object(body, code == 70))
            elif code == 28:
                cur.append(dict(depth=struct.unpack_from("<H", body, 0)[0], remove=True))
            elif code == 43:
                lab[read_string(body, 0)[0]] = len(frames)
            elif code in (2, 22, 32, 83):
                kinds[struct.unpack_from("<H", body, 0)[0]] = "shape"
            elif code in (46, 84):
                kinds[struct.unpack_from("<H", body, 0)[0]] = "morph"
            elif code in (7, 34):
                bid = struct.unpack_from("<H", body, 0)[0]; kinds[bid] = "button"
                if code == 34:                                      # its records: which character each state shows
                    pos, recs = 5, []
                    while body[pos]:
                        flags = body[pos]
                        ch, depth = struct.unpack_from("<HH", body, pos + 1)
                        b = Bits(body, pos + 5); m = read_matrix(b)
                        add, mul, n = b.read(1), b.read(1), b.read(4)
                        mm = [b.sread(n) / 256 for _ in range(4)] if mul else [1.0, 1.0, 1.0, 1.0]
                        aa = [b.sread(n) for _ in range(4)] if add else [0, 0, 0, 0]
                        b.align(); pos = b.pos
                        states = [s_ for i, s_ in enumerate(("up", "over", "down", "hit")) if flags >> i & 1]
                        recs.append(dict(char=ch, depth=depth, matrix=m, cx=mm + aa, states=states))
                        if flags & 0x30:
                            break                                   # filters / blend modes: not used by this game
                    buttons[bid] = recs
            elif code in (11, 33):
                kinds[struct.unpack_from("<H", body, 0)[0]] = "text"
            elif code in (10, 48, 75):
                kinds[struct.unpack_from("<H", body, 0)[0]] = "font"
            elif code == 37:
                cid, f = edit_text(body); fields[cid] = f; kinds[cid] = "field"
            elif code == 39:
                cid = struct.unpack_from("<H", body, 0)[0]; kinds[cid] = "sprite"
                walk(tags(body, 4, len(body)), cid)
            elif code == 56:
                n = struct.unpack_from("<H", body, 0)[0]; pos = 2
                for _ in range(n):
                    cid = struct.unpack_from("<H", body, pos)[0]; pos += 2
                    nm, pos = read_string(body, pos); names[nm] = cid
        if cur:
            frames.append(cur)
        timelines[sid] = frames; labels[sid] = lab

    body = zlib.decompress(data[8:]) if data[:3] == b"CWS" else data[8:]
    b = Bits(body); read_rect(b)
    walk(tags(body, b.pos + 4, len(body)), 0)                # id 0 is the main timeline
    return kinds, timelines, labels, fields, names, buttons


def _stage_list(timelines, labels):
    """The main timeline's display list on its "main" frame, without the title menu,
    the intro and the stage mask (what a running game shows)."""
    dl = {}
    for f in timelines[0][:labels[0]["main"] + 1]:
        for p in f:
            if p.get("remove"):
                dl.pop(p["depth"], None)
            else:
                dl[p["depth"]] = p
    return {d: p for d, p in dl.items() if d < 510 and "mask" not in p}


# ---------------------------------------------------------------- rasterising the leaves
def svg_origin(svg_text):
    """ffdec puts a character's origin at the root group's translation."""
    m = re.search(r'<g transform="matrix\(([^)]*)\)"', svg_text)
    v = [float(x) for x in m.group(1).split(",")]
    return v[4], v[5]


def ffdec(jar, args):
    subprocess.run(["java", "-jar", str(jar)] + args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    from flybrain import console_utf8  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("swf")
    ap.add_argument("--ffdec", required=True, help="path to JPEXS ffdec-cli.jar (Java 8+)")
    args = ap.parse_args()
    import resvg_py

    data = Path(args.swf).read_bytes()
    kinds, timelines, labels, fields, names, buttons = parse(data)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "leaves").mkdir(parents=True)
    leaves = {}
    with tempfile.TemporaryDirectory() as tmp:          # Java wants a plain path
        tmp = Path(tmp); swf = tmp / "game.swf"; swf.write_bytes(data)
        print("rasterising shapes and buttons (ffdec, zoom 2) ...", flush=True)
        ffdec(args.ffdec, ["-zoom", str(ZOOM), "-format", "shape:png,button:png", "-export", "shape,button", str(tmp / "png"), str(swf)])
        print("vector exports for origins and text ...", flush=True)
        ffdec(args.ffdec, ["-format", "shape:svg,button:svg,text:svg", "-export", "shape,button,text", str(tmp / "svg"), str(swf)])
        ffdec(args.ffdec, ["-format", "font:ttf", "-export", "font", str(tmp / "font"), str(swf)])
        for png in sorted((tmp / "png" / "shapes").glob("*.png")):
            sid = int(png.stem)
            ox, oy = svg_origin((tmp / "svg" / "shapes" / f"{sid}.svg").read_text(encoding="utf-8"))
            shutil.copy(png, OUT / "leaves" / f"{sid}.png")
            leaves[sid] = dict(file=f"{sid}.png", ox=ox * ZOOM, oy=oy * ZOOM)
        for d in sorted((tmp / "png" / "buttons").iterdir()):
            sid = int(d.name.split("_")[1])
            for state in ("1_up", "2_over", "3_down"):
                if (d / f"{state}.png").exists():
                    ox, oy = svg_origin((tmp / "svg" / "buttons" / d.name / f"{state}.svg").read_text(encoding="utf-8"))
                    name = f"{sid}_{state.split('_')[1]}.png"
                    shutil.copy(d / f"{state}.png", OUT / "leaves" / name)
                    leaves.setdefault(sid, {})[state.split("_")[1]] = dict(file=name, ox=ox * ZOOM, oy=oy * ZOOM)
        for svg in sorted((tmp / "svg" / "texts").glob("*.svg")):
            sid = int(svg.stem); text = svg.read_text(encoding="utf-8")
            ox, oy = svg_origin(text)
            png = bytes(resvg_py.svg_to_bytes(svg_string=text, zoom=ZOOM))
            Image.open(io.BytesIO(png)).save(OUT / "leaves" / f"{sid}.png")
            leaves[sid] = dict(file=f"{sid}.png", ox=ox * ZOOM, oy=oy * ZOOM)
        # sprites that tween morph shapes: rasterised whole, one picture per frame
        # (only those the game itself uses: the stage from the "main" frame, and the library symbols it attaches)
        used, todo = set(), [p["char"] for d, p in _stage_list(timelines, labels).items() if "char" in p] + list(names.values())
        while todo:
            c = todo.pop()
            if c in used:
                continue
            used.add(c)
            todo += [p["char"] for f in timelines.get(c, []) for p in f if "char" in p] + [r["char"] for r in buttons.get(c, [])]
        baked = sorted({sid for sid in used if sid in timelines
                        and any(kinds.get(p.get("char")) == "morph" for f in timelines[sid] for p in f)})
        if baked:
            ids = ",".join(map(str, baked))
            print(f"baking {len(baked)} sprites with morph shapes ...", flush=True)
            ffdec(args.ffdec, ["-selectid", ids, "-zoom", str(ZOOM), "-format", "sprite:png", "-export", "sprite", str(tmp / "bake_png"), str(swf)])
            ffdec(args.ffdec, ["-selectid", ids, "-format", "sprite:svg", "-export", "sprite", str(tmp / "bake_svg"), str(swf)])
            svgs = {d.name: d for d in (tmp / "bake_svg").rglob("DefineSprite_*") if d.is_dir()}
            for d in [d for d in (tmp / "bake_png").rglob("DefineSprite_*") if d.is_dir()]:
                sid = int(d.name.split("_")[1]); frames = {}
                for png in d.glob("*.png"):
                    f = int(png.stem) - 1
                    ox, oy = svg_origin((svgs[d.name] / f"{png.stem}.svg").read_text(encoding="utf-8"))
                    name = f"{sid}_f{f}.png"; shutil.copy(png, OUT / "leaves" / name)
                    frames[f] = dict(file=name, ox=ox * ZOOM, oy=oy * ZOOM)
                leaves[sid] = dict(frames=frames)
        fonts = {}
        (OUT / "fonts").mkdir()
        for ttf in (tmp / "font").rglob("*.ttf"):
            fid = int(ttf.stem.split("_")[0]); shutil.copy(ttf, OUT / "fonts" / f"{fid}.ttf"); fonts[fid] = f"{fid}.ttf"
    lib = dict(kinds={str(k): v for k, v in kinds.items()}, leaves={str(k): v for k, v in leaves.items()},
               timelines={str(k): v for k, v in timelines.items()}, labels={str(k): v for k, v in labels.items() if v},
               fields={str(k): v for k, v in fields.items()}, buttons={str(k): v for k, v in buttons.items()}, names=names, fonts={str(k): v for k, v in fonts.items()}, zoom=ZOOM)
    (OUT / "library.json").write_text(json.dumps(lib), encoding="utf-8")
    print(f"wrote {OUT}: {len(leaves)} leaves, {len(timelines)} timelines, {len(fields)} text fields, {len(fonts)} fonts")


if __name__ == "__main__":
    main()
