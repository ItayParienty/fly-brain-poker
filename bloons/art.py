"""The original game's pictures, put together the way Flash puts them together.

bloons/extract_art.py leaves the SWF's building blocks in data/btd1_art/:
rasterised shapes, buttons and static texts, the fonts, and every sprite's
timeline.  This module is a small Flash renderer over them: a sprite on a
given frame is its display list on that frame - each child with its matrix
and colour transform, drawn in depth order, sprites recursively.  Scripts
are not run; where the game's scripts change what is shown (the range ring
grown to 2 x range, the "can't place" ring, a frozen bloon's ice, the text
in a field) the caller says so with `set`:

    lib = Library()
    img, ox, oy = lib.picture(215, frame=lib.label(215, "tack"),
                              set={"radiusmc": dict(frame=1, scale=1.4, cantplace=dict(visible=True))})

Everything is drawn at twice the game's resolution and shrunk, as Flash's
anti-aliasing would have it.
"""
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flybrain.connectome import DATA_DIR

LEAF_BUDGET = 96 * 2**20           # bytes of shape images kept in memory (all of them: ~1 GB)

ART = DATA_DIR / "btd1_art"
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)      # a, b, c, d, tx, ty:  x' = a x + c y + tx,  y' = b x + d y + ty


def available():
    return (ART / "library.json").exists()


def compose(m, n):
    """The matrix that applies n, then m."""
    a, b, c, d, tx, ty = m
    a2, b2, c2, d2, tx2, ty2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2, a * tx2 + c * ty2 + tx, b * tx2 + d * ty2 + ty)


def swf_matrix(m):
    return (m["sx"], m["r0"], m["r1"], m["sy"], m["tx"], m["ty"])


def rotation(deg, x=0.0, y=0.0):
    r = math.radians(deg); c, s = math.cos(r), math.sin(r)
    return (c, s, -s, c, x, y)


class Library:
    def __init__(self, path=ART):
        self.path = Path(path)
        lib = json.loads((self.path / "library.json").read_text(encoding="utf-8"))
        self.zoom = lib["zoom"]
        self.kinds = {int(k): v for k, v in lib["kinds"].items()}
        self.leaves = {int(k): v for k, v in lib["leaves"].items()}
        self.timelines = {int(k): v for k, v in lib["timelines"].items()}
        self.labels = {int(k): v for k, v in lib["labels"].items()}
        self.fields = {int(k): v for k, v in lib["fields"].items()}
        self.buttons = {int(k): v for k, v in lib.get("buttons", {}).items()}
        self.fonts = {int(k): v for k, v in lib["fonts"].items()}
        self.names = lib["names"]
        self._images, self._lists, self._fonts = OrderedDict(), {}, {}
        self._image_bytes = 0

    # ------------------------------------------------ timelines
    def label(self, sid, name):
        return self.labels[sid][name]

    def display_list(self, sid, frame):
        """{depth: placement} on a frame (0-based), with the SWF's move/replace/remove rules."""
        key = (sid, frame)
        if key in self._lists:
            return self._lists[key]
        frames = self.timelines[sid]
        frame = min(frame, len(frames) - 1)
        dl = dict(self.display_list(sid, frame - 1)) if frame > 0 else {}
        for p in frames[frame]:
            d = p["depth"]
            if p.get("remove"):
                dl.pop(d, None); continue
            if p.get("move") and d in dl:
                cur = dict(dl[d])
                for k in ("char", "matrix", "cx", "name", "ratio", "mask"):
                    if k in p:
                        cur[k] = p[k]
                dl[d] = cur
            else:
                dl[d] = {k: v for k, v in p.items() if k not in ("depth", "move")}
        self._lists[key] = dl
        return dl

    # ------------------------------------------------ leaves
    def _leaf(self, sid, state="up"):
        """(premultiplied RGBA float image at 2x, origin x, origin y)."""
        key = (sid, state)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        info = self.leaves.get(sid)
        if info is None:
            self._images[key] = None
            return None
        if "frames" in info:
            info = info["frames"][state]
        elif "file" not in info:
            info = info.get(state) or info.get("up")
        im = np.asarray(Image.open(self.path / "leaves" / info["file"]).convert("RGBA")).astype(np.float32) / 255
        im[:, :, :3] *= im[:, :, 3:]
        self._images[key] = (im, info["ox"], info["oy"])
        self._image_bytes += im.nbytes
        while self._image_bytes > LEAF_BUDGET and len(self._images) > 1:      # the least recently used go
            _, old = self._images.popitem(last=False)
            self._image_bytes -= old[0].nbytes if old else 0
        return self._images[key]

    def _font(self, fid, size):
        key = (fid, size)
        if key not in self._fonts:
            self._fonts[key] = ImageFont.truetype(str(self.path / "fonts" / self.fonts[fid]), size)
        return self._fonts[key]

    def _field(self, fid, text):
        """A text field (DefineEditText) with the given text, as a leaf."""
        f = self.fields[fid]
        z = self.zoom
        x0, y0, x1, y1 = f["bounds"]
        w, h = int(math.ceil((x1 - x0) * z)) + 2, int(math.ceil((y1 - y0) * z)) + 2
        im = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(im)
        font = self._font(f["font"], max(1, int(round(f["size"] * z))))
        ascent, descent = font.getmetrics()
        lines = str(text).split("\r") if f.get("multiline") else [str(text).replace("\r", " ")]
        if f.get("wrap"):
            wrapped, width = [], (x1 - x0 - 4) * z
            for line in lines:
                cur = ""
                for word in line.split(" "):
                    if cur and d.textlength(cur + " " + word, font=font) > width:
                        wrapped.append(cur); cur = word
                    else:
                        cur = (cur + " " + word).strip()
                wrapped.append(cur)
            lines = wrapped
        step = ascent + descent + f.get("leading", 0) * z
        for i, line in enumerate(lines):
            tw = d.textlength(line, font=font)
            align = f.get("align", "left")
            x = 2 * z + (0 if align in ("left", "justify") else ((x1 - x0 - 4) * z - tw) / (2 if align == "center" else 1))
            d.text((x, 2 * z + ascent + i * step), line, fill=255, font=font, anchor="ls")
        cov = np.asarray(im).astype(np.float32) / 255
        r, g, b, a = [c / 255 for c in f.get("color", [0, 0, 0, 255])]
        out = np.zeros((h, w, 4), np.float32)
        out[:, :, 3] = cov * a
        out[:, :, 0], out[:, :, 1], out[:, :, 2] = r * out[:, :, 3], g * out[:, :, 3], b * out[:, :, 3]
        return out, -x0 * z, -y0 * z

    # ------------------------------------------------ drawing
    def draw(self, canvas, sid, frame=0, m=IDENTITY, cx=(), set=None, state="up", text=None, depths=None):
        """Draw character `sid` onto `canvas` (premultiplied float RGBA, 2x), with matrix m
        from the character's coordinates (1x) to the canvas's pixels.
        set: {child name: {frame, visible, scale, rotation, text, dx, dy, <child name>: {...}}}."""
        kind = "sprite" if sid == 0 else self.kinds.get(sid)       # 0 is the main timeline
        if kind == "sprite" and "frames" in self.leaves.get(sid, {}):
            kind = "baked"                                          # tweens a morph shape: drawn from its pictures
        if kind == "button" and sid in self.buttons:                # a state's records, like a one-frame sprite
            for r in sorted(self.buttons[sid], key=lambda r: r["depth"]):
                if state in r["states"]:
                    self.draw(canvas, r["char"], frame, compose(m, swf_matrix(r["matrix"])), cx + (r["cx"],))
            return
        if kind == "sprite":
            for depth in sorted(self.display_list(sid, frame)):
                if depths is not None and not depths[0] <= depth <= depths[1]:
                    continue
                p = self.display_list(sid, frame)[depth]
                if "char" not in p or "mask" in p or p.get("name", "").startswith("pathhit"):
                    continue
                o = (set or {}).get(p.get("name", ""), {}) if p.get("name") else {}
                if o.get("visible") is False:
                    continue
                pm = swf_matrix(p["matrix"]) if "matrix" in p else IDENTITY
                if "dx" in o or "dy" in o:
                    pm = compose((1, 0, 0, 1, o.get("dx", 0), o.get("dy", 0)), pm)
                if "scale" in o:
                    pm = compose(pm, (o["scale"], 0, 0, o["scale"], 0, 0))
                if "rotation" in o:
                    pm = compose(pm, rotation(o["rotation"]))
                sub = {k: v for k, v in o.items() if isinstance(v, dict)}
                self.draw(canvas, p["char"], o.get("frame", 0), compose(m, pm), cx + ((p["cx"],) if "cx" in p else ()),
                          sub, o.get("state", "up"), o.get("text"))
            return
        if kind == "field":
            leaf = self._field(sid, text if text is not None else self.fields[sid].get("text", ""))
        elif kind == "baked":
            frames = self.leaves[sid]["frames"]
            leaf = self._leaf(sid, str(min(frame, len(frames) - 1)))
        else:
            leaf = self._leaf(sid, state)
        if leaf is None:
            return
        img, ox, oy = leaf
        z = self.zoom
        # canvas pixel = m . ((u - ox) / z, (v - oy) / z); PIL wants the inverse
        a, b, c, d, tx, ty = compose(m, (1 / z, 0, 0, 1 / z, -ox / z, -oy / z))
        det = a * d - b * c
        if abs(det) < 1e-9:
            return
        ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
        itx, ity = -(ia * tx + ic * ty), -(ib * tx + id_ * ty)
        h, w = img.shape[:2]
        corners = [(a * u + c * v + tx, b * u + d * v + ty) for u, v in ((0, 0), (w, 0), (0, h), (w, h))]
        X0 = max(int(math.floor(min(p[0] for p in corners))) - 1, 0); X1 = min(int(math.ceil(max(p[0] for p in corners))) + 1, canvas.shape[1])
        Y0 = max(int(math.floor(min(p[1] for p in corners))) - 1, 0); Y1 = min(int(math.ceil(max(p[1] for p in corners))) + 1, canvas.shape[0])
        if X1 <= X0 or Y1 <= Y0:
            return
        out = np.empty((Y1 - Y0, X1 - X0, 4), np.float32)
        coeffs = (ia, ic, ia * X0 + ic * Y0 + itx, ib, id_, ib * X0 + id_ * Y0 + ity)
        for ch in range(4):
            band = Image.fromarray(img[:, :, ch], mode="F")
            out[:, :, ch] = np.asarray(band.transform((X1 - X0, Y1 - Y0), Image.AFFINE, coeffs, resample=Image.BILINEAR))
        for t in reversed(cx):                                   # the innermost colour transform first
            if list(t[:3]) == [1.0, 1.0, 1.0] and list(t[4:]) == [0, 0, 0, 0]:
                out *= t[3]                                      # the common case: transparency only
            else:
                alpha = np.maximum(out[:, :, 3:], 1e-6)
                straight = out[:, :, :3] / alpha
                na = np.clip(out[:, :, 3:] * t[3] + t[7] / 255, 0, 1)
                col = np.clip(straight * np.array(t[:3]) + np.array(t[4:7]) / 255, 0, 1)
                out = np.concatenate([col * na, na], axis=2)
        region = canvas[Y0:Y1, X0:X1]
        region *= (1 - out[:, :, 3:])
        region += out

    def picture(self, sid, frame=0, m=IDENTITY, set=None, state="up", text=None, box=None):
        """Draw one character on its own. Returns (straight RGBA uint8 image at 1x, origin x, origin y):
        the character's origin falls at (origin x, origin y) of the image."""
        z = self.zoom
        if box is None:
            box = (-160, -160, 160, 160)                          # x0, y0, x1, y1 around the origin, 1x px
        x0, y0, x1, y1 = box
        canvas = np.zeros((int((y1 - y0) * z), int((x1 - x0) * z), 4), np.float32)
        self.draw(canvas, sid, frame, compose((z, 0, 0, z, -x0 * z, -y0 * z), m), (), set, state, text)
        return shrink(canvas, z), -x0, -y0


def shrink(canvas, z):
    """2x premultiplied float -> 1x straight RGBA uint8, averaging (Flash-like anti-aliasing)."""
    h, w = canvas.shape[0] // z * z, canvas.shape[1] // z * z
    c = canvas[:h, :w].reshape(h // z, z, w // z, z, 4).mean(axis=(1, 3))
    a = c[:, :, 3:]
    rgb = np.where(a > 1e-6, c[:, :, :3] / np.maximum(a, 1e-6), 0)
    return np.clip(np.concatenate([rgb, a], axis=2) * 255 + 0.5, 0, 255).astype(np.uint8)


def trim(img, ox, oy):
    """Cut an RGBA picture down to its visible part, moving the origin with it."""
    ys, xs = np.nonzero(img[:, :, 3])
    if len(ys) == 0:
        return img[:1, :1].copy(), ox, oy
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1].copy(), ox - x0, oy - y0             # a copy: a view would keep the whole canvas
