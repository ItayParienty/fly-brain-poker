"""The fly at its desk: a trained fly plays Bloons live, for a browser to watch.

A background thread runs one fly - screen, eye, brain, read-out, mouse, game
- in real time (or faster), and streams what happens as server-sent events:
the screen as a JPEG, what each of the eye's 796 columns sees, where the fly
wants to look, four layers of its optic lobe, and every press. The front end
(bloons/desk/) puts the fly at a desk in front of a monitor showing that game.

The read-out is the evolving one from bloons/train.py (data/train/<run>/,
by default the run saved most recently), reloaded at the start of every
game, so the fly on the desk is always the latest one.

    python -m bloons.desk_server              # then open http://localhost:8767

Endpoints
    GET  /meta        the eye's layout: each column's place on the screen
    GET  /events      a stream of frames (text/event-stream)
    POST /speed       {"speed": 0.25 ... 4}   (1 = real time)
    POST /new         start a new game
"""
import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from bloons import rules as R
from bloons.fly_player import Arena, norm_path
from bloons.game import Game
from bloons.screen import Painter
from bloons.train import IDLE_SECONDS, unpack
from flybrain.connectome import DATA_DIR
from flybrain.flyvis_eye import make_eye
from flybrain.motor import Motor, resting_output

STATIC = Path(__file__).with_name("desk")
SHOW_FPS = 25                             # frames sent to the browser per second of wall time
LAYERS = [("L1", "למינה · L1"), ("Mi1", "ערוץ ON · Mi1"), ("Tm1", "ערוץ OFF · Tm1"), ("T4a", "תנועה · T4a")]
EVENTS_HE = {"pick": "הרים", "place": "הניח", "select": "בחר מגדל", "upgrade": "שדרג", "sell": "מכר מגדל",
             "broke": "אין מספיק כסף", "red": "ניסה להניח במקום אסור", "start": "לחץ Start Round",
             "restart": "לחץ Restart", "blocked": "כפתור שדרוג חסום", "wings:": "היכה בכנפיים: סיבוב חדש"}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


class Desk:
    def __init__(self, run, speed=1.0):
        self.run_dir = DATA_DIR / "train" / run
        self.c, self.eye, self.net = make_eye()
        self.motor = Motor(self.c, self.eye, n_agents=1, rest=resting_output(self.net, self.eye))
        self.motor.load_normalisation(norm_path())
        self.layer_idx = [self.motor.types.index(t) for t, _ in LAYERS]
        self.painter = Painter()
        self.speed = speed
        self.lock = threading.Condition()
        self.packet, self.seq = None, 0
        self.want_new = False
        self.new_game()

    def new_game(self):
        st = np.load(self.run_dir / "state.npz")
        self.generation = int(st["generation"])
        self.motor.set(unpack(st["theta"][None], self.motor.n_types))
        self.motor.reset()
        seed = int(time.time()) % 100000
        self.game = Game(seed=seed)
        self.arena = Arena(self.c, self.eye, self.net, self.motor, seeds=[seed], games=[self.game])
        self.events, self.logged, self.idle, self.ended = [], 0, 0, None

    def meta(self):
        owner = self.eye.owner[1::2, 1::2]                  # the eye's patches at half resolution, for drawing
        return dict(centres=np.round(self.eye.centres.astype(float), 1).tolist(), width=R.WIDTH, height=R.HEIGHT,
                    owner=base64.b64encode(np.ascontiguousarray(owner, dtype=np.uint16).tobytes()).decode(),
                    owner_size=[owner.shape[1], owner.shape[0]],
                    layers=[label for _, label in LAYERS], columns=self.c.n_columns,
                    neurons=int(self.c.n_neurons), types=self.motor.n_types)

    def run_forever(self):
        next_show = 0.0
        while True:
            t0 = time.perf_counter()
            if self.want_new or (self.ended is not None and time.time() - self.ended > 5):
                self.want_new = False
                self.new_game()
            if self.ended is None:
                colours = self.arena.frame()
                self._events()
                g = self.game
                self.idle = 0 if g.in_round else self.idle + 1
                restarted = any(e[2] == "restart" for e in self.arena.log[0])
                if g.over or g.won or restarted or self.idle > IDLE_SECONDS * R.FPS:
                    self.ended = time.time()
            else:
                colours = self._last_colours
            self._last_colours = colours
            now = time.perf_counter()
            if now >= next_show:
                self._publish(colours)
                next_show = now + 1.0 / SHOW_FPS
            spare = 1.0 / (R.FPS * self.speed) - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)

    def _events(self):
        for fr, step, ev in self.arena.log[0][self.logged:]:
            word = ev.split(" ")[0]
            if word in ("panel", "deselect"):
                continue
            he = EVENTS_HE.get(word, ev)
            if word in ("pick", "place"):
                he += " " + ev.split(" ")[1]
            elif word == "upgrade":
                he += " " + " ".join(ev.split(" ")[1:])
            self.events.append(dict(t=round(fr / R.FPS, 2), kind=word.rstrip(":"), text=he))
        self.logged = len(self.arena.log[0])
        self.events = self.events[-60:]

    def _publish(self, colours):
        g, m, motor = self.game, self.arena.mice[0], self.motor
        buf = io.BytesIO()
        Image.fromarray(self.painter.picture(g, m)).save(buf, "JPEG", quality=82)
        sal = motor.salience[:, 0].cpu().numpy()
        sal = (sal - sal.min()) / max(float(sal.max() - sal.min()), 1e-6)
        maps = motor.maps(self.arena.net.rate)[self.layer_idx, :, 0].cpu().numpy()
        packet = dict(
            screen=base64.b64encode(buf.getvalue()).decode(),
            eye=b64(np.clip(colours[0] * 255, 0, 255)), salience=b64(sal * 255),
            layers=[b64(np.clip(128 + 40 * m_, 0, 255)) for m_ in maps],
            pointer=[m.x, m.y], holding=m.tool, fixation=int(motor.fixation[0]),
            round=max(g.current_round, 1), money=g.money, lives=g.lives, towers=len(g.towers),
            in_round=g.in_round, over=g.over, won=g.won, ended=self.ended is not None,
            generation=self.generation, speed=self.speed, time=round(self.arena.frame_no / R.FPS, 2),
            events=self.events[-30:], last_event=len(self.events) and self.events[-1],
            presses=len(self.arena.log[0]))
        with self.lock:
            self.packet = json.dumps(packet); self.seq += 1
            self.lock.notify_all()


def make_handler(desk):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def _file(self, name):
            path = (STATIC / name).resolve()
            if STATIC.resolve() not in path.parents or not path.exists():
                self.send_error(404); return
            body = path.read_bytes()
            kind = {"html": "text/html; charset=utf-8", "js": "text/javascript", "css": "text/css"}.get(path.suffix[1:], "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/meta":
                self._json(desk.meta())
            elif self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with desk.lock:
                            desk.lock.wait_for(lambda: desk.seq != seen, timeout=5)
                            seen, packet = desk.seq, desk.packet
                        if packet:
                            self.wfile.write(b"data: " + packet.encode() + b"\n\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
            elif self.path in ("/", "/index.html"):
                self._file("index.html")
            else:
                self._file(self.path.lstrip("/"))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/speed":
                desk.speed = float(min(max(body.get("speed", 1), 0.1), 8)); self._json({"speed": desk.speed})
            elif self.path == "/new":
                desk.want_new = True; self._json({"ok": True})
            else:
                self.send_error(404)
    return Handler


def main():
    from flybrain import console_utf8  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--run", help="a training run in data/train/ (default: the one saved most recently)")
    ap.add_argument("--speed", type=float, default=0.5, help="1 = real time; the fly is quick, so it starts at half")
    args = ap.parse_args()
    run = args.run or max((DATA_DIR / "train").glob("*/state.npz"), key=lambda p: p.stat().st_mtime).parent.name
    desk = Desk(run, args.speed)
    threading.Thread(target=desk.run_forever, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(desk))
    print(f"the fly's desk at http://localhost:{args.port}   (read-out: {run}, generation {desk.generation})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
