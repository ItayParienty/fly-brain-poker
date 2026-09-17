"""A blackjack table with several flies, a dealer, and a seat for a human.

Runs the game in a background thread and serves its state as JSON so a
browser can draw it. Every fly decision comes with a recording of its brain -
which Kenyon cells fired on each timestep and how the HIT and STAND votes
built up - so the front end can show the circuit deciding, not just the
result.

    python table_server.py            # then open http://localhost:8765
    python table_server.py --learn    # flies keep learning while they play

Endpoints
    GET  /state                 full table state
    POST /join                  take the human seat (body: {"name": "..."})
    POST /leave                 give it back
    POST /action                {"action": "HIT" | "STAND"} on the human's turn
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from blackjack import HIT, STAND, Deck, Observation, card_value, hand_total
from blackjack_fly import make_fly
from connectome import load_circuit
from learning import LearningFly

FLY_NAMES = ["Zizi", "Bzzt", "Dorit"]
PACE = 1.6          # seconds between visible steps
HUMAN_TIMEOUT = 40  # seconds a human gets to act before standing automatically
WEIGHTS_DIR = Path(__file__).parent / "data"


class Seat:
    def __init__(self, name, kind, player=None):
        self.name = name
        self.kind = kind          # "fly" | "human" | "empty"
        self.player = player      # LearningFly for flies
        self.cards = []
        self.status = "idle"      # idle | waiting | thinking | hit | stand | bust | won | lost | push
        self.result = None
        self.chips = 0
        self.brain = None         # last decision's trace
        self.hands = 0

    @property
    def total(self):
        return hand_total(self.cards)[0] if self.cards else 0

    @property
    def soft(self):
        return hand_total(self.cards)[1] if self.cards else False

    def to_json(self):
        return {
            "name": self.name, "kind": self.kind,
            "cards": [repr(c) for c in self.cards],
            "total": self.total, "soft": self.soft,
            "status": self.status, "result": self.result,
            "chips": self.chips, "hands": self.hands,
            "brain": self.brain,
        }


class Table:
    def __init__(self, n_flies=3, learn=False, pace=PACE, pretrained=True):
        self.lock = threading.Lock()
        self.pace = pace
        self.learn = learn
        self.round = 0
        self.phase = "starting"
        self.dealer_cards = []
        self.dealer_hidden = True
        self.message = ""
        self.human_action = None
        self.human_event = threading.Event()

        circuit = load_circuit()
        self.seats = []
        for i in range(n_flies):
            fly = make_fly(circuit, name=FLY_NAMES[i % len(FLY_NAMES)])
            fly.record_trace = True
            learner = LearningFly(fly, learning_rate=0.01, reward_scale=1.0)
            if pretrained:
                self._load_weights(learner, i)
            self.seats.append(Seat(fly.name, "fly", learner))
        self.seats.append(Seat("", "empty"))

    def _load_weights(self, learner, index):
        path = WEIGHTS_DIR / f"fly_{index}.npz"
        if path.exists():
            blob = np.load(path)
            learner.fly.net.weights.data[:] = blob["weights"]
            learner.hands_played = int(blob["hands"])
            learner.expected_outcome = float(blob["expected"])

    def save_weights(self):
        for i, seat in enumerate(s for s in self.seats if s.kind == "fly"):
            np.savez_compressed(WEIGHTS_DIR / f"fly_{i}.npz",
                                weights=seat.player.fly.net.weights.data,
                                hands=seat.player.hands_played,
                                expected=seat.player.expected_outcome)

    # ---- public state --------------------------------------------------

    def state(self):
        with self.lock:
            dealer = [repr(c) for c in self.dealer_cards]
            if self.dealer_hidden and len(dealer) > 1:
                dealer = [dealer[0], "??"]
            return {
                "round": self.round,
                "phase": self.phase,
                "message": self.message,
                "learning": self.learn,
                "dealer": {"cards": dealer,
                           "total": hand_total(self.dealer_cards)[0]
                           if not self.dealer_hidden and self.dealer_cards else None},
                "seats": [s.to_json() for s in self.seats],
                "human_turn": self.phase == "human_turn",
            }

    def join(self, name):
        with self.lock:
            seat = self.seats[-1]
            if seat.kind == "human":
                return False
            # cards and status are left alone: a join can land mid-deal, and
            # clearing them here would wipe a card that was already dealt
            seat.kind, seat.name = "human", (name or "You")[:16]
            seat.chips, seat.hands = 0, 0
            return True

    def leave(self):
        with self.lock:
            seat = self.seats[-1]
            seat.kind, seat.name, seat.cards, seat.status = "empty", "", [], "idle"
        self.human_action = STAND
        self.human_event.set()

    def act(self, action):
        if action not in (HIT, STAND):
            return False
        self.human_action = action
        self.human_event.set()
        return True

    # ---- game loop -----------------------------------------------------

    def run_forever(self):
        while True:
            self.play_round()
            time.sleep(self.pace * 2)

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def play_round(self):
        self.round += 1
        deck = Deck(seed=None)
        with self.lock:
            self.dealer_cards, self.dealer_hidden = [], True
            self.phase, self.message = "dealing", f"סיבוב {self.round}"
            for seat in self.seats:
                seat.cards, seat.status, seat.result, seat.brain = [], "idle", None, None

        active = [s for s in self.seats if s.kind != "empty"]
        for _ in range(2):
            for seat in active:
                with self.lock:
                    seat.cards += deck.deal(1)
                time.sleep(self.pace * 0.25)
            with self.lock:
                self.dealer_cards += deck.deal(1)
            time.sleep(self.pace * 0.25)

        dealer_up = card_value(self.dealer_cards[0])

        for seat in active:
            if seat.total == 21:
                with self.lock:
                    seat.status = "blackjack"
                continue
            if seat.kind == "fly":
                self._fly_turn(seat, deck, dealer_up)
            else:
                self._human_turn(seat, deck)

        self._dealer_turn(deck)
        self._settle(active)
        if self.learn:
            self.save_weights()

    def _fly_turn(self, seat, deck, dealer_up):
        while True:
            with self.lock:
                seat.status, self.phase = "thinking", "fly_turn"
                self.message = f"{seat.name} חושב על {seat.total}..."
            obs = Observation(seat.total, seat.soft, dealer_up)
            action = seat.player.act(obs)
            fly = seat.player.fly
            trace = fly.last_trace
            # votes are scored against each pool's own baseline, exactly as the
            # decision is, so the bars the viewer sees agree with the action
            votes = [{a: (v[a] - fly.baseline[a][0]) / fly.baseline[a][1] for a in v}
                     for v in trace["votes"]]
            with self.lock:
                seat.brain = {"kc_frames": trace["kc_frames"],
                              "votes": votes,
                              "action": action,
                              "total": seat.total, "dealer": dealer_up}
            time.sleep(self.pace * 1.2)
            with self.lock:
                seat.status = "hit" if action == HIT else "stand"
                self.message = f"{seat.name}: {action}"
            if action == STAND:
                time.sleep(self.pace * 0.6)
                return
            with self.lock:
                seat.cards += deck.deal(1)
            time.sleep(self.pace * 0.8)
            if seat.total > 21:
                with self.lock:
                    seat.status = "bust"
                    self.message = f"{seat.name} נשרף עם {seat.total}"
                time.sleep(self.pace * 0.8)
                return

    def _human_turn(self, seat, deck):
        while True:
            with self.lock:
                seat.status, self.phase = "waiting", "human_turn"
                self.message = f"{seat.name}, תורך ({seat.total})"
            self.human_action = None
            self.human_event.clear()
            self.human_event.wait(timeout=HUMAN_TIMEOUT)
            action = self.human_action or STAND
            if seat.kind != "human":
                return
            with self.lock:
                seat.status = "hit" if action == HIT else "stand"
                self.phase = "fly_turn"
            if action == STAND:
                time.sleep(self.pace * 0.4)
                return
            with self.lock:
                seat.cards += deck.deal(1)
            time.sleep(self.pace * 0.6)
            if seat.total > 21:
                with self.lock:
                    seat.status = "bust"
                time.sleep(self.pace * 0.6)
                return

    def _dealer_turn(self, deck):
        with self.lock:
            self.phase, self.dealer_hidden = "dealer", False
            self.message = "הדילר משחק"
        time.sleep(self.pace)
        while hand_total(self.dealer_cards)[0] < 17:
            with self.lock:
                self.dealer_cards += deck.deal(1)
            time.sleep(self.pace * 0.8)

    def _settle(self, active):
        dealer_total = hand_total(self.dealer_cards)[0]
        dealer_bj = dealer_total == 21 and len(self.dealer_cards) == 2
        with self.lock:
            self.phase = "settle"
            for seat in active:
                total = seat.total
                if seat.status == "blackjack":
                    result = 0.0 if dealer_bj else 1.5
                elif total > 21:
                    result = -1.0
                elif dealer_total > 21 or total > dealer_total:
                    result = 1.0
                elif total < dealer_total:
                    result = -1.0
                else:
                    result = 0.0
                seat.result = result
                seat.chips += result
                seat.hands += 1
                seat.status = "won" if result > 0 else "lost" if result < 0 else "push"
                if seat.kind == "fly" and self.learn:
                    seat.player.reward(result)
                elif seat.kind == "fly":
                    seat.player.pending.clear()
            self.message = f"הדילר: {dealer_total}"
        time.sleep(self.pace * 2)


# ---- HTTP ----------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "table"


def make_handler(table):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, name):
            path = STATIC_DIR / name
            if not path.exists():
                self.send_error(404)
                return
            body = path.read_bytes()
            ctype = {"html": "text/html", "js": "application/javascript",
                     "css": "text/css"}.get(path.suffix[1:], "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/state":
                self._json(table.state())
            elif self.path in ("/", "/index.html"):
                self._file("index.html")
            else:
                self._file(self.path.lstrip("/"))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/join":
                self._json({"ok": table.join(body.get("name"))})
            elif self.path == "/leave":
                table.leave()
                self._json({"ok": True})
            elif self.path == "/action":
                self._json({"ok": table.act(body.get("action"))})
            else:
                self.send_error(404)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--flies", type=int, default=3)
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--pace", type=float, default=PACE)
    parser.add_argument("--fresh", action="store_true", help="ignore saved fly weights")
    args = parser.parse_args()

    table = Table(n_flies=args.flies, learn=args.learn, pace=args.pace,
                  pretrained=not args.fresh)
    threading.Thread(target=table.run_forever, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(table))
    print(f"table at http://localhost:{args.port}   (learning {'on' if args.learn else 'off'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
