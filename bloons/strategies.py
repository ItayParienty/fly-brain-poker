"""How hard is the clone?  Simple strategies against the full 50 rounds.

Each strategy buys whenever it can afford the next thing on its list and
puts every tower on the free spot that covers the most track (bot.best_spot).
Three seeds each; "WON n" means it finished round 50 with n lives left.

    python -m bloons.strategies
"""
from bloons import rules as R
from bloons.bot import ScriptedPlayer, best_spot
from bloons.game import Game


def play(policy, seed=0, act_every=10):
    g = Game(seed=seed)
    while not g.over and not g.won:
        policy(g); g.start_round()
        while g.in_round and not g.over:
            g.step()
            if g.frame % act_every == 0:
                policy(g)
    return g


def _upgrade(g, upgrades):
    for kind, which in upgrades:
        for t in g.towers:
            if t.kind == kind and not t.upgrades[which] and g.money >= R.UPGRADES[kind][which][1]:
                g.upgrade(t, which)


def greedy(kinds, upgrades=()):
    """Buy the upgrades, then the first affordable kind in `kinds`, as long as money lasts."""
    def policy(g):
        while True:
            _upgrade(g, upgrades)
            for kind in kinds:
                if g.money >= R.TOWERS[kind]["cost"]:
                    where, _ = best_spot(g, kind)
                    if where:
                        g.place(kind, *where); break
            else:
                return
    return policy


def rotation(kinds, upgrades=()):
    """Buy `kinds` in turn, each as soon as it is affordable."""
    state = {"i": 0}
    def policy(g):
        while True:
            _upgrade(g, upgrades)
            kind = kinds[state["i"] % len(kinds)]
            where, _ = best_spot(g, kind) if g.money >= R.TOWERS[kind]["cost"] else (None, 0)
            if where is None:
                return
            g.place(kind, *where); state["i"] += 1
    return policy


STRATEGIES = {
    "darts only": lambda: greedy(["Dart"]),
    "darts + piercing": lambda: greedy(["Dart"], [("Dart", 0)]),
    "darts + both upgrades": lambda: greedy(["Dart"], [("Dart", 0), ("Dart", 1)]),
    "tacks only": lambda: greedy(["Tack"]),
    "tacks + both upgrades": lambda: greedy(["Tack"], [("Tack", 0), ("Tack", 1)]),
    "dart, tack, dart, ... + piercing": lambda: rotation(["Dart", "Tack"], [("Dart", 0)]),
    "super when affordable, else dart": lambda: greedy(["Super", "Dart"], [("Dart", 0), ("Super", 0)]),
    "the scripted build order (bot.py)": None,
}

if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    for name, make in STRATEGIES.items():
        out = []
        for seed in range(3):
            g = play(ScriptedPlayer().act, seed, act_every=40) if make is None else play(make(), seed)
            out.append(f"WON {g.lives}" if g.won else f"lost in round {g.current_round}")
        print(f"{name:36s} " + ", ".join(out), flush=True)
