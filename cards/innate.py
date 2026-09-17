"""Does the measured wiring matter, or would any network do?

Before it learns anything, the circuit already raises more often with strong
hands than weak ones. That is worth taking seriously rather than celebrating:
it could come from the connectome, or it could be an artefact of how the
poker table was encoded, in which case any randomly wired network would do as
well and the connectome would be decoration.

Four versions of the same fly are compared:

  measured      - the connectome as FlyWire recorded it
  shuffled pools- MBONs assigned to actions at random instead of by the
                  transmitter they release, keeping group sizes identical
  rewired       - Kenyon cell to MBON connections randomly reassigned, keeping
                  the number of connections and the distribution of weights
  both          - shuffled pools and rewired

If the measured version is no better than the rest, the wiring is not what is
producing the behaviour.
"""

import numpy as np

from flybrain.connectome import load_circuit
from cards.control import discrimination
from cards.fly import Fly
from cards.opponents import TightOpponent
from cards.train import evaluate


def shuffle_pools(fly, rng):
    """Reassigns MBONs to action pools at random, preserving pool sizes."""
    all_mbons = np.concatenate([idx for idx in fly.pools.values()])
    shuffled = rng.permutation(all_mbons)
    out, start = {}, 0
    for action, idx in fly.pools.items():
        out[action] = np.sort(shuffled[start:start + len(idx)])
        start += len(idx)
    fly.pools = out
    return fly


def rewire_kc_to_mbon(fly, rng):
    """Randomly reassigns which Kenyon cell feeds which MBON, keeping the
    number of connections and the multiset of weights unchanged."""

    weights = fly.net.weights.tolil(copy=True)
    kc = fly.circuit.indices_of("Kenyon_Cell")
    mbon = np.concatenate([idx for idx in fly.pools.values()])
    kc_set = set(kc.tolist())

    for row in mbon:
        columns = np.array(weights.rows[row])
        values = np.array(weights.data[row])
        if len(columns) == 0:
            continue
        from_kc = np.array([c in kc_set for c in columns])
        if not from_kc.any():
            continue
        replacement = rng.choice(kc, size=int(from_kc.sum()), replace=False)
        keep_cols = columns[~from_kc].tolist()
        keep_vals = values[~from_kc].tolist()
        new_cols = keep_cols + replacement.tolist()
        new_vals = keep_vals + rng.permutation(values[from_kc]).tolist()
        order = np.argsort(new_cols)
        weights.rows[row] = [new_cols[i] for i in order]
        weights.data[row] = [new_vals[i] for i in order]

    fly.net.weights = weights.tocsr().astype(np.float32)
    return fly


def build(variant, circuit, rng):
    fly = Fly(circuit, name=variant, calibrate=False)
    if variant in ("shuffled pools", "both"):
        fly = shuffle_pools(fly, rng)
    if variant in ("rewired", "both"):
        fly = rewire_kc_to_mbon(fly, rng)
    fly._calibrate_baseline()
    return fly


def main(repeats=3):
    circuit = load_circuit()
    opponent = TightOpponent()
    variants = ["measured", "shuffled pools", "rewired", "both"]
    labels = {"measured": "החיווט הנמדד", "shuffled pools": "קבוצות מעורבבות",
              "rewired": "חיווט אקראי", "both": "שניהם"}

    print(f"אבחנה בין ידיים לפני כל למידה, {repeats} חזרות\n")
    print(f"  {'גרסה':<18}{'אבחנה':>10}{'צ׳יפים':>10}")
    print("  " + "-" * 38)

    for variant in variants:
        discs, chips = [], []
        for repeat in range(repeats):
            rng = np.random.default_rng(repeat)
            fly = build(variant, circuit, rng)
            score, log = evaluate(fly, opponent)
            discs.append(discrimination(log))
            chips.append(score)
            if variant == "measured":
                break  # deterministic, no need to repeat
        print(f"  {labels[variant]:<18}{np.mean(discs):>+9.1%}{np.mean(chips):>+10.3f}"
              + (f"  ±{np.std(discs):.1%}" if len(discs) > 1 else ""))


if __name__ == "__main__":
    from flybrain import console_utf8  # noqa: F401
    main()
