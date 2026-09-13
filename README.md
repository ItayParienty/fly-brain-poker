# Teaching a fly brain to play poker

A spiking simulation of the *Drosophila* mushroom body — the fly's actual
learning centre, wired exactly as the FlyWire connectome measured it — hooked
up to a poker table. The flies play for food. Dopamine neurons fire when they
win, and rewire the circuit accordingly.

No neural network is trained here in the machine-learning sense. The wiring
is the real scanned wiring, and the learning rule is the one the fly already
uses.

## Why a fly can learn poker at all

The classic *Drosophila* experiment is simple: present an odour, follow it
with sugar, and the fly learns to approach that odour. Four cell populations
do the work, and all of them are in the connectome:

```
    game state, injected as an "odour"
              |
   olfactory  |  2,281   sensory neurons
              v
      ALPN       685   relay station
              v
  Kenyon cells  5,177   each situation lights up a small, distinct subset
              |
              |  <-- 38,915 connections.  These are the ones that change.
              v
      MBON        96   opposing approach / avoid drives -> the decision
              ^
      DAN      331   dopamine: fires on reward, rewrites the arrow above
```

Learning happens at the Kenyon cell → MBON synapses, and only there. When a
dopamine neuron fires while a Kenyon cell is active, that cell's synapse onto
the MBON is *depressed*. The fly does not learn what to do; it learns what to
stop doing, and the surviving balance between opposing MBONs is the decision.

## The circuit is real, and it behaves

`verify_brain.py` checks three properties that have to hold before any
association can be learned. Numbers below are produced by the simulation, not
quoted from a paper:

| Property | Measured | Expected |
|---|---|---|
| Kenyon cell sparseness | **7.9%** active per stimulus | 5–10% in living flies |
| Separation between different stimuli | **14.1%** overlap | low = distinguishable |
| Reliability of the same stimulus | **100%** overlap | must be stable |

The one free parameter — how much current a synapse delivers, which no
connectome measures — was fixed by sweeping it until sparseness landed in the
biological range (`calibrate.py`). Everything else comes from the data.

### APL falls out of the data

The mushroom body needs strong inhibition to keep its code sparse. Searching
the connectome for whatever inhibits Kenyon cells returns two neurons making
**52,518** and **43,151** synapses onto them — one per hemisphere. FlyWire's
human annotators label them `APL-RHS` and `APL-LHS`.

Silencing them in simulation raises Kenyon cell activity from **7.9% to
21.7%**, a 2.7× loss of sparseness — the circuit stops being able to tell
situations apart, exactly as the literature describes.

## What this project does *not* claim

- The connectome gives synapse **counts** and, via neurotransmitter
  prediction, the **sign** of each connection. It does not give absolute
  synaptic strength. `DEFAULT_WEIGHT_SCALE` is a calibrated guess, and every
  connectome simulation has one.
- Neurotransmitter identity is itself a machine-learning prediction, not a
  measurement.
- A fly brain has no reason to be good at poker. Whether it becomes good is
  the experiment, not the assumption.

## Running it

```bash
pip install -r requirements.txt
python download_data.py      # ~50 MB from FlyWire, once
python connectome.py         # builds the circuit, prints its composition
python verify_brain.py       # the three checks above
python calibrate.py          # sweeps the one free parameter
```

Understanding the neuron model first:

```bash
python demo_neuron.py        # one neuron charging, leaking and firing
```

The poker engine runs standalone too:

```bash
python game.py               # one scripted heads-up hand
```

## Status

- [x] Load the FlyWire connectome, extract the mushroom body circuit
- [x] Vectorised leaky integrate-and-fire simulation (multiple flies in parallel)
- [x] Calibrate to biological sparseness, verify separation and reliability
- [x] Texas Hold'em engine (heads-up, fixed raise sizes)
- [ ] Encode game state as an odour; decode MBON balance into an action
- [ ] Dopamine-gated learning: winning a pot is food
- [ ] Multi-way table
- [ ] Visualisation, and a seat for a human player

## Data

Not included in this repository. See [CITATION.md](CITATION.md) — FlyWire data
is CC BY-NC-SA 4.0, non-commercial.
