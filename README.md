# Teaching a fly brain to play poker

A spiking simulation of the *Drosophila* mushroom body — the fly's learning
centre, wired as the FlyWire connectome measured it — sat down at a Texas
Hold'em table.

Nothing is trained here in the machine-learning sense. The wiring is the
scanned wiring, the plasticity rule is the one the fly already uses, and the
question is what such a circuit does when the thing it has to judge is a poker
hand instead of a smell.

Two results, one positive and one negative. The negative one is the more
interesting of the two.

## Result 1: the measured wiring carries poker-relevant structure

Before any learning at all, the circuit raises **21.6% more often with a
strong hand than a weak one**, and beats a tight opponent by 1.32 chips per
hand. That is surprising enough to be suspicious, so the wiring was taken
apart three ways (`innate.py`), each keeping the same sizes and weight
distributions:

| Circuit | Discrimination | Chips/hand |
|---|---|---|
| **Measured connectome** | **+21.6%** | **+1.317** |
| MBONs assigned to actions at random | −11.3% ±19.9% | +0.617 |
| Kenyon cell → MBON connections rewired at random | −10.8% ±13.2% | −0.092 |
| Both | +10.1% ±11.9% | +0.150 |

Every randomisation destroys the effect and replaces a stable number with
noise swinging ±20%. The behaviour comes from the recorded connectivity, not
from the encoding or from the architecture in general.

## Result 2: outcome-driven dopamine learning did not teach it the cards

Training with real hand outcomes was compared against training where the
reward keeps its size but has its sign randomised — the same perturbation, no
information (`experiment.py`, 4 repeats × 1,200 hands):

| Group | Discrimination | Chips/hand |
|---|---|---|
| No plasticity | +21.6% ±0.0% | +1.317 |
| Reward sign randomised | +24.6% ±0.3% | +1.346 |
| Real outcomes | +24.8% ±0.0% | +1.350 |

Real reward beats random reward by **0.2%**, against ±0.3% run-to-run noise.
Plasticity does move the circuit, but the reward signal contributes nothing:
almost all of the change is the circuit being disturbed rather than taught.

An earlier version of this project would have reported the first number in
that table against the last and claimed a fly had learned poker.

### Why not — and the check that rules out a broken implementation

The same plasticity code was given the task it evolved for: pair one odour
with punishment, leave another alone (`conditioning.py`, the Tully & Quinn
protocol).

```
punished odour   approach drive  −23.3%
control odour    approach drive   −0.8%
```

It learns, and the learning is specific to the punished stimulus. The
mechanism is intact; poker is what defeats it. Fly conditioning pairs a
stimulus with a *deterministic* outcome — this odour always precedes sugar.
A poker hand does not work that way: the identical decision wins or loses
depending on cards still to come, so the teaching signal is mostly variance.
Depression gated on a single trial's outcome has no way to average that out,
and 5,000 hands did not help either.

## Things that had to be right first

Most of the work was finding out what the circuit needs in order to represent
anything at all. Three bugs, each with a measurable signature:

**Saturation.** Depression-only learning is a one-way ratchet: every hand
weakens something and nothing recovers. Profit peaked at +0.98 chips/hand
around hand 200, then decayed to +0.08 by hand 600 with 53% of synapses
pinned at the floor — the fly went quiet and folded everything. Flies avoid
this by forgetting, and adding synaptic recovery turned the ratchet into a
stable balance.

**Saturation, again, at the input.** A Gaussian tuning curve leaks current
into its tails. Across five features that was enough to drive 62% of Kenyon
cells instead of the calibrated 8%, and the overlap between pocket aces and
seven-deuce reached 91% — every situation looked alike.

**No similarity gradient.** With one neuron per value, hand strength 0.10 and
0.20 shared 38.6% of their Kenyon cells, and 0.10 and 0.90 shared 37.3% —
i.e. nearby hands were exactly as different as opposite ones, so "a strong
hand" could never form as a category. Widening the tuning so each value spans
a band of neurons produced a real gradient: 100% overlap at a distance of
0.05, 36% at 0.4, 24% at 0.8. This single fix is what produced Result 1.

## The circuit

```
    game state, injected as an "odour"
              |
   olfactory  |  2,281   one band of neurons per feature
              v
      ALPN       685   relay
              v
  Kenyon cells  5,177   each situation lights a sparse, distinct subset
              |
              |  <-- 38,915 connections.  The only ones plasticity touches.
              v
      MBON        96   grouped by transmitter -> raise / fold / call
              ^
      DAN      331   dopamine: the outcome of the hand
              
      APL        2   one giant inhibitory neuron per hemisphere
```

`verify_brain.py` checks the three properties any association depends on.
These are simulation outputs, not quotations:

| Property | Measured | Expected |
|---|---|---|
| Kenyon cell sparseness | 7.9% active | 5–10% in living flies |
| Different stimuli overlap | 14.1% | low = distinguishable |
| Same stimulus repeated | 100% | must be stable |

**APL falls out of the data.** Asking the connectome what inhibits Kenyon
cells returns two neurons with 52,518 and 43,151 synapses onto them, one per
hemisphere; FlyWire's annotators label them `APL-RHS` and `APL-LHS`. Silencing
them raises Kenyon cell activity from 7.9% to 21.7% — a 2.7× loss of
sparseness, which is the role the literature assigns them.

## What this does not claim

- The connectome gives synapse **counts** and, through neurotransmitter
  prediction, the **sign** of a connection. It does not give absolute synaptic
  strength; `DEFAULT_WEIGHT_SCALE` is calibrated against measured Kenyon cell
  sparseness, and every connectome simulation has a number like it.
- Neurotransmitter identity is itself a prediction, not a measurement.
- Mapping cholinergic / glutamatergic / GABAergic MBONs onto raise / fold /
  call follows their reported valence, but the assignment is an
  interpretation.
- Hand strength and pot odds are computed and handed to the fly. It is given
  perception; what it would have to learn is what to do about it.
- Result 1 says the measured wiring produces a useful bias under this
  encoding. It does not say the fly understands poker.

## Running it

```bash
pip install -r requirements.txt
python download_data.py     # ~50 MB from FlyWire, once

python connectome.py        # build the circuit, print its composition
python verify_brain.py      # sparseness, separation, reliability, APL
python calibrate.py         # sweep the one free parameter

python baseline.py          # how an untrained fly plays
python innate.py            # Result 1: measured wiring vs randomised
python experiment.py        # Result 2: real reward vs randomised reward
python conditioning.py      # the classic odour protocol, as a positive control
```

Smaller pieces, worth reading first:

```bash
python demo_neuron.py       # one neuron charging, leaking, firing
python game.py              # one scripted heads-up hand
```

## Status

- [x] Load the FlyWire connectome, extract the mushroom body circuit
- [x] Vectorised leaky integrate-and-fire simulation, several flies in parallel
- [x] Calibrate to biological sparseness; verify separation and reliability
- [x] Texas Hold'em engine (heads-up, fixed raise sizes)
- [x] Encode the table as an odour, decode MBONs into an action
- [x] Dopamine-gated plasticity, with the controls to test whether it teaches
- [ ] A task whose feedback is deterministic enough for this rule to learn from
- [ ] Multi-way table
- [ ] Visualisation, and a seat for a human player

## Data

Not included here. `download_data.py` fetches the public FlyWire v783 snapshot.
See [CITATION.md](CITATION.md) — CC BY-NC-SA 4.0, non-commercial.
