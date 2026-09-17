# A fly brain at the card table

A spiking simulation of the *Drosophila* mushroom body — the fly's learning
centre, wired as the FlyWire connectome measured it — playing poker and
blackjack, in a 3D room you can sit down in.

Nothing here is trained in the machine-learning sense. The wiring is the
scanned wiring, the plasticity rule is the one the fly already uses, and the
question is what such a circuit does when what it has to judge is a hand of
cards instead of a smell.

<p align="center"><em>python table_server.py --learn → http://localhost:8765</em></p>

## What was found

**The measured wiring produces a stable policy; random wiring produces noise.**
Before any learning, the circuit's behaviour under a fixed encoding is
deterministic and repeatable. Randomising which MBONs drive which action, or
rewiring the Kenyon cell → MBON connections while keeping their number and
weights, turns that into a different policy on every seed, with a spread of
±10–19 points (`innate.py`). The recorded connectivity is what fixes the
behaviour.

**But the policy is not knowledge.** In poker the innate circuit raises 19.5%
*more* often with weak hands than strong ones — a bluffing habit that happens
to beat a tight opponent (+0.56 chips/hand) because bluffs work on a player
who folds. In blackjack it agrees with basic strategy on 68% of
frequency-weighted decisions and still hits a hard 18 against most dealer
cards. (A player on 21 is not offered a card — table rule, as in a casino.
Asked, the fly sometimes took one.) Which way the bias points is an accident of which olfactory neurons
the encoder assigned to card strength; only its stability is a property of
the wiring.

**Dopamine-gated depression did not teach it either game.** Training on real
outcomes is compared against training on outcomes with a randomly assigned
sign — the same perturbation of the synapses, carrying no information.

| Game | No plasticity | Random-sign reward | Real reward |
|---|---|---|---|
| Poker, discrimination | −19.5% | −17.0% ±0.2 | −17.0% ±0.5 |
| Poker, chips/hand | +0.558 | +0.404 | +0.379 |
| Blackjack, agreement with basic strategy | 67.7% | 58.4% ±0.2 | 60.8% ±0.1 |
| Blackjack, hits on hard 17–20 | 29.2% | 57.2% ±1.3 | 46.0% ±0.5 |
| Blackjack, return/hand | −0.234 | −0.409 ±0.004 | −0.330 ±0.007 |

In poker the reward's sign makes no difference at all. In blackjack it does —
real reward beats random-sign reward on every measure, by 10× the run-to-run
spread, so the signal carries information — but plasticity as a whole still
leaves the fly worse than it started, and at learning rates low enough not
to damage the innate policy it stops moving at all. The rule disturbs the
circuit more than it instructs it.

### Why blackjack shows a signal and poker does not

The plasticity rule pairs a stimulus with the outcome of a single trial and
depends on repetition to average out noise. A poker situation almost never
recurs — five continuous features and an opponent who reacts — so the noise in
one hand's result never cancels. A blackjack state is a total and a dealer
card; "16 against a 10" comes up dozens of times in a few hundred hands.

What the rule then learns is the obvious part. Hitting a hard 20 busts nine
times in ten and its signature is unmistakable; hard 16 against a 10 loses
about as often either way, and a rule that updates on single outcomes cannot
resolve a two-point difference in expectation. The same code passes the
classic odour-conditioning protocol cleanly (`conditioning.py`: a punished
odour loses 23% of its approach drive, a control odour 1%), so this is the
rule meeting a task it was not built for, not a broken implementation.

Two pieces of biology turned out to be load-bearing rather than decorative.
Depression-only learning is a one-way ratchet and saturates unless synapses
recover — without recovery the fly fell silent and folded everything after
a few hundred hands. And in a game you mostly lose, dopamine has to report
the outcome *relative to expectation*: scoring a routine loss as punishment
punishes whatever the fly does most, correct or not, and the policy flips
instead of converging. Reward prediction error, which is how dopamine is
described in mammals and increasingly in flies, fixed that.

## A result that was retracted

An earlier version of this README reported that the untrained circuit
discriminated strong from weak poker hands by **+21.6%**, with randomised
wiring collapsing to noise, and called it structure in the connectome. That
number was produced by a bug in the encoder: the *K* olfactory neurons
nearest to any feature value near the end of its range were simply the *K*
neurons at that end, so hand strengths above ~0.8 — and blackjack totals 18
through 21 — all drove the identical set of neurons and were
indistinguishable to the circuit. Fixing it (padding the slot axis by half a
window, `fly.py`) flipped the poker bias to −19.5% and lifted the untrained
blackjack fly from 59% to 68% agreement. The stability-versus-noise contrast
survived the fix; the claim about what the bias meant did not. Both the
original numbers and the corrected ones are in the commit history.

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
      MBON        96   grouped by transmitter -> the action
              ^
      DAN      331   dopamine: the outcome, relative to expectation

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

## The table

`table_server.py` runs a blackjack table with three flies, a dealer and a
seat for you, and serves it to a three.js front end. The flies are built from
their anatomy — thorax, striped abdomen, compound eyes, antennae, six jointed
legs, halteres, veined wings — and animated from the game: wings flutter while
one thinks, it hops on a hit, droops on a bust, jumps on a win, and grooms its
front legs when it has nothing to do. Cards are dealt from the shoe and flip
in the air; chips travel between seat and tray when a hand settles.

Every fly decision comes with a recording of the circuit making it: which of
the 5,177 Kenyon cells fired on each of the 60 timesteps, and how the HIT and
STAND votes built up. Click a fly to pin its brain. With `--learn` the flies
keep learning while they play and their synapses are saved between runs.

## What this does not claim

- The connectome gives synapse **counts** and, through neurotransmitter
  prediction, the **sign** of a connection. It does not give absolute
  synaptic strength; `DEFAULT_WEIGHT_SCALE` is calibrated against measured
  Kenyon cell sparseness, and every connectome simulation has a number like it.
- Neurotransmitter identity is itself a prediction, not a measurement.
- Mapping MBONs onto actions by transmitter follows their reported valence,
  but the assignment is an interpretation.
- Hand strength, pot odds and blackjack totals are computed and handed to the
  fly. It is given perception; what it would have to learn is what to do
  about it.

## Running it

```bash
pip install -r requirements.txt
python download_data.py          # ~50 MB from FlyWire, once

python table_server.py --learn   # the table, at http://localhost:8765

python connectome.py             # build the circuit, print its composition
python verify_brain.py           # sparseness, separation, reliability, APL
python calibrate.py              # sweep the one free parameter

python innate.py                 # measured wiring vs randomised (poker)
python experiment.py             # real vs random-sign reward (poker)
python blackjack_experiment.py   # the same, for blackjack
python conditioning.py           # classic odour conditioning, as a positive control
```

Smaller pieces, worth reading first:

```bash
python demo_neuron.py            # one neuron charging, leaking, firing
python game.py                   # one scripted heads-up poker hand
python blackjack.py              # basic strategy vs mimic-the-dealer vs always-stand
```

## Data

Not included here. `download_data.py` fetches the public FlyWire v783 snapshot.
See [CITATION.md](CITATION.md) — CC BY-NC-SA 4.0, non-commercial.
