# A fly brain at the card table

A spiking simulation of the *Drosophila* mushroom body — the fly's learning
centre, wired as the FlyWire connectome measured it — playing poker and
blackjack, in a 3D room you can sit down in.

Nothing here is trained in the machine-learning sense. The wiring is the
scanned wiring, the plasticity rule is the one the fly already uses, and the
question is what such a circuit does when what it has to judge is a hand of
cards instead of a smell.

<p align="center"><em>python -m cards.table_server --learn → http://localhost:8765</em></p>

## What was found

**The measured wiring produces a stable policy; random wiring produces noise.**
Before any learning, the circuit's behaviour under a fixed encoding is
deterministic and repeatable. Randomising which MBONs drive which action, or
rewiring the Kenyon cell → MBON connections while keeping their number and
weights, turns that into a different policy on every seed, with a spread of
±10–19 points (`cards/innate.py`). The recorded connectivity is what fixes the
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
classic odour-conditioning protocol cleanly (`cards/conditioning.py`: a punished
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
window, `cards/fly.py`) flipped the poker bias to −19.5% and lifted the untrained
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

`flybrain/verify_brain.py` checks the three properties any association depends on.
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

`cards/table_server.py` runs a blackjack table with three flies, a dealer and a
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

## Part two, in progress: an eye, a mouse, and Bloons Tower Defense 1

The cards hand the fly its perception. The second half of the project takes
that away: the fly gets one optic lobe of the connectome, a screen, and a
mouse, and the game is Bloons Tower Defense 1 (Ninja Kiwi, 2007). Nothing
about the game reaches the fly except through its photoreceptors.

**The game** (`bloons/`) is a headless, frame-exact reimplementation. Tower
stats, all 50 rounds, spawn timing, hit boxes and the bloons' paths — frame
by frame — are read from the original's decompiled ActionScript and
geometry (`bloons/extract_original.py`, run on a copy of the game you
supply; it is not in this repo). Against the original running in a browser,
one Dart tower at the same spot leaks 1 bloon in round 1 and 8 in round 2 in
both. The mouse behaves as the original's does, down to the tower-info box
on hover and the "Can't Afford" cover that swallows a press
(`bloons/ui.py`). Simple strategies give it a real difficulty gradient
(`bloons/strategies.py`): each buys whenever it can and puts every
tower where it covers the most track; three seeds each.

| strategy | outcome |
|---|---|
| Darts only | lost in rounds 11, 13, 13 |
| Darts, and every upgrade as soon as affordable | lost in rounds 10–12 |
| Tacks, and every upgrade as soon as affordable | lost in rounds 5–6 |
| Darts with Piercing | won once (13 lives), lost in rounds 13 and 22 |
| the scripted build order in `bloons/bot.py` | won all three, with 6–16 lives |
| **Tacks only** | **won all three, with 11–13 lives** |
| Dart and Tack in turn, with Piercing | won all three, with 10–24 lives |

Winning does not take a clever plan. It takes one kind of tower, placed well,
bought the moment it is affordable — which is exactly what a player who can
see and point quickly can do.

**The eye** (`flybrain/vision.py`, `flybrain/flyvis_eye.py`) is FlyWire's
right optic lobe — 48,882 neurons once the giant CT1 cell is split into its
per-column compartments. FlyWire's atlas places each columnar neuron in one
of the 796 columns of the eye's hexagonal lattice, so every neuron's patch of
the visual field is known; the screen is fitted to that lattice and light
enters at the photoreceptors. The connectome has no time constants or
resting potentials, so those, and a strength per pair of cell types, come
from the published flyvis model (Lappalainen et al. 2024). What the eye does
(`flybrain/verify_vision.py`):

| | |
|---|---|
| ON and OFF pathways | Mi1 and Tm3 rise with light, Tm1 and Tm9 fall — as in the fly |
| an object's position, decoded from L1 + L2 | to 21 px (chance ~180 px) |
| direction selectivity in T4/T5 | absent: DSI ≤ 0.09, every subtype the same direction |
| looming responses in LPLC2/LC4 | absent |

The last two fail although FlyWire's T4 inputs are offset by subtype exactly
as the literature describes — Mi9 on one side, Mi4 and C3 on the other. The
wiring for motion is there; the dynamics are not. flyvis's parameters were
fitted together with its own averaged lattice and do not carry over to
individual cells.

**The hands** (`flybrain/motor.py`, `bloons/fly_player.py`). Three read-outs,
each one weight per cell type and none tied to a place on the screen: the
**gaze** is a salience map over the 796 columns, and the pointer jumps to its
peak in one 12.5 ms step however far that is; the **proboscis** presses when a
sum over the fixated spot crosses zero; a **wing beat** presses Start Round. The
whole loop — screen, photoreceptors, optic lobe, pointer, game — runs at
~220 frames/s for one fly, 5.5× real time. The eye is fed by a sprite
compositor that computes each column's mean colour directly, identical to
rendering the frame and ~100× cheaper (`bloons/screen.py`).

Taught only what a bloon looks like — one weight per cell type, fitted by
regression while the fly watched one game (`bloons/watch.py`) — the fly keeps
its pointer on a bloon in 76% of frames of another game, against 14% for a
random spot. A bloon appearing on the track moves the lamina within 50 ms,
the medulla within 60–75 ms and the gaze map within 62 ms, so the fly's
reaction time is set by its eye rather than by moving a mouse. A lone bloon
on a still screen, though, is often outshone by other high-contrast
things — text above all.

**Is the optic lobe doing the seeing?** Not yet, as far as this measures
(`bloons/eye_vs_pixels.py`). Trained to find bloons, the same read-out does
this well from the optic lobe and from the raw photoreceptors of a
19-column patch:

| read-out | optic lobe | photoreceptors |
|---|---|---|
| one weight per feature | 65% | 58% |
| a small network shared by all columns | 94% | 97% |

The optic lobe's features help a read-out that cannot compute on its own,
and a read-out that can compute does not need them. Whatever the fly
achieves in the game will be measured against the same controls.

## Running it

```bash
pip install -r requirements.txt
python -m flybrain.download_data      # ~50 MB from FlyWire, once

python -m cards.table_server --learn  # the table, at http://localhost:8765

python -m flybrain.connectome         # build the circuit, print its composition
python -m flybrain.verify_brain       # sparseness, separation, reliability, APL
python -m flybrain.calibrate          # sweep the one free parameter

python -m cards.innate                # measured wiring vs randomised (poker)
python -m cards.experiment            # real vs random-sign reward (poker)
python -m cards.blackjack_experiment  # the same, for blackjack
python -m cards.conditioning          # classic odour conditioning, as a positive control

python -m bloons.game                 # the clone against the original: leaks with one Dart tower
python -m bloons.strategies           # simple strategies over all 50 rounds
python -m flybrain.verify_vision      # what the eye does: ON/OFF, objects, position, motion, looming
python -m bloons.watch                # the closed loop: aim, reaction time, speed, and a video
python -m bloons.eye_vs_pixels        # does the optic lobe help find bloons, or would pixels do?
```

Smaller pieces, worth reading first:

```bash
python -m flybrain.demo_neuron        # one neuron charging, leaking, firing
python -m cards.game                  # one scripted heads-up poker hand
python -m cards.blackjack             # basic strategy vs mimic-the-dealer vs always-stand
```

## Layout

```
flybrain/   the brain: FlyWire download, circuits, simulations, plasticity rule, the eye, the read-outs, checks
cards/      poker, blackjack, and the 3D table
bloons/     Bloons Tower Defense 1: the clone, its screen and mouse, and the fly that plays it
data/       FlyWire files and caches (not committed)
```

## Data

Not included here. `download_data.py` fetches the public FlyWire v783 snapshot.
See [CITATION.md](CITATION.md) — CC BY-NC-SA 4.0, non-commercial. Bloons Tower
Defense is Ninja Kiwi's; the clone is an independent research reimplementation,
and the original game file is not included.
