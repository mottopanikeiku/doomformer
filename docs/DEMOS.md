# Demos, results and the fine print

The code behind every demo is in [`demos/`](../demos), [`training/`](../training) and [`models/`](../models). The films
are on the [v0.1.0 release](https://github.com/mottopanikeiku/doomformer/releases/tag/v0.1.0).

## The fine print: what DOOM computes vs what Python does

The forward-pass arithmetic of an integer transformer runs on ViZDoom game mechanics. Voodoo dolls, gates and linedef
specials implement quantised matrix products and staircase activations. Python compiles the maps, transfers tile outputs
between games, supplies the inputs and samples characters. DOOM-generated readbacks are compared exactly (tolerance 0)
with a separate integer reference. The name model was trained on 11,341 of the 12,934 cleaned names. A Python procedural
generator builds a playable level from a generated name.

This is a demonstration of computing with game mechanics. It is slow (see [Speed](HOW_IT_WORKS.md#speed)). The
transformers use integer attention: ReLU-style attention in every demo below, softmax attention computed by the dolls in
`doomnames_softmax` (`demo --softmax`), `doomnames_onemap_softmax` (`demo --one-map --softmax`) and `doomnames_modern`
(which adds an integer RMSNorm and a gated MLP). None has LayerNorm; they are no drop-in float transformers. The engine is
ViZDoom 1.3.2's ZDoom engine with Freedoom2 assets (not vanilla 1993 DOOM). Transformer weights are trained offline in
PyTorch with quantisation-aware training. The perceptron demos update their weights inside DOOM, and the fly
circuits use scaled connectome weights. The transformer and perceptron marine demos imitate scripted experts from
privileged scene features. None of the demos is reward-driven RL or learning from pixels. Selected demonstration seeds
are listed in [Seeds and selection](#seeds-and-selection). Film captions abbreviate the results below with the same scope.

"Bit-exact" below means integer equality (no tolerance) between values read back from the running game and a separately
written integer reference. The values are sector light and floor and ceiling heights, exposed through ACS into ViZDoom
game variables. Each comparison covers one run and says nothing about inputs that were not run. Each negative control in
the repository is a variant that must fail the same comparison.

| Demo | DOOM computes | Python does | Topology | The bit-exact check compares |
|---|---|---|---|---|
| **Transformer name generator** (`doomformer/orchestrator.py`, `models/doombrain`, `doomnames`) | every tile's integer arithmetic: embedding lookup, q/k/v crossbars, q·k and attention-weighted products, output projection, MLP, unembedding to logits, staircase activations | compiles the tile maps and ACS (106 for `doombrain`, 38 for `doomnames`); copies tile outputs to the next tile's inputs (the bus); supplies the prompt; samples each letter from the logits (seeded, mirrored by the reference); schedules tiles; closes ablation doors; trains offline | **pool of games** (each tile instance is a logical map instance; live games are reused and evicted) | every exposed tile-output tensor and its negation vs `intref.forward` on the same tokens, and the generated string vs a reference generation. Thresholded activations, doll counts and partial sums are not compared |
| **Marine policy** (`doomformer/policy.py`, `models/doomplayer`, `doomplayer_xl`) | a 3-token transformer pass over (start, bearing, distance) → action logits | bins exact ViZDoom object positions into tokens (privileged); argmax over the action tokens → buttons, held 4 tics and released while the brain computes; offline training on aimbot labels. The 100-episode statistics use the NumPy reference | **pool of games** (12 workers) | tile tensors, negation twins, logits and chosen action vs `intref` on real in-game contexts (3 in the DOOM check, every decision in a filmed run) |
| **Walker** (`demos/walker.py`, `models/doomwalker`) | a 1-layer, d=8 transformer pass over 5 tokens → action logits | builds the level (`levelgen.py`); computes the tokens: bearing to a look-ahead waypoint on a precomputed route (privileged compass), enemy bearing/distance from ViZDoom objects, flags; argmax → buttons; offline training on a scripted expert. The held-out-level statistics use the NumPy reference | **pool of games** (55 games, 16 maps in the film run) | every exposed tile-output tensor and negation twin, logits included, vs `intref`, every decision of the film run (102/102; 120/120 lobotomised). The host maps the logits to the action deterministically |
| **One-game learner** (`demos/arena.py`, `demos/aimlearn.py`) | scores from 38 ceiling-height weights; the "act" dolls choosing the action from the pre-update scores; the perceptron update that rewrites the ceilings (persistent); the arena itself (demons, buddha mode, lives) | bins what the marine sees into 19 binary features; supplies aimbot labels (privileged exact positions); freezes and unfreezes demons (`Thing_Deactivate`); lives bookkeeping; video | **one game** (arena and brain in one map; the dolls are the marine's own voodoo dolls) | both scores, the action and all 38 weights vs a separately advanced NumPy perceptron, every decision |
| **His brain is the level** (`demos/brainlevel.py`, `doomformer/brainacs.py`) | the whole policy transformer (`models/doomplayer_onemap`) in the arena's own map, every decision, **and the choice of action**: two comparator layers of dolls take the exact argmax of the three action logits (lowest index on ties), and the winner's doll hands it to the level. Also the arena and the ACS level rules: the clock (freeze, sense, write the token ports, release the dolls, wait a fixed 4,160-tic budget), the 3 sensed tokens from actor positions (the bins of `policy.observe`, privileged exact positions), storing the dolls' action, relative turns, and a script-triggered pistol fired from his current facing | builds the map; during the run only advances tics and runs readout scripts (710 in the filmed run, 0 others); recomputes `policy.observe` from the pose sampled at sensing time and the frozen demons; checks; films | **one game** | the sensed tokens vs `policy.observe`; all 540 exposed values (525 transformer values and 15 comparator values: pairwise differences, win counts, one-hot winner), the logits and the action the level received vs `intref` and a NumPy twin of the comparator, every decision (50/50) |
| **Learn to aim** (`demos/aimlearn.py`, `doomformer/learn.py`) | the same in-map perceptron (scores, act dolls, update) | features, aimbot labels, and copying the chosen action into the arena | **two games**: an arena game and a learner-map game | both scores, the action and all 38 weights vs a NumPy twin, every decision |
| **Pure MLP** (`doomformer/pure.py`) | both layers of a 12→16→6 MLP in one map, with no Python between layers (a clock doll starts layer 2) | writes x and −x, reads the results; random weights and inputs; NumPy reference | **one game** | the 6 final outputs, the 16 hidden gate-floor heights layer 2 reads (plane heights read back via ACS) and the 16 hidden light outputs, every case |
| **Random spiking network** (`doomformer/spiking.py`) | the whole spiking step: threshold, reset, leak, synapses. Potentials are floor heights and persist in the map | writes the input spikes each step; converts the negated-potential readback; NumPy reference | **one game** | every neuron's spike and next potential, every step |
| **Fly marine** (`demos/fly.py`) | the same spiking step for a 200-neuron FlyWire-derived circuit, during the stimulus and while any neuron is still going to spike ("afterglow") | builds the corridor; scripted walking; synthetic taste-neuron spike trains from each snack's preassigned label; decides eat/refuse at 3 spikes summed over the two MN9 outputs; scripted eat/avoid motion; predicts whether more spikes will follow; **fast-forwards the silent leak decay between snacks and writes the resulting state into the map** | **one game** (corridor and circuit in one map) | active and afterglow steps only (219 of the 1,578 model steps of the seed-1 run). The reference computes the 1,359 rest steps |
| **The fly is alive** (`demos/flyloop.py`) | the whole loop: an in-map ACS clock rewinds, senses (zones, item presence, line of sight to the demon), draws every input spike with an in-map Park–Miller PRNG, runs the 255-neuron circuit and holds. Output spike dolls cross `ThrustThing`/`ThrustThingZ`/quake lines aimed at the marine's TID; conveyors and timed gates (ACS) move him between stations | builds the map; calibrates the step period on a separate twin game before the run; during the run only advances tics and sends readout scripts (every console command is counted: 0 others); replicates the PRNG and the circuit for checking; films | **one game** | every neuron's spike and potential every step (169/169), the PRNG state each step, and that the action lines fired exactly when the output neurons spiked |
| **Fly marine, day scenario** (`demos/fly.py`, `models/fly/brain.json`) | the same spiking step for the 255-neuron combined circuit | as above, plus dust puffs and synthetic Johnston's-organ trains at the dust patch; wakes the demon, freezes it at 200 units and supplies synthetic LC4 "looming" trains; decides groom/takeoff from aDN1/Giant Fiber spike counts; the takeoff is an ACS `ThrustThing`/`ThrustThingZ` on him | **one game** | active and afterglow steps (253 in the seed-1 run). The reference computes the silent decay between stations |

![the DOOM brain writing DEMON CANYON: a wall of its tile instances, each a DOOM game whose voodoo dolls compute part of the forward pass](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/brain.png)

**Trailer: [`trilogy.mp4`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/trilogy.mp4)** (48 s),
*DOOM Dreams of DOOM*. The name generator writes DEMON CANYON, a Python generator builds the level from the name, and a
marine with a 55-game DOOM-executed policy plays it. Halfway along the route his brain is shot: 3 neurons (`x0` units 1,
2, 5) are closed as doors in every consuming tile from decision 50 on. He keeps veering and turning right for the rest of
the film. All 120 decisions are bit-exact against the equally lobotomised reference. The lesion scan covers every single
neuron and 300 random blasts on all 5,848 observations the walker can see (`walker.all_contexts`). In a 170-decision
replay of seed 118 this lesion never reaches the exit (route progress 0.54). `x0 [1]` alone gets him to the end of the
route, but he never presses USE there and dies at decision 147. Both lesions were found by search (see
[Seeds and selection](#seeds-and-selection)).

## HIS BRAIN IS THE LEVEL

**[`brain_is_the_level_raw.mp4`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/brain_is_the_level_raw.mp4)**
(53 s, the thinking is time-lapsed): the marine of [ONE DOOM GAME](#the-other-demos), with a **transformer** brain
compiled into the map he fights in (`demos/brainlevel.py`, `doomformer/brainacs.py`). The dolls that compute it are
voodoo dolls of himself. Two stages:

| | Stage A: host senses and acts | Stage B: the level senses and acts, the dolls decide |
|---|---|---|
| sensing | Python `policy.observe` → token ports | ACS bins the nearest demon's bearing (17 bins) and distance (5 bins) from actor positions |
| transformer | dolls, whole forward pass, one map | dolls, whole forward pass, one map |
| action | Python argmax → buttons (4 tics) | dolls: exact argmax of the three action logits; the winner's doll hands it to ACS, which turns him (relative turns per tic) or triggers his pistol |
| decisions, kills (seed 4, first 50 decisions) | 50/50 exact, 5 kills | **50/50 exact, 5 kills** (5 shots, 26 → 21 bullets) |
| clock, wall time per decision | 3,672 tics per decision call, 31.8 s | 4,160-tic compute budget (the dolls settle in 3,824), 4,205 tics per full clock cycle; 35.3 s per decision call in the filmed run with time-lapse sampling |

**The dolls choose the action.** After the unembedding, two more layers of dolls take the argmax over L, R, F at the
last position. For each pair a < b, the unembedding also writes d_ab = logit_a − logit_b + 1 into a gate floor. These
are the same activations with weights Wun_a − Wun_b, so no logit is re-read or re-quantised. `doomplayer_onemap`'s Wun
lies in [−78, 82] and the pairwise differences in [−71, 139] ≤ 255; the worst port bound is 1,101 ≤ 12,000. A
one-threshold gate passes a doll iff d_ab ≥ 1, i.e. logit_a ≥ logit_b. The dolls count wins_a = Σ_{b>a} [s_a ≥ s_b] −
Σ_{b<a} [s_b ≥ s_a] + a, and a gate at 2 opens for exactly one action: the largest logit, the lower index on ties (as
`np.argmax`). That doll crosses `ACS_ExecuteAlways(213, 0, action)`, and script 213 only stores the number. The clock
compares no logits and reads no sector. It waits out the same 4,160-tic budget, then turns him or fires
(`doomformer/onemap.py`: `build_onemap(..., choose=...)`).

The map holds the arena and 1,345 dolls (1,337 for the transformer, 6 comparator dolls, 2 clock dolls) in 37,410
sectors (37,109 without the comparator). In the filmed run it builds in 1.4 s and loads in 0.7 s. A clock-free twin
settles in 3,824 tics on all 24 distinct contexts of the film (3,656 without the comparator), inside the unchanged
4,160-tic budget. While he thinks, every demon is dormant (`Thing_Deactivate`), so this is no real-time play and the film
time-lapses the thinking. The board shows the dolls' positions read every 128 tics (white = moved since the last sample).

In stage B, Python sends only readout scripts: 710 console commands in the filmed run (260 status reads and 9 readout
pages × 50, 0 others). It presses no buttons, and the readout scripts copy values into ACS variables that nothing in the
map reads. Each decision is checked:

- The sensed tokens are recomputed with `policy.observe` from the player pose that the map recorded at sensing time and
  the dormant demons' positions. This checks the selection and binning but not the map's own pose capture.
- All 540 exposed values are compared with `intref` and a NumPy twin of the comparator: 525 transformer values and the 15
  comparator values (d_ab and their negations, the win counts and their negations, the one-hot winner).
- The action the level received must equal the dolls' winner and `np.argmax` of the reference logits.

With the comparator, the filmed run matches a run in which ACS took the argmax in all 50 decisions: tokens, logits,
actions and every instrumented clock sample (pose, timer, step). No decision had tied action logits. Tamper tests
(`tests/research/test_tamper_stage_b.py`) flip one comparator linedef's sign in the compiled map (no winner, so the check
stops at the one-hot light) and make a winner line hand over the wrong action. Both are caught.

**The model.** The original `models/doomplayer` (d=16, 2 heads) also fits one map: 8,825 dolls, 528,366 sectors, built
in 20 s, loaded in 41 s. One forward pass in it is exact (all 969 exposed values and the action). It takes 29,088 tics and
1,861 s, which is 31 minutes per decision. That exceeds the former 20,000-tic cap. ViZDoom also lists at most 4,096
objects, so an object-list settle check saw fewer than half of the dolls and settled falsely. Both are fixed (see
[Engine facts](HOW_IT_WORKS.md#engine-facts)). So a policy
was purpose-trained for one map: `models/doomplayer_onemap`, L=1, d=8, H=1, MLP 16, 2-level staircases, the same aimbot
data and observation (`training/train_policy.py --L 1 --d 8 --dm 16 --K 2`). Held-out agreement with the aimbot is
96.7 % (original 97.2 %). Over 100 NumPy episodes (seeds 9000–9099) it gets **23.24 ± 0.46 kills per life**, against
23.35 ± 0.35 for the original and 24.50 ± 0.33 for the aimbot. The observation has only 86 distinct values, so this is a
small policy. The point is where it runs.

**Level design, not learned:** the sensing rule (privileged exact positions, binned like the host version), the argmax
(computed by dolls, but by fixed comparator layers), the turn size and the pistol. Stage B fires a script-triggered
pistol from his current facing: one 5-damage bullet with no spread and one bullet of ammo, never aimed at a chosen demon.
The stock pistol's random spread is not used. A demon's bite can slide him between sensing and a later pose read. The
first stage-B film run stopped at decision 31 with a token mismatch for that reason, so the check now reads the pose
recorded at sensing time. The physics is unchanged and the same seed was rerun. Probe maps cover turning and firing from
ACS, kill credit, repeatability and in-map sensing against `policy.observe`. Bearings within 0.0001° of a bin edge, or
near-ties between two demons, can select differently at ACS fixed-point precision, and any such case stops the run.

## The other demos

**[`one_game.mp4`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/one_game.mp4)** (44 s):
**ONE DOOM GAME.** The marine, his arena and his brain are one map in one DOOM game. The dolls that compute his choices
and learn from the aimbot are his own voodoo dolls: 156 copies of himself elsewhere in the same level. (DOOM makes the
last player-1 start the player and every earlier one a voodoo doll of him.) While the brain runs, demons are deactivated
(`Thing_Deactivate`) and the spawner is paused, and the engine keeps advancing tics. Deaths are lives in buddha mode. A
life ends at 1 health or when his 26 bullets are spent, so the game and his brain are never reloaded. His brain starts
blank. Kills per life in the film run: **4 → 24 → 26 → 22 → 24**. The current map compiler gives 4 → 22 → 15 → 20 → 16 for the same seed (see [Run it yourself](#run-it-yourself)). In the same arena and seed, the aimbot that supplies the labels gets 9,
26, 26, 25, 26, and a blank brain 0 every life (trajectories diverge once the actions differ). **931/931 decisions
bit-exact** in the film run (815/815 with the current compiler): both scores, his action and all 38 weights. One seed (4) is reported.

**Learn to aim: DOOM performs online supervised perceptron updates from an aimbot.** The arena marine starts with a blank
brain: 38 weights (19 binary features × 2 heads), all 0. He spins left and never fires (0 kills). His brain is one DOOM
map in which every weight is a ceiling height. Every decision, voodoo dolls compute two scores (fire? and turn
left/right?) from those ceilings, and two "act" dolls read the scores and choose his action. A second layer of dolls in
the same map then applies the perceptron rule against the aimbot's label. If the answer was wrong, they cross
`Ceiling_RaiseByValue`/`LowerByValue` lines under that input's weights. A cap gate limits stored weights to ±100, and the
scores read them clipped to ±3 in this task. The map is never reloaded, so the ceilings remember. The labels come from
privileged binned positions. Kills per episode: **5 → 18 → 24** (the aimbot baseline on the same episodes: 26, 25, 25).
**905/905 decisions bit-exact** against a NumPy twin: both scores, the action and all 38 weights. Two more seeds in DOOM:
22 → 22 → 22 kills (1,207/1,207 exact) and 6 → 25 → 23 (1,053/1,053). Python bins what he sees and supplies the aimbot's
labels; the dolls choose, add and learn. The learner alone has three negative-control variants (`rule`, `land`,
`freeze`) that must fail and do: 0/40, 1/40 and 31/40 examples exact.

**Pure DOOM:** whole MLPs also run in one map with no Python bus between layers (`doomformer/pure.py`). Layer 1's dolls
leave their sums as floor heights, and a clock doll drops layer 2's start barriers once layer 1 is done. A 12→16→6 MLP
is **40/40 cases exact on the final outputs and on the hidden layer as layer 2 reads it** (its 16 gate-floor heights,
read back from the game). The serialization is required: with `pure.SERIAL = 0` a run was 0/10 exact (outside the test
suite).

### One-map transformer

**[`onemap_shil_raw.mp4`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/onemap_shil_raw.mp4)**
(58 s, time-lapse): the whole map from above while the trained model writes SHIL. Every dot is one of its 3,548 voodoo
dolls, read from the game state every 8 tics, marching layer by layer. 5 forward passes; 2,680/2,680 exposed values and
every logit are bit-exact.

`doomformer/onemap.py` lowers a whole `intref` transformer forward pass into the layers of one pure map: embedding by
equality lanes, q/k/v crossbars, runtime × runtime attention products, attention-weighted sums, output projection,
residuals, MLP and unembedding. Python writes the token ids (and their negations), waits until every doll has stopped,
reads the logits and samples the next letter exactly as the reference does. At the default `opt` levels 0 to 2, no Python
and no ACS arithmetic run between layers (at `opt` 3 an ACS script copies each layer's light levels into the next layer's
gate heights; see [scheduling.md](scheduling.md)). Dolls that would hit the same sector in the same tic are spread over
time: the layer's clock doll releases them in **time-multiplexed waves**. Long layers are folded into rows.

| model | dolls | sectors | build / load | one forward pass | verified |
|---|---|---|---|---|---|
| tiny random transformer (L=1, d=4, H=1, T=3) | 593 | 12,620 | 0.4 s / 0.7 s | 1,810 tics, 4.6 s | 20 prompts + generated sequence, 3,347 exposed values exact |
| `models/doomnames_onemap`, trained (L=1, d=8, H=1, MLP 16, T=6) | 3,548 | 95,448 | 3.5 s / 11.1 s | 3,511–3,513 tics, ~84 s | 20 prompts + generated name **SHIL** (seed 17, temperature 12.5) = `intref.generate`; 12,142 exposed values exact (every exposed preactivation, attention score, logit and negation at the active positions) |

The trained model is purpose-built for one map. It is trained with `training/train_brain.py` on the 416 real level names
of at most 5 characters (328 train, 46 dev, 42 reserved) and reaches 25.5 % dev next-character accuracy. That is a much
smaller task than the 106-map `doombrain`. SHIL is not in the 12,934-name corpus. The original `doomnames` and
`doombrain` do not fit one map with this compiler. The planner counts 116,968 dolls and 15.2 million sectors for
`doomnames` (957,662 dolls for `doombrain`), mostly the long staircases that their trained activation thresholds need.
`tests/test_onemap.py` runs the tiny transformer in DOOM. A test releases two conflicting writes in one wave and must
catch it.

**[`doom_beats_doom.mp4`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/doom_beats_doom.mp4)**
(37 s): **a DOOM-generated name, a Python-built level, a DOOM-executed policy.** DEMON CANYON is a name written by the
DOOM brain. The Python procedural generator `demos/levelgen.py` built the level (geometry, theme and route from a hash of
the name and `random.Random`), and it is played in DOOM. The marine's brain is 55 DOOM games (`models/doomwalker`: 1
layer, d=8, 1,242 integer parameters including thresholds, trained offline on 170 *other* generated levels). He walks,
veers, opens doors and shoots twice in 102 decisions, all 102 bit-exact. With the NumPy integer reference policy he
reaches the exit in 34/40 episodes (85%) of the held-out DEMON CANYON level, matching his scripted expert. Random reaches
0/40. Across the 10 other held-out levels he reaches 46/80 exits (57.5%), again the same as the expert. The host supplies
a bearing to a precomputed route waypoint plus privileged scene features: enemy bearing and distance from ViZDoom objects
restricted to visible monsters, door/exit and blocked flags. The brain picks the action (host argmax → buttons). The film
uses seed 118, chosen for its two kills (see [Seeds and selection](#seeds-and-selection)). Full tables and caveats:
[`models/doomwalker/EVAL.md`](../models/doomwalker/EVAL.md).

A 2-layer d=32 policy (`models/doomplayer_xl`) spread over 230 DOOM games also plays the arena: 160 decisions, all
bit-exact, 8.5 s of thinking per move. Inside one "demon at point blank → FIRE" decision (tokens `[0, 12, 22]`), the
brain runs on up to 896 tiny copies of him, since voodoo dolls are player 1.

**[`full_brain.mp4`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/full_brain.mp4)** (28 s): the
4-layer brain (`models/doombrain_xl`: d=64, 8 heads, MLP 256) writes UNHOLY STRONGHOLD over 10,166 logical tile
instances. The run uses 208,598 voodoo dolls across its 210 compiled tile maps, 1.9 million game tics and 968 s of wall
time including video, and its exposed tile outputs match the reference. Still:
[`full_brain.png`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/full_brain.png).

**[`lobotomy.mp4`](https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/lobotomy.mp4)** (29 s): one
7-pellet shotgun blast to the marine's DOOM brain (`x0` units 0, 2, 3, 5, 7, 11, 14 of the 1-layer d=16 `doomplayer`,
`demos/shotgun.py`) makes him a pacifist that never fires. He gets 0 kills (healthy: 23.35 ± 0.35 over 100 NumPy-reference episodes, seeds 9000–9099)
over 74 decisions in DOOM, all bit-exact. A scan of every single neuron and 3,000 random 7-pellet blasts on all 86
observations this policy can see finds 16 of 96 single neurons and 2,531 of 3,000 blasts that change at least one
action. The pacifist is a pick: 4 of the 3,000 blasts are pacifists, all four on one site (`x0`), and the documented blast
is the first of them in scan order. Exhaustive counts ([Lesion statistics](#lesion-statistics)): 64 of the 11,440
7-subsets of `x0` are pacifists (0.56%) and no 7-subset of `ao`, `x1` or `x2` is (all subsets enumerated; none either in
20,000 random 7-subsets of `m`). Five neurons suffice and five is the global minimum: none of the 3,469,496 subsets of 1
to 4 of the 96 neurons is a pacifist, and 26 of the 61,124,064 five-subsets are (7 of them on `x0` alone, for example
`x0` units 0, 3, 11, 12, 14). The documented blast contains none of those five-neuron sets and none of its own proper
subsets is a pacifist. The film's last card, "shrugging off 83%", is the 2-layer `doomplayer_xl` leaving 2,478 of 3,000
random 7-pellet blasts (82.6%) without any changed action in the 86-entry table of observations (522 blasts change one;
2 of its 352 single neurons do). It is no statement about play.

## Results

| | DOOM brain (`models/doombrain`) | first model (`models/doomnames`) |
|---|---|---|
| architecture | 2 layers, d=32, 4 heads, MLP 96, context 24 | 1 layer, d=16, 2 heads, MLP 32, context 16 |
| integer parameters (including thresholds) | 24,279 (23,838 without) | 3,606 (3,438 without) |
| training data | 12,934 distinct cleaned level names (11,341 train, 300 dev, 1,293 held out) from 956 pre-deduplication source WADs; 901 WADs contribute retained names (927 labels in `levels.json` with rejected rows) | 136 official level names |
| next-character accuracy | 41.5 % on the 1,293 held-out names (7,710/18,588, NumPy integer reference) | 82.1 % on its own 136 training names (1,235/1,505; 1/136 names reproduced token-perfectly); 17.8 % on the 852 short enough held-out names of the new corpus (`models/doombrain/eval_report.txt`) |
| compiled DOOM maps (totals over distinct maps) | 106 maps, 56,390 voodoo dolls, 785,460 sectors | 38 maps, 10,462 dolls, 114,988 sectors |
| one generation | `DEMON CANYON`: 1,690 tile runs over 1,594 logical tile instances, 256,962 game tics | `THE COURTYARD`: 518 tile runs, 453 logical tile instances |
| verification | every exposed tile-output tensor and its negation is compared exactly with the integer reference on the reported run | same, on the reported run |

The 1,594 and 453 count distinct logical tile-instance identifiers over the run, not simultaneous processes: live games
are reused and evicted (bounded by `max_live`). `models/doombrain/run_model_log.txt` records the NECROGENER run. The
parameter, corpus and split counts come from the `models/` and `training/data/` artifacts (`levels.json`,
`train_info.json`). The 956 is the pre-deduplication WAD count of `training/data/SOURCES.md`.

Other names the DOOM brain wrote (temperature 150–200) are all absent from the cleaned corpus (12,934 names from DoomWiki
captures via the Internet Archive and id Software's released source). Whether any historical DOOM map uses them is not
established. With the nearest corpus name and its edit distance: DEMON CANYON (3: DEMON BAYOU), BLOOD SHOCK (4: BLOOD
SIEGE), NECROGENER (3: NECROGENESIS), DEMONIC CONTROL (2: DEMON CONTROL), GRAND TEMPLE (3: DEAD TEMPLE), DEAD SHIP (2:
DEATH SHIP), HELL AND THE LAB (6: ESCAPE THE LAB), BARBATORY (3: ORATORY).

### Talk to DOOM: `models/doomoracle`

A question-answering model: `QUESTION>ANSWER` over a 32-character vocabulary (`$ A-Z ' ! ? >`), 2 layers, d=32, 4 heads,
MLP 96, T=47. The training data is `training/data/oracle.json`: 189 facts (monsters, weapons, items, cheat codes,
levels/episodes, characters, id Software history). Each fact has a cited source URL (id Software's released source, or
DoomWiki through web.archive.org captures) and an evidence string. `training/data/make_oracle.py` re-checks each evidence
string against the cached source text with substring and parsed-field checks. The semantic correctness of each
paraphrase and fact is not checked (see [`ORACLE_SOURCES.md`](../training/data/ORACLE_SOURCES.md)). There are 3–7
question paraphrases per fact (10 facts have 7). One paraphrase per fact is held out, so the 189 held-out questions are
new wordings of facts the model has seen. Held-out facts are not evaluated. Train with `training/train_oracle.py` and
evaluate with `training/eval_oracle.py` (report: `models/doomoracle/EVAL_REPORT.md`). Exact-answer accuracy of the
integer model, by greedy generation in the NumPy integer reference: 94.7 % (606/640) on training paraphrases and 35.4 %
(67/189) on held-out paraphrases.

## The marine: a transformer that plays DOOM (`models/doomplayer`)

ViZDoom's `defend_the_center`: turn left, turn right or fire, 26 bullets, demons from every side. One decision is one
transformer pass over 3 tokens `[$, bearing of the nearest enemy (17 bins), distance (5 bins)]`. The host takes the
argmax over the three action tokens `L R F` and holds the buttons for 4 tics. The observations are privileged: exact
object positions from ViZDoom, binned by Python. A scripted aimbot with exact positions labels the states
(`training/train_policy.py`: 800 rollouts with action noise, 260,694 decisions). The model is a 1-layer, d=16, 2-head,
MLP 32 integer QModel (12 DOOM tile maps), trained offline by supervised imitation of those labels.

| 100 episodes (seeds 9000-9099), integer model via `intref.forward` (NumPy, not DOOM) | kills (mean ± SEM) | survival (tics, mean ± SEM) |
|---|---|---|
| integer model | 23.35 ± 0.35 | 1651 ± 23 |
| scripted aimbot (the teacher; the empirical baseline) | 24.50 ± 0.33 | 1725 ± 23 |
| random | 1.45 ± 0.12 | 337 ± 7 |

± is the standard error of the mean over the 100 episodes (`models/doomplayer/eval_policy.txt`). Action agreement with
the aimbot on held-out episodes is 97.2 % (`models/doomplayer/train_info.json`). That equals the empirical
majority-label ceiling for these binned observations on this held-out dataset, i.e. the agreement reached by predicting
each observation's most common teacher label. The observation takes only 86 distinct values, so the binning limits the
accuracy. On the states the integer model visits itself, agreement is 96.5 % (39,815/41,273).

A DOOM check runs 3 real in-game contexts through the DOOM brain: every exposed tile-output tensor and its negation twin
is exact, and the logits and action equal the reference. The first call took 4.1 s (warm-up), then about 1.15 s per
decision with 12 workers on a shared 16-core host. The 2-layer d=32 model (`models/doomplayer_xl`) took about 5.9 s per
decision in the same check. It has identical held-out agreement and identical episode statistics, so it is no more
accurate.

```sh
.venv/bin/python training/train_policy.py --out models/doomplayer --L 1 --d 16 --dm 32
```

## Seeds and selection

Demonstration seeds are illustrations, not effect-size estimates.

- Walker film, seed 118: chosen from the 120 film seeds in `models/doomwalker/film_dc_seeds.txt`. In the film
  configuration 84/120 reach the exit: 46 of those with 0 kills, 37 with 1 kill and only seed 118 with 2. The headless
  85% / 57.5% evaluation is a separate run from seed 9000 onwards, with the same seeds for the model, the expert and
  random.
- Lesions: the walker's `x0 [1, 2, 5]` (from a scan of all single neurons and 300 random blasts) and the player's 7-pellet
  pacifist (from 96 single neurons and 3,000 random blasts) were picked from the scan results for an interesting
  personality. They show that lesions can have large effects, not how often they do: the player's blast is the first of 4
  pacifists in 3,000 blasts, all four on `x0` (64 of 11,440 7-subsets of `x0`, 0.56%; none on the other sites), and
  2,531 of the 3,000 blasts change an action in `doomplayer`, 522 in `doomplayer_xl`.
- Names: DEMON CANYON (film seed 2, temperature 150) and the other names above are hand-picked samples. The number of
  seeds tried before picking them is not recorded. Unselected sampling statistics for seeds 0–99 per temperature are in
  `models/doombrain/eval_report.txt`.
- DEMON DEATH (the `L1.m` unit 33 lesion at temperature 150, seed 2) is a pick as well. 355 of the 416 single neurons
  change the name sampled at that seed, and `L1.m` 33 ranks 324th of 416 by teacher-forced logit KL
  ([Lesion statistics](#lesion-statistics)). It is not a selective neuron.
- Learning demos: learn to aim reports 3 seeds (2 for the main run, plus 1 and 3), all as they came, including the flat
  22 → 22 → 22. One-game reports one seed (4).
- His brain is the level: seed 4 for both stages. A run stops when it reaches its kill target (5 by default), so "50
  decisions, 5 kills" is the window that ends at the fifth kill (decision 50 in both stages). It is not a kill count over
  a fixed number of decisions.
- Fly: film seed 1. The on/off logic holds on 10 fresh seeds in each of two sets (0–9 and 16000–16009), see
  [FLY.md](FLY.md).
- Policy statistics: all 100 episodes above (seeds 9000–9099) are reported, none dropped.

## Run it yourself

Every "bigger experiment" in the README has a command. They run from a clone, from the repository root, after
`pip install -e .` (`demos/` is not part of the pip package). Each runs real headless DOOM games, checks every value DOOM
exposes against an integer reference with tolerance 0, and uses no film code. The same table, the flags and the notes are in
[`demos/README.md`](../demos/README.md).

| demo | command | what it proves | default run | documented run | section |
|---|---|---|---|---|---|
| His brain is the level | `python -m demos.brainlevel` | The policy transformer (1,345 voodoo dolls) and the argmax over his three actions run in the arena's own map. The level's clock senses, the dolls decide, the level acts. Python only advances tics and reads. Every decision's 540 values equal the reference. | 5 decisions, 4 min | `--decisions 50 --seed 4`, 31 min (measured) | [his brain is the level](#his-brain-is-the-level) |
| DOOM learns | `python -m demos.aimlearn` | A blank perceptron in his own map learns to aim from an aimbot; its 38 weights are ceiling heights. Both scores, the action and all 38 weights equal an integer NumPy perceptron every decision. | 2 lives, 261 decisions, 3 min | `--lives 5 --seed 4`, 11 min (measured, with `--gif`) | [the other demos](#the-other-demos) |
| The fruit-fly circuit | `python -m demos.flyloop` | The 255-neuron FlyWire circuit and the marine's level share one map; an in-map clock senses, draws the input spikes and runs the circuit. Every spike and potential equal the reference every step. Python sends only readout commands. | 40 steps, 3 min | `--steps 900 --seed 1`, ends at step 169, 13 min (measured) | [the fly is alive](FLY.md#the-fly-is-alive) |
| Lobotomy | `python -m demos.lobotomy` | A 7-pellet blast on `doomplayer` makes a pacifist (a 4-in-3,000 pick, all four on `x0`; 5 neurons are the minimum, see [Lesion statistics](#lesion-statistics)); scan of every single neuron and 3,000 blasts; lesioned decisions run in DOOM with the doors closed. | 3 decisions in DOOM, 1.3 min | `--decisions 26`, about 6 min (estimate at 14 s each) | [the other demos](#the-other-demos) |

Times are one run each on a shared 16-core aarch64 machine with up to three DOOM games and other jobs running. The default
runs were measured with `OMP_NUM_THREADS=1`. The documented runs are not repeated by the defaults: `brainlevel` fires its
first shot at decision 22, so the 5 default decisions are all turns. `--gif PATH` writes a small looping GIF (under 3 MB) of
the game's screen and the doll positions; the lobotomy has none.

### Default runs

The text below is the captured output of each default run, unedited.

**`python -m demos.brainlevel`** (224 s):

```text
calibrating the tic budget on a clock-free twin map (one transformer pass, about 40 s) ...
calibration (a clock-free twin of the map): the dolls settle in 3824 tics; compute budget 4302 tics; 36s
one map: his arena + his transformer + the action argmax (1,345 voodoo dolls of him, 37,410 sectors); built in 1.3s
decision   1/5  sees enemy 100-180 deg right, distant  his dolls chose TURN RIGHT (logits L/R/F [-350, 612, -347])  EXACT (540 values: tokens, every transformer scalar, logits, comparator, action; 4302 compute tics; 38.0s)  kills 0, bullets 26
decision   2/5  sees enemy 100-180 deg right, far  his dolls chose TURN RIGHT (logits L/R/F [-350, 612, -347])  EXACT (540 values: tokens, every transformer scalar, logits, comparator, action; 4302 compute tics; 37.6s)  kills 0, bullets 26
decision   3/5  sees enemy 100-180 deg right, far  his dolls chose TURN RIGHT (logits L/R/F [-350, 612, -347])  EXACT (540 values: tokens, every transformer scalar, logits, comparator, action; 4302 compute tics; 37.4s)  kills 0, bullets 26
decision   4/5  sees enemy 100-180 deg right, far  his dolls chose TURN RIGHT (logits L/R/F [-350, 612, -347])  EXACT (540 values: tokens, every transformer scalar, logits, comparator, action; 4302 compute tics; 37.1s)  kills 0, bullets 26
decision   5/5  sees enemy 100-180 deg right, far  his dolls chose TURN RIGHT (logits L/R/F [-350, 612, -347])  EXACT (540 values: tokens, every transformer scalar, logits, comparator, action; 4302 compute tics; 35.7s)  kills 0, bullets 26
5/5 decisions EXACT (2,700 values checked, tolerance 0); the level fired 0 shots and 0 demons died
Python sent 70 console commands, all readout scripts (non-readout: 0); it wrote no token, port or action and pressed no button. The level sensed the demon, the dolls computed and chose, the level turned him or fired.
wall 186s, 37.2s per decision (4302 compute tics + 32 hold + 4 action tics each)
```

**`python -m demos.aimlearn`** (180 s). It prints the first 10 decisions, then every 25th, and each life:

```text
one map: his arena + a perceptron that learns to aim (156 voodoo dolls of him, 4579 sectors); built in 0.1s
decision    1  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose L  scores [0, 0]  learned: weights changed  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision    2  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose R  scores [-2, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision    3  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose R  scores [-2, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision    4  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose R  scores [-2, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision    5  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose R  scores [-2, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision    6  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose R  scores [-2, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision    7  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose R  scores [-2, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision    8  life 1  sees enemy 60-100 deg right  aimbot says R  his dolls chose R  scores [-1, -1]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision    9  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose R  scores [-2, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision   10  life 1  sees enemy 100-180 deg right  aimbot says R  his dolls chose R  scores [-2, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision   25  life 1  sees enemy 7-12 deg right  aimbot says F  his dolls chose R  scores [0, -2]  learned: weights changed  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision   50  life 1  sees enemy 35-60 deg right  aimbot says R  his dolls chose R  scores [-1, -2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
life 1: 4 kills (killed), 71 decisions, agrees with the aimbot 70%, 71/71 exact so far, 48s
decision   75  life 2  sees enemy 35-60 deg left  aimbot says L  his dolls chose L  scores [-1, 2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision  100  life 2  sees enemy 4-7 deg left  aimbot says F  his dolls chose F  scores [1, 2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision  125  life 2  sees enemy 4-7 deg right  aimbot says R  his dolls chose R  scores [0, -2]  learned: weights changed  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision  150  life 2  sees enemy 4-7 deg left  aimbot says F  his dolls chose F  scores [1, 2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision  175  life 2  sees enemy 100-180 deg left  aimbot says L  his dolls chose L  scores [-1, 1]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision  200  life 2  sees enemy 4-7 deg left  aimbot says F  his dolls chose F  scores [1, 2]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision  225  life 2  sees no enemy  aimbot says L  his dolls chose L  scores [-2, 0]  learned: weights changed  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
decision  250  life 2  sees enemy 20-35 deg left  aimbot says L  his dolls chose L  scores [-2, 1]  no change  EXACT (41 values: 2 scores, the action, 38 weights; 1152 tics, 0.7s)
life 2: 22 kills (out of bullets), 190 decisions, agrees with the aimbot 87%, 261/261 exact so far, 179s
261/261 decisions EXACT (41 values each, 10701 in all; tolerance 0); weights changed in 58 of them; kills per life [4, 22]
Python's part: binned what he sees into features, supplied the aimbot's labels, froze and woke the demons, pressed the action the dolls chose. The dolls computed the scores, the action and every weight update.
wall 179s, 0.69s per decision (dolls thinking, checking and playing the action)
```

**`python -m demos.flyloop`** (193 s). It prints every 10th step and every step with an output spike:

```text
building the map and calibrating the step period on a clock-free twin map (about 10 s) ...
one map: his level + 255 fly neurons (1,222 voodoo dolls of him, 31,615 sectors); calibration {'worst_settle': 615, 'spikes': 255, 'period': 689}; step = 4 rewind + 3 write + 689 compute + 16 hold + 16 world = 728 tics; built in 8.1s
step    1  station 1  senses -              0 input spikes,   0 neurons spike  output -  pos (-1360,-6208,0) health 100  EXACT (255 potentials + 255 spikes + PRNG state; 1/1, 4s)
step   11  station 1  senses sugar          4 input spikes,   1 neurons spike  output -  pos (-1360,-6208,0) health 100  EXACT (255 potentials + 255 spikes + PRNG state; 11/11, 50s)
step   21  station 1  senses sugar         10 input spikes,   9 neurons spike  output -  pos (-1360,-6208,0) health 100  EXACT (255 potentials + 255 spikes + PRNG state; 21/21, 96s)
step   31  station 1  senses sugar          6 input spikes,  13 neurons spike  output -  pos (-1360,-6208,0) health 100  EXACT (255 potentials + 255 spikes + PRNG state; 31/31, 142s)
step   32  station 1  senses sugar          2 input spikes,   7 neurons spike  output {'MN9_left': 1, 'MN9_right': 1}  pos (-1360,-6145,0) health 100  EXACT (255 potentials + 255 spikes + PRNG state; 32/32, 146s)
step   36  station 1  senses sugar          5 input spikes,   9 neurons spike  output {'MN9_right': 1}  pos (-1360,-6124,0) health 101  EXACT (255 potentials + 255 spikes + PRNG state; 36/36, 165s)
step   37  station 1  senses -              0 input spikes,   8 neurons spike  output {'MN9_left': 1}  pos (-1360,-6124,0) health 101  EXACT (255 potentials + 255 spikes + PRNG state; 37/37, 170s)
station 1 sugar  senses (steps) sugar 35 | output spikes MN9_left 2, MN9_right 2 | 84 units north, ATE the health bonus | 39 steps
station 2 dust   senses (steps) - | output spikes none | dust still there | 1 step so far
station 3 bitter not reached
station 4 loom   not reached
40/40 steps EXACT (511 values each: every spike, every potential, the PRNG state; tolerance 0), PRNG 40/40, dolls at rest in the hold window 40/40, action lines vs output spikes: 0 differ; scenario complete: False
Python decisions in the loop: 0. Console commands Python sent: 444, all readout scripts (non-readout: 0); it wrote no port, potential or input and moved nothing.
step = 728 tics (compute 689, calibrated worst settle 615); wall 184s, 4.59s per step
```

**`python -m demos.lobotomy`** (80 s):

```text
scanning every single neuron and 3,000 random 7-pellet blasts on all 86 observations, about 30 s ...
scan of 86 observations (33s): 16 of 96 single neurons and 2531 of 3000 7-pellet blasts change at least one action; 4 of the blasts make a pacifist
personalities of the blasts: {'SUBTLY DIFFERENT': 1465, "WON'T SHOOT WHAT'S IN FRONT OF HIM": 1062, 'PACIFIST (never fires)': 4}
the lesion: {'x0': [0, 2, 3, 5, 7, 11, 14]} -> PACIFIST (never fires); actions on the 86 observations {'R': 58, 'L': 28} (healthy {'R': 33, 'F': 19, 'L': 34})
20 episodes (the integer reference picks the actions, seeds 9000..): healthy 23.45 kills, lesioned 0.00 kills (3s)
compiled the policy into verified tile maps in 1.2s
decision  1/3  sees enemy 12-20 deg right, point blank  healthy brain: F, lesioned brain in DOOM: R  EXACT (531 values, tensors, negation twins, logits, action; 14.1s)
decision  2/3  sees enemy in sights, near  healthy brain: F, lesioned brain in DOOM: R  EXACT (531 values, tensors, negation twins, logits, action; 13.9s)
decision  3/3  sees enemy 4-7 deg left, point blank  healthy brain: F, lesioned brain in DOOM: R  EXACT (531 values, tensors, negation twins, logits, action; 13.9s)
3/3 lesioned decisions EXACT in DOOM (1,593 values, tolerance 0); the doors changed the chosen action in 3 of them
Python's part: closed the doors of the lesioned neurons in every tile that consumes them, copied tile outputs to the next tile, and compared. The dolls did every product, sum and activation.
```

The scan prints counts only. What they mean (hit rates, the minimal pacifist, the name model) is in
[Lesion statistics](#lesion-statistics); `--minimal` and `--names` are separate runs.

### Documented runs, measured on this tree

| documented | measured by the command |
|---|---|
| His brain is the level, `--decisions 50 --seed 4`: 50/50 exact, 5 kills, 5 shots, 26 to 21 bullets | 50/50 exact, 27,000 values, 5 kills, 5 shots, 26 to 21 bullets, 610 readout commands and none other. The same with `--period 4160`, the film's budget (the default is 4,302), with identical decisions. 1,881 s and 1,821 s with three games running |
| Fly, `--steps 900 --seed 1`: 169/169 exact | 169/169 exact, PRNG 169/169, dolls at rest 169/169, 0 action-line differences, 1,863 readout commands and none other; station table as in [FLY.md](FLY.md): sugar 35 steps, MN9 4 spikes, ate; dust 38 steps, aDN1 1; sugar and bitter 59 steps, refused; looming 7 steps, Giant Fiber 1, peak 75. 797 s, 4.7 s per step |
| Lobotomy: 16 of 96 single neurons and 2,531 of 3,000 blasts change an action; the pacifist gets 0 kills | the same counts; 4 of the 3,000 blasts are pacifists, all on `x0`; 0.00 kills against 23.45 for the healthy brain over 20 episodes (seeds 9000 to 9019) |
| One-game learner, seed 4, film run: kills 4, 24, 26, 22, 24 and 931/931 exact; aimbot baseline 9, 26, 26, 25, 26 | `--lives 5 --seed 4`: kills 4, 22, 15, 20, 16 and 815/815 exact (634 s with `--gif`). `--driver aimbot`: 9, 26, 26, 25, 26 and 1,061/1,061 exact. The first life matches (4 kills in 71 decisions); lives 2 to 5 do not. See below |

The learner is exact in every run, so the claim of exact scores, actions and weights holds. The film's kill counts are from a
run made with the map compiler as it was on 18 October. A re-run of that snapshot on this machine gives the film numbers
again (4 and 24 kills in lives 1 and 2, 71 and 213 decisions). The compiler changed afterwards: the learner map has 4,539
sectors and 1,136 settle tics per decision in the snapshot and 4,579 sectors and 1,152 settle tics now, and the game
trajectory changes with it although every decision stays exact. The public `v0.1.0` tag already has the new compiler and
gives 4 and 22 kills in lives 1 and 2, as the current code does. Swapping the old `pure.py`, `tile.py`, `ops.py`, `udmf.py`
into the current tree restores 4 and 24; swapping the old `runtime.py`, `policy.py` and `aimlearn.py` in does not, and
three extra frozen tics per decision in the snapshot change nothing. Which compiler change moves the trajectory was not
isolated. The aimbot baseline in the same arena (9, 26, 26, 25, 26) is the same under both compilers. The film numbers
therefore cannot be reproduced from the repository; the current numbers can.

### Lesion statistics

What the lobotomy numbers mean, computed exactly instead of read off a hand-picked scan. `doomformer/lesion.py` evaluates
thousands of zero-ablation lesions at once (float64 GEMMs on the flattened batch; every sum is an integer below 2^53) and
equals `intref.forward(m, tokens, ablate=...)` bit for bit; `tests/test_lesion.py` checks this on `doomplayer`,
`doomplayer_xl` and random models, single- and multi-site lesions, last position and all positions. The intervention
API beyond zeroing is described in [interventions.md](interventions.md).

```sh
OMP_NUM_THREADS=1 python -m demos.lobotomy --minimal --null --json lesion_minimal.json   # 9.1 min under load
OMP_NUM_THREADS=1 python -m demos.lobotomy --names --json lesion_names.json              # 1.9 min under load
python -m demos.lobotomy --model doomplayer_xl                                           # the unchanged scan, 2.0 min
```

All numbers below are from these commands (one run each on the shared 16-core machine, load average 28 to 46, one core per
run; the `--minimal` run includes 3 decisions in DOOM with 1 worker and 20 episodes). Pacifist means: the lesioned brain chooses `F` on none of the 86 observations. `--max-size 6` is the largest
size enumerated exhaustively on `m` (32 units); sites of 16 units are enumerated at every size.

**Marine, per site.** Subsets of one activation site, all sizes.

| site | units | smallest pacifist size | pacifists at that size | hit rate at that size | pacifists among 7-subsets | 7-subsets that change at least one action |
|---|---|---|---|---|---|---|
| `x0` | 16 | 5 | 7 of 4,368 | 0.16% | 64 of 11,440 (0.56%) | 11,440 (100%) |
| `ao` | 16 | 8 | 16 of 12,870 | 0.12% | 0 of 11,440 | 11,430 (99.9%) |
| `x1` | 16 | 9 | 2 of 11,440 | 0.017% | 0 of 11,440 | 11,236 (98.2%) |
| `x2` | 16 | 13 | 1 of 560 | 0.18% | 0 of 11,440 | 9,078 (79.4%) |
| `m` | 32 | none | none | none | 0 of 20,000 (sampled; Wilson 95%: 0 to 0.019%) | 7,758 of 20,000 (sampled, 38.8%) |

All 65,535 subsets of each 16-unit site were enumerated. On `m`, sizes 1 to 6 (1,149,016 subsets) and sizes 29 to 32 were
enumerated and 20,000 random subsets were drawn for each size from 7 to 28; none is a pacifist. The number of subsets of
`x0` that change at least one action is 7 of 16 at size 1, 101 of 120 at size 2 and every subset from size 7 up. Pacifist
counts of `x0` by size 5 to 16: 7, 24, 64, 114, 138, 148, 106, 65, 24, 16, 5, 1 (the full table of every site and size is
in the printed output).

**Marine, all 96 neurons pooled** (any mix of sites).

| subset size | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| subsets | 96 | 4,560 | 142,880 | 3,321,960 | 61,124,064 |
| pacifists | 0 | 0 | 0 | 0 | 26 |

No pacifist has 4 or fewer neurons (3,469,496 subsets tested, all of them), so the global minimum is 5. There are 26 sets
of that size: 7 on `x0` alone, 9 with 4 `x0` units and 1 `ao` unit, 3 with 4 `x0` and 1 `x1`, 1 with 4 `x0` and 1 `x2`, 4
with 3 `x0`, 1 `ao` and 1 `x1`, and 2 with 3 `x0` and 2 `ao`; none uses `m`. Every one contains `x0` units 0 and 11. The
command prints the 26 sets. The size-5 pass took 210 s under load.

**The documented blast.** `x0` units 0, 2, 3, 5, 7, 11, 14 is one of the 64 pacifist 7-subsets of `x0` (all 64 contain units
0 and 11) and the first of the 4 pacifists among the scan's 3,000 blasts (the other three: `x0` 0 3 5 8 11 12 14; 0 2 3 5 10
11 15; 0 3 5 7 8 11 12). It contains none of the 7 pacifist 5-subsets of `x0` (`[0,3,11,12,14]`, `[0,3,11,12,15]`,
`[0,6,11,12,15]`, `[0,7,11,12,15]`, `[0,7,11,14,15]`, `[0,8,11,12,15]`, `[0,10,11,12,15]`), and none of its 126 proper
non-empty subsets is a pacifist: it is pacifist only as a whole. 35 of the 64 pacifist 7-subsets contain one of the seven
5-sets; each 5-set is contained in 55 7-subsets of `x0`, of which 10, 18, 3, 5, 3, 5 and 11 are pacifists.

**Hit rate of the original scan.** The scan draws a site uniformly, then 7 of its units. Replaying its random stream with
the batch evaluator gives the same 2,531 blasts that change an action and the same 4 pacifists, all on `x0`. Expected
pacifists from the exact per-site rates: `x0` 591 blasts, 4 observed, 3.31 expected; `ao` (601 blasts), `x1` (643), `m`
(569) and `x2` (596): 0 observed and 0 expected. In total 4 observed against 3.31 expected, with 4 of the 3,000 blasts
(0.13%) a pacifist. 3,000 uniform 7-subsets of all 96 neurons (any mix of sites) give 0 pacifists.

**Effect size** (`--tau 150`). Mean over the 86 observations of KL(softmax(healthy/tau) || softmax(lesioned/tau)) of the
three action-token logits, in nats; the full-vocabulary value is in brackets. The action-token logits of `doomplayer`
are integers with a median gap of 3,237 between the two largest (the training run's float logit scale is 0.00394 per
unit, so one nat is about 254 integer units), so at tau = 150 a changed action costs about gap/tau = 22 nats: the measure
is close to minus the log-probability the lesioned brain gives the healthy action. `--tau` sets it.

| lesion | KL | rank among all same-size subsets of `x0` | percentiles of that population (p5, p50, p95, max) |
|---|---|---|---|
| `x0` 0, 3, 11, 12, 14 (first minimal set) | 3.30 [3.27] | 216 of 4,368 from the top | 0.18, 0.90, 3.29, 5.48 (all 4,368 5-subsets) |
| the 7 minimal `x0` sets | 2.68 to 5.39 | ranks 3 to 427 of 4,368 | same |
| documented 7-blast | 5.18 [5.11] | 282 of 11,440 from the top | 0.55, 1.73, 4.57, 7.92 (all 11,440 7-subsets) |

All seven 5-sets are in the top 10% of the 4,368 5-subsets of `x0` by KL, and the documented blast is in the top 2.5% of the
7-subsets.

**Run in DOOM.** The minimal lesion `x0` [0, 3, 11, 12, 14] (first of the 7 in lexicographic order): 0.00 kills in 20
episodes against 23.45 for the healthy brain (seeds 9000 to 9019, the integer reference chooses; mean episode length 295
against 1,719 tics), and 3 of 3 decisions EXACT in DOOM with the doors closed (1,593 values, tolerance 0; the doors changed
the chosen action in all 3).

**Null: weight-shuffled copies.** `--null` permutes the entries of every weight matrix of `doomplayer` (`Wq`, `Wk`, `Wv`,
`Wo`, `Wup`, `Wdn`, `Wun`) at random for seeds 0 to 19 and keeps embeddings, biases, `r1`, `r2` and thresholds, then repeats
the exhaustive search per site (stopping at the first size with a pacifist; `m` up to size 6). No copy was skipped (every
copy fires somewhere). The copies fire on 3 to 85 of the 86 observations (the real brain: 19), and a copy that rarely
fires is easy to make pacifist, so the last two columns keep the 7 copies that fire on 9.5 to 38 observations.

| site | real model | copies with a pacifist (of 20) | median smallest size | near-rate copies with one (of 7) | median smallest size, near-rate |
|---|---|---|---|---|---|
| `x0` | 5 | 15 | 4 | 7 | 5 (3 to 6) |
| `ao` | 8 | 9 | 3 | 5 | 8 (3 to 13) |
| `x1` | 9 | 15 | 5 | 7 | 5 (3 to 10) |
| `m` | none up to 6 | 6 | 3 | 2 | 4.5 |
| `x2` | 13 | 1 | 1 | 1 | 1 |

Among copies with a similar firing rate a pacifist of 5 units on `x0` is typical (median 5, range 3 to 6), so "the minimum
is 5 units" is not small relative to a function with the same weight statistics. The real brain has no pacifist on `m`
(6 of 20 copies do) and needs 13 units on `x2` (the one copy with an `x2` pacifist needs 1).

**Name model** (`python -m demos.lobotomy --names`, `models/doombrain`, 2 layers, d=32, 4 heads, MLP 96, T=24). The 416
single neurons are 32 `x0` + 2 x (32 `ao` + 32 `x1` + 96 `m` + 32 `x2`). Teacher forcing runs over every position of the
1,293 validation names of `training/data/levels.json` (the fixed split of `training/train_brain.py`, 18,588 positions,
tau = 150, the model's sampling temperature). The text columns lesion one neuron and generate with `intref.generate`.

| | neurons that change the text (of 416) | distinct texts |
|---|---|---|
| T=150, seed 2 (healthy: DEMON CANYON) | 355 | 257 |
| T=150, seed 3 (healthy: BASE MALL) | 343 | 261 |
| T=150, seed 7 (healthy: THE DEATH DEATHER) | 326 | 237 |
| greedy (healthy: THE SECRET OF THE THE S) | 368 | 228 |

410 neurons change the text in at least one of the four runs, 267 in all four and 6 in none (all `L1.m`: 9, 10, 55, 64, 82,
85). The single-neuron teacher-forced KL (full vocabulary) has percentiles p0 0.023, p5 0.040, p25 0.102, p50 0.185,
p75 0.373, p95 0.757, p100 1.113; no neuron has KL 0, and the largest are `x0` units (13, 0, 3, 6, 25). Without the end
token the numbers are within 0.03 of these (p50 0.193). The three neurons that write DEMON DEATH at seed 2:

| neuron | KL | rank of 416 (1 = largest KL) |
|---|---|---|
| `L0.x1` 20 | 0.470 | 65 |
| `L1.x1` 29 | 0.250 | 154 |
| `L1.m` 33 | 0.087 | 324 |

By KL quartile (104 neurons each, lowest first) the neurons that change the seed-2 text number 55, 92, 104 and 104. The
data show that `L1.m` 33 is not a death neuron: 355 of 416 single neurons change the sampled name at that seed, 368 change
even the greedy text, and `L1.m` 33 is in the lowest KL quartile of the network. The lesion runs exactly in DOOM (see
[commands.md](commands.md)); it is no selective effect. Generation is autoregressive: a changed logit can change a letter,
and every later letter conditions on it.

## Induction: a transformer that learned an algorithm (`models/induction`)

The name models memorise: a 5-gram table beats the largest. This one is trained on a task a table cannot solve. A
sequence is BOS, a prefix of 5 to 8 distinct random tokens (16 letters), then a repeat of a run of 1 or 2 prefix
tokens (`$ KFCAPJD CA`). At a repeated token the answer is the token that followed its first occurrence (C then A,
A then P). The prefix is drawn without replacement, so given the current token every other prefix token is equally
likely: only looking back in the sequence answers it. Task, generator and rule: `doomformer/induction.py`; trainer:
`training/train_algo.py`; float baseline: `train_algo.py --float`; table: `training/eval_algo.py`.

Held-out (sequences the training stream never draws, and a second set whose queried token pairs never occur in
training), integer model L=2, d=16, 2 heads, K=7, softmax attention: 0.9995 on 4,000 random sequences and 0.9995 on
2,000 `unseen` sequences; float GPT of the same size 0.9991 and 0.9981; bigram table 0.069 and 0.000, 4-gram table 0.233
and 0.000, best heuristic 0.22 and 0.20. K=4, relu attention and d=32 rows are in
[`models/induction/EVAL.md`](../models/induction/EVAL.md). `training/analyze_induction.py` reads the circuit from the
integer attention weights: both layer-0 heads attend to the previous token, both layer-1 heads put 0.94 of their weight
on the token after the first occurrence, and zeroing either layer takes accuracy to chance.

```sh
python -m demos.induction --heldout 25 --workers 3     # 25 held-out sequences in tile maps, every value compared
python -m demos.induction KFCAPJDC                     # your sequence: what follows C?
python -m demos.induction --set unseen --heldout 5
```

The 25-sequence run (`models/induction/doom25_tiles.log`): 25/25 exact, 52 tile maps with 6,081 voodoo dolls and
191,566 sectors, 13,868 tile runs, 520,749 values compared at tolerance 0, 2,545 s wall with 3 games on a shared machine
(load average 25 to 45), DOOM's answer equals the rule at 38 of 38 queried tokens. The model does not fit one map. The
one-map model `models/induction_onemap` (L=2, d=8, K=2, context 7, 8,421 dolls) is weaker (0.958 and 0.859 held-out):
one held-out sequence in the single map is EXACT (2,247 values at tolerance 0, 332 s on a shared machine,
`models/induction_onemap/doom_onemap.log`).
`demos/induction.py` deals the tile runs of one stage over the game processes (`SharedGames`), about 1.7 times faster
per token than the stock logical-game naming (7.6 to 11.6 s against 13.6 to 17.0 s per position, one timing each, same
box and load).

**DOOM learns to read digits** (`doomformer learn digits`, [learning.md](learning.md#doom-learns-to-read-digits), logs in
[`models/digits/`](../models/digits)): a blank 10-class perceptron (370 ceiling-height weights) in one map; the dolls compute the
scores, the argmax and the update, and after each of 1,500 steps all 455 exposed values and all weights equal a NumPy twin.
Held-out accuracy 10 % to 71.2 % (textbook NumPy perceptron on the same stream 81.0 %). Unlike the kill counts of the one-game
learner above, which do not reproduce with the current compiler, these numbers come from the command.
