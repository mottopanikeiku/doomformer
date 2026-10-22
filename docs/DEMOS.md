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
transformers use integer attention without LayerNorm: ReLU-style attention in every demo below, softmax attention computed
by the dolls in `doomnames_softmax` (`demo --softmax`) and `doomnames_onemap_softmax` (`demo --one-map --softmax`). They
are no drop-in softmax/LayerNorm transformers. The engine is ViZDoom 1.3.2's ZDoom engine with Freedoom2 assets (not
vanilla 1993 DOOM). Transformer weights are trained
offline in PyTorch with quantisation-aware training. The perceptron demos update their weights inside DOOM, and the fly
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
blank. Kills per life: **4 → 24 → 26 → 22 → 24**. In the same arena and seed, the aimbot that supplies the labels gets 9,
26, 26, 25, 26, and a blank brain 0 every life (trajectories diverge once the actions differ). **931/931 decisions
bit-exact**: both scores, his action and all 38 weights. One seed (4) is reported.

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
reads the logits and samples the next letter exactly as the reference does. No Python and no ACS arithmetic run between
layers. Dolls that would hit the same sector in the same tic are spread over time: the layer's clock doll releases them in
**time-multiplexed waves**. Long layers are folded into rows.

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
`demos/shotgun.py`) makes him a pacifist that never fires. He gets 0 kills (healthy: 23.5 in NumPy-reference episodes)
over 74 decisions in DOOM, all bit-exact. A scan of every single neuron and 3,000 random 7-pellet blasts on all 86
observations this policy can see finds 16 of 96 single neurons and 2,531 of 3,000 blasts that change at least one
action. This pacifist was picked from the scan results. The larger 2-layer `doomplayer_xl` ignores 2,478 of 3,000 blasts
(82.6%), which the film's last card rounds to "shrugging off 83%".

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
  personality. They show that lesions can have large effects, not how often they do (see the 2,531 and 2,478 of 3,000
  counts above).
- Names: DEMON CANYON (film seed 2, temperature 150) and the other names above are hand-picked samples. The number of
  seeds tried before picking them is not recorded. Unselected sampling statistics for seeds 0–99 per temperature are in
  `models/doombrain/eval_report.txt`.
- Learning demos: learn to aim reports 3 seeds (2 for the main run, plus 1 and 3), all as they came, including the flat
  22 → 22 → 22. One-game reports one seed (4).
- His brain is the level: seed 4 for both stages. A run stops when it reaches its kill target (5 by default), so "50
  decisions, 5 kills" is the window that ends at the fifth kill (decision 50 in both stages). It is not a kill count over
  a fixed number of decisions.
- Fly: film seed 1. The on/off logic holds on 10 fresh seeds in each of two sets (0–9 and 16000–16009), see
  [FLY.md](FLY.md).
- Policy statistics: all 100 episodes above (seeds 9000–9099) are reported, none dropped.
