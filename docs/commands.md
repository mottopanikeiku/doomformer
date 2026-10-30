# Commands

Every command prints what DOOM computed and the result of the exact check. Sample output is from the shipped commands.
`[...]` marks a cut, and progress lines are left out. The `in N parallel DOOM games` line of each sample gives its game
count.

Times are from an idle 16-core arm64 Linux machine running up to 8 games at once. `hello`, `draw` and the one-map
commands use one game. Tile-map commands take proportionally longer on fewer games. Speed and cost:
[scaling.md](scaling.md).

| command | what it runs | time |
|---|---|---|
| [`doomformer hello`](#doomformer-hello) | a 4-6-5-3 integer MLP on 5 random inputs | 2 s |
| [`doomformer draw --digit 7`](#doomformer-draw---digit-7) | the marine walks a 7 on the DrawPad floor, the level reads it | 14 s |
| [`doomformer mnist`](#doomformer-mnist) | MNIST test digits, or your own with `--image`, the whole net in one map | 1.5 to 3 min per digit |
| `doomformer mnist --tiles` | the same net as 4 tile maps, Python copying values between them; `--all` reads 10,000 digits | about 1 s per digit |
| `doomformer mnist --opt 3` | the one-map net with the ACS copy bus | 272 game tics per digit |
| [`doomformer demo`](#doomformer-demo) | a 1-layer transformer writes a level name over 38 tile maps | 15 to 60 s |
| [`doomformer demo --softmax`](#softmax-attention-modern-and-the-flags) | softmax attention computed by the dolls | 1 to 3 min |
| [`doomformer demo --model doomnames_modern`](#softmax-attention-modern-and-the-flags) | RMSNorm and a gated MLP computed by the dolls | about 35 s |
| [`doomformer demo --nightmare`](#doomformer-demo---nightmare) | the 2-layer model, 106 tile maps | 3.5 to 7 min |
| [`doomformer demo --one-map`](#one-map-and---loop) | one forward pass of a smaller model in one map | about 90 s per letter |
| [`doomformer demo --one-map --loop`](#one-map-and---loop) | one map writes a whole name | 17 to 21 min |
| [`doomformer demo --one-map --loop --until-eos --prompt DAN`](#one-map-and---loop) | the map stops at the end-of-text token | 4 min |
| [`doomformer ask "what does iddqd do"`](#doomformer-ask) | a 2-layer transformer answers from 189 facts | about 11 min |
| `doomformer plan MODEL` | counts dolls and sectors without building the map | 20 s to 4 min |
| `doomformer doctor` | checks Python, ViZDoom and Freedoom; builds a missing ACC and compiles an ACS probe | about 1 s once ACC is built |
| [`doomformer facts`](#doomformer-fuzz-verify-facts-lint) | the engine facts as 28 probes in ViZDoom, PASS or FAIL per row | about 10 s |
| [`doomformer fuzz --nets 40`](verification.md) | random nets run in DOOM against the integer reference | 10 s |
| [`doomformer verify NET`](verification.md) | every input class of a small MLP's domain | seconds to minutes |
| [`doomformer verify --extract WAD --net NET`](verification.md#static-extraction) | static check, no game: recover the net from a one-map MLP and compare | about 6 s for MNIST |
| [`doomformer machine multiply 7 9`](#doomformer-machine) | a counter machine in one map, every step checked | 17 s |
| [`doomformer machine subleq all`](#doomformer-machine) | a SUBLEQ machine in one map, three programs on one sha256 | 5 min |
| [`doomformer learn digits --classes 4 --steps 300`](#doomformer-learn) | a blank perceptron learns digits in one map, every step checked | 5 min |
| [`doomformer tune --check`](tuning.md) | checks the committed timing constants; without `--check` re-measures them | minutes to hours |
| `doomformer bench idle` / `pass` / `tiles` | engine-time measurements ([scaling.md](scaling.md)) | minutes |
| `doomformer tune --hashes` | compares the opt 0 maps of six models with `doomformer/opt0_hashes.json` | 30 s |
| `doomformer lint WAD` | checks the wave schedule of a one-map brain ([scheduling.md](scheduling.md)) | 1 to 45 s |
| `doomformer lint --fuzz` | soundness campaign of the checker against DOOM | about 11 min for 1,000 nets |

## `doomformer hello`

A 4-6-5-3 integer MLP compiled into one map, five random inputs, every value checked. The first run also builds the
ACS compiler, about 10 s on a cold cache.

```text
tiny integer MLP 4-6-5-3 compiled into one DOOM map: 60 voodoo dolls, 1222 sectors
input 1/5  x=[-71, -82, 64, 37] -> y=[11, 4, -15]  EXACT (3 outputs + 22 hidden readouts equal the integer reference; 488 game tics)
...
5/5 EXACT
```

## `doomformer draw --digit 7`

The DrawPad level: a 28×28 floor canvas, a READ switch and a 7-segment display in one map. `--digit D --seed S` makes
the marine walk digit D and press READ. Without `--digit` you draw: arrow keys, hold Ctrl to draw, Space on READ. The
command builds at `--opt 3`, so a READ takes 273 game tics. How it works: [drawpad.md](drawpad.md).

```text
          ▄▄▄▄▄▄▄▄▄▄▄▄▄▄
       ▄████████████████
       ████▀▀     ▀████▀
       ▀▀▀        ▄███▀
                  ████
                  ███
                  ███
                  ███
                 ████
                 ███▀
                 ████
                 ▀███

DOOM reads: 7  (right)   -- the winner's doll lit it on the 7-segment display
EXACT: all 248 values DOOM computed (hidden layer, 45 score comparisons and 10 win counts, each with its twin, and the one-hot answer) equal the NumPy integer reference computed from the 784 pixel ports DOOM holds (tolerance 0); one display firing
pixels: the 784 ports equal the level's canvas (the tiles the pen inked)
pixels: the 784 ports equal the 3x3 brush stamped at every position the marine had with FIRE held (1 of them within 0.1 units of a tile edge, counted for either tile)
READ pass: 273 game tics = 8 s of real-time DOOM at 35 tics/s, 3 s wall | Python sent 58 commands, 29 readout scripts each followed by the ack script it waits for (0 waited more than one advance), and nothing else (0 rejected)
```

This transcript predates the readout guard. The `Python sent` line now also counts one guard before each page.

## `doomformer mnist`

The DrawPad network on MNIST test digits, 4 by default, one game per digit. `--digits N` samples N digits, `--all`
reads all 10,000, and `--image my7.png` reads your own. EXACT compares DOOM with the integer network. It does not say
the image was a digit. `--workers`, `--ray` and `--record PATH`: [scaling.md](scaling.md).

```text
$ doomformer mnist --digits 2 --seed 1

          ███
          ████                      ▄▄▄█████████▄▄
           ████                     ▀█████▀▀▀▀▀████
            ████                             ▄▄███▀
             ███▄                       ▄▄▄███████▄▄
             ████                       ████▀▀▀▀▀▀█████
              ████                                  ▀███
               ████                      ▄▄         ▄██▀
               ▀████                     █████▄▄██████▀
                ▀██                        ▀▀▀▀▀▀▀▀▀▀

DOOM reads: 1  right            DOOM reads: 3  right
test #46                        test #51

2/2 read correctly | 2/2 EXACT: every value DOOM computed (248 per digit: hidden layer, comparisons, win counts and
the one-hot answer) equals the NumPy integer reference (tolerance 0); Python wrote the 784 pixels, released the dolls and compared | 5,696 game tics per digit | 88 s
```

### `--tiles`, `--opt` and `plan`

`--tiles` runs the network as four tile maps in resident games. Python copies each layer's values into the next tile.
The 10,000-digit result is in [scaling.md](scaling.md#all-10000-mnist-test-digits-in-doom).

```text
$ doomformer mnist --digits 2 --seed 1 --tiles --workers 2
DOOM reads handwritten digits. The network (784 pixels -> 64 -> 45 score comparisons -> 10 win counts -> one-hot) runs as 4 TILE maps, one per layer: 1,351 voodoo dolls, 111,112 sectors in all. [...]
reading 2 digits with 2 workers; each holds the 4 tile games of the network (8 engine processes, 2 computing at once) (about 1.5 s per digit per worker) ...

[the two digits, as above]

DOOM reads: 1  right            DOOM reads: 3  right
test #46                        test #51

2/2 read correctly | 2/2 EXACT: every value DOOM computed (248 per digit: hidden layer, comparisons, win counts and the one-hot answer) equals the NumPy integer reference (tolerance 0); Python wrote the 784 pixels, copied the layer values between the 4 tile games and compared | 216 game tics per digit (mean) summed over the tiles | 4 s | 30.2 digits per minute in 2 tile workers
```

`--opt N` picks the one-map schedule ([scheduling.md](scheduling.md)). At `--opt 3` an ACS script copies each
layer's light levels into the next layer's gates.

```text
$ doomformer mnist --digits 2 --seed 1 --opt 3 --workers 2
DOOM reads handwritten digits. The whole network (784 pixels -> 64 -> 45 score comparisons -> 10 win counts -> one-hot) is ONE DOOM map:
1,355 voodoo dolls, 88,146 sectors; every ink pixel releases a doll that crosses the weight lines of the hidden layer, and dolls compare every pair of digit scores, so DOOM itself picks the digit.
reading 2 digits in 2 DOOM games at once [...]

[the two digits, as above]

DOOM reads: 1  right            DOOM reads: 3  right
test #46                        test #51

2/2 read correctly | 2/2 EXACT: every value DOOM computed (248 per digit: hidden layer, comparisons, win counts and the one-hot answer) equals the NumPy integer reference (tolerance 0); Python wrote the 784 pixels, released the dolls and compared | 272 game tics per digit (mean) | 8 s | 14.3 digits per minute in 2 DOOM games
```

`doomformer plan MODEL --opt N --explain` counts dolls and sectors without building, predicts tics and seconds per pass
and recommends one map or tiles. MODEL is a shipped name, a model directory or an MLP `.npz`. `--explain` adds the
schedule per layer and what each pass saves.

```text
$ doomformer plan models/mnist --explain --opt 3
counting dolls and sectors without building the map (large models take a few minutes) ...
MLP models/mnist: 1355 voodoo dolls, 88146 sectors, fits in one map: yes (span 31,969 x 5,192 units, limit 32,000 per axis)
  one map: about 276 game tics per pass, +-7.2 % (+-20 tics; calibrated on measured settles, docs/scheduling.md; the geometry bound the runner's timeout uses is 304)
  one map: predicted 2.8 s per pass (cost model on the 304-tic bound)
  tiles: 4 tile maps, 1351 dolls in total, about 216 game tics summed, predicted 752 ms per pass
  predicted seconds: engine time of one pass with the games loaded (about 90 ns per sector per tic, fitted on a quiet machine at load average 1 to 2; rms error 13 % one map, 22 % tiles on 11 held-out K3 and K8 nets; 6 other nets of K 2 to 8 measured 0 to 40 % above the prediction, rms 21 %, docs/scaling.md); a loaded or slower machine takes longer
  recommendation: tiles is faster by about 3.9x (df.run_many(net, xs, mode='tiles'); `doomformer mnist --tiles` for the MNIST net); one map keeps Python out of the loop between layers
  df.compile(mode='auto') would use: map

schedule of the one-map build, opt ['colour', 'copy', 'exclusive', 'lights', 'tuned']: 304 estimated game tics [...]

layer                   dolls  waves  bound    S    tics   sectors  dominated by
mlp_l0                    784      1      1   40      85    58,509  corridor length (40%)
mlp_l1                    512      1      1   40      81    28,679  corridor length (37%)
mlp_l2                     45      1      1   40      72       701  corridor length (29%)
mlp_l3                     10      1      1   40      66       126  corridor length (28%)

passes, cumulative (estimated settle tics, geometry counter):
                                        tics   delta   sectors    delta   waves
baseline (opt 0)                       5,676           158,550            1,312
+ exclusive dolls share waves          5,676      +0   158,550       +0   1,312
+ light targets never conflict         5,676      +0   158,550       +0   1,312
+ DSATUR / circle-method colouring     5,652     -24   158,544       -6   1,306
+ tuned timing constants               3,560  -2,092   158,538       -6   1,306
+ copy bus (one wave per layer) <-       304  -3,256    88,146  -70,392       4
bound: clique lower bound of the layer's conflict graph (waves cannot go below it); empty at opt 0.
```

## `doomformer demo`

A 1-layer transformer (`doomnames`, d=16, 2 heads) writes a level name over 38 tile maps. Python copies the integers
between games and samples each letter from the logits DOOM computed. Options: `--seed N`, `--letters N`, `--prompt TEXT`,
`--temperature T`, `--png PATH`.

```text
A transformer writes a DOOM level name; every number it computes is computed by DOOM.
model: the shipped doomnames: 1 layer, d=16, 2 heads, context 16, ReLU attention, trained on 136 DOOM level names
sampling: temperature 180, seed 20
38 tile maps (10,462 voodoo dolls, 135,912 sectors) built in 0 s; running them in 4 DOOM game processes ...

DOOM wrote: 'GHEER'   (not one of its training names)
  174 tile runs in 4 parallel DOOM games | 10,462 voodoo dolls in the tile maps | 6,902 values read back and compared with the NumPy integer reference (tolerance 0) | 22 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

Seed 20 is hand-picked. At the default temperature the model writes one of its 136 training names for 6 of 10 seeds
([`models/doomnames_softmax/EVAL.md`](../models/doomnames_softmax/EVAL.md)).

### Softmax attention, modern and the flags

`--softmax` runs the model trained with softmax attention. The construction is in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#softmax-attention).

```text
DOOM wrote: 'FINAL INEALL'   (not one of its training names)
  494 tile runs [...] | 4,886 voodoo dolls in the tile maps |
  106,834 values read back and compared with the NumPy integer reference (tolerance 0) [...]
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

`--model doomnames_modern` adds RMSNorm and a gated MLP to softmax attention ([HOW_IT_WORKS.md](HOW_IT_WORKS.md#rmsnorm)).

## `doomformer demo --nightmare`

The 2-layer, 24,279-parameter transformer trained on real DOOM level names, compiled into 106 maps.

```text
A transformer writes a DOOM level name; every number it computes is computed by DOOM.
model: the shipped doombrain: 2 layers, d=32, 4 heads, context 24, ReLU attention
sampling: temperature 150, seed 2
106 tile maps (56,390 voodoo dolls, 785,460 sectors) built in 10 s; running them in 6 DOOM game processes ...

DOOM wrote: 'DEMON CANYON'
  1,690 tile runs in 6 parallel DOOM games | 56,390 voodoo dolls in the tile maps | 120,666 values read back and compared with the NumPy integer reference (tolerance 0) | 210 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

DEMON CANYON is hand-picked. Unselected samples: [`models/doombrain/eval_report.txt`](../models/doombrain/eval_report.txt).
Lesions of single neurons: [DEMOS.md](DEMOS.md#lesion-statistics) and [interventions.md](interventions.md).

## One map, and `--loop`

`--one-map` runs a smaller model, `doomnames_onemap`, with no Python between layers. Python writes the token and samples
the letter after each pass.

```text
A transformer's forward pass, computed inside ONE DOOM map.
model: the shipped doomnames_onemap: 1 layer, d=8, 1 head, context 6, ReLU attention, trained on DOOM level names of up to 5 letters
map built in 5 s: 3,548 voodoo dolls cross lava lines in 95,448 sectors
DOOM is running 1 forward pass(es), about 85 s each on one core (3,500 game tics of dolls walking) ...

DOOM wrote: 'D'
  3,548 voodoo dolls | 95,448 sectors | 176 values read back and compared with the NumPy integer reference (tolerance 0) | 3,488 game tics per pass | 89 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same text
```

`--one-map --loop` builds one map that writes a whole name. The winner's doll writes the next token and the level runs
the next pass by itself. Python advances the clock and reads. Add `--softmax` for the softmax model. Results:
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#generation-without-python).

`--until-eos` (with `--loop`) stops the level after the pass whose newest token is EOS. The map is the same file for
every prompt of one length ([HOW_IT_WORKS.md](HOW_IT_WORKS.md#stopping-at-eos)).

## `doomformer ask`

A 2-layer transformer (d=32, 4 heads) trained on 189 facts about DOOM answers a question letter by letter, over tile
maps in at most 8 games.

```text
You ask DOOM: WHAT DOES IDDQD DO?
DOOM answers with a 2-layer transformer (d=32, 4 heads) trained on 189 facts about DOOM, split into tile maps over 6 DOOM games.
The first letter runs all 20 positions of the question through the network (3,160 tile runs): about 7 min with 6 DOOM games. Each later letter adds one position: about 28 s. (Fitted to two timed runs on a loaded machine; a quiet one was up to twice as fast.)
DOOM is reading your question ...
DOOM says: GOD MODE

  5,320 tile runs in 6 DOOM games | 313,248 values read back and compared with the NumPy integer reference (tolerance 0) | 653 s
EXACT: every value DOOM exposed equals the integer reference, and the reference gives the same answer
```

| you ask | DOOM answers | in the training facts? |
|---|---|---|
| what does iddqd do | GOD MODE | yes, a training paraphrase of fact 52 |
| what is love | INVULNERABILITY | no |
| what is a voodoo doll | BERSERK | no |
| can it run doom | SIXTY SECONDS | no |

The first row is a DOOM run, the others are the NumPy reference. `ask` exits with `MISMATCH` when DOOM differs. Outside
the 189 facts the answer is exact and meaningless. Accuracy:
[`models/doomoracle/EVAL_REPORT.md`](../models/doomoracle/EVAL_REPORT.md).

## `doomformer machine`

A counter or SUBLEQ machine in one map ([machine.md](machine.md)). Python advances the clock and compares the pc and
every register after every step with its interpreter. `doomformer machine list` shows the programs. Options
(`--tamper`, `--clock dolls`, `--export`, `--slots`, `--blind`, `--latch`) are in machine.md.

```text
multiply: 7 * 9
  9 instructions, registers a b c t; the interpreter halts after 464 steps with c = 63 (Python computes 63)
  map: 39 voodoo dolls, 578 sectors, span 17,129 x 4,136
  [...]
c = 63 (expected 63)
  464 steps | 39 dolls | 578 sectors | 126,340 game tics (216 to 216 per step) | 16 s
  audit: 9,610 readout commands, 0 other commands or game calls, 0 state loads (the map starts with the program and the arguments baked in)
  clock: the level's ACS script (rewind, barrier release, settle check, halt-latch read); data path and program counter: dolls
EXACT: the pc and every register of all 464 steps equal the interpreter's (24,180 values, tolerance 0); the halt latch stopped the level
```

## `doomformer learn`

A blank multiclass perceptron learns digits in one map: weights are ceiling heights, and the dolls compute the scores,
the argmax and the update ([learning.md](learning.md#doom-learns-to-read-digits)). After every step all exposed values
and weights are compared with a NumPy integer twin.

```text
$ doomformer learn digits --classes 4 --steps 300 --every 25
digits: 520 training and 200 held-out digits, 37 binary features (centre 6x6 pixels >= 8, and a bias), 4 classes, weights start at 0 (148 weights)
map: 314 dolls, 12,240 sectors, opt 3, estimated 387 tics per step; 170 values read and checked per step: 22 outputs and all 148 weights
-- after     0 examples: held-out accuracy of the weights DOOM holds  25.0% (50/200, NumPy forward over the read-back ceilings)
step     1  label 3  DOOM 0  margin   0  WRONG  updated  36  running   0.0%  EXACT 170/170  296 tics  1.50 s
step     2  label 2  DOOM 3  margin  11  WRONG  updated  30  running   0.0%  EXACT 170/170  296 tics  0.72 s
step     3  label 2  DOOM 2  margin   8  right  no update   running  33.3%  EXACT 170/170  296 tics  0.91 s
[...]
EXACT: 300 steps; after each, 170 exposed values and all 148 weights equal the NumPy integer twin (tolerance 0)
DOOM weights after 300 examples: held-out accuracy 83.5% (167/200); online accuracy over the stream 77.7% (67 updates)
NumPy perceptron, same stream, same clamp (independent code): held-out 83.5%, curve identical to DOOM's: True, 67 updates
NumPy perceptron, same stream, no clamp (textbook; not DOOM): held-out 89.5%, 67 updates
Python sent: 12,300 input writes (pixels and label), 299 rewinds, 300 starts, 0 motion checks, 900 readout commands, 0 other (a write to a weight would be rejected)
total: 88,800 game tics, 298 s for 300 steps (0.99 s and 296 tics per step; timing on a loaded shared machine unless stated)
DOOM forward pass (df.MLP + with_argmax, every value exact) on the first 100 held-out digits with the learned weights: 100/100 winners equal NumPy's argmax, 79/100 correct (14 s)
```

## `doomformer fuzz`, `verify`, `facts`, `lint`

Checks of the checker. Options and results: [verification.md](verification.md) and [scheduling.md](scheduling.md).
`fuzz` compiles random nets, runs them in DOOM and compares every value. `--plant BUG` must exit 1.

```text
$ doomformer fuzz --nets 40 --workers 2
  [10/40] net 8 mlp ok (8 passes, 184 values)
  [...]
fuzz: 40 mlp nets, 306 passes, 5,958 values compared at tolerance 0, 0 mismatches, 0 errors, 0 unsupported (all-zero layer), 0 refused, 60 inputs refused by the range check; seed 0, 9.2 s
```

`verify NET` runs every input class of a small MLP and prints `VERIFIED` only if all runs are exact.

```text
$ doomformer verify and_or.npz --domain 0 1 --mode all
and_or.npz: 2 inputs, 2 layer(s), unsigned input
mode all: every input of [0, 1]^2 (4 inputs) the net accepts: 4 runs; 0 candidates refused by the range check
VERIFIED: 24 values exact against the integer reference (tolerance 0), 0.2 s
```

`facts` runs the engine facts as probe maps in ViZDoom. Other engines: [Other ports](engine_limits.md#other-ports).

```text
$ doomformer facts
engine: ViZDoom 1.3.2 (ZDoom 2.8.1)

PASS  light_saturates    1/1 cases    0.1s  [acs]
PASS  light_same_tic     1/1 cases    0.2s  [acs]
PASS  mover_same_tic     1/1 cases    0.1s  [acs]
PASS  mover_active_lock  1/1 cases    0.1s  [acs]
PASS  mover_speed        1/1 cases    0.1s  [acs]
[...]
28/28 facts hold (55/55 cases) on ViZDoom 1.3.2 (ZDoom 2.8.1) in 8 s
```

`lint WAD` replays every doll of a compiled one-map brain tic by tic. It exits 1 on a hazard or a margin below its
minimum.
