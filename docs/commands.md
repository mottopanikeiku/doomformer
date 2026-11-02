# Commands

Start with `doomformer hello`, which compiles a small MLP and checks five inputs in about 2 s. `EXACT` means the exposed
DOOM values equal the integer reference at tolerance 0, while task accuracy is reported separately.

The output excerpts below omit progress lines. Unless noted otherwise, timings come from an idle 16-core arm64 Linux
machine with up to 8 games running at once, where one-map commands use one game and tile commands take longer with
fewer games, and [scaling.md](scaling.md) has the costs.

| command | what it runs | time |
|---|---|---|
| [`doomformer hello`](#doomformer-hello) | a 4-6-5-3 integer MLP on 5 random inputs | 2 s |
| [`doomformer draw --digit 7`](#doomformer-draw---digit-7) | the marine walks a 7 on the DrawPad floor, the level reads it | 14 s |
| [`doomformer draw --build DIR`](drawpad.md#in-gzdoom-without-python) | writes the DrawPad map as a WAD that GZDoom plays without Python | about 5 s |
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
| [`doomformer machine subleq all`](#doomformer-machine) | a SUBLEQ machine in one map, three programs using the same WAD | 5 min |
| [`doomformer learn digits --classes 4 --steps 300`](#doomformer-learn) | a blank perceptron learns digits in one map, every step checked | 5 min |
| [`doomformer tune --check`](tuning.md) | checks the saved timing constants; without `--check` re-measures them | minutes to hours |
| `doomformer bench idle` / `pass` / `tiles` | engine-time measurements ([scaling.md](scaling.md)) | minutes |
| `doomformer tune --hashes` | compares the opt 0 maps of six models with `doomformer/opt0_hashes.json` | 30 s |
| `doomformer lint WAD` | checks the wave schedule of a one-map brain ([scheduling.md](scheduling.md)) | 1 to 45 s |
| `doomformer lint --fuzz` | compares the schedule checker with DOOM on random nets | about 11 min for 1,000 nets |

## `doomformer hello`

This compiles a 4-6-5-3 integer MLP into one map, runs five random inputs and checks every value. The first run also
builds the ACS compiler, which takes about 10 s on a cold cache.

```text
tiny integer MLP 4-6-5-3 compiled into one DOOM map: 60 voodoo dolls, 1222 sectors
input 1/5  x=[-71, -82, 64, 37] -> y=[11, 4, -15]  EXACT (3 outputs + 22 hidden readouts equal the integer reference; 488 game tics)
...
5/5 EXACT
```

## `doomformer draw --digit 7`

The DrawPad level is one map with a 28×28 floor canvas, a READ switch and a 7-segment display. `--digit D --seed S`
makes the marine walk digit D and press READ, and without `--digit` you draw yourself with the arrow keys, holding Ctrl
to draw and pressing Space on READ.

The command builds at `--opt 3`, so a READ takes 273 game tics. With `--build DIR` it writes the map to
`DIR/drawpad.wad` for GZDoom instead of starting a game, `--export DIR` writes the self-test that other ports run, and
[drawpad.md](drawpad.md) explains how it all works.

<p align="center"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/drawpad.gif" alt="The marine walks a digit on the DrawPad floor and DOOM reads it" width="720"></p>

```text
DOOM reads: 7  (right)   -- the winner's doll lit it on the 7-segment display
EXACT: all 248 values DOOM computed (hidden layer, 45 score comparisons and 10 win counts, each with its twin, and the one-hot answer) equal the NumPy integer reference computed from the 784 pixel ports DOOM holds (tolerance 0); one display firing
pixels: the 784 ports equal the level's canvas (the tiles the pen inked)
```

## `doomformer mnist`

This runs the DrawPad network on MNIST test digits, reading 4 by default with one game per digit.

* `--digits N` samples N digits.
* `--all` reads all 10,000.
* `--image my7.png` reads your own image.
* `--workers`, `--ray` and `--record PATH`: see [scaling.md](scaling.md).

<p align="center"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/mnist.png" alt="MNIST digits read by the one-map network in DOOM" width="720"></p>

```text
$ doomformer mnist --digits 2 --seed 1

DOOM reads: 1  right            DOOM reads: 3  right
test #46                        test #51

2/2 read correctly | 2/2 EXACT: every value DOOM computed (248 per digit: hidden layer, comparisons, win counts and
the one-hot answer) equals the NumPy integer reference (tolerance 0); Python wrote the 784 pixels, released the dolls and compared | 5,696 game tics per digit | 88 s
```

### `--tiles`, `--opt` and `plan`

`--tiles` runs the network as four tile maps in resident games, with Python copying each layer's values into the next
tile, and the 10,000-digit result is in [scaling.md](scaling.md#all-10000-mnist-test-digits-in-doom).

```sh
doomformer mnist --digits 2 --seed 1 --tiles --workers 2
```

That run uses 1,351 dolls and 111,112 sectors across four maps per worker. Both digits are correct and all 248 values
per digit match the reference, in 216 game tics per digit summed over the tiles and 4 s in total.

`--opt N` picks the one-map schedule ([scheduling.md](scheduling.md)), and at `--opt 3` an ACS script copies each
layer's light levels into the next layer's gates.

```sh
doomformer mnist --digits 2 --seed 1 --opt 3 --workers 2
```

This build has 1,355 dolls and 88,146 sectors, and the same two digits are correct and exact in 272 game tics per
digit, 8 s in total.

`doomformer plan MODEL --opt N --explain` counts dolls and sectors without building the map, predicts tics and seconds
per pass and recommends one map or tiles. MODEL is a shipped name, a model directory or an MLP `.npz`, and `--explain`
adds the schedule per layer and what each pass saves.

```sh
doomformer plan models/mnist --explain --opt 3
```

For this model, `plan` reports a 31,969 × 5,192-unit map, 276 predicted game tics per pass and a 304-tic geometry bound,
with an estimated engine time of 2.8 s in one map or 752 ms in tiles once the games are loaded. `mode="auto"` chooses
the map because it fits, and `--explain` adds the per-layer schedule and the cumulative effect of each optimisation.

## `doomformer demo`

A 1-layer transformer, `doomnames` with d=16 and 2 heads, writes a DOOM level name over 38 tile maps. Python copies the
integers between games and samples each letter from the logits DOOM computed, and the options are `--seed N`,
`--letters N`, `--prompt TEXT`, `--temperature T` and `--png PATH`.

```text
model: the shipped doomnames: 1 layer, d=16, 2 heads, context 16, ReLU attention, trained on 136 DOOM level names
sampling: temperature 180, seed 20
38 tile maps (10,462 voodoo dolls, 135,912 sectors) built in 0 s; running them in 4 DOOM game processes ...

DOOM wrote: 'GHEER'   (not one of its training names)
  174 tile runs in 4 parallel DOOM games | 10,462 voodoo dolls in the tile maps | 6,902 values read back and compared with the NumPy integer reference (tolerance 0) | 22 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

### Softmax attention, modern and the flags

`--softmax` runs the model trained with softmax attention, whose construction is in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md#softmax-attention).

```text
DOOM wrote: 'FINAL INEALL'   (not one of its training names)
  494 tile runs [...] | 4,886 voodoo dolls in the tile maps |
  106,834 values read back and compared with the NumPy integer reference (tolerance 0) [...]
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

`--model doomnames_modern` adds RMSNorm and a gated MLP to softmax attention ([HOW_IT_WORKS.md](HOW_IT_WORKS.md#rmsnorm)).

## `doomformer demo --nightmare`

`--nightmare` runs `doombrain`, a 2-layer transformer with 24,279 parameters, trained on DOOM level names and compiled
into 106 tile maps.

<p align="center"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.1.0/demon_canyon.gif" alt="DOOM writes DEMON CANYON letter by letter" width="720"></p>

```text
model: the shipped doombrain: 2 layers, d=32, 4 heads, context 24, ReLU attention
sampling: temperature 150, seed 2
106 tile maps (56,390 voodoo dolls, 785,460 sectors) built in 10 s; running them in 6 DOOM game processes ...

DOOM wrote: 'DEMON CANYON'
  1,690 tile runs in 6 parallel DOOM games | 56,390 voodoo dolls in the tile maps | 120,666 values read back and compared with the NumPy integer reference (tolerance 0) | 210 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same name
```

More samples are in [`models/doombrain/eval_report.txt`](../models/doombrain/eval_report.txt), and single-neuron
lesions are in [DEMOS.md](DEMOS.md#lesion-statistics) and [interventions.md](interventions.md).

## One map, and `--loop`

`--one-map` runs a smaller model, `doomnames_onemap`, with no Python between layers, so Python only writes the token and
samples the letter after each pass.

```text
model: the shipped doomnames_onemap: 1 layer, d=8, 1 head, context 6, ReLU attention, trained on DOOM level names of up to 5 letters
map built in 5 s: 3,548 voodoo dolls cross lava lines in 95,448 sectors

DOOM wrote: 'D'
  3,548 voodoo dolls | 95,448 sectors | 176 values read back and compared with the NumPy integer reference (tolerance 0) | 3,488 game tics per pass | 89 s
EXACT: every value DOOM exposed equals the integer reference, and the reference writes the same text
```

`--one-map --loop` builds one map that writes a whole name, because the winner's doll writes the next token and the
level then runs the next pass by itself while Python only advances the clock and reads. `--softmax` switches to the
softmax model, and the results are in [HOW_IT_WORKS.md](HOW_IT_WORKS.md#generation-without-python).

`--until-eos` (with `--loop`) stops the level after the pass whose newest token is EOS, and the map is the same file
for every prompt of one length ([HOW_IT_WORKS.md](HOW_IT_WORKS.md#stopping-at-eos)).

## `doomformer ask`

A 2-layer transformer with d=32 and 4 heads, trained on 189 facts about DOOM, answers a question letter by letter over
tile maps in at most 8 games.

```text
You ask DOOM: WHAT DOES IDDQD DO?
DOOM answers with a 2-layer transformer (d=32, 4 heads) trained on 189 facts about DOOM, split into tile maps over 6 DOOM games.
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

The first row ran in DOOM and the others use the NumPy reference, and questions outside the 189 training facts can
return unrelated answers. `ask` exits with `MISMATCH` if DOOM differs from the reference, and its accuracy is in
[`models/doomoracle/EVAL_REPORT.md`](../models/doomoracle/EVAL_REPORT.md).

## `doomformer machine`

This runs a counter or SUBLEQ machine in one map ([machine.md](machine.md)), and after every step Python compares the
pc and every register with its interpreter. `doomformer machine list` shows the programs, and [machine.md](machine.md)
covers `--tamper`, `--clock dolls`, `--export`, `--slots`, `--blind` and `--latch`.

```text
multiply: 7 * 9
  9 instructions, registers a b c t; the interpreter halts after 464 steps with c = 63 (Python computes 63)
  map: 39 voodoo dolls, 578 sectors, span 17,129 x 4,136
  [...]
c = 63 (expected 63)
  464 steps | 39 dolls | 578 sectors | 126,340 game tics (216 to 216 per step) | 16 s
EXACT: the pc and every register of all 464 steps equal the interpreter's (24,180 values, tolerance 0); the halt latch stopped the level
```

## `doomformer learn`

A blank multiclass perceptron learns digits in one map, where its weights are ceiling heights and the dolls compute the
scores, the argmax and the update ([learning.md](learning.md#doom-learns-to-read-digits)). After every step, all
exposed values and weights are compared with a NumPy integer twin.

<p align="center"><img src="https://github.com/mottopanikeiku/doomformer/releases/download/v0.6.0/learn_1500.png" alt="Held-out accuracy of the perceptron DOOM trains over 1,500 steps" width="720"></p>

```sh
doomformer learn digits --classes 4 --steps 300 --every 25
```

The four-class example starts with 148 zero weights, learns from 520 training digits and checks 170 values per step,
including every weight. After 300 steps its held-out accuracy is 83.5% (167/200), the same as a NumPy perceptron with
the same clamp, which gets 89.5% without the clamp.

The run takes 88,800 game tics and 298 s on a loaded machine, and a separate DOOM forward pass with the learned weights
matches NumPy's winner on all 100 tested held-out digits and classifies 79 correctly.

## `doomformer fuzz`, `verify`, `facts`, `lint`

`fuzz` compiles random networks and compares exposed values with the integer reference, `--plant BUG` inserts a compiler
bug and must exit 1, and [verification.md](verification.md) has the options and results.

```text
$ doomformer fuzz --nets 40 --workers 2
  [10/40] net 8 mlp ok (8 passes, 184 values)
  [...]
fuzz: 40 mlp nets, 306 passes, 5,958 values compared at tolerance 0, 0 mismatches, 0 errors, 0 unsupported (all-zero layer), 0 refused, 60 inputs refused by the range check; seed 0, 9.2 s
```

`verify NET` runs every input class of a small MLP; `--mode all` enumerates the chosen input box.

```text
$ doomformer verify and_or.npz --domain 0 1 --mode all
and_or.npz: 2 inputs, 2 layer(s), unsigned input
mode all: every input of [0, 1]^2 (4 inputs) the net accepts: 4 runs; 0 candidates refused by the range check
VERIFIED: 24 values exact against the integer reference (tolerance 0), 0.2 s
```

`facts` runs the engine facts as probe maps in ViZDoom, and [Other ports](engine_limits.md#other-ports) covers other
engines.

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

`lint WAD` replays every doll of a compiled one-map brain tic by tic and exits 1 on a hazard or a too-small margin
([scheduling.md](scheduling.md)).
