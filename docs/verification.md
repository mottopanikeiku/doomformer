# Verification

What the exactness check compares, how much of it is tested, and what `fuzz`, `fuzz --mutate` and `verify` add. Every
number here comes from a command in this page; the JSON records are in [models/verify/](../models/verify/).

## What is compared

Every pass of a brain (`brain(x)`, `brain.logits(tokens)`, `brain.step(spikes)`, `df.run_many`) compares every value
the map exposes with the integer reference (`intref`, five lines of NumPy for an MLP) at tolerance 0. The exposed values
of an MLP are its output lights, every hidden light `h{l}_{j}` and every hidden gate floor `-h{l}_{j}`; of a transformer
every pre-activation, score, softmax and RMSNorm intermediate, logit. Three guards make the check itself hard to
weaken:

* The set of names a pass reads back must equal the set the lowering exposes (`MLP.exposed_names()`,
  `onemap.exposed_names()`), otherwise `df.ReadoutError`. Before, deleting a name from `meta.pages` made `checked` one
  smaller and nothing complained.
* `df.compile(net)` defaults to `guard=True`, including spiking nets; tile maps always guard their readouts.
  Before **each page**, an ACS script fills every USER slot with the poison value 100,003 (outside the light/height
  range). Python checks that the guard ran, then requires the page to overwrite the poison. A dropped guard or page
  raises `RuntimeError`, even when the input repeats or adjacent pages contain identical values. Previously the
  opt-in guard ran only before the first page. `guard=False` explicitly opts out for historical one-map reproduction.
  The guard changes only readback globals, not model state. It costs one more tic per page (`TileMeta.read_tics`).
  `compile_pure` guards by default too, so the digit learner, the counter, stored-program and SUBLEQ machines (both
  clocks), self-generation, `df.Agent`, *his brain is the level* and the live-demons run read through it. Where a map
  holds its results for a fixed window, the window is sized from the guarded readout and is as long as before whenever
  the guarded pages fit its old margin (up to 24 pages for the machines and self-generation; the policy's 9 pages
  take 18 of the levels' 32 tics), so those runs' tics per step do not change; larger maps get a longer window. The
  readout scripts change no game state, so the level plays the same game however Python spends the window. One demo
  keeps the unguarded readback on purpose: the fly loop. Its 9 pages take 18 tics guarded, more than its 16-tic HOLD
  window, which is world time (the walkway carries him through it); a longer window would change the documented
  runs. It still compares every spike and potential at every step, so there a dropped page is caught only when the
  stale values differ from the reference. Custom readout loops (such as DrawPad's ACK handshake) keep their own
  protocol; the script alone does not protect a custom reader.
* `df.compile(transformer)` runs `intref.check_limits` over the parameters and `kit.limit_sequences(net)` (every single
  token after BOS plus 48 seeded random sequences) and raises `RangeError` before building. `limits=False` skips it.
  That sample is not every token sequence, so a one-map `brain.logits(tokens)` also runs `check_limits` on the sequence it
  is about to run (same range rules and the plane-move bound below) and raises `RangeError` before DOOM does; before
  this change an out-of-range sequence failed at run time as a DOOM-vs-reference mismatch (saturated light at 32,767).
  `limits=False` skips this check too. Tile brains do not run it.

The check compares what DOOM computes with what the lowering says it computes. It does not compare the lowering with the
mathematics of the original float model (that is `quantization_report`), and a value the map does not expose is not
compared.

## The plane-move bound

In one map a value that feeds a gate is a plane height: dolls raise and lower it with `Floor_`/`Ceiling_RaiseByValue` and
`LowerByValue` lines at mover speed 150,000, which is 18,750 units per tic in 16.16 fixed point. A move that starts with
the plane above 14,018 units (32,768 - 18,750) still ends at the right height, but its mover stays busy for 3 tics
instead of 1, and a move requested on that plane in tics 1 and 2 is dropped. The edge is not symmetric and differs by one
unit between floors and ceilings (`doomformer facts --only mover_busy`, [models/verify/mover_busy_probe.txt](../models/verify/mover_busy_probe.txt);
the longer sweep is [models/verify/mover_busy_edges.txt](../models/verify/mover_busy_edges.txt)):

| plane | move | starts fine at | busy from |
|---|---|---|---|
| floor | raise | 14,018 | 14,019 |
| floor | lower | -14,017 | -14,018 |
| ceiling | raise | 14,017 | 14,018 |
| ceiling | lower | -14,018 | -14,019 |

The compiler's waves keep different dolls' moves on one plane at least 4 tics apart, so a 3-tic window is harmless
inside the bound and loses a move outside it. This was a miscompile of v0.5.0, found by `doomformer fuzz --edge
--seed 902`, net 5; replay [replay_mlp_edge_seed902_net5.json](../models/verify/replay_mlp_edge_seed902_net5.json)):
`MLP([[[3,255,255,0,255]],[[-255]]], [[-14282],[185]], [[713,1423,3182,11867],[8540]])` at `x = [-12000, 5933, 6252,
-353, 11483]`. The gate plane `-h1_0` starts at 14,282; four dolls raise it by 3 and nine lower it by 255; the fourth raise
starts above 14,018, so one 255 lower is lost: DOOM 12,254, reference 11,999. The final value (11,999) and every partial
sum (at most 14,294) were inside the old contract (|h| <= 12,000, partial sums <= 30,000), so `net.check` accepted it.

**The contract.** A plane with bias `b` whose terms raise it by `P` in total and lower it by `N` in total (dolls cross the
lines in any order) needs `b + P <= 14,018` when `P > 0` and `b - N >= -14,018` when `N > 0`. Every plane also has to start inside
`[-15,000, 16,000]` (next subsection). `P` stands for the whole sum of raises, not for the sum without the last one, so the rule is
stricter than the engine by at most the smallest raise (one weight). It is enforced in three places (the start bound in the first two):

* `net.check(x)` (MLP, every pass of a one-map brain, `run_many` in map mode): the plane rule on the activations `x`
  produces, so every input `check` accepts gives DOOM = reference at every `opt`. Tile mode has no doll-moved planes
  (Python writes every gate value, the accumulators are lights): tile brains and `run_many(mode="tiles")` call
  `check(x, planes=False)`.
* `df.compile(net)` (map mode): for an MLP the worst case over ALL inputs (the staircase counts propagated as intervals;
  exact for layer 0), for a transformer `intref.check_limits(planes=True)` over `kit.limit_sequences`. A violation raises
  `RangeError` naming the plane and the bound. `limits=False` skips this check, and `net.check(x)` still refuses the
  inputs that violate it. Nets from `from_numpy` and `from_torch` keep `|b| + sum|W| K <= 12,000` and stay inside.
* `SpikingBrain.step`: the potentials are state, unbounded at compile time, so each step is checked before it runs
  (the fire layer's moves from the state, then the synapse layer's moves from the result).

**Where a plane may start.** A floor port's sector has its ceiling at 16,000 (`CEIL`, `tile.py`), and a ceiling port's
sector has its floor at -15,000 (`FLOOR_PIT`, `pure.py`). The map is built with the plane at its bias, so the first pass
is exact; the reset script moves it back with `Floor_MoveToValue`/`Ceiling_MoveToValue`, and a floor stops at the ceiling
and a ceiling at the pit floor. A floor plane that starts at 16,001 and never rises is off by one on every later pass
(DOOM 11,920, reference 11,921; the repro `MLP([[[255,255,255,255]],[[-255]]], [[-16001],[0]], [[10,20,30,40],[100]],
signed_input=False)` at `x = [45]*4`), at 16,010 by 10, at 17,000 by 1,000 at opt 0, 1 and 2; opt 3 is exact (one move from
the light, no clamp in its path); 16,000 is exact. The old contract said such a plane "may start anywhere": that was wrong
above 16,000. Measured in [plane_start_probe.txt](../models/verify/plane_start_probe.txt): the ACS clamp (floor 16,000,
ceiling -15,000) and the low side. Floors have no clamp below; they are exact down to -25,500, and a reset that lowers
a plane through a third mover step (start more than 37,500 below the previous pass's final value, i.e. below -25,500 when
that value is 12,000) reads a wrapped height. The bound is `START_MIN = -15,000`, `START_MAX = 16,000` for every plane,
floor or ceiling, the intersection that is safe for both (floors are exact below -15,000 but the check does not know the
kind of a transformer plane). For an MLP the plane is the twin `-h` and starts at `-bias`: a hidden bias must lie in
`[-16,000, 15,000]`. For a transformer every bias is at most 255 in magnitude (`W_MAX`), so the bound holds; a spiking
state starts within +-12,000 (`V_MAX`). `check`, the compile-time check and `check_limits(planes=True)` enforce it
(`planes.start_errors`); the fuzz generator draws `--edge` nets whose twin plane starts 0 to 9 units inside either end.
Not tested: ceiling-plane starts in a real map (only in the ACS clamp probe).

Measured on the reduced net with the check bypassed (`doomformer verify --plane-sweep`, no arguments: these seven rows; record
[plane_bias_sweep.json](../models/verify/plane_bias_sweep.json); "peak" is the rule's `b + P`):

| bias | peak | `check` | DOOM `-h1_0` | reference | |
|---|---|---|---|---|---|
| -12,000 | 12,012 | accepts | 9,717 | 9,717 | exact |
| -12,001 | 12,013 | accepts | 9,718 | 9,718 | exact |
| -14,006 | 14,018 | accepts | 11,723 | 11,723 | exact |
| -14,007 | 14,019 | refuses | 11,724 | 11,724 | exact (the rule is one move conservative) |
| -14,009 | 14,021 | refuses | 11,726 | 11,726 | exact (fourth raise starts at 14,018) |
| -14,010 | 14,022 | refuses | 11,982 | 11,727 | differs by 255 (fourth raise starts at 14,019) |
| -14,282 | 14,294 | refuses | 12,254 | 11,999 | differs by 255 (the replay) |

Schedule levels 0, 1 and 2 give the same rows (at opt 2 the replay reads 12,251). Level 3 passes the whole sweep, the replay
included ([plane_bias_sweep_opt3.json](../models/verify/plane_bias_sweep_opt3.json)): it copies each hidden light into its
plane with one `Floor_MoveToValue`, so no doll moves the plane. The check does not know `opt` and refuses the input at
every level. Shipped transformers: the largest plane partial sum over `kit.limit_sequences` is 13,224
(`doomnames_modern`, 794 below the bound) and under 10,000 for the others. `doomnames_modern` has no one-map build (its map
spans 31,989 x 1,313,928 units, so it runs in tiles); over every name of its corpus the largest plane partial sum is
14,848, so a one-map build would break the rule on some names, and `tests/test_arch.py` checks its limits without it. Not tested: ceiling-plane edges inside a real map (they are measured only in the ACS probe) and transformer
sequences outside `limit_sequences`.

## Coverage: how many of the reference's values does a pass compare

`df.coverage(brain_or_net)` (also `doomformer verify --coverage NET [--tiles]`; module `doomformer/exposure.py`) runs one
pass and lists, for the MLP, the spiking net or the transformer, every value the integer reference defines for that pass,
how many the map reads back, how many the check compares, and which are missing. The reference side is built from
`net.reference(x)` / `IntNet.step` / `intref.forward`, not from the lowering, so a lowering that stops exposing a value
shows up. For a one-map brain and for MLP tiles the comparison is measured: the pass's readback is replayed once per
exposed name with that value changed by 1, and the name counts only if the brain's own check raises `MismatchError`
naming it (`detected`). `brain.info["checked"]` must equal the reference total. The staircase counts (input counts,
hidden activations, attention and MLP activations, gate comparisons) are defined by the reference and exposed by no map;
the report lists their number, and they are checked only through the sums they feed.

MNIST (`doomformer verify --coverage models/mnist/mlp.npz`, [one map](../models/verify/coverage_mnist_map.json) and
[tiles](../models/verify/coverage_mnist_tiles.json), input zeros): the reference defines 129 values (119 hidden
pre-activations, 10 outputs) and the lowering creates 119 negated twins; the check compared 248 of 248, a change by 1 was
detected for 248 of 248 names; not exposed: 903 staircase counts and 1,351 gate comparisons.

Tests: `tests/test_exposure.py` fails if the compared count is below the reference's total for an MLP (one map, tiles), a
spiking net and a transformer (one map, tiles), and has a negative control that drops one readout name (editing the page
and the brain's expected set, which bypasses the v0.5.0 name-set guard) and requires the report to name it.
Spiking maps read back the spikes and the potentials; the twin planes `-s` are read by the synapse gates only and are
listed as not exposed.

## Method: translation validation

Each compile is followed by a validation of that one run of the compiler, instead of a proof that the compiler is
correct for all nets. This is the notion Pnueli, Siegel and Singerman call translation validation: "each individual
translation (i.e. a run of the compiler) is followed by a validation phase which verifies that the target code produced
on this run correctly implements the submitted source program" (A. Pnueli, M. Siegel, E. Singerman, *Translation
validation*, TACAS 1998, LNCS 1384, pp. 151-166, <https://link.springer.com/chapter/10.1007/BFb0054170>, abstract page
opened for this document). Their validator is automatic and builds a refinement proof between two models. Doomformer's
validator is weaker: a test. It runs the target (the map in the engine) on chosen inputs and compares values with the
source semantics, so it can find a wrong translation but proves nothing about inputs it did not run. `verify` closes
that gap for small nets under one stated assumption; `fuzz` samples many nets. Related work: [related_work.md](related_work.md).

## `doomformer fuzz`

```text
doomformer fuzz --nets 40 --seed 0 --workers 3          # MLPs
doomformer fuzz --kind transformer --nets 100 --workers 3 --inputs 3
doomformer fuzz --kind spiking --nets 100
doomformer fuzz --edge --nets 500                       # weights +-255, hidden values at +-12,000, thresholds at 12,000,
                                                        # hidden planes at the plane-move bound
doomformer fuzz --edge --nets 5000 --seed random --workers 6   # a fresh seed, printed ("fuzz: random seed N")
doomformer fuzz --edge --nets 5000 --tiles              # MLPs in tile mode (one game per layer)
doomformer fuzz --replay fuzz-failures.json             # re-run the cases saved by a failing run
doomformer fuzz --nets 12 --plant gate-gt               # self-test: must exit 1
doomformer fuzz --nets 300 --opt 1                      # the compiler's schedule level (when df.compile has `opt`)
```

Net `i` of seed `S` is drawn from `SeedSequence([S, i, kind])`, so a case does not depend on the worker count. Each
worker compiles a net into a temporary WAD, starts a game, runs the net's inputs and closes it; a mismatch, an
unexpected exception or a dead worker is a failure (exit status 1), the failing cases are written to a replay file
(`--save-failures`, default `fuzz-failures.json`) and the summary to `--json` (with `opt`, `tiles` and `source_sha256`, a
hash of every `doomformer/*.py` the run used). MLPs compile with `limits=False`: the per-input contract (`net.check`
accepts, DOOM must agree) is what is fuzzed, and edge nets are meant to leave the all-inputs bound. A net whose inputs
are all refused is reported as `refused` with the first refusal, not as a pass. `--seed random` draws a fresh seed and
prints it first; a net of that seed is `--seed N --start i --nets 1`. The generator changed with the plane-move bound
(edge nets also shift a hidden bias so that a plane lands 0 to 3 units under the bound), so the nets of the seeds in the
older tables below are those of the v0.5.0 generator.

Generators:

* MLP: depth 1 to 4, 1 to 5 inputs, widths 1 to 8, signed or unsigned input, 1 to 4 thresholds per staircase with
  duplicates (first thresholds up to 12,000; later thresholds drawn from the hidden values the calibration inputs
  produce, so inputs land on them), weights up to +-255 (a quarter of the nets, and every `--edge` net, are range-edge
  nets: many weights at +-255, biases up to +-12,000, thresholds at 12,000 or 11,999; the shift below can move a bias
  further, to about +-14,300). Inputs: zero, +-1, +-12,000, the
  `t-1, t, t+1` neighbours of each first-layer threshold with both signs, random inputs, and inputs for which a hidden unit
  sits exactly on a later threshold neighbour or on +-12,000 / +-11,999 (the generator shifts that unit's bias), and, for
  `--edge` nets of depth 2 or more, inputs for which a hidden plane's partial sum `b + P` lands 0 to 3 under the
  plane-move bound or `b - N` 0 to 3 above its negative. Inputs the range check refuses are counted, not run
  (`inputs refused`).
* Transformer: all 8 combinations of `attn` relu/softmax, `norm` none/rms and `mlp` relu/reglu, 1 to 2 layers, `d` 1 to 4,
  `H` 1 to 2, `T` 2 to 3, vocabulary 3 to 5, duplicate thresholds, weights +-3, accepted when `check_limits` passes;
  inputs: BOS only, short random prefixes and one full-length sequence.
* Spiking: 2 to 6 neurons, 1 to 3 inputs, theta 2 to 8, leak 0 to 2, 8 steps (the state stays in the map).
* All-zero weight layers cannot be lowered (no doll carries them): `df.compile` raises `ValueError: weights[1] is all
  zero ...` for an MLP and `layer N (name) has no dolls: every weight into it is zero ...` for a transformer. The MLP
  generator makes a few of them (3 %) and requires that message; they are reported as `unsupported`, not as passes.

Planted bugs (`--plant NAME`, the harness's own test; each patches one lowering function): `gate-gt` (`x > t` instead of
`x >= t` at an input gate), `w255` (weight lines clamp at 254), `bias-twin` (the negated twin of a negative hidden bias is
off by one), `dup-thr` (duplicate thresholds collapse). `tests/test_verification.py` runs 12 nets per bug in the test
process and requires at least one mismatch for each; CI runs `fuzz --nets 40` and then requires `fuzz --nets 12 --plant
gate-gt` to fail. Measured on the v0.6.1 tree: `doomformer fuzz --nets 40 --seed 0 --workers 2` ran 40 nets, 306 passes, 5,946 values, 0
mismatches, 60 inputs refused by the range check in 9.2 s (v0.5.0: 304 / 5,914 / 61; v0.6.0 printed 308 / 5,975 / 57; v0.7.1 prints 306 / 5,958 / 60: the
generator changed with each plane bound); `--nets 12 --plant gate-gt` found the bug (10 of 12 nets mismatch, first at net 1) in 3.3 s (shared
machine, load average 16).

### Results

Commands run on the tree of this release (shared 16-core machine, load average 10 to 40, `OMP_NUM_THREADS=1`, 2 games
at once; records in [models/verify/](../models/verify/)). `values` counts the exposed values compared at tolerance 0 over
all passes; `unsupported` nets have an all-zero weight layer and were refused with the message above (the zero-layer
generators are part of the campaign, they are not exact passes).

| command | nets | passes | values compared | mismatches | unsupported | wall time |
|---|---|---|---|---|---|---|
| `fuzz --nets 2000 --seed 0 --workers 2` (MLP) | 2,000 | 15,374 | 276,594 | 0 | 42 | 542 s |
| `fuzz --edge --nets 500 --seed 1 --workers 2` (MLP, range edge) | 500 | 3,799 | 68,750 | 0 | 0 | 118 s |
| `fuzz --nets 300 --seed 2 --workers 2 --opt 1` (MLP, schedule level 1) | 300 | 2,317 | 40,620 | 0 | 5 | 62 s |
| `fuzz --nets 300 --seed 3 --workers 2 --opt 3` (MLP, schedule level 3) | 300 | 2,280 | 38,892 | 0 | 10 | 45 s |
| `fuzz --kind transformer --nets 100 --seed 0 --workers 2 --inputs 3` | 100 | 297 | 54,675 | 0 | 1 | 1,060 s |
| `fuzz --kind spiking --nets 100 --seed 0 --workers 2` | 100 | 800 | 6,432 | 0 | 0 | 23 s |

The 100 transformers cover all 8 variants (attn relu/softmax, norm none/rms, mlp relu/reglu; 9 to 18 nets each, 297
passes). That run started before the scheduling fix listed below and was not repeated on the final tree (the repeat was
cut off); the MLP rows were run on the final tree. `--inputs`, `--opt` and the seeds are the ones printed.
Inputs refused by the range check (2,025 in the 2,000-net run) are not counted as passes.

#### Campaign v2 (after the plane-move bound)

Seeds drawn once with `secrets.randbelow(2**31)` and not used before: 1546066799 78056607 1712291335 1659255519
1705628535 570427258 1447945148 1090071893 327667345 1382960168. Command per chunk (1,000 nets; `S` the seed, `MODE` one
of `--opt 0`, `--opt 3`, `--tiles`; seeds 1, 4, 7, 10 of the list ran at opt 0, seeds 2, 5, 8 at opt 3, seeds 3, 6, 9 in
tile mode):

```text
doomformer fuzz --edge --nets 1000 --start K --seed S --workers 6 MODE -q --json models/verify/v2/edge_MODE_seedS_cC.json
```

Shared 16-core machine at load average 30 to 45, six games at once, so the wall times are 2 to 3 times those of an idle
machine. The target was 5,000 nets per seed; this ran 1,000 to 2,000 per seed (15,000 nets in all) before I stopped it,
because the machine ran at 2 to 7 nets per second. The records are per chunk
([models/verify/v2/](../models/verify/v2/)); the table sums the chunks of one seed. Every chunk ran on one source tree
(`source_sha256` 28a519c0c52354a0 in the JSON, a snapshot taken after the plane-move fix; that hash is of a working-tree snapshot
and differs from the hash of the tagged v0.6.0 tree, which I did not re-run). From v0.6.1 on every chunk JSON records `start`
(the first net index) and `source_sha256` of the code that ran, so a chunk is `--seed S --start K --nets 1000`; the chunks
below do not carry `start`.

| seed | mode | nets | passes | values compared | mismatches | errors | inputs refused by the range check | seconds |
|---|---|---|---|---|---|---|---|---|
| 1546066799 | opt 0 | 2,000 | 15,162 | 269,363 | 0 | 0 | 6,536 | 654 |
| 78056607 | opt 3 | 2,000 | 15,091 | 262,458 | 0 | 0 | 6,487 | 377 |
| 1712291335 | tiles | 2,000 | 15,211 | 258,873 | 0 | 0 | 6,039 | 655 |
| 1659255519 | opt 0 | 2,000 | 15,167 | 267,664 | 0 | 0 | 6,397 | 452 |
| 1705628535 | opt 3 | 2,000 | 15,073 | 261,365 | 0 | 0 | 6,401 | 266 |
| 570427258 | tiles | 1,000 | 7,603 | 134,524 | 0 | 0 | 3,092 | 421 |
| 1447945148 | opt 0 | 1,000 | 7,526 | 129,867 | 0 | 0 | 3,192 | 319 |
| 1090071893 | opt 3 | 1,000 | 7,563 | 130,583 | 0 | 0 | 3,430 | 143 |
| 327667345 | tiles | 1,000 | 7,650 | 140,626 | 0 | 0 | 2,824 | 236 |
| 1382960168 | opt 0 | 1,000 | 7,559 | 132,309 | 0 | 0 | 3,288 | 170 |
| **all** | opt 0 6,000, opt 3 5,000, tiles 4,000 | **15,000** | **113,605** | **1,987,632** | **0** | **0** | | |

#### Campaign v3 (after the start bound, v0.6.1 tree)

Fresh seeds from `secrets.randbelow(2**31)`: 2109258237 (opt 0), 19503025 (opt 3), 1794520407 (tiles); 1,000 edge nets each,
3 games, idle machine, `--start 0`; command `doomformer fuzz --edge --nets 1000 --start 0 --seed S --workers 3 MODE -q --json
models/verify/v2/edge_MODE_seedS_c0_r3.json`. These chunks record `start` and `source_sha256` (869459c8dbb46665, the tree
when they ran; `mutate.py`, tests and docs changed afterwards and nothing else).

| seed | mode | nets | passes | values compared | mismatches | errors | inputs refused | seconds |
|---|---|---|---|---|---|---|---|---|
| 2109258237 | opt 0 | 1,000 | 7,598 | 136,241 | 0 | 0 | 3,167 | 144 |
| 19503025 | opt 3 | 1,000 | 7,508 | 133,202 | 0 | 0 | 3,360 | 92 |
| 1794520407 | tiles | 1,000 | 7,576 | 133,036 | 0 | 0 | 3,260 | 181 |

The `--edge` generator now also draws nets whose twin plane starts 0 to 9 units inside `[-15,000, 16,000]` ("wide one-way
planes"; roughly 1 net in 100 has a plane start beyond +-14,018).

#### Campaign v4 (high-side starts, spiking at the edge, v0.7.1 tree)

`--edge` MLP nets now also draw a unit-0 plane that starts in [15,000, 16,000] and only lowers (about 5 % of nets; 42 of the
1,000 opt 0 nets and 26 of the 500 opt 3 nets start a plane at or above 15,000, 8 and 9 below -14,018; before, 10 of 10,000
started above 14,018). Fresh seeds from `secrets`, `--start 0`, 3 games, idle machine; `source_sha256` in each JSON
(the tree when they ran; later changes are `mutate.py` labels, tests and docs).

| seed | mode | nets | passes | values compared | mismatches | errors | inputs refused | seconds |
|---|---|---|---|---|---|---|---|---|
| 326047183 | opt 0 | 1,000 | 7,380 | 132,757 | 0 | 0 | 3,924 | 140 |
| 1732841896 | opt 3 | 500 | 3,643 | 63,945 | 0 | 0 | 2,148 | 46 |
| 959462705 | tiles | 500 | 3,759 | 63,461 | 0 | 0 | 1,677 | 89 |
| 317167311 | spiking `--edge` | 200 | 12,794 | 105,936 | 0 | 0 | 250 | 62 |

The tiles run generates its inputs with the tile-mode range check (no plane rules), so it has no high-side draw. Spiking
`--edge` now draws a threshold of 2,000 to 12,000 and weights up to 255, and builds each net's input trajectory with the
reference (a step the V_MAX or plane check would refuse is replaced by no input): 67 of the 200 nets reach |v| >= 10,000 and
the largest is 11,999, so potentials sit near +-12,000 and the plane start bound of +-14,018 is only approached, not
crossed (the check forbids more). Transformers stay unchanged under `--edge` (weights +-3, biases within `W_MAX`).

`--edge` changes MLP and spiking nets only: `gen_transformer` ignores it (weights +-3, every bias within `W_MAX`), so the
transformer runs are the ordinary generator, and so was the 300-net spiking run below; the per-pass transformer plane check
is exercised at the edge only by `tests/test_planes.py`.

On the same tree: `fuzz --kind spiking --nets 300 --seed 31337 --workers 3` (2,400 passes, 19,016 values, 0 mismatches,
118 s) and `fuzz --kind transformer --nets 100 --seed 31338 --workers 3 --inputs 3` (297 passes, 60,890 values, 0
mismatches, 99 ok and 1 net with an all-zero weight matrix reported unsupported, 1,859 s); records `spiking_300_seed31337.json` and
`transformer_100_seed31338.json` (source hash 8d487aae38667bf3, an earlier snapshot of the same tree: it lacks the
per-pass transformer range check and `verify --coverage`, which do not change what these runs build). Edge nets with
`--tiles` are generated with the tile-mode range check (no plane rule), so they include inputs that the one-map check
refuses. Tile mode compiles each net into one tile per layer and runs every input through `TiledMLPBrain` with one live
game per layer.

CI: every push runs `fuzz --edge --nets 60 --seed random --workers 2` after the fixed-seed step, and
`.github/workflows/fuzz.yml` runs a longer campaign (opt 0, opt 3, tiles, spiking, transformers) every Monday with random
seeds; each step prints `fuzz: random seed N`, and a failing step uploads its replay file.
A manual run of `fuzz.yml` on the v0.7.0 commit 7ed8236 ([run 40747096084](https://github.com/mottopanikeiku/doomformer/actions/runs/40747096084),
13 m 27 s, success) printed these seeds and results, all with 0 mismatches and 0 errors: opt 0 seed 2084419975 (1,000 nets,
7,611 passes, 132,715 values), opt 3 seed 1544036178 (1,000 nets, 7,555 passes, 132,904 values), tiles seed 1245050383
(300 nets, 2,281 passes, 40,699 values), spiking seed 485719738 (100 nets, 800 passes, 6,304 values), transformers seed
1366333554 (20 nets, 60 passes, 11,269 values). An earlier manual run at v0.6.0 (40689922442) also passed. A seed that fails is reproduced
with `--seed N`. Seeds in CI are fresh each run, so a clean CI run says nothing about the seeds in the table above.

### What the fuzzer found

| found by | what | state |
|---|---|---|
| `fuzz --nets 2000 --seed 0`, net 591 | a barrier that needs two mover tics left a doll one tic late and one plane move was lost: `-h1_0: DOOM -1461 != reference -1305` for `MLP([[[156]],[[3]]], [[1617],[0]], [[5748,8058],[11999]])` at `x = -12000` | fixed in `pure.py` (per-layer barrier height); replay file [replay_mlp_seed0_net591.json](../models/verify/replay_mlp_seed0_net591.json) is exact on the fixed tree |
| `fuzz --edge --nets 500 --seed 1`, net 246 | the same cause with thresholds `[5740, 11999]`, `W = 255`, `x = 12000`: `-h1_0: DOOM -255 != reference -510` | fixed with it; [replay_mlp_edge_seed1_net246.json](../models/verify/replay_mlp_edge_seed1_net246.json) |
| `fuzz --edge --nets 400 --seed 902`, net 5 (a run on v0.5.0) | the plane-move bound: a plane starting above 14,018 lost a move; `net.check` accepted the input: DOOM 12,254, reference 11,999 | contract changed, see [the plane-move bound](#the-plane-move-bound); replay [replay_mlp_edge_seed902_net5.json](../models/verify/replay_mlp_edge_seed902_net5.json) now reports one pass and 18 refused inputs; regression `tests/test_planes.py` |
| `fuzz --kind transformer --nets 6 --seed 0`, net 4 | a gated MLP unit with an all-zero `Wdn` column still allocated its `u` gate port; the readout read a sector that does not exist (`L0.zu:0:1: DOOM 0 != reference -4`) | fixed in `ops.gated`; test `test_gated_unit_with_a_zero_wdn_column_has_no_stray_port` |
| the transformer generator | `check_limits` raised on empty level sets (softmax with `R * len(thr_v) == 1`, `d == 1`) | fixed in `intref.check_limits` |

Both scheduling cases failed at `--opt 0`, `1` and `3` before the fix and are exact at all levels after it. The earlier
records `fuzz_mlp_2000_before_fix.json` and `fuzz_mlp_edge_before_fix.json` show the failing runs (1 mismatch each).

## `doomformer fuzz --mutate`

Mutation analysis of the compiled map of a 3-4-3 two-layer MLP (`mutate.demo_net()`, 1,222 sectors). 
`doomformer fuzz --mutate --per-class 25 --workers 3 --seed 0 --json models/verify/mutation.json` compiles the net,
captures the ACS source the compiler produced, and builds one mutant per site: every linedef special argument (+-1) and
special number (raise <-> lower), a retargeting of every effect line to another target, every sector floor, ceiling and
light (+-1), every integer literal of the ACS (+-1, recompiled with acc) and the `>> 16` of the plane readouts (as `>> 15`).
Up to 25 sites per class are sampled (5,578 sites exist; 1,004 were run). A mutant runs the net's 14 inputs (zero, +-1,
+-12,000, threshold neighbours, random); a survivor is re-run on 42 more. Then EVERY class that still has a survivor
gets a deep set: up to 6 of its survivors run on 150 random inputs at four scales plus 96 threshold neighbours (each
input at t-1, t, t+1 of every first-layer threshold, the others at 0 or random; 246 inputs). A class where any of them
dies loses its rule (`ctx["demoted"]`, by family: all `sector-floor-walk:*` subclasses together). A mutant is killed when
any value differs from the integer reference, or the game fails to start, crashes or does not settle. A survivor of a
class that kept its rule is given a one-line reason by a rule table (`mutate.reason`); the rules are expectations from
the design, not proofs, and a class is "kept" only because the sampled survivors (not all of them) survived the deep set.

Result of `fuzz --mutate --per-class 25 --workers 3 --seed 0 --controls` (record
[mutation_v2.json](../models/verify/mutation_v2.json); 775 s on 3 games, machine idle):

| | |
|---|---|
| sites / run | 5,578 / 1,004 |
| killed | 297 (29.6 %) |
| survivors | 707: 590 labelled equivalent by a rule, 117 **input gaps** (walking floors), 0 unexplained |
| classes that kept a survivor and went through the deep set | 40: 224 of the 710 survivors present before it (at most 6 per class) |
| killed only by the deep set | 3 (2 `ACS_ExecuteAlways` script numbers, 1 walking floor) |

Every sampled site of the classes that carry a weight, a sign, a target, a doll identity or an output's initial value
is killed (light weight and plane weight with their tags, effect sign, effect target, script number, doll TID,
start-barrier tag). Per-class counts are in the JSON (`classes`, with `deep_survivors`, `deep_tested`, `deep_killed`).
The 29.6 % is not a quality score: 590 of the 707 survivors are labelled equivalent (light levels no script reads,
ceilings 16,000 units above the dolls, barrier heights of 13,000, output display heights, mover speeds, the motion script
of maps with fewer than 4,096 objects, the unused map argument of `ACS_ExecuteAlways`, the point arguments of
`GetSector*Z`, the redundancy rules below), and each of those labels is supported only by the deep-set survivors of its
class, not by an argument. The v0.5.0 record (301 killed, 703 survivors, 33 without a rule) stays in `mutation.json`
and is superseded: its table and the "70 % cannot change a pass" sentence are removed from this page (the 70 % was
false for the 862 walking-floor sites); the 33 survivors are explained below.

**Walking floors (confirmed).** The v0.5.0 label "+-1 on a walking floor stays inside the 24-unit
step-up slack" was false and had no deep run. A lane's walkway floor sets its gate: the gate passes when its floor is
within 24 of the walkway, so lowering the walkway by 1 moves that lane's threshold by one, and an input exactly on the
threshold shows it. On the 3-4-3 net `sector 231 (tag 1) heightfloor -26 -> -27` is exact at x = [0,0,0] and [3,3,3] and
wrong at [2,2,2] (h1_0: DOOM 12, reference 15; `tests/test_mutation_rules.py::test_walking_floor_mutants_are_not_equivalent`).
A separate run killed 19 of 75 survivors of the class `FLOOR4_8/192` on 320 more inputs; the five sites that differed between the
two earlier records live or die with the random inputs drawn. In this run the deep set (which now includes the threshold
neighbours) killed 1 of the 6 tested survivors of that subclass, so the whole family is demoted: its 117 survivors are
reported as **input gaps** (not equivalent, not killed by this input set), not as explained. All 117 were then run once on the deep set (below).

### The 33 survivors of the v0.5.0 run, explained

I ran every site of the classes with an unexplained survivor, not only the sampled ones, on the 3-4-3 net with 24 + 100
inputs (1,676 sites had no rule in the v0.5.0 table; 182 survived: 146 `acs-reset`, 6 `move-to-tag`, 6 `line-other` and
24 `scroller` sites, the rest were killed), and the same classes on three random forward nets (`fuzz` seeds 11, 12, 13,
2 layers, 63 to 72 inputs); counts per class in [mutation_survivor_study.json](../models/verify/mutation_survivor_study.json).
Then I changed whole mechanisms at once. The rules are in `mutate.reason` (`_redundancy_reason`); every survivor now gets the deep set; `fuzz --mutate --controls` prints the aggregate maps below (record
`mutation_v2.json`, 14 inputs; `tests/test_mutation_rules.py` pins them).

| class (survivors among the 33) | rule | aggregate control on the 3-4-3 map | result |
|---|---|---|---|
| scroller args (16) | the carry belt is 4 stacked scrollers (`CONVEYOR stack=4`); changing one leaves 3 carrying | 1, 2, 3, 4 of 4 removed | exact, exact, **fails**, **fails** (the dolls do not move) |
| tag and arg3 of a freeze line, `special 37` (12) | a freeze line stops a late doll of a finished layer; a forward pass has none | all 3 freeze lines removed | exact |
| wave barrier reset, tag +-1 or sign flag (4) | one barrier reset wrongly leaves the other waves to serialise the moves | first reset removed / every other / all; first below the walkway / all | exact / **fails** / **fails**; exact / **fails** |
| freeze reset, flag 1 -> 2 (1) | any nonzero flag negates the height: the mutant is identical | | rule only |

Not among the 33 but in the full run and now covered by rules: the layer-0 start barrier reset (script 7 lowers the barrier
every run), freeze reset height +-1 (inside the 24-unit step slack) and the reset of the last layer's freeze barrier
(nothing raises it).

On the three further nets the same classes survive: `scroller` 24 of 24 sites on each net, freeze lines (`move-to-tag`
and `line-other`) 8 of 8, 8 of 8 and 12 of 12, barrier resets (tag or flag, closed height 13,000) 28 of 32, 64 of 72 and
40 of 48. Single-site faults of redundant mechanisms are not seen by a forward pass. The learning map is the contrast: of
its 8 `move-to-tag` and 8 `line-other` (`special 37`) sites 6 of each are killed (`mutation_learning.json`), and
`test_freeze_barriers_are_necessary` shows the mechanism is needed there. The wave-barrier rule is the weakest: I did not
find why a single barrier set 13,000 units below the walkway is harmless, and the claim rests on the controls, not on a
mechanism. [INFERENCE, untested: a `Floor_LowerToLowest` on a floor below its neighbours may raise it.]

The overall result of the `--controls` run is the table above.

### Mutating tile maps and the learning map

`fuzz --mutate --tile K` mutates operation tile K of the tile-mode lowering of the same net (ports written with the
reference's previous-layer values, every output compared); `--learning` mutates the online perceptron map of
`doomformer.learn` (F = 6, K = 3, cap 100; 40 examples, every spike and weight compared after every example).

| target | sites | run | killed | kill rate | survivors without a rule | weight, sign, target classes killed | record |
|---|---|---|---|---|---|---|---|
| 3-4-3 net, one map | 5,578 | 1,004 | 297 | 29.6 % | 0 (117 input gaps) | 25 of 25 in each | mutation_v2.json |
| tile 0 (layer 0) | 1,926 | 893 | 245 | 27.4 % | 8 (`acs-other` 4, `acs-reset` 4) | 25 of 25 in each | mutation_tile0.json |
| tile 1 (layer 1) | 1,034 | 779 | 189 | 24.3 % | 9 (`acs-other` 4, `acs-reset` 5) | all sampled killed | mutation_tile1.json |
| perceptron (learning map) | 5,911 | 923 | 190 | 20.6 % | 74 | plane weight 13 of 25, plane weight tag 10 of 25, effect sign 12 of 25, effect target 11 of 25, barrier tag 14 of 25 | mutation_learning.json |

The tile and learning rows ran before the deep set was applied to every class (v0.6.1 changed `run_mutation` only; their
walking-floor and other labelled survivors have not been through it). The learning map's weight and target lines are not all killed by 40 examples: a perceptron updates only on a mistake and
a line is crossed only when its feature is on, so many lines are never exercised in 40 examples. I read that as a gap in
the examples, not as equivalence; I did not run more examples. The 8 and 9 unexplained tile survivors are ACS literals I
did not look at.

**Labels on a net with planes at the clamps.** The equivalence labels were measured on the 3-4-3 net, whose planes stay
far below 14,000. I ran them again on [plane_edge_net.npz](../models/verify/plane_edge_net.npz) (4 inputs, 3 hidden units; the
twin planes start at 16,000, -15,000 and -11,978; `plane_errors() == []`), every site of each class (up to 80) on 151 inputs,
2 or more passes each ([mutation_plane_edge_labels.txt](../models/verify/mutation_plane_edge_labels.txt)):

| class | sites | killed on the plane-edge net | label |
|---|---|---|---|
| `sector-ceiling-gate` | 56 | 1 (`heightceiling 16000 -> 15999` of the plane that starts at 16,000: -h1_0 DOOM 11,919, reference 11,920) | conditioned |
| `sector-floor-gate` | 56 | 6 (the three planes' start heights +-1: -15,000, 16,000 and -11,978) | conditioned |
| `sector-floor-barrier`, `sector-ceiling-barrier` | 80 + 80 | 0 | kept |
| `plane-mover-speed`, `barrier-speed`, `move-to-speed` | 80 + 58 + 8 | 0 | kept |
| `move-to-height`, `sector-floor-output`, `sector-ceiling-output` | 8 + 10 + 10 | 0 | kept |

The two gate labels now read "... on nets whose planes stay within +-10,000" (`mutate.PLANES_LOW`, `planes_within`): a gate
sector's ceiling of 16,000 is the clamp of a plane that starts there, and a plane's start height is a live value. On a net
that fails that test `mutate.reason` gives these classes no rule. The 3-4-3 numbers above are unchanged by this (it passes
the test); `tests/test_mutation_rules.py` pins the conditioned labels and the kill on the plane-edge net. The sampled harness
(`fuzz --mutate --net plane_edge_net.npz`, [mutation_plane_edge.json](../models/verify/mutation_plane_edge.json)) is a weak
instrument on this net: only 1 of its first candidate inputs passes the range check, so 171 of its 788 survivors have no
rule (weight, TID and script classes among them), and its 6-per-class deep sample does not reach the one site that matters
(it found 0 of 56 in the ceiling-gate class); the exhaustive table above is the evidence, not that run.

**The 117 input gaps on the deep set.** All 117 walking-floor survivors of the 3-4-3 run, once each on the 14 + 246 inputs
([mutation_input_gaps.json](../models/verify/mutation_input_gaps.json)): 4 die (all in the subclass `FLOOR4_8/192`: 4 of its
17), 113 survive. So 96 % are still unclassified: for the other four subclasses (25 sites each, 0 killed) I have neither a
kill nor an argument that the lanes they serve are never at a threshold in these inputs.

Planted compiler bugs (`fuzz --plant`, above) are the complementary test: they change the compiler, not the map.

## `doomformer verify`

```text
doomformer verify net.npz                               # --mode classes over |x| <= 12,000 for every input
doomformer verify net.npz --mode all --domain -40 40    # every input of the box
doomformer verify net.npz --wad tampered.wad            # check a given compiled map: a changed weight line fails
doomformer verify --validate 20                         # classes == all on 20 small nets; tampered maps fail both
```

`net.npz` is an `MLP.save` file. `all` runs every integer input of the box `[lo, hi]^n_in` (default the whole legal range
`[-12000, 12000]`, which only a net with one input can enumerate: 24,001 runs, about 15 ms each) that `net.check` accepts.
`classes` runs one pass per input class and per threshold neighbourhood.

**Why classes cover the domain.** The map sees an input `x_i` only through the staircase count `a_i = adc(x_i,
thresholds[0])`: one doll per threshold and sign, each passing or not passing its gate. Every later layer, weight line
and readout depends on the count vector `a = (a_0 ... a_{n-1})`; the reference is the same function of `a`; and
`net.check` depends on `a` and on `|x_i| <= 12,000`. `a_i` is constant on the intervals between the breakpoints `t`
(the count jumps at `x = t`) and `1 - t` (signed input: it jumps between `-t` and `-t + 1`) of the first layer's
thresholds, so the box splits into the product of the per-input class counts, each class with one `a`. `verify` runs the
two end points of every class (so `t-1` and `t` on both sides of each breakpoint) and, for each input and breakpoint, the
three values `t-1, t, t+1` with the other inputs at 0. It reports how many domain inputs the runs stand for.

**The assumption (A).** DOOM's result on `x` equals its result on the class representative if the engine's response to
`x` depends on `x` only through the counts: the gate comparison of the port value against the gate's limit is a correct
integer comparison for every `|x| <= 12,000` (16.16 fixed point is exact there), and nothing else depends on the size of
`x` (port moves finish within 2 tics at any distance). `all` needs no assumption. `--validate` tests A on small nets.

**What `VERIFIED` means.** For this compiled map in this engine build, every run of the listed inputs reproduced the integer
reference in every exposed value at tolerance 0; with `all`, those runs are every input of the box; with `classes`, they
are one or more representatives of every class, which covers the box under A. It does not mean: the net is correct for
the float model it was quantised from; inputs outside the box or refused by the range check are safe (they are refused
before any run); unexposed values are right; another engine version behaves the same; or the ACS is correct for maps it
was not compiled for. The run is deterministic only as far as ZDoom is.

### `--validate`: classes against all

`doomformer verify --validate 20 --seed 0` (record [verify_validate.json](../models/verify/verify_validate.json)): on 20
small random MLPs (1 to 3 layers, 1 or 2 inputs, weights +-9) the verdict of `classes` equals the verdict of `all` on
20 of 20 nets, all 20 clean maps are `VERIFIED` by both, and a map with one weight line changed by +1 (the first
`Light_RaiseByValue` / `Light_LowerByValue` line) is `FAILED` by both on 20 of 20. Nets 0 and 1 are 1-input nets over the
entire legal domain `[-12000, 12000]`: `all` ran 24,001 inputs (308 s and 232 s for the clean map on the shared machine,
about 13 ms per pass; the tampered runs stop at the first mismatch) against 14 and 8 runs of `classes`. The other 18
nets use a small box that contains every breakpoint (domains of +-4 to +-30). Total: 808 runs for `classes`, 105,724
planned runs for `all`. The run took 964 s. This validates A on these 20 nets; it is not a proof of A for other nets.

Directly: `doomformer verify net.npz` on a 1-input net prints `VERIFIED` with 5 classes over 24,001 inputs in under a
second of game time, and the same command with `--wad tampered.wad` prints `FAILED` and the first differing input.

## Static extraction

```text
doomformer verify --extract net.wad --net net.npz      # no game: read the net out of the map, compare, exit 1 on a difference
doomformer extract --dump net.wad                      # the recovered netlist per layer, the checks, FINDING and NOT CHECKED lines
```

The exactness checks above show that DOOM computed the same values for the inputs that were run. The static extraction
answers a different question about the same file: which weights, biases and thresholds does the map hold, and do the
lines and scripts that start the dolls say what the compiler writes? `extract` reads the compiled `.wad` and nothing else
(the TEXTMAP with its sectors, linedefs, things, and the BEHAVIOR lump), recovers the MLP, and `Netlist.diff(net)`
compares it with the source net. The check needs no input, and it covers a weight line behind a gate that no run reaches.

**What EQUAL means.** `EQUAL` is a static statement about the arithmetic netlist (every lane's gate and weight lines,
every bias), the plane contract (`planes.plane_errors()` on the recovered net: start heights and the plane-move bound
for all inputs; `verify --extract` prints `plane contract: ok` or fails with the bound), and the items marked "read"
in the table below. It is not a statement about timing, about triggers that are not listed, or about the readout slots'
names; `extract --dump` and `verify --extract` print the `NOT CHECKED:` list. It does not show that the engine executes
the map as read; the exact-in-DOOM checks above remain the evidence for that. The extractor encodes what each special
does by its own reading of the engine facts (`doomformer/extract.py`, module docstring) and imports nothing from the
lowering code. It does not read the `.json` that `compile` writes, so the names the runtime writes into each port
(`x{i}`, `-x{i}`) are not checked against the map.

**Method.** Lanes are found from the start barriers (tag 2 for layer 0, 60000 + s for layer s) and walked along x
through their two-sided linedefs. A lane gives its port (the gate sector's tag), its gate limit (the walkway floor plus
24; the threshold is the negated limit) and the ordered effect lines behind the gate. Per layer the lines are summed
into rows: a hidden unit writes the light of `h` and the plane of `-h` in consecutive lines, an output writes its light;
a bias is the start height of the plane (hidden) or the start level of the light (output). Opt 3 maps write the light
only; a small interpreter runs the compact ACS bytecode of the copy scripts on two sets of test light levels and reads
which light each plane is set to. The same interpreter runs scripts 1, 2, 3, 4, 7 and the readout pages. Every
linedef with a special is classified and all five arguments and the trigger flags are compared with what the compiler
writes. The comparison with the net is per lane: lane k of input i into row j.

| item | status |
|---|---|
| layer count and widths, weights per lane, biases, thresholds of every layer including the input staircase | read; compared lane by lane |
| signed or relu input | read as the port structure (pairs `-x`, `x` with equal gates and negated lines). A relu net whose input columns come in negated pairs has the same structure, so the map cannot tell the two apart; `equals` compares in the port form of the net under its own `signed_input` |
| twin accumulators (light of `h`, plane of `-h`) | read; both start values and both amounts of every lane must agree with each other and with the net (opt 3: one light, the copy script is decoded) |
| units no later weight reads | read when their row has a nonzero weight; a zero row leaves two lights compared as a multiset of start values |
| plane contract | read: `plane_errors()` on the recovered net (an unread unit has no plane and is outside it) |
| every line special: class, tag, all five arguments, mover speed (150,000 to 262,144), trigger flags (exactly `playercross` and `repeatspecial`, two-sided), for weight, release, freeze, script-start and teleport lines and the conveyor scrollers; two-sided lines without flags | read |
| weight lines that lie outside every lane; blocking two-sided lines; line specials of any other class | read: findings |
| teleport lines: a single destination thing with that id exists, inside a lane or clock sector, spawning in single player | read; the destination's position within the clock chain is not (timing) |
| reset script (3) | read: restores every light, internal plane, barrier and freeze walkway to its TEXTMAP value exactly once, at an allowed mover speed, and sends every doll (TID of its self-tag line) to a destination thing of the right id inside its own home sector |
| scripts 1, 2, 4, 7 | read: the port writers, the TID script and the layer-0 release produce the specials the runtime relies on |
| readout pages (100 and up) | read: together they read every light and every internal plane exactly once; the names behind the slots are not in the map |
| copy scripts (opt 3) and their start lines | read |
| lane and clock geometry: sector specials, conveyor tag, steps up of at most 24, headroom of 56, barriers that block closed and open to a walkable step, closed sector boundaries, wave, start and freeze barriers released, one doll per home plus the player, thing flags | read |
| vertex coordinates, sidedefs, textures; lane lengths, wave spacing, clock chain positions | not read (timing and geometry) |
| light and ceiling of sectors that are not gates, barriers, lane ground or accumulators; thing angles, coop and dm flags, thing positions inside a home | not read |
| motion script 8, guard script 9, world scripts | not read |
| inputs whose weights are all zero; hidden units with no reader and a zero row; leading all-zero output rows of a net with hidden layers | not in the map; `expected_netlist` drops the same things (see the module docstring) |
| tile maps, transformer maps, other maps | `ExtractError` (tile and transformer maps were tried) |

### Extraction mutation score

`python models/extract/mutation_score.py --opt N` changes one field of the compiled map of a 3-4-3 signed MLP
(`small_net()`: two weight layers, 24 doll lanes) and reads the map back: every integer field of the TEXTMAP (+1, and
the sign flip when the value is not 0), every float coordinate (+1), every true linedef and thing flag (set false), every
byte of the BEHAVIOR lump (+1). A verdict of DIFF means a difference, a finding or `ExtractError`. A sample of the edits
that stay EQUAL, two per field kind, then ran in DOOM on 12 inputs at tolerance 0 (`SAME`: every pass exact).

| | opt 0 | opt 3 |
|---|---|---|
| edits | 23,622 | 19,754 |
| detected (difference, finding or `ExtractError`) | 17,906 | 14,890 |
| undetected (EQUAL) | 5,716 | 4,864 |
| undetected edits run in DOOM (12 inputs, tolerance 0) | 50 | 48 |
| ... DOOM caught it (mismatch) | 0 | 0 |
| ... DOOM error (no settle, exception) | 0 | 0 |
| ... DOOM crash or hang | 0 | 0 |
| ... SAME: exact on all 12 inputs | 50 | 48 |

Undetected edits by field kind (undetected / edits of that kind). Every sampled one was SAME in DOOM.

| field kind | opt 0 | opt 3 | why it stays EQUAL |
|---|---|---|---|
| BEHAVIOR: bytes of scripts 1, 2, 3, 7, copy scripts, readout pages | 114 / 1,241 | 72 / 1,012 | a mover speed edited inside the accepted range 150,000 to 262,144 (all 114 at opt 0; 64 of 72 at opt 3, the other 8 change no script output) |
| BEHAVIOR: legacy ACS0 script directory | 92 / 92 | 104 / 104 | not used behind the ACSe header; the extractor reads the chunk table |
| BEHAVIOR: motion script 8 | 236 / 279 | 243 / 292 | not read (runtime settle only for maps with 4,096 or more objects) |
| linedef blocking flag off | 952 / 952 | 814 / 814 | one-sided lines block anyway |
| linedef sidedef index | 511 / 3,527 | 442 / 2,975 | another sidedef of the same sector (boundaries still close) |
| mover speed +1 (lines 20, 21, 23, 37) | 75 / 336 | 5 / 197 | inside the accepted range 150,000 to 262,144 |
| sector floor and ceiling | 697 / 1,721 | 559 / 1,466 | ground ceilings with more headroom, floors within the step rule, lights, closets, input gates (written by Python) |
| sector light level | 868 / 880 | 730 / 742 | sectors that are not accumulators |
| sector special | 2 / 441 | 2 / 372 | world room and closets (not part of the machine) |
| thing angle, coop and dm flags | 159 / 635 | 159 / 635 | not read |
| thing position +1 | 106 / 106 | 106 / 106 | still inside the home sector; not read |
| vertex coordinates | 1,904 / 1,904 | 1,628 / 1,628 | geometry and timing, not read |

Every edit of the other 66 field kinds is detected, among them every weight, bias and threshold field, the special number, tag and arguments of every line class, every trigger and repeat flag, every sector tag, every thing type and id, the reset script's literals other than speeds, and every sector tag.


### Extraction results

Commands run on the tree of this release (shared machine, `OMP_NUM_THREADS=1`); records in
[models/extract/](../models/extract/).

| check | command | result |
|---|---|---|
| 500 random MLPs, opt 0 | `python models/extract/campaign.py --nets 500 --seed 0 --opt 0` | 490 compiled (10 refused by the compiler: an all-zero layer), 0 failures: 484 equal with plane contract ok, 6 equal netlist that violate the plane contract (they compile only with `limits=False`) ([json](../models/extract/fuzz_seed0_opt0.json)) |
| the same seed, opt 1, 2, 3 | `... --nets 200 --seed 0 --opt 1` (2, 3) | 199 compiled each, 0 failures: 196 equal with contract ok, 3 violate it |
| range-edge nets, opt 0 and 3 | `... --nets 200 --seed 1 --opt 0 --edge` (3) | 200 compiled each, 0 failures: 191 equal with contract ok, 9 violate it |
| plane start -16,001 (`bias` outside [-15,000, 16,000]) | `doomformer verify --extract bad.wad --net bad.npz` | netlist equal, `plane contract: 1 violation(s)` with the bound in the finding, `NOT EQUAL`, exit 1 (test `test_plane_contract_is_part_of_equal`) |
| shipped MNIST net (784-64-45-10-10) | `doomformer verify --extract mnist_opt0.wad --net models/mnist/mlp.npz` (map: `df.compile(MLP.load("models/mnist/mlp.npz"), wad, mode="map", opt=N)`) | opt 0, 1, 2, 3: EQUAL, plane contract ok; 73,766 lane weights, 129 biases, 1,351 lane thresholds; 7.3 s and 785 MB at opt 0, 4.6 s at opt 3 ([log](../models/extract/mnist.txt)) |
| negative controls, fuzz net (seed 0, index 10: 2-5-5-7-3, signed) | `python models/extract/controls.py --fuzz-seed 0 --index 10 --opt 0` (3) | 19 controls at opt 0, 18 at opt 3, all detected: nine edit one weight, sign, retarget, gate height, pad, start value, plane line, twin line, plane start; the others drop a trigger or repeat flag of a release, weight, script-start or freeze line, change a mover speed or the freeze height, move a teleport destination's id, change the reset script's light tag ([opt 0](../models/extract/controls_fuzz_idx10_opt0.txt), [opt 3](../models/extract/controls_fuzz_idx10_opt3.txt)) |
| negative controls, MNIST | `python models/extract/controls.py --net models/mnist/mlp.npz` | the same 19 controls, all detected ([log](../models/extract/controls_mnist.txt)) |
| the controls of the fuzz net in DOOM | `doomformer verify net10.npz --wad CONTROL.wad` (61 runs each) | 15 of 19 FAILED. VERIFIED: the two last-layer threshold edits (the unit that gate reads takes only the values -2, 4, 7 and 10 on the 61 runs and on 40,000 random inputs, so no run separates thresholds 3 and 4), and the freeze-line edits (a freeze barrier that these inputs do not need; `tests/test_doom_exact.py::test_freeze_barriers_are_necessary` has a case that does) ([log](../models/extract/controls_fuzz_idx10_opt0_dynamic.txt)) |
| the static claim against DOOM on the same map | `python models/extract/doom_pair.py --seed 0 --indices 1 5 6 10 14 17 24 33 --opt 0` | 8 nets, static EQUAL, 64 passes and 2,160 values exact at tolerance 0; nets 10, 14, 17 at opt 1, 2, 3: 24 passes and 824 values exact per level ([json](../models/extract/doom_pair_opt0.json)) |
| mutation score of the extractor | `python models/extract/mutation_score.py --opt 0` (3) | the table above ([opt 0](../models/extract/mutation_score_opt0.json), [opt 3](../models/extract/mutation_score_opt3.json)) |


### Extraction limits

* EQUAL compares the map with the net as the lowering reads it (one doll per threshold and sign, lines per weight). That
  the engine runs those dolls and lines as modelled is the assumption of everything above, tested by the exactness checks.
* The semantics of each special is the extractor's own reading of the engine facts. If it is wrong, extraction and
  compiler can agree and both be wrong; a DOOM run exposes that.
* Timing is outside the netlist: wave spacing, clock positions, the order in which the clock doll crosses its lines and
  the settle estimate decide whether the dolls arrive in time. The mutation score shows what a single edit of such a field
  does on the 3-4-3 net, not that other nets are safe.
* The mutation score is single edits on one small net at opt 0 and opt 3, and the DOOM sample is two edits per kind on
  12 inputs. "SAME" means equal on those inputs, not equivalent.
* Of the 490 compiled fuzz nets (seed 0), 36 have an all-zero input column, 75 a unit nothing reads with a nonzero row,
  15 one with a zero row and 9 a leading all-zero output row (a net can have several); they are compared as described in
  the table.
* Tile maps (one layer per map) are not read, and neither is any map that is not a one-map MLP.

## Not covered

* The ACS. The page and reset scripts are exercised by every run and mutated by `--mutate` (literals, `>> 16`). Static
  extraction runs scripts 1, 2, 3, 4, 7, the opt 3 copy scripts and the readout pages of an MLP map and compares them with
  the TEXTMAP (table in Static extraction); the motion script, the guard script and the scripts of other maps have no
  static argument, and there is no input-independent proof.
* Inputs that were not run: `fuzz` samples nets and inputs, `fuzz --mutate` samples mutants, `verify --mode classes`
  stands on assumption A, and `verify` is for MLPs (a transformer's domain, V^T token sequences with `d`-dimensional
  staircases, is not enumerable here).
* Large maps: a map with at least 4,096 objects uses the motion script of `runtime.settle`; the fuzzed nets are far smaller,
  so that script and its ACS are not exercised by `fuzz` (the mutants of script 8 survive because of it).
* Learning maps: not fuzzed. They were mutated once (above, 20.6 % killed, 74 survivors without a rule); the freeze
  barriers that forward nets do not need are exercised only by `test_freeze_barriers_are_necessary` and that mutation run.
* Tile mode: MLPs are fuzzed (`--tiles`, 4,000 edge nets above) and two tiles are mutated; transformers in tiles are
  covered by `df.coverage` and `tests/test_tiled.py`, not by `fuzz`. Ray, worlds, actions and `guard` on tiles: not fuzzed.
  Spiking maps are fuzzed without the readout guard (not supported for them).
* Other engines: the fuzzers, `verify` and the mutation harness ran in ViZDoom 1.3.2's ZDoom only. The probe maps and the
  `hello` brain ran in five other engines ([engine_limits.md](engine_limits.md#other-ports)); the original DOOM engine is
  not tested.
* The transformer range check at compile is over sampled sequences, and each one-map pass checks its own sequence before it
  runs; there is no static bound over all of V^T. Ceiling-plane edges of the plane-move bound are measured in the ACS probe
  only.
* The staircase counts are not exposed by any map (MNIST: 903 counts and 1,351 gate comparisons); they are checked through
  the sums they feed. Spiking maps do not read back the twin planes `-s`.
* Mutation survivors with a rule-based reason are expected equivalents, not proven ones; the aggregate controls support the
  redundancy rules on one net; the wave-barrier rule has no mechanism behind it.

## Reproduce

Commands and seeds are the ones printed above; the JSON records hold the per-run counts. Timings were measured on a
16-core arm64 machine shared with other jobs (load average 25 to 40), at most 3 games at once; an idle machine is
about 2 to 3 times faster.
