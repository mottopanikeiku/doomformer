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
* `df.compile(net, guard=True)` (one map only) adds an ACS script that fills every USER slot with a fresh nonce before
  each read. A readout page that never runs returns the nonce and the pass raises `RuntimeError: readout page 0 did not
  run`. Before, USER1.. kept the last pass's values, so a dead page passed whenever the input repeated. `guard` is off
  by default so that default maps stay byte-identical; `fuzz` and `verify` compile with it. A default map keeps the old
  weakness: a dead page is caught only when the stale values differ from the reference.
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
lines in any order) needs `b + P <= 14,018` when `P > 0` and `b - N >= -14,018` when `N > 0`. A plane that never rises
may start anywhere. `P` stands for the whole sum of raises, not for the sum without the last one, so the rule is
stricter than the engine by at most the smallest raise (one weight). It is enforced in three places:

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

Measured on the reduced net with the check bypassed (`doomformer verify --plane-sweep`, record
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
gate-gt` to fail. Measured: `doomformer fuzz --nets 40 --seed 0 --workers 2` ran 40 nets, 304 passes, 5,914 values, 0
mismatches in 10.3 s; `--nets 12 --plant gate-gt` found the bug (10 of 12 nets mismatch, first at net 1) in 3.3 s (shared
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
(`source_sha256` 28a519c0c52354a0 in the JSON, a snapshot taken after the plane-move fix).

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

On the same tree: `fuzz --kind spiking --nets 300 --seed 31337 --workers 3` (2,400 passes, 19,016 values, 0 mismatches,
118 s) and `fuzz --kind transformer --nets 100 --seed 31338 --workers 3 --inputs 3` (297 passes, 60,890 values, 0
mismatches, 1 net with an all-zero weight matrix reported unsupported, 1,859 s); records `spiking_300_seed31337.json` and
`transformer_100_seed31338.json` (source hash 8d487aae38667bf3, an earlier snapshot of the same tree: it lacks the
per-pass transformer range check and `verify --coverage`, which do not change what these runs build). Edge nets with
`--tiles` are generated with the tile-mode range check (no plane rule), so they include inputs that the one-map check
refuses. Tile mode compiles each net into one tile per layer and runs every input through `TiledMLPBrain` with one live
game per layer.

CI: every push runs `fuzz --edge --nets 60 --seed random --workers 2` after the fixed-seed step, and
`.github/workflows/fuzz.yml` runs a longer campaign (opt 0, opt 3, tiles, spiking, transformers) every Monday with random
seeds; each step prints `fuzz: random seed N`, and a failing step uploads its replay file. A seed that fails is reproduced
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
+-12,000, threshold neighbours, random); a survivor is re-run on 42 more, and an unexplained survivor on 300 more at four
scales. A mutant is killed when any value differs from the integer reference, or the game fails to start, crashes or does
not settle. A survivor is given a one-line reason by a rule table (`mutate.reason`); the rules are expectations from the
design, not proofs.

Result of that command (704.6 s on 3 games, shared machine):

| class | sites | run | killed | survived |
|---|---|---|---|---|
| light weight (`Light_Raise/LowerByValue` amount) | 210 | 25 | 25 | 0 |
| light weight target tag | 210 | 25 | 25 | 0 |
| plane weight (`Floor_Raise/Lower` amount, `Ceiling_...`) | 144 | 25 | 25 | 0 |
| plane weight target tag | 144 | 25 | 25 | 0 |
| effect sign (raise <-> lower) | 177 | 25 | 25 | 0 |
| effect retargeted to another output | 177 | 25 | 25 | 0 |
| ACS_ExecuteAlways script number (doll tagging) | 64 | 25 | 25 | 0 |
| doll TID, teleport TID | 64 | 25 | 25 | 0 |
| start-barrier tag (`Floor_LowerToLowest` arg0) | 62 | 25 | 25 | 0 |
| ACS `>> 16` of the plane readouts | 4 | 4 | 4 | 0 |
| ACS doll-tag script literals | 2 | 2 | 2 | 0 |
| output light initial value | 14 | 14 | 14 | 0 |
| ACS readout literals (sector tags, indices) | 38 | 25 | 16 | 9 |
| ACS reset script literals | 556 | 25 | 7 | 18 |
| ACS start-barrier script literals | 4 | 4 | 2 | 2 |
| sector floor of a gate | 60 | 25 | 5 | 20 |
| sector floor of a walking sector | 862 | 125 | 12 | 113 |
| ACS_ExecuteAlways map argument | 64 | 25 | 12 | 13 |
| scroller arguments | 40 | 25 | 2 | 23 |
| mover speeds (plane, barrier, move-to), ACS port-write script | 216 | 60 | 0 | 60 |
| ceilings, lights of walking, barrier, gate and room sectors; floors of barriers, rooms, outputs | 2,436 | 415 | 0 | 415 |
| ACS motion script | 12 | 12 | 0 | 12 |
| `Floor_MoveToValue` line (special 37): tag, height, extra arg | 18 | 18 | 0 | 18 |
| **all** | **5,578** | **1,004** | **301** | **703** |

(The table groups the 51 classes of `models/verify/mutation.json`; the rows add up to the last line.) Every sampled
mutant of the 12 classes that carry a weight, a sign, a target, a doll identity or an output's initial value is killed
(first 12 rows). The overall 30.0 % is not a quality score: 70 % of the sites are sector
heights and lights, mover speeds and similar constants where a +-1 change cannot change a pass. Killed by: 273 value
mismatches, 26 `AttributeError` (the game ended and its state was gone) and 2 `RuntimeError` (not settled); I did not
inspect those 28 one by one.

Survivors by reason (703): 670 rule-explained as expected equivalents (light levels no script reads; ceilings 16,000
units above the dolls; barrier heights of 13,000; output display heights; mover speed 150,000 = 18,750 units per tic,
so +-1 moves no finish tic; the motion script, which runs only on maps with at least 4,096 objects; the 24-unit step-up
slack of walking floors; the unused map argument of `ACS_ExecuteAlways`; the point arguments of `GetSector*Z`), and
33 that no rule explained in that run. The next section replaces those 33.

### The 33 survivors, explained (v2)

I ran every site of the classes with an unexplained survivor, not only the sampled ones, on the 3-4-3 net with 24 + 100
inputs (1,676 sites had no rule in the v0.5.0 table; 182 survived: 146 `acs-reset`, 6 `move-to-tag`, 6 `line-other` and
24 `scroller` sites, the rest were killed), and the same classes on three random forward nets (`fuzz` seeds 11, 12, 13,
2 layers, 63 to 72 inputs); counts per class in [mutation_survivor_study.json](../models/verify/mutation_survivor_study.json).
Then I changed whole mechanisms at once. The rules are in `mutate.reason` (`_redundancy_reason`); a survivor with such a
reason still gets the 300-input deep set; `fuzz --mutate --controls` prints the aggregate maps below (record
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

Overall result of `fuzz --mutate --per-class 25 --workers 3 --seed 0 --controls` ([mutation_v2.json](../models/verify/mutation_v2.json),
1,496 s, shared machine): 5,578 sites, 1,004 run, 296 killed (29.5 %), 708 survived, 0 without a rule. Every sampled site
of the classes light weight, plane weight, their tags, effect sign, effect target, `ACS_ExecuteAlways` script number, doll
TID and start-barrier tag is killed. The v0.5.0 record killed 301 with the same seed; I did not trace the difference of 5
(timeouts and crashes count as kills, and the machine load differs). "0 without a rule" means every survivor matches a
rule, not that every survivor is proven equivalent.

### Mutating tile maps and the learning map

`fuzz --mutate --tile K` mutates operation tile K of the tile-mode lowering of the same net (ports written with the
reference's previous-layer values, every output compared); `--learning` mutates the online perceptron map of
`doomformer.learn` (F = 6, K = 3, cap 100; 40 examples, every spike and weight compared after every example).

| target | sites | run | killed | kill rate | survivors without a rule | weight, sign, target classes killed | record |
|---|---|---|---|---|---|---|---|
| 3-4-3 net, one map | 5,578 | 1,004 | 296 | 29.5 % | 0 | 25 of 25 in each | mutation_v2.json |
| tile 0 (layer 0) | 1,926 | 893 | 245 | 27.4 % | 8 (`acs-other` 4, `acs-reset` 4) | 25 of 25 in each | mutation_tile0.json |
| tile 1 (layer 1) | 1,034 | 779 | 189 | 24.3 % | 9 (`acs-other` 4, `acs-reset` 5) | all sampled killed | mutation_tile1.json |
| perceptron (learning map) | 5,911 | 923 | 190 | 20.6 % | 74 | plane weight 13 of 25, plane weight tag 10 of 25, effect sign 12 of 25, effect target 11 of 25, barrier tag 14 of 25 | mutation_learning.json |

The learning map's weight and target lines are not all killed by 40 examples: a perceptron updates only on a mistake and
a line is crossed only when its feature is on, so many lines are never exercised in 40 examples. I read that as a gap in
the examples, not as equivalence; I did not run more examples. The 8 and 9 unexplained tile survivors are ACS literals I
did not look at.

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

## Not covered

* The ACS. The page and reset scripts are exercised by every run and mutated by `--mutate` (literals, `>> 16`), but there
  is no static argument about them and no input-independent proof.
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
