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
  The check covers those sequences, not every token sequence; an out-of-range sequence still fails at run time as a
  DOOM-vs-reference mismatch (saturated light at 32,767), which is sound but later.

The check compares what DOOM computes with what the lowering says it computes. It does not compare the lowering with the
mathematics of the original float model (that is `quantization_report`), and a value the map does not expose is not
compared.

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
doomformer fuzz --edge --nets 500                       # weights +-255, hidden values at +-12,000, thresholds at 12,000
doomformer fuzz --replay fuzz-failures.json             # re-run the cases saved by a failing run
doomformer fuzz --nets 12 --plant gate-gt               # self-test: must exit 1
doomformer fuzz --nets 300 --opt 1                      # the compiler's schedule level (when df.compile has `opt`)
```

Net `i` of seed `S` is drawn from `SeedSequence([S, i, kind])`, so a case does not depend on the worker count. Each
worker compiles a net into a temporary WAD, starts a game, runs the net's inputs and closes it; a mismatch, an
unexpected exception or a dead worker is a failure (exit status 1), the failing cases are written to a replay file
(`--save-failures`, default `fuzz-failures.json`) and the summary to `--json`.

Generators:

* MLP: depth 1 to 4, 1 to 5 inputs, widths 1 to 8, signed or unsigned input, 1 to 4 thresholds per staircase with
  duplicates (first thresholds up to 12,000; later thresholds drawn from the hidden values the calibration inputs
  produce, so inputs land on them), weights up to +-255 (a quarter of the nets, and every `--edge` net, are range-edge
  nets: many weights at +-255, biases up to +-12,000, thresholds at 12,000 or 11,999). Inputs: zero, +-1, +-12,000, the
  `t-1, t, t+1` neighbours of each first-layer threshold with both signs, random inputs, and inputs for which a hidden unit
  sits exactly on a later threshold neighbour or on +-12,000 / +-11,999 (the generator shifts that unit's bias). Inputs
  the range check refuses are counted, not run (`inputs refused`).
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
passes). The transformer run started before the scheduling fix listed below and was not repeated on the final tree (the
repeat was cut off); the MLP rows were run on the final tree. `--inputs`, `--opt` and the seeds are the ones printed.
Inputs refused by the range check (2,025 in the 2,000-net run) are not counted as passes.

### What the fuzzer found

| found by | what | state |
|---|---|---|
| `fuzz --nets 2000 --seed 0`, net 591 | a barrier that needs two mover tics left a doll one tic late and one plane move was lost: `-h1_0: DOOM -1461 != reference -1305` for `MLP([[[156]],[[3]]], [[1617],[0]], [[5748,8058],[11999]])` at `x = -12000` | fixed in `pure.py` (per-layer barrier height); replay file [replay_mlp_seed0_net591.json](../models/verify/replay_mlp_seed0_net591.json) is exact on the fixed tree |
| `fuzz --edge --nets 500 --seed 1`, net 246 | the same cause with thresholds `[5740, 11999]`, `W = 255`, `x = 12000`: `-h1_0: DOOM -255 != reference -510` | fixed with it; [replay_mlp_edge_seed1_net246.json](../models/verify/replay_mlp_edge_seed1_net246.json) |
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
33 not explained by any rule: scroller arg2 1 -> 0 (16), `Floor_MoveToValue` arg3 (6), the tag of a `Floor_MoveToValue`
line +-1 (6) and of its ACS twin in the reset script (5). The last three are other layers' freeze barriers, which a pass
resets and moves in pairs; I expect them to be equivalent, but no rule or test says so. They are the places to read for
a coverage gap. The ACS readout and reset literals that survive (27 of 50 sampled) are mostly the explained kinds
above (point arguments, closed heights, speeds).

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
* Tile mode and `run_many(mode="tiles")`, Ray, worlds, actions and `guard` on tiles: not fuzzed here. Spiking maps are
  fuzzed without the readout guard (not supported for them).
* Other engines: ViZDoom 1.3.2's ZDoom only. GZDoom, Zandronum and the original DOOM engine are not tested.
* The transformer range check is over sampled sequences (see above), not a static bound over all of V^T.
* Mutation survivors with a rule-based reason are expected equivalents, not proven ones; the unexplained ones are listed
  above and are the coverage gaps of the input set.

## Reproduce

Commands and seeds are the ones printed above; the JSON records hold the per-run counts. Timings were measured on a
16-core arm64 machine shared with other jobs (load average 25 to 40), at most 3 games at once; an idle machine is
about 2 to 3 times faster.
