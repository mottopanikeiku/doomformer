# Tuning the one-map schedule

`doomformer tune` ([`doomformer/tune.py`](../doomformer/tune.py)) measures where each timing constant of the one-map
schedule (`pure.Timing`, map units of the clock doll, about 16 units per game tic) starts to lose values, and writes
[`doomformer/tuning.json`](../doomformer/tuning.json). `--opt 2` builds with its `shipped` constants, `--opt 3` with
`shipped` updated by `shipped_copy` (`pure.Timing.tuned`).

## Method

- One constant changes, the others stay at their defaults. Maps are built with the opt=1 passes (opt=3 passes for the
  copy-bus constants) and an explicit `Timing`.
- Every input runs through the map in ViZDoom and every exposed value is compared with the NumPy integer reference at
  tolerance 0 (`MLPBrain` / `TransformerBrain` checks). A value fails on a net if any input mismatches.
- Inputs per MLP: 4 extremes (all +max, all -max, all zero, alternating), boundary vectors (every unit at
  `+-(t + d)` for an input threshold `t` and `d` in 0, 1, 10, 23, 24, so the lane gates stand 0 to 24 units above
  their walkways and the dolls step onto them), and seeded random inputs. Transformers: 3 extremes and random token
  lists (every token lies within 23 of an embedding threshold). Seeds and counts are in the table.
- Scan: start at the default (sync_margin, whose default is the legacy schedule: 128), descend until some input fails,
  bisect between the largest failing value and the smallest passing value above it, test the shipped value.
- edge: the smallest tested value at and above which every net passes every input. largest failing: the largest
  tested value with a failure.
- Shipped: serial 1.25 x edge, sync_margin max(64, 1.25 x edge), clock_margin 2 x edge, rounded up to 8; opt=3
  clock_margin max(256, 1.25 x edge) rounded up to 16. A constant without a table ships its default.
- Game tics are simulated, so a map with the same bytes and inputs gives the same readback; a value whose map equals an
  already measured map reuses its result (the cells marked `reused`).
- Every new map also gets the static checker's prediction (`doomformer.schedule.check_schedule`, `predicted_slack`
  in the table: per constant the smallest margin over the layers minus its measured minimum, in tics).

## Results (2026-10-27, ViZDoom 1.3.2)

Measured under a time box (the final table was due at 23:25): coarse scans, 2 games at once, on a shared 16-core
machine with load average 17 to 48. Wall times in the table are under that load; pass/fail and game tics do not depend
on it.

| constant | default | edge | largest failing | shipped | nets, inputs per net | how |
|---|---|---|---|---|---|---|
| sync_margin | none (sync_wait 400) | at most -64 (no failure in 128 .. -64) | none | 64 (4 tics) | hello, signed 6-10-4 K8 (24 each: 4 extremes, 10 boundary, 10 random), tiny_model (13) | `--coarse` |
| serial | 64 | 29 (1.81 tics) | 28 (mlp24 2 of 10) | 40 (2.5 tics) | hello, signed, mlp24 24-32-16-8 K3, tiny_model (10 each: 4 extremes, 6 random) | preliminary scan |
| clock_margin (opt 2) | 512 | 80 (5 tics), the other constants shipped; 72 with the others at their defaults | 72 (tiny_model 6 of 33), others shipped | 192 (12 tics = 2.4 x the edge; the rule gives 160) | hello, mlp24, tiny_model, softmax fixture (49, 49, 33, 33 inputs: extremes, boundary vectors, 30 random) | re-scan, 2026-10-28 |
| corr_gap | 64 | not tuned | | 64 | | default |
| tele_gap | 64 | not measured | | 64 | | default |
| chunk_lead | 16 | not measured | | 16 | | default |
| opt=3 clock_margin | 512 | at most 192 (512, 448, 384, 320, 256, 192 pass) | 128 (mnist 1 of 1, mlp24 23 of 24) | 256 | mnist (1 digit), mlp24, hello (24 each), tiny_model (13) | `--copy`, stopped after 128 |
| opt=3 copy_settle | 64 | not measured | | 64 | | default |

clock_margin ships 192, not the rule's 160 (2 x the edge of 80): the static checker needs up to 176 units on MNIST (176, 160 and 144 per
layer) and flags doomplayer_onemap at 144 (a freeze line one tic before the last doll leaves a freeze sector); a doll
starting from rest lags about 154 units plus a tic. 192 is that bound rounded up to 16, plus 16. At opt=3 the copy line
fires before the last light hits of a long crossbar when clock_margin is too small: 128 lost values on MNIST and mlp24.

The serial table comes from the first, preliminary scan (opt=1 then had no corridor rule, the sync barrier used the
legacy sync_wait 400, random inputs only). The serial edge sits between 1.75 and 1.81 tics of clock; values from 16 up
pass on some nets and fail on others (two releases can share a tic below 2 tics), so 40 keeps half a tic above the 2-tic
lifetime of a plane mover. It was not re-scanned with the other constants shipped (the lint campaign's edge probes
cover serial at the checker's boundary, `docs/scheduling.md`).

**The clock_margin edge, re-measured (2026-10-28).** A later run found `tiny_model` failing 5 of 12 prompts at
clock_margin 72, the value this table listed as the edge. The listed edge came from a scan that varies one constant
and keeps the others at their defaults (serial 64, sync_wait 400); opt 2 builds with serial 40 and sync_margin 64, and
there 72 does fail. Re-scanned with 30 random inputs per net plus the extremes and boundary vectors (49 per MLP, 33 per
transformer), 4 nets, resolution 8 units:

| clock_margin | others at defaults | others shipped (what opt 2 builds) |
|---|---|---|
| 128, 120, 112, 96, 88, 80 | pass | pass |
| 72 | pass | **fail**: tiny_model 6/33 |
| 64 | fail: tiny_model 6/33, softmax 27/33 | fail: tiny_model 6/33, softmax 27/33 |
| 56, 48 | fail | not run |
| edge (smallest value at and above which all pass) | 72 | **80** (5 tics) |

The shipped 192 is 2.4 x the true edge of the schedule it ships in (the rule's 2 x would be 160; 192 is the checker's
bound over MNIST and `doomplayer_onemap`, see above). The check inputs of the old table reproduce the old edge (72
with defaults); the failure at 72 appears only with the shipped serial and sync_margin, so a table of one-at-a-time edges
does not bound the combined schedule. `doomformer tune --shipped-others` scans that way. The rows are in
`doomformer/tuning.json` (`constants.clock_margin`, with `others_default` for the old numbers).

**serial and the opt 3 clock_margin, re-scanned with the others shipped (2026-10-28).** Same nets (hello, mlp24, tiny_model, softmax
fixture; 49, 49, 33 and 33 inputs), `tune --const serial --shipped-others --scan-top 48 --resolution 4` and `tune --copy --const
clock_margin --shipped-others --scan-top 256 --resolution 16`:

| constant | passes | fails | edge | shipped | shipped / edge |
|---|---|---|---|---|---|
| serial (opt 2 schedule) | 48, 40, 32, 24 | 20 (mlp24 31 of 49), 16 (mlp24 33 of 49) | 24 (1.5 tics; 29 with the others at their defaults) | 40 | 1.7 |
| opt 3 clock_margin | 256, 192, 160 | 144 (mlp24 48 of 49), 128 (mlp24 49 of 49) | 160 (10 tics) | 256 | 1.6 |

Both shipped values are at least 1.5 times the edge of the schedule they ship in, so `tuning.json` is unchanged; the rows are in
this paragraph, not in the table (the table keeps the first scan's serial rows). The scans stop at the first failing values, and 1.5 x is
the margin the shipped values are held to. MNIST at opt 3 with clock_margin 144 loses values (above), consistent with an edge
near 160; MNIST was not part of this scan.

sync_margin replaces sync_wait at opt >= 2: each layer's first release follows the modelled arrival of its slowest lane
doll (`pure.lane_arrival_units`, from the checker's doll model) plus sync_margin. An earlier, constant sync_wait of 64
(edge 30 on small nets) lost one score value on every prompt of doomnames_onemap (0 of 3) and doomnames_onemap_softmax
(0 of 2): their dolls reach the sync barriers later than those of the small nets, and the checker predicted that
(margins -10 to -3 tics). A per-layer margin does not depend on the layer geometry.

corr_gap is not tuned. A corridor layer spaces its waves `max(span + corr_gap, serial)` apart; two same-target hits of
consecutive waves are `S + p_B - p_A` apart (p: the target cell's offset in each doll's corridor), so only corr_gap
itself bounds the separation for any geometry. A negative corr_gap passed on tiny_model down to -59 (the checker
predicted the edge exactly) but is geometry specific, and a rule that spaced waves by the spread of the cell offsets
failed in the checker on the softmax model (dolls move slower than 16 units per tic through corridors).

## Joint runs

All shipped constants together, fresh inputs (seed 200 + net index; 10 random plus the extremes, plus 15 boundary
vectors on the MLPs):

| schedule | hello | mlp24 | tiny_model | softmax fixture | doomnames_onemap |
|---|---|---|---|---|---|
| opt=1 passes, `shipped` (what opt=2 builds) | 29/29, 240 tics | 29/29, 608 | 13/13, 736 | 13/13, 1,120 | 3/3, 1,128 |
| opt=3 passes, `shipped` + `shipped_copy` | 29/29, 184 | 29/29, 200 | 13/13, 512 | 13/13, 568 | not run |

For comparison, at opt=1 with the default constants the same nets take 384, 960, 1,048 and 1,488 tics.

## Commands

```
doomformer tune --const serial,clock_margin --nets hello,signed,mlp24,tiny --inputs 6 --workers 2
doomformer tune --const clock_margin --nets hello,mlp24,tiny,softmax --inputs 30 --shipped-others --scan-top 128 --resolution 8 --workers 3
doomformer tune --hashes           # opt 0 sha256 of six models against doomformer/opt0_hashes.json (no game)
doomformer tune --coarse --const sync_margin --nets hello,signed,tiny --inputs 10 --workers 1
doomformer tune --copy --coarse --no-predict --const clock_margin --nets mnist,mlp24,hello,tiny --inputs 10 --workers 1
doomformer tune --joint --no-predict --nets hello,mlp24,tiny,softmax,doomnames_onemap --inputs 10 --workers 1
doomformer tune --joint --copy --no-predict --nets hello,mlp24,tiny,softmax --inputs 10 --workers 1
doomformer tune --check            # the committed constants at opt=2 on fresh inputs; exit 1 on any mismatch
```

The serial table was written by an earlier tune.py (no boundary inputs, sync_wait 400); the first command regenerates it
with today's input sets. The clock_margin (opt 2) row is the re-scan above. The opt=3 clock_margin scan was stopped after
128, under the time box. The logs (one JSON line per map run) are kept outside the repository.

## `tune --check`, run end to end (2026-10-28)

`doomformer tune --check --trained --inputs 30 --workers 2 --no-predict` builds opt 2 maps from the committed
`tuning.json` and runs every reference net on fresh inputs (seeds 200 + net index: extremes, boundary vectors, 30
random; the trained models run their prompts): 331 of 331 passes exact, 5.6 minutes (hello 49, mlp24 49, mlp16 54,
signed 54, fold 54, tiny_model 33, softmax fixture 33, `doomnames_onemap` 3 prompts, its softmax twin 2). With `--copy`
(opt 3 maps; hello, mlp24, tiny_model, softmax, the two trained models): 169 of 169 exact, 4.1 minutes. Logs:
[`models/schedule/revalidate/`](../models/schedule/revalidate) (`tune-check-opt2.log`, `tune-check-opt3.log`,
`commands.txt`). Machine load was 12 to 46; passes and game tics do not depend on it.

`induction_onemap` (T = 7, 2 layers; the static checker flags it at opt 2 and 3 for a sync wait of 2 tics against the minimum of 4,
which DOOM does not bear out) is a reference net of `--check` too: 3 prompts (`ABCAB`, `DEFD`, `GHIJG`), 3 of 3 exact at opt 2
(2,160 tics) and at opt 3 (944 tics); `tune-check-induction-opt2.log`, `tune-check-induction-opt3.log`.

Not tested: macOS, other ViZDoom versions; the trained models in `--check` run 2 or 3 prompts, not 30 inputs. A
`copy_settle` below 1 at opt 3 now raises a `ValueError` naming `Timing.copy_settle` (it used to fail inside the build with
"overlapping sectors").
