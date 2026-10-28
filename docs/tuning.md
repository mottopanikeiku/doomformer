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
| clock_margin | 512 | 72 (4.5 tics) | 71 (tiny_model 3 of 9) | 192 (12 tics; the rule gives 144) | as serial | preliminary scan, then the checker's bound |
| corr_gap | 64 | not tuned | | 64 | | default |
| tele_gap | 64 | not measured | | 64 | | default |
| chunk_lead | 16 | not measured | | 16 | | default |
| opt=3 clock_margin | 512 | at most 192 (512, 448, 384, 320, 256, 192 pass) | 128 (mnist 1 of 1, mlp24 23 of 24) | 256 | mnist (1 digit), mlp24, hello (24 each), tiny_model (13) | `--copy`, stopped after 128 |
| opt=3 copy_settle | 64 | not measured | | 64 | | default |

clock_margin ships 192, not the rule's 144: the static checker needs up to 176 units on MNIST (176, 160 and 144 per
layer) and flags doomplayer_onemap at 144 (a freeze line one tic before the last doll leaves a freeze sector); a doll
starting from rest lags about 154 units plus a tic. 192 is that bound rounded up to 16, plus 16. At opt=3 the copy line
fires before the last light hits of a long crossbar when clock_margin is too small: 128 lost values on MNIST and mlp24.

The serial and clock_margin tables come from the first, preliminary scan (opt=1 then had no corridor rule, the sync
barrier used the legacy sync_wait 400, random inputs only). The serial edge sits between 1.75 and 1.81 tics of clock;
values from 16 up pass on some nets and fail on others (two releases can share a tic below 2 tics), so 40 keeps half a
tic above the 2-tic lifetime of a plane mover.

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
doomformer tune --coarse --const sync_margin --nets hello,signed,tiny --inputs 10 --workers 1
doomformer tune --copy --coarse --no-predict --const clock_margin --nets mnist,mlp24,hello,tiny --inputs 10 --workers 1
doomformer tune --joint --no-predict --nets hello,mlp24,tiny,softmax,doomnames_onemap --inputs 10 --workers 1
doomformer tune --joint --copy --no-predict --nets hello,mlp24,tiny,softmax --inputs 10 --workers 1
doomformer tune --check            # the committed constants at opt=2 on fresh inputs; exit 1 on any mismatch
```

The serial and clock_margin tables were written by an earlier tune.py (no boundary inputs, sync_wait 400); the first
command regenerates them with today's input sets. The opt=3 clock_margin scan was stopped after 128, under the time
box. The logs (one JSON line per map run) are kept outside the repository. Not tested: macOS, other ViZDoom versions,
`doomformer tune --check` end to end (its maps are the joint runs' maps, built from the committed file).
