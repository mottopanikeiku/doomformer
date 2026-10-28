# Lint soundness campaign (seed 0, nets 0..999)

command: `doomformer lint --fuzz --nets 1000 --seed 0 --workers 6 --fuzz-out models/schedule/lintfuzz-seed0`
command: `doomformer lint --fuzz --edge --nets 500 --seed 0 --workers 6 --fuzz-out models/schedule/lintfuzz-seed0`

The random campaign ran in full (1,000 nets x opt 1, 2, 3 = 3,000 jobs). The edge campaign (`--nets 500`) was stopped by hand after 282 nets (846 jobs) because the shared machine's load average was above 40; every row it wrote is in lintfuzz.jsonl and is counted here. Re-running the same command resumes it.

## random: random timing constants (each constant replaced with probability 1/2 by a draw from its range around the edges of docs/tuning.md)

3000 maps over 983 nets; 2947 scored (build, static verdict, then 16 inputs per map in DOOM at tolerance 0; 52,147 DOOM passes).

| opt | maps scored | false negatives | flagged, DOOM wrong (TP) | flagged, DOOM exact (conservative) | clean, exact (TN) |
|---|---|---|---|---|---|
| 1 | 983 | 0 | 122 | 389 | 472 |
| 2 | 983 | 0 | 41 | 284 | 658 |
| 3 | 981 | 0 | 115 | 262 | 604 |
| all | 2947 | 0 | 278 | 935 | 1734 |

False negatives: **0**. DOOM was wrong on 278 maps. Conservative rate (flagged maps DOOM ran exactly, of all flagged): 77.1 % (935 of 1213). Not scored: build_error 2, unsupported 51.

Why the checker flagged (hazard = two plane writes that can share a tic; the other names are the timing margins below their minimum):

| flagged for | DOOM wrong (TP) | DOOM exact (conservative) |
|---|---|---|
| sync_wait | 83 | 920 |
| clock_margin | 80 | 2 |
| clock_margin + sync_wait | 39 | 2 |
| hazard + serial | 37 | 3 |
| hazard + serial + sync_wait | 21 | 0 |
| hazard | 9 | 6 |
| hazard + sync_wait | 8 | 2 |
| hazard + clock_margin | 1 | 0 |

## edge: the checker's own boundary: per net, opt level and constant (serial, clock_margin; the others shipped) the smallest value the checker calls clean (edge-clean) and one below it (edge-flagged)

2456 maps over 280 nets; 2450 scored (build, static verdict, then 16 inputs per map in DOOM at tolerance 0; 45,402 DOOM passes).

| opt | maps scored | false negatives | flagged, DOOM wrong (TP) | flagged, DOOM exact (conservative) | clean, exact (TN) |
|---|---|---|---|---|---|
| 1 | 835 | 0 | 180 | 97 | 558 |
| 2 | 836 | 0 | 181 | 97 | 558 |
| 3 | 779 | 0 | 181 | 38 | 560 |
| all | 2450 | 0 | 542 | 232 | 1676 |

False negatives: **0**. DOOM was wrong on 542 maps. Conservative rate (flagged maps DOOM ran exactly, of all flagged): 30.0 % (232 of 774). Not scored: unsupported 6.

Why the checker flagged (hazard = two plane writes that can share a tic; the other names are the timing margins below their minimum):

| flagged for | DOOM wrong (TP) | DOOM exact (conservative) |
|---|---|---|
| clock_margin | 230 | 119 |
| hazard + serial | 248 | 88 |
| hazard | 63 | 23 |
| serial | 1 | 2 |

| constant, role | FN | TP | conservative | TN |
|---|---|---|---|---|
| clock_margin edge-clean | 0 | 0 | 0 | 838 |
| clock_margin edge-flagged | 0 | 230 | 119 | 0 |
| serial edge-clean | 0 | 0 | 0 | 838 |
| serial edge-flagged | 0 | 312 | 113 | 0 |
