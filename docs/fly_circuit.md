# The FlyWire circuits: method, numbers and deviations

How `models/fly/feeding.json` and `models/fly/brain.json` were made, how they compare with the float whole-brain model, and
where they deviate from it. Narrative and films: [`FLY.md`](FLY.md). Data licence and attribution:
[`models/fly/NOTICE.md`](../models/fly/NOTICE.md).

Files: `demos/fly.py` (the library and the code behind every table below), `models/fly/feeding.json` (the integer
network, **plain global-scale variant**) and `models/fly/brain.json` (the combined network). The feeding-circuit run took
297 s wall and the combined-circuit run 368 s, both on 4 threads. All numbers below come from those runs unless marked as
taken from the literature. The "fitted alternative" of the feeding circuit (below) is not shipped.

# Feeding subcircuit as a DOOM IntNet (`models/fly/feeding.json`)

## Method

1. **Float full-brain reference** (`fly.Brain`, reimplemented from the Shiu et al. 2024 equations, not copied): 138,639
   neurons, 15.09M neuron pairs (FlyWire v783 via Eon's parquet), `dv/dt=(v0-v+g)/20ms`, `dg/dt=-g/5ms`, threshold -45 mV,
   reset -52 mV and g=0, refractory 2.2 ms (v,g frozen, input still accumulates in g), delay 1.8 ms,
   `w = 0.275 mV x signed count`, dt 0.1 ms, exact exponential update, sparse (only spiking columns per step). Poisson-driven
   neurons get +68.75 mV per event (no refractory). 1 s trials, 5 seeds, ~4 s per trial. The reference notebook has no saved
   outputs, so the comparison is with the paper's text: sugar-GRN activation drives MN9, adding bitter GRN activity inhibits
   it (Shiu et al., Nature 2024, Fig. 1c, 3b).
2. **Subcircuit** (`select_subcircuit`): candidates = non-GRN neurons firing >= 0.5 Hz in the float model under sugar,
   sugar+bitter or bitter (506). Rank by activity-weighted backward influence on MN9 (share of a post neuron's
   rate-weighted input that comes from each pre neuron, propagated 5 hops back from MN9, max over conditions). Keep the top
   200 (MN9 L/R included). The 23 sugar and 41 of 42 bitter GRNs are **input lines** (columns of U), never simulated.
   Synapses with |count| < 4 dropped (smallest threshold that gives nnz <= 4000).
3. **Integerization** (`integerize`): `w = clip(round(count x g x theta/7mV x 0.275mV x 5ms/20ms), +-200)`, i.e. a synapse
   of count c moves v by its time-integrated effect c x 0.06875 mV, with theta standing for the 7 mV threshold gap;
   `leak = round(lam x theta x dt/20ms)`; step = dt ms of fly time; inputs are seeded Bernoulli trains p = 200 Hz x dt per
   GRN line. **Plain variant: ONE global gain g (and one inhibitory/excitatory ratio), no per-neuron gains, no damping.**
   Grid over dt {1,2} ms, theta {100,200,300}, lam {0.15,0.25,0.35}, g (8 values 0.27-0.53), inh/exc {1,0.7,0.5}: 432
   candidates, 45 valid. Valid = no overflow, max|v| <= 0.8 V_MAX, sugar MN9 spikes within [0.5,2] x float count,
   sugar+bitter MN9 < 10 % of sugar, silent fixed point after the input stops (3 seeds x 3 conditions), then the best by
   mean log spike-count error vs float on seeds 300+ that also passes a rest test on 40 more seeds (400+) x 3 conditions.
   Chosen: dt 1 ms, theta 100, leak 1, g 0.4, inh/exc 1 (weights = round(0.3929 x count)); T = 120 steps = 120 ms.
4. **Validation**: `IntNet.run` (the contract incl. the leak rule `[v>=leak]`/`[v<=-leak]`), seeds 0-9, against fresh float
   full-brain seeds (100+, disjoint from the float seeds 10-14 used for selection/fit).

## Results

Float full brain, 5 seeds x 1000 ms:

```
condition     MN9_left Hz   MN9_right Hz  MN9 spikes in 0-240 ms (L / R)  neurons ever active  neurons >= 1 Hz
none          0.0 +- 0.0    0.0 +- 0.0    0.0 / 0.0                       0                    0
sugar         123.8 +- 3.3  71.6 +- 2.4   28.0 / 17.0                     442                  403
sugar+bitter  0.0 +- 0.0    0.0 +- 0.0    0.0 / 0.0                       335                  299
bitter        0.0 +- 0.0    0.0 +- 0.0    0.0 / 0.0                       156                  151
LC4 -> Giant Fiber (3 seeds): P9 100 Hz: GF 0.0/0.0 Hz;  P9 100 Hz + LC4 200 Hz: GF_1 136.7 +- 0.9, GF_2 142.3 +- 2.5 Hz
```

Consistent with the paper qualitatively (sugar -> MN9 on, bitter abolishes it; looming LC4 -> Giant Fiber). Absolute rates
are not comparable (paper: unilateral activation, v630).

Circuit: N = 200, M = 64 (23 sugar + 41 bitter), nnz(W)+nnz(U) = 3144 + 650 = 3794, max |w| 200, T 120: all DOOM limits
met. Kept 3794 of 9143 neuron pairs among the selected set = 91.6 % of the synapse mass (counts); MN9 L/R keep 2386/5579
and 2670/5886 of their input synapses. Dropped: all pairs with |count| < 4 and all 306 non-selected active candidates
(reported, not recovered). All 200 selected neurons are active in at least one float condition.

Plain integer model (`models/fly/feeding.json`), MN9 spikes L/R in 120 steps (120 ms):

```
condition     int mean (10 seeds)  float full brain (fresh seeds)  Jaccard active  rate corr  max|v|
none          0.0/0.0              0.0/0.0                        1.00            nan        0
sugar         25.9/17.8            12.8/6.6                       0.84            0.78       1290
sugar+bitter  0.0/0.0              0.2/0.0                        0.76            0.90       1255
bitter        0.0/0.0              0.0/0.0                        0.65            0.88       1510
```

Per seed (sugar, L/R): 20/15, 32/21, 31/20, 27/17, 22/12, 28/19, 21/16, 31/21, 19/14, 28/23; every seed of none /
sugar+bitter / bitter gives 0/0. (i)-(iv) hold: sugar MN9 > 0 in 10/10 seeds, sugar+bitter 0 spikes in 10/10 (vs 43.7 mean
for sugar), no input silent, max |v| 1510 of 12000.

**Rest after the stimulus**: zero input after the 120 input steps; all 200 neurons reach a silent fixed point (no spike, v
static, all v < theta). After sugar the last spike is <= 12 steps after the stop, v is static after <= 421 steps; after
sugar+bitter 1308, bitter 1507 steps (slow only because deeply inhibited neurons recover at leak = 1 per step). 0 failures
in 60 fresh seeds per condition (500-559); 0 of 120 in the search. The silent fixed point is **empirical, not guaranteed**:
the excitatory-only branching ratio rho(W+/theta) is 1.6 (>1). One grid candidate latched into a self-sustaining
~24-neuron reverberation (overflow) on 1 of 10 seeds, which is why rest is part of the validity test.

## Deviations

* The plain model overshoots sugar MN9: 25.9/17.8 vs 12.8/6.6 spikes (2.0x / 2.7x; L+R 43.7 vs 19.4 = 2.25x on the
  validation comparison; it passed the [0.5,2]x acceptance band on the selection seeds with 41.0 vs the fit-seed float
  count, i.e. just inside). Active-set Jaccard 0.84 / 0.76 / 0.65 and per-neuron rate correlation 0.78 / 0.90 / 0.88: it is
  functionally right (on/off logic), not rate-accurate. The pruned *float* subnet (no integerization) gives 13.7/7.3, so
  pruning costs almost nothing; the overshoot is integerization.
* Dose response is steeper than float: held-out sugar at 50 / 100 / 400 Hz, float 1.3/1.7, 6.7/5.0, 17.0/9.7 vs plain
  integer 0.0/0.0, 0.3/0.2, 70.3/45.5 spikes in 120 ms (a threshold nonlinearity: the integer neuron has no refractory
  saturation). Do not use graded sugar intensities without re-calibrating.
* Outside the Bernoulli regime there is no |v| guarantee: all 64 lines on at every step overflows (12154 > 12000).
* Time: one step = 1 ms of fly time, synaptic delay 1 step vs 1.8 ms, delta synapses instead of the alpha synapse, linear
  instead of exponential leak, no refractory period. Neuron-level behaviour is rate-matched only in the fitted alternative.
* Pruning keeps strong connections; weak synapses (< 4 counts) and neurons outside the top 200 are gone, so interventions
  (silencing etc.) in this subnet are not equivalent to the whole brain.
* Fly side: bilateral GRN activation (notebook), not unilateral as in the paper; v783 not v630.

### Fitted alternative (reported only, not shipped)

Per-neuron rate-matched excitatory/inhibitory gains (open-loop least squares on float rates, then closed-loop log-ratio
calibration on seeds 200+) plus a global recurrent damping 0.205 (loop-gain bound rho_max 0.9), dt 1 ms, theta 200, leak 5.
These are ~400 fitted parameters beyond the connectome, so it weakens the "the fly's wiring runs in DOOM" claim. It is more
accurate: sugar MN9 12.5/6.9 vs float 12.8/6.6, Jaccard 0.95 / 0.87 / 1.00, rate corr 0.92 / 0.99 / 0.99, max |v| 3301,
sugar+bitter and none give 0 spikes. But it **fails the rest criterion**: 11 of 60 fresh sugar trials latch into
overflowing reverberation; 0/60 for sugar+bitter and bitter. It survives the all-lines-on stress (11637). The plain file is
the one in `models/fly/`.

## Licence of the data (terms as read on 2026-10-16)

* **FlyWire public-release data (v783) is CC BY-NC 4.0** (Attribution-NonCommercial): https://home.flywire.ai/guidelines
  ("FlyWire's public release data is made available under license CC BY-NC 4.0"); annotations contributed on FlyWire are
  also released CC-BY-NC 4.0 (https://home.flywire.ai/tos). The Nature paper text itself is CC BY 4.0, which does not cover
  the data (Dorkenwald et al. 2024, https://doi.org/10.1038/s41586-024-07558-y; data via Codex
  https://codex.flywire.ai/api/download and Zenodo https://doi.org/10.5281/zenodo.10676866, whose licence field is
  unverified).
* The parquet as redistributed by Eon (https://github.com/eonsystemspbc/fly-brain) / Shiu
  (https://github.com/philshiu/Drosophila_brain_model): the repository licence is GPL-2.0-or-later (Eon; Shiu code is MIT),
  which covers **code**. No relicensing statement for the data files was found; they are derived from the CC BY-NC 4.0
  FlyWire release, so the NC condition travels with them (a reading of the licences, not legal advice).
* **`models/fly/feeding.json`** (200 FlyWire ids, ~3800 integer weights derived from synapse counts) is committed: CC BY-NC
  4.0 permits sharing adapted material for **non-commercial** purposes with attribution, a licence link and an indication of
  changes. It is not licensed for commercial use without permission from FlyWire (flywire@princeton.edu). The file carries
  the notice in `meta.licence`.
* Required attribution (FlyWire citation guide,
  https://docs.google.com/spreadsheets/d/1eOPxOYoalArVDIhzVis3CsKjETzkdFkRtXSHHY3XdSU): co-cite Dorkenwald et al. 2024
  (https://doi.org/10.1038/s41586-024-07558-y) and Schlegel et al. 2024 (https://doi.org/10.1038/s41586-024-07686-5); for
  synapses/connectivity also Zheng et al. 2018 (https://doi.org/10.1016/j.cell.2018.06.019), Buhmann et al. 2021
  (https://doi.org/10.1038/s41592-021-01183-7), Heinrich et al. 2018 (https://doi.org/10.1007/978-3-030-00934-2_36);
  neurotransmitter signs: Eckstein, Bates et al. 2024 (https://doi.org/10.1016/j.cell.2024.03.016); model: Shiu et al. 2024
  (https://doi.org/10.1038/s41586-024-07763-9). The raw data is not in this repository. `demos/fly.py` reimplements the
  equations (no GPL code copied; the GRN id lists are data, taken from the Shiu example notebook).

---

# Combined brain: feeding + escape + grooming in ONE IntNet (`models/fly/brain.json`)

`models/fly/feeding.json` is untouched. Code: `demos/fly.py` (section "Combined brain": `select_multi`, `build_plain`,
`search_brain`, `eval_brain`, `behaviour_check`, `rest_test`).

## Float full-brain findings (reimplementation in `demos/fly.py`, FlyWire v783)

* **LC4 -> Giant Fiber: LC4 alone is enough.** LC4 (104 neurons, 200 Hz Poisson) drives GF_1/GF_2 at 139.7 / 144.3 Hz; P9
  (100 Hz) alone gives 0; P9+LC4 gives 134.3 / 146.3 Hz (no change). Dose (separate run, 3 seeds, 1 s): LC4 50 / 100 /
  200 / 400 Hz -> GF 12.7|39.3 / 68|67 / 139|142 / 237|254 Hz. So in the embodiment P9 may be on or off; it is kept as one
  input line (p = 0.1) only to test that.
* **JO -> aDN1: works, but only aDN1_right.** All 209 notebook JO neurons: aDN1_right 5 / 47 / 68 Hz at 100 / 200 / 300 Hz,
  **aDN1_left is 0 in every JO condition** (also each of the 10 JO groups alone). Of the groups only JO_EDM (40 neurons)
  drives aDN1_right alone (18 Hz at 300 Hz); the rest of the response needs the other groups together (the 1090 neurons
  active at 300 Hz). Grooming is therefore the single output aDN1_right; aDN1_left is dropped (not forced). 300 Hz (the
  notebook's JO rate) is used.
* No float crosstalk: sugar/bitter never drive GF or aDN1, LC4 never MN9/aDN1, JO never MN9/GF.

```
=== float full brain, 3 seeds x 1000 ms, output rates in Hz (mean +- sd) ===
condition     MN9_left      MN9_right    GF_1          GF_2          aDN1_right   neurons ever active
------------  ------------  -----------  ------------  ------------  -----------  -------------------
sugar         125.0 +- 3.7  70.7 +- 1.2  0.0 +- 0.0    0.0 +- 0.0    0.0 +- 0.0   437                
sugar+bitter  0.0 +- 0.0    0.0 +- 0.0   0.0 +- 0.0    0.0 +- 0.0    0.0 +- 0.0   334                
bitter        0.0 +- 0.0    0.0 +- 0.0   0.0 +- 0.0    0.0 +- 0.0    0.0 +- 0.0   152                
lc4           0.0 +- 0.0    0.0 +- 0.0   139.7 +- 1.2  144.3 +- 0.9  0.0 +- 0.0   844                
lc4+p9        0.0 +- 0.0    0.0 +- 0.0   134.3 +- 4.5  146.3 +- 1.2  0.0 +- 0.0   876                
jo            0.0 +- 0.0    0.0 +- 0.0   0.0 +- 0.0    0.0 +- 0.0    68.0 +- 1.6  1090               
p9            0.0 +- 0.0    0.0 +- 0.0   0.0 +- 0.0    0.0 +- 0.0    0.0 +- 0.0   144                

=== float full brain: which Johnston's-organ groups drive aDN1 (2 seeds x 500 ms, Hz) ===
stimulus                       aDN1_right  aDN1_left
-----------------------------  ----------  ---------
all JO @ 100 Hz (209 neurons)  5.0         0.0      
all JO @ 200 Hz (209 neurons)  47.0        0.0      
all JO @ 300 Hz (209 neurons)  68.0        0.0      
JO_EV @ 300 Hz (22 neurons)    0.0         0.0      
JO_EDC @ 300 Hz (13 neurons)   0.0         0.0      
JO_EDM @ 300 Hz (40 neurons)   18.0        0.0      
JO_EDP @ 300 Hz (12 neurons)   0.0         0.0      
JO_EVL @ 300 Hz (27 neurons)   0.0         0.0      
JO_EVM @ 300 Hz (43 neurons)   0.0         0.0      
JO_EVP @ 300 Hz (18 neurons)   0.0         0.0      
JO_CA @ 300 Hz (17 neurons)    0.0         0.0      
JO_CL @ 300 Hz (8 neurons)     0.0         0.0      
JO_CM @ 300 Hz (9 neurons)     0.0         0.0
```

## Joint selection

Candidates: non-input neurons >= 0.5 Hz in any of sugar, sugar+bitter, bitter, lc4, lc4+p9, jo (1968). Per behaviour the
candidates are ranked by the same activity-weighted backward influence as before on that behaviour's outputs (feeding: MN9
L/R; escape: GF 1/2; grooming: aDN1_right); the top 90 / 25 / 140 are united (255 neurons, outputs always in). Escape needs
very few neurons (a handful of LC4 -> LC4-target -> GF steps); grooming needs the most. Input lines (never simulated): all
23 sugar, 40 of 42 bitter, P9 1 of 2, and the LC4 / JO lines with the most synapse mass (LC4 capped at 40 of 104, JO 96 of
209; the 27 dropped lines are the weakest). Pruned by |count| >= 3.

```
=== joint subcircuit (before integer rounding) ===
quantity                                                    value                                        
----------------------------------------------------------  ---------------------------------------------
candidates (non-input, >= 0.5 Hz in a selection condition)  1968                                         
quotas feeding / escape / grooming (top by influence)       90 / 25 / 140                                
N simulated neurons (limit 260)                             255                                          
M input lines (limit 200)                                   200: sugar 23, bitter 40, lc4 40, jo 96, p9 1
nnz(W) + nnz(U) (limit 6000)                                3297 + 1755 = 5052                           
pruning |synapse count| >= c_min                            3                                            
input lines dropped (least mass; LC4 capped at 40)          27                                           
pairs kept / considered                                     5052 / 10478                                 
synapse mass kept / considered                              88471 / 98411 (89.9%)
```

The pruned *float* subnet (no integerization, 500 ms, 2 seeds) reproduces all three behaviours: MN9 108 Hz, GF 90 Hz,
aDN1_right 59 Hz vs 98 / 142 / 68 Hz in the full brain (separate run with these quotas).

## Plain integerization (one global scale; no per-neuron gains, no damping)

`w = clip(round(count x g x theta/7mV x 0.275mV x 5ms/20ms), +-200)`, inhibitory synapses additionally x inh_ratio; leak =
round(lam x theta x dt/20 ms); exact contract incl. the `[v>=leak]` leak rule; one step = 1 ms; T = 120 steps = 120 ms.
Chosen: **dt 1 ms, theta 700, leak 5 (lam 0.15), g 0.5 (weights = round(0.4911 x count)), inhibitory x 0.7**. Valid = no
overflow, max|v| <= 0.8 V_MAX, every behaviour's output within [0.4, 2.5] x the float count when the float is on and <= 10 %
of the on count when off (this includes crosstalk, sugar+bitter and all mixtures), silent rest, then the lowest log
spike-count error that also passes rest on 20 more seeds x 11 conditions (0/220).

**Input rates (per group, Bernoulli p per 1 ms step; also in `meta.input_groups / input_hz / input_p_per_step`):** sugar 200
Hz -> p 0.2 (23 lines), bitter 200 Hz -> 0.2 (40 lines), LC4 200 Hz -> 0.2 (40 lines), JO 300 Hz -> 0.3 (96 lines), P9 100
Hz -> 0.1 (1 line). Stimulus conditions are in `meta.conditions`.

```
plain integerization grid: 128 candidates, 10 valid (149 s)

=== 10 best valid by log error ===
dt ms  theta  lam   gain g  inh/exc  log err  max|v|  feeding sugar  escape lc4  grooming jo  rest fails
-----  -----  ----  ------  -------  -------  ------  -------------  ----------  -----------  ----------
1.0    700    0.15  0.5     0.7      0.151    7016    19.8           55.2        4.6          0/220     
1.0    700    0.15  0.46    0.7      0.154    6404    17.0           48.8        3.5          -/-       
1.0    700    0.15  0.54    0.7      0.155    7668    23.5           62.1        5.2          -/-       
1.0    700    0.15  0.58    0.7      0.164    8508    27.0           68.2        6.1          -/-       
1.0    700    0.25  0.58    0.7      0.169    7626    23.6           67.8        4.6          -/-       
1.0    700    0.25  0.54    0.7      0.169    6978    20.5           62.0        3.8          -/-       
1.0    500    0.15  0.54    1.0      0.178    7778    27.9           53.8        4.6          -/-       
1.0    700    0.15  0.58    1.0      0.179    9596    13.6           53.4        4.2          -/-       
1.0    700    0.15  0.54    1.0      0.180    8683    12.0           50.9        3.5          -/-       
1.0    500    0.15  0.58    1.0      0.185    8898    30.5           63.4        5.2          -/-       
chosen: {'dt': 1.0, 'theta': 700, 'lam': 0.15, 'gain': 0.5, 'inh_ratio': 0.7, 'valid': True, 'err': 0.15089412721083603, 'vmax': 7016, 'feeding': 19.8, 'escape': 55.2, 'grooming': 4.6, 'rest_fail': 0, 'rest_trials': 220}

=== DOOM capacity check (this map: 260 / 200 / 6000 / 200 / 120) ===
N    M    nnz(W)+nnz(U)  max |w|  T    ok  
---  ---  -------------  -------  ---  ----
255  200  5052           200      120  True
```

## Validation (IntNet.run, 10 seeds per stimulus, vs float full brain on 5 fresh seeds, 120 ms window)

```
=== CROSSTALK: integer output spikes in 120 steps = 120 ms, mean [min-max] over 10 IntNet.run seeds / float full-brain mean (5 fresh seeds) ===
stimulus      MN9_left            MN9_right          GF_1                 GF_2                 aDN1_right     
------------  ------------------  -----------------  -------------------  -------------------  ---------------
none          0.0 [0-0] / 0.0     0.0 [0-0] / 0.0    0.0 [0-0] / 0.0      0.0 [0-0] / 0.0      0.0 [0-0] / 0.0
sugar         10.3 [9-12] / 12.8  10.0 [9-12] / 6.6  0.0 [0-0] / 0.0      0.0 [0-0] / 0.0      0.0 [0-0] / 0.0
bitter        0.0 [0-0] / 0.0     0.0 [0-0] / 0.0    0.0 [0-0] / 0.0      0.0 [0-0] / 0.0      0.0 [0-0] / 0.0
sugar+bitter  0.0 [0-0] / 0.2     0.0 [0-0] / 0.0    0.0 [0-0] / 0.0      0.0 [0-0] / 0.0      0.0 [0-0] / 0.0
lc4           0.0 [0-0] / 0.0     0.0 [0-0] / 0.0    27.0 [25-29] / 17.2  28.9 [26-31] / 18.2  0.0 [0-0] / 0.0
p9            0.0 [0-0] / 0.0     0.0 [0-0] / 0.0    0.0 [0-0] / 0.0      0.0 [0-0] / 0.0      0.0 [0-0] / 0.0
lc4+p9        0.0 [0-0] / 0.0     0.0 [0-0] / 0.0    27.1 [25-29] / 16.4  28.9 [26-31] / 18.6  0.0 [0-0] / 0.0
jo            0.0 [0-0] / 0.0     0.0 [0-0] / 0.0    0.0 [0-0] / 0.0      0.0 [0-0] / 0.0      4.0 [3-5] / 7.0
sugar+lc4     10.2 [9-12] / 12.8  9.7 [8-12] / 6.6   26.7 [25-28] / 17.6  28.4 [26-30] / 17.4  0.0 [0-0] / 0.0
sugar+jo      10.2 [9-12] / 14.6  9.7 [9-11] / 7.4   0.0 [0-0] / 0.0      0.0 [0-0] / 0.0      4.2 [4-5] / 7.0
lc4+jo        0.0 [0-0] / 0.0     0.0 [0-0] / 0.0    26.3 [25-28] / 17.0  29.1 [26-31] / 19.4  3.9 [3-4] / 6.4

=== fidelity per stimulus (active = mean >= 0.5 spikes in window; corr = per-neuron count correlation) ===
stimulus      active float  active integer  Jaccard  rate corr  max |v| (all seeds)
------------  ------------  --------------  -------  ---------  -------------------
none          0             0               1.00     nan        0                  
sugar         80            82              0.93     0.91       1405               
bitter        12            12              0.85     0.91       7047               
sugar+bitter  53            56              0.85     0.94       5606               
lc4           26            25              0.96     0.94       4005               
p9            0             0               1.00     nan        27                 
lc4+p9        26            25              0.96     0.94       4005               
jo            137           130             0.91     0.85       1992               
sugar+lc4     106           107             0.94     0.92       3860               
sugar+jo      217           211             0.93     0.82       2104               
lc4+jo        150           152             0.91     0.88       1625               
max |v| over all validation runs: 7047 (bound 12000)
behaviour check (on within [0.4, 2.5] x float, off <= 10% of on): True

=== REST after the input stops (120 input steps, then up to 3000 steps of zero input; worst of 10 seeds; last column: failures in 60 fresh seeds 500-559; 58 s) ===
stimulus      spikes after stop  last spike (steps after stop)  v static from step  silent fixed point  failures / 60
------------  -----------------  -----------------------------  ------------------  ------------------  -------------
none          0                  0                              0                   True                0            
sugar         35                 7                              144                 True                0            
bitter        4                  2                              1409                True                0            
sugar+bitter  21                 5                              1125                True                0            
lc4           16                 6                              824                 True                0            
p9            0                  0                              4                   True                0            
lc4+p9        15                 6                              824                 True                0            
jo            28                 3                              423                 True                0            
sugar+lc4     45                 9                              795                 True                0            
sugar+jo      60                 12                             430                 True                0            
lc4+jo        35                 6                              159                 True                0            
rest failures in total: 0 of 660
stress, all 200 input lines on every step for 120 steps: OVERFLOW (v[720575940620553094] = 13068 outside +-12000)

saved models/fly/brain.json (105417 bytes); IntNet.load round trip identical: True; limits: {'n': 255, 'm': 200, 'nnz': 5052, 'w': 200, 't': 120, 'ok': True}
total 368 s

real	6m8.167s
user	9m7.342s
sys	0m6.370s
```

## Deviations

* **Rates are not matched, logic is.** Escape overshoots (GF 27-29 vs 17-18 spikes in 120 ms, 1.6x), grooming undershoots
  (aDN1_right 4.0 vs 7.0, 0.57x; its window count is small, 3-5 spikes), feeding is left 10.3 vs 12.8 and right 10.0 vs 6.6
  (float is asymmetric, the integer net is symmetric). All outputs lie inside the [0.4, 2.5] acceptance band by
  construction, not tighter.
* Active-set Jaccard 0.85-0.96 (bitter alone 0.85, sugar+bitter 0.85, since the 12 / 53 active neurons are few), per-neuron
  rate correlation 0.82-0.94; the worst are the JO mixtures (0.82-0.88), where 130-210 neurons are active.
* **Rest is empirical.** 0 failures in 660 trials (60 fresh seeds x 11 stimuli). Slowest to go static: bitter (1409 steps)
  and sugar+bitter (1125) because deeply inhibited neurons recover at leak 5 per step; spikes stop within 12 steps after the
  input stops in every condition.
* **No |v| guarantee outside the Bernoulli regime:** all 200 lines on at every step overflows (13068 > 12000). Max |v| in
  validation is 7047 (bitter), 7016 in the grid.
* **Feeding dose response and held-out strengths were not re-validated for the combined net**; the feeding-only net's
  steeper-than-float dose response (above) very likely applies (the feeding output is on a steep g-threshold: g 0.34 -> 5
  spikes, 0.46 -> 190 at theta 100 in the 260-neuron exploration).
* Single-stimulus design: simultaneous mixtures were tested for sugar+lc4, sugar+jo, lc4+jo only (each behaviour stays on,
  others stay silent); not triples. aDN1_left is absent (the float brain does not drive it). Time / synapse simplifications
  as for the feeding net (1 ms step, delta synapse, linear leak, no refractory period).
* Licence: same as above (FlyWire CC BY-NC 4.0; `meta.licence` carries the notice); `brain.json` adds 255 FlyWire ids and
  200 input ids (LC4, JO, GRN, P9 neurons) from the same release.
