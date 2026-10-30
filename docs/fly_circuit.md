# The FlyWire circuits: method, numbers and deviations

How `models/fly/feeding.json` and `models/fly/brain.json` are built, how they compare with the float whole-brain model,
and where they deviate. The demos: [`FLY.md`](FLY.md). Data licence: [`models/fly/NOTICE.md`](../models/fly/NOTICE.md).

`demos/fly.py` produces every table below. It builds the feeding circuit in 297 s and the combined one in 368 s on 4
threads.

The integer engine is `doomformer/spiking.py`: one shared threshold and leak, binary inputs, bounded potentials. A
random 180-neuron, 5,102-synapse network is 40/40 steps exact, and its `leak` negative control is caught.

# Feeding subcircuit as a DOOM IntNet (`models/fly/feeding.json`)

## Method

1. **Float full-brain reference** (`fly.Brain`). It reimplements the Shiu et al. 2024 equations on all 138,639 neurons
   and 15.09M neuron pairs of FlyWire v783.

   | parameter | value |
   |---|---|
   | dynamics | `dv/dt=(v0-v+g)/20ms`, `dg/dt=-g/5ms` |
   | threshold, reset | -45 mV, -52 mV and g=0 |
   | refractory, delay | 2.2 ms, 1.8 ms |
   | weight | `w = 0.275 mV x signed count` |
   | step | dt 0.1 ms, exact exponential update |
   | Poisson input | +68.75 mV per event |
   | trials | 1 s, 5 seeds, ~4 s per trial |

   The comparison is with the paper's text (Fig. 1c, 3b).
2. **Subcircuit** (`select_subcircuit`). Candidates are the 506 non-GRN neurons firing >= 0.5 Hz under sugar,
   sugar+bitter or bitter. They are ranked by activity-weighted influence on MN9, propagated 5 hops back. The top 200 are
   kept. The taste neurons are input lines. Synapses with |count| < 4 are dropped.
3. **Integerization** (`integerize`):
   `w = clip(round(count x g x theta/7mV x 0.275mV x 5ms/20ms), +-200)` and `leak = round(lam x theta x dt/20ms)`. Inputs
   are seeded Bernoulli trains, p = 200 Hz x dt per line. One global gain, no per-neuron gains, no damping.
4. **Grid search**: 432 candidates, 45 valid. Valid means no overflow, max|v| <= 0.8 V_MAX, sugar MN9 within [0.5,2] x
   float, sugar+bitter below 10 % of sugar, and a silent rest.
   Chosen: dt 1 ms, theta 100, leak 1, g 0.4, inh/exc 1, so weights = round(0.3929 x count). T = 120 steps = 120 ms.
5. **Validation**: `IntNet.run` on seeds 0-9 against fresh float seeds (100+), disjoint from the selection seeds.

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

This matches the paper qualitatively. Absolute rates are not comparable.

| circuit | value |
|---|---|
| neurons N, input lines M | 200, 64 (23 sugar + 41 bitter) |
| nnz(W) + nnz(U) | 3144 + 650 = 3794 |
| max \|w\|, T | 200, 120 |
| pairs kept | 3794 of 9143, 91.6 % of the synapse mass |
| MN9 L/R input synapses kept | 2386/5579, 2670/5886 |

All DOOM limits are met.

Plain integer model, MN9 spikes L/R in 120 steps (120 ms):

```
condition     int mean (10 seeds)  float full brain (fresh seeds)  Jaccard active  rate corr  max|v|
none          0.0/0.0              0.0/0.0                        1.00            nan        0
sugar         25.9/17.8            12.8/6.6                       0.84            0.78       1290
sugar+bitter  0.0/0.0              0.2/0.0                        0.76            0.90       1255
bitter        0.0/0.0              0.0/0.0                        0.65            0.88       1510
```

Sugar fires MN9 in 10/10 seeds, from 19/14 to 32/21 spikes. Sugar+bitter, bitter and no input give 0 spikes in 10/10.
Max |v| is 1510 of 12000.

**Rest after the stimulus.** With zero input all 200 neurons reach a silent fixed point. After sugar the last spike comes
within 12 steps and v is static within 421 steps. After sugar+bitter this takes 1308 steps, after bitter 1507, because
deeply inhibited neurons recover at leak 1 per step. 0 failures in 60 fresh seeds per condition.

The silent fixed point is empirical: the excitatory branching ratio rho(W+/theta) is 1.6.

## Deviations

* **Sugar overshoots.** MN9 fires 25.9/17.8 against 12.8/6.6 float spikes, 2.25x for L+R. The circuit is right on the
  on/off logic, not on rates. The pruned float subnet gives 13.7/7.3, so the overshoot comes from integerization.
* **Steeper dose response.** Held-out sugar rates give these MN9 spikes in 120 ms. Re-calibrate before using graded
  intensities.

  | sugar | float | integer |
  |---|---|---|
  | 50 Hz | 1.3/1.7 | 0.0/0.0 |
  | 100 Hz | 6.7/5.0 | 0.3/0.2 |
  | 400 Hz | 17.0/9.7 | 70.3/45.5 |

* **No |v| guarantee outside the Bernoulli regime.** All 64 lines on at every step overflows (12154 > 12000).
* **Time.** One step is 1 ms. Synaptic delay is 1 step instead of 1.8 ms. Synapses are delta, leak is linear, and there is
  no refractory period.
* **Pruning.** Silencing in this subnet is not equivalent to the whole brain.
* **Fly side.** GRN activation is bilateral as in the notebook, and the connectome is v783, not v630.

### Fitted alternative (not shipped)

Per-neuron rate-matched gains add ~400 fitted parameters. They match rates better: sugar MN9 12.5/6.9 against float 12.8/6.6. It fails
the rest criterion: 11 of 60 fresh sugar trials latch into overflowing reverberation.

## Licence of the data (terms as read on 2026-10-16)

* **FlyWire public-release data (v783) is CC BY-NC 4.0**: https://home.flywire.ai/guidelines. FlyWire annotations are
  also CC-BY-NC 4.0 (https://home.flywire.ai/tos). The Nature paper text is CC BY 4.0, which does not cover the data
  (Dorkenwald et al. 2024, https://doi.org/10.1038/s41586-024-07558-y). Data via Codex
  https://codex.flywire.ai/api/download and Zenodo https://doi.org/10.5281/zenodo.10676866, licence field unverified.
* The parquet redistributed by Eon (https://github.com/eonsystemspbc/fly-brain) and Shiu
  (https://github.com/philshiu/Drosophila_brain_model) sits in GPL-2.0-or-later and MIT code repositories. No relicensing
  statement for the data was found. The data derives from the FlyWire release, so the NC condition travels with it. This
  is a reading of the licences, not legal advice.
* **`models/fly/feeding.json`** (200 FlyWire ids, ~3800 integer weights) is committed under CC BY-NC 4.0 for
  non-commercial use with attribution. Commercial use needs permission from FlyWire (flywire@princeton.edu). The file
  carries the notice in `meta.licence`.
* The raw data is not in this repository. `demos/fly.py` reimplements the equations and copies no GPL code. The GRN id
  lists are data from the Shiu example notebook.

Required attribution, from the [FlyWire citation guide](https://docs.google.com/spreadsheets/d/1eOPxOYoalArVDIhzVis3CsKjETzkdFkRtXSHHY3XdSU):

| for | cite |
|---|---|
| the data | Dorkenwald et al. 2024 (https://doi.org/10.1038/s41586-024-07558-y), Schlegel et al. 2024 (https://doi.org/10.1038/s41586-024-07686-5) |
| synapses | Zheng et al. 2018 (https://doi.org/10.1016/j.cell.2018.06.019), Buhmann et al. 2021 (https://doi.org/10.1038/s41592-021-01183-7), Heinrich et al. 2018 (https://doi.org/10.1007/978-3-030-00934-2_36) |
| neurotransmitter signs | Eckstein, Bates et al. 2024 (https://doi.org/10.1016/j.cell.2024.03.016) |
| model | Shiu et al. 2024 (https://doi.org/10.1038/s41586-024-07763-9) |

---

# Combined brain: feeding + escape + grooming in ONE IntNet (`models/fly/brain.json`)

Code: `demos/fly.py`, section "Combined brain" (`select_multi`, `build_plain`, `search_brain`, `eval_brain`,
`behaviour_check`, `rest_test`).

## Float full-brain findings (reimplementation in `demos/fly.py`, FlyWire v783)

* **LC4 alone drives the Giant Fiber.** P9 alone gives 0 and adds nothing to LC4. P9 stays as one input line (p = 0.1).
  LC4 dose, 3 seeds, 1 s:

  | LC4 | 50 Hz | 100 Hz | 200 Hz | 400 Hz |
  |---|---|---|---|---|
  | GF_1, GF_2 (Hz) | 12.7, 39.3 | 68, 67 | 139, 142 | 237, 254 |

* **Johnston's organ drives aDN1_right only.** aDN1_left is 0 in every JO condition, so grooming uses aDN1_right alone.
  Only JO_EDM drives it as a single group. The circuit uses 300 Hz, the notebook's JO rate.
* **No float crosstalk.** Sugar and bitter never drive GF or aDN1, LC4 never MN9 or aDN1, JO never MN9 or GF.

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
JO_EDM @ 300 Hz (40 neurons)   18.0        0.0      
[...] the other nine JO groups @ 300 Hz: 0.0 / 0.0
```

## Joint selection

Candidates are the 1968 non-input neurons >= 0.5 Hz in any selection condition. Each behaviour ranks them by influence on
its outputs. The top 90 (feeding), 25 (escape) and 140 (grooming) are united into 255 neurons. Input lines keep the most
synapse mass: LC4 is capped at 40 of 104 and JO at 96 of 209. Synapses need |count| >= 3.

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

The pruned float subnet reproduces all three behaviours (500 ms, 2 seeds):

| output | pruned float | full brain |
|---|---|---|
| MN9 | 108 Hz | 98 Hz |
| GF | 90 Hz | 142 Hz |
| aDN1_right | 59 Hz | 68 Hz |

## Plain integerization (one global scale; no per-neuron gains, no damping)

The weight formula is the feeding one, with inhibitory synapses scaled by inh_ratio. Chosen: **dt 1 ms, theta 700,
leak 5 (lam 0.15), g 0.5 (weights = round(0.4911 x count)), inhibitory x 0.7**. A candidate is valid when every output
lies within [0.4, 2.5] x float when on and below 10 % of the on count when off. This covers crosstalk and mixtures.
Rest must hold on 20 more seeds x 11 conditions.

Input rates are Bernoulli p per 1 ms step, also in `meta.input_groups`, `input_hz`, `input_p_per_step`:

| group | lines | rate | p |
|---|---|---|---|
| sugar | 23 | 200 Hz | 0.2 |
| bitter | 40 | 200 Hz | 0.2 |
| LC4 | 40 | 200 Hz | 0.2 |
| JO | 96 | 300 Hz | 0.3 |
| P9 | 1 | 100 Hz | 0.1 |

```
plain integerization grid: 128 candidates, 10 valid (149 s)

=== 10 best valid by log error ===
dt ms  theta  lam   gain g  inh/exc  log err  max|v|  feeding sugar  escape lc4  grooming jo  rest fails
-----  -----  ----  ------  -------  -------  ------  -------------  ----------  -----------  ----------
1.0    700    0.15  0.5     0.7      0.151    7016    19.8           55.2        4.6          0/220     
1.0    700    0.15  0.46    0.7      0.154    6404    17.0           48.8        3.5          -/-       
1.0    700    0.15  0.54    0.7      0.155    7668    23.5           62.1        5.2          -/-       
[...]

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
[...]
lc4+jo        35                 6                              159                 True                0            
rest failures in total: 0 of 660
stress, all 200 input lines on every step for 120 steps: OVERFLOW (v[720575940620553094] = 13068 outside +-12000)

saved models/fly/brain.json (105417 bytes); IntNet.load round trip identical: True; limits: {'n': 255, 'm': 200, 'nnz': 5052, 'w': 200, 't': 120, 'ok': True}
```

## Deviations

* **Rates are not matched, logic is.** Escape overshoots at 1.6x the float spikes. Grooming undershoots at 0.57x, on
  small counts of 3-5 spikes. The float feeding output is asymmetric, the integer one symmetric. All outputs lie inside
  the [0.4, 2.5] band by construction, not tighter.
* **Fidelity.** Active-set Jaccard is 0.85-0.96 and per-neuron rate correlation 0.82-0.94. The JO mixtures are worst.
* **Rest is empirical.** 0 failures in 660 trials. Bitter and sugar+bitter are slowest to go static, because deeply
  inhibited neurons recover at leak 5 per step.
* **No |v| guarantee outside the Bernoulli regime.** All 200 lines on overflows (13068 > 12000).
* **Feeding dose response is not validated for this net.** The feeding-only net's steep dose response likely applies.
* **Mixtures.** Only sugar+lc4, sugar+jo and lc4+jo are tested, not triples. aDN1_left is absent. Time and synapse
  simplifications are as for the feeding net.
* **Licence.** As above. `brain.json` adds 255 FlyWire ids and 200 input ids from the same release.
