# doomnames_softmax

The doomnames transformer with softmax attention (`attn="softmax"`, `R=2`). Shape: 1 layer, d=16, 2 heads, dm=32,
context 16, K=KA=KM=7. Data: the 136 DOOM level names in `corpus.json`.

Attention, per head h, query t, keys u <= t:

```
e_tu  = adc_relu(q_t . k_u, thr_a)             in [0, 7]
den_t = 1 + sum_u e_tu                          (the 1 is a null slot: logit at thr_a[0], value 0)
num_t = sum_u e_tu * v_u
ao_t  = trunc(R * num_t / den_t)                in [-13, 13]   (R*7 - 1 = 13 comparison levels per sign)
```

thr_a is trained as `c + sigma*ln(k+1)`, so e is a quantised exponential of the score. The division by the sum of
weights happens at run time. There is no thr_o array.

```sh
python training/train_qat.py --attn softmax --R 2   # doomnames recipe: seed 1, steps 300/6000/4000, lr 0.02/0.01/0.004, --recal
python training/eval_model.py models/doomnames_softmax
```

`train_info.json` holds args, c, sigma and thr_a. Torch int64 forward equals `intref.forward` on 136/136 names.
`intref.check_limits` is clean: max |P|, |N| 1392, max den 65.

## Results

Teacher-forced accuracy is over the 1505 target tokens of the 136 training names. Float accuracy is the train_qat float
stage.

| | doomnames (relu, shipped) | doomnames_softmax (R=2, kept) | softmax R=1 (not kept) |
|---|---|---|---|
| float teacher-forced accuracy | 0.8173 | 0.8272 | 0.8140 |
| integer teacher-forced accuracy | 1235/1505 = 0.8206 | 1221/1505 = 0.8113 | 1214/1505 = 0.8066 |
| names reproduced token-perfectly (teacher-forced) | 1/136 | 1/136 | 1/136 |
| greedy, empty prompt | THE COURTYARD (corpus) | THE CHASM (corpus) | THE CITADEL (corpus) |
| greedy, 21 prompts (empty + first letters) | 20/21 in corpus | 16/21 in corpus | 16/21 in corpus |
| demo names (temperature 180, seeds 20..29) | 6/10 in corpus | 3/10 in corpus | 5/10 in corpus |
| training wall time | 1766 s (shipped record) | 1505 s | 1510 s |

Softmax attention costs 0.9 points of integer accuracy against the relu model and samples fewer corpus names at
temperature 180.

Names the default demo writes (`intref.generate(m, "", max_new=15, temperature=180, seed=s)`):

| seed | doomnames (relu) | doomnames_softmax (R=2) |
|---|---|---|
| 20 | GHEER | FINAL INEALL |
| 21 | THE LIVING END (corpus) | THE FRQUWOUS |
| 22 | HANGER (corpus) | HALLINGHOL DT |
| 23 | THE OMEN (corpus) | SPARROH S |
| 24 | HANGER (corpus) | HELL BOF LASEEP |
| 25 | DARK CR COUBE | D BIBBS BBANIZK |
| 26 | ONSLAUGHT (corpus) | MONSTER CONDO (corpus) |
| 27 | THE TERRALERP | SPAWNING VATS (corpus) |
| 28 | TOXIN REFINERY (corpus) | THE BLEEDING (corpus) |
| 29 | BLOOD KE LARO | BIRVAFIDEER |

## The exponential in thr_a

Learned c = 1.515, sigma = 34.25. The R=1 model has thr_a = 2 23 36 45 51 57 62.

| k | c + sigma*ln(k+1) | thr_a[k] |
|---|---|---|
| 0 | 1.51 | 2 |
| 1 | 25.26 | 25 |
| 2 | 39.15 | 39 |
| 3 | 49.00 | 49 |
| 4 | 56.64 | 57 |
| 5 | 62.89 | 63 |
| 6 | 68.17 | 68 |

Over scores -50..399 the staircase equals `min(7, floor(exp((score - c)/sigma)))`, except at scores 25, 39 and 68. There
a threshold rounded down moves the step one unit earlier. Every positive score of 2 or more gets weight at least 1.

Counts over all causal (query, key) pairs of the training names (den 1..65, mean 12.6):

| e | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| pairs | 9395 | 3035 | 1987 | 1076 | 674 | 484 | 290 | 2539 |
