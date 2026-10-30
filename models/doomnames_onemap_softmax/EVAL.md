# doomnames_onemap_softmax

The one-map name transformer with softmax attention (`attn="softmax"`, `R=1`) and KA=4 attention levels. Shape:
1 layer, d=8, 1 head, dm=16, context 6, K=KM=2. Data and split: those of `doomnames_onemap`, 416 names.

Attention, per query t, keys u <= t:

```
e_tu  = adc_relu(q_t . k_u, thr_a)             in [0, 4]
den_t = 1 + sum_u e_tu                          (the 1 is a null slot: logit at thr_a[0], value 0)
num_t = sum_u e_tu * v_u
ao_t  = trunc(R * num_t / den_t)                in [-1, 1]   (R*2 - 1 = 1 comparison level per sign)
```

```sh
python training/train_brain.py --attn softmax --R 1 --KA 4 --seed 3   # otherwise the doomnames_onemap recipe
```

`train_info.json` holds args, c, sigma and thr_a. The log is `train_log.txt`.

## Exactness

* `training/eval_brain.py`: `intref.check_limits` is clean on all 416 names, and torch int64 forward equals
  `intref.forward` on 416/416.
* DOOM: `doomformer demo --one-map --softmax` and a greedy generation (DAGE, five passes) are exact in every value.

## Choosing KA, R and the seed

With K=2, attention scores are small integers (-22..24). Rounding often merges thresholds, for example thr_a = 1, 2, 2,
2, where e takes only 0, 1 or 4. Runs are scored with `intref.forward` on dev (251 tokens) and val (228).

| attention | runs (seeds) | dev accuracy, mean | val accuracy, mean |
|---|---|---|---|
| relu, KA=2 (the `doomnames_onemap` recipe, retrained here) | 11 (0..9, 17) | 0.2383 | 0.2253 |
| softmax KA=4 R=1 | 11 (0..9, 17) | 0.2514 | 0.2313 |
| softmax KA=4 R=2 | 11 (0..9, 17) | 0.2409 | 0.2348 |
| softmax KA=2, 3, 5, 7, R=1 and R=2 | 3 each (1, 2, 17) | 0.2377 to 0.2550 | 0.2281 to 0.2558 |

Selection uses dev only: KA=4, four distinct thresholds, highest dev accuracy, then lowest dev cross-entropy. The relu
row is retrained runs, which do not reproduce the shipped `doomnames_onemap` weights. The comparison below uses the
shipped model.

## Results

Integer teacher-forced next-character accuracy, `training/eval_brain.py`:

| | doomnames_onemap (relu, shipped) | doomnames_onemap_softmax (KA=4, R=1) |
|---|---|---|
| train accuracy (328 names) | 518/1772 = 0.2923 | 511/1772 = 0.2884 |
| dev accuracy (46 names) | 64/251 = 0.2550 | 62/251 = 0.2470 |
| val accuracy (42 names) | 55/228 = 0.2412 | 55/228 = 0.2412 |
| val NLL at temperature 50 (nats/char) | 2.874 | 2.873 |
| float dev accuracy (train_brain float stage) | 0.2390 | 0.2550 |
| greedy, empty prompt | DAN | DAGE |
| greedy, 27 prompts (empty + A..Z) in corpus | 2/27 (LUST, RUST) | 2/27 (CAGE, RAGE) |
| map dolls / sectors (`df.plan`) | 3,548 / 95,448 | 4,209 / 123,066 |
| game tics per one-map pass (measured) | 3,488 | 4,144 |
| wall time per letter, one core (measured) | 83 s | 116 to 128 s |
| training wall time | 20 s | 22 s |

The difference is inside the seed spread (dev standard deviation 0.013 to 0.017). Both models get about one character
in four.

Greedy names of the softmax model, empty prompt then A..Z:

```text
DAGE, A ANAAE, B BOR, C CAGE, D DAGE, E EDANA, F FINES, G GAGU, H HIDRE, I IB, J JAGEL, K KERAE, L LORES, M MAGE,
N NAGE, O OREAS, P PAGE, Q QUBAL, R RAGE, S SNAAE, T TAIS, U UNIST, V VINES, W WAGE, X XERER, Y YAGL, Z ZOR
```

## The exponential in thr_a

Learned c = 1.225, sigma = 2.608.

| k | c + sigma*ln(k+1) | thr_a[k] |
|---|---|---|
| 0 | 1.23 | 1 |
| 1 | 3.03 | 3 |
| 2 | 4.09 | 4 |
| 3 | 4.84 | 5 |

Over scores -30..39 the staircase equals `min(4, floor(exp((score - c)/sigma)))`, except at scores 1, 3 and 4. There a
threshold rounded down moves the step one unit earlier.

Counts over all causal (query, key) pairs of the training names (den ranges 1..18, mean 4.95):

| e | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| pairs | 3515 | 474 | 142 | 322 | 1319 |
