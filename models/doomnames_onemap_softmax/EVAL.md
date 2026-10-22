# doomnames_onemap_softmax

The one-map name transformer (1 layer, d=8, 1 head, dm=16, context 6, K=KM=2, the 416 level names of up to 5 letters
in `corpus.json`, the same file and split as `doomnames_onemap`) with softmax attention in place of ReLU attention
(`attn="softmax"`, `R=1` in `config.json`) and KA=4 attention levels.

Attention, per query t, keys u <= t:

```
e_tu  = adc_relu(q_t . k_u, thr_a)             in [0, 4]
den_t = 1 + sum_u e_tu                          (the 1 is a null slot: logit at thr_a[0], value 0)
num_t = sum_u e_tu * v_u
ao_t  = trunc(R * num_t / den_t)                in [-1, 1]   (R*2 - 1 = 1 comparison level per sign)
```

Trained with `training/train_brain.py --attn softmax --R 1 --KA 4 --seed 3` and otherwise the `doomnames_onemap` recipe
(L=1, d=8, dm=16, K=KM=2, S=16, batch 32, steps 600/600/600, lr 0.01/0.005/0.002, 1 thread, no `--recal`); args,
wall time (22 s), c, sigma and thr_a are in `train_info.json`, the log in `train_log.txt`.

Exactness: `training/eval_brain.py models/doomnames_onemap_softmax --corpus models/doomnames_onemap/corpus.json` gives
`intref.check_limits` clean over all 416 names and torch int64 forward == `intref.forward` logits on 416/416. In DOOM,
`doomformer demo --one-map --softmax` and a greedy generation from an empty prompt (DAGE, five passes) were exact in
every exposed value.

## Choosing KA, R and the seed

With K=2, q and k are in [-2, 2], so the attention scores of this model are small integers (-22..24 over the training
names). The learned exponential has c near 1 and sigma of 1 to 3 score units, and rounding often merges thresholds:
many runs end with thr_a such as 1, 2, 2, 2 (e only 0, 1 or 4). Runs, all with the recipe above, scored with
`intref.forward` on the fixed split (dev 251 target tokens, val 228):

| attention | runs (seeds) | dev accuracy, mean | val accuracy, mean |
|---|---|---|---|
| relu, KA=2 (the `doomnames_onemap` recipe, retrained here) | 11 (0..9, 17) | 0.2383 | 0.2253 |
| softmax KA=4 R=1 | 11 (0..9, 17) | 0.2514 | 0.2313 |
| softmax KA=4 R=2 | 11 (0..9, 17) | 0.2409 | 0.2348 |
| softmax KA=2, 3, 5, 7, R=1 and R=2 | 3 each (1, 2, 17) | 0.2377 to 0.2550 | 0.2281 to 0.2558 |

Selection used dev only: KA=4, runs whose four thresholds are distinct so that e takes every value 0..4 (R=1 seeds 3
and 7, R=2 seeds 3, 4, 7 and 17), highest dev accuracy (62/251 for R=1 seed 3, R=2 seed 3 and R=2 seed 17), ties
broken by dev cross-entropy (2.492, 2.584, 2.544). The highest dev accuracy of any KA=4 run, 72/251 (R=1 seed 6), has
thr_a = 1, 2, 2, 2 and was not kept. Retraining `doomnames_onemap` from its recorded args on this machine does not
reproduce its shipped weights (the unmodified and the modified `train_brain.py` give identical weights to each other),
so the table's relu row is the retrained runs and the comparison below is with the shipped model.

## Results

Same code (`training/eval_brain.py`, corpus `models/doomnames_onemap/corpus.json`) for both models. Accuracy is integer
teacher-forced next-character accuracy.

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

Softmax is 7 tokens worse on train and 2 tokens worse on dev, equal on val. Both models are weak (about one character
in four); the run-to-run spread over seeds (dev standard deviation 0.013 to 0.017 in the 11-seed rows above) is larger
than the difference.

Greedy names, softmax model: DAGE, A ANAAE, B BOR, C CAGE, D DAGE, E EDANA, F FINES, G GAGU, H HIDRE, I IB, J JAGEL,
K KERAE, L LORES, M MAGE, N NAGE, O OREAS, P PAGE, Q QUBAL, R RAGE, S SNAAE, T TAIS, U UNIST, V VINES, W WAGE, X XERER,
Y YAGL, Z ZOR.

## The exponential in thr_a

Learned c = 1.225, sigma = 2.608.

| k | c + sigma*ln(k+1) | thr_a[k] |
|---|---|---|
| 0 | 1.23 | 1 |
| 1 | 3.03 | 3 |
| 2 | 4.09 | 4 |
| 3 | 4.84 | 5 |

Over scores -30..39 the staircase equals `min(4, floor(exp((score - c)/sigma)))` (0 below c) except at scores 1, 3
and 4, where rounding a threshold down moves the step one score unit earlier. Over all causal (query, key) pairs of
the training names, e = 0..4 occurs 3515, 474, 142, 322, 1319 times; den ranges 1..18, mean 4.95.
