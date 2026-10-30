# induction

An integer transformer that learned an algorithm. It has 2 layers, d=16, 2 heads of 8, dm=32, K=KA=KM=7 and softmax
attention (`attn="softmax"`, R=2). Context is 10 over 17 tokens: BOS and the letters A to P. It holds 5,105 weights and
biases and 105 threshold integers. `df.plan` gives 6,081 voodoo dolls and 191,566 sectors in 52 tile maps. It does not
fit one map.

## The task

A sequence is BOS, a prefix of 5 to 8 distinct random tokens, then a repeat of a run of 1 or 2 prefix tokens
(`$ KFCAPJD CA`). At every repeated token the answer is the token that followed its first occurrence: C -> A, A -> P.
Given the current token, the answer is uniform over the other prefix tokens. No count table over tokens can learn it.
The generator and the exact rule are in [`doomformer/induction.py`](../../doomformer/induction.py).

Training draws fresh random sequences every step, batches of 128. Three held-out sets are fixed by seed, and the
generator drops any training row with the same hash.

| set | sequences | queried tokens | use |
|---|---|---|---|
| `test` | 4,000 | 5,525 | random sequences |
| `unseen` | 2,000 | 2,124 | every queried (token, answer) pair never stands adjacent in training |
| `dev` | 2,000 | | picks the best checkpoint |

A model that stored token transitions has seen none of the `unseen` pairs.

## Training

```sh
python training/train_algo.py --out models/induction --attn softmax --R 2 --d 16 --dm 32 --bs 128 \
  --steps1 1000 --steps2 1000 --steps3 2000 --threads 2
```

Seed 0. The recipe is that of the name models: smooth-staircase stage, activation QAT, full QAT with rounded weights
and thresholds, best dev checkpoint. `train_info.json` holds args, learned exp parameters and thresholds. The log is
`train_log.txt`. The float baselines are `float_d16.json` and `float_d32.json` (`train_algo.py --float`, AdamW, 5,000
steps).

## Results

```sh
python training/eval_algo.py models/induction models/induction_relu models/induction_k4 models/induction_relu_k4 \
  models/induction_d32 --float models/induction/float_d16.json --float models/induction/float_d32.json
```

Accuracy is over all queried tokens of the set. "first" is the first queried token of each sequence, the pure
look-back. Tables and heuristics are fit on 400,000 training sequences. One training seed per row. A row carries about
0.1 point of noise.

| model | test | first | unseen | first |
|---|---|---|---|---|
| unigram | 0.0601 | 0.0575 | 0.0000 | 0.0000 |
| bigram table | 0.0688 | 0.0680 | 0.0000 | 0.0000 |
| trigram table (backs off) | 0.0695 | 0.0828 | 0.0000 | 0.0000 |
| 4-gram table (backs off) | 0.2331 | 0.2490 | 0.0000 | 0.0000 |
| copy the token 5 back (best fixed lag) | 0.2235 | 0.2082 | 0.2034 | 0.1925 |
| guess a token already seen (expected value) | 0.1946 | 0.1901 | 0.1830 | 0.1813 |
| the rule (look back), the labels | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| float GPT, d=16, softmax attention, LayerNorm, GELU (5,201 parameters) | 0.9991 | 0.9992 | 0.9981 | 0.9980 |
| float GPT, d=32 (18,577 parameters) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **induction**: integer, K=7, softmax attention R=2, d=16 (this model) | **0.9995** | 0.9992 | **0.9995** | 0.9995 |
| &nbsp;&nbsp;its smooth-staircase stage (unrounded weights, the "float" stage of the recipe) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| induction_relu: integer, K=7, relu attention, d=16 | 0.9993 | 0.9990 | 0.9986 | 0.9985 |
| induction_k4: integer, K=4, softmax attention R=2, d=16 | 0.9984 | 0.9988 | 0.9929 | 0.9940 |
| induction_relu_k4: integer, K=4, relu attention, d=16, seed 1 | 0.9986 | 0.9982 | 0.9944 | 0.9945 |
| &nbsp;&nbsp;the same recipe, seed 0 | 0.6775 | 0.6715 | 0.3757 | 0.3745 |
| induction_d32: integer, K=7, softmax attention R=2, d=32 | 0.9998 | 0.9998 | 0.9995 | 0.9995 |

* Count tables sit near chance, 1/16. The 4-gram table is higher because its window covers much of a short sequence.
* On `unseen` every table scores 0 by construction. It never counted the pair it needs.
* Every integer model that learned the task is within 0.1 point of the float GPT on `test` and 0.6 point on `unseen`.
* Relu attention at K=4 learned from seed 1 and failed from seed 0 (`induction_relu_k4/train_log_seed0.txt`).

## What it learned

`python training/analyze_induction.py models/induction` measures attention on 1,375 queried tokens of 1,000 held-out
sequences. Take a queried token t that repeats the token first seen at j. The table gives each head's share of
attention on t-1, on j+1 (the answer) and on BOS. The last column ablates one head's 8 units of `L{l}.ao` with intref
`ablate` and measures accuracy on 500 sequences.

| head | on t-1 | on j+1 | on BOS | on t-1, all t | accuracy with it zeroed |
|---|---|---|---|---|---|
| L0.h0 | 0.592 | 0.089 | 0.032 | 0.811 | 0.665 |
| L0.h1 | 0.529 | 0.082 | 0.047 | 0.746 | 0.941 |
| L1.h0 | 0.085 | **0.935** | 0.001 | 0.057 | 0.991 |
| L1.h1 | 0.083 | **0.942** | 0.003 | 0.049 | 0.986 |

| zeroed | accuracy |
|---|---|
| nothing | 1.000 |
| both layer-0 heads | 0.106 |
| both layer-1 heads | 0.063 |

This is the induction circuit. The layer-0 heads copy the previous token into the next position, so position j+1
carries the token at j. The layer-1 heads match the current token against that and read the token at j+1. The two heads
of each layer are partly redundant. The variants show the same structure (their EVAL.md files). Causal tests by
patching in DOOM: [docs/interventions.md](../../docs/interventions.md#the-induction-circuit).

## Exactness

* Torch int64 forward equals `intref.forward` on all logits of 2,000 held-out sequences, for every model here
  (`train_info.json`, `held_out.torch_vs_intref`).
* `intref.check_limits` is clean on all 6,000 held-out sequences (`eval_algo.py ... --limits`) for `induction`,
  `induction_relu` and `induction_k4`, and on 600 for the others.
* DOOM, tile maps: the first 25 `test` sequences, every exposed value against `intref.forward` at tolerance 0.

| sequences | exact | tile maps | tile runs | values checked | queried tokens equal to the rule | wall, 3 games, shared machine |
|---|---|---|---|---|---|---|
| 25 | 25/25 | 52 | 13,868 | 520,749 | 38/38 | 2,545 s |

The values cover every pre-activation, score, den, P/N comparison, logit and negation twin of every position. Python
copies the integers between games. The BOS position runs once and is re-checked with every sequence. Output:
[doom25_tiles.log](doom25_tiles.log). The one-map run is [`induction_onemap`](../induction_onemap/EVAL.md).

`python -m demos.induction --heldout 25 --workers 3` repeats the tile run. `python -m demos.induction KFCAPJDC` runs
your own sequence.

## Limits

* One seed per row.
* A three-line rule solves the task exactly. The model matches the rule and does not exceed it.
* Sequences are at most 10 tokens over 16 tokens. Nothing is claimed for longer contexts or other vocabularies.
* The DOOM runs cover the listed sequences only.
