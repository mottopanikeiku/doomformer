# models/quant/transformer: a char-level GPT through `Transformer.from_torch`

What this measures: a small torch GPT (`doomformer.torch_models.GPT`) is trained with the staircase recipe
(`torch_models.fit`), lowered with `df.Transformer.from_torch(net, vocab)` to a `QModel`, and compared on held-out text with
a plain float GPT of the same shape and with n-gram models; held-out names then run through DOOM in tile mode.
Code: [`doomformer/torch_models.py`](../../doomformer/torch_models.py), [`training/quant_transformer.py`](../../training/quant_transformer.py);
docs: [`docs/interop.md`](../../docs/interop.md) (Transformers); tests: [`tests/test_torch_models.py`](../../tests/test_torch_models.py).

## Commands

```
python training/quant_transformer.py train --name relu_none --steps1 2000 --steps2 2500 --steps3 2000 --float-steps 4000 --threads 4
python training/quant_transformer.py train --name soft_rms_reglu --attn softmax --norm rms --Rn 4 --mlp reglu --steps1 1000 --steps2 1500 --steps3 1000 --float-steps 0
python training/quant_transformer.py report --name relu_none          # report.json: perplexities, n-grams, exactness check
python training/quant_transformer.py doom --name relu_none --n 3 --workers 3     # doom.json
```

Training took 2,895 s (`relu_none`, 6,500 steps, 4 threads, `nice`, a shared machine) and 2,082 s (`soft_rms_reglu`, 3,500 steps).
Seed 0, one run each: no seed spread was measured.

## Data

`training/data/levels.json`, field `clean`: 12,934 distinct level names of the DOOM family (upper-case letters, space,
apostrophe, `!`), the fixed split of `training/train_brain.py:load_corpus`: 11,341 train, 300 dev (checkpoint selection),
1,293 val (reporting only). T = 24, vocabulary 30. Every number is bits per character over the characters of a name plus its end
marker, teacher forced, on the 1,293 val names (18,588 predictions); perplexity = 2^bits.

## Models

| name | architecture | parameters | steps |
|---|---|---|---|
| `relu_none` | L=2, d=32, H=4, dm=96, K=KA=KM=7, relu attention, no norm, relu MLP | 24,280 | 2,000 + 2,500 + 2,000 |
| `soft_rms_reglu` | the same shape, softmax attention (R=1), integer RMSNorm (Rn=4), gated MLP | 30,284 | 1,000 + 1,500 + 1,000 |
| plain float GPT | `torch_models.FloatGPT`: L=2, d=32, H=4, dm=96, RMSNorm, softmax attention, ReLU MLP, no staircases | 4,000 AdamW steps, best dev checkpoint |

Stages of `fit`: float stage = smooth staircases with unrounded weights (a float model of the same graph, not a plain
transformer); act-QAT = hard staircases; integer = everything rounded (this is the lowered model).

## Held-out perplexity

| model / stage | bits per char | perplexity |
|---|---|---|
| unigram | 4.364 | 20.6 |
| bigram | 3.730 | 13.3 |
| 4-gram | 2.731 | 6.6 |
| 5-gram | 2.546 | 5.84 |
| 6-gram | 2.548 | 5.85 |
| `relu_none` float stage (smooth staircases) | 2.875 | 7.34 |
| `relu_none` act-QAT | 2.964 | 7.80 |
| **`relu_none` integer (the lowered QModel)** | **2.954** | **7.75** |
| plain float GPT, same shape | 2.915 | 7.54 |
| `soft_rms_reglu` float stage | 3.174 | 9.02 |
| `soft_rms_reglu` integer | 3.214 | 9.28 |

n-grams: interpolated Witten-Bell over characters of `<BOS> name`, order n conditions on the n-1 previous symbols (BOS
included), fitted on the 11,341 training names only (the same data the GPTs train on; dev is not used by the n-grams).

* **Gap, integer against float:** `relu_none` +5.6 % perplexity against its own float stage and +2.7 % against the
  plain float GPT; `soft_rms_reglu` +2.8 % against its float stage. All inside the 10 % target.
* **The model does not beat its 5-gram.** The best integer model has perplexity 7.75 against 5.84 (+33 %), and its
  top-1 next-char accuracy on val is 39.0 %. The plain float GPT of the same shape is also behind (7.54), so the shortfall is capacity and
  data, not the quantisation.
* The integer model is the lowered `QModel`: `intref.forward` on 100 val names equals the torch integer forward bit for bit (100 of 100,
  both models; `report.json`: `qmodel_equals_torch_integer_forward`). Integer logits are multiplied by one learned float
  (`int_logit_scale` in `info.json`, 0.00311 for `relu_none`) to give the probabilities above.
* `soft_rms_reglu` is worse in absolute terms: fewer steps (3,500 against 6,500) and a harder model to train; it is
  included to show that the variants lower the same way, not as a comparison of the variants.

## Summary of the comparison with n-gram baselines

* **Level names** (above): the integer model (7.75) does not beat its 5-gram (5.84), and the plain float GPT of the same
  shape (7.54, 4,000 AdamW steps against 6,500 recipe steps) does not either. Integer against the float stage +5.6 %.
* **Second corpus, my own small run** (`pydoc_relu_none`; one seed; 434 s training; `--corpus pydoc`): the reference
  prose of Python's `pydoc_data.topics`, lower-cased, alphabet `$ a-z . , ( )`, windows of 23 characters (the model reads
  BOS + 23 and predicts 23). Topics split 69 / 4 / 8 (train / dev / val) at random (seed 1234); train windows have stride 6
  (52,676); dev and val windows are kept only if their 23 characters occur nowhere in the train text, so nothing is scored
  on text any model or n-gram saw (158 dev, 1,500 val windows, 34,500 predictions). Same architecture as `relu_none`
  (24,410 parameters), 500 + 700 + 500 recipe steps, plain float GPT 1,700 steps (the same number of optimiser steps).

| model / stage (pydoc, novel val windows) | bits per char | perplexity |
|---|---|---|
| 2-gram / 4-gram | 3.501 / 2.342 | 11.3 / 5.07 |
| 5-gram / 6-gram / 8-gram | 2.279 / 2.411 / 2.773 | 4.85 / 5.32 / 6.84 |
| GPT float stage | 2.617 | 6.13 |
| GPT integer (lowered) | 2.705 | 6.52 |
| plain float GPT, 1,700 steps | 2.477 | 5.57 |

  Here too the integer model does **not** beat the best n-gram (6.52 against 4.85, +34 %). Integer against its float stage
  +6.3 %; against the plain float GPT at the same step count +17.2 % (so the gap to a plain float model is above the 10 %
  target at equal budget; on the level names it was +2.7 % against a plain float GPT that overfit). The lowered `QModel`
  equals the torch integer forward on 100 of 100 val windows.
* **Not reconciled:** an independent run on a larger training budget and its own split of the same text reported the integer
  GPT ahead of the best n-gram by 20 to 30 % on novel windows (perplexity 6.31 against 7.85) and +30 % behind a plain float GPT
  trained 3,000 steps (+18 % at 1,800 steps). I did not reproduce that: my split, window and budget differ (its 5-gram
  perplexity on novel windows is 7.85, mine 4.85, so the splits are not comparable). Whether a model beats its n-gram here
  depends on the split and the training budget; the two runs I made do not.

## DOOM

`relu_none` (2 layers, d=32, 4 heads, relu attention; `df.plan(mode="tiles")`: 56,390 dolls, 803,146 sectors, 106 tile maps, compiled
in 3 to 11 s) on three held-out names of at most 6 characters (the first three of the val split: HUNSKI, FANBOY, SMOL),
`df.compile(qm, mode="tiles", workers=3)`, `brain.logits(tokens)`: every tile run is compared with the NumPy integer reference
at tolerance 0 (`MismatchError` otherwise) and the logits equal `intref.forward`.

| name | tokens | values compared | tile runs | wall | result |
|---|---|---|---|---|---|
| HUNSKI | 7 | 9,170 | 742 | 102 s | EXACT |
| FANBOY | 7 | 9,170 | 742 | 101 s | EXACT |
| SMOL | 5 | 6,470 | 490 | 66 s | EXACT |

3 of 3 exact, 271 s wall in all, three games at once on a shared machine (load average 6 to 10, so the times say nothing
about speed). `doom.json` holds the same rows (its `tile_runs` column is cumulative). A first attempt with the first three
names of the split (16, 7 and 21 characters) was stopped after 12 minutes without a result; I did not investigate whether it
was only slow, so no claim is made for long names.

## Limits

* Two corpora, one seed each, small models; the perplexity differences between the variants are not established.
* `from_torch` lowers a `GPT` trained through the staircase recipe. It does not import a transformer trained elsewhere;
  a float GPT cannot be lowered by rounding (`FloatGPT` is refused with a message).
* The DOOM run covers 3 held-out names of `relu_none`, not the whole val set; `soft_rms_reglu` was not run in DOOM.
