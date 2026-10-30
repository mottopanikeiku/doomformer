# models/quant/transformer: a char-level GPT through `Transformer.from_torch`

A small torch GPT (`doomformer.torch_models.GPT`) trains with the staircase recipe (`torch_models.fit`).
`df.Transformer.from_torch(net, vocab)` lowers it to a `QModel`. This record compares the lowered model on held-out text
with a plain float GPT of the same shape and with n-gram models. Three held-out names then run through DOOM in tile mode.
Code: [`doomformer/torch_models.py`](../../doomformer/torch_models.py), [`training/quant_transformer.py`](../../training/quant_transformer.py).
Docs: [`docs/interop.md`](../../docs/interop.md). Tests: [`tests/test_torch_models.py`](../../tests/test_torch_models.py).

## Commands

```
python training/quant_transformer.py train --name relu_none --steps1 2000 --steps2 2500 --steps3 2000 --float-steps 4000 --threads 4
python training/quant_transformer.py train --name soft_rms_reglu --attn softmax --norm rms --Rn 4 --mlp reglu --steps1 1000 --steps2 1500 --steps3 1000 --float-steps 0
python training/quant_transformer.py report --name relu_none          # report.json: perplexities, n-grams, exactness check
python training/quant_transformer.py doom --name relu_none --n 3 --workers 3     # doom.json
```

Seed 0, one run per model.

## Data

* Corpus: `training/data/levels.json`, field `clean`, 12,934 distinct DOOM level names. T = 24, vocabulary 30.
* Split: the fixed split of `training/train_brain.py:load_corpus`, 11,341 train, 300 dev, 1,293 val.
* Dev selects the checkpoint. Val is for reporting only.
* Metric: bits per character on the val names plus end marker, teacher forced, 18,588 predictions. Perplexity is 2^bits.

## Models

| name | architecture | parameters | steps |
|---|---|---|---|
| `relu_none` | L=2, d=32, H=4, dm=96, K=KA=KM=7, relu attention, no norm, relu MLP | 24,280 | 2,000 + 2,500 + 2,000 |
| `soft_rms_reglu` | the same shape, softmax attention (R=1), integer RMSNorm (Rn=4), gated MLP | 30,284 | 1,000 + 1,500 + 1,000 |
| plain float GPT | `torch_models.FloatGPT`: L=2, d=32, H=4, dm=96, RMSNorm, softmax attention, ReLU MLP, no staircases | 4,000 AdamW steps, best dev checkpoint |

`fit` has three stages. The float stage uses smooth staircases and unrounded weights. Act-QAT uses hard staircases. The
integer stage rounds everything and is the lowered model.

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

The n-grams are interpolated Witten-Bell over the characters of `<BOS> name`, fitted on the training names only.

* Integer against float: `relu_none` is +5.6 % perplexity against its float stage and +2.7 % against the plain float
  GPT. `soft_rms_reglu` is +2.8 % against its float stage. All are inside the 10 % target.
* The model does not beat its 5-gram. The plain float GPT does not either, so the shortfall is capacity and data.
* `intref.forward` equals the torch integer forward bit for bit on 100 of 100 val names, for both models
  (`report.json`). One learned float, `int_logit_scale` in `info.json`, turns integer logits into probabilities.
* `soft_rms_reglu` has fewer steps. It shows that the variants lower the same way and is not a comparison of variants.

## Second corpus: pydoc

`pydoc_relu_none` (`--corpus pydoc`) uses the same architecture as `relu_none` on the prose of Python's
`pydoc_data.topics`. One seed.

* Text: lower-cased, alphabet `$ a-z . , ( )`, windows of 23 characters.
* Split: topics 69 / 4 / 8 (train / dev / val) at random, seed 1234. Train windows have stride 6.
* Dev and val keep only windows whose 23 characters occur nowhere in the train text: 1,500 val windows.
* Steps: 500 + 700 + 500 recipe steps. The plain float GPT gets 1,700 steps, the same optimiser step count.

| model / stage (pydoc, novel val windows) | bits per char | perplexity |
|---|---|---|
| 2-gram / 4-gram | 3.501 / 2.342 | 11.3 / 5.07 |
| 5-gram / 6-gram / 8-gram | 2.279 / 2.411 / 2.773 | 4.85 / 5.32 / 6.84 |
| GPT float stage | 2.617 | 6.13 |
| GPT integer (lowered) | 2.705 | 6.52 |
| plain float GPT, 1,700 steps | 2.477 | 5.57 |

The integer model does not beat the best n-gram here either. It is +6.3 % against its float stage and +17.2 % against
the plain float GPT at equal steps, above the 10 % target. The lowered `QModel` equals the torch integer forward on 100
of 100 val windows. Whether a model beats its n-gram depends on the split and the training budget.

## DOOM

`doom` compiles `relu_none` with `df.compile(qm, mode="tiles", workers=3)` and runs three held-out names of at most 6
characters. `df.plan(mode="tiles")` gives 56,390 dolls and 803,146 sectors in 106 tile maps. Every tile run is compared
with the NumPy integer reference at tolerance 0, and the logits equal `intref.forward`.

| name | tokens | values compared | tile runs | wall | result |
|---|---|---|---|---|---|
| HUNSKI | 7 | 9,170 | 742 | 102 s | EXACT |
| FANBOY | 7 | 9,170 | 742 | 101 s | EXACT |
| SMOL | 5 | 6,470 | 490 | 66 s | EXACT |

Three games ran at once on a shared machine, so the times say nothing about speed. `doom.json` holds the same rows. Its
`tile_runs` column is cumulative.

## Limits

* Two corpora, one seed each, small models. Perplexity differences between the variants are not established.
* `from_torch` lowers a `GPT` trained through the staircase recipe. It refuses `FloatGPT` and any transformer trained
  elsewhere.
* The DOOM run covers 3 short held-out names of `relu_none`. No claim is made for long names. `soft_rms_reglu` has no
  DOOM run.
