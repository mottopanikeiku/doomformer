# doomnames_modern

The doomnames transformer with every block option: RMSNorm (`Rn=6`), softmax attention (`R=2`) and a gated MLP
(`mlp="reglu"`, dm=21). Shape: 1 layer, d=16, 2 heads, context 16, K=KA=KM=7, 3,432 weights and biases. Data: the 136
DOOM level names in `corpus.json`. The dolls compute the RMS, the exponentials and both divisions. Definitions:
[`intref.py`](../../doomformer/intref.py) and [HOW_IT_WORKS.md](../../docs/HOW_IT_WORKS.md#rmsnorm).

```sh
python training/train_qat.py --attn softmax --R 2 --norm rms --Rn 6 --mlp reglu --dm 21   # doomnames recipe, seed 1
```

Torch int64 forward equals `intref.forward` on 136/136 names. `intref.check_limits` is clean.

## Results

Integer teacher-forced accuracy over the 1,505 target tokens of the 136 training names, `training/eval_model.py`. Same
recipe and seed for every row.

| model | norm | attention | MLP | accuracy |
|---|---|---|---|---|
| doomnames (shipped) | none | relu | relu | 1235/1505 = 0.8206 |
| doomnames_softmax (shipped) | none | softmax R=2 | relu | 1221/1505 = 0.8113 |
| rms, Rn=8 | rms | relu | relu | 0.8100 |
| rms, Rn=3 | rms | relu | relu | 1188/1505 = 0.7894 |
| reglu, dm=21 | none | relu | reglu | 1218/1505 = 0.8093 |
| softmax + reglu | none | softmax R=2 | reglu | 1207/1505 = 0.8020 |
| softmax + rms, Rn=8 | rms | softmax R=2 | relu | 0.8133 |
| modern, Rn=2 | rms | softmax R=2 | reglu | 0.6897 |
| modern, Rn=3 | rms | softmax R=2 | reglu | 0.7608 |
| modern, Rn=4 | rms | softmax R=2 | reglu | 1168/1505 = 0.7761 |
| **modern, Rn=6 (this model)** | rms | softmax R=2 | reglu | **1211/1505 = 0.8047** |
| modern, Rn=8 | rms | softmax R=2 | reglu | 0.8020 |
| modern, Rn=12 | rms | softmax R=2 | reglu | 0.7987 |

No option beats the plain model on this corpus. 136 names is a memorisation task for a 3,400-weight model, and these
options exist for scale. With Rn of 3 or less the normalised values have too few levels and accuracy drops.

Greedy generation from 21 prompts (empty and first letters) writes 17 training names and 4 new ones: ENTRYWENT,
LITFOSTOKH, REALM OF OFIND, UNRULY AN. The default demo settings (temperature 180) for seeds 20 to 29 write 9 training
names and THE SPIRRENGED (seed 27).
