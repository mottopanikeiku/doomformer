# induction_d32

A variant of [`models/induction`](../induction/EVAL.md): L=2, 2 heads, context 10, softmax attention R=2, K=KA=KM=7, d=32, dm=64, 18,385 weights and biases. The task, held-out sets
and baselines are in that file.

```sh
python training/train_algo.py --out models/induction_d32 --attn softmax --R 2 --bs 128 \
  --steps1 1500 --steps2 1500 --steps3 2500 --threads 2
```

Seed 0. `train_info.json` holds args and thresholds. The log is `train_log.txt`.

## Results

| held-out set | integer accuracy | first queried token | float GPT, d=16 |
|---|---|---|---|
| `test` | 0.9998 | 0.9998 | 1.0000 |
| `unseen` | 0.9995 | 0.9995 | 1.0000 |

## Exactness

* Torch int64 forward equals `intref.forward` on 2,000 held-out sequences.
* `intref.check_limits` is clean on all 6,000 held-out sequences (`python training/eval_algo.py models/induction_d32 --limits`).

## Circuit

`python training/analyze_induction.py models/induction_d32`:

| measure | value |
|---|---|
| L1.h0 on the token after the first occurrence | 0.976 |
| L1.h0 zeroed | 0.397 |
| L1.h1 zeroed | 0.999 |
| L0.h1 zeroed (previous-token head) | 0.195 |
| both L0 heads zeroed | 0.139 |
| both L1 heads zeroed | 0.076 |
