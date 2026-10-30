# induction_k4

A variant of [`models/induction`](../induction/EVAL.md): L=2, 2 heads, context 10, softmax attention R=2, K=KA=KM=4, d=16, dm=32. The task, held-out sets
and baselines are in that file.

```sh
python training/train_algo.py --out models/induction_k4 --attn softmax --R 2 --K 4 --d 16 --dm 32 --bs 128 \
  --steps1 1000 --steps2 1000 --steps3 2000 --threads 2
```

Seed 0. `train_info.json` holds args and thresholds. The log is `train_log.txt`.

## Results

| held-out set | integer accuracy | first queried token | float GPT, d=16 |
|---|---|---|---|
| `test` | 0.9984 | 0.9988 | 0.9991 |
| `unseen` | 0.9929 | 0.9940 | 0.9981 |

## Exactness

* Torch int64 forward equals `intref.forward` on 2,000 held-out sequences.
* `intref.check_limits` is clean on all 6,000 held-out sequences (`python training/eval_algo.py models/induction_k4 --limits`).

## Circuit

`python training/analyze_induction.py models/induction_k4`:

| measure | value |
|---|---|
| L1 heads on the token after the first occurrence | 0.895, 0.905 |
| L0.h1 zeroed (previous-token head) | 0.421 |
| both L0 heads zeroed | 0.118 |
| both L1 heads zeroed | 0.073 |
