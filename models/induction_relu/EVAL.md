# induction_relu

A variant of [`models/induction`](../induction/EVAL.md): L=2, 2 heads, context 10, relu attention (`thr_o`), K=KA=KM=7, d=16, dm=32. The task, held-out sets
and baselines are in that file.

```sh
python training/train_algo.py --out models/induction_relu --d 16 --dm 32 --bs 128 \
  --steps1 3000 --lr1 0.02 --steps2 1000 --steps3 2000 --threads 2
```

Seed 0. `train_info.json` holds args and thresholds. The log is `train_log.txt`.

## Results

| held-out set | integer accuracy | first queried token | float GPT, d=16 |
|---|---|---|---|
| `test` | 0.9993 | 0.9990 | 0.9991 |
| `unseen` | 0.9986 | 0.9985 | 0.9981 |

## Exactness

* Torch int64 forward equals `intref.forward` on 2,000 held-out sequences.
* `intref.check_limits` is clean on all 6,000 held-out sequences (`python training/eval_algo.py models/induction_relu --limits`).

## Circuit

`python training/analyze_induction.py models/induction_relu`:

| measure | value |
|---|---|
| L1 heads on the token after the first occurrence | 0.927, 0.908 |
| L0.h1 zeroed (previous-token head) | 0.166 |
| both L0 heads zeroed | 0.120 |
| both L1 heads zeroed | 0.087 |
