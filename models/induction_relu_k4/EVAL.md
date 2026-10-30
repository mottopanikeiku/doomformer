# induction_relu_k4

A variant of [`models/induction`](../induction/EVAL.md): L=2, 2 heads, context 10, relu attention, K=KA=KM=4, d=16, dm=32. The task, held-out sets
and baselines are in that file.

```sh
python training/train_algo.py --out models/induction_relu_k4 --K 4 --d 16 --dm 32 --bs 128 \
  --steps1 4000 --lr1 0.02 --steps2 1000 --steps3 1500 --threads 2 --seed 1
```

Seed 1. `train_info.json` holds args and thresholds. The log is `train_log.txt`.

## Results

| held-out set | integer accuracy | first queried token | float GPT, d=16 |
|---|---|---|---|
| `test` | 0.9986 | 0.9982 | 0.9991 |
| `unseen` | 0.9944 | 0.9945 | 0.9981 |

The same recipe with seed 0 stays at 0.6775 on `test` and 0.3757 on `unseen` (`train_log_seed0.txt`,
`train_info_seed0.json`). Its float stage sits on a 27% plateau and full QAT never recovers. The shipped weights are
seed 1.

## Exactness

* Torch int64 forward equals `intref.forward` on 2,000 held-out sequences.
* `intref.check_limits` is clean on all 6,000 held-out sequences (`python training/eval_algo.py models/induction_relu_k4 --limits`).
