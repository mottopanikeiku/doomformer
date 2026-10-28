# induction_relu_k4

Variant of [`models/induction`](../induction/EVAL.md) (relu attention, K=4, d=16, seed 1). L=2, d=16, 2 heads, dm=32, K=KA=KM=4, relu attention, context 10. Task, held-out sets and the full
table with the count-table and float baselines: see that file.

Trained with `python training/train_algo.py --out models/induction_relu_k4 --K 4 --d 16 --dm 32 --bs 128 --steps1 4000 --lr1 0.02 --steps2 1000 --steps3 1500 --threads 2 --seed 1` (seed 1, 1345 s on a loaded machine; args, thresholds and log in
`train_info.json` and `train_log.txt`).

| held-out set | integer accuracy | first queried token | float GPT, d=16 |
|---|---|---|---|
| `test` (4,000 random sequences) | 0.9986 | 0.9982 | 0.9991 |
| `unseen` (2,000 sequences, queried token pairs never in training) | 0.9944 | 0.9945 | 0.9981 |

Bigram table: 0.0688 and 0.0000, 4-gram table: 0.2331 and 0.0000 (same sets).
The same recipe with seed 0 (`train_log_seed0.txt`, `train_info_seed0.json`) stayed at 0.6775 on `test` and 0.3757 on `unseen`: its float stage sat on a 27% plateau for 3,000 steps and full QAT never recovered. The shipped weights are seed 1.
Exactness: torch int64 forward equals `intref.forward` on 2000 held-out sequences (0 differ);
`intref.check_limits`: clean (600 sequences in `train_info.json`; all 6,000 held-out sequences with
`python training/eval_algo.py models/induction_relu_k4 --limits`).
