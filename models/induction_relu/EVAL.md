# induction_relu

Variant of [`models/induction`](../induction/EVAL.md) (relu attention, K=7, d=16). L=2, d=16, 2 heads, dm=32, K=KA=KM=7, relu attention (`thr_o`), context 10. Task, held-out sets and the full
table with the count-table and float baselines: see that file.

Trained with `python training/train_algo.py --out models/induction_relu --d 16 --dm 32 --bs 128 --steps1 3000 --lr1 0.02 --steps2 1000 --steps3 2000 --threads 2` (seed 0, 1711 s on a loaded machine; args, thresholds and log in
`train_info.json` and `train_log.txt`).

| held-out set | integer accuracy | first queried token | float GPT, d=16 |
|---|---|---|---|
| `test` (4,000 random sequences) | 0.9993 | 0.9990 | 0.9991 |
| `unseen` (2,000 sequences, queried token pairs never in training) | 0.9986 | 0.9985 | 0.9981 |

Bigram table: 0.0688 and 0.0000, 4-gram table: 0.2331 and 0.0000 (same sets).

Exactness: torch int64 forward equals `intref.forward` on 2000 held-out sequences (0 differ);
`intref.check_limits`: clean (600 sequences in `train_info.json`; all 6,000 held-out sequences with
`python training/eval_algo.py models/induction_relu --limits`).

Circuit (`python training/analyze_induction.py models/induction_relu`): L1 heads put 0.927 and 0.908 of their weight on the token after the first occurrence; L0.h1 is the previous-token head (zeroing it: 0.166); both L0 heads zeroed 0.120, both L1 heads 0.087.
