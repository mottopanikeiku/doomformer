# induction_d32

Variant of [`models/induction`](../induction/EVAL.md) (softmax attention, K=7, d=32). L=2, d=32, 2 heads, dm=64, K=KA=KM=7, softmax attention R=2, context 10 (18,385 weights and biases). Task, held-out sets and the full
table with the count-table and float baselines: see that file.

Trained with `python training/train_algo.py --out models/induction_d32 --attn softmax --R 2 --bs 128 --steps1 1500 --steps2 1500 --steps3 2500 --threads 2` (seed 0, 3716 s on a loaded machine; args, thresholds and log in
`train_info.json` and `train_log.txt`).

| held-out set | integer accuracy | first queried token | float GPT, d=16 |
|---|---|---|---|
| `test` (4,000 random sequences) | 0.9998 | 0.9998 | 1.0000 |
| `unseen` (2,000 sequences, queried token pairs never in training) | 0.9995 | 0.9995 | 1.0000 |

Bigram table: 0.0688 and 0.0000, 4-gram table: 0.2331 and 0.0000 (same sets).

Exactness: torch int64 forward equals `intref.forward` on 2000 held-out sequences (0 differ);
`intref.check_limits`: clean (600 sequences in `train_info.json`; all 6,000 held-out sequences with
`python training/eval_algo.py models/induction_d32 --limits`).

Circuit (`python training/analyze_induction.py models/induction_d32`): L1.h0 puts 0.976 of its weight on the token after the first occurrence and carries the answer (zeroing it: 0.397; L1.h1 zeroed: 0.999); L0.h1 is the previous-token head (zeroed: 0.195); both L0 heads zeroed 0.139, both L1 heads 0.076.
