# induction_onemap

The induction task at the one size that fits one DOOM map: L=2, d=8, 2 heads of 4, dm=16, K=KA=KM=2, relu attention,
context 7 (BOS, 5 distinct prefix tokens, 1 repeated token; 2,096,640 possible sequences), 17 tokens. See
[`models/induction`](../induction/EVAL.md) for the task and the held-out sets (here at context 7: `test` 4,000
sequences, `unseen` 2,000, 4,000 and 2,000 queried tokens). `df.plan`: 8,421 dolls, 244,890 sectors, span 31,986 x
12,056, 5,704 estimated settle tics in one map; 1,496 dolls and 22,839 sectors in 38 tile maps.

Trained with `python training/train_algo.py --out models/induction_onemap --T 7 --d 8 --H 2 --dm 16 --K 2 --bs 128
--steps1 4000 --lr1 0.02 --steps2 2000 --steps3 3000 --threads 1` (seed 0, 441 s on a loaded machine).

`python training/eval_algo.py models/induction_onemap --T 7 --float models/induction_onemap/float_d8_t7.json`:

| model (context 7) | test | first | unseen | first |
|---|---|---|---|---|
| bigram table | 0.0693 | 0.0693 | 0.0000 | 0.0000 |
| 4-gram table | 0.3708 | 0.3708 | 0.0000 | 0.0000 |
| copy the token 4 back (best fixed lag) | 0.2610 | 0.2610 | 0.2540 | 0.2540 |
| float GPT, d=8, softmax attention (1,561 parameters) | 0.9995 | 0.9995 | 0.9985 | 0.9985 |
| this model, smooth-staircase stage (unrounded weights) | 0.9400 | 0.9400 | 0.8885 | 0.8885 |
| **this model, integer** | **0.9580** | 0.9580 | **0.8590** | 0.8590 |

With 2 levels per sign (5 values per unit) and 8 dimensions the 16 tokens do not separate cleanly: the integer model is
4 points below float on `test` and 14 on `unseen`; it beats every table. (Short sequences make the 4-gram table
stronger here: with a 5-token prefix its window covers almost the whole sequence.)

Exactness: torch int64 forward equals `intref.forward` on 2,000 held-out sequences (0 differ); `intref.check_limits`
clean on 600. DOOM, one map (`python -m demos.induction --model models/induction_onemap --mode map --heldout 1`): the first held-out sequence (`$ E A K H N H`, 6 tokens) in the single compiled map of 8,421 dolls and 244,890 sectors: EXACT, 2,247 values compared at tolerance 0, DOOM answers H -> N as the rule does, 332 s wall on a shared machine (compile 10 s); output in [doom_onemap.log](doom_onemap.log). One sequence only.
