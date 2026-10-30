# induction_onemap

The induction task at the one size that fits one DOOM map. L=2, d=8, 2 heads of 4, dm=16, K=KA=KM=2, relu attention,
17 tokens. Context is 7: BOS, 5 distinct prefix tokens, 1 repeated token. The task and held-out sets are those of
[`models/induction`](../induction/EVAL.md), at context 7.

| plan (`df.plan`) | dolls | sectors | maps |
|---|---|---|---|
| one map, span 31,986 x 12,056, 5,704 estimated settle tics | 8,421 | 244,890 | 1 |
| tiles | 1,496 | 22,839 | 38 |

```sh
python training/train_algo.py --out models/induction_onemap --T 7 --d 8 --H 2 --dm 16 --K 2 --bs 128 \
  --steps1 4000 --lr1 0.02 --steps2 2000 --steps3 3000 --threads 1
```

Seed 0.

## Results

`python training/eval_algo.py models/induction_onemap --T 7 --float models/induction_onemap/float_d8_t7.json`:

| model (context 7) | test | first | unseen | first |
|---|---|---|---|---|
| bigram table | 0.0693 | 0.0693 | 0.0000 | 0.0000 |
| 4-gram table | 0.3708 | 0.3708 | 0.0000 | 0.0000 |
| copy the token 4 back (best fixed lag) | 0.2610 | 0.2610 | 0.2540 | 0.2540 |
| float GPT, d=8, softmax attention (1,561 parameters) | 0.9995 | 0.9995 | 0.9985 | 0.9985 |
| this model, smooth-staircase stage (unrounded weights) | 0.9400 | 0.9400 | 0.8885 | 0.8885 |
| **this model, integer** | **0.9580** | 0.9580 | **0.8590** | 0.8590 |

The integer model beats every table and trails float by 4 points on `test` and 14 on `unseen`. With 5 values per unit
and 8 dimensions the 16 tokens do not separate cleanly. The 4-gram table is strong here because its window covers
almost the whole sequence.

## Exactness

* Torch int64 forward equals `intref.forward` on 2,000 held-out sequences. `intref.check_limits` is clean on 600.
* DOOM, one map: `python -m demos.induction --model models/induction_onemap --mode map --heldout 1` runs the first
  held-out sequence, `$ E A K H N H`. All 2,247 values match at tolerance 0, and DOOM answers H -> N as the rule does.
  Output: [doom_onemap.log](doom_onemap.log).

## Limits

* One seed, and one sequence in DOOM.
