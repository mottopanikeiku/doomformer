# induction

An integer transformer that learned an algorithm, not a table: 2 layers, d=16, 2 heads of 8, dm=32, K=KA=KM=7, softmax
attention (`attn="softmax"`, R=2), context 10, 17 tokens (BOS and the letters A to P). 5,105 weights and biases, 105
threshold integers. `df.plan`: 6,081 voodoo dolls and 191,566 sectors in 52 tile maps; it does not fit one map.

## The task

A sequence is BOS, a prefix of 5 to 8 distinct random tokens, then a repeat of a run of 1 or 2 tokens of the prefix
(`$ KFCAPJD CA`). At every repeated token the answer is the token that followed its first occurrence (here C -> A, A ->
P). The prefix is drawn without replacement from 16 tokens, so given the current token the answer is uniform over the
other prefix tokens: no count table over tokens can learn it, and the answer exists only in the sequence itself. The
task, its generator and its exact rule are in [`doomformer/induction.py`](../../doomformer/induction.py).

Training draws fresh random sequences every step (batches of 128; 4.3 billion different sequences exist). Three held-out sets are fixed by seed and never drawn in training (the generator drops any row with
the same hash): `test` (4,000 random sequences, 5,525 queried tokens), `unseen` (2,000 sequences, 2,124 queried tokens)
and `dev` (2,000, picks the best checkpoint). In `unseen` every queried token pair (current token, answer) is one of 24
ordered token pairs that never stand next to each other in any training sequence, so a model that stored token
transitions has never seen any of them.

Trained with `python training/train_algo.py --out models/induction --attn softmax --R 2 --d 16 --dm 32 --bs 128
--steps1 1000 --steps2 1000 --steps3 2000 --threads 2` (seed 0, 1,285 s on a loaded machine; the recipe of the name
models: smooth-staircase stage, activation-QAT, full QAT with rounded weights and thresholds, best dev checkpoint).
Args, learned exp parameters and thresholds are in `train_info.json`, the log in `train_log.txt`, the float baselines
in `float_d16.json` and `float_d32.json` (`train_algo.py --float --d 16 --dm 32`, AdamW, 5,000 steps).

## Results

`python training/eval_algo.py models/induction models/induction_relu models/induction_k4 models/induction_relu_k4
models/induction_d32 --float models/induction/float_d16.json --float models/induction/float_d32.json`. Accuracy over
all queried tokens of the set ("first": at the first queried token of each sequence, the pure look-back). The tables
and heuristics are fit on 400,000 training sequences (their supervised positions, the most favourable data for them).
One training seed per row; 4,000 sequences put about 0.1 point of noise on a row.

| model | test | first | unseen | first |
|---|---|---|---|---|
| unigram | 0.0601 | 0.0575 | 0.0000 | 0.0000 |
| bigram table | 0.0688 | 0.0680 | 0.0000 | 0.0000 |
| trigram table (backs off) | 0.0695 | 0.0828 | 0.0000 | 0.0000 |
| 4-gram table (backs off) | 0.2331 | 0.2490 | 0.0000 | 0.0000 |
| copy the token 5 back (best fixed lag) | 0.2235 | 0.2082 | 0.2034 | 0.1925 |
| guess a token already seen (expected value) | 0.1946 | 0.1901 | 0.1830 | 0.1813 |
| the rule (look back), the labels | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| float GPT, d=16, softmax attention, LayerNorm, GELU (5,201 parameters) | 0.9991 | 0.9992 | 0.9981 | 0.9980 |
| float GPT, d=32 (18,577 parameters) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **induction**: integer, K=7, softmax attention R=2, d=16 (this model) | **0.9995** | 0.9992 | **0.9995** | 0.9995 |
| &nbsp;&nbsp;its smooth-staircase stage (unrounded weights, the "float" stage of the recipe) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| induction_relu: integer, K=7, relu attention, d=16 | 0.9993 | 0.9990 | 0.9986 | 0.9985 |
| induction_k4: integer, K=4, softmax attention R=2, d=16 | 0.9984 | 0.9988 | 0.9929 | 0.9940 |
| induction_relu_k4: integer, K=4, relu attention, d=16, seed 1 | 0.9986 | 0.9982 | 0.9944 | 0.9945 |
| &nbsp;&nbsp;the same recipe, seed 0 | 0.6775 | 0.6715 | 0.3757 | 0.3745 |
| induction_d32: integer, K=7, softmax attention R=2, d=32 | 0.9998 | 0.9998 | 0.9995 | 0.9995 |

The count tables sit at 6 to 7% (chance is 1/16) except the 4-gram: with 5 to 8 prefix tokens its context window covers
much of a short sequence, so it recognises some of the 4-token windows it saw in training (23%). On `unseen` every
table scores 0 by construction (the pair it would need was never counted; the unigram's single guess is a token that
is never the answer of a held-out pair). The heuristics that look at position or at the set of tokens seen reach 19 to
23%. Integer accuracy is within 0.1 point of the float GPT of the same size on `test` and within 0.6 point on `unseen`
for every model that learned the task (K=7 and K=4, both attention kinds). `induction_relu_k4` shows the recipe's seed risk: relu
attention at K=4 learned the task from seed 1 and stayed at 67.8% (`train_log_seed0.txt` there) from seed 0, whose
float stage never left a 27% plateau; the softmax models and relu K=7 learned from seed 0. Training wall times
(`train_log.txt`) are on a machine with load average 30 to 45: 1,285 s for this model, 1,711 s relu, 3,716 s d=32.

## What it learned

`python training/analyze_induction.py models/induction` (1,375 queried tokens of 1,000 held-out sequences, NumPy on the
integer attention weights e_ts; "zeroed": intref `ablate` of one head's 8 units of `L{l}.ao`, accuracy at the queried
tokens of 500 sequences). Share of a head's attention weight at a queried token t (a repeat of the token first seen at
j) on position t-1, on j+1 (the token that followed the first occurrence, the answer) and on position 0:

| head | on t-1 | on j+1 | on BOS | on t-1, all t | accuracy with it zeroed |
|---|---|---|---|---|---|
| L0.h0 | 0.592 | 0.089 | 0.032 | 0.811 | 0.665 |
| L0.h1 | 0.529 | 0.082 | 0.047 | 0.746 | 0.941 |
| L1.h0 | 0.085 | **0.935** | 0.001 | 0.057 | 0.991 |
| L1.h1 | 0.083 | **0.942** | 0.003 | 0.049 | 0.986 |

Nothing zeroed: 1.000. Both layer-0 heads zeroed: 0.106. Both layer-1 heads zeroed: 0.063 (chance 0.06). This is the
induction circuit: the layer-0 heads copy the previous token into the next position's residual stream, so position
j+1 carries the token at j; the layer-1 heads match the current token against that and read the token at j+1. The two
heads of each layer are partly redundant. The same two-layer structure is in `induction_relu` (L1 heads put 0.927 and
0.908 of their weight on j+1; L0.h1 is the previous-token head, zeroing it gives 0.166), `induction_k4` (0.895 and
0.905; L0.h1 zeroed 0.421) and `induction_d32` (L1.h0 puts 0.976 on j+1 and zeroing it gives 0.397; L0.h1 zeroed
0.195). The causal tests with patching in DOOM are `df.patch_recovery` on this model (see docs/DEMOS.md).

## Exactness

- Torch int64 forward (`train_brain.int_forward` on the exported integers) equals `intref.forward` on all logits of
  2,000 held-out sequences (0 differ), for every model here (`train_info.json`, `held_out.torch_vs_intref`).
- `intref.check_limits` is clean on all 6,000 held-out sequences (`test` and `unseen`) for `induction`,
  `induction_relu` and `induction_k4` (`training/eval_algo.py ... --limits`), and on 600 for the others
  (`train_info.json`).
- DOOM, tile maps (`df.compile(net, mode="tiles")`, 3 games at a time, Python copies the integers between games), the
  first 25 sequences of the held-out `test` set, every exposed value against `intref.forward` at tolerance 0:
  25/25 exact. 52 tile maps (6,081 voodoo dolls, 191,566 sectors in all), 13,868 tile runs, 520,749
  values checked (every pre-activation, score, den, P/N comparison, logit and negation twin of every position), 2,545 s
  wall for the 25 sequences with 3 DOOM games on a machine shared with other jobs (load average 25 to 45), compile
  1 s from cache. DOOM's answer equals the rule at all 38 queried tokens of the 25 sequences. The BOS position is run
  once and reused (its values are re-checked against the reference with every sequence). Output:
  [doom25_tiles.log](doom25_tiles.log).
- DOOM, one map: `induction_onemap` (below).

`python -m demos.induction --heldout 25 --workers 3` repeats the tile run; `python -m demos.induction KFCAPJDC` runs
your own sequence.

## induction_onemap

The shape trained and run in one map is tiny: L=2, d=8, 2 heads of 4, dm=16, K=2, relu attention, context 7 (BOS, 5
prefix tokens, 1 repeated token). `df.plan`: 8,421 dolls, 244,890 sectors, 5,704 estimated settle tics, span 31,986 x
12,056. Shapes planned with untrained weights: d=8, K=3, context 8 fits (15,510 dolls, 561,434 sectors, 9,848 estimated
tics; not trained or run); d=12 at K=3 and context 8 does not fit (span 35,336 on one axis, over 32,000), nor does
d=16. Trained with `python training/train_algo.py --out
models/induction_onemap --T 7 --d 8 --H 2 --dm 16 --K 2 --bs 128 --steps1 4000 --lr1 0.02 --steps2 2000 --steps3 3000
--threads 1`. Held-out at context 7 (`python training/eval_algo.py models/induction_onemap --T 7 --float
models/induction_onemap/float_d8_t7.json`): integer 0.9580 on `test` and 0.8590 on `unseen`, against 0.9995 and 0.9985
for a float GPT of the same shape (1,561 parameters); the 4-gram table scores 0.3708 and 0.0000, the bigram table 0.0693
and 0.0000. Here the integer model is 4 to 14 points below float: with 2 levels per sign (5 values) 16 tokens do not
separate cleanly in 8 dimensions. DOOM, one map (`python -m demos.induction --model models/induction_onemap --mode map --heldout 1`): the first held-out sequence (`$ E A K H N H`, 6 tokens) in the single compiled map of 8,421 dolls and 244,890 sectors: EXACT, 2,247 values compared at tolerance 0, DOOM answers H -> N as the rule does, 332 s wall on a shared machine (compile 10 s); output in [doom_onemap.log](../induction_onemap/doom_onemap.log). One sequence only.

## Limits

One seed per row. The task is exactly solvable by a three-line rule; the model matches the rule and does not exceed it.
Sequences are at most 10 tokens over 16 tokens (at most 7 over 16 for the one-map model); nothing is claimed for longer
contexts or other vocabularies. The DOOM runs cover the sequences listed, with this model's integer weights, and say
nothing about sequences that were not run.
