# mnist: the network DOOM reads digits with

`mlp.npz` is the network of `doomformer mnist` and `doomformer draw`: 784 binarised pixels -> 64 hidden units (8-level
staircase) -> 10 digit scores -> 45 pairwise comparisons -> 10 win counts -> one-hot (`df.with_argmax`, so the voodoo
dolls pick the digit). One DOOM map: 1,355 voodoo dolls, 158,550 sectors. Training: `training/train_mnist.py`, recipe in
[docs/drawpad.md](../../docs/drawpad.md#training-trainingtrain_mnistpy). Every number below comes from a file in this
directory.

| file | what |
|---|---|
| `mlp.npz` | the integer network (`df.MLP.load`), sha256 `f31c1b4d...04e7c4` |
| `digits.npz` | the first 100 MNIST test digits, bit-packed, with labels |
| `train_log.txt` | training output: accuracy of the float, integer and winner nets, and the map plan |
| `doom_100.txt` | `doomformer mnist --digits 100 --workers 6 --seed 0`: the 100 shipped digits in DOOM |
| `doom_1000.json` | `doomformer mnist --digits 1000 --seed 0 --workers 14 --record ...`: 1,000 test digits in DOOM, one line each |
| `doom_walked_30.txt` | `doomformer draw --digit D --seed S` for D = 0..9, S = 0..2: digits the marine walks on the DrawPad floor |

## NumPy integer reference

| net | MNIST test (10,000) | walked digits, strokes (5,000) | walked digits, SimWalker (5,000) |
|---|---|---|---|
| float model | 97.58 % | | |
| integer score net (argmax of the 10 scores) | 97.21 % | 98.72 % | 98.90 % |
| `mlp.npz` (DOOM's one-hot; score layer x0.2545) | 97.16 % | 98.70 % | 98.92 % |

## In DOOM

Every pass compares all 248 values DOOM computed (64 hidden values, 45 comparisons and 10 win counts, each with its
twin, and the 10 one-hot outputs) with the NumPy integer reference, tolerance 0.

| run | read correctly | exact | game tics |
|---|---|---|---|
| 100 shipped test digits (`doom_100.txt`) | 99/100 (test #62 read 5, is 9) | 100/100 | 5,696 per digit |
| 1,000 test digits sampled with seed 0 from the 10,000 (`doom_1000.json`) | 966/1,000 | 1,000/1,000 | 5,696 per digit, 5,696,000 in all |
| 30 walked digits (`doom_walked_30.txt`) | 30/30 | 30/30 | 5,705 per READ |

The 1,000-digit run took 7,421 s on one 16-core Linux arm64 machine with 14 DOOM games at once (8.1 digits per minute;
one pass took 83 to 153 s of wall time, 103 s on average, while other jobs shared the machine). Its 5,696,000 game tics
are 45 hours of real-time DOOM. The largest process of the run used 1.24 GB. No pass was retried.

DOOM's outputs in `doom_1000.json` can be checked without DOOM: rebuild the same 1,000 digits
(`doomformer.cli._mnist_digits` with `digits=1000, seed=0`) and compare `y` with `net.reference(net.encode(x))["y"]`;
all 1,000 rows match. 966/1,000 is the integer net's accuracy on that sample, since DOOM's answers equal the
reference's.
