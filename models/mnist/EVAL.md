# mnist: the network DOOM reads digits with

`mlp.npz` is the network of `doomformer mnist` and `doomformer draw`. It maps 784 binarised pixels to 64 hidden units
(8-level staircase), 10 digit scores, 45 pairwise comparisons, 10 win counts and a one-hot output (`df.with_argmax`).
The voodoo dolls pick the digit. One DOOM map holds 1,355 voodoo dolls and 158,550 sectors. Training:
`training/train_mnist.py`, recipe in [docs/drawpad.md](../../docs/drawpad.md#training-trainingtrain_mnistpy). Every
number below comes from a file in this directory.

| file | what | command |
|---|---|---|
| `mlp.npz` | the integer network (`df.MLP.load`), sha256 `f31c1b4d...04e7c4` | |
| `digits.npz` | the first 100 MNIST test digits, bit-packed, with labels | |
| `train_log.txt` | accuracy of the float, integer and winner nets, and the map plan | `training/train_mnist.py` |
| `doom_100.txt` | the 100 shipped digits in DOOM | `doomformer mnist --digits 100 --workers 6 --seed 0` |
| `doom_1000.json` | 1,000 test digits in DOOM, one map, one line each | `doomformer mnist --digits 1000 --seed 0 --workers 14 --record ...` |
| `doom_tiles_10000.json` | all 10,000 test digits in DOOM as four tile maps, one line each | `doomformer mnist --tiles --all --workers 6 --record ...` |
| `doom_walked_30.txt` | digits the marine walks on the DrawPad floor, D = 0..9, S = 0..2 | `doomformer draw --digit D --seed S` |

## NumPy integer reference

Accuracy of the nets computed in NumPy, without DOOM:

| net | MNIST test (10,000) | walked digits, strokes (5,000) | walked digits, SimWalker (5,000) |
|---|---|---|---|
| float model | 97.58 % | | |
| integer score net (argmax of the 10 scores) | 97.21 % | 98.72 % | 98.90 % |
| `mlp.npz` (DOOM's one-hot; score layer x0.2545) | 97.16 % | 98.70 % | 98.92 % |

## In DOOM

Every pass compares all 248 values DOOM computed with the NumPy integer reference at tolerance 0. They are the 64
hidden values, 45 comparisons and 10 win counts, each with its twin, and the 10 one-hot outputs.

| run | read correctly | exact | game tics |
|---|---|---|---|
| 100 shipped test digits (`doom_100.txt`) | 99/100 (test #62 read 5, is 9) | 100/100 | 5,696 per digit |
| 1,000 test digits sampled with seed 0 from the 10,000 (`doom_1000.json`) | 966/1,000 | 1,000/1,000 | 5,696 per digit, 5,696,000 in all |
| all 10,000 test digits, tile maps (`doom_tiles_10000.json`) | 9,716/10,000 (97.16 %) | 10,000/10,000 | 216 per digit (4 tiles, summed) |
| 30 walked digits (`doom_walked_30.txt`) | 30/30 | 30/30 | 5,705 per READ |

DOOM's answers equal the reference's, so each accuracy above is the integer net's accuracy on that sample. The
10,000-digit tile run reproduces the `mlp.npz` row of the NumPy table. Anyone can check the files without DOOM.
Rebuild the 1,000 digits with `doomformer.cli._mnist_digits(digits=1000, seed=0)` and compare `y` with
`net.reference(net.encode(x))["y"]`.

`doom_1000.json` is the one-map design, with no Python between layers. `doom_tiles_10000.json` is the tile design,
where Python copies each layer's values into the next tile's game. Wall times and the price of each design:
[docs/scaling.md](../../docs/scaling.md#one-map-versus-tiles-the-price).
