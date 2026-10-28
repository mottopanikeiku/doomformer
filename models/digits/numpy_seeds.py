"""NumPy only (no DOOM): the held-out accuracy of the same perceptron on 20 different example streams.

    python models/digits/numpy_seeds.py > models/digits/numpy_seeds.txt

`clamped` is the learner DOOM runs (weights enter the score clamped to [-3, 3], digit_specs); `plain` is the textbook
integer perceptron without the clamp. Seed 0 is the stream of the DOOM run in learn_1500.log."""
import numpy as np

from doomformer import learn_digits as ld

data = ld.load_digits()
marks = {100, 300, 500, 1000, 1500}
runs = {name: [] for name in ("clamped", "plain")}
for seed in range(20):
    order = ld.stream(len(data.x_train), 1500, seed)
    runs["clamped"].append(ld.plain_perceptron(data, order, marks, clip=ld.CLIP)["curve"])
    runs["plain"].append(ld.plain_perceptron(data, order, marks, clip=None)["curve"])
print(f"held-out accuracy (%) of {len(data.y_test)} digits by examples seen, 20 streams (seeds 0..19, seed 0 is the DOOM stream of learn_1500.log), numpy {np.__version__}")
print("variant   examples  mean  sd   min   max   seed 0")
for name, curves in runs.items():
    for m in sorted(marks):
        a = 100 * np.array([c[m] for c in curves])
        print(f"{name:8s} {m:8d} {a.mean():5.1f} {a.std():4.1f} {a.min():5.1f} {a.max():5.1f} {a[0]:7.1f}")
