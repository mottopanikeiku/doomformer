"""Measured cost of one learner step by map size and schedule level: python models/digits/cost_curve.py > cost_curve.txt

Each row builds the map for C classes x n features (the first n columns of the shipped digit features), runs STEPS
training steps in one DOOM game (every value and all weights checked against the twin, tolerance 0) and reports the
median wall time per step. Timings depend on the load of the machine; the loadavg at the start is printed."""
import os
import statistics
import time

import numpy as np

from doomformer import learn_digits as ld

STEPS = 4
rows = [(4, 6, 3), (4, 37, 3), (10, 12, 3), (10, 25, 3), (10, 37, 3), (10, 65, 3),
        (4, 6, 0), (4, 6, 2), (10, 37, 2), (10, 37, 1), (10, 37, 0)]
data = ld.load_digits()
print(f"loadavg at start: {os.getloadavg()[0]:.1f}; {os.cpu_count()} cores; {STEPS} steps per row (1 for opt 0 to 2 at 10 x 37)")
print("classes features opt  dolls   sectors  tics/step  s/step (median)  exact")
for C, n, opt in rows:
    steps = 1 if (opt < 3 and (C, n) == (10, 37)) else STEPS
    keep = data.y_train < C
    x = data.x_train[keep][:, :n] if n <= data.n_features else np.concatenate(
        [data.x_train[keep], np.zeros((keep.sum(), n - data.n_features), dtype=np.int64)], axis=1)
    y = data.y_train[keep]
    spec = ld.DigitPerceptron(n, C)
    t0 = time.time()
    with spec.compile(opt=opt) as learner:
        times = []
        for k in range(steps):
            times.append(learner.step(x[k], int(y[k])).seconds)
        info = learner.info
        print(f"{C:7d} {n:8d} {opt:3d} {info['dolls']:6,} {info['sectors']:9,} {info['tics']:10,} "
              f"{statistics.median(times):12.2f}      yes ({steps} steps)", flush=True)
