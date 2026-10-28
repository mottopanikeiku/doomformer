"""Draw learn_1500.png from learn_1500.json (the run's learning curve and the ceiling heights DOOM learned).

    python models/digits/figure.py models/digits/learn_1500.json models/digits/learn_1500.png

Left: held-out accuracy against examples seen, for the weights read from DOOM (the same numbers as the NumPy twin with
the same clamp) and for the textbook NumPy perceptron without the clamp on the same stream. Right: the 10 x 36 pixel
weights DOOM holds after the run, one 6x6 tile per digit class (red: positive ceiling, blue: negative, white: 0; the
colour scale is the largest |w| of the run)."""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

src, dst = sys.argv[1:3]
r = json.loads(open(src).read())
W, H = 1180, 520
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
x0, y0, x1, y1 = 70, 60, 560, 440
d.rectangle([x0, y0, x1, y1], outline="black")
steps = r["steps"]
for a in range(0, 101, 20):
    y = y1 - (y1 - y0) * a / 100
    d.line([x0, y, x1, y], fill=(220, 220, 220))
    d.text((x0 - 38, y - 6), f"{a}%", fill="black")
for s in range(0, steps + 1, max(1, steps // 6 // 50 * 50 or 50)):
    x = x0 + (x1 - x0) * s / steps
    d.line([x, y1, x, y1 + 5], fill="black")
    d.text((x - 12, y1 + 8), str(s), fill="black")


def line(curve, colour):
    pts = sorted((int(k), v) for k, v in curve.items())
    xy = [(x0 + (x1 - x0) * k / steps, y1 - (y1 - y0) * v) for k, v in pts]
    d.line(xy, fill=colour, width=3)


line(r["curve_numpy_plain"] | {"0": r["curve"]["0"]}, (150, 150, 150))
line(r["curve"], (200, 30, 30))
d.text((x0, 20), f"DOOM learns digits: {steps} online steps, {r['classes']} classes, {r['features']} binary features", fill="black")
d.text((x0 + 10, y0 + 10), "red: weights read from DOOM's ceilings (equal to NumPy, same clamp, every step)", fill=(200, 30, 30))
d.text((x0 + 10, y0 + 26), "grey: textbook NumPy perceptron, no clamp, same stream", fill=(110, 110, 110))
d.text((x0 + 160, y1 + 30), "examples seen", fill="black")
d.text((x0, y1 + 50), f"held-out accuracy {100 * r['heldout']:.1f}% on 500 digits; online accuracy over the stream "
       f"{100 * r['online_accuracy']:.1f}%", fill="black")
w = np.array(r["weights"])[:, :36].reshape(-1, 6, 6)
m = max(1, int(np.abs(w).max()))
cell, gap = 14, 18
for c in range(len(w)):
    cx, cy = 620 + (c % 5) * (6 * cell + gap), 60 + (c // 5) * (6 * cell + gap + 70)
    for i in range(6):
        for j in range(6):
            v = w[c, i, j] / m
            col = (255, int(255 * (1 - v)), int(255 * (1 - v))) if v > 0 else (int(255 * (1 + v)), int(255 * (1 + v)), 255)
            d.rectangle([cx + j * cell, cy + i * cell, cx + (j + 1) * cell - 1, cy + (i + 1) * cell - 1], fill=col)
    d.text((cx + 28, cy + 6 * cell + 4), f"digit {c}", fill="black")
d.text((620, 20), f"ceiling heights after the run (scale +-{m})", fill="black")
img.save(dst)
