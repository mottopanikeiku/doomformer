"""09 - Record a GIF of DOOM computing XOR.

`brain.record(inputs, path)` runs the same verified passes as `brain(x)` and, while DOOM runs, samples the game every
8 tics (it only READS state: where every voodoo doll is) and draws a film:

* the compiled map seen from above (the WAD's linedefs, to scale, cut into bands stacked top to bottom; orange = lines
  carrying a special, i.e. weights, barriers, teleports),
* every voodoo doll as a dot coloured by its layer, white while it is moving; the long corridors are the clock dolls
  that release each layer's waves,
* what Python wrote (the input), and, only after the pass finished and the exact comparison with the integer
  reference passed, what DOOM read back and the word EXACT.

What DOOM does here: the whole XOR network (staircase activations, sums of weights). What Python does: writes the two
input bits, reads the result back, compares it with a NumPy integer reference, and draws the picture.

    python examples/09_record_a_gif.py [--view topdown|firstperson|both] [--width 800] [--mp4]

Writes examples/out/09_record_a_gif.gif (and .mp4 with --mp4). Exits non-zero on any mismatch, if a recorded result
differs from the XOR truth table, or if the GIF is not under 5 MB.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

import doomformer as df

OUT = Path(__file__).resolve().parent / "out"

# hidden unit 0 = OR(a, b), hidden unit 1 = AND(a, b), output = OR - AND = XOR (same net as example 01)
xor = df.MLP(
    weights=[np.array([[1, 1], [1, 1]]), np.array([[1, -1]])],
    biases=[np.array([0, -1]), np.array([0])],
    thresholds=[np.array([1]), np.array([1])],
    signed_input=False,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="topdown", choices=["topdown", "firstperson", "both"])
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--mp4", action="store_true", help="also write an .mp4")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    inputs = [list(b) for b in itertools.product([0, 1], repeat=2)]
    ok = True
    with df.compile(xor) as brain:
        for suffix in ([".gif", ".mp4"] if args.mp4 else [".gif"]):
            path = OUT / f"09_record_a_gif{suffix}"
            rec = brain.record(inputs, path, view=args.view, width=args.width)   # raises df.MismatchError on a mismatch
            print(rec)
            print(f"  wrote {path}: {rec.bytes / 1e6:.2f} MB")
            ok &= rec.verified and rec.bytes < 5_000_000
        for x, y in zip(inputs, rec.results):
            ref = xor.reference(np.array(x))["y"]
            right = int(y[0]) == (x[0] ^ x[1]) and np.array_equal(y, ref)
            ok &= right
            print(f"  x={x}  DOOM y={int(y[0])}  reference y={int(ref[0])}  xor={x[0] ^ x[1]}  EXACT" if right else
                  f"  x={x}  DOOM y={y.tolist()}  reference {ref.tolist()}  WRONG")
    print("OK: recorded passes verified, GIF written" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
