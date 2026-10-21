"""01 - XOR, computed by DOOM.

A hand-written integer MLP (2 inputs -> 2 hidden units -> 1 output) that computes XOR, compiled into a DOOM map.

What happens in the game: every input bit is a "port": the floor height of a gate sector. Voodoo dolls (copies of
the player) walk through the gates; a doll only gets through if the port is high enough (that is the staircase
activation). Each doll that gets through crosses a line that raises or lowers an output sector's light level by the
weight, and that light level IS the output value. Python writes the inputs and reads the outputs; the arithmetic in
between is done by DOOM.

Hidden unit 0 = OR(a, b), hidden unit 1 = AND(a, b), output = OR - AND = XOR.
Part 2 repeats the trick with 3-bit parity, a function a single neuron cannot compute.

    python examples/01_xor.py

Exits non-zero if DOOM and the independent integer reference ever differ.
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np

import doomformer as df
from doomformer.record import topdown

# Layer 0: a_0 = adc_relu(x, [1]) = [x >= 1]  (a plain bit; signed_input=False means "no negative inputs")
# Layer 1: h_1 = b_0 + W_0 @ a_0 ; a_1 = adc_relu(h_1, [1]) = [h_1 >= 1]
# Layer 2: y = b_1 + W_1 @ a_1   (the last layer has no activation: the output is the raw sum)
xor = df.MLP(
    weights=[np.array([[1, 1],       # OR  = [a + b >= 1]
                       [1, 1]]),     # AND = [a + b - 1 >= 1]
             np.array([[1, -1]])],   # XOR = OR - AND
    biases=[np.array([0, -1]), np.array([0])],
    thresholds=[np.array([1]), np.array([1])],
    signed_input=False,
)

# Three-bit parity: parity(s) = [s >= 1] - [s >= 2] + [s >= 3] for s = number of ones.
parity3 = df.MLP(
    weights=[np.ones((3, 3), dtype=np.int64), np.array([[1, -1, 1]])],
    biases=[np.array([0, -1, -2]), np.array([0])],
    thresholds=[np.array([1]), np.array([1])],
    signed_input=False,
)


OUT = Path(__file__).resolve().parent / "out"


def run(name: str, net: df.MLP, n_bits: int, parity_of, image: str | None = None) -> bool:
    plan = df.plan(net)
    print(f"\n{name}: {plan['dolls']} voodoo dolls, {plan['sectors']} sectors")
    ok = True
    with df.compile(net) as brain:
        print(f"  map: {brain.info['wad']}")
        for bits in itertools.product([0, 1], repeat=n_bits):
            x = np.array(bits)
            t0 = time.time()
            try:
                y = brain(x)                    # runs DOOM, compares with net.reference, raises on any difference
                status = "EXACT"
            except df.MismatchError as e:
                y, status, ok = None, f"MISMATCH ({e})", False
            ref = net.reference(x)
            want = parity_of(bits)
            right = y is not None and int(y[0]) == want
            ok &= right
            print(f"  inputs {bits}  DOOM y={None if y is None else int(y[0])}  reference y={int(ref['y'][0])}  "
                  f"truth={want}  {status}  ({time.time() - t0:.1f}s)")
        if image:
            OUT.mkdir(exist_ok=True)
            topdown(brain.wad, title=f"{name} network, seen from above: {plan['dolls']} voodoo dolls, "
                    f"{plan['sectors']} sectors (orange = lines with a weight special)").save(OUT / image)
            print(f"  wrote {OUT / image}")
    return ok


def main() -> int:
    ok = run("XOR", xor, 2, lambda b: b[0] ^ b[1], image="01_xor_map.png")
    ok &= run("3-bit parity", parity3, 3, lambda b: sum(b) % 2)
    print("\nALL EXACT: DOOM computed XOR and parity bit-for-bit like the integer reference" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
