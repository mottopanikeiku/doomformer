"""05 - A whole transformer in one DOOM map.

Loads the trained name-generating transformer (the shipped model "doomnames_onemap": 1 layer, d=8, 1 head, MLP 16, context 6;
3,548 voodoo dolls, 95,448 sectors) and lets DOOM run its forward pass to write letters. Embeddings, projections,
causal attention scores, attention-weighted values, MLP and logits are computed by dolls crossing lines in ONE map.
Python only writes the token ids, waits, reads the values back, and samples the next token from the logits
(argmax at temperature 0, seeded sampling otherwise, exactly like intref.generate).

EVERY step is verified: all exposed pre-activations, attention scores and logits must equal the integer reference
(intref.forward) bit for bit, or df.MismatchError is raised.

THIS IS SLOW: 85-96 seconds per letter on one core (about 3,500 game tics of dolls walking; measured), plus a few
seconds to build and load the map. The default generates a single letter. For a whole name use --max-new 5 (a 5-step
run measured 7 minutes).
--tiny uses a random-weight toy transformer (onemap.tiny_model) that runs in seconds; it writes nonsense, but the
bit-exact checking is the same.

    python examples/05_one_map_transformer.py [--prompt SH] [--max-new 1] [--temperature 0] [--seed 0] [--tiny]

Exits non-zero on any mismatch.
"""
import argparse
import sys
import time
from pathlib import Path

import doomformer as df
from doomformer import intref, onemap
from doomformer.record import topdown

OUT = Path(__file__).resolve().parent / "out"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="", help="characters to start from (vocab: letters, space, ' and !)")
    ap.add_argument("--max-new", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="random-weight toy model, fast")
    args = ap.parse_args()

    if args.tiny:
        net = onemap.tiny_model()                    # a QModel; df.Transformer is intref.QModel
    else:
        net = df.Transformer.load("doomnames_onemap")
        print("Warning: ~90 s per generated letter. Ctrl-C to stop; use --tiny for a fast demo.", flush=True)
    model = net
    print(f"plan (nothing built yet): {df.plan(net)}", flush=True)

    t0 = time.time()
    with df.compile(net) as brain:
        print(f"map built in {time.time() - t0:.0f}s: {brain.info['dolls']} dolls, "
              f"{brain.info['sectors']} sectors; the game loads on the first pass", flush=True)
        t1 = time.time()
        text = brain.generate(args.prompt, temperature=args.temperature, seed=args.seed, max_new=args.max_new)
        elapsed = time.time() - t1
        print(f"DOOM generated {text!r} in {elapsed:.0f}s, every step verified bit-exact "
              f"(tics of the last pass: {brain.info.get('tics')})")
        OUT.mkdir(exist_ok=True)
        png = OUT / ("05_onemap_tiny.png" if args.tiny else "05_onemap.png")
        topdown(brain.wad, size=(1000, 520),
                title=f"the transformer's map from above ({brain.info['dolls']} dolls, {brain.info['sectors']} sectors); "
                      f"prompt {args.prompt!r} -> DOOM wrote {text!r}").save(png)
        print(f"wrote {png}")
    ref = intref.generate(model, args.prompt, max_new=args.max_new, temperature=args.temperature, seed=args.seed)
    print(f"integer reference generated {ref!r}")
    ok = ref == text
    print("OK: identical" if ok else "FAILED: DOOM and reference differ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
