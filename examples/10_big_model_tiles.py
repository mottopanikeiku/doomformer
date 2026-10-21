"""10 - The trained d=16 doomnames transformer across two DOOM workers.

DOOM computes tile arithmetic; ACS moves gates and exposes sector lights.
Python copies the integer bus between games and samples the next token. Every
exposed tensor and negation twin is checked at tolerance zero on every call.
By default it completes the short name FEAR from prompt F, including the EOS
decision. It exits nonzero on any integer or text mismatch.

    python examples/10_big_model_tiles.py
"""
import argparse
import json
import time

import doomformer as df
from doomformer import intref
from doomformer.tiled import TiledTransformerBrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="F")
    parser.add_argument("--max-new", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    net = df.Transformer.load("doomnames")
    started = time.perf_counter()
    with df.compile(net, mode="tiles", workers=args.workers) as brain:
        assert isinstance(brain, TiledTransformerBrain)
        built = time.perf_counter()
        text = brain.generate(args.prompt, max_new=args.max_new,
                              temperature=args.temperature, seed=args.seed)
        wall = time.perf_counter() - built
        ref = intref.generate(net, args.prompt, max_new=args.max_new,
                              temperature=args.temperature, seed=args.seed)
        if text != ref:
            raise RuntimeError(f"text mismatch: DOOM {text!r}, reference {ref!r}")
        print(json.dumps(dict(text=text, reference=ref, exact=True,
                              compile_seconds=round(built - started, 3),
                              generate_seconds=round(wall, 3),
                              total_seconds=round(time.perf_counter() - started, 3),
                              **brain.info), indent=2))
        print(f"OK: DOOM wrote {text!r}, identical to the integer reference; every exposed value exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
