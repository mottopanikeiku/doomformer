#!/usr/bin/env python3
"""Acceptance checks for levels.json.

    python verify_levels.py stats                 # unique clean count, length distribution, charset, per tier
    python verify_levels.py sample [-n 15] [--seed S]   # re-open the source of random entries, confirm the name is there
    python verify_levels.py overlap               # how many of training/data.py NAMES are in the clean set
"""
from __future__ import annotations

import argparse
import collections
import html as htmllib
import importlib.util
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wbfetch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path: str | None = None) -> list[dict]:
    with open(path or os.path.join(HERE, "levels.json"), encoding="utf-8") as f:
        return json.load(f)


def stats(rows: list[dict]) -> None:
    cl = [r["clean"] for r in rows if r["clean"]]
    print(f"entries={len(rows)}  non-null clean={len(cl)}  unique non-null clean={len(set(cl))}")
    lens = collections.Counter(len(c) for c in cl)
    print("length distribution (chars -> count):")
    for k in sorted(lens):
        print(f"  {k:2d}  {lens[k]:5d}  {'#' * (lens[k] // 20)}")
    mean = sum(len(c) for c in cl) / len(cl)
    print(f"mean length {mean:.2f}, median {sorted(len(c) for c in cl)[len(cl)//2]}, max {max(map(len, cl))}")
    chars = collections.Counter("".join(cl))
    print("charset:", "".join(sorted(chars)), f"({len(chars)} symbols)")
    print("char counts:", dict(sorted(chars.items(), key=lambda kv: -kv[1])))
    print("per tier (non-null clean):", dict(collections.Counter(r["tier"] for r in rows if r["clean"])))
    print("per tier (null clean)    :", dict(collections.Counter(r["tier"] for r in rows if not r["clean"])))
    print("distinct WADs with >=1 clean name:", len({r["wad"] for r in rows if r["clean"]}))


def raw_url(u: str) -> str:
    m = re.match(r"(https://web\.archive\.org/web/\d{14})/(.*)$", u)
    return f"{m.group(1)}id_/{m.group(2)}" if m else u


def sample(rows: list[dict], n: int, seed: int) -> int:
    rng = random.Random(seed)
    pool = [r for r in rows if r["clean"]]
    bad = 0
    for r in rng.sample(pool, n):
        url = r["source_url"]
        try:
            data, final = wbfetch._get(raw_url(url))        # live request, bypasses the cache
            text = htmllib.unescape(data.decode("utf-8", "replace"))
            text = text.replace("\u2019", "'")
            if "web.archive.org" in url:   # DoomWiki: the exact article/link title "<slot>: <name> (<wad>)" must occur
                needle = f"{r['slot']}: {r['name']} ({r['wad']})"
                ok = needle in text
            else:                          # id source: the C string literal "E1M1: Name" / "level 7: name"
                lead = r["slot"] if r["slot"].startswith("E") else f"level {int(r['slot'][3:])}"
                needle = f'"{lead}: {r["name"]}"'
                ok = needle in text
            verdict = "OK" if ok else "FAIL"
            detail = f"needle={needle!r} found={ok} bytes={len(data)}"
        except Exception as e:  # noqa: BLE001
            verdict, detail = "FAIL", f"fetch error: {e}"
        bad += verdict != "OK"
        print(f"{verdict:4} {r['slot']:5} {r['name']!r:38} [{r['wad']}] -> {r['clean']!r}\n       {url}\n       {detail}")
    return bad


def overlap(rows: list[dict]) -> None:
    spec = importlib.util.spec_from_file_location("doomformer_data", os.path.join(HERE, "..", "data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = list(mod.NAMES)
    cs = {r["clean"] for r in rows if r["clean"]}
    hit = [n for n in names if n in cs]
    miss = [n for n in names if n not in cs]
    print(f"data.py NAMES: {len(names)}; present in levels.json clean set: {len(hit)}; missing: {len(miss)}")
    if miss:
        print("missing:", miss)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "sample", "overlap"])
    ap.add_argument("-n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    rows = load(a.file)
    if a.cmd == "stats":
        stats(rows)
    elif a.cmd == "overlap":
        overlap(rows)
    else:
        sys.exit(1 if sample(rows, a.n, a.seed) else 0)
