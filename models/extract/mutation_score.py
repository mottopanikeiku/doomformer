"""Mutation score of the static extractor: change one field of a compiled map, read it back, count what is detected.

    python models/extract/mutation_score.py --opt 0 --json models/extract/mutation_score_opt0.json
    python models/extract/mutation_score.py --combine models/extract/mutation_score.json models/extract/mutation_score_opt*.json

Edits: every integer field of the TEXTMAP (+1, and the sign flip when the value is not 0), every float coordinate of a
vertex and a thing (+1), every true boolean linedef or thing flag (set false), every byte of the BEHAVIOR lump (+1).
Extractor verdict per edit: DIFF (the netlist differs, a finding is raised, or ExtractError) or EQUAL. A stratified sample
of the EQUAL edits (by field kind) is then run in DOOM on 12 inputs at tolerance 0 (at most 3 games at once): CAUGHT (a
mismatch), ERROR (the pass did not settle or raised), CRASH (the game died or hung), SAME (every input exact: equivalent
on these inputs). Exit status 0.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from doomformer import extract as ex
from doomformer.kit import MLP
from doomformer.kit import compile as df_compile

INT_FIELD = re.compile(r"\b(\w+) = (-?\d+);")
FLOAT_FIELD = re.compile(r"\b(x|y) = (-?\d+\.\d+);")
BOOL_FIELD = re.compile(r"\b(\w+) = true;")


def small_net() -> MLP:
    """3 inputs (signed), 4 hidden units, 3 outputs: two weight layers, thresholds 2 and 3 lanes."""
    rng = np.random.default_rng(7)
    W0 = rng.integers(-6, 7, (4, 3))
    W1 = rng.integers(-5, 6, (3, 4))
    W0[0, 0] = 3
    W1[0, 0] = -4
    return MLP([W0, W1], [rng.integers(-5, 6, 4), rng.integers(-5, 6, 3)], [np.array([2, 5]), np.array([3, 6, 9])],
               signed_input=True)


def inputs_for(net: MLP, n: int = 12, seed: int = 3) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        x = rng.integers(-14, 15, net.n_in)
        try:
            net.check(x)
        except Exception:
            continue
        out.append([int(v) for v in x])
    return out


def split_blocks(text: str) -> list[tuple[str, str]]:
    return [(ln.split(" ", 1)[0], ln) for ln in text.split("\n")]


def kind_of_line(kind: str, ln: str, key: str) -> str:
    if kind == "linedef":
        m = re.search(r"special = (\d+);", ln)
        sp = m.group(1) if m else "0"
        return f"linedef[special {sp}].{key}"
    return f"{kind}.{key}"


def textmap_edits(text: str):
    """Yield (kind, description, new_text) for every single edit of the TEXTMAP."""
    lines = text.split("\n")
    for n, ln in enumerate(lines):
        kind = ln.split(" ", 1)[0]
        if kind not in ("vertex", "linedef", "sidedef", "sector", "thing"):
            continue
        fields = [(m.group(1), m.start(), m.end(), m.group(2)) for m in INT_FIELD.finditer(ln)]
        for key, a, b, val in fields:
            v = int(val)
            for tag, new in (("+1", v + 1), ("negate", -v)):
                if new == v:
                    continue
                yield (kind_of_line(kind, ln, key), f"{kind} {n} {key} {v} -> {new}",
                       "\n".join(lines[:n] + [ln[:a] + f"{key} = {new};" + ln[b:]] + lines[n + 1:]), tag)
        for m in FLOAT_FIELD.finditer(ln):
            v = float(m.group(2))
            yield (f"{kind}.{m.group(1)} (coordinate)", f"{kind} {n} {m.group(1)} {v} -> {v + 1}",
                   "\n".join(lines[:n] + [ln[:m.start()] + f"{m.group(1)} = {v + 1:.1f};" + ln[m.end():]] + lines[n + 1:]),
                   "+1")
        if kind in ("linedef", "thing"):
            for m in BOOL_FIELD.finditer(ln):
                yield (kind_of_line(kind, ln, m.group(1)) + " (flag)", f"{kind} {n} {m.group(1)} true -> false",
                       "\n".join(lines[:n] + [ln[:m.start()] + f"{m.group(1)} = false;" + ln[m.end():]] + lines[n + 1:]),
                       "flag off")


def behavior_kind(beh: bytes, i: int, scripts: dict) -> str:
    dirofs = struct.unpack("<i", beh[4:8])[0]
    if i < 8:
        return "behavior header"
    if i >= dirofs:
        return "behavior legacy directory (ACS0, unused behind the ACSe header)"
    if i >= dirofs - 8:
        return "behavior ACSe header"
    starts = sorted((off, num) for num, (off, _) in scripts.items())
    num = None
    for off, n in starts:
        if off <= i:
            num = n
    group = "reset script 3" if num == 3 else f"script {num}" if num in (1, 2, 4, 7, 8, 9) else \
        "copy script" if num is not None and 30 <= num < 100 else "readout page" if num is not None and 100 <= num < 200 else \
        f"script {num}"
    return f"behavior {group}"


def verdict(lumps: dict[str, bytes], net: MLP) -> tuple[str, str]:
    try:
        nl = ex.extract_lumps(lumps)
    except (ex.ExtractError, ValueError, KeyError, IndexError, struct.error) as e:
        return "DIFF", f"refused: {type(e).__name__}: {str(e)[:80]}"
    diffs = nl.diff(net)
    if diffs:
        return "DIFF", str(diffs[0])[:100]
    if nl.findings:
        return "DIFF", "finding: " + nl.findings[0][:90]
    return "EQUAL", ""


def write_wad(path: Path, lumps: list[tuple[str, bytes]]) -> None:
    data, directory = b"", b""
    for name, blob in lumps:
        directory += struct.pack("<ii8s", 12 + len(data), len(blob), name.encode())
        data += blob
    path.write_bytes(struct.pack("<4sii", b"PWAD", len(lumps), 12 + len(data)) + data + directory)


def doom_worker(argv) -> int:
    """Child process: run the inputs on one mutant map; print one JSON line."""
    net_path, wad, xs = argv[0], Path(argv[1]), json.loads(argv[2])
    from doomformer.verify import _run
    try:
        r = _run(MLP.load(net_path), wad, xs, 1, True)
        print(json.dumps(dict(status="CAUGHT" if r["mismatches"] else "SAME")))
    except Exception as e:
        print(json.dumps(dict(status="ERROR", detail=f"{type(e).__name__}: {e}"[:160])))
    return 0


def run_doom(job) -> dict:
    net_path, wad, xs = job
    try:
        p = subprocess.run([sys.executable, __file__, "--doom-worker", str(net_path), str(wad), json.dumps(xs)],
                           capture_output=True, text=True, timeout=180, env=dict(os.environ, OMP_NUM_THREADS="1"))
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line.startswith("{") else dict(status="CRASH", detail=f"exit {p.returncode}")
    except subprocess.TimeoutExpired:
        return dict(status="CRASH", detail="timeout 180 s")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "--doom-worker":
        return doom_worker(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--opt", type=int, default=0)
    ap.add_argument("--per-kind", type=int, default=2, help="EQUAL edits per field kind sent to DOOM (default 2)")
    ap.add_argument("--max-doom", type=int, default=150)
    ap.add_argument("--games", type=int, default=3, help="DOOM games at once (default 3)")
    ap.add_argument("--no-doom", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--combine", nargs="+", metavar="FILE", help="instead: merge result files into the first path")
    args = ap.parse_args(argv)
    if args.combine:
        out, *ins = args.combine
        recs = [json.loads(Path(f).read_text()) for f in ins]
        Path(out).write_text(json.dumps(dict(runs=recs), indent=1) + "\n")
        return 0
    net = small_net()
    xs = inputs_for(net)
    work = Path(tempfile.mkdtemp(prefix="dfmut-"))
    base = work / "base.wad"
    df_compile(net, base, mode="map", opt=args.opt, limits=False).close()
    net_path = work / "net.npz"
    net.save(net_path)
    lumps = ex.read_lumps(base)
    text = lumps["TEXTMAP"].decode()
    beh = lumps["BEHAVIOR"]
    assert verdict(lumps, net)[0] == "EQUAL", "the unedited map must read back equal"
    scripts = ex.acs_scripts(beh)
    t0 = time.perf_counter()
    rows = []                                   # (kind, variant, desc, extractor verdict, detail, path info)
    mutants: dict[int, tuple] = {}              # row index -> ("TEXTMAP", edit number) or ("BEHAVIOR", byte)
    for k, (kind, desc, new_text, variant) in enumerate(textmap_edits(text)):
        v, detail = verdict({**lumps, "TEXTMAP": new_text.encode()}, net)
        rows.append([kind, variant, desc, v, detail])
        if v == "EQUAL":
            mutants[len(rows) - 1] = ("TEXTMAP", k)
    for i in range(len(beh)):
        b = bytearray(beh)
        b[i] = (b[i] + 1) & 255
        kind = behavior_kind(beh, i, scripts)
        v, detail = verdict({**lumps, "BEHAVIOR": bytes(b)}, net)
        rows.append([kind, "+1", f"BEHAVIOR byte {i} {beh[i]} -> {(beh[i] + 1) & 255}", v, detail])
        if v == "EQUAL":
            mutants[len(rows) - 1] = ("BEHAVIOR", i)
    t_extract = time.perf_counter() - t0
    # DOOM on a stratified sample of the EQUAL edits
    by_kind: dict[str, list[int]] = defaultdict(list)
    for idx in mutants:
        by_kind[rows[idx][0]].append(idx)
    rng = random.Random(0)
    chosen = []
    for kind, idxs in sorted(by_kind.items()):
        rng.shuffle(idxs)
        chosen += idxs[:args.per_kind]
    rng.shuffle(chosen)
    chosen = chosen[:args.max_doom]
    doom: dict[int, dict] = {}
    if not args.no_doom:
        jobs = []
        want_text = {mutants[i][1]: i for i in chosen if mutants[i][0] == "TEXTMAP"}
        texts = {want_text[k]: t for k, (_, _, t, _) in enumerate(textmap_edits(text)) if k in want_text}
        for idx in chosen:
            lump, ref = mutants[idx]
            if lump == "TEXTMAP":
                blob = texts[idx].encode()
            else:
                b = bytearray(beh)
                b[ref] = (b[ref] + 1) & 255
                blob = bytes(b)
            p = work / f"m{idx}.wad"
            write_wad(p, [(n, blob if n == lump else b) for n, b in ex.read_lumps(base).items()])
            shutil.copyfile(base.with_suffix(".json"), p.with_suffix(".json"))
            jobs.append((net_path, p, xs))
        with ThreadPoolExecutor(args.games) as pool:
            for idx, res in zip(chosen, pool.map(run_doom, jobs)):
                doom[idx] = res
    # report
    kinds: dict[str, Counter] = defaultdict(Counter)
    for idx, (kind, variant, desc, v, detail) in enumerate(rows):
        kinds[kind]["edits"] += 1
        kinds[kind]["detected" if v == "DIFF" else "undetected"] += 1
        if idx in doom:
            kinds[kind]["doom_" + doom[idx]["status"]] += 1
    tot = Counter()
    for c in kinds.values():
        tot.update(c)
    out = dict(opt=args.opt, net="3-4-3 signed, thresholds [2,5] and [3,6,9] (small_net)", inputs=xs, edits=tot["edits"],
               detected=tot["detected"], undetected=tot["undetected"], doom_sampled=len(doom),
               doom={k[5:]: v for k, v in tot.items() if k.startswith("doom_")},
               extract_seconds=round(t_extract, 1),
               kinds={k: dict(c) for k, c in sorted(kinds.items())},
               undetected_examples={k: [r[2] for r in rows if r[0] == k and r[3] == "EQUAL"][:2] for k in sorted(kinds)
                                    if kinds[k]["undetected"]},
               doom_caught_examples=[rows[i][2] for i in chosen if doom.get(i, {}).get("status") == "CAUGHT"][:20])
    print(f"opt {args.opt}: {out['edits']} edits, {out['detected']} detected, {out['undetected']} undetected; DOOM on "
          f"{out['doom_sampled']} of the undetected: {out['doom']}")
    for k, c in sorted(kinds.items()):
        if c["undetected"]:
            print(f"  {k}: {c['edits']} edits, {c['undetected']} undetected, doom {{ {', '.join(f'{a[5:]} {b}' for a, b in c.items() if a.startswith('doom_'))} }}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
