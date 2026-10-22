"""Acceptance report for models/doombrain (any L-layer QModel dir) on the fixed train/dev/val split.

Usage: python training/eval_brain.py [model_dir] [--temperature T] [--old models/doomnames]
Reports: check_limits over every corpus sequence, max |z| per tensor, torch-int64 == intref.forward
bit-exactness, integer teacher-forced accuracy on train/dev/val (and of the old 1-layer model on the same
splits), a temperature sweep (val CE of the integer logits, sampled novelty / word validity), 30 sampled
generations (seeds 0..29, prompt ''), greedy generations for '' and 'A'..'Z'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from doomformer import intref  # noqa: E402
from train_brain import LEVELS, encode, encode_target, int_forward, load_corpus, repo_rel  # noqa: E402

try:
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein
except ImportError:  # novelty distance is optional
    process = Levenshtein = None

ZKEYS = ("zq", "zk", "zv", "zo", "z1", "zm", "zg", "zu", "z2", "s_att", "P_att", "N_att", "s_mlp", "P_mlp", "N_mlp")


def nearest(name, names):
    """(edit distance, closest corpus name) or (-1, '') if rapidfuzz is missing."""
    if process is None or not name:
        return -1, ""
    best, _, _ = process.extractOne(name, names, scorer=Levenshtein.distance)
    return Levenshtein.distance(name, best), best


def encodable(m, name):
    return len(name) + 1 <= m.T and all(c in m.vocab for c in name)


def accuracy(m, names):
    names = [n for n in names if encodable(m, n)]
    hit = tot = seq = 0
    for n in names:
        pred = intref.forward(m, encode(m.vocab, n)).argmax(-1)
        ok = pred == np.array(encode_target(m.vocab, n))
        hit, tot, seq = hit + int(ok.sum()), tot + len(ok), seq + int(ok.all())
    return hit, tot, seq, len(names)


def val_ce(m, names, temps):
    """Mean next-char NLL (nats) of softmax(logits / T) of the integer model, per temperature."""
    L = [(intref.forward(m, encode(m.vocab, n)), np.array(encode_target(m.vocab, n))) for n in names if encodable(m, n)]
    out = {}
    for T in temps:
        nll = n = 0
        for lg, t in L:
            z = lg / T
            z = z - z.max(-1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
            nll -= lp[np.arange(len(t)), t].sum()
            n += len(t)
        out[T] = nll / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=str(HERE.parent / "models" / "doombrain"))
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--old", default=str(HERE.parent / "models" / "doomnames"))
    ap.add_argument("--corpus", default=str(LEVELS))
    ap.add_argument("--dev", type=int, default=300)
    a = ap.parse_args()
    path = Path(a.model)
    m = intref.QModel.load(path)
    corp = load_corpus(Path(a.corpus), a.dev)
    assert m.vocab == corp["vocab"], (m.vocab, corp["vocab"])
    names = corp["names"]
    nameset = set(names)
    seqs = [encode(m.vocab, n) for n in names]
    print(f"model: {repo_rel(path)}\nhyperparameters: L={m.L} T={m.T} d={m.d} H={m.H} dm={m.dm} K={m.K} KA={m.KA} KM={m.KM} "
          f"V={len(m.vocab)} attn={m.attn} R={m.R} norm={m.norm} Rn={m.Rn} mlp={m.mlp}  "
          f"params={sum(v.size for v in m.p.values())}")
    print(f"corpus {corp['source']}: {len(names)} names (train {len(corp['train'])}, dev {len(corp['dev'])}, "
          f"val {len(corp['val'])}); longest {max(map(len, names))} chars")

    errs = intref.check_limits(m, seqs)
    print(f"\n[1] check_limits over {len(seqs)} sequences -> {errs!r}  ({'PASS' if not errs else 'FAIL'})")
    worst = {}
    for s in seqs:
        tr = {}
        intref.forward(m, s, tr)
        for k, v in tr.items():
            if k.split(".")[-1] in ZKEYS + ("z0", "score", "logits"):
                worst[k] = max(worst.get(k, 0), int(np.abs(v).max()))
    print("    max |z| per tensor (final sums):", worst)
    print("    max |param| per array:", {k: int(np.abs(v).max()) for k, v in m.p.items() if "thr" not in k})
    print("    thresholds (min..max, min gap):", {k: (int(v.min()), int(v.max()),
                                                    int(np.diff(v, axis=-1).min()) if v.shape[-1] > 1 else 0)
                                                 for k, v in m.p.items() if "thr" in k})

    tok = torch.zeros(len(seqs), m.T, dtype=torch.long)
    for i, s in enumerate(seqs):
        tok[i, : len(s)] = torch.tensor(s)
    lt = int_forward(m.p, tok, m.H, m.L, m.attn, m.R, m.norm, m.Rn, m.mlp).numpy()
    exact = sum(int(np.array_equal(intref.forward(m, s), lt[i, : len(s)])) for i, s in enumerate(seqs))
    print(f"\n[2] torch int64 forward == intref.forward logits (all positions): {exact}/{len(seqs)} sequences "
          f"= {100 * exact / len(seqs):.1f}%  ({'PASS' if exact == len(seqs) else 'FAIL'})")

    print("\n[3] integer teacher-forced next-char accuracy (intref.forward)")
    old = intref.QModel.load(Path(a.old)) if Path(a.old).exists() else None
    old_train = set()
    if old is not None:
        from data import NAMES as OLD_NAMES
        old_train = set(OLD_NAMES)
    info = path / "train_info.json"
    res = {}
    for split in ("train", "dev", "val"):
        h, t, sq, n = accuracy(m, corp[split])
        res[split] = h / t
        print(f"    {path.name:10s} {split:5s}: {h}/{t} = {h / t:.4f}   ({sq}/{n} names token-perfect)")
    if old is not None:
        for split in ("train", "dev", "val"):
            h, t, sq, n = accuracy(old, corp[split])
            print(f"    {Path(a.old).name:10s} {split:5s}: {h}/{t} = {h / max(t, 1):.4f}   "
                  f"({n}/{len(corp[split])} names encodable: <=15 chars, in-vocab; {sq} token-perfect)")
        for split in ("dev", "val"):
            unseen = [n for n in corp[split] if n not in old_train]
            h, t, sq, n = accuracy(old, unseen)
            h2, t2, _, _ = accuracy(m, [x for x in unseen if encodable(old, x)])
            print(f"    on {split} names NOT in doomnames' 136 training names, encodable by both ({n}): "
                  f"doomnames {h}/{t} = {h / max(t, 1):.4f}   {path.name} {h2}/{t2} = {h2 / max(t2, 1):.4f}")
    if info.exists():
        ti = json.loads(info.read_text())
        print(f"    training wall time: {ti['wall_s']:.0f} s; learned float logit scale {ti['int_logit_scale']:.5f} "
              f"(=> nominal temperature {1 / ti['int_logit_scale']:.0f} integer-logit units)")

    # --- temperature sweep
    temps = [50, 100, 150, 200, 300, 400, 600, 800, 1200, 2000]
    ces = val_ce(m, corp["val"], temps)
    print("\n[4] val NLL (nats/char) of softmax(int logits / T):", {k: round(float(v), 3) for k, v in ces.items()})
    best_T = min(ces, key=ces.get)
    words = {w for n in names for w in n.split()}
    print(f"    CE-optimal T on val: {best_T}")
    print("    sampling stats over seeds 0..99 (prompt ''): T: novel%  word-valid%  mean-len  distinct")
    for T in sorted(set(temps[:8] + [best_T])):
        outs = [intref.generate(m, "", temperature=T, seed=s) for s in range(100)]
        nov = np.mean([o not in nameset and o != "" for o in outs])
        ws = [w for o in outs for w in o.split()]
        wv = np.mean([w in words for w in ws]) if ws else 0
        print(f"      {T:5d}: {100 * nov:5.1f}  {100 * wv:5.1f}  {np.mean([len(o) for o in outs]):5.1f}  {len(set(outs))}")
    T = a.temperature if a.temperature is not None else best_T
    print(f"\n[5] 30 sampled generations at temperature {T} (seeds 0..29, prompt '')")
    nnov = 0
    for s in range(30):
        o = intref.generate(m, "", temperature=T, seed=s)
        tag = "in-corpus" if o in nameset else "NOVEL"
        nnov += tag == "NOVEL"
        dist, near = nearest(o, names)
        print(f"    seed {s:2d}: {o!r:28} {tag:9}  nearest corpus name (edit dist {dist}): {near!r}")
    print(f"    {nnov}/30 novel")
    print("\n[6] greedy (intref.generate)")
    n_nov = tot = 0
    for pr in [""] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        if pr and pr not in m.vocab:
            continue
        o = intref.generate(m, pr)
        tag = "in-corpus" if o in nameset else "NOVEL"
        n_nov, tot = n_nov + (tag == "NOVEL"), tot + 1
        print(f"    {pr or '<empty>':>7} -> {o!r:28} {tag}")
    print(f"    {n_nov}/{tot} novel")


if __name__ == "__main__":
    main()
