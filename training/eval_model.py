"""Acceptance report for models/doomnames (or any QModel dir trained on training/data.py's names).

Usage: python training/eval_model.py [model_dir]
Checks: check_limits, torch-int64 == intref.forward bit-exactness, teacher-forced accuracy of the integer
model (and of the float model if train_qat's float-stage accuracy is in train_info.json), generation per prompt,
and the names the default demo writes (`intref.generate`, temperature 180, seeds 20..29, and greedy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import NAMES, VOCAB, encode, encode_target  # noqa: E402
from doomformer import intref  # noqa: E402
from train_brain import repo_rel  # noqa: E402
from train_qat import int_forward  # noqa: E402

DEMO_TEMPERATURE, DEMO_SEEDS = 180.0, range(20, 30)


def main(path: Path):
    m = intref.QModel.load(path)
    assert m.vocab == VOCAB
    seqs = [encode(n) for n in NAMES]
    tgts = [encode_target(n) for n in NAMES]
    cfg = dict(T=m.T, d=m.d, H=m.H, dm=m.dm, K=m.K, KA=m.KA, KM=m.KM, L=m.L, V=len(m.vocab), attn=m.attn, R=m.R)
    print(f"model: {repo_rel(path)}\nhyperparameters: {cfg}")
    print(f"training names: {len(NAMES)}  target tokens: {sum(len(t) for t in tgts)}")

    errs = intref.check_limits(m, seqs)
    print(f"\n[1] check_limits over {len(seqs)} sequences -> {errs!r}  ({'PASS' if not errs else 'FAIL'})")

    # per-tensor max |z| (full accumulations, as in intref's trace)
    keys = ["zq", "zk", "zv", "score", "zo"] + (["den", "P", "N"] if m.attn == "softmax" else []) + ["z1", "zm", "z2"]
    keys = ["z0"] + [f"L{l}.{k}" for l in range(m.L) for k in keys] + ["logits"]
    worst = {}
    for s in seqs:
        tr = {}
        intref.forward(m, s, tr)
        for k in keys:
            worst[k] = max(worst.get(k, 0), int(np.abs(tr[k]).max()))
    print("    max |z| per tensor (final sums):", worst)
    print("    max |param| per array:", {k: int(np.abs(v).max()) for k, v in m.p.items() if "thr" not in k})
    print("    thresholds:", {k: (int(v.min()), int(v.max())) for k, v in m.p.items() if "thr" in k})
    for l in range(m.L):
        print(f"    L{l}.thr_a: {m.p[f'L{l}.thr_a'].tolist()}")

    # bit-exactness: standalone torch int64 forward vs intref.forward
    T = m.T
    tok = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        tok[i, : len(s)] = torch.tensor(s)
    lt = int_forward(m.p, tok, m.H, m.attn, m.R).numpy()
    exact = 0
    acc_n = acc_d = 0
    seq_ok = 0
    for i, (s, t) in enumerate(zip(seqs, tgts)):
        ref = intref.forward(m, s)
        exact += int(np.array_equal(ref, lt[i, : len(s)]))
        pred = ref.argmax(-1)
        ok = pred == np.array(t)
        acc_n += int(ok.sum())
        acc_d += len(t)
        seq_ok += int(ok.all())
    print(f"\n[2] torch int64 forward == intref.forward logits (all positions): {exact}/{len(seqs)} sequences "
          f"= {100 * exact / len(seqs):.1f}%  ({'PASS' if exact == len(seqs) else 'FAIL'})")
    print(f"[3] integer model teacher-forced next-char accuracy (intref.forward): {acc_n}/{acc_d} = {acc_n / acc_d:.4f}; "
          f"names reproduced token-perfectly: {seq_ok}/{len(seqs)}")
    info = path / "train_info.json"
    if info.exists():
        ti = json.loads(info.read_text())
        print(f"    float model (smooth ADC, unrounded weights; train_qat float stage) accuracy: {ti['float_acc']:.4f}")
        print(f"    training wall time: {ti['wall_s']:.0f} s")

    print("\n[4] generation (intref.generate, greedy)")
    names = set(NAMES)
    starts = sorted({n[0] for n in NAMES if n[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})
    novel = 0
    total = 0
    for pr in [""] + [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c in starts]:
        out = intref.generate(m, pr)
        tag = "TRAIN" if out in names else "NOVEL"
        novel += tag == "NOVEL"
        total += 1
        print(f"    {pr or '<empty>':>7} -> {out!r:24} {tag}")
    print(f"    {novel}/{total} novel, {total - novel}/{total} verbatim")

    print(f"\n[5] demo names (intref.generate, empty prompt, max_new {m.T - 1}, temperature {DEMO_TEMPERATURE:g})")
    train = 0
    for seed in DEMO_SEEDS:
        out = intref.generate(m, "", max_new=m.T - 1, temperature=DEMO_TEMPERATURE, seed=seed)
        train += out in names
        print(f"    seed {seed} -> {out!r:24} {'TRAIN' if out in names else 'NOVEL'}")
    print(f"    {train}/{len(DEMO_SEEDS)} in the training corpus")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "models" / "doomnames")
