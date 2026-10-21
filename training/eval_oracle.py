"""Acceptance report for models/doomoracle (QModel dir) on training/data/oracle.json.

Checks, all on the INTEGER model (doomformer.intref):
  1. intref.check_limits(...) == [] on every training/held-out sequence (QUESTION>ANSWER with BOS)
  2. the pure-torch int64 forward of train_brain.int_forward equals intref.forward on 100 % of sequences
  3. exact-answer accuracy of greedy generation  intref.generate(m, question + '>')  on train and held-out paraphrases
  4. 15 held-out samples, plus a few questions that are not in the data at all
Writes <model>/EVAL_REPORT.md and <model>/eval_report.json.

Usage: python training/eval_oracle.py --model models/doomoracle
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from doomformer import intref  # noqa: E402
from train_brain import int_forward, repo_rel  # noqa: E402
from train_oracle import ORACLE, SEP, load_oracle, sequences  # noqa: E402

# never in oracle.json: unknown facts and unseen phrasings of known ones
NOVEL = [
    ("UNKNOWN FACT", "WHAT DOES IDDT DO?"),
    ("UNKNOWN FACT", "WHO COMPOSED THE MUSIC?"),
    ("UNKNOWN FACT", "WHAT IS E ONE M FIVE?"),
    ("NEW PHRASING", "HOW STRONG IS AN IMP?"),
    ("NEW PHRASING", "WHICH MAP HOLDS THE SPIDER?"),
    ("NEW PHRASING", "WHAT GOD MODE CODE?"),
    ("OUT OF SCOPE", "WHAT IS THE CAPITAL OF FRANCE?"),
]


def answer_of(m, question: str) -> str:
    out = intref.generate(m, question + SEP)
    return out[len(question) + 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE.parent / "models" / "doomoracle"))
    ap.add_argument("--data", default=str(ORACLE))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model_dir = Path(a.model)
    m = intref.QModel.load(model_dir)
    corp = load_oracle(Path(a.data))
    vocab = m.vocab
    assert vocab == corp["vocab"], (vocab, corp["vocab"])
    rows = corp["train"] + corp["val"]
    seqs = sequences(vocab, rows)
    rep = dict(model=repo_rel(model_dir), L=m.L, d=m.d, H=m.H, dm=m.dm, K=m.K, T=m.T, vocab=vocab,
               n_facts=corp["n_facts"], n_train=len(corp["train"]), n_heldout=len(corp["val"]))

    # 1. DOOM limits
    errs = intref.check_limits(m, seqs)
    rep["check_limits"] = errs
    print("check_limits:", errs if errs else "[]")

    # 2. torch-int forward == intref.forward
    same = 0
    for s in seqs:
        ref = intref.forward(m, s)
        out = int_forward(m.p, torch.tensor([s]), m.H, m.L)[0].numpy()
        same += int(np.array_equal(ref, out))
    rep["torch_int_equals_intref"] = f"{same}/{len(seqs)}"
    print("torch-int == intref.forward:", rep["torch_int_equals_intref"])

    # 3. greedy exact-answer accuracy
    def evaluate(rows_):
        ok, bad = 0, []
        for fid, q, ans in rows_:
            got = answer_of(m, q)
            if got == ans:
                ok += 1
            else:
                bad.append((fid, q, ans, got))
        return ok, bad

    tr_ok, tr_bad = evaluate(corp["train"])
    va_ok, va_bad = evaluate(corp["val"])
    rep["train_exact"] = f"{tr_ok}/{len(corp['train'])}"
    rep["heldout_exact"] = f"{va_ok}/{len(corp['val'])}"
    rep["train_exact_pct"] = 100 * tr_ok / len(corp["train"])
    rep["heldout_exact_pct"] = 100 * va_ok / len(corp["val"])
    print(f"train exact {rep['train_exact']} ({rep['train_exact_pct']:.1f}%)  held-out exact {rep['heldout_exact']} ({rep['heldout_exact_pct']:.1f}%)")

    # per-category held-out accuracy
    cat_of = {f["id"]: f["category"] for f in corp["facts"]}
    by_cat = defaultdict(lambda: [0, 0])
    for fid, q, ans in corp["val"]:
        by_cat[cat_of[fid]][1] += 1
    wrong = {(fid, q) for fid, q, _, _ in va_bad}
    for fid, q, ans in corp["val"]:
        by_cat[cat_of[fid]][0] += int((fid, q) not in wrong)
    rep["heldout_by_category"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())}
    # facts whose answer is right for at least one held-out / how many facts have all train paraphrases right
    stop_ok = Counter()
    for fid, q, ans in corp["val"]:
        got = answer_of(m, q)
        # whether generation stopped by itself with '$' (a missing stop shows as an over-long answer)
        stop_ok[len(got) <= 18] += 1
    rep["heldout_answers_within_18_chars"] = f"{stop_ok[True]}/{len(corp['val'])}"

    # 4. samples
    rnd = random.Random(a.seed)
    samp = rnd.sample(corp["val"], 15)
    rep["heldout_samples"] = [dict(question=q, expected=ans, model=answer_of(m, q)) for _, q, ans in samp]
    rep["novel"] = [dict(kind=k, question=q, model=answer_of(m, q)) for k, q in NOVEL]
    # a novel question must really be absent from the data
    known = {q for _, q, _ in rows}
    assert not any(q in known for _, q in NOVEL), "NOVEL question is in the data"

    rep["heldout_errors_sample"] = [dict(question=q, expected=e, model=g) for _, q, e, g in va_bad[:30]]
    (model_dir / "eval_report.json").write_text(json.dumps(rep, indent=1))

    lines = ["# doomoracle evaluation", "",
             f"Model `{model_dir.name}`: L={m.L}, d={m.d}, H={m.H}, dm={m.dm}, K={m.K}/{m.KA}/{m.KM}, T={m.T}, vocab `{vocab}`. "
             f"Data: {rep['n_facts']} facts; {rep['n_train']} training paraphrases, {rep['n_heldout']} held-out paraphrases (one per fact).", "",
             "| check | result |", "|---|---|",
             f"| `intref.check_limits` on all {len(seqs)} sequences | `{errs if errs else '[]'}` |",
             f"| torch int64 forward == `intref.forward` | {rep['torch_int_equals_intref']} sequences |",
             f"| exact answer, greedy `intref.generate(m, q + '>')`, TRAIN paraphrases | {rep['train_exact']} = {rep['train_exact_pct']:.1f} % |",
             f"| exact answer, greedy, HELD-OUT paraphrases | {rep['heldout_exact']} = {rep['heldout_exact_pct']:.1f} % |", "",
             "## Held-out accuracy by category", "", "| category | exact |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rep["heldout_by_category"].items()]
    lines += ["", "## 15 held-out samples (random, seed %d)" % a.seed, "", "| question | gold | model | |", "|---|---|---|---|"]
    lines += [f"| {s['question']} | {s['expected']} | {s['model']} | {'ok' if s['model'] == s['expected'] else 'WRONG'} |" for s in rep["heldout_samples"]]
    lines += ["", "## Questions that are not in the data", "", "| kind | question | model says |", "|---|---|---|"]
    lines += [f"| {n['kind']} | {n['question']} | {n['model']} |" for n in rep["novel"]]
    (model_dir / "EVAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
