"""Held-out evaluation of induction models and the trivial baselines, one table.

    python training/eval_algo.py models/induction models/induction_softmax [--float runs/float/float_info.json]
                                 [--fit 400000] [--limits] [--json out.json]

Rows: count tables and heuristics (fit on `--fit` fresh training sequences), the float GPT (its float_info.json from
`train_algo.py --float`), the smooth-staircase stage of each integer model (unrounded weights; train_info.json) and
each integer model (int64 torch forward of the exported integers, equal to intref.forward). Columns, per held-out set:
accuracy over all queried tokens and at the first queried token of each sequence (the pure look-back).
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

import train_algo as ta  # noqa: E402
from doomformer import induction as I  # noqa: E402
from doomformer import intref  # noqa: E402


def label(path: Path, qm: intref.QModel) -> str:
    return f"{path.name}: {qm.attn} attention, K={qm.K}/{qm.KA}/{qm.KM}" + (f", R={qm.R}" if qm.attn == "softmax" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--float", dest="float_info", action="append", default=[], help="float_info.json of a float GPT")
    ap.add_argument("--fit", type=int, default=400_000, help="training sequences the count tables are fit on")
    ap.add_argument("--T", type=int, default=I.T, help="context length of the task (the models must have it)")
    ap.add_argument("--limits", action="store_true", help="also run intref.check_limits on every held-out sequence")
    ap.add_argument("--json", help="write the numbers here")
    a = ap.parse_args()
    t = a.T
    sets = I.test_sets(t)
    rows = []                                                        # (name, {set: (tok_acc, first_acc)})
    train = I.train_batch(np.random.default_rng(99), a.fit, t)
    per_set = {name: I.baselines(train[:2], sets[name]) for name in ("test", "unseen")}
    for (key, a_res), (_, b_res) in zip(per_set["test"].items(), per_set["unseen"].items()):   # best lag may differ
        rows.append((key, {"test": a_res, "unseen": b_res}))
    for f in a.float_info:
        info = json.loads(Path(f).read_text())
        rows.append((f"float GPT, d={info['args']['d']}, softmax attention ({info['params']:,} parameters)",
                     {n: (info[n]["tok_acc"], info[n]["first_acc"]) for n in ("test", "unseen")}))
    for m in a.models:
        path = Path(m)
        qm = intref.QModel.load(path)
        if qm.T != t:
            raise SystemExit(f"{m}: context {qm.T}, but the task is evaluated at --T {t}")
        info = json.loads((path / "train_info.json").read_text())
        sm = info["smooth_stage_unrounded"]
        rows.append((label(path, qm) + ", smooth staircase stage (unrounded)",
                     {n: (sm[n]["tok_acc"], sm[n]["first_acc"]) for n in ("test", "unseen")}))
        res = {}
        for n in ("test", "unseen"):
            s = sets[n]
            met = ta.metrics(ta.int_logits(qm, s["tok"]).argmax(-1), torch.as_tensor(s["tgt"]))
            res[n] = (met["tok_acc"], met["first_acc"])
        rows.append((label(path, qm) + ", integer", res))
        if a.limits:
            seqs = [r[:n].tolist() for k in ("test", "unseen") for r, n in zip(sets[k]["tok"], sets[k]["length"])]
            print(f"check_limits {path.name}: {len(seqs)} held-out sequences: {intref.check_limits(qm, seqs) or 'clean'}")
    w = max(len(r[0]) for r in rows)
    print(f"held-out sets: test {len(sets['test']['tok'])} random sequences, unseen {len(sets['unseen']['tok'])} "
          f"sequences whose queried token pairs never occur in training; count tables fit on {a.fit:,} sequences")
    print(f"{'model':<{w}}  {'test':>8} {'first':>8}  {'unseen':>8} {'first':>8}")
    for name, res in rows:
        print(f"{name:<{w}}  {res['test'][0]:8.4f} {res['test'][1]:8.4f}  {res['unseen'][0]:8.4f} {res['unseen'][1]:8.4f}")
    if a.json:
        Path(a.json).write_text(json.dumps({name: res for name, res in rows}, indent=1))


if __name__ == "__main__":
    main()
