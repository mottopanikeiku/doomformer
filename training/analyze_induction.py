"""Attention patterns of a trained induction model, in NumPy (no torch), on held-out sequences.

    python training/analyze_induction.py models/induction [--n 2000]

For every layer and head, at every queried token t (a repeat of the first occurrence at j, target at j + 1), the
share of the head's attention weight that lands on
  previous   position t - 1           (what a previous-token head reads)
  source     position j               (the first occurrence of the current token)
  target     position j + 1           (the token that followed it: what an induction head must read)
  bos        position 0
  self       position t
  other      every other position (the first five can coincide, e.g. j + 1 = t - 1, so the columns may sum above 1)
and, over all positions t >= 1 of the sequences, the share on t - 1 ("previous, all t"). The weight of a head is
its integer attention weight a_ts (relu attention) or e_ts (softmax attention), normalised over s <= t; a row with
no weight counts as no attention (listed as "none").

Second table: accuracy at the queried tokens with one head's output zeroed (intref `ablate`, key "L{l}.ao", the
head's d/H units), one head at a time and the two heads of the circuit together.

`causal_scores` gives the same numbers as a function, for the interventions in demos/ and docs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doomformer import induction as I  # noqa: E402
from doomformer import intref  # noqa: E402

COLUMNS = ("previous", "source", "target", "bos", "self", "other", "none")


def attention(qm: intref.QModel, tokens: list[int]) -> np.ndarray:
    """Integer attention weights [L, H, n, n] of one sequence (relu: a, softmax: e), causal."""
    tr = {}
    intref.forward(qm, tokens, tr)
    key = "e" if qm.attn == "softmax" else "a"
    return np.stack([tr[f"L{l}.{key}"] for l in range(qm.L)])


def head_shares(qm: intref.QModel, seqs: np.ndarray, lengths: np.ndarray, tgts: np.ndarray) -> dict:
    """Shares per head (see the module docstring): {"queried": [L, H, len(COLUMNS)], "previous_all": [L, H]}."""
    acc = np.zeros((qm.L, qm.H, len(COLUMNS)))
    nq = 0
    prev_all = np.zeros((qm.L, qm.H))
    n_all = 0
    for row, n, tg in zip(seqs, lengths, tgts):
        toks = row[:n].tolist()
        a = attention(qm, toks).astype(np.float64)                   # [L, H, n, n]
        tot = a.sum(-1, keepdims=True)
        w = np.divide(a, tot, out=np.zeros_like(a), where=tot > 0)
        for t in range(1, n):
            prev_all += w[:, :, t, t - 1]
            n_all += 1
        for t in np.flatnonzero(tg[:n] != I.IGNORE):
            j = toks.index(toks[t])                                  # first occurrence of the current token
            wt = w[:, :, t, :]
            pos = {"previous": t - 1, "source": j, "target": j + 1, "bos": 0, "self": t}
            for c, (name, p) in enumerate(pos.items()):
                acc[:, :, c] += wt[:, :, p]
            union = sorted(set(pos.values()))                        # positions can coincide (j + 1 == t - 1)
            have = tot[:, :, t, 0] > 0
            acc[:, :, 5] += np.where(have, 1.0 - wt[:, :, union].sum(-1), 0.0)
            acc[:, :, 6] += ~have
            nq += 1
    return dict(queried=acc / nq, previous_all=prev_all / n_all, n_queried=nq)


def head_ablation(qm: intref.QModel, seqs, lengths, tgts) -> dict:
    """Accuracy at the queried tokens with heads zeroed: key None nothing, (l, h) one head, (l, None) a whole layer."""
    out = {}
    for key in [None] + [(l, h) for l in range(qm.L) for h in range(qm.H)] + [(l, None) for l in range(qm.L)]:
        units = [] if key is None else (list(range(qm.d)) if key[1] is None else
                                        list(range(key[1] * qm.dh, (key[1] + 1) * qm.dh)))
        ablate = None if key is None else {f"L{key[0]}.ao": units}
        hit = tot = 0
        for row, n, tg in zip(seqs, lengths, tgts):
            sup = tg[:n] != I.IGNORE
            pred = intref.forward(qm, row[:n].tolist(), ablate=ablate).argmax(-1)
            hit += int((pred[sup] == tg[:n][sup]).sum())
            tot += int(sup.sum())
        out[key] = hit / tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=2000, help="held-out sequences")
    a = ap.parse_args()
    qm = intref.QModel.load(a.model)
    s = I.test_sets(qm.T)["test"]
    res = head_shares(qm, s["tok"][: a.n], s["length"][: a.n], s["tgt"][: a.n])
    print(f"{a.model}: L={qm.L} H={qm.H} attn={qm.attn}; {a.n} held-out sequences, {res['n_queried']} queried tokens")
    print(f"{'head':>6} " + " ".join(f"{c:>9}" for c in COLUMNS) + f" {'previous, all t':>16}")
    for l in range(qm.L):
        for h in range(qm.H):
            print(f"{'L%d.h%d' % (l, h):>6} " + " ".join(f"{x:9.3f}" for x in res["queried"][l, h])
                  + f" {res['previous_all'][l, h]:16.3f}")
    n = min(a.n, 500)
    ab = head_ablation(qm, s["tok"][:n], s["length"][:n], s["tgt"][:n])
    print(f"\naccuracy at the queried tokens with one head zeroed ({n} sequences):")
    for key, acc in ab.items():
        name = "none" if key is None else ("L%d (both)" % key[0] if key[1] is None else "L%d.h%d" % key)
        print(f"  {name:>10}  {acc:.4f}")


if __name__ == "__main__":
    main()
