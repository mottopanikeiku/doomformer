"""Quantisation-aware training of "doomoracle": a tiny question-answering transformer (L<=2, d<=32, H=d/8, dm<=96).

Reuses the multi-layer QAT network of train_brain.py (`Net`: float -> act-QAT -> full QAT, integer-exact in stage 3).
Differences to train_brain.py:
* data: training/data/oracle.json (facts, 3-6 question paraphrases each, one held out per fact);
  an example is  BOS + QUESTION + '>' + ANSWER  and the model's job is to continue QUESTION> with ANSWER and '$'.
* loss only on the answer characters and the final '$' (the positions that read '>' and the answer so far),
  while every real position is still tracked for the DOOM partial-sum limits and used for threshold calibration.
* metric: exact answer (teacher-forced argmax == target on every answer position == greedy generation).
* checkpoint selection uses the TRAIN set only (the held-out paraphrases are never used to pick a checkpoint).

Usage: python training/train_oracle.py --out models/doomoracle --steps2 12000 --steps3 4000 --lr2 0.01 --lr3 0.003 --alpha 0.5 --S 64
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from train_brain import IGNORE, Net, loss_fn, record_args, repo_rel  # noqa: E402

ORACLE = HERE / "data" / "oracle.json"
SEP = ">"


# ----------------------------------------------------------------------------- data
def load_oracle(path: Path = ORACLE, facts: set[int] | None = None, dev_per_fact: int = 0, dev_seed: int = 7):
    """-> train / val (the held-out paraphrase of each fact) / dev.  dev (tuning only, default empty) is carved out of train:
    `dev_per_fact` further paraphrases per fact (only facts keeping >= 2 train paraphrases), so that hyper-parameters can be
    tuned without ever looking at `val`."""
    import random
    d = json.loads(Path(path).read_text())
    vocab = d["vocab"]
    assert len(vocab) <= 32 and vocab[0] == "$"
    rnd = random.Random(dev_seed)
    train, val, dev = [], [], []
    for f in d["facts"]:
        if facts is not None and f["id"] not in facts:
            continue
        rest = [q for q in f["questions"] if q != f["heldout"]]
        val.append((f["id"], f["heldout"], f["answer"]))
        k = min(dev_per_fact, len(rest) - 2) if dev_per_fact else 0
        dq = set(rnd.sample(rest, k)) if k > 0 else set()
        for q in rest:
            (dev if q in dq else train).append((f["id"], q, f["answer"]))
    T = max(len(q) + len(a) + 2 for _, q, a in train + val + dev)
    return dict(vocab=vocab, train=train, val=val, dev=dev, T=T, facts=d["facts"], n_facts=len(val))


def pack(vocab: str, rows, T: int):
    """-> tok [N,T] (0 padded), tgt [N,T] (IGNORE except answer chars + final '$'), valid [N,T] (real positions)."""
    tok = torch.zeros(len(rows), T, dtype=torch.long)
    tgt = torch.full((len(rows), T), IGNORE, dtype=torch.long)
    valid = torch.zeros(len(rows), T, dtype=torch.bool)
    for i, (_, q, a) in enumerate(rows):
        x = [0] + [vocab.index(c) for c in q + SEP + a]
        y = x[1:] + [0]
        p = len(q) + 1                                   # index of '>' in x: first position whose output is an answer char
        tok[i, : len(x)] = torch.tensor(x)
        tgt[i, p: len(x)] = torch.tensor(y[p:])
        valid[i, : len(x)] = True
    return tok, tgt, valid


def sequences(vocab: str, rows) -> list[list[int]]:
    return [[0] + [vocab.index(c) for c in q + SEP + a] for _, q, a in rows]


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(net, data, bs=512):
    """Integer-model (quant, hard) answer-token accuracy, exact-answer accuracy and CE on data=(tok,tgt,valid)."""
    saved = (net.quant, net.hard, net.training)
    net.quant, net.hard = True, True
    net.eval()
    tok, tgt, valid = data
    hit = cnt = exact = 0
    ce = 0.0
    worst = {}
    for i in range(0, len(tok), bs):
        t, g, v = tok[i:i + bs], tgt[i:i + bs], valid[i:i + bs]
        n = int(v.sum(1).max())
        t, g, v = t[:, :n], g[:, :n], v[:, :n]
        c, _, logits, w = loss_fn(net, t, g, v)
        m = g != IGNORE
        pred = logits.argmax(-1)
        hit += ((pred == g) & m).sum().item()
        exact += (((pred == g) | ~m).all(1)).sum().item()
        ce += c.item() * m.sum().item()
        cnt += m.sum().item()
        for k, x in w.items():
            worst[k] = max(worst.get(k, 0), int(x))
    net.quant, net.hard = saved[0], saved[1]
    net.train(saved[2])
    return hit / cnt, exact / len(tok), ce / cnt, worst


def augment(rows, vocab, T, p_word, p_char, p_shift, rnd):
    """Train-time noise on the QUESTION only: random word deletion (>= 1 word kept), random character substitution and
    (p_shift) a few leading spaces so that no word sits at a fixed absolute position."""
    out = []
    for fid, q, a in rows:
        w = q.split(" ")
        if p_word > 0 and len(w) > 1:
            keep = [x for x in w if rnd.random() >= p_word]
            w = keep or [rnd.choice(w)]
        q2 = " ".join(w)
        if p_char > 0:
            q2 = "".join(rnd.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") if (c != " " and rnd.random() < p_char) else c for c in q2)
        if p_shift > 0 and rnd.random() < p_shift:                       # position jitter: leading spaces
            k = rnd.choice((1, 2, 3))
            if len(q2) + k + len(a) + 2 <= T:
                q2 = " " * k + q2
        out.append((fid, q2[:28 + 3], a))
    return pack(vocab, out, T)


def run_stage(net, name, steps, lr, quant, hard, alpha, train, evalset, bs, log, lr_end=0.05, eval_every=100,
              keep_best=False, extra_eval=None, aug=None):
    """aug = (rows, vocab, T, p_word, p_char, p_shift): re-pack every batch from `rows` with question noise instead of using `train`."""
    import random
    rnd = random.Random(1234)
    tok, tgt, valid = train
    net.quant, net.hard, net.alpha = quant, hard, alpha
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    best, best_state = (-1.0, 1e9), None
    N = len(tok)
    lens = valid.sum(1)
    batches = []
    for step in range(steps):
        cur = lr * (lr_end + (1 - lr_end) * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        if not batches:                       # length-bucketed shuffled batches: shorter padded length, same data per epoch
            perm = torch.randperm(N)
            for c in range(0, N, bs * 8):
                chunk = perm[c:c + bs * 8]
                chunk = chunk[lens[chunk].argsort()]
                batches += [chunk[b:b + bs] for b in range(0, len(chunk), bs) if len(chunk[b:b + bs]) == bs]
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        idx = batches.pop()
        if aug is not None:
            t, g_, v = augment([aug[0][i] for i in idx.tolist()], *aug[1:], rnd)
        else:
            t, g_, v = tok[idx], tgt[idx], valid[idx]
        n = int(v.sum(1).max())
        t, g_, v = t[:, :n], g_[:, :n], v[:, :n]
        ce, pen, _, _ = loss_fn(net, t, g_, v)
        opt.zero_grad()
        (ce + pen).backward()
        opt.step()
        if step % eval_every == 0 or step == steps - 1:
            acc, ex, ice, worst = evaluate(net, evalset)
            if keep_best and (ex, -ice) > (best[0], -best[1]):
                best, best_state = (ex, ice), copy.deepcopy(net.state_dict()) | {"_units": copy.deepcopy(net.units)}
            if step % (eval_every * 4) == 0 or step == steps - 1:
                xs = ""
                if extra_eval is not None:
                    xs = " | " + " ".join(f"{k} exact {evaluate(net, v_)[1]:.4f}" for k, v_ in extra_eval.items())
                log(f"[{name}] step {step:5d} lr {cur:.4f} train-ce {ce.item():.4f} pen {float(pen):.3f} | "
                    f"train int tok-acc {acc:.4f} exact {ex:.4f} ce {ice:.4f} maxz {max(worst.values())}{xs}")
    if keep_best and best_state is not None:
        units = best_state.pop("_units")
        net.load_state_dict(best_state)
        net.units = units
        log(f"[{name}] restored best TRAIN int exact {best[0]:.4f} ce {best[1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "models" / "doomoracle"))
    ap.add_argument("--data", default=str(ORACLE))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--dm", type=int, default=96)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--KA", type=int, default=7)
    ap.add_argument("--KM", type=int, default=7)
    ap.add_argument("--S", type=float, default=64.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps1", type=int, default=0)
    ap.add_argument("--steps2", type=int, default=12000)
    ap.add_argument("--steps3", type=int, default=4000)
    ap.add_argument("--lr1", type=float, default=0.01)
    ap.add_argument("--lr2", type=float, default=0.01)
    ap.add_argument("--lr3", type=float, default=0.003)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dev-per-fact", type=int, default=0, help="tuning: carve this many train paraphrases per fact into a dev set")
    ap.add_argument("--p-word", type=float, default=0.0, help="augmentation: word-deletion probability on train questions")
    ap.add_argument("--p-char", type=float, default=0.0, help="augmentation: character substitution probability")
    ap.add_argument("--p-shift", type=float, default=0.0, help="augmentation: probability of 1-3 leading spaces (position jitter)")
    ap.add_argument("--drop", type=float, default=0.0, help="activation dropout")
    ap.add_argument("--max-facts", type=int, default=0, help="use only the first N facts (pilot runs)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(getattr(torch, a.dtype))
    torch.manual_seed(a.seed)
    H = a.d // 8
    only = set(range(a.max_facts)) if a.max_facts else None
    corp = load_oracle(Path(a.data), only, a.dev_per_fact)
    vocab, T = corp["vocab"], corp["T"]
    assert a.L <= 2 and a.d <= 32 and a.d % 8 == 0 and a.dm <= 96 and T <= 48
    assert max(a.K, a.KA, a.KM) <= 7 and len(vocab) <= 32

    t0 = time.time()

    def log(s):
        print(f"{time.time() - t0:7.1f}s {s}", flush=True)

    train, val = pack(vocab, corp["train"], T), pack(vocab, corp["val"], T)
    dev = pack(vocab, corp["dev"], T) if corp["dev"] else None
    extra = {"dev": dev} if dev is not None else None
    aug = (corp["train"], vocab, T, a.p_word, a.p_char, a.p_shift) if (a.p_word or a.p_char or a.p_shift) else None
    net = Net(len(vocab), T, a.d, H, a.dm, a.K, a.KA, a.KM, a.L, a.S, a.seed, a.drop)
    log(f"oracle: {corp['n_facts']} facts; train {len(corp['train'])} val(held-out) {len(corp['val'])}; T={T} "
        f"vocab {len(vocab)}; params {sum(p.numel() for p in net.parameters())}")

    # calibrate thresholds from observed pre-activations on a train sample
    net.quant, net.hard, net.calibrating = False, True, True
    sub = torch.randperm(len(train[0]))[:600]
    tk, _, vd = (x[sub] for x in train)
    net.valid = vd
    net.eval()
    with torch.no_grad():
        net(tk, vd)
    net.calibrating = False

    run_stage(net, "float", a.steps1, a.lr1, False, False, a.alpha, train, train, a.bs, log, extra_eval=extra, aug=aug)
    run_stage(net, "act-qat", a.steps2, a.lr2, False, True, a.alpha, train, train, a.bs, log, extra_eval=extra, aug=aug)
    run_stage(net, "full-qat", a.steps3, a.lr3, True, True, a.alpha, train, train, a.bs, log, keep_best=True, extra_eval=extra, aug=aug)

    qm = net.qmodel(vocab)
    out = Path(a.out)
    qm.save(out)
    tr_acc, tr_ex, tr_ce, worst = evaluate(net, train)
    tuning = dev is not None                          # tuning runs never touch the held-out (val) paraphrases
    va_acc, va_ex, va_ce, _ = (None, None, None, None) if tuning else evaluate(net, val)
    dv = evaluate(net, dev)[:3] if tuning else None
    (out / "train_info.json").write_text(json.dumps(
        dict(train_tok_acc=tr_acc, train_exact=tr_ex, val_tok_acc=va_acc, val_exact_teacher_forced=va_ex, dev=dv,
             seed=a.seed, wall_s=time.time() - t0, args=record_args(a), n_train=len(corp["train"]), n_val=len(corp["val"]),
             n_facts=corp["n_facts"], T=T, worst_partial_sums=worst), indent=1))
    log(f"final torch-int train: tok-acc {tr_acc:.4f} exact {tr_ex:.4f} | "
        + (f"dev tok-acc {dv[0]:.4f} exact {dv[1]:.4f}" if tuning else f"held-out: tok-acc {va_acc:.4f} exact {va_ex:.4f}"))
    log(f"saved {repo_rel(a.out)}")


if __name__ == "__main__":
    main()
