"""Train the integer transformer on the induction task (doomformer.induction), and its float baseline.

    python training/train_algo.py --out models/induction                      # the shipped recipe (relu attention)
    python training/train_algo.py --out models/induction_softmax --attn softmax --R 2
    python training/train_algo.py --float --out build/induction_float         # an ordinary float transformer

The integer model is `train_brain.Net` (the recipe of the name models: float stage with smooth staircases, act-QAT,
full QAT with rounded weights and thresholds) trained on a stream of fresh random sequences. No sequence of the
held-out sets (test, unseen, dev) is ever drawn, so the held-out accuracies in train_info.json are on sequences the
model has not seen (doomformer/induction.py). The dev set picks the best full-QAT checkpoint.

--float trains a standard pre-LayerNorm GPT with softmax attention and GELU (same L, d, H, dm, T, same data) in float32,
the baseline for "float vs integer".

Every integer model is checked at the end: the exported integers, run through the pure int64 torch forward
(`train_brain.int_forward`), give the same logits as `intref.forward` on 2,000 held-out sequences, and
`intref.check_limits` is clean on 600 of them.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import train_brain as tb  # noqa: E402
from doomformer import induction as I  # noqa: E402
from doomformer import intref  # noqa: E402

IGNORE = I.IGNORE
PLIM, FLIM = tb.PLIM, tb.FLIM


# ----------------------------------------------------------------------------- data
def to_torch(tok, tgt, length, t: int):
    tok, tgt = torch.as_tensor(tok), torch.as_tensor(tgt)
    live = torch.arange(t)[None] < torch.as_tensor(length)[:, None]
    return tok, tgt, live


def metrics(pred: torch.Tensor, tgt: torch.Tensor) -> dict:
    """Accuracy over the queried tokens, at the first queried token of each sequence, and whole sequences."""
    sup = tgt != IGNORE
    hit = (pred == tgt) & sup
    first = torch.zeros_like(sup)
    first[torch.arange(len(sup)), sup.int().argmax(1)] = True
    first &= sup
    return dict(tok_acc=hit.sum().item() / sup.sum().item(), first_acc=(hit & first).sum().item() / first.sum().item(),
                seq_acc=((pred == tgt) | ~sup).all(1).double().mean().item(), n=len(tgt))


# ----------------------------------------------------------------------------- float baseline
class FloatGPT(nn.Module):
    """Pre-LayerNorm decoder: learned token and position embeddings, softmax attention, GELU MLP, untied head."""

    def __init__(self, V, T, d, H, dm, L):
        super().__init__()
        self.H = H
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d), qkv=nn.Linear(d, 3 * d), proj=nn.Linear(d, d), ln2=nn.LayerNorm(d),
            up=nn.Linear(d, dm), dn=nn.Linear(dm, d))) for _ in range(L))
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, V)

    def forward(self, tok):
        B, n = tok.shape
        x = self.tok(tok) + self.pos(torch.arange(n))[None]
        for b in self.blocks:
            q, k, v = b["qkv"](b["ln1"](x)).reshape(B, n, 3, self.H, -1).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, n, -1)
            x = x + b["proj"](a)
            x = x + b["dn"](F.gelu(b["up"](b["ln2"](x))))
        return self.head(self.lnf(x))


def train_float(a, log):
    t = a.T
    net = FloatGPT(len(I.VOCAB), t, a.d, a.H, a.dm, a.L)
    n_par = sum(p.numel() for p in net.parameters())
    log(f"float GPT: L={a.L} d={a.d} H={a.H} dm={a.dm} T={t}, {n_par} parameters")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr_float, weight_decay=0.01)
    sets, rng = I.test_sets(t), np.random.default_rng(a.seed)
    for step in range(a.steps_float):
        cur = a.lr_float * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / a.steps_float)))
        for g in opt.param_groups:
            g["lr"] = cur
        tok, tgt, _ = to_torch(*I.train_batch(rng, a.bs, t), t)
        loss = F.cross_entropy(net(tok).reshape(-1, len(I.VOCAB)), tgt.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == a.steps_float - 1:
            m = float_eval(net, sets["dev"], t)
            log(f"[float] step {step:5d} lr {cur:.4f} loss {loss.item():.4f} dev acc {m['tok_acc']:.4f} "
                f"first {m['first_acc']:.4f}")
    res = {name: float_eval(net, sets[name], t) for name in ("test", "unseen", "dev")}
    return net, res, n_par


@torch.no_grad()
def float_eval(net, s, t):
    tok, tgt, _ = to_torch(s["tok"], s["tgt"], s["length"], t)
    net.eval()
    out = metrics(net(tok).argmax(-1), tgt)
    net.train()
    return out


# ----------------------------------------------------------------------------- integer model
def loss_fn(net, tok, tgt, live):
    logits, worst, fin = net(tok, live)
    ce = F.cross_entropy((logits * net.lg.exp()).reshape(-1, net.V), tgt.reshape(-1), ignore_index=IGNORE)
    pen = (sum(F.relu(w - PLIM) for w in worst.values()) + sum(F.relu(w - FLIM) for w in fin.values())) / 1000.0
    return ce, pen, logits, {**worst, **{"fin_" + k: v for k, v in fin.items()}}


@torch.no_grad()
def evaluate(net, s, t):
    """Integer-model (quant, hard) metrics on a set dict; also the cross-entropy and the worst partial sums."""
    saved = (net.quant, net.hard, net.training)
    net.quant, net.hard = True, True
    net.eval()
    tok, tgt, live = to_torch(s["tok"], s["tgt"], s["length"], t)
    ce, _, logits, worst = loss_fn(net, tok, tgt, live)
    net.quant, net.hard = saved[:2]
    net.train(saved[2])
    return dict(metrics(logits.argmax(-1), tgt), ce=ce.item(), maxz=max(int(v) for v in worst.values()))


def run_stage(net, name, steps, lr, quant, hard, alpha, a, rng, dev, log, keep_best=False, lr_end=0.05):
    net.quant, net.hard, net.alpha = quant, hard, alpha
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    best, best_state = (-1.0, 1e9), None
    for step in range(steps):
        cur = lr * (lr_end + (1 - lr_end) * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        tok, tgt, live = to_torch(*I.train_batch(rng, a.bs, a.T), a.T)
        ce, pen, _, _ = loss_fn(net, tok, tgt, live)
        opt.zero_grad()
        (ce + pen).backward()
        opt.step()
        if step % a.eval_every == 0 or step == steps - 1:
            m = evaluate(net, dev, a.T)
            if keep_best and (m["tok_acc"], -m["ce"]) > (best[0], -best[1]):
                best, best_state = (m["tok_acc"], m["ce"]), copy.deepcopy(net.state_dict()) | {"_units": copy.deepcopy(net.units)}
            if step % (a.eval_every * 5) == 0 or step == steps - 1:
                log(f"[{name}] step {step:5d} lr {cur:.4f} train-ce {ce.item():.4f} pen {float(pen):.3f} | dev int acc "
                    f"{m['tok_acc']:.4f} first {m['first_acc']:.4f} ce {m['ce']:.4f} maxz {m['maxz']}")
    if keep_best and best_state is not None:
        units = best_state.pop("_units")
        net.load_state_dict(best_state)
        net.units = units
        log(f"[{name}] restored best dev int acc {best[0]:.4f} ce {best[1]:.4f}")


def int_logits(qm: intref.QModel, tok: np.ndarray) -> torch.Tensor:
    """Pure int64 torch forward of the exported integers (an independent evaluation of the contract)."""
    return tb.int_forward(qm.p, torch.as_tensor(tok), qm.H, qm.L, qm.attn, qm.R, qm.norm, qm.Rn, qm.mlp)


def int_report(qm: intref.QModel, t: int, n_check: int = 2000, n_limits: int = 600) -> dict:
    """Held-out metrics of the exported integer model (int64 torch forward), the torch == intref check and
    check_limits. Everything a model's EVAL.md quotes."""
    sets = I.test_sets(t)
    out = {}
    for name, s in sets.items():
        pred = int_logits(qm, s["tok"]).argmax(-1)
        out[name] = metrics(pred, torch.as_tensor(s["tgt"]))
    tok, length = sets["test"]["tok"][:n_check], sets["test"]["length"][:n_check]
    lg = int_logits(qm, tok).numpy()
    bad = sum(not np.array_equal(lg[i, :n], intref.forward(qm, tok[i, :n].tolist())) for i, n in enumerate(length))
    out["torch_vs_intref"] = dict(sequences=len(tok), differing=int(bad))
    seqs = [row[:n].tolist() for row, n in zip(sets["test"]["tok"][:n_limits], sets["test"]["length"][:n_limits])]
    out["check_limits"] = intref.check_limits(qm, seqs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "models" / "induction"))
    ap.add_argument("--float", action="store_true", help="train the float GPT baseline instead of the integer model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=I.T)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--dm", type=int, default=64)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--KA", type=int, default=None, help="default: K")
    ap.add_argument("--KM", type=int, default=None, help="default: K")
    ap.add_argument("--S", type=float, default=64.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps1", type=int, default=1500)
    ap.add_argument("--steps2", type=int, default=2000)
    ap.add_argument("--steps3", type=int, default=3000)
    ap.add_argument("--lr1", type=float, default=0.01)
    ap.add_argument("--lr2", type=float, default=0.005)
    ap.add_argument("--lr3", type=float, default=0.002)
    ap.add_argument("--alpha", type=float, default=0.5, help="surrogate sigmoid width / threshold step")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--recal", action="store_true", help="re-calibrate thresholds after the float stage")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--attn", choices=("relu", "softmax"), default="relu")
    ap.add_argument("--R", type=int, default=1, help="softmax only: output resolution")
    ap.add_argument("--norm", choices=("none", "rms"), default="none")
    ap.add_argument("--Rn", type=int, default=1)
    ap.add_argument("--mlp", choices=("relu", "reglu"), default="relu")
    ap.add_argument("--steps-float", dest="steps_float", type=int, default=6000)
    ap.add_argument("--lr-float", dest="lr_float", type=float, default=0.003)
    a = ap.parse_args()
    a.KA, a.KM = a.KA or a.K, a.KM or a.K
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(getattr(torch, a.dtype))
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, lines = time.time(), []

    def log(s):
        line = f"{time.time() - t0:7.1f}s {s}"
        lines.append(line)
        print(line, flush=True)

    sets = I.test_sets(a.T)
    log(f"task: induction, T={a.T}, vocab {len(I.VOCAB)}; held out: " + ", ".join(f"{k} {len(v['tok'])}" for k, v in
                                                                                  sets.items()))
    if a.float:
        net, res, n_par = train_float(a, log)
        info = dict(kind="float GPT", params=n_par, seed=a.seed, wall_s=time.time() - t0, args=tb.record_args(a), **res)
        (out / "float_info.json").write_text(json.dumps(info, indent=1))
        (out / "float_log.txt").write_text("\n".join(lines) + "\n")
        log("float held-out: " + ", ".join(f"{k} {v['tok_acc']:.4f} (first {v['first_acc']:.4f})" for k, v in res.items()))
        return

    rng = np.random.default_rng(a.seed)
    dev = sets["dev"]
    net = tb.Net(len(I.VOCAB), a.T, a.d, a.H, a.dm, a.K, a.KA, a.KM, a.L, a.S, a.seed, 0.0, attn=a.attn, R=a.R,
                 norm=a.norm, Rn=a.Rn, mlp=a.mlp)
    n_par = sum(p.numel() for p in net.parameters() if p.dim() > 0 and p is not net.lg) + 0
    log(f"integer model: L={a.L} d={a.d} H={a.H} dm={a.dm} K={a.K}/{a.KA}/{a.KM} attn={a.attn} norm={a.norm} "
        f"mlp={a.mlp}; {sum(p.numel() for p in net.p.values())} weights")

    def calibrate():
        net.quant, net.hard, net.calibrating = False, True, True
        tok, tgt, live = to_torch(*I.train_batch(rng, 1024, a.T), a.T)
        net.valid = live
        net.eval()
        with torch.no_grad():
            net(tok, live)
        net.calibrating = False

    calibrate()
    run_stage(net, "float", a.steps1, a.lr1, False, False, a.alpha, a, rng, dev, log)
    net.quant, net.hard = False, False
    net.eval()
    with torch.no_grad():
        tok, tgt, live = to_torch(dev["tok"], dev["tgt"], dev["length"], a.T)
        lg, _, _ = net(tok, live)
    fm = metrics(lg.argmax(-1), tgt)
    log(f"smooth-staircase stage (unrounded weights) dev acc {fm['tok_acc']:.4f} first {fm['first_acc']:.4f}")
    if a.recal:
        calibrate()
    run_stage(net, "act-qat", a.steps2, a.lr2, False, True, a.alpha, a, rng, dev, log)
    run_stage(net, "full-qat", a.steps3, a.lr3, True, True, a.alpha, a, rng, dev, log, keep_best=True)

    qm = net.qmodel(I.VOCAB)
    qm.save(out)
    rep = int_report(qm, a.T)
    smooth = {}
    net.quant, net.hard = False, False
    net.eval()
    with torch.no_grad():
        for name in ("test", "unseen"):
            s = sets[name]
            tok, tgt, live = to_torch(s["tok"], s["tgt"], s["length"], a.T)
            net.quant, net.hard = False, False
            smooth[name] = metrics(net(tok, live)[0].argmax(-1), tgt)
    info = dict(smooth_stage_unrounded=smooth, held_out=rep, seed=a.seed, wall_s=time.time() - t0,
                int_logit_scale=float(net.lg.exp()), args=tb.record_args(a), **net.arch_info(), **net.attn_info())
    (out / "train_info.json").write_text(json.dumps(info, indent=1))
    log("integer held-out: " + ", ".join(f"{k} {rep[k]['tok_acc']:.4f} (first {rep[k]['first_acc']:.4f})"
                                         for k in ("test", "unseen", "dev")))
    log(f"torch int64 vs intref: {rep['torch_vs_intref']}; check_limits: {rep['check_limits'] or 'clean'}")
    log(f"saved {tb.repo_rel(a.out)}")
    (out / "train_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
