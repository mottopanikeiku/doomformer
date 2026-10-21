"""Quantisation-aware training of the bigger DOOM-name transformer ("doombrain", L<=2, d<=32, H=d/8).

Adapts train_qat.py to the L-layer QModel format ("L{l}.<name>"), the real-level-name corpus
(training/data/levels.json, field `clean`) and mini-batch training.  The torch model mirrors
`doomformer.intref.forward` exactly; every tensor is an integer-valued float64 tensor when
`quant=True, hard=True`, so the same code is the integer model.

Stages (as in train_qat): 1 float (smooth staircases) -> 2 act-QAT (hard staircases, float weights)
-> 3 full QAT (weights, biases, thresholds rounded, straight-through).  Thresholds are cumulative sums
of positive increments (positive and nondecreasing by construction) calibrated from activation quantiles.

Split (fixed seed): 10 % of the (deduplicated) names are `val` (touched only for reporting), `dev`
names (taken from the rest) select checkpoints / hyper-parameters, the rest is `train`.

Usage: python training/train_brain.py --out models/doombrain [--seed 0] [--L 2 --d 32 --dm 96]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
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

from doomformer import intref  # noqa: E402

Z_MAX, W_MAX, PORT_MAX = intref.Z_MAX, intref.W_MAX, intref.PORT_MAX
PLIM = 0.9 * Z_MAX          # training-time ceiling for partial sums (contract limit is Z_MAX)
FLIM = 0.9 * PORT_MAX       # training-time ceiling for final z feeding gates (contract limit PORT_MAX)
IGNORE = -100
BASE_VOCAB = "$ ABCDEFGHIJKLMNOPQRSTUVWXYZ'!"
LEVELS = HERE / "data" / "levels.json"
SPLIT_SEED = 12345
ROOT = HERE.parent


def repo_rel(path) -> str:
    """`path` relative to the repository root when it lies inside it, so run records hold no local absolute paths."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def record_args(a) -> dict:
    """`vars(a)` for train_info.json, with absolute paths inside the repository made repo-relative."""
    return {k: repo_rel(v) if isinstance(v, str) and Path(v).is_absolute() else v for k, v in vars(a).items()}


# ----------------------------------------------------------------------------- data
def load_corpus(path: Path = LEVELS, dev: int = 150, seed: int = SPLIT_SEED):
    """-> dict(names, vocab, train, dev, val, T).  Deterministic, deduplicated, 10 % val."""
    names = None
    if Path(path).exists():
        rows = json.loads(Path(path).read_text())
        names = sorted({r["clean"] for r in rows if r.get("clean")})
        source = repo_rel(path)
    if names is None:
        from data import NAMES
        names, source = sorted(set(NAMES)), "training/data.py NAMES"
    extra = sorted({c for n in names for c in n} - set(BASE_VOCAB))
    vocab = BASE_VOCAB + "".join(extra)
    assert len(vocab) <= 32, vocab
    rnd = random.Random(seed)
    order = names[:]
    rnd.shuffle(order)
    nval = round(0.1 * len(order))
    val, rest = order[:nval], order[nval:]
    dev = min(dev, len(rest) // 8)
    return dict(names=names, vocab=vocab, val=val, dev=rest[:dev], train=rest[dev:],
                T=max(len(n) for n in names) + 1, source=source)


def encode(vocab: str, name: str) -> list[int]:
    return [0] + [vocab.index(c) for c in name]


def encode_target(vocab: str, name: str) -> list[int]:
    return [vocab.index(c) for c in name] + [0]


def make_batch(vocab: str, names: list[str], T: int):
    tok = torch.zeros(len(names), T, dtype=torch.long)
    tgt = torch.full((len(names), T), IGNORE, dtype=torch.long)
    for i, n in enumerate(names):
        x, y = encode(vocab, n), encode_target(vocab, n)
        tok[i, : len(x)] = torch.tensor(x)
        tgt[i, : len(y)] = torch.tensor(y)
    return tok, tgt


def round_ste(x):
    return x + (torch.round(x) - x).detach()


def inv_softplus(y):
    return math.log(math.expm1(y))


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    SITES = {  # name -> (levels attr, relu?)
        "q": ("K", False), "k": ("K", False), "v": ("K", False), "a": ("KA", True), "o": ("K", False),
        "x1": ("K", False), "m": ("KM", True), "x2": ("K", False),
    }

    def __init__(self, V, T, d, H, dm, K, KA, KM, L, S=128.0, seed=0, drop=0.0):
        super().__init__()
        self.V, self.T, self.d, self.H, self.dm, self.K, self.KA, self.KM, self.L = V, T, d, H, dm, K, KA, KM, L
        self.dh, self.S, self.drop = d // H, S, drop
        g = torch.Generator().manual_seed(seed)

        def nrm(*shape, std):
            return nn.Parameter(torch.randn(*shape, generator=g) * std)

        p = nn.ParameterDict()
        p["tok_emb"], p["pos_emb"] = nrm(V, d, std=1.0), nrm(T, d, std=0.7)
        for l in range(L):
            for w in ("Wq", "Wk", "Wv", "Wo"):
                p[f"{l}_{w}"] = nrm(d, d, std=0.3)
            for b in ("bq", "bk", "bv", "bo", "bdn"):
                p[f"{l}_{b}"] = nn.Parameter(torch.zeros(d))
            p[f"{l}_Wup"], p[f"{l}_bup"] = nrm(dm, d, std=0.3), nn.Parameter(torch.zeros(dm))
            p[f"{l}_Wdn"] = nrm(d, dm, std=0.3 * math.sqrt(d / dm))
            p[f"{l}_r1"], p[f"{l}_r2"] = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        p["Wun"], p["bun"] = nrm(V, d, std=0.3), nn.Parameter(torch.zeros(V))
        self.p = p
        self.raw = nn.ParameterDict()
        self.units = {}
        for site in ["x0"] + [f"{l}_{n}" for l in range(L) for n in self.SITES]:
            kk = getattr(self, "K" if site == "x0" else self.SITES[site.split("_")[1]][0])
            per_pos = site.endswith("_o")
            shape = (T, kk) if per_pos else (kk,)
            self.raw[site] = nn.Parameter(torch.full(shape, inv_softplus(1.0)))
            self.units[site] = torch.full(shape[:-1] + (1,), float(S))
        self.lg = nn.Parameter(torch.tensor(math.log(1.0 / S)))      # float logits = exp(lg) * int logits
        self.quant, self.hard, self.alpha = False, False, 0.5
        self.calibrating, self.valid = False, None

    # --- integer-valued parameters
    def W(self, name):
        w = (self.p[name] * self.S).clamp(-W_MAX, W_MAX)
        return round_ste(w) if self.quant else w

    def thr(self, site):
        inc = self.units[site] * F.softplus(self.raw[site])
        t = inc.cumsum(-1).clamp(1, PORT_MAX)
        return (round_ste(t) if self.quant else t), inc.detach()

    @staticmethod
    def relu_site(site):
        return site.split("_")[-1] in ("a", "m")

    # --- calibration of a staircase from the observed pre-activation
    @torch.no_grad()
    def calibrate(self, site, z):
        kk = getattr(self, "K" if site == "x0" else self.SITES[site.split("_")[1]][0])
        relu, valid, n = self.relu_site(site), self.valid, z.shape[1] if site.endswith("_o") else None
        if site.endswith("_a"):
            nn_ = z.shape[-1]
            m = valid[:, None, :, None] & valid[:, None, None, :] & (torch.tril(torch.ones(nn_, nn_)) > 0)[None, None]
            m = m.expand_as(z)
        else:
            m = valid[:, :, None].expand_as(z)

        def step_of(sample):
            sample = sample[sample > 0] if relu else sample.abs()
            if sample.numel() < 8:
                return None
            q = torch.quantile(sample[:2_000_000], 0.985 if not relu else 0.97).item()
            return max(q / (kk - 0.5), 2.0)

        if site.endswith("_o"):
            steps, prev = [], None
            for t in range(self.T):
                s = step_of(z[:, t][m[:, t]]) if t < z.shape[1] else None
                if s is None:
                    s = (prev * (t + 1) / t) if prev is not None else float(self.S)
                steps.append(s)
                prev = s
            self.units[site] = torch.tensor(steps)[:, None]
        else:
            self.units[site] = torch.tensor([step_of(z[m]) or float(self.S)])
        init = torch.ones(kk)
        init[0] = 0.5
        self.raw[site].copy_(torch.log(torch.expm1(init)).expand_as(self.raw[site]))

    # --- staircase with sigmoid surrogate gradient
    def stair(self, site, z, relu=False):
        if self.calibrating:
            self.calibrate(site, z)
        thr, inc = self.thr(site)
        tau = (self.alpha * inc).clamp(min=1.0)
        if site.endswith("_o"):
            n = z.shape[1]
            thr, tau = thr[None, :n, None, :], tau[None, :n, None, :]

        def count(u):
            u = u[..., None]
            soft = torch.sigmoid((u - thr) / tau).sum(-1)
            if self.hard:
                return soft + ((u >= thr).to(soft.dtype).sum(-1) - soft).detach()
            return soft

        out = count(z) if relu else count(z) - count(-z)
        return out

    def dropout(self, x):
        if self.training and self.drop > 0:
            return x * (torch.rand_like(x) >= self.drop).to(x.dtype)
        return x

    # --- forward; returns integer-valued logits and per-site worst partial sums
    def forward(self, tok, valid=None):
        B, n = tok.shape
        H, dh, d = self.H, self.dh, self.d
        worst, fin = {}, {}
        mask = valid[:, :, None] if valid is not None else None

        def track(key, hi, lo):
            m = torch.maximum(hi.abs(), lo.abs())
            if mask is not None:
                m = m.masked_fill(~mask, 0)
            worst[key] = torch.maximum(worst[key], m.max()) if key in worst else m.max()

        def final(key, z):
            m = z.abs()
            if mask is not None:
                m = m.masked_fill(~mask, 0)
            fin[key] = m.max()

        def lin(key, x, Wn, bn, extra=None):
            Wm, b = self.W(Wn), self.W(bn)
            xp, xn, Wp, Wq = x.clamp(min=0), x.clamp(max=0), Wm.clamp(min=0), Wm.clamp(max=0)
            z = x @ Wm.T + b
            hi = xp @ Wp.T + xn @ Wq.T + b
            lo = xp @ Wq.T + xn @ Wp.T + b
            if extra is not None:
                z = z + extra
                hi, lo = hi + extra.clamp(min=0), lo + extra.clamp(max=0)
            if key != "logits":
                final(key, z)
            track(key, hi, lo)
            return z

        E, P = self.W("tok_emb"), self.W("pos_emb")
        z0 = E[tok] + P[None, :n]
        x = self.dropout(self.stair("x0", z0))
        tril = torch.tril(torch.ones(n, n))[None, None]
        for l in range(self.L):
            s = lambda nme: f"{l}_{nme}"
            zq, zk, zv = (lin(f"{l}.z{u}", x, s("W" + u), s("b" + u)) for u in "qkv")
            q, k, v = self.stair(s("q"), zq), self.stair(s("k"), zk), self.stair(s("v"), zv)
            qh, kh, vh = (u.reshape(B, n, H, dh) for u in (q, k, v))
            score = torch.einsum("bthd,bshd->bhts", qh, kh)
            a = self.stair(s("a"), score, relu=True) * tril
            zo = torch.einsum("bhts,bshe->bthe", a, vh).reshape(B, n, d)
            ao = self.dropout(self.stair(s("o"), zo))
            z1 = lin(f"{l}.z1", ao, s("Wo"), s("bo"), extra=self.W(s("r1"))[None, None] * x)
            x1 = self.dropout(self.stair(s("x1"), z1))
            zm = lin(f"{l}.zm", x1, s("Wup"), s("bup"))
            m = self.dropout(self.stair(s("m"), zm, relu=True))
            z2 = lin(f"{l}.z2", m, s("Wdn"), s("bdn"), extra=self.W(s("r2"))[None, None] * x1)
            x = self.dropout(self.stair(s("x2"), z2))
        logits = lin("logits", x, "Wun", "bun")
        return logits, worst, fin

    # --- export
    @torch.no_grad()
    def export(self) -> dict[str, np.ndarray]:
        q0 = self.quant
        self.quant = True
        out = {}
        for n in self.p:
            name = n if "_" not in n or n in ("tok_emb", "pos_emb") else f"L{n.split('_')[0]}.{n.split('_', 1)[1]}"
            out[name] = self.W(n).numpy().astype(np.int64)
        for site in self.raw:
            name = "thr_x0" if site == "x0" else f"L{site.split('_')[0]}.thr_{site.split('_', 1)[1]}"
            out[name] = self.thr(site)[0].numpy().astype(np.int64)
        self.quant = q0
        return out

    def qmodel(self, vocab) -> intref.QModel:
        return intref.QModel(vocab=vocab, T=self.T, d=self.d, H=self.H, dm=self.dm, K=self.K, KA=self.KA,
                             KM=self.KM, L=self.L, p=self.export())


# ----------------------------------------------------------------------------- standalone int64 torch forward
def int_forward(p: dict, tok: torch.Tensor, H: int, L: int) -> torch.Tensor:
    """Pure int64 torch evaluation of the contract from exported integers. tok [B,n] long."""
    p = {k: torch.as_tensor(v, dtype=torch.int64) for k, v in p.items()}
    B, n = tok.shape
    d = p["tok_emb"].shape[1]
    dh = d // H

    def relu(z, thr):
        return (z[..., None] >= thr).sum(-1)

    def sgn(z, thr):
        return relu(z, thr) - relu(-z, thr)

    x = sgn(p["tok_emb"][tok] + p["pos_emb"][None, :n], p["thr_x0"])
    tril = torch.tril(torch.ones(n, n, dtype=torch.int64))
    for l in range(L):
        g = lambda nme: p[f"L{l}.{nme}"]
        q, k, v = (sgn(x @ g("W" + u).T + g("b" + u), g("thr_" + u)) for u in "qkv")
        qh, kh, vh = (u.reshape(B, n, H, dh) for u in (q, k, v))
        score = torch.einsum("bthd,bshd->bhts", qh, kh)
        a = relu(score, g("thr_a")) * tril
        zo = torch.einsum("bhts,bshe->bthe", a, vh).reshape(B, n, d)
        ao = sgn(zo, g("thr_o")[None, :n, None, :])
        x1 = sgn(ao @ g("Wo").T + g("r1") * x + g("bo"), g("thr_x1"))
        m = relu(x1 @ g("Wup").T + g("bup"), g("thr_m"))
        x = sgn(m @ g("Wdn").T + g("r2") * x1 + g("bdn"), g("thr_x2"))
    return x @ p["Wun"].T + p["bun"]


# ----------------------------------------------------------------------------- training
def loss_fn(net, tok, tgt, valid):
    logits, worst, fin = net(tok, valid)
    ce = F.cross_entropy((logits * net.lg.exp()).reshape(-1, net.V), tgt.reshape(-1), ignore_index=IGNORE)
    pen = (sum(F.relu(w - PLIM) for w in worst.values()) + sum(F.relu(w - FLIM) for w in fin.values())) / 1000.0
    return ce, pen, logits, {**worst, **{"fin_" + k: v for k, v in fin.items()}}


@torch.no_grad()
def evaluate(net, data, bs=512):
    """Integer-model (quant, hard) accuracy / CE on data=(tok,tgt,valid), in chunks."""
    saved = (net.quant, net.hard, net.training)
    net.quant, net.hard = True, True
    net.eval()
    tok, tgt, valid = data
    hit = ce = cnt = seqok = 0.0
    worst = {}
    for i in range(0, len(tok), bs):
        t, g, v = tok[i:i + bs], tgt[i:i + bs], valid[i:i + bs]
        n = int(v.sum(1).max())
        t, g, v = t[:, :n], g[:, :n], v[:, :n]
        c, _, logits, w = loss_fn(net, t, g, v)
        pred = logits.argmax(-1)
        hit += (pred[v] == g[v]).sum().item()
        ce += c.item() * v.sum().item()
        cnt += v.sum().item()
        seqok += (((pred == g) | ~v).all(1)).sum().item()
        for k, x in w.items():
            worst[k] = max(worst.get(k, 0), int(x))
    net.quant, net.hard = saved[0], saved[1]
    net.train(saved[2])
    return hit / cnt, seqok / len(tok), ce / cnt, worst


def run_stage(net, name, steps, lr, quant, hard, alpha, train, dev, bs, log, wd=0.0, lr_end=0.05, eval_every=100,
              keep_best=False):
    tok, tgt, valid = train
    net.quant, net.hard, net.alpha = quant, hard, alpha
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    best, best_state = (1e9,), None
    N = len(tok)
    perm, pos = torch.randperm(N), 0
    for step in range(steps):
        cur = lr * (lr_end + (1 - lr_end) * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        if pos + bs > N:
            perm, pos = torch.randperm(N), 0
        idx = perm[pos:pos + bs]
        pos += bs
        t, g_, v = tok[idx], tgt[idx], valid[idx]
        n = int(v.sum(1).max())
        t, g_, v = t[:, :n], g_[:, :n], v[:, :n]
        ce, pen, _, _ = loss_fn(net, t, g_, v)
        opt.zero_grad()
        (ce + pen).backward()
        opt.step()
        if wd:
            with torch.no_grad():
                for p in net.p.values():
                    p.mul_(1 - cur * wd)
        if step % eval_every == 0 or step == steps - 1:
            acc, seq, ice, worst = evaluate(net, dev)
            if keep_best and ice < best[0]:
                best, best_state = (ice,), copy.deepcopy(net.state_dict()) | {"_units": copy.deepcopy(net.units)}
            if step % (eval_every * 4) == 0 or step == steps - 1:
                log(f"[{name}] step {step:5d} lr {cur:.4f} train-ce {ce.item():.4f} pen {float(pen):.3f} | "
                    f"dev int acc {acc:.4f} ce {ice:.4f} maxz {max(worst.values())}")
    if keep_best and best_state is not None:
        units = best_state.pop("_units")
        net.load_state_dict(best_state)
        net.units = units
        log(f"[{name}] restored best dev int ce {best[0]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "models" / "doombrain"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--dm", type=int, default=96)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--KA", type=int, default=7)
    ap.add_argument("--KM", type=int, default=7)
    ap.add_argument("--S", type=float, default=128.0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps1", type=int, default=3000)
    ap.add_argument("--steps2", type=int, default=2000)
    ap.add_argument("--steps3", type=int, default=3000)
    ap.add_argument("--lr1", type=float, default=0.01)
    ap.add_argument("--lr2", type=float, default=0.005)
    ap.add_argument("--lr3", type=float, default=0.002)
    ap.add_argument("--wd", type=float, default=0.0, help="decoupled weight decay on float weights")
    ap.add_argument("--drop", type=float, default=0.0, help="activation dropout (train only)")
    ap.add_argument("--alpha", type=float, default=0.5, help="surrogate sigmoid width / threshold step")
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--recal", action="store_true", help="re-calibrate thresholds after the float stage")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="exact for the integer stages: every integer-valued tensor stays far below 2^24")
    ap.add_argument("--corpus", default=str(LEVELS))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(getattr(torch, a.dtype))
    torch.manual_seed(a.seed)
    H = a.d // 8
    corp = load_corpus(Path(a.corpus), a.dev)
    vocab, T = corp["vocab"], corp["T"]
    assert a.L <= 2 and a.d <= 32 and a.d % 8 == 0 and a.dm <= 96 and T <= 24
    assert max(a.K, a.KA, a.KM) <= 7 and len(vocab) <= 32

    t0 = time.time()

    def log(s):
        print(f"{time.time() - t0:7.1f}s {s}", flush=True)

    def pack(names):
        tok, tgt = make_batch(vocab, names, T)
        return tok, tgt, tgt != IGNORE

    train, dev = pack(corp["train"]), pack(corp["dev"])
    net = Net(len(vocab), T, a.d, H, a.dm, a.K, a.KA, a.KM, a.L, a.S, a.seed, a.drop)
    log(f"corpus {corp['source']}: {len(corp['names'])} names; train {len(corp['train'])} dev {len(corp['dev'])} "
        f"val {len(corp['val'])}; T={T} vocab {len(vocab)}; params {sum(p.numel() for p in net.parameters())}")

    def calibrate():
        net.quant, net.hard, net.calibrating = False, True, True
        sub = torch.randperm(len(train[0]))[:600]
        tk, _, vd = (x[sub] for x in train)
        net.valid = vd
        net.eval()
        with torch.no_grad():
            net(tk, vd)
        net.calibrating = False

    calibrate()
    run_stage(net, "float", a.steps1, a.lr1, False, False, a.alpha, train, dev, a.bs, log, wd=a.wd)
    net.quant, net.hard = False, False
    net.eval()
    with torch.no_grad():
        lg, _, _ = net(dev[0], dev[2])
    facc = (lg.argmax(-1)[dev[2]] == dev[1][dev[2]]).double().mean().item()
    log(f"float model (smooth ADC, unrounded) dev teacher-forced acc {facc:.4f}")
    if a.recal:
        calibrate()
    run_stage(net, "act-qat", a.steps2, a.lr2, False, True, a.alpha, train, dev, a.bs, log, wd=a.wd)
    run_stage(net, "full-qat", a.steps3, a.lr3, True, True, a.alpha, train, dev, a.bs, log, wd=a.wd, keep_best=True)

    qm = net.qmodel(vocab)
    out = Path(a.out)
    qm.save(out)
    acc, seq, ice, worst = evaluate(net, dev)
    tacc, _, _, _ = evaluate(net, train)
    (out / "train_info.json").write_text(json.dumps(
        dict(float_dev_acc=facc, dev_int_acc=acc, train_int_acc=tacc, seed=a.seed, wall_s=time.time() - t0,
             int_logit_scale=float(net.lg.exp()), args=record_args(a), split_seed=SPLIT_SEED,
             n_train=len(corp["train"]), n_dev=len(corp["dev"]), n_val=len(corp["val"])), indent=1))
    log(f"final torch-int train acc {tacc:.4f} dev acc {acc:.4f} ce {ice:.4f} worst partial sums {worst}")
    log(f"saved {repo_rel(a.out)}")


if __name__ == "__main__":
    main()
