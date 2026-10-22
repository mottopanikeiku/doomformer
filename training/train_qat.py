"""Quantisation-aware training of the DOOM-name transformer.

The torch model mirrors `doomformer.intref.forward` exactly:
  x0 = adc(tok+pos); q,k,v = adc(W x0 + b); a = adc_relu(q.k) (causal); ao = adc(sum a v, thr_o[t]);
  x1 = adc(Wo ao + r1*x0 + bo); m = adc_relu(Wup x1 + bup); x2 = adc(Wdn m + r2*x1 + bdn);
  logits = Wun x2 + bun.
Every tensor of the integer model is an integer-valued float64 tensor, so the same code is the
integer model when `quant=True, hard=True`.

--attn softmax (intref attn="softmax") replaces the attention normalisation: e = adc_relu(q.k, thr_a) (causal),
den = 1 + sum e, ao = trunc(R * sum e v / den) in [-(R*K-1), R*K-1]; there is no thr_o. thr_a is parametrised as
c + sigma*ln(k+1) (c, sigma > 0 learned), so e approximates exp((score - c)/sigma). The backward pass (and the
float stage) uses R * sum e v / den in place of the trunc.

--norm rms --Rn N (intref norm="rms") normalises the input of q/k/v (x0) and of the MLP (x1): s = sum c^2 over the
staircase counts c, r = #{k: s >= d k^2} = floor(sqrt(s/d)), n = trunc(Rn * c / (1 + r)). The backward pass (and the
float stage) uses Rn * c / (1 + r) with r's sigmoid surrogate; gains fold into the following weights (no arrays).
--mlp reglu (intref mlp="reglu"): m = adc_relu(Wg x + bg, thr_g) * adc_signed(Wu x + bu, thr_u) in place of
adc_relu(Wup x + bup, thr_m); the product of the two staircases is exact in the hard path, surrogates multiply.

Stages
  1. float   : weights unrounded, staircases replaced by smooth sums of sigmoids.
  2. act-QAT : weights unrounded, hard staircases (straight-through: backward = sigmoid surrogate).
  3. full QAT: weights/biases/embeddings/residuals and thresholds rounded with STE, hard staircases.
Thresholds are parametrised as cumulative sums of positive increments, so they are always
positive and nondecreasing; they are calibrated from activation quantiles before stage 1 (softmax thr_a:
c at the median positive score, top threshold at the 97th percentile).

Usage: python training/train_qat.py --out models/doomnames [--seed 0] [--attn softmax --R 1] [--norm rms --Rn 4]
       [--mlp reglu --dm 21]
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

from data import NAMES, VOCAB, encode, encode_target  # noqa: E402
from doomformer import intref  # noqa: E402
from train_brain import record_args, repo_rel, rmsnorm_ste  # noqa: E402

torch.set_default_dtype(torch.float64)
Z_MAX, W_MAX = intref.Z_MAX, intref.W_MAX
PORT_MAX = intref.PORT_MAX
PLIM = 0.9 * Z_MAX          # training-time ceiling for partial sums (contract limit is Z_MAX)
FLIM = 0.9 * PORT_MAX       # training-time ceiling for final z feeding gates (contract limit PORT_MAX)
IGNORE = -100
EXP_C_Q = 0.5              # softmax calibration: quantile of the positive scores where e first reaches 1


# ----------------------------------------------------------------------------- data
def make_batch(names: list[str], T: int):
    tok = torch.zeros(len(names), T, dtype=torch.long)
    tgt = torch.full((len(names), T), IGNORE, dtype=torch.long)
    for i, n in enumerate(names):
        x, y = encode(n), encode_target(n)
        tok[i, : len(x)] = torch.tensor(x)
        tgt[i, : len(y)] = torch.tensor(y)
    return tok, tgt


def round_ste(x):
    return x + (torch.round(x) - x).detach()


def inv_softplus(y):
    return math.log(math.expm1(y))


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    SITES = {  # name -> (number of levels attr, relu?)
        "x0": ("K", False), "q": ("K", False), "k": ("K", False), "v": ("K", False),
        "a": ("KA", True), "o": ("K", False), "x1": ("K", False), "m": ("KM", True), "g": ("KM", True),
        "u": ("KM", False), "x2": ("K", False),
    }

    def __init__(self, V, T, d, H, dm, K, KA, KM, S=128.0, seed=0, attn="relu", R=1, norm="none", Rn=1, mlp="relu"):
        super().__init__()
        self.V, self.T, self.d, self.H, self.dm, self.K, self.KA, self.KM = V, T, d, H, dm, K, KA, KM
        self.dh, self.S = d // H, S
        assert attn in ("relu", "softmax") and R >= 1 and R * K >= 2
        assert norm in ("none", "rms") and Rn >= 1 and (norm == "rms" or Rn == 1) and mlp in ("relu", "reglu")
        self.attn, self.R, self.norm, self.Rn, self.mlp = attn, R, norm, Rn, mlp
        skip = ({"o"} if attn == "softmax" else set()) | ({"m"} if mlp == "reglu" else {"g", "u"})
        self.sites = [n for n in self.SITES if n not in skip]    # softmax: no thr_o; reglu: thr_g, thr_u for thr_m
        g = torch.Generator().manual_seed(seed)

        def nrm(*shape, std):
            return nn.Parameter(torch.randn(*shape, generator=g) * std)

        p = nn.ParameterDict()
        p["tok_emb"], p["pos_emb"] = nrm(V, d, std=1.0), nrm(T, d, std=0.7)
        for w in ("Wq", "Wk", "Wv", "Wo"):
            p[w] = nrm(d, d, std=0.3)
        for b in ("bq", "bk", "bv", "bo", "bdn"):
            p[b] = nn.Parameter(torch.zeros(d))
        if mlp == "reglu":
            p["Wg"], p["bg"] = nrm(dm, d, std=0.3), nn.Parameter(torch.zeros(dm))
            p["Wu"], p["bu"] = nrm(dm, d, std=0.3), nn.Parameter(torch.zeros(dm))
        else:
            p["Wup"], p["bup"] = nrm(dm, d, std=0.3), nn.Parameter(torch.zeros(dm))
        p["Wdn"] = nrm(d, dm, std=0.3 * math.sqrt(d / dm))
        p["r1"], p["r2"] = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        p["Wun"], p["bun"] = nrm(V, d, std=0.3), nn.Parameter(torch.zeros(V))
        self.p = p
        self.raw = nn.ParameterDict()
        self.units = {}
        for n in self.sites:
            L = getattr(self, self.SITES[n][0])
            shape = (T, L) if n == "o" else (L,)
            if n == "a" and attn == "softmax":       # thr_a = c + sigma ln(k+1)
                for u in ("a_c", "a_s"):
                    self.raw[u] = nn.Parameter(torch.full((1,), inv_softplus(1.0)))
                    self.units[u] = torch.full((1,), float(S))
                ln = torch.log(torch.arange(1, KA + 2, dtype=torch.float64))
                self.lnk = ln[:KA]                              # ln(k+1), k = 0..KA-1
                self.dlnk = torch.cat([ln[1:2], ln[1:KA] - ln[:KA - 1]])   # surrogate widths: spacing in ln
                continue
            self.raw[n] = nn.Parameter(torch.full(shape, inv_softplus(1.0)))
            self.units[n] = torch.full(shape[:-1] + (1,), float(S))
        self.lg = nn.Parameter(torch.tensor(math.log(1.0 / S)))      # float logits = exp(lg) * int logits
        self.quant, self.hard, self.alpha = False, False, 0.5
        self.calibrating, self.valid = False, None
        self.tril = torch.tril(torch.ones(T, T))

    # --- integer-valued parameters
    def W(self, name):
        w = (self.p[name] * self.S).clamp(-W_MAX, W_MAX)
        return round_ste(w) if self.quant else w

    def exp_cs(self):
        """Softmax mode: (c, sigma) of thr_a = c + sigma ln(k+1), both positive."""
        return self.units["a_c"] * F.softplus(self.raw["a_c"]), self.units["a_s"] * F.softplus(self.raw["a_s"])

    def thr(self, name):
        if name == "a" and self.attn == "softmax":
            c, s = self.exp_cs()
            t = (c + s * self.lnk).clamp(1, PORT_MAX)
            return (round_ste(t) if self.quant else t), (s * self.dlnk).detach()
        inc = self.units[name] * F.softplus(self.raw[name])
        t = inc.cumsum(-1).clamp(1, PORT_MAX)
        return (round_ste(t) if self.quant else t), inc.detach()

    # --- calibration of a staircase from the observed pre-activation
    @torch.no_grad()
    def calibrate(self, name, z, relu):
        kk = getattr(self, self.SITES[name][0])
        valid = self.valid
        if name == "a":
            m = valid[:, None, :, None] & valid[:, None, None, :] & (self.tril > 0)[None, None]
            m = m.expand_as(z)
        else:
            m = valid[:, :, None].expand_as(z)
        if name == "a" and self.attn == "softmax":
            # exp staircase: e = 1 from the median positive score, e = KA at the 97th percentile
            sample = z[m][z[m] > 0]
            if sample.numel() >= 8:
                c, top = (torch.quantile(sample, q).item() for q in (EXP_C_Q, 0.97))
            else:
                c, top = float(self.S), 3.0 * self.S
            c = max(c, 1.0)
            self.units["a_c"] = torch.tensor([c])
            self.units["a_s"] = torch.tensor([max((top - c) / max(math.log(kk), 1.0), 1.0)])
            self.raw["a_c"].fill_(inv_softplus(1.0))
            self.raw["a_s"].fill_(inv_softplus(1.0))
            return

        def step_of(sample):
            sample = sample[sample > 0] if relu else sample.abs()
            if sample.numel() < 8:
                return None
            q = torch.quantile(sample, 0.985 if not relu else 0.97).item()
            return max(q / (kk - 0.5), 2.0)

        if name == "o":
            steps, prev = [], None
            for t in range(self.T):
                s = step_of(z[:, t][m[:, t]])
                if s is None:
                    s = (prev * (t + 1) / t) if prev is not None else float(self.S)
                steps.append(s)
                prev = s
            self.units[name] = torch.tensor(steps)[:, None]
        else:
            self.units[name] = torch.tensor([step_of(z[m]) or float(self.S)])
        init = torch.ones(kk)
        init[0] = 0.5
        self.raw[name].copy_(torch.log(torch.expm1(init)).expand_as(self.raw[name]))

    # --- staircase with sigmoid surrogate gradient
    def stair(self, name, z, relu=False):
        if self.calibrating:
            self.calibrate(name, z, relu)
        thr, inc = self.thr(name)
        tau = (self.alpha * inc).clamp(min=1.0)
        if name == "o":
            thr, tau = thr[None, :, None, :], tau[None, :, None, :]

        def count(u):
            u = u[..., None]
            soft = torch.sigmoid((u - thr) / tau).sum(-1)
            if self.hard:
                return soft + ((u >= thr).to(soft.dtype).sum(-1) - soft).detach()
            return soft

        return count(z) if relu else count(z) - count(-z)

    # --- forward; returns integer-valued logits and per-site worst partial sums
    def forward(self, tok, valid=None):
        B, T = tok.shape
        H, dh, d = self.H, self.dh, self.d
        worst, fin = {}, {}

        def track(key, hi, lo):
            m = torch.maximum(hi.abs(), lo.abs())
            if valid is not None and m.dim() >= 2 and m.shape[:2] == valid.shape:
                m = m.masked_fill(~valid.reshape(valid.shape + (1,) * (m.dim() - 2)), 0)
            worst[key] = torch.maximum(worst[key], m.max()) if key in worst else m.max()

        def final(key, z):
            m = z.abs()
            if valid is not None and m.dim() >= 2 and m.shape[:2] == valid.shape:
                m = m.masked_fill(~valid.reshape(valid.shape + (1,) * (m.dim() - 2)), 0)
            fin[key] = m.max()

        def lin(key, x, Wn, bn, extra=None):
            Wm, b = self.W(Wn), self.W(bn)
            terms = x[:, :, None, :] * Wm[None, None]
            if extra is not None:
                terms = torch.cat([terms, extra[:, :, :, None]], -1)
            z = terms.sum(-1) + b
            if key != "logits":
                final(key, z)
            track(key, b + terms.clamp(min=0).sum(-1), b + terms.clamp(max=0).sum(-1))
            return z

        E, P = self.W("tok_emb"), self.W("pos_emb")
        e, pe = E[tok], P[None, :T].expand(B, T, d)
        z0 = e + pe
        track("z0", e.clamp(min=0) + pe.clamp(min=0), e.clamp(max=0) + pe.clamp(max=0))
        final("z0", z0)
        x0 = self.stair("x0", z0)
        xa = self.rmsnorm(x0) if self.norm == "rms" else x0
        q = self.stair("q", lin("zq", xa, "Wq", "bq"))
        k = self.stair("k", lin("zk", xa, "Wk", "bk"))
        v = self.stair("v", lin("zv", xa, "Wv", "bv"))
        qh, kh, vh = (u.reshape(B, T, H, dh) for u in (q, k, v))
        sterms = qh[:, :, None] * kh[:, None, :]                      # [B,t,s,H,dh]
        score = sterms.sum(-1).permute(0, 3, 1, 2)                   # [B,H,t,s]
        if valid is not None:
            vv = (valid[:, :, None] & valid[:, None, :])[:, :, :, None, None]
            sterms = sterms.masked_fill(~vv, 0)
        final("score", score)
        track("score", sterms.clamp(min=0).sum(-1), sterms.clamp(max=0).sum(-1))
        a = self.stair("a", score, relu=True) * self.tril[None, None, :T, :T]
        oterms = torch.einsum("bhts,bshe->bthes", a, vh)
        zo = oterms.sum(-1).reshape(B, T, d)
        final("zo", zo)
        track("zo", oterms.clamp(min=0).sum(-1), oterms.clamp(max=0).sum(-1))
        if self.attn == "softmax":
            # den = 1 + sum_s e (null slot); ao = trunc(R num / den), backward through R num / den
            R, LV = self.R, self.R * self.K - 1
            dent = 1 + a.sum(-1).permute(0, 2, 1)                     # [B,t,H]
            final("den", dent)
            track("den", dent, dent)
            den = dent.repeat_interleave(dh, -1)                     # [B,t,d]
            ratio = R * zo / den
            ao = torch.trunc(ratio.detach()) + (ratio - ratio.detach()) if self.hard else ratio
            # gate-fed P = R num - l den + 1 and N = -R num - l den + 1, l = 1..LV (extremes at l = 1, LV)
            final("PN", torch.stack([s * R * zo - l * den + 1 for s in (1, -1) for l in (1, LV)], -1))
            big = R * torch.maximum(oterms.clamp(min=0).sum(-1), -oterms.clamp(max=0).sum(-1)).reshape(B, T, d)
            track("PN", 1 + big, 1 - LV * den - big)
        else:
            ao = self.stair("o", zo)
        z1 = lin("z1", ao, "Wo", "bo", extra=self.W("r1")[None, None] * x0)
        x1 = self.stair("x1", z1)
        xm = self.rmsnorm(x1) if self.norm == "rms" else x1
        if self.mlp == "reglu":
            m = self.stair("g", lin("zg", xm, "Wg", "bg"), relu=True) * self.stair("u", lin("zu", xm, "Wu", "bu"))
        else:
            m = self.stair("m", lin("zm", xm, "Wup", "bup"), relu=True)
        z2 = lin("z2", m, "Wdn", "bdn", extra=self.W("r2")[None, None] * x1)
        x2 = self.stair("x2", z2)
        logits = lin("logits", x2, "Wun", "bun")
        return logits, worst, fin

    # --- export
    @torch.no_grad()
    def export(self) -> dict[str, np.ndarray]:
        """Integer arrays named as in intref (globals bare, layer arrays "L0.<name>")."""
        q0 = self.quant
        self.quant = True
        out = {n: self.W(n).numpy().astype(np.int64) for n in self.p}
        for n in self.sites:
            out["thr_" + n] = self.thr(n)[0].numpy().astype(np.int64)
        self.quant = q0
        return {(n if n in intref.GLOBAL_ARRAYS else "L0." + n): v for n, v in out.items()}

    def qmodel(self) -> intref.QModel:
        kw = dict(attn="softmax", R=self.R) if self.attn == "softmax" else {}
        return intref.QModel(vocab=VOCAB, T=self.T, d=self.d, H=self.H, dm=self.dm, K=self.K, KA=self.KA,
                             KM=self.KM, L=1, p=self.export(), norm=self.norm, Rn=self.Rn, mlp=self.mlp, **kw)

    def rmsnorm(self, c):
        """intref.rmsnorm with straight-through surrogates (train_brain.rmsnorm_ste)."""
        return rmsnorm_ste(c, self.d, self.K, self.Rn, self.alpha, self.hard)

    def arch_info(self) -> dict:
        """norm / mlp settings for train_info."""
        return dict(norm=self.norm, Rn=self.Rn, mlp=self.mlp)

    @torch.no_grad()
    def attn_info(self) -> dict:
        """Softmax mode: the learned exp parameters of thr_a and the resulting integer thresholds (for train_info)."""
        if self.attn != "softmax":
            return {}
        c, s = self.exp_cs()
        return dict(attn="softmax", R=self.R, c=float(c), sigma=float(s),
                    thr_a=[int(t) for t in self.export()["L0.thr_a"]])


# ----------------------------------------------------------------------------- standalone int64 torch forward
def int_forward(p: dict, tok: torch.Tensor, H: int, attn: str = "relu", R: int = 1, norm: str = "none", Rn: int = 1,
                mlp: str = "relu") -> torch.Tensor:
    """Pure int64 torch evaluation of the contract from exported integers ("L{l}.<name>" layer arrays). tok [B,T] long.
    RMSNorm here is a trunc division (intref counts comparisons), an independent check of that identity."""
    p = {k: torch.as_tensor(v, dtype=torch.int64) for k, v in p.items()}
    B, T = tok.shape
    d = p["tok_emb"].shape[1]
    dh = d // H
    L = len({k.split(".")[0] for k in p if "." in k})
    causal = torch.tril(torch.ones(T, T, dtype=torch.int64))

    def relu(z, thr):
        return (z[..., None] >= thr).sum(-1)

    def sgn(z, thr):
        return relu(z, thr) - relu(-z, thr)

    def rms(c, levels):
        if norm != "rms":
            return c
        r = relu((c * c).sum(-1, keepdim=True), d * torch.arange(1, levels + 1) ** 2)
        return torch.div(Rn * c, 1 + r, rounding_mode="trunc")

    x = sgn(p["tok_emb"][tok] + p["pos_emb"][None, :T], p["thr_x0"])
    for l in range(L):
        q_ = {k.split(".", 1)[1]: v for k, v in p.items() if k.startswith(f"L{l}.")}
        xa = rms(x, len(p["thr_x0"] if l == 0 else p[f"L{l - 1}.thr_x2"]))
        q, k, v = (sgn(xa @ q_[W].T + q_[b], q_[t]) for W, b, t in
                   (("Wq", "bq", "thr_q"), ("Wk", "bk", "thr_k"), ("Wv", "bv", "thr_v")))
        qh, kh, vh = (u.reshape(B, T, H, dh) for u in (q, k, v))
        score = torch.einsum("bthd,bshd->bhts", qh, kh)
        a = relu(score, q_["thr_a"]) * causal
        zo = torch.einsum("bhts,bshe->bthe", a, vh).reshape(B, T, d)
        if attn == "softmax":
            den = (1 + a.sum(-1)).permute(0, 2, 1).repeat_interleave(dh, -1)     # [B,t,d]
            ao = torch.div(R * zo, den, rounding_mode="trunc")
        else:
            ao = sgn(zo, q_["thr_o"][None, :T, None, :])
        x1 = sgn(ao @ q_["Wo"].T + q_["r1"] * x + q_["bo"], q_["thr_x1"])
        xm = rms(x1, len(q_["thr_x1"]))
        if mlp == "reglu":
            m = relu(xm @ q_["Wg"].T + q_["bg"], q_["thr_g"]) * sgn(xm @ q_["Wu"].T + q_["bu"], q_["thr_u"])
        else:
            m = relu(xm @ q_["Wup"].T + q_["bup"], q_["thr_m"])
        x = sgn(m @ q_["Wdn"].T + q_["r2"] * x1 + q_["bdn"], q_["thr_x2"])
    return x @ p["Wun"].T + p["bun"]


# ----------------------------------------------------------------------------- training
def loss_fn(net, tok, tgt, valid):
    logits, worst, fin = net(tok, valid)
    ce = F.cross_entropy((logits * net.lg.exp()).reshape(-1, net.V), tgt.reshape(-1), ignore_index=IGNORE)
    pen = (sum(F.relu(w - PLIM) for w in worst.values()) + sum(F.relu(w - FLIM) for w in fin.values())) / 1000.0
    return ce, pen, logits, {**worst, **{"fin_" + k: v for k, v in fin.items()}}


@torch.no_grad()
def evaluate(net, tok, tgt, valid):
    saved = (net.quant, net.hard)
    net.quant, net.hard = True, True
    ce, pen, logits, worst = loss_fn(net, tok, tgt, valid)
    net.quant, net.hard = saved
    pred = logits.argmax(-1)
    acc = (pred[valid] == tgt[valid]).double().mean().item()
    seq = ((pred == tgt) | ~valid).all(1).double().mean().item()
    return acc, seq, ce.item(), {k: int(v) for k, v in worst.items()}


def run_stage(net, name, steps, lr, quant, hard, alpha, data, log, lr_end=0.05, eval_every=25, keep_best=False):
    tok, tgt, valid = data
    net.quant, net.hard, net.alpha = quant, hard, alpha
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state = (-1.0, 1e9), None
    for step in range(steps):
        cur = lr * (lr_end + (1 - lr_end) * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        ce, pen, _, _ = loss_fn(net, tok, tgt, valid)
        opt.zero_grad()
        (ce + pen).backward()
        opt.step()
        if step % eval_every == 0 or step == steps - 1:
            acc, seq, ice, worst = evaluate(net, tok, tgt, valid)
            if keep_best and (acc, -ice) > (best[0], -best[1]):
                best, best_state = (acc, ice), copy.deepcopy(net.state_dict()) | {"_units": copy.deepcopy(net.units)}
            if step % (eval_every * 8) == 0 or step == steps - 1:
                log(f"[{name}] step {step:5d} lr {cur:.4f} train-ce {ce.item():.4f} pen {float(pen):.3f} | "
                    f"int acc {acc:.4f} seq-acc {seq:.4f} int-ce {ice:.4f} maxz {max(worst.values())}")
    if keep_best and best_state is not None:
        units = best_state.pop("_units")
        net.load_state_dict(best_state)
        net.units = units
        log(f"[{name}] restored best int acc {best[0]:.4f} ce {best[1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "models" / "doomnames"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--dm", type=int, default=32)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--KA", type=int, default=7)
    ap.add_argument("--KM", type=int, default=7)
    ap.add_argument("--S", type=float, default=128.0)
    ap.add_argument("--steps1", type=int, default=2000)
    ap.add_argument("--steps2", type=int, default=3000)
    ap.add_argument("--steps3", type=int, default=5000)
    ap.add_argument("--lr1", type=float, default=0.02)
    ap.add_argument("--lr2", type=float, default=0.01)
    ap.add_argument("--lr3", type=float, default=0.004)
    ap.add_argument("--recal", action="store_true", help="re-calibrate thresholds after the float stage")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--attn", choices=("relu", "softmax"), default="relu",
                    help="relu: ReLU staircase weights, fixed 1/(t+1) in thr_o; softmax: exp staircase, ao = trunc(R num/den)")
    ap.add_argument("--R", type=int, default=1, help="softmax only: output resolution, ao in [-(R*K-1), R*K-1]")
    ap.add_argument("--norm", choices=("none", "rms"), default="none",
                    help="rms: integer RMSNorm before q/k/v and the MLP, n = trunc(Rn c / (1 + floor(sqrt(s/d))))")
    ap.add_argument("--Rn", type=int, default=1, help="rms only: output resolution of the norm")
    ap.add_argument("--mlp", choices=("relu", "reglu"), default="relu",
                    help="reglu: m = relu-staircase(Wg x + bg) * signed-staircase(Wu x + bu) (pick dm for the "
                         "parameter count: 3*d*dm + 2*dm against 2*d*dm + dm)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    assert a.T <= 16 and a.d <= 16 and a.H == 2 and a.dm <= 32 and max(a.K, a.KA, a.KM) <= 7 and len(VOCAB) <= 32

    t0 = time.time()
    def log(s):
        print(f"{time.time() - t0:7.1f}s {s}", flush=True)

    tok, tgt = make_batch(NAMES, a.T)
    valid = tgt != IGNORE
    data = (tok, tgt, valid)
    net = Net(len(VOCAB), a.T, a.d, a.H, a.dm, a.K, a.KA, a.KM, a.S, a.seed, attn=a.attn, R=a.R, norm=a.norm, Rn=a.Rn,
              mlp=a.mlp)
    log(f"{len(NAMES)} names, {int(valid.sum())} target tokens, vocab {len(VOCAB)}, params {sum(p.numel() for p in net.parameters())}")

    def calibrate():
        net.quant, net.hard, net.calibrating, net.valid = False, True, True, valid
        with torch.no_grad():
            net(tok, valid)
        net.calibrating = False

    calibrate()
    run_stage(net, "float", a.steps1, a.lr1, False, False, 0.5, data, log)
    # float model accuracy: unrounded weights, smooth staircases
    net.quant, net.hard = False, False
    with torch.no_grad():
        lg, _, _ = net(tok, valid)
    facc = (lg.argmax(-1)[valid] == tgt[valid]).double().mean().item()
    log(f"float model (smooth ADC, unrounded) teacher-forced acc {facc:.4f}")
    if a.recal:
        calibrate()
    run_stage(net, "act-qat", a.steps2, a.lr2, False, True, 0.5, data, log)
    run_stage(net, "full-qat", a.steps3, a.lr3, True, True, 0.5, data, log, keep_best=True)

    qm = net.qmodel()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    qm.save(Path(a.out))
    acc, seq, ice, worst = evaluate(net, tok, tgt, valid)
    (Path(a.out) / "train_info.json").write_text(json.dumps(
        dict(float_acc=facc, int_acc=acc, seed=a.seed, wall_s=time.time() - t0, args=record_args(a),
             **net.arch_info(), **net.attn_info()), indent=1))
    log(f"final torch-int acc {acc:.4f} seq-exact {seq:.4f} worst partial sums {worst}")
    log(f"saved {repo_rel(a.out)}")


if __name__ == "__main__":
    main()
