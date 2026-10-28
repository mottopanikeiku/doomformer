"""Small integer-faithful GPT for DOOM: the torch model behind `Transformer.from_torch`, with its quantisation-aware
training recipe.

`GPT` mirrors `doomformer.intref.forward` exactly. Its parameters are floats; the forward pass has three modes:

* float stage (`quant=False, hard=False`): every staircase is a sum of sigmoids, weights are not rounded;
* act-QAT (`quant=False, hard=True`): hard staircases (count of passed thresholds, sigmoid surrogate gradient);
* full QAT (`quant=True, hard=True`): weights, biases, embeddings, residual gains and thresholds are rounded with a
  straight-through gradient, every tensor is an integer-valued float and the forward pass IS the integer model.

`fit` runs the three stages (the recipe of training/train_brain.py, which now imports this module), and
`Transformer.from_torch(model, vocab)` (`doomformer.intref.QModel.from_torch`) exports the rounded parameters as a
`QModel`: the same integers the training scripts write to models/. Architecture variants: attention `relu` (counts of
passed score thresholds, `thr_o[t]` folding 1/(t+1)) or `softmax` (quantised exponential, null slot, comparison
division); norm `none` or `rms` (integer RMSNorm before q/k/v and the MLP); MLP `relu` or `reglu` (gated).
Limits: d % H == 0; `Transformer.from_torch` enforces what `intref.check_limits` checks (|w| <= 255, thresholds in 1..12,000,
every gate-fed value within +-12,000 and every partial sum within +-30,000 on its calibration sequences), not a count of
levels or layers; see docs/interop.md (Transformers).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import intref

Z_MAX, W_MAX, PORT_MAX = intref.Z_MAX, intref.W_MAX, intref.PORT_MAX
PLIM = 0.9 * Z_MAX          # training-time ceiling for partial sums (contract limit is Z_MAX)
FLIM = 0.9 * PORT_MAX       # training-time ceiling for final z feeding gates (contract limit PORT_MAX)
IGNORE = -100
EXP_C_Q = 0.5              # softmax calibration: quantile of the positive scores where e first reaches 1


def round_ste(x):
    return x + (torch.round(x) - x).detach()


def inv_softplus(y):
    return math.log(math.expm1(y))


def rmsnorm_ste(c, d: int, K: int, Rn: int, alpha: float, hard: bool):
    """intref.rmsnorm on staircase counts c [..., d] (|c| <= K): s = sum c^2, r = #{k <= K: s >= d k^2},
    n = trunc(Rn c / (1 + r)). hard: exact integers forward, backward through Rn c / (1 + r) with r's sigmoid surrogate
    (widths alpha times the threshold spacing d (2k - 1)); not hard (float stage): Rn c / (1 + soft r)."""
    k = torch.arange(1, K + 1, dtype=c.dtype)
    thr, tau = d * k ** 2, (alpha * d * (2 * k - 1)).clamp(min=1.0)
    if hard:
        c = c.detach().round() + (c - c.detach())          # exact integer value, same gradient
    s = (c * c).sum(-1, keepdim=True)
    soft = torch.sigmoid((s[..., None] - thr) / tau).sum(-1)
    if not hard:
        return Rn * c / (1 + soft)
    r = (s.detach()[..., None] >= thr).to(c.dtype).sum(-1) + (soft - soft.detach())
    ratio = Rn * c / (1 + r)
    return torch.trunc(ratio.detach()) + (ratio - ratio.detach())


# ----------------------------------------------------------------------------- model
class Net(nn.Module):
    SITES = {  # name -> (levels attr, relu?)
        "q": ("K", False), "k": ("K", False), "v": ("K", False), "a": ("KA", True), "o": ("K", False),
        "x1": ("K", False), "m": ("KM", True), "g": ("KM", True), "u": ("KM", False), "x2": ("K", False),
    }

    def __init__(self, V, T, d, H, dm, K, KA, KM, L, S=128.0, seed=0, drop=0.0, attn="relu", R=1, norm="none", Rn=1,
                 mlp="relu"):
        super().__init__()
        self.V, self.T, self.d, self.H, self.dm, self.K, self.KA, self.KM, self.L = V, T, d, H, dm, K, KA, KM, L
        self.dh, self.S, self.drop = d // H, S, drop
        for what, value, ok in (("attn", attn, ("relu", "softmax")), ("norm", norm, ("none", "rms")),
                                ("mlp", mlp, ("relu", "reglu"))):
            if value not in ok:
                raise ValueError(f"{what}={value!r} is not supported: {what} is one of {ok} (the variants the integer "
                                 "reference computes)")
        if d % H or min(V, T, d, H, dm, K, KA, KM, L, R, Rn) < 1:
            raise ValueError(f"sizes must be positive and d divisible by H (V={V}, T={T}, d={d}, H={H}, dm={dm}, K={K}, "
                             f"KA={KA}, KM={KM}, L={L}, R={R}, Rn={Rn})")
        if R * K < 2:
            raise ValueError("R * K must be at least 2")
        if attn == "relu" and R != 1:
            raise ValueError("R is a softmax-attention setting; relu attention needs R = 1")
        if norm == "none" and Rn != 1:
            raise ValueError("Rn is an RMSNorm setting; norm='none' needs Rn = 1")
        self.config = dict(V=V, T=T, d=d, H=H, dm=dm, K=K, KA=KA, KM=KM, L=L, S=S, seed=seed, drop=drop, attn=attn, R=R,
                           norm=norm, Rn=Rn, mlp=mlp)
        self.attn, self.R, self.norm, self.Rn, self.mlp = attn, R, norm, Rn, mlp
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
            if mlp == "reglu":
                p[f"{l}_Wg"], p[f"{l}_bg"] = nrm(dm, d, std=0.3), nn.Parameter(torch.zeros(dm))
                p[f"{l}_Wu"], p[f"{l}_bu"] = nrm(dm, d, std=0.3), nn.Parameter(torch.zeros(dm))
            else:
                p[f"{l}_Wup"], p[f"{l}_bup"] = nrm(dm, d, std=0.3), nn.Parameter(torch.zeros(dm))
            p[f"{l}_Wdn"] = nrm(d, dm, std=0.3 * math.sqrt(d / dm))
            p[f"{l}_r1"], p[f"{l}_r2"] = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        p["Wun"], p["bun"] = nrm(V, d, std=0.3), nn.Parameter(torch.zeros(V))
        self.p = p
        self.raw = nn.ParameterDict()
        self.units = {}
        # softmax mode has no thr_o; its thr_a = c + sigma ln(k+1) with c, sigma > 0 learned (raw "<site>c", "<site>s");
        # the gated MLP has thr_g, thr_u in place of thr_m
        skip = ({"o"} if attn == "softmax" else set()) | ({"m"} if mlp == "reglu" else {"g", "u"})
        self.sites = ["x0"] + [f"{l}_{n}" for l in range(L) for n in self.SITES if n not in skip]
        ln = torch.log(torch.arange(1, KA + 2, dtype=torch.get_default_dtype()))
        self.lnk = ln[:KA]                                             # ln(k+1), k = 0..KA-1
        self.dlnk = torch.cat([ln[1:2], ln[1:KA] - ln[:KA - 1]])       # surrogate widths: spacing in ln
        for site in self.sites:
            kk = getattr(self, "K" if site == "x0" else self.SITES[site.split("_")[1]][0])
            if self.exp_site(site):
                for u in (site + "c", site + "s"):
                    self.raw[u] = nn.Parameter(torch.full((1,), inv_softplus(1.0)))
                    self.units[u] = torch.full((1,), float(S))
                continue
            per_pos = site.endswith("_o")
            shape = (T, kk) if per_pos else (kk,)
            self.raw[site] = nn.Parameter(torch.full(shape, inv_softplus(1.0)))
            self.units[site] = torch.full(shape[:-1] + (1,), float(S))
        self.lg = nn.Parameter(torch.tensor(math.log(1.0 / S)))      # float logits = exp(lg) * int logits
        self.quant, self.hard, self.alpha = False, False, 0.5
        self.calibrating, self.valid = False, None
        self._calibrated = False

    # --- integer-valued parameters
    def W(self, name):
        w = (self.p[name] * self.S).clamp(-W_MAX, W_MAX)
        return round_ste(w) if self.quant else w

    def exp_site(self, site) -> bool:
        """Softmax mode: the attention-weight staircase, thresholds c + sigma ln(k+1) (a quantised exponential)."""
        return self.attn == "softmax" and site.endswith("_a")

    def exp_cs(self, site):
        """(c, sigma) of an exp_site's thresholds, both positive."""
        return (self.units[site + "c"] * F.softplus(self.raw[site + "c"]),
                self.units[site + "s"] * F.softplus(self.raw[site + "s"]))

    def thr(self, site):
        if self.exp_site(site):
            c, s = self.exp_cs(site)
            t = (c + s * self.lnk).clamp(1, PORT_MAX)
            return (round_ste(t) if self.quant else t), (s * self.dlnk).detach()
        inc = self.units[site] * F.softplus(self.raw[site])
        t = inc.cumsum(-1).clamp(1, PORT_MAX)
        return (round_ste(t) if self.quant else t), inc.detach()

    @staticmethod
    def relu_site(site):
        return site.split("_")[-1] in ("a", "m", "g")

    # --- calibration of a staircase from the observed pre-activation
    @torch.no_grad()
    def calibrate(self, site, z):
        self._calibrated = True
        kk = getattr(self, "K" if site == "x0" else self.SITES[site.split("_")[1]][0])
        relu, valid = self.relu_site(site), self.valid
        if site.endswith("_a"):
            nn_ = z.shape[-1]
            m = valid[:, None, :, None] & valid[:, None, None, :] & (torch.tril(torch.ones(nn_, nn_)) > 0)[None, None]
            m = m.expand_as(z)
        else:
            m = valid[:, :, None].expand_as(z)
        if self.exp_site(site):
            # exp staircase: e = 1 from the median positive score, e = KA at the 97th percentile
            sample = z[m][z[m] > 0][:2_000_000]
            if sample.numel() >= 8:
                c, top = (torch.quantile(sample, q).item() for q in (EXP_C_Q, 0.97))
            else:
                c, top = float(self.S), 3.0 * self.S
            c = max(c, 1.0)
            self.units[site + "c"] = torch.tensor([c])
            self.units[site + "s"] = torch.tensor([max((top - c) / max(math.log(kk), 1.0), 1.0)])
            self.raw[site + "c"].fill_(inv_softplus(1.0))
            self.raw[site + "s"].fill_(inv_softplus(1.0))
            return

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
            xa = rmsnorm_ste(x, d, self.K, self.Rn, self.alpha, self.hard) if self.norm == "rms" else x
            zq, zk, zv = (lin(f"{l}.z{u}", xa, s("W" + u), s("b" + u)) for u in "qkv")
            q, k, v = self.stair(s("q"), zq), self.stair(s("k"), zk), self.stair(s("v"), zv)
            qh, kh, vh = (u.reshape(B, n, H, dh) for u in (q, k, v))
            score = torch.einsum("bthd,bshd->bhts", qh, kh)
            a = self.stair(s("a"), score, relu=True) * tril
            if self.attn == "softmax":
                # den = 1 + sum_s e (null slot); ao = trunc(R num / den), backward through R num / den
                R, LV = self.R, self.R * self.K - 1
                oterms = torch.einsum("bhts,bshe->bthes", a, vh)                # [B,t,H,dh,s]
                zo = oterms.sum(-1).reshape(B, n, d)
                dent = 1 + a.sum(-1).permute(0, 2, 1)                          # [B,t,H]
                final(f"{l}.den", dent)
                track(f"{l}.den", dent, dent)
                den = dent.repeat_interleave(dh, -1)                           # [B,t,d]
                ratio = R * zo / den
                ao = torch.trunc(ratio.detach()) + (ratio - ratio.detach()) if self.hard else ratio
                # gate-fed P = R num - l den + 1 and N = -R num - l den + 1, l = 1..LV (extremes at l = 1, LV)
                final(f"{l}.PN", torch.stack([sg * R * zo - lv * den + 1 for sg in (1, -1) for lv in (1, LV)],
                                             -1).abs().amax(-1))
                big = R * torch.maximum(oterms.clamp(min=0).sum(-1), -oterms.clamp(max=0).sum(-1)).reshape(B, n, d)
                track(f"{l}.PN", 1 + big, 1 - LV * den - big)
                ao = self.dropout(ao)
            else:
                zo = torch.einsum("bhts,bshe->bthe", a, vh).reshape(B, n, d)
                ao = self.dropout(self.stair(s("o"), zo))
            z1 = lin(f"{l}.z1", ao, s("Wo"), s("bo"), extra=self.W(s("r1"))[None, None] * x)
            x1 = self.dropout(self.stair(s("x1"), z1))
            xm = rmsnorm_ste(x1, d, self.K, self.Rn, self.alpha, self.hard) if self.norm == "rms" else x1
            if self.mlp == "reglu":
                m = self.dropout(self.stair(s("g"), lin(f"{l}.zg", xm, s("Wg"), s("bg")), relu=True)
                                 * self.stair(s("u"), lin(f"{l}.zu", xm, s("Wu"), s("bu"))))
            else:
                zm = lin(f"{l}.zm", xm, s("Wup"), s("bup"))
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
        for site in self.sites:
            name = "thr_x0" if site == "x0" else f"L{site.split('_')[0]}.thr_{site.split('_', 1)[1]}"
            out[name] = self.thr(site)[0].numpy().astype(np.int64)
        self.quant = q0
        return out

    def qmodel(self, vocab) -> intref.QModel:
        kw = dict(attn="softmax", R=self.R) if self.attn == "softmax" else {}
        return intref.QModel(vocab=vocab, T=self.T, d=self.d, H=self.H, dm=self.dm, K=self.K, KA=self.KA,
                             KM=self.KM, L=self.L, p=self.export(), norm=self.norm, Rn=self.Rn, mlp=self.mlp, **kw)

    def arch_info(self) -> dict:
        """norm / mlp settings for train_info."""
        return dict(norm=self.norm, Rn=self.Rn, mlp=self.mlp)

    @torch.no_grad()
    def attn_info(self) -> dict:
        """Softmax mode: per layer, the learned exp parameters of thr_a and the integer thresholds (for train_info)."""
        if self.attn != "softmax":
            return {}
        cs = [self.exp_cs(f"{l}_a") for l in range(self.L)]
        p = self.export()
        return dict(attn="softmax", R=self.R, c=[float(c) for c, _ in cs], sigma=[float(s) for _, s in cs],
                    thr_a=[[int(t) for t in p[f"L{l}.thr_a"]] for l in range(self.L)])


# ----------------------------------------------------------------------------- standalone int64 torch forward
def int_forward(p: dict, tok: torch.Tensor, H: int, L: int, attn: str = "relu", R: int = 1, norm: str = "none",
                Rn: int = 1, mlp: str = "relu") -> torch.Tensor:
    """Pure int64 torch evaluation of the contract from exported integers. tok [B,n] long. RMSNorm here is a trunc
    division (intref counts comparisons), an independent check of that identity."""
    p = {k: torch.as_tensor(v, dtype=torch.int64) for k, v in p.items()}
    B, n = tok.shape
    d = p["tok_emb"].shape[1]
    dh = d // H

    def relu(z, thr):
        return (z[..., None] >= thr).sum(-1)

    def sgn(z, thr):
        return relu(z, thr) - relu(-z, thr)

    def rms(c, levels):
        if norm != "rms":
            return c
        r = relu((c * c).sum(-1, keepdim=True), d * torch.arange(1, levels + 1) ** 2)
        return torch.div(Rn * c, 1 + r, rounding_mode="trunc")

    x = sgn(p["tok_emb"][tok] + p["pos_emb"][None, :n], p["thr_x0"])
    tril = torch.tril(torch.ones(n, n, dtype=torch.int64))
    for l in range(L):
        g = lambda nme: p[f"L{l}.{nme}"]
        xa = rms(x, len(p["thr_x0"] if l == 0 else p[f"L{l - 1}.thr_x2"]))
        q, k, v = (sgn(xa @ g("W" + u).T + g("b" + u), g("thr_" + u)) for u in "qkv")
        qh, kh, vh = (u.reshape(B, n, H, dh) for u in (q, k, v))
        score = torch.einsum("bthd,bshd->bhts", qh, kh)
        a = relu(score, g("thr_a")) * tril
        zo = torch.einsum("bhts,bshe->bthe", a, vh).reshape(B, n, d)
        if attn == "softmax":
            den = (1 + a.sum(-1)).permute(0, 2, 1).repeat_interleave(dh, -1)     # [B,n,d]
            ao = torch.div(R * zo, den, rounding_mode="trunc")
        else:
            ao = sgn(zo, g("thr_o")[None, :n, None, :])
        x1 = sgn(ao @ g("Wo").T + g("r1") * x + g("bo"), g("thr_x1"))
        xm = rms(x1, len(g("thr_x1")))
        if mlp == "reglu":
            m = relu(xm @ g("Wg").T + g("bg"), g("thr_g")) * sgn(xm @ g("Wu").T + g("bu"), g("thr_u"))
        else:
            m = relu(xm @ g("Wup").T + g("bup"), g("thr_m"))
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
                log(f"[{name}] step {step:5d} lr {cur:.4f} train-ce {ce.item():.4f} pen {float(pen.detach()):.3f} | "
                    f"dev int acc {acc:.4f} ce {ice:.4f} maxz {max(worst.values())}")
    if keep_best and best_state is not None:
        units = best_state.pop("_units")
        net.load_state_dict(best_state)
        net.units = units
        log(f"[{name}] restored best dev int ce {best[0]:.4f}")


# ----------------------------------------------------------------------------- public API
GPT = Net            # the documented name: `GPT(V, T, d, H, dm, K, KA, KM, L, ...)`


def pack(rows: list[list[int]], targets: list[list[int]], T: int):
    """(tok, tgt, valid) tensors from token rows and next-token targets (padded with token 0 / IGNORE to T): the
    format `fit`, `evaluate` and `loss_fn` take. A row and its target have the same length <= T."""
    tok = torch.zeros(len(rows), T, dtype=torch.long)
    tgt = torch.full((len(rows), T), IGNORE, dtype=torch.long)
    for i, (x, y) in enumerate(zip(rows, targets)):
        if len(x) != len(y) or len(x) > T:
            raise ValueError(f"row {i}: {len(x)} tokens, {len(y)} targets, T = {T}")
        tok[i, :len(x)] = torch.tensor(x)
        tgt[i, :len(y)] = torch.tensor(y)
    return tok, tgt, tgt != IGNORE


def _check_calibration_set(train) -> None:
    tok, _, valid = train
    if not bool(valid.any()):
        raise ValueError("the calibration set has no valid positions (every target is IGNORE)")
    seen = tok[valid.bool() if valid.dtype != torch.bool else valid]
    if bool((tok == tok.flatten()[0]).all()) or bool((seen == seen.flatten()[0]).all()):
        raise ValueError("the calibration set has a single distinct token: every staircase would see a constant "
                         "and get degenerate thresholds; calibrate on real text")


def calibrate(model: Net, train, n: int = 600, seed: int | None = None) -> None:
    """Set every staircase of `model` from the activations of `n` random rows of `train` = (tok, tgt, valid): the
    thresholds start at quantiles of what the staircase sees (the recipe's step 0)."""
    _check_calibration_set(train)
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    sub = torch.randperm(len(train[0]), generator=g)[:n]
    tk, _, vd = (x[sub] for x in train)
    saved = (model.quant, model.hard, model.training)
    model.quant, model.hard, model.calibrating, model.valid = False, True, True, vd
    model.eval()
    with torch.no_grad():
        model(tk, vd)
    model.calibrating = False
    model.quant, model.hard = saved[0], saved[1]
    model.train(saved[2])


def fit(model: Net, train, dev, *, steps=(3000, 2000, 3000), lrs=(0.01, 0.005, 0.002), bs: int = 256, alpha: float = 0.5,
        wd: float = 0.0, recal: bool = False, log=None, seed: int | None = None) -> dict:
    """The three-stage recipe of training/train_brain.py on `train` / `dev` = (tok, tgt, valid) (see `pack`):
    calibrate the thresholds, then float stage (smooth staircases), act-QAT (hard staircases, float weights) and full
    QAT (everything rounded, straight-through; the checkpoint with the best dev integer cross-entropy is kept).
    steps / lrs: one entry per stage. Returns {"float_dev_acc", "dev_int_acc", "dev_int_ce", "worst"}; afterwards
    `model` IS the integer model (`model.quant = model.hard = True`) and `Transformer.from_torch` exports it."""
    log = log or (lambda s: None)
    if seed is not None:
        torch.manual_seed(seed)
    calibrate(model, train, seed=seed)
    run_stage(model, "float", steps[0], lrs[0], False, False, alpha, train, dev, bs, log, wd=wd)
    model.quant, model.hard = False, False
    model.eval()
    with torch.no_grad():
        lg, _, _ = model(dev[0], dev[2])
    facc = (lg.argmax(-1)[dev[2]] == dev[1][dev[2]]).double().mean().item()
    log(f"float model (smooth staircases, unrounded) dev teacher-forced acc {facc:.4f}")
    if recal:
        calibrate(model, train, seed=seed)
    run_stage(model, "act-qat", steps[1], lrs[1], False, True, alpha, train, dev, bs, log, wd=wd)
    run_stage(model, "full-qat", steps[2], lrs[2], True, True, alpha, train, dev, bs, log, wd=wd, keep_best=True)
    acc, _, ice, worst = evaluate(model, dev)
    model.quant, model.hard = True, True
    model.eval()
    return dict(float_dev_acc=facc, dev_int_acc=acc, dev_int_ce=ice, worst=worst)


def save(model: Net, path) -> None:
    """Write `model` (configuration, parameters, thresholds' units, calibration flag) to one file; `load` restores a GPT
    whose `Transformer.from_torch` gives the same QModel (same sha256 of the parameters)."""
    torch.save(dict(config=model.config, state=model.state_dict(), units={k: v.clone() for k, v in model.units.items()},
                    calibrated=model._calibrated, quant=model.quant, hard=model.hard, alpha=model.alpha), path)


def load(path) -> Net:
    """The GPT `save` wrote, in the mode it was saved in (integer after `fit`)."""
    ck = torch.load(path, weights_only=True)
    model = Net(**ck["config"])
    model.load_state_dict(ck["state"])
    model.units = {k: v.clone() for k, v in ck["units"].items()}
    model._calibrated, model.quant, model.hard, model.alpha = ck["calibrated"], ck["quant"], ck["hard"], ck["alpha"]
    model.eval()
    return model


def to_qmodel(model: Net, vocab: str, calib_tokens=None) -> "intref.QModel":
    """The integer model of `model`: every parameter rounded (`export`), thresholds as trained. `calib_tokens`
    (tok, tgt, valid) calibrates the thresholds first if the model was never calibrated or trained."""
    if not isinstance(model, Net):
        raise TypeError(
            f"Transformer.from_torch lowers a doomformer.torch_models.GPT, not a {type(model).__name__}: a transformer "
            "is a chain of staircases whose thresholds, residual gains and rounded weights are trained together "
            "(torch_models.fit), so an arbitrary float transformer cannot be lowered by rounding. Train a GPT with "
            "torch_models.fit (or copy your data into its recipe) and lower that")
    if len(vocab) != model.V:
        raise ValueError(f"vocab has {len(vocab)} symbols, the model {model.V}")
    if not model._calibrated:
        if calib_tokens is None:
            raise ValueError("this GPT has never been calibrated or trained: pass calib_tokens=(tok, tgt, valid) or run "
                             "torch_models.fit")
        calibrate(model, calib_tokens)
    from .kit import limit_sequences

    qm = model.qmodel(vocab)
    errs = intref.check_limits(qm, limit_sequences(qm))
    if errs:
        raise ValueError("the lowered model leaves DOOM's exact ranges: " + "; ".join(errs[:4]))
    return qm


class FloatGPT(nn.Module):
    """A plain float GPT of the same shape (pre-norm RMSNorm with gain, causal softmax attention, ReLU MLP, learned
    positions, no staircases): the reference a `GPT`'s perplexity is compared with. Not lowerable."""

    def __init__(self, V: int, T: int, d: int, H: int, dm: int, L: int):
        super().__init__()
        self.V, self.T, self.H = V, T, H
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(nn.ModuleDict(dict(
            n1=nn.RMSNorm(d), qkv=nn.Linear(d, 3 * d), o=nn.Linear(d, d), n2=nn.RMSNorm(d),
            up=nn.Linear(d, dm), dn=nn.Linear(dm, d))) for _ in range(L))
        self.out = nn.Linear(d, V)

    def forward(self, tok):
        B, n = tok.shape
        x = self.tok(tok) + self.pos(torch.arange(n))[None]
        for b in self.blocks:
            q, k, v = (u.reshape(B, n, self.H, -1).transpose(1, 2) for u in b["qkv"](b["n1"](x)).chunk(3, -1))
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, n, -1)
            x = x + b["o"](a)
            x = x + b["dn"](F.relu(b["up"](b["n2"](x))))
        return self.out(x)
