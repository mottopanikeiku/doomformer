"""Integer reference semantics of the DOOM transformer ("the contract").

Every operation here maps 1:1 onto a DOOM mechanism; DOOM tiles and the training code must
reproduce these integers bit-for-bit.

Primitives
----------
adc_signed(z, thr)  staircase activation. thr: nondecreasing positive ints (len K).
                    c+ = #{k: z >= thr[k]},  c- = #{k: -z >= thr[k]};  returns c+ - c-  in [-K, K]
adc_relu(z, thr)    c+ only                  in [0, K]
crossbar            z = b + W @ x            (dolls cross weight lines; light accumulates)
multiplier          z = sum_i a_i * b_i      (a dolls climb a corridor of b's gates)

Model (L layers, attention "relu" or "softmax", normalisation "none" or "rms", MLP "relu" or "reglu")
-----------------------------------------------------------------------------------------------------
x   = adc_signed(tok_emb[tok_t] + pos_emb[t], thr_x0)
for each layer l (parameters "L{l}.<name>"):
  xa    = x, or rmsnorm(x) with norm="rms"                         (see below)
  q,k,v = adc_signed(W @ xa + b, thr_q|thr_k|thr_v)              per position
  a_ts  = adc_relu(sum_{d in head} q_t,d * k_s,d, thr_a)          causal s <= t
  relu:    ao_t = adc_signed(sum_{s<=t} a_ts * v_s (per head), thr_o[t])  (thr_o[t] can fold a 1/(t+1))
  softmax: ao_t = trunc(R * sum_{s<=t} a_ts * v_s / (1 + sum_{s<=t} a_ts))  (see below; no thr_o)
  x1    = adc_signed(Wo @ ao + r1 * x + bo, thr_x1)               (r1: per-dim residual weight)
  xm    = x1, or rmsnorm(x1) with norm="rms"
  relu:  m = adc_relu(Wup @ xm + bup, thr_m)
  reglu: m = adc_relu(Wg @ xm + bg, thr_g) * adc_signed(Wu @ xm + bu, thr_u)   (see below)
  x     = adc_signed(Wdn @ m + r2 * x1 + bdn, thr_x2)
logits = Wun @ x + bun ; greedy: argmax (lowest index on ties)

Softmax attention (config attn="softmax", integer R >= 1)
--------------------------------------------------------
e_ts  = adc_relu(score_ts, thr_a) in [0, KA]: with log-spaced thresholds thr_a[k] ~ c + sigma*ln(k+1) this is
        min(KA, floor(exp((score - c) / sigma))), a quantised exponential (trace key "e")
den_t = 1 + sum_{s<=t} e_ts per head (the 1 is a null slot whose logit sits at the first threshold, value 0)
num_t = sum_{s<=t} e_ts * v_s  (trace key "zo"),  LV = R * len(thr_v) - 1
P_t,l = R*num_t - l*den_t + 1 ,  N_t,l = -R*num_t - l*den_t + 1        l = 1..LV  (trace keys "P", "N": [n, d, LV])
ao_t  = sum_l [P_t,l >= 1] - sum_l [N_t,l >= 1] = trunc(R*num_t / den_t)   in [-LV, LV]
The division is a count of comparisons: #{l <= LV : l*den <= R*num}; |num| < KV*den bounds the quotient by LV.

RMSNorm (config norm="rms", integer Rn >= 1; site "att" normalises x, site "mlp" normalises x1)
-----------------------------------------------------------------------------------------------
c     = the staircase counts of the normalised vector (x or x1): d values, |c_i| <= Kc = len(its thresholds)
s     = sum_i c_i^2                                       in [0, d*Kc^2]     (trace "s_att" / "s_mlp": [n])
r     = adc_relu(s, thr_r),  thr_r = d*k^2 for k = 1..Kc  = floor(sqrt(s/d))  (trace "r_att" / "r_mlp": [n])
LN    = min(Rn*Kc, isqrt(Rn^2*d - 1))                     comparison levels per sign
P_i,l = Rn*c_i - l*(1+r) + 1 ,  N_i,l = -Rn*c_i - l*(1+r) + 1        l = 1..LN  (trace "P_att", "N_att": [n, d, LN])
n_i   = sum_l [P_i,l >= 1] - sum_l [N_i,l >= 1] = trunc(Rn*c_i / (1+r))   in [-LN, LN]  (trace "n_att" / "n_mlp")
r is exactly floor(sqrt(s/d)) (the integer RMS) because s <= d*Kc^2. The 1 in 1+r keeps the divisor >= 1 (s = 0
gives r = 0 and n = 0); the + 1 in P/N turns l*(1+r) <= +-Rn*c_i into the gate test P >= 1 / N >= 1.
LN suffices: for c_i >= 0 the P count is min(LN, q), q = floor(Rn*c_i/(1+r)), and every N_i,l <= 1 - l*(1+r) < 1
(c_i < 0: mirrored). q <= Rn*Kc since 1+r >= 1; and c_i^2 <= s < d*(1+r)^2, so (Rn*c_i/(1+r))^2 < Rn^2*d,
q^2 <= Rn^2*d - 1, q <= isqrt(Rn^2*d - 1). So q <= LN and the count never saturates.
A learned gain gamma_i on n_i folds into the next weights (Wq[:, i] * gamma_i), so there are no gain arrays. The
residuals (r1 * x, r2 * x1) read the un-normalised values.

Gated MLP (config mlp="reglu"; arrays Wg, bg, thr_g, Wu, bu, thr_u replace Wup, bup, thr_m)
-----------------------------------------------------------------------------------------
zg = Wg @ xm + bg ,  zu = Wu @ xm + bu                              (trace "zg", "zu": [n, dm])
g  = adc_relu(zg, thr_g) in [0, len(thr_g)] ,  u = adc_signed(zu, thr_u) in [-len(thr_u), len(thr_u)]
m  = g * u  (trace "m") ;  z2 = Wdn @ m + r2 * x1 + bdn
In DOOM m is never a value: the g lanes of unit j climb a corridor of zu_j's gates (one corridor per sign of u),
and every passed stage adds +-Wdn[:, j] to the z2 outputs.

Ablation (`forward(..., ablate=...)`): zero listed units of an activation before any consumer reads
it. Keys: "x0", "L{l}.ao", "L{l}.x1", "L{l}.m", "L{l}.x2" (with norm="rms" an ablated x unit is also absent from
s and n). In DOOM this is closing the zone doors of that unit's blocks in every consuming tile.

DOOM constraints the params must satisfy (checked by `check_limits`)
-------------------------------------------------------------------
* every value that feeds a gate (all z except logits) has |z| <= PORT_MAX: gate heights are 16.16
  fixed point and every floor/ceiling difference the engine computes must stay below 32768
* every weight, bias, residual weight is an int with |w| <= W_MAX
* every partial sum of every accumulation stays within +-Z_MAX: sector light is a saturating
  int16 and dolls cross lines in arbitrary order (bias + all positive terms, bias + all negative terms)
* thresholds: positive nondecreasing ints <= PORT_MAX (norm="rms": also the fixed thr_r, d*Kc^2 <= PORT_MAX)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import shipped

Z_MAX = 30000      # partial sums (sector light, saturating int16)
PORT_MAX = 12000   # values written into gate heights
W_MAX = 255
GLOBAL_ARRAYS = ("tok_emb", "pos_emb", "thr_x0", "Wun", "bun")
LAYER_ARRAYS = ("Wq", "bq", "Wk", "bk", "Wv", "bv", "thr_q", "thr_k", "thr_v", "thr_a", "thr_o", "Wo", "r1",
                "bo", "thr_x1", "Wup", "bup", "thr_m", "Wdn", "r2", "bdn", "thr_x2")
ATTN = ("relu", "softmax")
NORM = ("none", "rms")
MLP = ("relu", "reglu")
RELU_MLP = ("Wup", "bup", "thr_m")
REGLU_MLP = ("Wg", "bg", "thr_g", "Wu", "bu", "thr_u")


def layer_arrays(attn: str = "relu", mlp: str = "relu") -> tuple[str, ...]:
    """Per-layer array names; softmax attention divides at run time and has no thr_o; the gated MLP (reglu) has
    Wg, bg, thr_g, Wu, bu, thr_u in place of Wup, bup, thr_m."""
    if attn not in ATTN:
        raise ValueError(f"attn must be one of {ATTN}, not {attn!r}")
    if mlp not in MLP:
        raise ValueError(f"mlp must be one of {MLP}, not {mlp!r}")
    names = LAYER_ARRAYS if attn == "relu" else tuple(n for n in LAYER_ARRAYS if n != "thr_o")
    if mlp == "reglu":
        i = names.index(RELU_MLP[0])
        names = names[:i] + REGLU_MLP + names[i + len(RELU_MLP):]
    return names


def param_names(L: int, attn: str = "relu", mlp: str = "relu") -> list[str]:
    return list(GLOBAL_ARRAYS) + [f"L{l}.{n}" for l in range(L) for n in layer_arrays(attn, mlp)]


@dataclass
class QModel:
    vocab: str          # index -> character; index 0 is the BOS/EOS marker
    T: int              # context length
    d: int
    H: int
    dm: int
    K: int              # levels per sign for signed activations
    KA: int             # levels for attention weights
    KM: int             # levels for MLP hidden
    L: int              # layers
    p: dict             # name -> np.ndarray[int64]
    attn: str = "relu"  # "relu" (thr_o staircase) or "softmax" (runtime division, see the module docstring)
    R: int = 1          # softmax only: ao = trunc(R * num / den), LV = R * len(thr_v) - 1 levels per sign
    norm: str = "none"  # "none" or "rms" (integer RMSNorm before q/k/v and before the MLP, see the module docstring)
    Rn: int = 1         # rms only: n = trunc(Rn * c / (1 + r))
    mlp: str = "relu"   # "relu" (Wup, thr_m) or "reglu" (m = relu-staircase(Wg x) * signed-staircase(Wu x))

    def __post_init__(self):
        layer_arrays(self.attn, self.mlp)
        if self.norm not in NORM:
            raise ValueError(f"norm must be one of {NORM}, not {self.norm!r}")
        for key in ("R", "Rn"):
            v = getattr(self, key)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f"{key} must be an int >= 1, not {v!r}")
            setattr(self, key, int(v))
        if self.attn == "relu" and self.R != 1:
            raise ValueError("R is a softmax-attention setting; relu attention needs R == 1")
        if self.norm == "none" and self.Rn != 1:
            raise ValueError("Rn is an RMSNorm setting; norm='none' needs Rn == 1")

    @property
    def dh(self) -> int:
        return self.d // self.H

    def layer(self, l: int) -> dict:
        return {n: self.p[f"L{l}.{n}"] for n in layer_arrays(self.attn, self.mlp)}

    def lv(self, l: int) -> int:
        """Softmax attention: comparison levels per sign of layer l's ao (R * len(thr_v) - 1)."""
        return self.R * len(self.p[f"L{l}.thr_v"]) - 1

    def thr_in(self, l: int) -> np.ndarray:
        """Thresholds of the staircase that produces layer l's input x."""
        return self.p["thr_x0"] if l == 0 else self.p[f"L{l - 1}.thr_x2"]

    def thr_norm(self, l: int, site: str) -> np.ndarray:
        """Thresholds of the staircase an RMSNorm site of layer l reads: "att" -> x (thr_in), "mlp" -> x1 (thr_x1)."""
        return self.thr_in(l) if site == "att" else self.p[f"L{l}.thr_x1"]

    def thr_r(self, l: int, site: str) -> np.ndarray:
        """RMSNorm: the fixed thresholds d*k^2 (k = 1..Kc) of r = floor(sqrt(s/d))."""
        return self.d * np.arange(1, len(self.thr_norm(l, site)) + 1, dtype=np.int64) ** 2

    def ln(self, l: int, site: str) -> int:
        """RMSNorm: comparison levels per sign of n, min(Rn*Kc, isqrt(Rn^2*d - 1)) (see the module docstring)."""
        return min(self.Rn * len(self.thr_norm(l, site)), math.isqrt(self.Rn ** 2 * self.d - 1))

    @classmethod
    def load(cls, path: Path | str) -> "QModel":
        """Load a model directory (config.json + params.npz), given as a path or the name of a shipped model.
        Configs without "attn"/"R" are relu-attention models, without "norm"/"Rn"/"mlp" plain relu-MLP models."""
        path = shipped.resolve(path)
        cfg = json.loads((path / "config.json").read_text())
        arrs = np.load(path / "params.npz")
        names = param_names(cfg["L"], cfg.get("attn", "relu"), cfg.get("mlp", "relu"))
        return cls(**cfg, p={k: arrs[k].astype(np.int64) for k in names})

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        cfg = dict(vocab=self.vocab, T=self.T, d=self.d, H=self.H, dm=self.dm, K=self.K, KA=self.KA,
                   KM=self.KM, L=self.L, attn=self.attn, R=self.R, norm=self.norm, Rn=self.Rn, mlp=self.mlp)
        (path / "config.json").write_text(json.dumps(cfg, indent=1))
        np.savez(path / "params.npz", **{k: np.asarray(self.p[k], dtype=np.int64)
                                         for k in param_names(self.L, self.attn, self.mlp)})


def adc_relu(z, thr):
    z = np.asarray(z, dtype=np.int64)
    return (z[..., None] >= thr).sum(-1)


def adc_signed(z, thr):
    z = np.asarray(z, dtype=np.int64)
    return adc_relu(z, thr) - adc_relu(-z, thr)


def _ablated(x: np.ndarray, key: str, ablate: dict | None) -> np.ndarray:
    if not ablate or not ablate.get(key):
        return x
    x = x.copy()
    x[..., list(ablate[key])] = 0
    return x


def rmsnorm(m: QModel, l: int, site: str, c: np.ndarray, trace: dict | None = None) -> np.ndarray:
    """Integer RMSNorm of the staircase counts c [n, d] at layer l, site "att" (x) or "mlp" (x1): returns
    n = trunc(Rn * c / (1 + r)), r = floor(sqrt(sum c^2 / d)); `trace` receives s, r, P, N, n as "<key>_<site>"."""
    s = (c * c).sum(-1)
    r = adc_relu(s, m.thr_r(l, site))
    lv = np.arange(1, m.ln(l, site) + 1)
    den = (1 + r)[..., None, None]
    P = m.Rn * c[..., None] - lv * den + 1
    N = -m.Rn * c[..., None] - lv * den + 1
    n = (P >= 1).sum(-1) - (N >= 1).sum(-1)
    if trace is not None:
        trace.update({f"s_{site}": s, f"r_{site}": r, f"P_{site}": P, f"N_{site}": N, f"n_{site}": n})
    return n


def forward(m: QModel, tokens: list[int], trace: dict | None = None, ablate: dict | None = None) -> np.ndarray:
    """Logits [len(tokens), V]. `trace` receives every pre-activation (DOOM-visible) and activation."""
    p, n = m.p, len(tokens)
    assert 1 <= n <= m.T
    tr = trace if trace is not None else {}
    z0 = p["tok_emb"][tokens] + p["pos_emb"][np.arange(n)]
    x = _ablated(adc_signed(z0, p["thr_x0"]), "x0", ablate)
    tr.update(z0=z0, x0=x)
    causal = np.tril(np.ones((n, n), dtype=np.int64))
    for l in range(m.L):
        q_ = m.layer(l)
        pre = f"L{l}."
        nrm = {}
        xa = rmsnorm(m, l, "att", x, nrm) if m.norm == "rms" else x
        zq, zk, zv = (xa @ q_[W].T + q_[b] for W, b in (("Wq", "bq"), ("Wk", "bk"), ("Wv", "bv")))
        q, k, v = adc_signed(zq, q_["thr_q"]), adc_signed(zk, q_["thr_k"]), adc_signed(zv, q_["thr_v"])
        qh, kh, vh = (u.reshape(n, m.H, m.dh) for u in (q, k, v))
        score = np.einsum("thd,shd->hts", qh, kh)
        a = adc_relu(score, q_["thr_a"]) * causal
        zo = np.einsum("hts,she->the", a, vh).reshape(n, m.d)
        if m.attn == "softmax":
            den = 1 + a.sum(-1).T                                     # [n, H]: null slot + every e
            lv = np.arange(1, m.lv(l) + 1)
            den_j = np.repeat(den, m.dh, axis=1)[..., None]           # [n, d, 1]
            P = m.R * zo[..., None] - lv * den_j + 1
            N = -m.R * zo[..., None] - lv * den_j + 1
            ao = (P >= 1).sum(-1) - (N >= 1).sum(-1)
            attn = dict(e=a, den=den, P=P, N=N)
        else:
            ao = np.stack([adc_signed(zo[t], q_["thr_o"][t]) for t in range(n)])
            attn = dict(a=a)
        ao = _ablated(ao, pre + "ao", ablate)
        z1 = ao @ q_["Wo"].T + q_["r1"] * x + q_["bo"]
        x1 = _ablated(adc_signed(z1, q_["thr_x1"]), pre + "x1", ablate)
        xm = rmsnorm(m, l, "mlp", x1, nrm) if m.norm == "rms" else x1
        if m.mlp == "reglu":
            zg, zu = xm @ q_["Wg"].T + q_["bg"], xm @ q_["Wu"].T + q_["bu"]
            g, u = adc_relu(zg, q_["thr_g"]), adc_signed(zu, q_["thr_u"])
            mh = _ablated(g * u, pre + "m", ablate)
            mlp = dict(zg=zg, zu=zu, g=g, u=u)
        else:
            zm = xm @ q_["Wup"].T + q_["bup"]
            mh = _ablated(adc_relu(zm, q_["thr_m"]), pre + "m", ablate)
            mlp = dict(zm=zm)
        z2 = mh @ q_["Wdn"].T + q_["r2"] * x1 + q_["bdn"]
        x_next = _ablated(adc_signed(z2, q_["thr_x2"]), pre + "x2", ablate)
        tr.update({pre + k_: v_ for k_, v_ in dict(x=x, **nrm, zq=zq, zk=zk, zv=zv, q=q, k=k, v=v, score=score,
                                                     **attn, zo=zo, ao=ao, z1=z1, x1=x1, **mlp, m=mh, z2=z2,
                                                     x2=x_next).items()})
        x = x_next
    logits = x @ p["Wun"].T + p["bun"]
    tr["logits"] = logits
    return logits


def choose(logits: np.ndarray, temperature: float = 0.0, rng: np.random.Generator | None = None) -> int:
    """Greedy (temperature 0: argmax, lowest index on ties) or seeded sampling from softmax(logits/T)."""
    if temperature <= 0:
        return int(np.argmax(logits))
    z = (logits - logits.max()) / temperature
    pr = np.exp(z)
    return int(rng.choice(len(pr), p=pr / pr.sum()))


def generate(m: QModel, prompt: str = "", max_new: int | None = None, temperature: float = 0.0,
             seed: int = 0, ablate: dict | None = None) -> str:
    toks = [0] + [m.vocab.index(c) for c in prompt]
    rng = np.random.default_rng(seed)
    max_new = max_new if max_new is not None else m.T - len(toks)
    for _ in range(max_new):
        if len(toks) >= m.T:
            break
        nxt = choose(forward(m, toks, ablate=ablate)[-1], temperature, rng)
        if nxt == 0:
            break
        toks.append(nxt)
    return "".join(m.vocab[t] for t in toks[1:])


def check_limits(m: QModel, sequences: list[list[int]]) -> list[str]:
    """Return violations of DOOM constraints (empty list = OK)."""
    errs = []
    for name in param_names(m.L, m.attn, m.mlp):
        a = m.p[name]
        if name.split(".")[-1].startswith("thr"):
            if (a < 1).any() or (a > PORT_MAX).any() or (np.diff(a, axis=-1) < 0).any():
                errs.append(f"{name}: thresholds must be positive nondecreasing <= {PORT_MAX}")
        elif np.abs(a).max() > W_MAX:
            errs.append(f"{name}: |w| max {np.abs(a).max()} > {W_MAX}")
    sites = ("att", "mlp") if m.norm == "rms" else ()
    for l in range(m.L):
        for site in sites:
            if m.thr_r(l, site)[-1] > PORT_MAX:
                errs.append(f"L{l}.thr_r ({site}): d*Kc^2 = {m.thr_r(l, site)[-1]} > {PORT_MAX}")
    worst, final = {}, {}

    def extreme(key, terms, bias=0):
        """Dolls cross in arbitrary order and light saturates at int16: bound every partial sum."""
        hi = bias + np.clip(terms, 0, None).sum(-1)
        lo = bias + np.clip(terms, None, 0).sum(-1)
        worst[key] = max(worst.get(key, 0), int(np.abs(hi).max()), int(np.abs(lo).max()))

    def xbar(x, W):
        return x[:, None, :] * W[None, :, :]

    p = m.p
    for seq in sequences:
        tr, toks = {}, seq[: m.T]
        forward(m, toks, tr)
        n = len(toks)
        extreme("z0", np.stack([p["tok_emb"][toks], p["pos_emb"][:n]], -1))
        gate_fed = ["z0"]
        for l in range(m.L):
            q_, pre = m.layer(l), f"L{l}."
            for site in sites:
                # s: one doll per passed lane k adds 2k-1; P/N: each c doll adds +-Rn, each r doll adds -l
                c, r = tr[pre + ("x" if site == "att" else "x1")], tr[pre + "r_" + site]
                lv = np.arange(1, m.ln(l, site) + 1)
                extreme(pre + "s_" + site, c * c)
                r_l = np.broadcast_to(-lv * r[:, None, None], (n, m.d, len(lv)))
                for key, sign in (("P_", 1), ("N_", -1)):
                    rc = np.broadcast_to(sign * m.Rn * c[..., None], r_l.shape)
                    extreme(pre + key + site, np.stack([rc, r_l], -1), 1 - lv)
                gate_fed += [pre + key + site for key in ("s_", "P_", "N_")]
            xa, xm = (tr[pre + "n_att"], tr[pre + "n_mlp"]) if sites else (tr[pre + "x"], tr[pre + "x1"])
            for z, W, b in (("zq", "Wq", "bq"), ("zk", "Wk", "bk"), ("zv", "Wv", "bv")):
                extreme(pre + z, xbar(xa, q_[W]), q_[b])
            qh, kh, vh = (tr[pre + u].reshape(n, m.H, m.dh) for u in ("q", "k", "v"))
            extreme(pre + "score", np.einsum("thd,shd->htsd", qh, kh))
            a = tr[pre + ("e" if m.attn == "softmax" else "a")]
            ev = np.einsum("hts,she->thes", a, vh).reshape(n, m.d, n)
            extreme(pre + "zo", ev)
            if m.attn == "softmax":
                # each e doll adds -l to P_l and N_l once; each passed v stage adds +-R
                lv = np.arange(1, m.lv(l) + 1)
                e_j = np.repeat(a.transpose(1, 0, 2), m.dh, axis=1)[:, :, None, :] * -lv[:, None]
                for key, sign in (("P", 1), ("N", -1)):
                    rv = np.broadcast_to(sign * m.R * ev[:, :, None, :], e_j.shape)
                    extreme(pre + key, np.concatenate([rv, e_j], -1), 1 - lv)
                extreme(pre + "den", a.transpose(1, 0, 2), 1)
            extreme(pre + "z1", np.concatenate([xbar(tr[pre + "ao"], q_["Wo"]),
                                                (q_["r1"] * tr[pre + "x"])[..., None]], -1), q_["bo"])
            mlp_z = ("zg", "zu") if m.mlp == "reglu" else ("zm",)
            for z, W, b in (("zg", "Wg", "bg"), ("zu", "Wu", "bu")) if m.mlp == "reglu" else (("zm", "Wup", "bup"),):
                extreme(pre + z, xbar(xm, q_[W]), q_[b])
            extreme(pre + "z2", np.concatenate([xbar(tr[pre + "m"], q_["Wdn"]),
                                                (q_["r2"] * tr[pre + "x1"])[..., None]], -1), q_["bdn"])
            attn_fed = ("P", "N") if m.attn == "softmax" else ("zo",)
            gate_fed += [pre + z for z in ("zq", "zk", "zv", "score", *attn_fed, "z1", *mlp_z, "z2")]
        extreme("logits", xbar(tr[f"L{m.L - 1}.x2"], p["Wun"]), p["bun"])
        for key in gate_fed:
            final[key] = max(final.get(key, 0), int(np.abs(tr[key]).max()))
    errs += [f"partial sums of {k} reach {v} > {Z_MAX}" for k, v in worst.items() if v > Z_MAX]
    errs += [f"|{k}| reaches {v} > {PORT_MAX} (feeds gates)" for k, v in final.items() if v > PORT_MAX]
    return errs
