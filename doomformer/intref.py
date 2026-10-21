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

Model (L layers, ReLU attention, no LayerNorm)
----------------------------------------------
x   = adc_signed(tok_emb[tok_t] + pos_emb[t], thr_x0)
for each layer l (parameters "L{l}.<name>"):
  q,k,v = adc_signed(W @ x + b, thr_q|thr_k|thr_v)               per position
  a_ts  = adc_relu(sum_{d in head} q_t,d * k_s,d, thr_a)          causal s <= t
  ao_t  = adc_signed(sum_{s<=t} a_ts * v_s (per head), thr_o[t])  (thr_o[t] can fold a 1/(t+1))
  x1    = adc_signed(Wo @ ao + r1 * x + bo, thr_x1)               (r1: per-dim residual weight)
  m     = adc_relu(Wup @ x1 + bup, thr_m)
  x     = adc_signed(Wdn @ m + r2 * x1 + bdn, thr_x2)
logits = Wun @ x + bun ; greedy: argmax (lowest index on ties)

Ablation (`forward(..., ablate=...)`): zero listed units of an activation before any consumer reads
it. Keys: "x0", "L{l}.ao", "L{l}.x1", "L{l}.m", "L{l}.x2". In DOOM this is closing the zone doors of
that unit's blocks in every consuming tile.

DOOM constraints the params must satisfy (checked by `check_limits`)
-------------------------------------------------------------------
* every value that feeds a gate (all z except logits) has |z| <= PORT_MAX: gate heights are 16.16
  fixed point and every floor/ceiling difference the engine computes must stay below 32768
* every weight, bias, residual weight is an int with |w| <= W_MAX
* every partial sum of every accumulation stays within +-Z_MAX: sector light is a saturating
  int16 and dolls cross lines in arbitrary order (bias + all positive terms, bias + all negative terms)
* thresholds: positive nondecreasing ints <= PORT_MAX
"""
from __future__ import annotations

import json
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


def param_names(L: int) -> list[str]:
    return list(GLOBAL_ARRAYS) + [f"L{l}.{n}" for l in range(L) for n in LAYER_ARRAYS]


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

    @property
    def dh(self) -> int:
        return self.d // self.H

    def layer(self, l: int) -> dict:
        return {n: self.p[f"L{l}.{n}"] for n in LAYER_ARRAYS}

    def thr_in(self, l: int) -> np.ndarray:
        """Thresholds of the staircase that produces layer l's input x."""
        return self.p["thr_x0"] if l == 0 else self.p[f"L{l - 1}.thr_x2"]

    @classmethod
    def load(cls, path: Path | str) -> "QModel":
        """Load a model directory (config.json + params.npz), given as a path or the name of a shipped model."""
        path = shipped.resolve(path)
        cfg = json.loads((path / "config.json").read_text())
        arrs = np.load(path / "params.npz")
        return cls(**cfg, p={k: arrs[k].astype(np.int64) for k in param_names(cfg["L"])})

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        cfg = dict(vocab=self.vocab, T=self.T, d=self.d, H=self.H, dm=self.dm, K=self.K, KA=self.KA,
                   KM=self.KM, L=self.L)
        (path / "config.json").write_text(json.dumps(cfg, indent=1))
        np.savez(path / "params.npz", **{k: np.asarray(self.p[k], dtype=np.int64) for k in param_names(self.L)})


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
        zq, zk, zv = (x @ q_[W].T + q_[b] for W, b in (("Wq", "bq"), ("Wk", "bk"), ("Wv", "bv")))
        q, k, v = adc_signed(zq, q_["thr_q"]), adc_signed(zk, q_["thr_k"]), adc_signed(zv, q_["thr_v"])
        qh, kh, vh = (u.reshape(n, m.H, m.dh) for u in (q, k, v))
        score = np.einsum("thd,shd->hts", qh, kh)
        a = adc_relu(score, q_["thr_a"]) * causal
        zo = np.einsum("hts,she->the", a, vh).reshape(n, m.d)
        ao = _ablated(np.stack([adc_signed(zo[t], q_["thr_o"][t]) for t in range(n)]), pre + "ao", ablate)
        z1 = ao @ q_["Wo"].T + q_["r1"] * x + q_["bo"]
        x1 = _ablated(adc_signed(z1, q_["thr_x1"]), pre + "x1", ablate)
        zm = x1 @ q_["Wup"].T + q_["bup"]
        mh = _ablated(adc_relu(zm, q_["thr_m"]), pre + "m", ablate)
        z2 = mh @ q_["Wdn"].T + q_["r2"] * x1 + q_["bdn"]
        x_next = _ablated(adc_signed(z2, q_["thr_x2"]), pre + "x2", ablate)
        tr.update({pre + k_: v_ for k_, v_ in dict(x=x, zq=zq, zk=zk, zv=zv, q=q, k=k, v=v, score=score, a=a,
                                                     zo=zo, ao=ao, z1=z1, x1=x1, zm=zm, m=mh, z2=z2,
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
    for name in param_names(m.L):
        a = m.p[name]
        if name.split(".")[-1].startswith("thr"):
            if (a < 1).any() or (a > PORT_MAX).any() or (np.diff(a, axis=-1) < 0).any():
                errs.append(f"{name}: thresholds must be positive nondecreasing <= {PORT_MAX}")
        elif np.abs(a).max() > W_MAX:
            errs.append(f"{name}: |w| max {np.abs(a).max()} > {W_MAX}")
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
            for z, W, b in (("zq", "Wq", "bq"), ("zk", "Wk", "bk"), ("zv", "Wv", "bv")):
                extreme(pre + z, xbar(tr[pre + "x"], q_[W]), q_[b])
            qh, kh, vh = (tr[pre + u].reshape(n, m.H, m.dh) for u in ("q", "k", "v"))
            extreme(pre + "score", np.einsum("thd,shd->htsd", qh, kh))
            extreme(pre + "zo", np.einsum("hts,she->thes", tr[pre + "a"], vh).reshape(n, m.d, n))
            extreme(pre + "z1", np.concatenate([xbar(tr[pre + "ao"], q_["Wo"]),
                                                (q_["r1"] * tr[pre + "x"])[..., None]], -1), q_["bo"])
            extreme(pre + "zm", xbar(tr[pre + "x1"], q_["Wup"]), q_["bup"])
            extreme(pre + "z2", np.concatenate([xbar(tr[pre + "m"], q_["Wdn"]),
                                                (q_["r2"] * tr[pre + "x1"])[..., None]], -1), q_["bdn"])
            gate_fed += [pre + z for z in ("zq", "zk", "zv", "score", "zo", "z1", "zm", "z2")]
        extreme("logits", xbar(tr[f"L{m.L - 1}.x2"], p["Wun"]), p["bun"])
        for key in gate_fed:
            final[key] = max(final.get(key, 0), int(np.abs(tr[key]).max()))
    errs += [f"partial sums of {k} reach {v} > {Z_MAX}" for k, v in worst.items() if v > Z_MAX]
    errs += [f"|{k}| reaches {v} > {PORT_MAX} (feeds gates)" for k, v in final.items() if v > PORT_MAX]
    return errs
