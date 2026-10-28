"""Causal interventions on the integer transformer: zero, mean, resample, scale, set, add_direction, project_out.

An intervention names a site (an integer activation of the model), the positions and the units (or heads) it
changes, and a mode. `intref.forward(m, tokens, interventions=[...])` applies it to the activation array before any
consumer reads it, exactly where `ablate=` zeroes: the result is the reference of the intervened model.
`DoomTransformer(model, tiles, interventions=[...])` runs the same intervened model on the dolls and
`check_tokens` compares every exposed value with that reference at tolerance 0.

Sites (layer l, "L{l}.<name>"; x0 has no layer)
-----------------------------------------------
x0, L{l}.x1, L{l}.x2, L{l}.q, L{l}.k, L{l}.v   signed staircase counts, d units
L{l}.ao                                         attention output counts (relu attention: signed staircase of
                                                zo with thr_o[t]; softmax attention: the count of the P/N
                                                comparison lanes, up to LV per sign), d units
L{l}.m                                          MLP hidden counts, dm units (relu MLP: staircase of zm; gated MLP
                                                (reglu): zero only)
ao, q, k and v are split in H heads of dh consecutive units: `heads=[h]` means units h*dh .. (h+1)*dh - 1.

Modes
-----
zero                 the activation is 0
set                  the activation is `value` (int, or an array broadcast to [positions, units])
mean                 the activation is the rounded mean (half up) of a clean dataset's activation: per position
                     (by="position") or over all positions (by="all")
resample             the activation is the one of a clean run (`clean` tokens) at `src_pos` (default: same position)
scale                the activation becomes trunc(x * num / den), clipped to the site's range
direction            x + alpha * u (an integer direction u over the selected units), clipped to the site's range
project              x - c * u with c = trunc((x . u) / (u . u)), clipped: the component along u, up to one integer step
Resolved values (set, mean, resample) are fixed integers before the run; scale, direction and project read the run's own
activation (DOOM: the host takes the count of the pre-activation the dolls computed, applies the mode, writes the new z).
A value must be reachable: the staircase counts #{k: z >= thr[k]} skip a level when two thresholds are equal.

How DOOM applies an intervention
--------------------------------
zero at x0, ao, x1, m, x2: the zone doors of the unit's blocks in every consuming tile are closed (the mechanism of
`ablate=`); the doors are chosen per position. Every other case is a patch of the bus: the host reads the
pre-activation z the dolls computed (z0, zq, zk, zv, zo, z1, zm, z2), computes the staircase count of that integer,
applies the intervention to the count, and writes z' (and -z') = the smallest integer with that count into the ports
of the consuming tiles. Softmax attention's ao has no single z: the host writes the P/N lanes of the new count
(P_k = 1 for k < v, or N_k = 1 for k < -v, all others 0) into the out tile's ports. The patched values are then consumed by dolls only. The check covers: the DOOM-read z of every
tile output (which depends on the patches upstream), all tensors and negation twins downstream, the logits, and for
every patched cell that the count of z' equals the reference's patched activation. It does not cover the choice of
the patch value itself: a wrong patch value yields an internally consistent wrong run (see `tests/test_interventions.py`
for the predicted-difference test).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from .intref import QModel

LEGACY = ("x0", "ao", "x1", "m", "x2")           # sites with zone doors (what ablate= closes)
SITE_NAMES = ("ao", "x1", "m", "x2", "q", "k", "v")
HEAD_SITES = ("ao", "q", "k", "v")
MODES = ("zero", "set", "mean", "resample", "scale", "direction", "project")
_SITE = re.compile(r"^(?:x0|L(\d+)\.(ao|x1|m|x2|q|k|v))$")


def site_key(name: str, layer: int | None = None) -> str:
    """"x0" for the embedding, else "L{layer}.{name}"."""
    return "x0" if name == "x0" else f"L{layer}.{name}"


def parse_site(key: str, m: QModel) -> tuple[int | None, str]:
    """(layer or None, kind) of a site key; ValueError for an unknown site or a layer the model does not have."""
    hit = _SITE.match(str(key))
    if not hit:
        raise ValueError(f"unknown site {key!r}: use 'x0' or 'L<layer>.<{'|'.join(SITE_NAMES)}>'")
    if hit.group(1) is None:
        return None, "x0"
    layer = int(hit.group(1))
    if layer >= m.L:
        raise ValueError(f"site {key!r}: the model has {m.L} layer(s)")
    return layer, hit.group(2)


def width(m: QModel, key: str) -> int:
    return m.dm if key.endswith(".m") else m.d


def all_sites(m: QModel) -> list[str]:
    """Every site key of the model, in forward order."""
    return ["x0"] + [f"L{l}.{n}" for l in range(m.L) for n in ("q", "k", "v", "ao", "x1", "m", "x2")]


def _tuple(v) -> tuple | None:
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return (int(v),)
    return tuple(int(i) for i in v)


@dataclass(frozen=True)
class Intervention:
    """One intervention (see the module docstring); build it with zero/set_value/mean/resample/scale/add_direction/
    project_out."""
    site: str
    mode: str
    units: tuple | None = None       # unit indices of the site (None: all, unless heads is given)
    heads: tuple | None = None       # head indices (ao, q, k, v): the heads' units
    pos: tuple | None = None         # positions (None: every position, or the donor's for resample)
    value: Any = None                # set: int or array broadcast to [len(pos), len(units)]
    num: int = 1                     # scale: x -> trunc(x * num / den)
    den: int = 1
    clean: Any = None                # resample: one token list; mean: a list of token lists
    src_pos: tuple | None = None     # resample: donor positions (default: pos)
    by: str = "position"             # mean: "position" or "all"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {self.mode!r}")
        for name in ("units", "heads", "pos", "src_pos"):
            object.__setattr__(self, name, _tuple(getattr(self, name)))
        if self.units is not None and self.heads is not None:
            raise ValueError("give units or heads, not both")
        if self.mode == "scale" and (not isinstance(self.num, (int, np.integer)) or self.den < 1):
            raise ValueError("scale needs an integer num and an integer den >= 1")
        if self.mode in ("direction", "project"):
            u = np.asarray(self.value)
            if u.ndim != 1 or not np.issubdtype(u.dtype, np.integer) or not u.any():
                raise ValueError("a direction is a nonzero 1-d integer vector over the selected units")
            if self.mode == "direction" and (isinstance(self.num, bool) or not isinstance(self.num, (int, np.integer))):
                raise ValueError("alpha must be an integer")
        if self.by not in ("position", "all"):
            raise ValueError("by must be 'position' or 'all'")


def zero(site: str, *, units=None, heads=None, pos=None) -> Intervention:
    return Intervention(site, "zero", units=units, heads=heads, pos=pos)


def set_value(site: str, value, *, units=None, heads=None, pos=None) -> Intervention:
    return Intervention(site, "set", units=units, heads=heads, pos=pos, value=value)


def scale(site: str, num: int, den: int = 1, *, units=None, heads=None, pos=None) -> Intervention:
    return Intervention(site, "scale", units=units, heads=heads, pos=pos, num=int(num), den=int(den))


def add_direction(site: str, u, alpha: int = 1, *, units=None, heads=None, pos=None) -> Intervention:
    """x -> x + alpha * u at the chosen positions, clipped to the site's count range. u: a nonzero integer vector with
    one entry per selected unit (the direction); alpha: an integer (negative subtracts). The sum is exact in integers; the
    clip is the only nonlinearity (a staircase cannot output a count beyond its levels)."""
    return Intervention(site, "direction", units=units, heads=heads, pos=pos, value=np.asarray(u), num=int(alpha))


def project_out(site: str, u, *, units=None, heads=None, pos=None) -> Intervention:
    """Remove the component of x along the integer vector u: x -> x - c * u with c = trunc((x . u) / (u . u)) (rounded
    toward zero, so after the step |x' . u| < u . u: the component is removed up to one integer step, not to exactly 0
    unless u . u divides x . u)."""
    return Intervention(site, "project", units=units, heads=heads, pos=pos, value=np.asarray(u))


def resample(site: str, clean, *, units=None, heads=None, pos=None, src_pos=None) -> Intervention:
    """The activation of the clean run `clean` (token list) at src_pos (default: the same position)."""
    return Intervention(site, "resample", units=units, heads=heads, pos=pos, clean=list(clean),
                        src_pos=src_pos)


def mean(site: str, clean, *, units=None, heads=None, pos=None, by: str = "position") -> Intervention:
    """The rounded (half up) mean activation over the clean token lists `clean`."""
    return Intervention(site, "mean", units=units, heads=heads, pos=pos, clean=[list(c) for c in clean], by=by)


@dataclass(frozen=True)
class Op:
    """A resolved intervention: explicit positions and units, values fixed (set) or computed per run (scale, direction,
    project)."""
    key: str
    mode: str                        # zero, set, scale, direction, project
    pos: tuple
    units: tuple
    values: Any = None               # set: int64 array [len(pos), len(units)]; direction, project: the vector [len(units)]
    num: int = 1
    den: int = 1


class Plan:
    """Resolved interventions of one model, in application order. `ablate` zeros come first."""

    def __init__(self, m: QModel, ops: list[Op]):
        self.m, self.ops = m, ops
        self.by_key: dict[str, list[Op]] = {}
        for op in ops:
            self.by_key.setdefault(op.key, []).append(op)
        for key, group in self.by_key.items():
            layer, kind = parse_site(key, m)
            if kind == "m" and m.mlp == "reglu" and any(o.mode != "zero" for o in group):
                raise ValueError(f"{key}: the gated MLP supports zero only (m = g * u is a product of two gates)")

    def __bool__(self):
        return bool(self.ops)

    # -- thresholds of the staircase that produces a site
    def thr(self, key: str, t: int) -> tuple[np.ndarray, bool]:
        """(threshold row of position t, signed) of the staircase behind site `key`."""
        p, layer, kind = self.m.p, *parse_site(key, self.m)
        if kind == "x0":
            return p["thr_x0"], True
        pre = f"L{layer}."
        if kind == "ao":
            return p[pre + "thr_o"][t], True
        if kind == "m":
            return p[pre + "thr_m"], False
        return p[pre + {"x1": "thr_x1", "x2": "thr_x2", "q": "thr_q", "k": "thr_k", "v": "thr_v"}[kind]], True

    def lanes(self, key: str) -> bool:
        """True for the softmax-attention ao: a count of P/N comparison lanes, not the staircase of one value."""
        layer, kind = parse_site(key, self.m)
        return kind == "ao" and self.m.attn == "softmax"

    def levels(self, key: str) -> tuple[int, bool]:
        """(largest count, counts may be negative) of site `key`."""
        if self.lanes(key):
            return self.m.lv(parse_site(key, self.m)[0]), True
        thr, signed = self.thr(key, 0)
        return len(thr), signed

    def reachable(self, v: np.ndarray, key: str, t: int) -> np.ndarray:
        """Which counts of `v` site `key` can output at position t (a staircase skips a level where two thresholds are
        equal; the P/N lanes of softmax ao reach every count up to LV)."""
        v = np.asarray(v)
        if self.lanes(key):
            return np.abs(v) <= self.levels(key)[0]
        thr, signed = self.thr(key, t)
        K = len(thr)
        a = np.abs(v)
        ok = (a <= K) & ((v >= 0) | signed)
        nxt = np.append(thr, np.iinfo(np.int64).max)
        lvl = np.clip(a, 1, K)
        return ok & ((a == 0) | (thr[lvl - 1] < nxt[lvl]))

    def preimage(self, v: np.ndarray, key: str, t: int) -> np.ndarray:
        """The smallest-magnitude integer z whose staircase count is v (v must be reachable)."""
        thr, _ = self.thr(key, t)
        a = np.abs(v)
        z = np.where(a == 0, 0, thr[np.clip(a, 1, len(thr)) - 1])
        return np.where(v < 0, -z, z)

    def count(self, z: np.ndarray, key: str, t: int) -> np.ndarray:
        """Staircase count of z at site `key`, position t."""
        thr, signed = self.thr(key, t)
        z = np.asarray(z, dtype=np.int64)
        up = (z[..., None] >= thr).sum(-1)
        return up - (((-z)[..., None] >= thr).sum(-1) if signed else 0)

    def apply(self, x: np.ndarray, key: str, pos0: int = 0) -> np.ndarray:
        """x [n, units]: the activation of rows = positions pos0 .. pos0 + n - 1; returns x with every op of `key`
        applied in order (a copy). Positions outside x are skipped."""
        group = self.by_key.get(key)
        if not group:
            return x
        x = x.copy()
        n = len(x)
        for op in group:
            sel = [(i, p - pos0) for i, p in enumerate(op.pos) if 0 <= p - pos0 < n]
            if not sel:
                continue
            ii, rows = np.array([s[0] for s in sel]), np.array([s[1] for s in sel])
            cols = np.array(op.units)
            cur = x[np.ix_(rows, cols)]
            if op.mode == "zero":
                new = np.zeros_like(cur)
            elif op.mode == "set":
                new = op.values[ii]
            elif op.mode == "scale":
                new = np.sign(cur) * (np.abs(cur) * op.num // op.den)
                top, signed = self.levels(key)
                new = np.clip(new, -top if signed else 0, top)
            else:
                u = np.asarray(op.values, dtype=np.int64)
                if op.mode == "direction":
                    new = cur + op.num * u[None, :]
                else:
                    dot = cur @ u
                    new = cur - (np.sign(dot) * (np.abs(dot) // int(u @ u)))[:, None] * u[None, :]
                top, signed = self.levels(key)
                new = np.clip(new, -top if signed else 0, top)
            if op.mode != "zero":
                for r, row in zip(rows, new):
                    bad = ~self.reachable(row, key, r + pos0)
                    if bad.any():
                        raise ValueError(f"{key} position {r + pos0}: count {int(row[bad][0])} is not reachable by "
                                         f"its staircase (levels out of range or equal thresholds)")
            x[np.ix_(rows, cols)] = new
        return x

    # -- DOOM side
    def door_site(self, key: str) -> bool:
        """True when every op of `key` is a zero at a site with zone doors: DOOM closes doors, no bus patch."""
        group = self.by_key.get(key, [])
        return bool(group) and key.split(".")[-1] in LEGACY and all(o.mode == "zero" for o in group)

    def door_units(self, key: str, t: int) -> list[int]:
        """Units whose doors are closed at position t for site `key` (empty unless door_site)."""
        if not self.door_site(key):
            return []
        return sorted({u for o in self.by_key[key] if t in o.pos for u in o.units})

    def patched_keys(self) -> list[str]:
        """Sites DOOM patches on the bus."""
        return [k for k in self.by_key if not self.door_site(k)]

    def signature(self, t: int) -> tuple:
        """Hashable description of everything this plan does at positions <= t (cache key of the DOOM prefix)."""
        out = []
        for op in self.ops:
            if any(p <= t for p in op.pos):
                vals = None if op.values is None else op.values.tobytes()
                out.append((op.key, op.mode, op.pos, op.units, vals, op.num, op.den))
        return tuple(out)


def _units(iv: Intervention, m: QModel, key: str) -> tuple:
    n = width(m, key)
    kind = key.split(".")[-1]
    if iv.heads is not None:
        if kind not in HEAD_SITES:
            raise ValueError(f"{key}: heads apply to {HEAD_SITES} only")
        if any(not 0 <= h < m.H for h in iv.heads):
            raise ValueError(f"{key}: heads must be in 0..{m.H - 1}")
        return tuple(h * m.dh + e for h in iv.heads for e in range(m.dh))
    if iv.units is None:
        return tuple(range(n))
    if any(not 0 <= u < n for u in iv.units):
        raise ValueError(f"{key}: units must be in 0..{n - 1}")
    return iv.units


def _donor(m: QModel, tokens, key: str, cache: dict) -> np.ndarray:
    from .intref import forward
    tokens = tuple(int(t) for t in tokens)
    if tokens not in cache:
        tr: dict = {}
        forward(m, list(tokens), tr)
        cache[tokens] = tr
    return cache[tokens][key]


def compile_plan(m: QModel, ablate: dict | None = None, interventions: Iterable | "Plan" | None = None,
                 cache: dict | None = None) -> Plan:
    """Resolve `ablate` (zeros, all positions) and `interventions` (Intervention objects, or dicts of their fields)
    into a Plan. Donor runs for mean and resample are plain intref.forward calls of the same model; `cache` (a dict
    you keep) holds their traces between calls."""
    if isinstance(interventions, Plan):
        if ablate:
            raise ValueError("pass ablate inside the Plan, not next to it")
        return interventions
    ivs = [Intervention(**i) if isinstance(i, dict) else i for i in (interventions or [])]
    ivs = [zero(k, units=list(u)) for k, u in (ablate or {}).items() if len(u)] + ivs
    ops, cache = [], {} if cache is None else cache
    for iv in ivs:
        parse_site(iv.site, m)
        key, units = iv.site, _units(iv, m, iv.site)
        pos = iv.pos
        if iv.mode == "zero":
            ops.append(Op(key, "zero", pos if pos is not None else tuple(range(m.T)), units))
            continue
        if iv.mode == "scale":
            ops.append(Op(key, "scale", pos if pos is not None else tuple(range(m.T)), units, num=iv.num, den=iv.den))
            continue
        if iv.mode in ("direction", "project"):
            if len(iv.value) != len(units):
                raise ValueError(f"{key}: the direction has {len(iv.value)} entries for {len(units)} selected units")
            ops.append(Op(key, iv.mode, pos if pos is not None else tuple(range(m.T)), units,
                          values=np.asarray(iv.value, dtype=np.int64), num=iv.num))
            continue
        if iv.mode == "set":
            pos = pos if pos is not None else tuple(range(m.T))
            vals = np.broadcast_to(np.asarray(iv.value, dtype=np.int64), (len(pos), len(units))).copy()
        elif iv.mode == "resample":
            if not iv.clean:
                raise ValueError("resample needs the clean tokens")
            src = iv.src_pos if iv.src_pos is not None else (pos if pos is not None else tuple(range(len(iv.clean))))
            pos = pos if pos is not None else src
            if len(src) != len(pos):
                raise ValueError("src_pos and pos must have the same length")
            act = _donor(m, iv.clean, key, cache)
            if any(not 0 <= s < len(act) for s in src):
                raise ValueError(f"src_pos outside the clean run ({len(act)} positions)")
            vals = act[np.ix_(list(src), list(units))].astype(np.int64)
        else:                                                    # mean
            if not iv.clean:
                raise ValueError("mean needs a list of clean token lists")
            acts = [_donor(m, c, key, cache) for c in iv.clean]
            rounded = lambda rows: (2 * np.sum(rows, 0) + len(rows)) // (2 * len(rows))
            if iv.by == "all":
                pooled = np.concatenate([a[:, list(units)] for a in acts])
                pos = pos if pos is not None else tuple(range(m.T))
                vals = np.tile(rounded(pooled), (len(pos), 1)).astype(np.int64)
            else:
                longest = max(len(a) for a in acts)
                pos = pos if pos is not None else tuple(range(longest))
                rows = []
                for p in pos:
                    have = [a[p, list(units)] for a in acts if p < len(a)]
                    if not have:
                        raise ValueError(f"mean: no clean sequence reaches position {p}")
                    rows.append(rounded(np.array(have)))
                vals = np.array(rows, dtype=np.int64)
        if any(not 0 <= p < m.T for p in pos):
            raise ValueError(f"positions must be in 0..{m.T - 1}")
        ops.append(Op(key, "set", tuple(pos), units, values=vals))
    return Plan(m, ops)


def fold(m: QModel, interventions: Iterable | Plan | None = None, ablate: dict | None = None) -> QModel:
    """A copy of the model with constant interventions compiled into its weights: for every position the consumers of an
    intervened unit read a fixed count v, so column u of each consumer weight is zeroed and W[:, u] * v moves into the
    consumer's bias (the residual weights r1, r2 likewise). The folded model's pre-activations, scores and logits equal
    the intervened reference's; only the intervened sites' own activation arrays differ (the folded model still computes
    them, nobody reads them). This puts an intervention into a one-map brain, which has no host between its layers.

    Supported: zero, set, and mean(by="all") at every position, at x0, ao, x1, m, x2, for norm="none". Not foldable:
    scale and per-position values (they are not constants), q/k/v (their consumers multiply them), RMSNorm models (the
    norm reads the unit directly)."""
    plan = compile_plan(m, ablate, interventions)
    if m.norm != "none":
        raise ValueError("fold: RMSNorm models read the unit in the norm; use the tile path (DoomTransformer)")
    final: dict[tuple[str, int], int] = {}
    for op in plan.ops:
        _, kind = parse_site(op.key, m)
        if kind in ("q", "k", "v"):
            raise ValueError(f"fold: {op.key} is read by products (q.k, a.v); use the tile path")
        if op.mode in ("scale", "direction", "project"):
            raise ValueError(f"fold: {op.mode} depends on the run's activation; use the tile path")
        if op.pos != tuple(range(m.T)):
            raise ValueError("fold: the intervention must cover every position (pos=None)")
        vals = np.zeros((m.T, len(op.units)), dtype=np.int64) if op.mode == "zero" else op.values
        if (vals != vals[0]).any():
            raise ValueError("fold: the value must be the same at every position")
        if op.mode != "zero":
            for t in range(m.T):
                if not plan.reachable(vals[t], op.key, t).all():
                    raise ValueError(f"{op.key}: a value is not reachable by its staircase")
        for u, v in zip(op.units, vals[0]):
            final[(op.key, u)] = int(v)
    p = {k: v.copy() for k, v in m.p.items()}

    def column(w: str, b: str, u: int, v: int):
        p[b] = p[b] + p[w][:, u] * v
        p[w][:, u] = 0

    def residual(r: str, b: str, u: int, v: int):
        p[b] = p[b].copy()
        p[b][u] += p[r][u] * v
        p[r][u] = 0

    for (key, u), v in final.items():
        layer, kind = parse_site(key, m)
        if kind == "x0" or (kind == "x2" and layer + 1 < m.L):
            nxt = 0 if kind == "x0" else layer + 1
            for n in "qkv":
                column(f"L{nxt}.W{n}", f"L{nxt}.b{n}", u, v)
            residual(f"L{nxt}.r1", f"L{nxt}.bo", u, v)
        elif kind == "x2":
            column("Wun", "bun", u, v)
        elif kind == "ao":
            column(f"L{layer}.Wo", f"L{layer}.bo", u, v)
        elif kind == "x1":
            for w, b in ((("Wg", "bg"), ("Wu", "bu")) if m.mlp == "reglu" else (("Wup", "bup"),)):
                column(f"L{layer}.{w}", f"L{layer}.{b}", u, v)
            residual(f"L{layer}.r2", f"L{layer}.bdn", u, v)
        else:                                                       # m
            column(f"L{layer}.Wdn", f"L{layer}.bdn", u, v)
    return replace(m, p=p)


LANE_DOOR_TAG0 = 52000      # sector tags of runtime lane doors: one tag per (site, unit); the tile doors use 50000 + block


def lane_doors(m: QModel, ablate: dict) -> tuple[dict[str, int], dict[tuple[str, int], int]]:
    """Doors for a one-map brain that a level script can close while the level runs (pure.compile_pure(doors=)).

    ablate: the keys and units of `ablate=` (zero at x0, L{l}.ao, L{l}.x1, L{l}.m, L{l}.x2). Returns (doors, tags):
    `doors` maps the value name a lane's gate port carries (x0 unit i at position t is "z0:t:i", L{l}.ao "L{l}.zo:t:i",
    L{l}.x1 "L{l}.z1:t:i", L{l}.m "L{l}.zm:t:i", L{l}.x2 "L{l}.z2:t:i") to a tag, and `tags` maps (key, unit) to its tag.
    Closing a tag removes every consumer lane of that unit at every position: zero ablation, as `fold` and `ablate=`
    compute it. relu attention, no norm and the relu MLP only (softmax ao is read through P/N lanes, RMSNorm reads the
    unit in the norm, the gated MLP's m is a product)."""
    if m.attn != "relu" or m.norm != "none" or m.mlp != "relu":
        raise ValueError("runtime lane doors: relu attention, norm none and the relu MLP only")
    plan = compile_plan(m, ablate)
    names = {"x0": "z0", "ao": "zo", "x1": "z1", "m": "zm", "x2": "z2"}
    doors: dict[str, int] = {}
    tags: dict[tuple[str, int], int] = {}
    for op in plan.ops:
        layer, kind = parse_site(op.key, m)
        if kind not in names or op.mode != "zero":
            raise ValueError(f"runtime lane doors: zero at x0, ao, x1, m or x2 only (got {op.mode} at {op.key})")
        stem = names[kind] if layer is None else f"L{layer}.{names[kind]}"
        for u in op.units:
            tag = tags.setdefault((op.key, int(u)), LANE_DOOR_TAG0 + len(tags))
            for t in range(m.T):
                doors[f"{stem}:{t}:{u}"] = tag
    return doors, tags
