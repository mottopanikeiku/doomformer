"""Compile an integer QModel (doomformer.intref) into DOOM tile maps.

Tile kinds (one WAD each); ports are tile-local names, the orchestrator binds them to bus values:
  embed           tok, pos            -> z0_*            equality lanes (lookup tables)
  L{l}.qkv        in_*                -> zq_*, zk_*, zv_*  crossbar, staircase of the layer input
  L{l}.score      q*, k*              -> score           multiplier corridors (one head, one key position)
  L{l}.wsum_{t}   a0..a{t}, v0..v{t}  -> o               multiplier corridors over exactly t+1 positions
  L{l}.out_{t}    zo_*, in_*          -> z1_*            crossbar, adc_signed(thr_o[t]) + residual
  L{l}.mlp_up     z1_*                -> zm_*            crossbar, adc_signed(thr_x1)
  L{l}.mlp_dn     zm_*, z1_*          -> z2_*            crossbar, adc_relu(thr_m) + residual
  unembed         in_*                -> logit_*         crossbar, staircase of the last layer output
Softmax attention (attn="softmax") replaces wsum/out:
  L{l}.den_{t}    a0..a{t}            -> den             crossbar of the e lanes (thr_a), bias 1
  L{l}.wsum_{t}   a0..a{t}, v0..v{t}  -> o, P_k, N_k     ops.attend corridors (P_k, N_k: l = k+1, bias -k)
  L{l}.out        P_i_k, N_i_k, in_*  -> z1_*            crossbar of [P >= 1] (Wo) and [N >= 1] (-Wo) + residual
RMSNorm (norm="rms") adds two tiles before qkv (site "att", in_* = the layer input) and before mlp_up (site "mlp",
in_* = z1), and qkv / mlp_up then read the normalised units through indicator lanes instead of the staircase:
  L{l}.sumsq_{site}  in_*             -> s               ops.square: lane k of either sign adds 2k-1, so s = sum c^2
  L{l}.norm_{site}   in_*, s          -> P_i_k, N_i_k, r crossbar of in_i (+-Rn) and of s (thr_r = d*k^2: -(k+1),
                                                         and +1 to r); P_i_k, N_i_k: l = k+1, bias -k
The gated MLP (mlp="reglu") changes the MLP tiles:
  L{l}.mlp_up     z1_* | P/N          -> zg_*, zu_*      crossbar (Wg, Wu)
  L{l}.mlp_dn     zg_*, zu_*, z1_*    -> z2_*            ops.gated: zg_j's lanes climb zu_j's corridors (+-Wdn[:, j])
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import _mp
from .intref import QModel
from .ops import attend, crossbar, equals, gated, levels, multiply, output_pair, output_single, square
from .tile import TileMeta, TileSpec, compile_tile


def tile_specs(m: QModel) -> list[TileSpec]:
    p, d, dh, dm, V, T = m.p, m.d, m.dh, m.dm, len(m.vocab), m.T
    specs = []

    s = TileSpec("embed")
    z0 = [output_pair(s, f"z0_{j}") for j in range(d)]
    for t in range(V):
        equals(s, "tok", t, [(z0[j], p["tok_emb"][t, j]) for j in range(d)])
    for pos in range(T):
        equals(s, "pos", pos, [(z0[j], p["pos_emb"][pos, j]) for j in range(d)])
    specs.append(s)

    for l in range(m.L):
        q_, pre, thr_in = m.layer(l), f"L{l}.", m.thr_in(l)

        if m.norm == "rms":
            specs += norm_specs(m, l, "att")
        s = TileSpec(pre + "qkv")
        outs = {n: [output_pair(s, f"z{n}_{j}", q_[f"b{n}"][j]) for j in range(d)] for n in "qkv"}
        for i in range(d):
            _read(s, m, l, "att", i, f"in_{i}", thr_in, [(outs[n][j], q_[f"W{n}"][j, i]) for n in "qkv"
                                                         for j in range(d)])
        specs.append(s)

        s = TileSpec(pre + "score")
        pair = output_pair(s, "score")
        for k in range(dh):
            multiply(s, f"q{k}", q_["thr_q"], True, f"k{k}", q_["thr_k"], True, pair)
        specs.append(s)

        if m.attn == "softmax":
            specs += softmax_specs(m, l)
        else:
            for t in range(T):
                s = TileSpec(f"{pre}wsum_{t}")
                pair = output_pair(s, "o")
                for sl in range(t + 1):
                    multiply(s, f"a{sl}", q_["thr_a"], False, f"v{sl}", q_["thr_v"], True, pair)
                specs.append(s)

            for t in range(T):
                s = TileSpec(f"{pre}out_{t}")
                z1 = [output_pair(s, f"z1_{j}", q_["bo"][j]) for j in range(d)]
                for i in range(d):
                    crossbar(s, f"zo_{i}", q_["thr_o"][t], [(z1[j], q_["Wo"][j, i]) for j in range(d)])
                    crossbar(s, f"in_{i}", thr_in, [(z1[i], q_["r1"][i])])
                specs.append(s)

        if m.norm == "rms":
            specs += norm_specs(m, l, "mlp")
        s = TileSpec(pre + "mlp_up")
        if m.mlp == "reglu":
            zg = [output_pair(s, f"zg_{j}", q_["bg"][j]) for j in range(dm)]
            zu = [output_pair(s, f"zu_{j}", q_["bu"][j]) for j in range(dm)]
            for i in range(d):
                _read(s, m, l, "mlp", i, f"z1_{i}", q_["thr_x1"], [(zg[j], q_["Wg"][j, i]) for j in range(dm)]
                      + [(zu[j], q_["Wu"][j, i]) for j in range(dm)])
        else:
            zm = [output_pair(s, f"zm_{j}", q_["bup"][j]) for j in range(dm)]
            for i in range(d):
                _read(s, m, l, "mlp", i, f"z1_{i}", q_["thr_x1"], [(zm[j], q_["Wup"][j, i]) for j in range(dm)])
        specs.append(s)

        s = TileSpec(pre + "mlp_dn")
        z2 = [output_pair(s, f"z2_{j}", q_["bdn"][j]) for j in range(d)]
        for i in range(dm):
            if m.mlp == "reglu":
                gated(s, f"zg_{i}", q_["thr_g"], f"zu_{i}", q_["thr_u"], [(z2[j], q_["Wdn"][j, i]) for j in range(d)])
            else:
                crossbar(s, f"zm_{i}", q_["thr_m"], [(z2[j], q_["Wdn"][j, i]) for j in range(d)], signed=False)
        for i in range(d):
            crossbar(s, f"z1_{i}", q_["thr_x1"], [(z2[i], q_["r2"][i])])
        specs.append(s)

    s = TileSpec("unembed")
    lg = [output_single(s, f"logit_{v}", p["bun"][v]) for v in range(V)]
    for i in range(d):
        crossbar(s, f"in_{i}", m.thr_in(m.L), [(lg[v], p["Wun"][v, i]) for v in range(V)])
    specs.append(s)
    return specs


def _read(spec: TileSpec, m: QModel, l: int, site: str, i: int, x: str, thr, targets):
    """A consumer's read of input unit i: with norm="rms" the indicator lanes of the norm tile's P_i_k / N_i_k (the
    count n_i), otherwise the staircase of port x."""
    if m.norm == "rms":
        LN = m.ln(l, site)
        levels(spec, [f"P_{i}_{k}" for k in range(LN)], [f"N_{i}_{k}" for k in range(LN)], targets)
    else:
        crossbar(spec, x, thr, targets)


def norm_specs(m: QModel, l: int, site: str) -> list[TileSpec]:
    """RMSNorm tiles of layer l at `site` (see the module docstring); same arithmetic as onemap.norm_layers."""
    pre, d, thr, LN = f"L{l}.", m.d, m.thr_norm(l, site), m.ln(l, site)
    a = TileSpec(f"{pre}sumsq_{site}")
    s = output_pair(a, "s")
    for i in range(d):
        square(a, f"in_{i}", thr, s)
    b = TileSpec(f"{pre}norm_{site}")
    P = [[output_pair(b, f"P_{i}_{k}", -k) for k in range(LN)] for i in range(d)]
    N = [[output_pair(b, f"N_{i}_{k}", -k) for k in range(LN)] for i in range(d)]
    r = output_single(b, "r")
    for i in range(d):
        crossbar(b, f"in_{i}", thr, [(p, m.Rn) for p in P[i]] + [(q, -m.Rn) for q in N[i]])
    crossbar(b, "s", m.thr_r(l, site), [(r, 1)] + [(row[k], -(k + 1)) for rows in (P, N) for row in rows
                                                   for k in range(LN)], signed=False)
    return [a, b]


def softmax_specs(m: QModel, l: int) -> list[TileSpec]:
    """Softmax attention tiles of layer l (see the module docstring); same arithmetic as onemap.softmax_layers."""
    q_, pre, d, LV = m.layer(l), f"L{l}.", m.d, m.lv(l)
    specs = []
    for t in range(m.T):
        s = TileSpec(f"{pre}den_{t}")
        den = output_pair(s, "den", 1)
        for sl in range(t + 1):
            crossbar(s, f"a{sl}", q_["thr_a"], [(den, 1)], signed=False)
        specs.append(s)
    for t in range(m.T):
        s = TileSpec(f"{pre}wsum_{t}")
        num = output_pair(s, "o")
        P = [output_pair(s, f"P_{k}", -k) for k in range(LV)]
        N = [output_pair(s, f"N_{k}", -k) for k in range(LV)]
        for sl in range(t + 1):
            attend(s, f"a{sl}", q_["thr_a"], f"v{sl}", q_["thr_v"], num, P, N, m.R)
        specs.append(s)
    s = TileSpec(pre + "out")
    z1 = [output_pair(s, f"z1_{j}", q_["bo"][j]) for j in range(d)]
    for i in range(d):
        for k in range(LV):
            for key, sign in (("P", 1), ("N", -1)):
                crossbar(s, f"{key}_{i}_{k}", [1], [(z1[j], sign * q_["Wo"][j, i]) for j in range(d)], signed=False)
        crossbar(s, f"in_{i}", m.thr_in(l), [(z1[i], q_["r1"][i])])
    specs.append(s)
    return specs


def _compiler_version() -> str:
    """Hash of the code that turns a spec into a WAD: any change there invalidates every cached tile."""
    h = hashlib.sha256()
    for f in ("tile.py", "udmf.py"):
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def spec_key(spec: TileSpec, version: str) -> str:
    blob = json.dumps(dataclasses.asdict(spec), sort_keys=True, default=int)
    return hashlib.sha256((version + blob).encode()).hexdigest()[:24]


def _compile(spec: TileSpec, wad: Path, key: str) -> TileMeta:
    meta = compile_tile(spec, wad)
    wad.with_suffix(".key").write_text(key)       # written last: a crash mid-compile leaves no valid stamp
    return meta


def build(m: QModel, out_dir: Path, jobs: int | None = None) -> dict:
    """Compile every tile into out_dir; returns a manifest {tile: {dolls, sectors, ports, outputs}}.
    Tiles whose spec and compiler source are unchanged since the last build are reused (content hash in
    <tile>.key); the rest compile in parallel on `jobs` processes (default: all cores)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = _compiler_version()
    metas: dict[str, TileMeta] = {}
    todo = []
    specs = tile_specs(m)
    for spec in specs:
        wad, key = out_dir / f"{spec.name}.wad", spec_key(spec, version)
        stamp = wad.with_suffix(".key")
        if wad.exists() and stamp.exists() and stamp.read_text() == key:
            metas[spec.name] = TileMeta.load(wad.with_suffix(".json"))
        else:
            todo.append((spec, wad, key))
    if todo:
        # spawn (fork warns and can deadlock once threads exist); detached_main: scripts may lack a __main__ guard
        with _mp.detached_main(), ProcessPoolExecutor(max_workers=jobs or os.cpu_count(),
                                                      mp_context=_mp.context()) as ex:
            for (spec, _, _), meta in zip(todo, ex.map(_compile, *zip(*todo))):
                metas[spec.name] = meta
    manifest = {}
    for spec in specs:
        meta = metas[spec.name]
        manifest[spec.name] = dict(dolls=meta.n_dolls, sectors=meta.n_sectors, ports=len(meta.ports),
                                   outputs=len(meta.outputs))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
