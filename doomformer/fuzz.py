"""Differential fuzzing of the compiler against the integer reference: `doomformer fuzz`.

Each case is a random integer net (MLP, spiking net or transformer) plus inputs. The net is compiled into a DOOM map,
the inputs run through a real game and every exposed value is compared with intref at tolerance 0 (the checks of
df.compile's brains: values, the set of names read back, a readout guard). A mismatch, an unexpected exception or a
planted bug the fuzzer misses is a failure; the exit status is 1 then.

    doomformer fuzz --nets 40 --seed 0 --workers 3          # MLPs
    doomformer fuzz --kind transformer --nets 20            # all norm / mlp / attn variants
    doomformer fuzz --edge --nets 200                       # weights +-255, hidden values at +-12,000
    doomformer fuzz --replay fuzz-failures.json             # re-run the saved failing cases
    doomformer fuzz --nets 40 --plant gate-gt               # self-test: a planted compiler bug must be found

Net i of seed S is drawn from numpy's SeedSequence([S, i, kind]), so a case is the same for any worker count. Inputs
of an MLP: zero, +-1, +-12,000, the threshold-1 / threshold / threshold+1 neighbours of the input staircase (both
signs), and, for hidden layers, inputs that put a hidden unit exactly on a threshold neighbour or on +-12,000 /
+-11,999 (the generator shifts that unit's bias). Inputs the net's range check refuses are counted, not run.
A net whose layer has all-zero weights cannot be lowered (no doll carries it): the generator makes a few MLPs of
that kind and requires the compiler's clear message, reported as `unsupported`; a transformer whose random weight
matrix came out all zero is reported the same way (its message names the layer).
"""
from __future__ import annotations

import contextlib
import inspect
import json
import os
import re
import secrets
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import _mp
from . import planes as _planes

KINDS = ("mlp", "spiking", "transformer")
FORMAT = "doomformer-fuzz-replay-1"
PORT_MAX = 12000
PLANE_MAX = _planes.PLANE_MAX
START_MAX, START_MIN = _planes.START_MAX, _planes.START_MIN
W_MAX = 255
UNSUPPORTED_MSG = "all zero"            # fragment of kit's all-zero-layer message
VOCAB = "$ABCDEFGH"

PLANTS = {
    "gate-gt": "an input gate compares x > t instead of x >= t (ops.adc_lanes: limit one lower)",
    "w255": "weight lines clamp |w| at 254 (ops.fx)",
    "bias-twin": "the negated twin of a negative hidden bias is off by one (ops.output_pair)",
    "dup-thr": "duplicate thresholds collapse into one gate (ops.adc_lanes)",
}


# --- planted compiler bugs (the fuzzer's self-test) ---------------------------------------------------------------
@contextlib.contextmanager
def plant(name: str | None):
    """Monkeypatch one known lowering bug into doomformer.ops for the duration (and restore it)."""
    if name is None:
        yield
        return
    if name not in PLANTS:
        raise ValueError(f"unknown plant {name!r}; known: {', '.join(PLANTS)}")
    from . import kit, ops
    from .tile import Effect, Gate
    saved = dict(adc_lanes=ops.adc_lanes, fx=ops.fx, output_pair=ops.output_pair)
    kit_pair = kit.output_pair
    if name == "gate-gt":
        ops.adc_lanes = lambda spec, x, thr, sign: [Gate(spec.port(ops.neg(x) if sign > 0 else x, "floor"),
                                                         -int(t) - 1, "floor") for t in thr]
    elif name == "dup-thr":
        ops.adc_lanes = lambda spec, x, thr, sign: saved["adc_lanes"](spec, x, sorted(set(int(t) for t in thr)), sign)
    elif name == "w255":
        def fx(pair, w):
            w = max(-254, min(254, int(w)))
            return [] if not w else [Effect(pair[0], w)] + ([Effect(pair[1], -w)] if pair[1] is not None else [])
        ops.fx = fx
    elif name == "bias-twin":
        ops.output_pair = kit.output_pair = lambda spec, nm, bias=0: (
            spec.output(nm, bias), spec.output(ops.neg(nm), -int(bias) + (1 if bias < 0 else 0)))
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(ops, k, v)
        kit.output_pair = kit_pair


# --- (de)serialisation: a case is plain JSON ------------------------------------------------------------------------
def _lst(a):
    return np.asarray(a).tolist()


def spec_of(net) -> dict:
    from .intref import QModel
    from .kit import MLP
    if isinstance(net, MLP):
        return dict(weights=[_lst(W) for W in net.weights], biases=[_lst(b) for b in net.biases],
                    thresholds=[_lst(t) for t in net.thresholds], signed_input=net.signed_input)
    if isinstance(net, QModel):
        cfg = dict(vocab=net.vocab, T=net.T, d=net.d, H=net.H, dm=net.dm, K=net.K, KA=net.KA, KM=net.KM, L=net.L,
                   attn=net.attn, R=net.R, norm=net.norm, Rn=net.Rn, mlp=net.mlp)
        return dict(config=cfg, p={k: _lst(v) for k, v in net.p.items()})
    return dict(names=list(net.names), inputs=list(net.inputs), W=_lst(net.W), U=_lst(net.U), theta=int(net.theta),
                leak=int(net.leak))


def net_of(kind: str, spec: dict):
    if kind == "mlp":
        from .kit import MLP
        return MLP(spec["weights"], spec["biases"], spec["thresholds"], signed_input=spec["signed_input"])
    if kind == "transformer":
        from .intref import QModel
        return QModel(**spec["config"], p={k: np.asarray(v, dtype=np.int64) for k, v in spec["p"].items()})
    from .spiking import IntNet
    return IntNet(spec["names"], spec["inputs"], spec["W"], spec["U"], spec["theta"], spec["leak"])


def rng_for(seed: int, index: int, kind: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(index), KINDS.index(kind)]))


# --- generators -----------------------------------------------------------------------------------------------------
def _draw_thr(rng, K: int, hi: int, dup: float = 0.3, edge: bool = False) -> np.ndarray:
    t = np.sort(rng.integers(1, max(1, min(hi, PORT_MAX)) + 1, K))
    for k in range(1, K):
        if rng.random() < dup:
            t[k] = t[k - 1]
    if edge and rng.random() < 0.4:
        t[-1] = PORT_MAX - int(rng.integers(0, 2))
    return np.sort(t)


def _thr_from(rng, h: np.ndarray, K: int, edge: bool) -> np.ndarray:
    """Thresholds drawn from the hidden values the calibration inputs produced, so inputs land on them."""
    pos = np.unique(h[h >= 1])
    if not len(pos):
        return _draw_thr(rng, K, 8, edge=edge)
    t = np.sort(rng.choice(pos, K, replace=True))
    if edge and rng.random() < 0.3:
        t[-1] = PORT_MAX - int(rng.integers(0, 2))
    return np.sort(np.clip(t, 1, PORT_MAX))


def _mlp_inputs(rng, net, extra: list[np.ndarray], n: int, planes: bool = True) -> tuple[list[list[int]], int]:
    """At most n inputs the net accepts (net.check); returns (inputs, how many candidates it refused)."""
    from .kit import RangeError
    t0 = net.thresholds[0]
    n_in = net.n_in
    cand: list[np.ndarray] = list(extra)
    cand += [np.zeros(n_in, np.int64), np.ones(n_in, np.int64), -np.ones(n_in, np.int64)]
    for _ in range(3):                                              # threshold neighbourhoods, both signs
        x = rng.integers(-2, 3, n_in)
        i = int(rng.integers(0, n_in))
        x[i] = int(rng.choice(t0)) + int(rng.integers(-1, 2))
        cand.append(x if rng.random() < 0.5 else _neg_at(x, i))
    for v in (PORT_MAX, -PORT_MAX):
        cand.append(np.full(n_in, v, np.int64))
        cand.append(rng.choice([PORT_MAX, -PORT_MAX, 0, 1], n_in))
    scale = int(min(PORT_MAX, max(2, 2 * int(t0.max()))))
    cand += [rng.integers(-scale, scale + 1, n_in) for _ in range(4)]
    out, refused, seen = [], 0, set()
    for x in cand:
        x = np.clip(np.asarray(x, dtype=np.int64), -PORT_MAX, PORT_MAX)
        key = tuple(x.tolist())
        if key in seen:
            continue
        seen.add(key)
        try:
            net.check(x, planes=planes)
        except RangeError:
            refused += 1
            continue
        out.append([int(v) for v in x])
    order = rng.permutation(len(out)) if len(out) > n else np.arange(len(out))
    keep = sorted(order[:n].tolist())
    return [out[i] for i in keep], refused


def _neg_at(x: np.ndarray, i: int) -> np.ndarray:
    y = x.copy()
    y[i] = -y[i]
    return y


def gen_mlp(rng, edge: bool, n_inputs: int = 8, planes: bool = True) -> dict:
    from .intref import adc_relu, adc_signed
    from .kit import MLP, RangeError
    for attempt in range(40):
        depth = int(rng.integers(1, 5))
        n_in = int(rng.integers(1, 6))
        widths = [n_in] + [int(rng.integers(1, 9)) for _ in range(depth)]
        signed = bool(rng.random() < 0.7)
        mag = 255 if edge and rng.random() < 0.7 else int(rng.choice([3, 9, 40, 255])) >> (attempt // 10)
        bmag = 12000 if edge and rng.random() < 0.5 else int(rng.choice([3, 50, 2000]))
        zero_layer = None if edge or rng.random() > 0.03 else int(rng.integers(0, depth))
        Ws, bs = [], []
        for l in range(depth):
            W = rng.integers(-max(1, mag), max(1, mag) + 1, (widths[l + 1], widths[l]))
            W[rng.random(W.shape) < 0.25] = 0
            if edge or rng.random() < 0.1:
                W[rng.random(W.shape) < 0.3] = rng.choice([-W_MAX, W_MAX])
            if not W.any():
                W[0, 0] = 1
            if l == zero_layer:
                W[:] = 0
            Ws.append(W)
            bs.append(rng.integers(-bmag, bmag + 1, widths[l + 1]))
        K0 = int(rng.integers(1, 5))
        thr = [_draw_thr(rng, K0, int(rng.choice([3, 20, 300, PORT_MAX])), edge=edge)]
        xs = rng.integers(-2 * int(thr[0].max()) - 1, 2 * int(thr[0].max()) + 2, (32, n_in)).clip(-PORT_MAX, PORT_MAX)
        a = adc_signed(xs, thr[0]) if signed else adc_relu(xs, thr[0])
        for l in range(depth - 1):
            h = a @ Ws[l].T + bs[l]
            thr.append(_thr_from(rng, h, int(rng.integers(1, 5)), edge))
            a = adc_relu(h, thr[-1])
        net = MLP(Ws, bs, thr, signed_input=signed)
        ok = []
        for x in xs:
            try:
                net.check(x, planes=planes)
                ok.append(x)
            except RangeError:
                pass
        if len(ok) >= 2:
            break
    else:
        raise RuntimeError("fuzz: could not draw an MLP the range check accepts")
    extra = []
    if depth >= 2 and ok:                                           # a hidden unit on a threshold or the +-12,000 edge
        for _ in range(2):
            l, j = int(rng.integers(0, depth - 1)), None
            x = ok[int(rng.integers(0, len(ok)))]
            j = int(rng.integers(0, widths[l + 1]))
            h = int(net.reference(x)["hidden"][l][j])
            targets = [PORT_MAX, -PORT_MAX, PORT_MAX - 1, -PORT_MAX + 1] if edge else []
            targets += [int(t) + int(d) for t in net.thresholds[l + 1] for d in (-1, 0, 1)]
            target = int(rng.choice(targets))
            b2 = [b.copy() for b in net.biases]
            b2[l][j] += target - h
            try:
                shifted = MLP(net.weights, b2, net.thresholds, signed_input=signed)
                shifted.check(x, planes=planes)
            except (RangeError, ValueError):
                continue
            net = shifted
            extra += [x, x + 1, x - 1]
    if edge and planes and depth >= 2 and ok:                                  # a hidden plane right at the plane-move bound
        for _ in range(2):
            l = int(rng.integers(0, depth - 1))
            x = ok[int(rng.integers(0, len(ok)))]
            j = int(rng.integers(0, widths[l + 1]))
            gap = int(rng.integers(0, 4))                           # land the partial sum 0..3 below PLANE_MAX
            terms = net.reference(x)["activations"][l] * net.weights[l][j]
            up, down = int(np.clip(terms, 0, None).sum()), int(-np.clip(terms, None, 0).sum())
            wants = ([PLANE_MAX - gap - up] if up else []) + ([gap + down - PLANE_MAX] if down else [])
            if not wants:
                continue
            b2 = [b.copy() for b in net.biases]
            b2[l][j] = int(rng.choice(wants))
            try:
                shifted = MLP(net.weights, b2, net.thresholds, signed_input=signed)
                shifted.check(x, planes=planes)
            except (RangeError, ValueError):
                continue
            net = shifted
            extra += [x, x + 1, x - 1]
    if edge and planes and depth >= 2 and ok:                       # wide one-way planes at the start bounds
        for _ in range(2):
            target = int(rng.choice([START_MAX, START_MAX - 1, START_MAX - 9, START_MIN, START_MIN + 1, START_MIN + 9]))
            pairs = [(l, j, x) for l in range(depth - 1) for x in ok[:6]
                     for j, d in enumerate((net.reference(x)["activations"][l] * net.weights[l]).sum(1))
                     if abs(d - target) <= PORT_MAX]                # h ends at -target + d: inside +-12,000
            if not pairs:
                continue
            l, j, x = pairs[int(rng.integers(0, len(pairs)))]
            b2 = [b.copy() for b in net.biases]
            b2[l][j] = -target                                      # the twin plane -h starts at `target`
            try:
                shifted = MLP(net.weights, b2, net.thresholds, signed_input=signed)
                shifted.check(x, planes=planes)
            except (RangeError, ValueError):
                continue
            net = shifted
            extra += [x, x + 1, x - 1]
    inputs, refused = _mlp_inputs(rng, net, [np.asarray(x) for x in extra] + [np.asarray(x) for x in ok[:2]], n_inputs,
                                  planes)
    return dict(kind="mlp", spec=spec_of(net), inputs=inputs, refused_inputs=refused,
                planned="unsupported" if zero_layer is not None else "ok",
                note=f"depth {depth}, widths {widths}, {'signed' if signed else 'unsigned'} input"
                     + (", edge" if edge else ""))


def gen_spiking(rng, edge: bool, n_inputs: int = 8) -> dict:
    from .spiking import IntNet
    n, m = int(rng.integers(2, 7)), int(rng.integers(1, 4))
    theta = int(rng.integers(2, 9))
    leak = int(rng.integers(0, 3))
    mag = 3
    W = rng.integers(-mag, mag + 1, (n, n))
    U = rng.integers(-mag, mag + 2, (n, m))
    net = IntNet([f"n{i}" for i in range(n)], [f"u{i}" for i in range(m)], W, U, theta, leak)
    steps = [[int(v) for v in (rng.random(m) < 0.6)] for _ in range(max(4, n_inputs))]
    return dict(kind="spiking", spec=spec_of(net), inputs=steps, refused_inputs=0, planned="ok",
                note=f"{n} neurons, {m} inputs, theta {theta}, leak {leak}")


def gen_transformer(rng, edge: bool, n_inputs: int = 4) -> dict:
    from .intref import QModel, check_limits
    attn = str(rng.choice(["relu", "softmax"]))
    norm = str(rng.choice(["none", "rms"]))
    mlp = str(rng.choice(["relu", "reglu"]))
    for attempt in range(60):
        V, T = int(rng.integers(3, 6)), int(rng.integers(2, 4))
        H, dh = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        d, dm = H * dh, int(rng.integers(2, 5))
        K, KA, KM = (int(rng.integers(1, 4)) for _ in range(3))
        L = 2 if rng.random() < 0.15 else 1
        R = int(rng.integers(1, 3)) if attn == "softmax" else 1
        Rn = int(rng.integers(2 if d == 1 else 1, 4)) if norm == "rms" else 1     # d == 1, Rn == 1: no levels
        mag = 3 if attempt < 30 else 1
        w = lambda *shape: rng.integers(-mag, mag + 1, shape, dtype=np.int64)          # noqa: E731
        thr = lambda k, hi=5: _draw_thr(rng, k, hi, dup=0.3)                           # noqa: E731
        p = dict(tok_emb=w(V, d), pos_emb=w(T, d), thr_x0=thr(K), Wun=w(V, d), bun=w(V))
        for l in range(L):
            q = {}
            for k in "qkv":
                n_thr = int(rng.integers(2 if (k == "v" and attn == "softmax" and R == 1) else 1, 4))
                q["W" + k], q["b" + k], q["thr_" + k] = w(d, d), w(d), thr(n_thr)
            q["thr_a"] = thr(KA, 8)
            if attn == "relu":
                q["thr_o"] = np.stack([thr(KM) for _ in range(T)])
            q.update(Wo=w(d, d), r1=w(d), bo=w(d), thr_x1=thr(K), Wdn=w(d, dm), r2=w(d), bdn=w(d), thr_x2=thr(K))
            if mlp == "reglu":
                q.update(Wg=w(dm, d), bg=w(dm), thr_g=thr(KM), Wu=w(dm, d), bu=w(dm), thr_u=thr(KM))
            else:
                q.update(Wup=w(dm, d), bup=w(dm), thr_m=thr(KM))
            p.update({f"L{l}.{k}": v for k, v in q.items()})
        m = QModel(VOCAB[:V], T, d, H, dm, K, KA, KM, L, p, attn=attn, R=R, norm=norm, Rn=Rn, mlp=mlp)
        seqs = [[0] + [int(t) for t in rng.integers(0, V, int(rng.integers(0, T)))] for _ in range(n_inputs - 1)]
        seqs.append([int(t) for t in rng.integers(0, V, T)])
        if not check_limits(m, seqs, planes=True):
            break
    else:
        raise RuntimeError("fuzz: could not draw a transformer inside the range")
    return dict(kind="transformer", spec=spec_of(m), inputs=seqs, refused_inputs=0, planned="ok",
                note=f"attn {attn}, norm {norm}, mlp {mlp}, L={L} d={d} H={H} T={T} V={V}")


def gen_case(kind: str, seed: int, index: int, edge: bool = False, n_inputs: int | None = None,
             planes: bool = True) -> dict:
    rng = rng_for(seed, index, kind)
    if kind == "mlp":
        # a quarter of the nets are range-edge nets even without --edge
        case = gen_mlp(rng, edge or rng.random() < 0.25, n_inputs or 8, planes)
    elif kind == "spiking":
        case = gen_spiking(rng, edge, n_inputs or 8)
    else:
        case = gen_transformer(rng, edge, n_inputs or 4)
    return dict(id=index, seed=int(seed), **case)


# --- running one case -----------------------------------------------------------------------------------------------
_OPTS: dict = {}


def _init_worker(plant_name, opts):
    _mp.exit_with_parent()
    _OPTS.update(opts)
    if plant_name:
        plant(plant_name).__enter__()                              # for the life of the worker


def _compile_kwargs(opts: dict, kind: str = "mlp") -> dict:
    from .kit import compile as df_compile
    kw = {}
    if kind == "spiking":                                          # no guard, no opt levels for stateful maps
        return kw
    if kind == "mlp":                                              # the per-input contract is what is fuzzed: nets
        kw["limits"] = False                                       # whose worst case leaves the plane range are run too
    if opts.get("tiles"):
        if kind != "mlp":
            raise ValueError("--tiles fuzzes MLPs only (tile mode of transformers: doomformer verify, tests/test_tiled.py)")
        return kw                                                  # tiles: no guard (one map only), no schedule levels
    if opts.get("guard", True):
        kw["guard"] = True
    if opts.get("opt") is not None:
        if "opt" not in inspect.signature(df_compile).parameters:
            raise ValueError("this build's df.compile has no `opt` option")
        opt = str(opts["opt"])
        kw["opt"] = int(opt) if opt.isdigit() else frozenset(opt.split(","))
    return kw


def run_case(case: dict, opts: dict | None = None) -> dict:
    """Compile and run one case in a fresh game; returns the result record (status ok | mismatch | unsupported |
    refused | error)."""
    from .kit import MismatchError, RangeError, ReadoutError
    from .kit import compile as df_compile
    opts = opts if opts is not None else _OPTS
    kind = case["kind"]
    res = dict(id=case["id"], kind=kind, status="ok", passes=0, values=0, refused_inputs=case.get("refused_inputs", 0),
               note=case.get("note", ""), seconds=0.0)
    t0 = time.perf_counter()
    try:
        net = net_of(kind, case["spec"])
        kw = _compile_kwargs(opts, kind)
    except Exception as e:
        return dict(res, status="error", detail=f"{type(e).__name__}: {e}")
    with tempfile.TemporaryDirectory(prefix="dfz-") as d:
        try:
            if opts.get("tiles"):
                from .orchestrator import TilePool
                brain = df_compile(net, Path(d) / "net.tiles", mode="tiles", workers=1, **kw)
                brain._pool = TilePool(brain.wad, 1, max_live=net.n_layers + 1)    # one live game per layer
            else:
                brain = df_compile(net, Path(d) / "net.wad", mode="map", **kw)
        except RangeError as e:
            return dict(res, status="refused", detail=str(e), seconds=time.perf_counter() - t0)
        except ValueError as e:
            expected = (case.get("planned") == "unsupported" and UNSUPPORTED_MSG in str(e)) or (
                kind == "transformer" and "has no dolls" in str(e))     # an all-zero weight matrix
            return dict(res, status="unsupported" if expected else "error", detail=str(e),
                        seconds=time.perf_counter() - t0)
        except Exception as e:
            return dict(res, status="error", detail=f"compile: {type(e).__name__}: {e}")
        if case.get("planned") == "unsupported":
            brain.close()
            return dict(res, status="error", detail="an all-zero layer compiled; it was expected to be refused")
        with brain:
            try:
                for x in case["inputs"]:
                    try:
                        if kind == "mlp" and opts.get("tiles"):
                            got = brain.run(np.asarray(x), check=True)
                            if set(got) != net.exposed_names():
                                raise ReadoutError(f"tile readback names differ: missing "
                                                   f"{sorted(net.exposed_names() - set(got))[:4]}")
                        elif kind == "mlp":
                            brain(np.asarray(x))
                        elif kind == "spiking":
                            brain.step(np.asarray(x))
                        else:
                            brain.logits(x)
                    except RangeError as e:
                        res["refused_inputs"] += 1
                        res.setdefault("first_refusal", str(e)[:300])
                        continue
                    res["passes"] += 1
                    res["values"] += int(brain.info["checked"])
            except MismatchError as e:
                res.update(status="mismatch", input=x, mismatches=[list(m) for m in e.mismatches[:8]],
                           detail=str(e)[:400])
            except Exception as e:
                res.update(status="error", input=x, detail=f"{type(e).__name__}: {e}"[:400])
    if res["status"] == "ok" and not res["passes"] and res.get("first_refusal"):
        res.update(status="refused", detail="every input is refused by the range check: " + res["first_refusal"])
    res["seconds"] = round(time.perf_counter() - t0, 3)
    return res


def _work(case: dict) -> dict:
    return run_case(case)


# --- campaign -------------------------------------------------------------------------------------------------------
def run_cases(cases: list[dict], *, workers: int = 1, plant_name: str | None = None, opts: dict | None = None,
              progress=None, stop_on_fail: bool = False) -> list[dict]:
    """Run cases; workers == 1 runs in this process (so a monkeypatched compiler is seen), else a spawn pool of
    whole-net workers. Results are returned in case order."""
    opts = dict(opts or {})
    _compile_kwargs(opts)                                          # fail early on an unsupported option
    results: dict[int, dict] = {}
    if workers <= 1 or len(cases) <= 1:
        with plant(plant_name):
            for c in cases:
                r = run_case(c, opts)
                results[c["id"]] = r
                if progress:
                    progress(r, len(results), len(cases))
                if stop_on_fail and r["status"] in ("mismatch", "error"):
                    break
    else:
        with _mp.detached_main(), ProcessPoolExecutor(max_workers=min(workers, len(cases)), mp_context=_mp.context(),
                                                      initializer=_init_worker,
                                                      initargs=(plant_name, opts)) as pool:
            futs = {pool.submit(_work, c): c for c in cases}
            try:
                for f in as_completed(futs):
                    c = futs[f]
                    try:
                        r = f.result()
                    except Exception as e:                         # a worker died
                        r = dict(id=c["id"], kind=c["kind"], status="error", passes=0, values=0, refused_inputs=0,
                                 note=c.get("note", ""), seconds=0.0, detail=f"worker: {type(e).__name__}: {e}")
                    results[c["id"]] = r
                    if progress:
                        progress(r, len(results), len(cases))
                    if stop_on_fail and r["status"] in ("mismatch", "error"):
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    return [results[c["id"]] for c in cases if c["id"] in results]


def summarize(results: list[dict], **meta) -> dict:
    n = lambda s: sum(1 for r in results if r["status"] == s)       # noqa: E731
    variants: dict[str, dict[str, int]] = {}
    for r in results:                                               # transformers: attn / norm / mlp variant of each net
        m = re.match(r"attn (\w+), norm (\w+), mlp (\w+)", r.get("note", ""))
        if m:
            v = variants.setdefault("/".join(m.groups()), dict(nets=0, passes=0, values=0))
            v["nets"] += 1
            v["passes"] += r["passes"]
            v["values"] += r["values"]
    return dict(**meta, nets=len(results), ok=n("ok"), mismatches=n("mismatch"), errors=n("error"),
                unsupported=n("unsupported"), refused=n("refused"), passes=sum(r["passes"] for r in results),
                values=sum(r["values"] for r in results), refused_inputs=sum(r["refused_inputs"] for r in results),
                variants=dict(sorted(variants.items())) or None,
                failures=[r for r in results if r["status"] in ("mismatch", "error")])


def fuzz(nets: int = 40, seed: int = 0, kind: str = "mlp", edge: bool = False, workers: int = 1,
         plant_name: str | None = None, opts: dict | None = None, inputs: int | None = None, start: int = 0,
         progress=None, stop_on_fail: bool = False) -> dict:
    """Generate `nets` cases of `kind` (indices start..start+nets-1 of `seed`), run them and return the summary dict
    (with `failures`, the failing result records, and `cases`, the failing cases for a replay file)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    cases = [gen_case(kind, seed, i, edge, inputs, not (opts or {}).get("tiles")) for i in range(start, start + nets)]
    t0 = time.perf_counter()
    results = run_cases(cases, workers=workers, plant_name=plant_name, opts=opts, progress=progress,
                        stop_on_fail=stop_on_fail)
    out = summarize(results, kind=kind, seed=seed, edge=edge, plant=plant_name, workers=workers,
                    opt=(opts or {}).get("opt"), tiles=bool((opts or {}).get("tiles")), start=start, source_sha256=source_hash(),
                    seconds=round(time.perf_counter() - t0, 1))
    bad = {r["id"] for r in out["failures"]}
    out["cases"] = [c for c in cases if c["id"] in bad]
    return out


def source_hash() -> str:
    """Short SHA-256 over every doomformer/*.py (the compiler and checkers a campaign ran on)."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def save_replay(path, cases: list[dict]) -> None:
    Path(path).write_text(json.dumps(dict(format=FORMAT, cases=cases)))


def load_replay(path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    return data["cases"]


# --- CLI ------------------------------------------------------------------------------------------------------------
def _seed(text: str):
    return "random" if text == "random" else int(text)


def add_arguments(p) -> None:
    p.add_argument("--nets", type=int, default=40, help="random nets to generate (default 40)")
    p.add_argument("--seed", type=_seed, default=0, help="seed (an integer, or `random`: a fresh one, printed); net i "
                   "of a seed is the same for any worker count")
    p.add_argument("--kind", choices=KINDS, default="mlp", help="what to fuzz (default mlp)")
    p.add_argument("--workers", type=int, default=1, help="DOOM games at once (default 1; at most 3 on a shared box)")
    p.add_argument("--edge", action="store_true", help="range-edge nets only: weights +-255, hidden values at "
                   "+-12,000, thresholds at 12,000, large biases")
    p.add_argument("--inputs", type=int, help="inputs per net (default 8 for mlp and spiking, 4 for transformer)")
    p.add_argument("--start", type=int, default=0, help="first net index (default 0)")
    p.add_argument("--opt", help="schedule level (0-3, or a comma list of pass names), passed to df.compile as `opt`")
    p.add_argument("--no-guard", action="store_true", help="compile without the readout guard")
    p.add_argument("--tiles", action="store_true", help="MLPs in tile mode (one game per layer, Python copies the "
                   "integer layer values) instead of one map")
    p.add_argument("--plant", choices=sorted(PLANTS), help="self-test: plant a known compiler bug; exit 1 when found")
    p.add_argument("--mutate", action="store_true", help="mutation harness instead: see doomformer.mutate")
    p.add_argument("--per-class", type=int, default=20, help="--mutate: mutants sampled per class (default 20)")
    p.add_argument("--tile", type=int, metavar="K", help="--mutate: mutate operation tile K of the tile-mode lowering "
                   "(the map of one layer) instead of the one-map brain")
    p.add_argument("--learning", action="store_true", help="--mutate: mutate the online perceptron map (the learning "
                   "map: weights are ceiling heights the dolls change) instead of an MLP")
    p.add_argument("--controls", action="store_true", help="--mutate: also run the aggregate controls of the survivor "
                   "rules (whole mechanisms changed at once)")
    p.add_argument("--net", metavar="FILE", help="--mutate: an MLP .npz to mutate (default: the built-in 3-4-3 net)")
    p.add_argument("--replay", metavar="FILE", help="re-run the cases of a replay file (written on failure)")
    p.add_argument("--json", metavar="FILE", help="write the summary record as JSON")
    p.add_argument("--save-failures", metavar="FILE", default="fuzz-failures.json",
                   help="replay file written when a case fails (default fuzz-failures.json)")
    p.add_argument("--stop-on-fail", action="store_true", help="stop at the first mismatch or error")
    p.add_argument("-q", "--quiet", action="store_true", help="no per-net progress")


def run_cli(args) -> int:
    if args.mutate:
        from . import mutate
        return mutate.run_cli(args)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    opts = dict(guard=not args.no_guard, opt=args.opt, tiles=args.tiles)
    if args.tiles and args.kind != "mlp" and not args.replay:
        print("doomformer fuzz: --tiles fuzzes MLPs only", file=sys.stderr)
        return 2
    if args.seed == "random":                                      # a fresh seed per run (CI), printed so a failure replays
        args.seed = secrets.randbelow(2 ** 31)
        print(f"fuzz: random seed {args.seed}", flush=True)

    def progress(r, done, total):
        if r["status"] in ("ok", "unsupported") and (args.quiet or (done % 10 and done != total)):
            return
        tag = {"ok": "ok", "mismatch": "MISMATCH", "error": "ERROR", "unsupported": "unsupported",
               "refused": "refused"}[r["status"]]
        print(f"  [{done}/{total}] net {r['id']} {r['kind']} {tag} ({r['passes']} passes, {r['values']} values)"
              + (f": {r.get('detail', '')[:160]}" if tag not in ("ok",) else ""), flush=True)

    t0 = time.perf_counter()
    try:
        if args.replay:
            cases = load_replay(args.replay)
            results = run_cases(cases, workers=args.workers, plant_name=args.plant, opts=opts, progress=progress)
            out = summarize(results, kind="replay", seed=None, edge=None, plant=args.plant, workers=args.workers,
                            seconds=round(time.perf_counter() - t0, 1))
            bad = {r["id"] for r in out["failures"]}
            out["cases"] = [c for c in cases if c["id"] in bad]
        else:
            out = fuzz(args.nets, args.seed, args.kind, args.edge, args.workers, args.plant, opts, args.inputs,
                       args.start, progress, args.stop_on_fail)
    except (ValueError, RuntimeError) as e:
        print(f"doomformer fuzz: {e}", file=sys.stderr)
        return 2
    failed = bool(out["failures"])
    print(f"fuzz: {out['nets']} {out['kind']} nets, {out['passes']} passes, {out['values']:,} values compared at "
          f"tolerance 0, {out['mismatches']} mismatches, {out['errors']} errors, {out['unsupported']} unsupported "
          f"(all-zero layer), {out['refused']} refused, {out['refused_inputs']} inputs refused by the range check; "
          f"seed {out['seed']}, {out['seconds']} s")
    if args.json:
        Path(args.json).write_text(json.dumps({k: v for k, v in out.items() if k != "cases"}, indent=1) + "\n")
    if failed and out["cases"] and not args.plant:                 # a planted bug is expected: no replay file
        save_replay(args.save_failures, out["cases"])
        print(f"fuzz: {len(out['cases'])} failing case(s) saved to {args.save_failures} "
              f"(re-run: doomformer fuzz --replay {args.save_failures})")
    if args.plant:
        print(f"fuzz: planted bug {args.plant!r} ({PLANTS[args.plant]}) " + ("FOUND" if failed else "NOT FOUND"))
    return 1 if failed else 0
