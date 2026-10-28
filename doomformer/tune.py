"""Failure edges of the one-map schedule's timing constants, measured with the oracle (`doomformer tune`).

pure.Timing holds the schedule's timing constants in map units of the clock doll (about 16 units per tic). `doomformer
tune` builds reference nets with one constant changed and the others at their defaults, runs seeded random and extreme
inputs through every map in DOOM and compares every exposed value with the integer reference at tolerance 0. Per
constant it scans down from the default in decreasing steps until some input fails, bisects (on multiples of 8, then to
1 unit) between the largest failing value and the smallest passing value above it, and then tests every 8 units for 32
units above the edge and the shipped value. It records value -> failing inputs / inputs for every net.

edge     the smallest tested value v such that every net passes every input at v and at every tested value above v
shipped  the edge times a safety multiple (MULTIPLE), rounded up to a multiple of 8 (shipped_value)

Two modes: "plane" (opt=1: plane accumulators, the six constants of `shipped`, which opt=2 builds) and "copy" (the copy
bus of opt=3: sync_wait, clock_margin and copy_settle, written as `shipped_copy`). doomformer/tuning.json holds the
tables, the edges and both shipped dicts (pure.Timing.tuned). A joint run checks all shipped values of a mode together
on fresh inputs (`--joint`); `--check` builds opt=2 (or opt=3) maps from the committed table and exits 1 on any
mismatch.

The engine is deterministic (game tics are simulated): the same map bytes with the same inputs give the same readback,
so a value whose map equals an already measured map (a constant that does not change a net, e.g. corr_gap on an MLP)
reuses that result. At most `--workers` DOOM games run at once, each in a spawned worker process; a map's inputs are
split over the games.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .pure import OPT_PASSES, TUNING_FILE, Timing

UNITS_PER_TIC = 16           # the clock doll's speed on the conveyor, map units per game tic
# safety multiples of the edge: serial and corr_gap guard a physical limit (two plane movers on one sector in the same
# tic: the second is lost), sync_margin a modelled arrival time, the others settling times that depend on the net
MULTIPLE = {"serial": 1.25, "corr_gap": 1.25, "sync_margin": 1.25, "clock_margin": 2.0, "tele_gap": 2.0,
            "chunk_lead": 2.0, "copy_settle": 2.0}
SHIP_AT_LEAST = {"sync_margin": 64}       # 4 tics past the modelled arrival (the checker's safe margin)
OTHERS: Timing | None = None              # --shipped-others: the constants a scan does not vary (default: Timing())
TOP = {"sync_margin": 128}                # scan start of a constant whose default is None (the legacy schedule)
LOWEST = {"serial": 8, "corr_gap": 0, "sync_margin": -64, "clock_margin": 0, "tele_gap": 0, "chunk_lead": 0,
          "copy_settle": 0}
STEP = {"sync_margin": 16}                # descent step below a small top (default 8)
# nets left out of a constant's scan: fold's lanes are those of any crossbar MLP (it is there for the clock teleports)
SKIP = {"sync_margin": {"fold"}, "clock_margin": {"fold"}, "copy_settle": {"fold"}}
NETS_ALL = ("hello", "mlp24", "mlp16", "signed", "fold", "tiny", "softmax", "arch")      # generated nets, seed order
SCAN_SEED, JOINT_SEED = 100, 200   # input seed of a net: SCAN_SEED (scans) or JOINT_SEED (joint) + index in NETS_ALL
BOUNDARY_D = (0, 1, 10, 23, 24)   # MLP boundary inputs: every unit at +-(t + d) for one input threshold t
BOUNDARY_MAX = 20                  # at most this many boundary vectors per net
COARSE_STEP, COARSE_RESOLUTION, COARSE_BOUNDARY = 32, 8, 10      # --coarse
TRAINED = {"doomnames_onemap": ("DOOM", "HELL", "ZONE"),        # shipped one-map models: name -> prompts
           "doomnames_onemap_softmax": ("DOOM", "HELL"),
           "induction_onemap": ("ABCAB", "DEFD", "GHIJG")}


@dataclass(frozen=True)
class Mode:
    name: str
    passes: frozenset        # compile_pure passes of the scans and the joint run (timing given explicitly)
    check_opt: int           # what --check builds (timing read from tuning.json by the compiler)
    constants: tuple         # measuring order
    nets: tuple              # default reference nets
    table: str               # tuning.json keys: tables, shipped dict, joint run
    shipped: str
    joint: str


MODES = {
    "plane": Mode("plane", OPT_PASSES[1], 2,
                  ("sync_margin", "serial", "clock_margin", "corr_gap", "tele_gap", "chunk_lead"),
                  ("hello", "mlp24", "mlp16", "signed", "fold", "tiny", "softmax"),
                  "constants", "shipped", "joint"),
    "copy": Mode("copy", OPT_PASSES[1] | {"copy"}, 3, ("clock_margin", "copy_settle"),
                 ("hello", "mlp24", "tiny", "softmax"), "constants_copy", "shipped_copy", "joint_copy"),
}
# Timing(**shipped); sync_wait is the legacy constant (a schedule with sync_margin set does not use it)
SHIPPED_KEYS = ("serial", "sync_wait", "sync_margin", "clock_margin", "corr_gap", "tele_gap", "chunk_lead")


# --- reference nets ---------------------------------------------------------------------------------------------------
def _mlp(sizes: list[int], K: list[int], seed: int, lo: int = 15, hi: int = 40):
    """Random-weight MLP drawn as `doomformer hello` and the tuning probes draw it: weights in [-20, 20], biases in
    [-30, 30], layer l's thresholds the cumulative sums of K[l] draws in [lo, hi)."""
    from .kit import MLP
    rng = np.random.default_rng(seed)
    n = len(sizes) - 1
    return MLP([rng.integers(-20, 21, (sizes[l + 1], sizes[l])) for l in range(n)],
               [rng.integers(-30, 31, sizes[l + 1]) for l in range(n)],
               [np.cumsum(rng.integers(lo, hi, K[l])) for l in range(n)])


def _fold_mlp():
    """2-1-1 MLP whose first layer has 400 dolls on one plane: 100 thresholds per input, signed. The plus and minus
    dolls of an input are exclusive, so the layer has 200 waves: two clock chunks at serial 64 (12000 // 64 = 187 waves
    per chunk), so the clock teleports once and tele_gap/chunk_lead matter."""
    from .kit import MLP
    return MLP([np.array([[3, -2]]), np.array([[1]])], [np.array([5]), np.array([0])],
               [4 * np.arange(1, 101), np.array([1, 60, 200])])


def _softmax_model():
    """tests/test_softmax.py softmax_model(seed=1, R=2): T=3, d=4, H=2, softmax attention."""
    from .intref import QModel
    rng = np.random.default_rng(1)
    V, T, d, H, dm, K, KA = 5, 3, 4, 2, 4, 2, 3
    w = lambda shape: rng.integers(-2, 3, shape, dtype=np.int64)
    p = dict(tok_emb=w((V, d)), pos_emb=w((T, d)), thr_x0=np.array([1, 2]), Wun=w((V, d)), bun=w(V))
    for k in "qkv":
        p["L0.W" + k], p["L0.b" + k] = w((d, d)), w(d)
        p["L0.thr_" + k] = np.array([1, 3])
    p["L0.thr_a"] = np.array([1, 2, 4])
    p.update({"L0.Wo": w((d, d)), "L0.r1": w(d), "L0.bo": w(d), "L0.thr_x1": np.array([1, 2]),
              "L0.Wup": w((dm, d)), "L0.bup": w(dm), "L0.thr_m": np.array([1, 2]), "L0.Wdn": w((d, dm)),
              "L0.r2": w(d), "L0.bdn": w(d), "L0.thr_x2": np.array([1, 2])})
    return QModel("$ABCD", T, d, H, dm, K, KA, K, 1, p, attn="softmax", R=2)


def _arch_model():
    """tests/test_arch.py arch_model(seed=1, norm="rms", mlp="reglu", attn="softmax"): RMSNorm, gated MLP, softmax."""
    from .intref import QModel
    rng = np.random.default_rng(1)
    V, T, d, H, dm, K = 5, 3, 4, 2, 3, 2
    w = lambda shape: rng.integers(-2, 3, shape, dtype=np.int64)
    p = dict(tok_emb=w((V, d)), pos_emb=w((T, d)), thr_x0=np.array([1, 2]), Wun=w((V, d)), bun=w(V))
    p["tok_emb"][0] = -p["pos_emb"][0]
    p["tok_emb"][4] = [5, -5, 5, -5]
    for k in "qkv":
        p["L0.W" + k], p["L0.b" + k] = w((d, d)), w(d)
        p["L0.thr_" + k] = np.array([1, 3])
    p["L0.thr_a"] = np.array([1, 2])
    p.update({"L0.Wo": w((d, d)), "L0.r1": 2 * w(d), "L0.bo": w(d), "L0.thr_x1": np.array([1, 2]),
              "L0.Wdn": w((d, dm)), "L0.r2": w(d), "L0.bdn": w(d), "L0.thr_x2": np.array([1, 2])})
    p["L0.bv"][:] = 0
    p["L0.bo"][:] = 0
    p.update({"L0.Wg": w((dm, d)), "L0.bg": w(dm), "L0.thr_g": np.array([1, 2]),
              "L0.Wu": w((dm, d)), "L0.bu": w(dm), "L0.thr_u": np.array([1, 3])})
    return QModel("$ABCD", T, d, H, dm, K, K, K, 1, p, attn="softmax", R=2, norm="rms", Rn=2, mlp="reglu")


def _tiny():
    from .onemap import tiny_model
    return tiny_model()


def _trained(name: str):
    from .intref import QModel
    return QModel.load(name)


MNIST_DIGITS = 1             # shipped test digits per MNIST run (one pass is about 25 s of game time)


def _mnist():
    from . import shipped
    from .kit import MLP
    return MLP.load(shipped.data("mnist/mlp.npz"))


def mnist_digits(k: int) -> list[list[int]]:
    """The first k shipped MNIST test digits as 784 0/1 inputs."""
    from . import shipped
    d = np.load(shipped.data("mnist/digits.npz"))
    return [[int(v) for v in row] for row in np.unpackbits(d["bits"], axis=1)[:k, :784]]


NETS: dict[str, tuple[Callable, str]] = {
    "hello": (lambda: _mlp([4, 6, 5, 3], [3, 3, 3], 0), "4-6-5-3 MLP of `doomformer hello` (seed 0), K=3"),
    "mlp24": (lambda: _mlp([24, 32, 16, 8], [3, 3, 3], 7), "24-32-16-8 MLP, K=3, seed 7 (probe serial_sweep2)"),
    "mlp16": (lambda: _mlp([16, 48, 4], [4, 4], 4), "16-48-4 MLP, K=4, seed 4 (probe serial_sweep)"),
    "signed": (lambda: _mlp([6, 10, 4], [8, 3], 3, lo=10, hi=25),
               "6-10-4 MLP, seed 3, signed input with 8 thresholds (sign pairs share waves)"),
    "fold": (_fold_mlp, "2-1-1 MLP, 100 signed thresholds per input: 200 waves, two clock chunks at serial 64"),
    "tiny": (_tiny, "onemap.tiny_model(): relu-attention transformer, T=3, d=4 (corridor layers)"),
    "softmax": (_softmax_model, "tests/test_softmax.py softmax_model(): softmax attention, T=3, d=4, H=2"),
    "arch": (_arch_model, "tests/test_arch.py arch_model(attn='softmax'): RMSNorm, ReGLU MLP, softmax attention"),
    **{name: ((lambda name=name: _trained(name)), f"shipped {name} (trained, T=6, d=8, dm=16) on the prompts "
                                                   f"{', '.join(prompts)}") for name, prompts in TRAINED.items()},
    "mnist": (_mnist, f"shipped MNIST MLP (models/mnist/mlp.npz) on the first {MNIST_DIGITS} shipped test digit(s): "
                      "the longest crossbars (784 inputs)"),
}


def make_net(name: str):
    if name not in NETS:
        raise ValueError(f"unknown net {name!r}; known: {', '.join(NETS)}")
    return NETS[name][0]()


def _kind(net) -> str:
    from .kit import _kind as kind
    return kind(net)


def _x_max(net) -> int:
    """Largest |input| of an MLP's inputs: past its top input threshold."""
    return max(150, int(net.thresholds[0].max()) + 20)


def make_inputs(net, n: int, seed: int, boundary_max: int = BOUNDARY_MAX) -> tuple[list, int, int]:
    """(inputs, extremes, boundary): the extreme inputs (MLP: all +max, all -max, all zero, alternating; transformer:
    all top token, all BOS, alternating), then the boundary inputs (MLP only: every unit at s * (t + d) for an input
    threshold t, d in BOUNDARY_D and random signs s, so the lane gates stand 0 to 24 units above their walkways and the
    dolls step onto them; at most boundary_max pairs (t, d), evenly spread), then n random ones from default_rng(seed)
    (MLP: uniform in [-max, max]; transformer: uniform length 1..T, uniform tokens). Inputs outside the net's exact range
    are left out (extremes, boundary) or drawn again (random)."""
    rng = np.random.default_rng(seed)
    if _kind(net) == "mlp":
        from .kit import RangeError
        b, k = _x_max(net), net.n_in
        extremes = [[b] * k, [-b] * k, [0] * k, [b if i % 2 == 0 else -b for i in range(k)]]
        pairs = [(int(t), d) for d in BOUNDARY_D for t in net.thresholds[0]]
        pairs = [pairs[i * len(pairs) // boundary_max] for i in range(boundary_max)] if len(pairs) > boundary_max \
            else pairs
        signs = np.random.default_rng([seed, 1]).choice([-1, 1], (len(pairs), k))
        boundary = [[int(s) * (t + d) for s in row] for (t, d), row in zip(pairs, signs)]

        def ok(x) -> bool:
            try:
                net.check(np.asarray(x, dtype=np.int64))
            except RangeError:
                return False
            return True

        def draw():
            return [int(v) for v in rng.integers(-b, b + 1, k)]
    else:
        from .intref import check_limits
        V, T = len(net.vocab), net.T
        extremes = [[V - 1] * T, [0] * T, [0 if i % 2 == 0 else V - 1 for i in range(T)]]
        boundary = []

        def ok(x) -> bool:
            return not check_limits(net, [x])

        def draw():
            return [int(t) for t in rng.integers(0, V, int(rng.integers(1, T + 1)))]
    out = [x for x in extremes if ok(x)]
    n_ext = len(out)
    out += [x for x in boundary if ok(x)]
    kept = len(out)
    for _ in range(100 * (n + 1)):
        if len(out) == kept + n:
            break
        x = draw()
        if ok(x):
            out.append(x)
    if len(out) < kept + n:
        raise RuntimeError(f"cannot draw {n} in-range inputs")
    return out, n_ext, kept - n_ext


@dataclass
class Ref:
    """A reference net with its input set: extremes, boundary inputs, random inputs (in this order)."""
    name: str
    net: object
    inputs: list
    seed: int | None
    extremes: int
    boundary: int = 0

    def describe(self) -> dict:
        return dict(seed=self.seed, inputs=len(self.inputs), extremes=self.extremes, boundary=self.boundary,
                    random=len(self.inputs) - self.extremes - self.boundary)


def reference(names, n: int, seed0: int, boundary_max: int = BOUNDARY_MAX) -> dict[str, Ref]:
    """Ref per net name; the shipped models of TRAINED get their prompts (n does not apply)."""
    refs = {}
    for name in names:
        net = make_net(name)
        if name in TRAINED:
            refs[name] = Ref(name, net, [[0] + [net.vocab.index(c) for c in p] for p in TRAINED[name]], None, 0)
            continue
        if name == "mnist":
            refs[name] = Ref(name, net, mnist_digits(MNIST_DIGITS), None, 0)
            continue
        seed = seed0 + NETS_ALL.index(name)
        inputs, n_ext, n_bound = make_inputs(net, n, seed, boundary_max)
        refs[name] = Ref(name, net, inputs, seed, n_ext, n_bound)
    return refs


# --- running maps -----------------------------------------------------------------------------------------------------
def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    from . import _mp
    _mp.exit_with_parent()


def run_map(net, wad: str, inputs: list) -> dict:
    """Run the inputs through the compiled map in one game. rows: [status, tics, bad values, first bad names] per input;
    status ok, mismatch (an exposed value differs from the reference) or error (no settle, a missing readout)."""
    from .fleet import open_brain
    from .kit import MismatchError
    t0 = time.perf_counter()
    rows = []
    brain = open_brain(net, Path(wad))
    try:
        for x in inputs:
            try:
                if brain.kind == "transformer":
                    brain.logits(x)
                else:
                    brain(np.asarray(x, dtype=np.int64))
                rows.append(["ok", brain.info["tics"], 0, []])
            except MismatchError as e:
                rows.append(["mismatch", brain.info["tics"], len(e.mismatches), [m[0] for m in e.mismatches[:3]]])
            except RuntimeError as e:
                rows.append(["error", None, 0, [str(e)[:200]]])
                brain.close()
                brain = open_brain(net, Path(wad))
    finally:
        brain.close()
    return dict(rows=rows, seconds=time.perf_counter() - t0)


def summarize(rows: list, seconds: float) -> dict:
    """fail, n, tics [min, max], bad: [input index, status, values, names] of the first 6 failing inputs."""
    tics = [r[1] for r in rows if r[1] is not None]
    bad = [[i, r[0], r[2], r[3]] for i, r in enumerate(rows) if r[0] != "ok"]
    return dict(fail=len(bad), n=len(rows), tics=[min(tics), max(tics)] if tics else None, bad=bad[:6],
                seconds_per_pass=round(seconds / max(1, len(rows)), 2))


class _Done:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


def predicted_slack(wad: Path) -> dict:
    """The static schedule checker's prediction for a built map (doomformer.schedule.check_schedule): per Timing field,
    the smallest margin over the layers minus its measured minimum, in tics (negative: a predicted failure), and the
    number of predicted hazards. {"error": text} when the checker is missing or fails."""
    try:
        from .schedule import check_schedule
        rep = check_schedule(wad)
    except Exception as e:                                # the checker is optional here: record why it is absent
        return {"error": f"{type(e).__name__}: {e}"[:160]}
    out: dict = {}
    for m in rep.margins:
        slack = m["tics"] - m["minimum"]
        out[m["name"]] = min(out.get(m["name"], slack), slack)
    out["hazards"] = len(rep.hazards)
    return out


class Bench:
    """Builds maps in this process and runs them in up to `workers` spawned games (0: in this process). Results are
    cached by map bytes and input set. passes: the compile_pure passes (or opt level) of every build. predict: also
    record predicted_slack of every new map."""

    def __init__(self, refs: dict[str, Ref], workers: int, workdir: Path, passes, log: Path | None = None,
                 predict: bool = False):
        self.refs, self.workers, self.workdir, self.passes, self.log = refs, workers, Path(workdir), passes, log
        self.predict = predict
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        self.pool = None
        self.passes_run = 0

    def __enter__(self):
        if self.workers:
            from . import _mp
            self._detached = _mp.detached_main()
            self._detached.__enter__()
            self.pool = cf.ProcessPoolExecutor(self.workers, mp_context=_mp.context(), initializer=_worker_init)
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self._detached.__exit__(None, None, None)

    def build(self, name: str, timing: Timing | None):
        """(cache key, wad, stats) or {"build_error": text}."""
        from .kit import _build
        ref = self.refs[name]
        tmp = self.workdir / f".build-{os.getpid()}.wad"
        try:
            meta = _build(ref.net, tmp, opt=self.passes, timing=timing)
        except (ValueError, ZeroDivisionError) as e:
            return {"build_error": f"{type(e).__name__}: {e}"[:200]}
        data = tmp.read_bytes()
        est = meta.layout["estimated_settle_tics"]          # TileRunner gives up after twice it
        key = hashlib.sha256(data + f"|{est}|{json.dumps(ref.inputs)}".encode()).hexdigest()[:20]
        wad = self.workdir / f"{name}-{key}.wad"
        os.replace(tmp, wad)
        os.replace(tmp.with_suffix(".json"), wad.with_suffix(".json"))
        stats = meta.layout["stage_stats"]
        return key, wad, dict(waves=[s["waves"] for s in stats], spacing=[s["wave_spacing"] for s in stats],
                              estimated_tics=est, dolls=meta.n_dolls, map=hashlib.sha256(data).hexdigest()[:12])

    def run(self, jobs: list[tuple[str, Timing | None]]) -> list[dict]:
        """Result per (net, timing): summarize() plus waves, spacing, estimated_tics, dolls, map (sha256 prefix of the
        WAD) and reused (this map ran before with the same inputs)."""
        out: list = [None] * len(jobs)
        pending: dict[str, list[int]] = {}
        started = {}
        for j, (name, timing) in enumerate(jobs):
            built = self.build(name, timing)
            if isinstance(built, dict):
                out[j] = dict(built, fail=len(self.refs[name].inputs), n=len(self.refs[name].inputs))
                continue
            key, wad, stats = built
            if key in self.cache:
                out[j] = dict(self.cache[key], **stats, reused=True)
                wad.unlink(missing_ok=True)
                wad.with_suffix(".json").unlink(missing_ok=True)
                continue
            if key in pending:
                pending[key].append(j)
                continue
            pending[key] = [j]
            inputs = self.refs[name].inputs
            k = max(1, min(self.workers, len(inputs)))
            parts = [inputs[i * len(inputs) // k:(i + 1) * len(inputs) // k] for i in range(k)]
            net = self.refs[name].net
            futs = [_Done(run_map(net, str(wad), p)) if self.pool is None else
                    self.pool.submit(run_map, net, str(wad), p) for p in parts]
            started[key] = (futs, stats, wad, predicted_slack(wad) if self.predict else None)
        for key, (futs, stats, wad, predicted) in started.items():
            res = [f.result() for f in futs]
            rows = [r for part in res for r in part["rows"]]
            self.passes_run += len(rows)
            summary = summarize(rows, sum(part["seconds"] for part in res))
            if predicted is not None:
                summary["predicted_slack"] = predicted
            self.cache[key] = summary
            for k, j in enumerate(pending[key]):
                out[j] = dict(summary, **stats, reused=k > 0)
            wad.unlink(missing_ok=True)
            wad.with_suffix(".json").unlink(missing_ok=True)
        if self.log is not None:
            with open(self.log, "a") as f:
                for (name, timing), r in zip(jobs, out):
                    f.write(json.dumps(dict(net=name, timing=None if timing is None else asdict(timing), **r)) + "\n")
        return out


# --- scanning ---------------------------------------------------------------------------------------------------------
def round8(v: float) -> int:
    """v rounded up to a multiple of 8."""
    return int(math.ceil(v / 8 - 1e-9) * 8)


# opt=3 (copy bus): the copy line must follow the last light hit of the longest crossbar; (multiple, at least, rounding)
COPY_RULE = {"clock_margin": (1.25, 256, 16)}
COPY_SCAN = {"clock_margin": dict(step=64, resolution=16)}     # descent step and bisection resolution at opt=3


def shipped_value(const: str, edge: int, copy: bool = False) -> int:
    """edge times MULTIPLE[const], rounded up to a multiple of 8 (a non-positive edge counts as 0), and at least
    SHIP_AT_LEAST[const]; at opt=3 (copy) COPY_RULE overrides multiple, minimum and rounding."""
    m, least, unit = COPY_RULE[const] if copy and const in COPY_RULE else (MULTIPLE[const],
                                                                            SHIP_AT_LEAST.get(const, 0), 8)
    return max(unit * math.ceil(max(edge, 0) * m / unit - 1e-9), least)


def scan_top(const: str) -> int:
    """Where a scan starts: the default, or TOP[const] for a constant whose default is None (the legacy schedule)."""
    return TOP.get(const, getattr(Timing(), const))


def descent(const: str, lowest: int, step: int | None = None) -> list[int]:
    """Coarse values below the scan top, in decreasing steps: every `step` units if given, else every STEP[const] (8)
    units from a top <= 128 (serial, corr_gap, tele_gap, chunk_lead, copy_settle, sync_margin), halving from a larger
    one (clock_margin)."""
    d = scan_top(const)
    if step or d <= 128:
        vals = list(range(d - (step or STEP.get(const, 8)), lowest - 1, -(step or STEP.get(const, 8))))
    else:
        vals, v = [], d
        while v > 8:
            v = 8 * (v // 16)
            vals.append(v)
    vals = [v for v in vals if lowest <= v < d]
    return vals + ([lowest] if lowest not in vals and lowest < d else [])


def passes_all(row: dict) -> bool:
    return all(r["fail"] == 0 for r in row.values())


def edge_of(rows: dict[int, dict]) -> tuple[int, int | None]:
    """(edge, largest failing value) of tested rows {value: {net: result}}."""
    failing = [v for v, row in rows.items() if not passes_all(row)]
    top = max(failing) if failing else None
    above = [v for v in rows if top is None or v > top]
    if not above:
        raise RuntimeError(f"the default value fails; failing values {sorted(failing)}")
    return min(above), top


def scan(bench: Bench, const: str, nets: list[str], lowest: int, echo: Callable[[str], None] = print,
         step: int | None = None, resolution: int = 1, confirm: bool = True, copy: bool = False) -> dict:
    """Measure the failure edge of one constant (the others at their defaults); the table entry of tuning.json.
    step: descent step (default: descent's); resolution: bisect until the largest failing value and the edge are this
    close; confirm: also test every 8 units up to 32 above the edge (the shipped value is always tested)."""
    default, top_value = getattr(Timing(), const), scan_top(const)
    rows: dict[int, dict[str, dict]] = {}
    t0 = time.perf_counter()

    def test(values) -> bool:
        values = [v for v in dict.fromkeys(values) if v not in rows and v >= lowest]
        if not values:
            return False
        res = bench.run([(n, dataclasses.replace(OTHERS or Timing(), **{const: v})) for v in values for n in nets])
        for k, v in enumerate(values):
            rows[v] = {n: res[k * len(nets) + i] for i, n in enumerate(nets)}
            cells = "  ".join(f"{n} {r['fail']}/{r['n']}" + ("=" if r.get("reused") else "")
                              + (f" ({_slack(r, const):+d})" if isinstance(_slack(r, const), int) else "")
                              for n, r in rows[v].items())
            echo(f"  {const} {v:5d} ({v / UNITS_PER_TIC:6.2f} tics): {'pass' if passes_all(rows[v]) else 'FAIL'}  "
                 f"{cells}  [{(time.perf_counter() - t0) / 60:.0f} min]")
        return True

    test([top_value])
    per_value = sum(r["seconds_per_pass"] * r["n"] for r in rows[top_value].values()) / max(1, bench.workers)
    coarse = descent(const, lowest, step)
    echo(f"  {const}: up to {len(coarse) + 10} more values at about {per_value / 60:.1f} min each (estimate at the "
         "first value; maps that repeat cost nothing)")
    for v in coarse:                                         # descend until some input fails
        test([v])
        if not passes_all(rows[v]):
            break
    while True:
        edge, top = edge_of(rows)
        if top is not None and edge - top > resolution:      # bisect: on multiples of 8, then to `resolution`
            mid = 8 * ((edge + top) // 16) if edge - top > 8 else (edge + top) // 2
            if not top < mid < edge:
                mid = (edge + top) // 2
            test([mid])
            continue
        ship = shipped_value(const, edge, copy)              # every 8 units just above the edge, and the shipped value
        grid = range(round8(edge + 1), min(top_value, edge + 32) + 1, 8) if confirm else []
        if test([*grid, ship]):
            continue
        break
    edge, top = edge_of(rows)
    ship = shipped_value(const, edge, copy)
    return dict(default=default, scan_top=top_value, edge=edge, edge_tics=edge / UNITS_PER_TIC, largest_failing=top,
                largest_failing_tics=None if top is None else top / UNITS_PER_TIC, lowest_tested=min(rows),
                resolution=resolution if top is not None else None,
                multiple=COPY_RULE[const][0] if copy and const in COPY_RULE else MULTIPLE[const],
                ship_at_least=COPY_RULE[const][1] if copy and const in COPY_RULE else SHIP_AT_LEAST.get(const, 0),
                shipped=ship,
                shipped_tics=ship / UNITS_PER_TIC,
                predicted_slack={label: {n: _slack(rows[v][n], const) for n in nets}
                                 for label, v in (("scan_top", top_value), ("shipped", ship), ("edge", edge),
                                                  ("largest_failing", top)) if v in rows},
                nets=list(nets), inputs={n: bench.refs[n].describe() for n in nets},
                minutes=round((time.perf_counter() - t0) / 60, 1),
                table=[dict(value=v, tics=v / UNITS_PER_TIC, nets={n: _cell(r) for n, r in rows[v].items()})
                       for v in sorted(rows, reverse=True)])


def _slack(r: dict, const: str):
    """The checker's predicted slack (tics) of `const` in a result, None if it has none (no such layer, no checker)."""
    return r.get("predicted_slack", {}).get(const)


def _cell(r: dict) -> dict:
    keep = ("fail", "n", "tics", "waves", "spacing", "map", "reused", "bad", "build_error", "seconds_per_pass",
            "predicted_slack")
    return {k: r[k] for k in keep if k in r}


# --- tuning.json ------------------------------------------------------------------------------------------------------
def load(path: Path = TUNING_FILE) -> dict:
    return json.loads(Path(path).read_text())


def write(path: Path, data: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def environment() -> dict:
    import vizdoom
    return dict(date=datetime.date.today().isoformat(), vizdoom=vizdoom.__version__,
                machine=f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores",
                load_average=[round(v, 1) for v in os.getloadavg()] if hasattr(os, "getloadavg") else None)


def shipped_dicts(data: dict) -> tuple[dict, dict]:
    """(shipped, shipped_copy): the plane constants and the copy-mode constants that differ from them. A constant
    without a table ships its default (the legacy, conservative value)."""
    d = asdict(Timing())
    plane = {c: data.get("constants", {}).get(c, {}).get("shipped", d[c]) for c in SHIPPED_KEYS}
    copy = {}
    for c in MODES["copy"].constants:
        v = data.get("constants_copy", {}).get(c, {}).get("shipped", d[c])
        if c not in plane or v != plane[c]:
            copy[c] = v
    return plane, copy


def mode_timing(data: dict, mode: Mode) -> Timing:
    plane, copy = shipped_dicts(data)
    return Timing(**{**plane, **(copy if mode.name == "copy" else {})})


# --- command line -----------------------------------------------------------------------------------------------------
def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--check", action="store_true",
                   help="only build opt=2 maps (--copy: opt=3) from the committed tuning.json and run them on the "
                        "reference nets with fresh inputs; exit 1 on any mismatch")
    p.add_argument("--hashes", action="store_true",
                   help="rebuild the opt 0 maps of tiny, doomnames_onemap, doomnames_onemap_softmax, doomplayer_onemap, "
                        "induction_onemap and mnist (no game) and compare their sha256 with opt0_hashes.json; exit 1 on "
                        "a difference")
    p.add_argument("--update-hashes", action="store_true", help="with --hashes: rewrite opt0_hashes.json")
    p.add_argument("--joint", action="store_true",
                   help="only the joint run: every shipped constant of the table together on every reference net, "
                        "recorded in the table (--inputs defaults to 100)")
    p.add_argument("--copy", action="store_true",
                   help="the copy-bus schedule of opt=3 (clock_margin, copy_settle; `shipped_copy`)")
    p.add_argument("--const", default=None,
                   help="constants to re-measure, comma separated (default: all of the mode; plane: "
                        f"{','.join(MODES['plane'].constants)}; copy: {','.join(MODES['copy'].constants)})")
    p.add_argument("--nets", default=None,
                   help=f"reference nets, comma separated (default plane: {','.join(MODES['plane'].nets)}; copy: "
                        f"{','.join(MODES['copy'].nets)}; known: {', '.join(NETS)})")
    p.add_argument("--inputs", type=int, default=None,
                   help="random inputs per net and value besides the extremes (default 30; --joint 100)")
    p.add_argument("--workers", type=int, default=2, help="DOOM games at once (default 2; 0 runs in this process)")
    p.add_argument("--trained", action="store_true",
                   help="with --joint/--check: also run the shipped one-map models on a few prompts ("
                        + "; ".join(f"{n}: {', '.join(p)}" for n, p in TRAINED.items()) + ")")
    p.add_argument("--out", default=str(TUNING_FILE), help="table to update (default: the package's tuning.json)")
    p.add_argument("--workdir", default=None, help="where maps are built (default: a temporary directory)")
    p.add_argument("--log", default=None, help="append every map run as a JSON line to this file")
    p.add_argument("--provisional", action="store_true", help="mark the written table provisional")
    p.add_argument("--no-predict", action="store_true",
                   help="skip the static checker's prediction of every map (doomformer.schedule; slow on large maps)")
    p.add_argument("--shipped-others", action="store_true",
                   help="scans: the constants not being varied stay at their shipped values (serial 40, sync_margin 64, "
                        "...) instead of the legacy defaults, so the edge is that of the schedule opt 2 builds")
    p.add_argument("--scan-top", type=int, default=None, help="scans: start the descent at this value, not the default")
    p.add_argument("--resolution", type=int, default=None, help="scans: bisect until the largest failing value and the "
                                                                 "edge are this many units apart (default 1)")
    p.add_argument("--coarse", action="store_true",
                   help=f"time-boxed scans: descend in steps of {COARSE_STEP}, bisect to {COARSE_RESOLUTION} units, "
                        f"test only the shipped value above the edge, at most {COARSE_BOUNDARY} boundary inputs")


def _command(args) -> str:
    parts = ["doomformer tune"]
    for flag in ("check", "joint", "copy", "trained", "provisional", "coarse", "no_predict", "shipped_others"):
        if getattr(args, flag, False):
            parts.append(f"--{flag.replace('_', '-')}")
    for opt in ("const", "nets", "inputs", "scan_top", "resolution"):
        if getattr(args, opt) is not None:
            parts.append(f"--{opt} {getattr(args, opt)}")
    return " ".join(parts + [f"--workers {args.workers}"])


def run(args) -> int:
    os.environ["OMP_NUM_THREADS"] = "1"
    if args.hashes or args.update_hashes:
        return hashes(args)
    mode = MODES["copy" if args.copy else "plane"]
    nets = args.nets.split(",") if args.nets else list(mode.nets)
    consts = args.const.split(",") if args.const else list(mode.constants)
    for n in nets:
        if n not in NETS:
            raise SystemExit(f"tune: unknown net {n!r}; known: {', '.join(NETS)}")
    for c in consts:
        if c not in mode.constants:
            raise SystemExit(f"tune: {c!r} is not a {mode.name} constant ({', '.join(mode.constants)})")
    if args.workers < 0:
        raise SystemExit("tune: --workers must be >= 0")
    if args.trained:
        nets += [n for n in TRAINED if n not in nets]
    out = Path(args.out)
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="doomformer-tune-"))
    log = Path(args.log) if args.log else None
    try:
        if args.check or args.joint:
            return _joint(args, mode, nets, out, workdir, log)
        return _measure(args, mode, nets, consts, out, workdir, log)
    finally:
        if not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _joint(args, mode: Mode, nets: list[str], out: Path, workdir: Path, log) -> int:
    n = args.inputs if args.inputs is not None else (100 if args.joint else 30)
    refs = reference(nets, n, JOINT_SEED)
    if args.check:
        passes, timing = mode.check_opt, None
        shown = asdict(Timing.tuned(copy=mode.name == "copy"))
        what = f"tune --check: opt={mode.check_opt} with the constants of {TUNING_FILE.name} {shown}"
    else:
        passes, timing = mode.passes, mode_timing(load(out), mode)
        what = f"tune --joint ({mode.name}): {asdict(timing)}"
    print(f"{what} on {', '.join(nets)}; {n} random inputs per net plus extremes", flush=True)
    t0 = time.perf_counter()
    with Bench(refs, args.workers, workdir, passes, log, predict=not args.no_predict) as bench:
        res = bench.run([(name, timing) for name in nets])
    cells = {}
    for name, r in zip(nets, res):
        cells[name] = dict(_cell(r), **refs[name].describe())
        print(f"  {name}: {r['n'] - r['fail']}/{r['n']} exact, tics {r.get('tics')}, waves {r.get('waves')}, "
              f"{r.get('seconds_per_pass')} s per pass" + (f"  failing inputs: {r['bad']}" if r["fail"] else ""),
              flush=True)
    bad, total = sum(r["fail"] for r in res), sum(r["n"] for r in res)
    minutes = round((time.perf_counter() - t0) / 60, 1)
    print(f"{'EXACT' if not bad else 'MISMATCH'}: {total - bad} of {total} passes exact ({minutes} min)")
    if args.joint:
        data = load(out)
        data[mode.joint] = dict(timing=asdict(timing), passes=sorted(mode.passes), failures=bad, passes_run=total,
                                nets=cells, minutes=minutes, command=_command(args), **environment())
        _finish(data, args)
        write(out, data)
    return 0 if not bad else 1


def _measure(args, mode: Mode, nets: list[str], consts: list[str], out: Path, workdir: Path, log) -> int:
    n = args.inputs if args.inputs is not None else 30
    refs = reference(nets, n, SCAN_SEED, COARSE_BOUNDARY if args.coarse else BOUNDARY_MAX)
    data = load(out) if out.exists() else {}
    if args.shipped_others:
        global OTHERS
        OTHERS = Timing.tuned(copy=mode.name == "copy")
    print(f"tune ({mode.name}): {', '.join(consts)} on {', '.join(nets)}, {n} random inputs per net plus extremes, "
          f"{args.workers} games at once. A full scan takes hours; every constant prints an estimate.", flush=True)
    with Bench(refs, args.workers, workdir, mode.passes, log, predict=not args.no_predict) as bench:
        for const in consts:
            lowest = LOWEST[const]
            print(f"{const}: default {getattr(Timing(), const)}, lowest {lowest}", flush=True)
            scan_kw = dict(step=COARSE_STEP, resolution=COARSE_RESOLUTION, confirm=False) if args.coarse else {}
            if args.resolution:
                scan_kw["resolution"] = args.resolution
            if args.scan_top is not None:
                TOP[const] = args.scan_top
            if mode.name == "copy":
                scan_kw.update(COPY_SCAN.get(const, {}), confirm=False)
            entry = scan(bench, const, [m for m in nets if m not in SKIP.get(const, ())], lowest,
                         echo=lambda s: print(s, flush=True), copy=mode.name == "copy", **scan_kw)
            entry.update(command=_command(args), **environment())
            data.setdefault(mode.table, {})[const] = entry
            print(f"{const}: edge {entry['edge']} ({entry['edge_tics']} tics), largest failing "
                  f"{entry['largest_failing']}, shipped {entry['shipped']} = {entry['multiple']} x edge rounded up "
                  f"to 8 ({entry['minutes']} min)", flush=True)
            _finish(data, args)
            write(out, data)
    return 0


def _record_command(data: dict, args) -> None:
    data.setdefault("commands", [])
    if _command(args) not in data["commands"]:
        data["commands"].append(_command(args))


def _finish(data: dict, args) -> None:
    data["format"] = "doomformer-tuning-1"
    plane = MODES["plane"]          # final once the plane joint run of the current shipped values passed
    joint_ok = (data.get(plane.joint, {}).get("failures") == 0
                and data[plane.joint]["timing"] == asdict(mode_timing(data, plane)))
    data["provisional"] = bool(args.provisional) or not joint_ok
    data["units_per_tic"] = UNITS_PER_TIC
    data["modes"] = {m.name: dict(passes=sorted(m.passes), check_opt=m.check_opt, constants=list(m.constants),
                                  nets=list(m.nets)) for m in MODES.values()}
    data["shipped"], data["shipped_copy"] = shipped_dicts(data)
    data["not_measured"] = {m.shipped: [c for c in m.constants if c not in data.get(m.table, {})]
                            for m in MODES.values()}
    data["net_descriptions"] = {n: NETS[n][1] for n in NETS}
    env = environment()
    data["date"], data["vizdoom"] = env["date"], env["vizdoom"]
    loads = [e["load_average"][0] for t in (m.table for m in MODES.values()) for e in data.get(t, {}).values()
             if e.get("load_average")]
    loads += [data[m.joint]["load_average"][0] for m in MODES.values() if data.get(m.joint, {}).get("load_average")]
    if loads:
        data["machine_load_note"] = (
            f"measured on a shared machine ({env['machine']}) running other jobs: 1-minute load average "
            f"{min(loads)} to {max(loads)} when the entries were written. minutes and seconds_per_pass are wall times "
            "under that load; pass/fail and game tics do not depend on it (game tics are simulated)")
    _record_command(data, args)


# --- opt 0 byte identity ----------------------------------------------------------------------------------------------
HASH_FILE = Path(__file__).with_name("opt0_hashes.json")
HASH_MODELS = ("tiny", "doomnames_onemap", "doomnames_onemap_softmax", "doomplayer_onemap", "induction_onemap", "mnist")


def _hash_net(name: str):
    return make_net(name) if name in NETS else _trained(name)


def opt0_hashes(names=HASH_MODELS, echo=print) -> dict:
    """sha256 of the .wad and .json that `df.compile(net)` (opt 0, no flags) builds for each model, plus the sector count."""
    from .kit import _build
    out = {}
    with tempfile.TemporaryDirectory(prefix="doomformer-hash-") as d:
        for name in names:
            t0 = time.perf_counter()
            wad = Path(d) / f"{name}.wad"
            meta = _build(_hash_net(name), wad)
            sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()       # noqa: E731
            out[name] = dict(wad=sha(wad), json=sha(wad.with_suffix(".json")), sectors=meta.n_sectors,
                             dolls=meta.n_dolls)
            echo(f"  {name}: {meta.n_sectors} sectors, wad {out[name]['wad'][:16]}, json {out[name]['json'][:16]} "
                 f"({time.perf_counter() - t0:.0f} s)")
            wad.unlink(missing_ok=True)
            wad.with_suffix(".json").unlink(missing_ok=True)
    return out


def hashes(args) -> int:
    """`tune --hashes`: rebuild the opt 0 maps of HASH_MODELS and compare with opt0_hashes.json (exit 1 on a
    difference); `--update-hashes` rewrites the file instead."""
    names = args.nets.split(",") if args.nets else list(HASH_MODELS)
    for n in names:
        if n not in HASH_MODELS:
            raise SystemExit(f"tune --hashes: {n!r} is not one of {', '.join(HASH_MODELS)}")
    print(f"opt 0 build hashes of {', '.join(names)} (df.compile(net), no flags):", flush=True)
    got = opt0_hashes(names)
    if args.update_hashes:
        data = json.loads(HASH_FILE.read_text()) if HASH_FILE.is_file() else {"models": {}}
        data["models"].update(got)
        data["note"] = ("sha256 of the .wad and .json of the opt 0 build. Equal to what v0.4.1 builds for tiny, "
                        "doomnames_onemap, doomnames_onemap_softmax, doomplayer_onemap and mnist; induction_onemap "
                        "first appears in v0.5.0 (its hash is the v0.5.0 one)")
        data.update(command="doomformer tune --update-hashes", **{k: v for k, v in environment().items()
                                                                  if k in ("date", "vizdoom")})
        HASH_FILE.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {HASH_FILE}")
        return 0
    want = json.loads(HASH_FILE.read_text())["models"]
    bad = [n for n in names if got[n]["wad"] != want[n]["wad"] or got[n]["json"] != want[n]["json"]]
    for n in bad:
        print(f"DIFFERENT {n}: wad {got[n]['wad']} (recorded {want[n]['wad']}), json {got[n]['json']} "
              f"(recorded {want[n]['json']})")
    print(f"{'IDENTICAL' if not bad else 'DIFFERENT'}: {len(names) - len(bad)} of {len(names)} opt 0 maps match "
          f"{HASH_FILE.name}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m doomformer.tune", description=__doc__.split("\n\n")[0])
    add_arguments(ap)
    return run(ap.parse_args(argv))


if __name__ == "__main__":         # run the package module, so spawned workers can import what they unpickle
    from doomformer.tune import main as _main
    raise SystemExit(_main())
