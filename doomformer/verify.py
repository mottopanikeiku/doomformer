"""Full-domain verification of a small MLP: `doomformer verify NET`.

    doomformer verify net.npz                          # --mode classes over |x| <= 12,000 per input
    doomformer verify net.npz --mode all --domain -40 40
    doomformer verify net.npz --wad tampered.wad       # check a given compiled map instead of compiling one
    doomformer verify --validate 20                    # classes == all on small nets, and a tampered map fails both

Two modes, both compare every exposed value of every run with the integer reference at tolerance 0:

all      every input of the integer box [lo, hi]^n_in the net accepts (net.check), one DOOM pass each.
classes  one pass per representative of every input class, plus both sides of every threshold neighbourhood.

Why classes cover the domain. The map sees an input x_i only through the staircase count a_i = adc(x_i, thresholds[0])
(one doll per threshold and sign): its other lanes, gates, weights and every later layer read values computed from
the count vector a = (a_0 ... a_n). The reference is the same function of a, and net.check is a function of a and of
|x_i| <= 12,000. a_i is constant on the intervals between the breakpoints t (a jumps at x = t) and -t + 1 (a jumps
between -t and -t + 1; signed input) of the first layer's thresholds, so the box splits into prod_i C_i classes
with one a each, and DOOM's result on any x is its result on the class representative IF the engine's response to
x depends on x only through a. That one assumption (A) is the whole gap between `classes` and `all`: A says the
gate comparison "port value against the gate's limit" is a correct integer comparison for every |x| <= 12,000 (16.16
fixed point is exact there) and nothing else about x's size matters (movers finish within 2 tics at any distance).
`all` does not need A; it needs time (about 15 ms per input on an idle core). What is verified: for the compiled
map and this engine build, every class representative reproduces the reference in every exposed value. What is not:
inputs outside the box or refused by net.check, values the map does not expose, other engine versions, and (classes)
any input whose engine behaviour differs from its representative's, which `--validate` tests on small nets.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PORT_MAX = 12000

ASSUMPTION = ("assumption A: the engine's response to an input x depends on x only through its staircase counts "
              "(the gate comparison is exact for |x| <= 12,000)")


# --- classes --------------------------------------------------------------------------------------------------------
def breakpoints(thresholds, signed: bool) -> list[int]:
    """x values where the staircase count of one input starts a new value: t (x >= t) and -t + 1 (signed: -x >= t)."""
    b = {int(t) for t in thresholds}
    if signed:
        b |= {1 - int(t) for t in thresholds}
    return sorted(b)


def input_classes(thresholds, signed: bool, lo: int, hi: int) -> list[tuple[int, int]]:
    """The maximal intervals of [lo, hi] on which the staircase count is constant, as (first, last)."""
    starts = [lo] + [b for b in breakpoints(thresholds, signed) if lo < b <= hi]
    return [(s, (starts[i + 1] - 1) if i + 1 < len(starts) else hi) for i, s in enumerate(starts)]


def _count(x, thr, signed):
    from .intref import adc_relu, adc_signed
    return int((adc_signed if signed else adc_relu)(np.asarray([x]), thr)[0])


def plan_inputs(net, mode: str, lo: int = -PORT_MAX, hi: int = PORT_MAX, max_inputs: int = 30000,
                max_classes: int = 20000) -> dict:
    """The inputs to run: {xs, covered (domain inputs they stand for), classes, domain_inputs, refused (candidates the
    net's range check refuses), refused_covered}. Raises ValueError when the enumeration exceeds the budget."""
    from .kit import RangeError
    n = net.n_in
    thr, signed = net.thresholds[0], net.signed_input
    total = (hi - lo + 1) ** n
    if mode == "all":
        if total > max_inputs:
            raise ValueError(f"--mode all would run {total:,} inputs over [{lo}, {hi}]^{n}; narrow --domain, raise "
                             f"--max-inputs ({max_inputs:,}) or use --mode classes")
        cand = [list(t) for t in itertools.product(range(lo, hi + 1), repeat=n)]
        classes = None
    else:
        cls = input_classes(thr, signed, lo, hi)
        for first, last in cls:                                   # the class property itself, on its end points
            assert _count(first, thr, signed) == _count(last, thr, signed), "class is not constant"
        if len(cls) ** n > max_classes:
            raise ValueError(f"{len(cls)} classes per input ^ {n} inputs = {len(cls) ** n:,} class tuples; raise "
                             f"--max-classes ({max_classes:,}) or use a smaller net")
        idx = list(itertools.product(range(len(cls)), repeat=n))
        reps: dict[tuple, int] = {}
        for t in idx:
            for pick in (0, 1):
                reps.setdefault(tuple(cls[i][pick] for i in t), 0)
        base = [min(max(0, lo), hi)] * n
        for i in range(n):                                        # both sides of every threshold neighbourhood
            for b in breakpoints(thr, signed):
                for v in (b - 1, b, b + 1):
                    if lo <= v <= hi:
                        x = list(base)
                        x[i] = v
                        reps.setdefault(tuple(x), 0)
        cand = [list(x) for x in reps]
        classes = len(idx)
    xs, refused = [], 0
    for x in cand:
        try:
            net.check(x)
            xs.append(x)
        except RangeError:
            refused += 1
    covered = None
    if mode == "all":
        covered = len(xs)
    else:                                                          # inputs of the classes whose representative passes
        covered = 0
        for t in idx:
            first = [cls[i][0] for i in t]
            try:
                net.check(first)
            except RangeError:
                continue
            size = 1
            for i in t:
                size *= cls[i][1] - cls[i][0] + 1
            covered += size
    return dict(xs=xs, covered=covered, classes=classes, domain_inputs=total, refused=refused)


# --- running --------------------------------------------------------------------------------------------------------
def _run(net, wad: Path, xs: list, workers: int, fail_fast: bool) -> dict:
    """Run xs on the map `wad`: {mismatches: [(index, [(name, doom, ref)])], values, error}."""
    from .fleet import _Collector, _pass, _run_local, open_brain
    from .kit import MismatchError

    class Tally(_Collector):
        bad: list
        values = 0

        def take(self, i, game, status, *rest, retried=False):
            if status == "mismatch":
                self.bad.append((i, rest[0]))
                self.done += 1
                return
            if status == "ok":
                self.values += rest[1]
            super().take(i, game, status, *rest, retried=retried)

    out = dict(mismatches=[], values=0)
    if workers <= 1 or len(xs) < 4:
        brain = open_brain(net, wad)
        try:
            for i, x in enumerate(xs):
                r = _pass(brain, np.asarray(x))
                if r[0] == "mismatch":
                    out["mismatches"].append((i, r[1]))
                    if fail_fast:
                        break
                elif r[0] == "error":
                    raise RuntimeError(f"input {x}: {r[1]}")
                else:
                    out["values"] += r[2]
        finally:
            brain.close()
        return out
    col = (_Collector if fail_fast else Tally)(len(xs), None)
    if not fail_fast:
        col.bad = []
    try:
        _run_local(net, wad, xs, min(workers, len(xs)), col)
    except MismatchError as e:
        out["mismatches"].append((getattr(e, "index", None), e.mismatches))
        return out
    if not fail_fast:
        out["mismatches"], out["values"] = col.bad, col.values
    else:
        out["values"] = None
    return out


def verify(net, mode: str = "classes", *, domain: tuple[int, int] = (-PORT_MAX, PORT_MAX), wad=None, workers: int = 1,
           max_inputs: int = 30000, max_classes: int = 20000, fail_fast: bool = True, progress=None) -> dict:
    """Verify `net` (an MLP) over its input domain; returns a JSON-ready dict with `verdict` "VERIFIED" or "FAILED".
    wad: a compiled map (with its .json) to check instead of compiling `net` (a tampered map must FAIL)."""
    from .kit import MLP
    from .kit import compile as df_compile
    if not isinstance(net, MLP):
        raise TypeError("verify takes a df.MLP")
    if mode not in ("all", "classes"):
        raise ValueError("mode must be 'all' or 'classes'")
    lo, hi = int(domain[0]), int(domain[1])
    if not (-PORT_MAX <= lo <= hi <= PORT_MAX):
        raise ValueError(f"domain must lie inside [-{PORT_MAX}, {PORT_MAX}] (inputs are written into gate floors)")
    t0 = time.perf_counter()
    plan = plan_inputs(net, mode, lo, hi, max_inputs, max_classes)
    xs = plan["xs"]
    with tempfile.TemporaryDirectory(prefix="dfv-") as d:
        path = Path(wad) if wad is not None else Path(d) / "net.wad"
        if wad is None:
            df_compile(net, path, mode="map", guard=True, limits=False).close()
        elif not path.with_suffix(".json").exists():
            raise FileNotFoundError(f"{path.with_suffix('.json')}: the map's metadata must sit next to the WAD")
        res = _run(net, path, xs, workers, fail_fast)
    bad = res["mismatches"]
    out = dict(mode=mode, domain=[lo, hi], n_in=net.n_in, layers=net.n_layers, signed_input=net.signed_input,
               classes=plan["classes"], domain_inputs=plan["domain_inputs"], covered=plan["covered"],
               inputs_run=len(xs), candidates_refused=plan["refused"], values=res["values"],
               mismatching_inputs=len(bad), verdict="FAILED" if bad else "VERIFIED",
               first_mismatch=None, assumption=None if mode == "all" else ASSUMPTION,
               seconds=round(time.perf_counter() - t0, 1))
    if bad:
        i, mm = bad[0]
        out["first_mismatch"] = dict(input=xs[i] if i is not None else None, values=[list(m) for m in mm[:6]])
    return out


def describe(r: dict, name: str = "net") -> str:
    n = f"{name}: {r['n_in']} inputs, {r['layers']} layer(s), {'signed' if r['signed_input'] else 'unsigned'} input"
    box = f"[{r['domain'][0]}, {r['domain'][1]}]^{r['n_in']} ({r['domain_inputs']:,} inputs)"
    if r["mode"] == "all":
        what = f"every input of {box} the net accepts: {r['inputs_run']:,} runs"
    else:
        what = (f"{r['classes']:,} input classes of {box}: {r['inputs_run']:,} runs (one per class end point and "
                f"threshold neighbour), standing for {r['covered']:,} inputs")
    lines = [n, f"mode {r['mode']}: {what}; {r['candidates_refused']} candidates refused by the range check"]
    if r["verdict"] == "VERIFIED":
        v = f"{r['values']:,} values" if r["values"] is not None else "every value"
        lines.append(f"VERIFIED: {v} exact against the integer reference (tolerance 0), {r['seconds']} s")
        if r["assumption"]:
            lines.append(f"  holds under {r['assumption']}")
    else:
        fm = r["first_mismatch"]
        lines.append(f"FAILED: {r['mismatching_inputs']} input(s) differ; first x = {fm['input']}: "
                     + ", ".join(f"{n} DOOM {d} != reference {ref}" for n, d, ref in fm["values"][:4]))
    return "\n".join(lines)


# --- validating the classes mode ------------------------------------------------------------------------------------
def small_net(rng, n_in: int, tmax: int):
    """A small MLP (depth 1 to 3, widths <= 4, weights +-9, first thresholds <= tmax) whose hidden thresholds come from
    its own hidden values, so that every layer's staircase is exercised."""
    from .fuzz import _draw_thr, _thr_from
    from .intref import adc_relu, adc_signed
    from .kit import MLP
    depth = int(rng.integers(1, 4))
    widths = [n_in] + [int(rng.integers(1, 5)) for _ in range(depth)]
    signed = bool(rng.random() < 0.7)
    Ws = []
    for l in range(depth):
        W = rng.integers(-9, 10, (widths[l + 1], widths[l]))
        W[rng.random(W.shape) < 0.2] = 0
        if not W.any():
            W[0, 0] = 3
        Ws.append(W)
    bs = [rng.integers(-6, 7, w) for w in widths[1:]]
    thr = [_draw_thr(rng, int(rng.integers(1, 4)), tmax)]
    xs = rng.integers(-tmax - 2, tmax + 3, (48, n_in))
    a = adc_signed(xs, thr[0]) if signed else adc_relu(xs, thr[0])
    for l in range(depth - 1):
        h = a @ Ws[l].T + bs[l]
        thr.append(_thr_from(rng, h, int(rng.integers(1, 4)), False))
        a = adc_relu(h, thr[-1])
    return MLP(Ws, bs, thr, signed_input=signed)


def validate(n_nets: int = 20, seed: int = 0, workers: int = 1, full_domain: int = 2, progress=None) -> dict:
    """On `n_nets` small nets: verdict(classes) == verdict(all) over the same domain, and a map with one weight line
    changed by +1 FAILS both. The first `full_domain` nets are 1-input nets verified over the whole +-12,000 domain
    (24,001 inputs); the rest use n_in 1 or 2 over a small box that contains every breakpoint."""
    from .kit import compile as df_compile
    from .mutate import tamper_weight
    rows = []
    for k in range(n_nets):
        rng = np.random.default_rng(np.random.SeedSequence([seed, k, 99]))
        full = k < full_domain
        n_in = 1 if full else int(rng.integers(1, 3))
        tmax = 12 if (n_in == 2 and not full) else int(rng.choice([6, 20, 40]))
        net = small_net(rng, n_in, tmax)
        lim = PORT_MAX if full else int(max(int(net.thresholds[0].max()), 1) + 3)
        dom = (-lim, lim)
        row = dict(net=k, n_in=n_in, layers=net.n_layers, domain=list(dom), thresholds0=net.thresholds[0].tolist())
        with tempfile.TemporaryDirectory(prefix="dfval-") as d:
            wad, bad = Path(d) / "ok.wad", Path(d) / "bad.wad"
            df_compile(net, wad, mode="map", guard=True, limits=False).close()
            row["tamper"] = tamper_weight(wad, bad, 1)
            for tag, w in (("clean", wad), ("tampered", bad)):
                for mode in ("classes", "all"):
                    r = verify(net, mode, domain=dom, wad=w, workers=workers, max_inputs=10 ** 6, fail_fast=True)
                    row[f"{tag}_{mode}"] = dict(verdict=r["verdict"], runs=r["inputs_run"], seconds=r["seconds"],
                                                covered=r["covered"])
        row["agree"] = (row["clean_classes"]["verdict"] == row["clean_all"]["verdict"]
                        and row["tampered_classes"]["verdict"] == row["tampered_all"]["verdict"])
        row["tamper_detected"] = row["tampered_classes"]["verdict"] == row["tampered_all"]["verdict"] == "FAILED"
        rows.append(row)
        if progress:
            progress(row)
    return dict(seed=seed, nets=rows, agree=sum(r["agree"] for r in rows), n=len(rows),
                clean_verified=sum(r["clean_all"]["verdict"] == "VERIFIED" for r in rows),
                tamper_detected=sum(r["tamper_detected"] for r in rows),
                runs_all=sum(r["clean_all"]["runs"] + r["tampered_all"]["runs"] for r in rows),
                runs_classes=sum(r["clean_classes"]["runs"] + r["tampered_classes"]["runs"] for r in rows))


# --- CLI ------------------------------------------------------------------------------------------------------------
def add_arguments(p) -> None:
    p.add_argument("net", nargs="?", help="an MLP saved with MLP.save (.npz)")
    p.add_argument("--mode", choices=("classes", "all"), default="classes",
                   help="classes: one run per input class and threshold neighbour (default); all: every input")
    p.add_argument("--domain", nargs=2, type=int, metavar=("LO", "HI"), default=[-PORT_MAX, PORT_MAX],
                   help="integer box per input (default -12000 12000, the whole legal range)")
    p.add_argument("--wad", help="check this compiled map (with its .json) instead of compiling the net")
    p.add_argument("--workers", type=int, default=1, help="DOOM games at once (default 1)")
    p.add_argument("--max-inputs", type=int, default=30000, help="largest --mode all enumeration (default 30000)")
    p.add_argument("--max-classes", type=int, default=20000, help="largest class product (default 20000)")
    p.add_argument("--all-failures", action="store_true", help="count every mismatching input (default: stop at first)")
    p.add_argument("--validate", type=int, metavar="N", help="instead: classes == all on N small nets, tampered fail")
    p.add_argument("--opt", type=int, help="--plane-sweep: schedule level of the map")
    p.add_argument("--plane-sweep", nargs="*", type=int, metavar="BIAS", help="instead: DOOM at the plane-move bound "
                   "(planes.py): the reduced net of fuzz seed 902 with these biases, check bypassed (default -12000 "
                   "-12001 -14006 -14007 -14009 -14010 -14282); --json writes the table")
    p.add_argument("--coverage", action="store_true", help="instead: which of the values the reference defines does a "
                   "pass compare (doomformer.exposure); NET may also be a shipped transformer name")
    p.add_argument("--tiles", action="store_true", help="--coverage: compile into tiles instead of one map")
    p.add_argument("--x", type=int, nargs="+", metavar="V", help="--coverage: the input (default: zeros / the first "
                   "full-length token sequence the range check accepts)")
    p.add_argument("--seed", type=int, default=0, help="seed of --validate")
    p.add_argument("--extract", metavar="WAD", help="instead: static check, no game: recover the net from this compiled "
                   "map alone and compare it with NET (doomformer.extract); exit 1 on a difference")
    p.add_argument("--net", dest="net_opt", metavar="NET", help="NET given as an option (same as the positional)")
    p.add_argument("--json", metavar="FILE", help="write the result record as JSON")


def _coverage_cli(args) -> int:
    from .exposure import coverage
    from .kit import MLP, Transformer
    path = Path(args.net)
    net = MLP.load(path) if path.suffix == ".npz" and path.is_file() else Transformer.load(args.net)
    cov = coverage(net, args.x, mode="tiles" if args.tiles else "map")
    print(cov)
    if args.json:
        Path(args.json).write_text(json.dumps(cov.as_dict(), indent=1) + "\n")
    return 0 if cov.complete else 1


def run_cli(args) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        if args.validate is not None:
            def prog(row):
                print(f"  net {row['net']}: {row['n_in']} in, {row['layers']} layers, domain {row['domain']}: classes "
                      f"{row['clean_classes']['verdict']} ({row['clean_classes']['runs']} runs) / all "
                      f"{row['clean_all']['verdict']} ({row['clean_all']['runs']} runs); tampered "
                      f"{row['tampered_classes']['verdict']} / {row['tampered_all']['verdict']}", flush=True)
            rep = validate(args.validate, args.seed, args.workers, progress=prog)
            print(f"validate: classes verdict == all verdict on {rep['agree']}/{rep['n']} nets (clean: "
                  f"{rep['clean_verified']} VERIFIED), tampered weight line failed both on {rep['tamper_detected']}/"
                  f"{rep['n']}; {rep['runs_classes']:,} runs (classes) vs {rep['runs_all']:,} (all)")
            if args.json:
                Path(args.json).write_text(json.dumps(rep, indent=1) + "\n")
            return 0 if rep["agree"] == rep["n"] and rep["clean_verified"] == rep["n"] else 1
        if args.plane_sweep is not None:
            from .planes import main as plane_main
            return plane_main([str(b) for b in args.plane_sweep] + (["--json", args.json] if args.json else [])
                              + (["--opt", str(args.opt)] if args.opt is not None else []))
        if args.extract:
            from .extract import run_compare
            net_path = args.net_opt or args.net
            if not net_path:
                print("doomformer verify: --extract WAD needs --net NET", file=sys.stderr)
                return 2
            return run_compare(args.extract, net_path, dump=False, json_path=args.json)
        args.net = args.net or args.net_opt
        if not args.net:
            print("doomformer verify: give a NET (.npz saved with MLP.save) or --validate N", file=sys.stderr)
            return 2
        if args.coverage:
            return _coverage_cli(args)
        from .kit import MLP
        net = MLP.load(args.net)
        r = verify(net, args.mode, domain=tuple(args.domain), wad=args.wad, workers=args.workers,
                   max_inputs=args.max_inputs, max_classes=args.max_classes, fail_fast=not args.all_failures)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"doomformer verify: {e}", file=sys.stderr)
        return 2
    print(describe(r, Path(args.net).name))
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=1) + "\n")
    return 0 if r["verdict"] == "VERIFIED" else 1
