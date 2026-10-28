"""Soundness campaign of the schedule checker: `doomformer lint --fuzz`.

Each job is one random net (the generators of `doomformer fuzz`, plus a stratum of mid-size nets drawn as the tuner's
reference MLPs), one opt level (1, 2 or 3) and random timing constants drawn around the failure edges of
docs/tuning.md. The job builds the map, asks the static checker (`check_schedule`: hazards and margins) for its
verdict, runs the net's boundary-seeking inputs (thresholds, thresholds +-1, extremes, random; 16 per net) in DOOM and
compares every exposed value with the integer reference at tolerance 0. Rows classify as

  false negative   the checker reports the map clean, DOOM mismatches (must be 0)
  true positive    flagged, DOOM mismatches
  conservative     flagged, DOOM exact on every input
  true negative    clean, exact

A job that raises while building is counted apart (build_error). A DOOM run that does not settle counts as a
mismatch. Seeds make every job reproducible: job (seed, index, opt) draws its net, inputs and constants from
numpy SeedSequences of those numbers, whatever the worker count.
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

OPTS = (1, 2, 3)
MID_FRACTION = 0.12            # share of the nets drawn as mid-size reference MLPs (more waves, longer corridors)
N_INPUTS = 16

# range each constant is drawn from, in map units (about 16 per tic): the failure edges of docs/tuning.md lie inside
RANGES = {"serial": (8, 72), "clock_margin": (0, 200), "corr_gap": (-64, 64), "tele_gap": (-32, 64),
          "chunk_lead": (-16, 32), "sync_wait": (0, 160), "sync_margin": (-64, 96), "copy_settle": (0, 96)}
CONSTANTS = {1: ("serial", "clock_margin", "sync_wait", "corr_gap", "tele_gap", "chunk_lead"),
             2: ("serial", "clock_margin", "sync_margin", "corr_gap", "tele_gap", "chunk_lead"),
             3: ("serial", "clock_margin", "sync_margin", "corr_gap", "tele_gap", "chunk_lead", "copy_settle")}
OPT3_CLOCK = (0, 320)


def draw_timing(seed: int, index: int, opt: int):
    """Random Timing of a job: the opt level's base constants (defaults at opt 1, tuned at 2, tuned copy at 3) with each
    constant replaced with probability 1/2 (at least one) by a uniform draw from RANGES."""
    from .pure import Timing
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(index), opt, 77]))
    base = Timing() if opt == 1 else Timing.tuned(copy=opt == 3)
    names = CONSTANTS[opt]
    pick = [n for n in names if rng.random() < 0.5] or [names[int(rng.integers(len(names)))]]
    over = {}
    for n in pick:
        lo, hi = OPT3_CLOCK if (n == "clock_margin" and opt == 3) else RANGES[n]
        over[n] = int(rng.integers(lo, hi + 1))
    if opt == 1:
        over.setdefault("sync_margin", None)
    return dataclasses.replace(base, **over), over


def draw_case(seed: int, index: int) -> dict:
    """Net `index` of `seed`: {kind: "mlp" | "mid", spec, inputs, note}. About MID_FRACTION of the nets are mid-size."""
    from . import fuzz, tune
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(index), 991]))
    if rng.random() < MID_FRACTION:
        n_layers = int(rng.integers(2, 4))
        sizes = [int(rng.integers(4, 17))] + [int(rng.integers(4, 25)) for _ in range(n_layers - 1)] + \
                [int(rng.integers(3, 9))]
        K = [int(rng.integers(2, 5)) for _ in range(n_layers)]
        net = tune._mlp(sizes, K, int(rng.integers(0, 2 ** 31)))
        inputs, _, _ = tune.make_inputs(net, N_INPUTS, int(rng.integers(0, 2 ** 31)))
        return dict(id=index, kind="mid", spec=fuzz.spec_of(net), inputs=[[int(v) for v in x] for x in inputs],
                    note=f"mid-size, sizes {sizes}, K {K}")
    case = fuzz.gen_case("mlp", seed, index, n_inputs=N_INPUTS)
    case["kind"] = "mlp"
    return case


def _evaluate(net, case, opt: int, timing, rec: dict, build_only: bool = False) -> dict:
    """Build the map with `timing`, ask the checker, run the case's inputs in DOOM; fills `rec` (see the module doc)."""
    from . import tune
    from .kit import RangeError, _build
    from .schedule import check_schedule
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="dfl-") as d:
        wad = Path(d) / "net.wad"
        try:
            meta = _build(net, wad, opt=opt, timing=timing)
        except (ValueError, ZeroDivisionError, RangeError) as e:
            return dict(rec, verdict="build_error", detail=f"{type(e).__name__}: {e}"[:200])
        rec["estimated_tics"] = meta.layout["estimated_settle_tics"]
        try:
            rep = check_schedule(wad, net=net, witnesses=0)
        except Exception as e:                                                       # the checker must not crash
            return dict(rec, verdict="checker_error", detail=f"{type(e).__name__}: {e}"[:300])
        flagged = not rep.ok
        rec.update(flagged=flagged, hazards=len(rep.hazards), hazard_count=int(rep.hazard_count),
                   margins_failed=sorted({m["name"] for m in rep.margins if not m["ok"]}))
        if build_only:
            return rec
        run = tune.run_map(net, str(wad), case["inputs"])
        rows = run["rows"]
    bad = [i for i, r in enumerate(rows) if r[0] != "ok"]
    ok_tics = [r[1] for r in rows if r[1]]
    rec.update(doom_fail=len(bad), doom_bad_inputs=bad[:8], doom_first=rows[bad[0]][3] if bad else [],
               doom_status=sorted({rows[i][0] for i in bad}), tics=[min(ok_tics), max(ok_tics)] if ok_tics else None,
               seconds=round(time.perf_counter() - t0, 1))
    rec["verdict"] = ("TP" if bad else "conservative") if flagged else ("FN" if bad else "TN")
    if rec["verdict"] == "FN":
        rec["case"] = case
        rec["timing"] = dataclasses.asdict(timing)
    return rec


def _head(seed: int, index: int, opt: int):
    from . import fuzz
    case = draw_case(seed, index)
    rec = dict(seed=seed, index=index, opt=opt, note=case["note"], kind=case["kind"], inputs=len(case["inputs"]))
    return case, rec, fuzz.net_of("mlp", case["spec"])


def run_job(seed: int, index: int, opt: int, build_only: bool = False) -> list[dict]:
    """One random job: [record]. verdict in {"FN", "TP", "conservative", "TN", "build_error", "unsupported",
    "checker_error"}."""
    case, rec, net = _head(seed, index, opt)
    timing, over = draw_timing(seed, index, opt)
    rec.update(mode="random", const=None, constants=over)
    if case.get("planned") == "unsupported":
        return [dict(rec, verdict="unsupported")]
    return [_evaluate(net, case, opt, timing, rec, build_only)]


EDGE_CONSTANTS = {1: ("serial", "clock_margin"), 2: ("serial", "clock_margin"), 3: ("serial", "clock_margin")}
EDGE_RANGE = {"serial": (8, 64), "clock_margin": (0, 512)}


def run_edge_job(seed: int, index: int, opt: int) -> list[dict]:
    """The checker's own boundary: for each constant of EDGE_CONSTANTS, all others at their shipped values (defaults at
    opt 1), bisect the smallest value v at which the checker reports the map clean (the checker is monotone in these
    constants on every net seen; v is also verified: v clean, v - 1 flagged), then run v in DOOM (role "edge-clean":
    DOOM wrong here is a false negative) and v - 1 (role "edge-flagged")."""
    from .kit import RangeError, _build
    from .pure import Timing
    from .schedule import check_schedule
    case, base_rec, net = _head(seed, index, opt)
    if case.get("planned") == "unsupported":
        return [dict(base_rec, mode="edge", const=None, verdict="unsupported")]
    base = Timing() if opt == 1 else Timing.tuned(copy=opt == 3)
    out = []

    def clean(const: str, v: int):
        with tempfile.TemporaryDirectory(prefix="dfl-") as d:
            wad = Path(d) / "net.wad"
            _build(net, wad, opt=opt, timing=dataclasses.replace(base, **{const: v}))
            return check_schedule(wad, net=net, witnesses=0).ok

    for const in EDGE_CONSTANTS[opt]:
        lo, hi = EDGE_RANGE[const]
        rec = dict(base_rec, mode="edge", const=const)
        try:
            if not clean(const, hi):
                out.append(dict(rec, verdict="build_error", detail=f"flagged at {hi}"))
                continue
            if clean(const, lo):
                edge = lo
            else:
                while hi - lo > 1:                                              # lo flagged, hi clean
                    mid = (lo + hi) // 2
                    lo, hi = (lo, mid) if clean(const, mid) else (mid, hi)
                edge = hi
        except (ValueError, ZeroDivisionError, RangeError) as e:
            out.append(dict(rec, verdict="build_error", detail=f"{type(e).__name__}: {e}"[:200]))
            continue
        for role, v in (("edge-clean", edge), ("edge-flagged", edge - 1)):
            if v < EDGE_RANGE[const][0]:
                continue
            r = _evaluate(net, case, opt, dataclasses.replace(base, **{const: v}),
                          dict(rec, role=role, value=v, constants={const: v}))
            out.append(r)
    return out


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    from . import _mp
    _mp.exit_with_parent()


def _job(args):
    seed, index, opt, edge = args
    try:
        return (run_edge_job if edge else run_job)(seed, index, opt)
    except Exception as e:                                                          # keep the campaign going
        return [dict(seed=seed, index=index, opt=opt, mode="edge" if edge else "random", const=None,
                     verdict="job_error", detail=f"{type(e).__name__}: {e}"[:300])]


def _mode(r: dict) -> str:
    return r.get("mode", "random")


def summarize(rows: list[dict], **meta) -> dict:
    """Counts per mode (random draws, edge probes), per opt level and per edge role."""
    out = dict(meta, jobs=len(rows))
    for mode in ("random", "edge"):
        sub = [r for r in rows if _mode(r) == mode]
        v = Counter(r["verdict"] for r in sub)
        scored = [r for r in sub if r["verdict"] in ("FN", "TP", "conservative", "TN")]
        per_opt = {}
        for opt in OPTS:
            c = Counter(r["verdict"] for r in scored if r["opt"] == opt)
            per_opt[opt] = {k: c.get(k, 0) for k in ("FN", "TP", "conservative", "TN")}
        flagged = v["TP"] + v["conservative"]
        entry = dict(maps=len(sub), scored=len(scored), verdicts=dict(v), per_opt=per_opt, false_negatives=v["FN"],
                     flagged=flagged, conservative_rate=round(v["conservative"] / flagged, 4) if flagged else None,
                     doom_wrong=v["TP"] + v["FN"], inputs_run=sum(r.get("inputs", 0) for r in scored),
                     nets=len({(r["seed"], r["index"]) for r in scored}))
        reasons: dict[str, Counter] = {}
        for r in scored:
            if r["flagged"]:
                key = " + ".join((["hazard"] if r["hazards"] else []) + r["margins_failed"])
                reasons.setdefault(key, Counter())[r["verdict"]] += 1
        entry["flag_reasons"] = {k: dict(c) for k, c in sorted(reasons.items(), key=lambda kv: -sum(kv[1].values()))}
        if mode == "edge":
            by = {}
            for r in scored:
                k = f"{r['const']} {r['role']}"
                c = by.setdefault(k, Counter())
                c[r["verdict"]] += 1
            entry["by_constant_role"] = {k: dict(c) for k, c in sorted(by.items())}
        out[mode] = entry
    out["false_negatives"] = out["random"]["false_negatives"] + out["edge"]["false_negatives"]
    return out


def _table(entry: dict) -> list[str]:
    out = ["| opt | maps scored | false negatives | flagged, DOOM wrong (TP) | flagged, DOOM exact (conservative) | "
           "clean, exact (TN) |", "|---|---|---|---|---|---|"]
    for opt, c in entry["per_opt"].items():
        out.append(f"| {opt} | {sum(c.values())} | {c['FN']} | {c['TP']} | {c['conservative']} | {c['TN']} |")
    t = entry["verdicts"]
    out.append(f"| all | {entry['scored']} | {t.get('FN', 0)} | {t.get('TP', 0)} | {t.get('conservative', 0)} | "
               f"{t.get('TN', 0)} |")
    return out


def markdown(s: dict, rows: list[dict]) -> str:
    out = [f"# Lint soundness campaign (seed {s['seed']}, nets {s['start']}..{s['start'] + s['nets_requested'] - 1})",
           "", *[f"command: `{c}`" for c in s.get("commands", [s["command"]])], ""]
    out += [s["notes"], ""] if s.get("notes") else []
    for mode, what in (("random", "random timing constants (each constant replaced with probability 1/2 by a draw from "
                                  "its range around the edges of docs/tuning.md)"),
                       ("edge", "the checker's own boundary: per net, opt level and constant (serial, clock_margin; "
                                "the others shipped) the smallest value the checker calls clean (edge-clean) and one "
                                "below it (edge-flagged)")):
        e = s[mode]
        if not e["maps"]:
            continue
        rate = f"{100 * e['conservative_rate']:.1f} %" if e["conservative_rate"] is not None else "n/a"
        t = e["verdicts"]
        other = ", ".join(f"{k} {n}" for k, n in sorted(t.items()) if k not in ("FN", "TP", "conservative", "TN"))
        out += [f"## {mode}: {what}", "",
                f"{e['maps']} maps over {e['nets']} nets; {e['scored']} scored (build, static verdict, then "
                f"{N_INPUTS} inputs per map in DOOM at tolerance 0; {e['inputs_run']:,} DOOM passes).", ""] + _table(e) + [
                "", f"False negatives: **{e['false_negatives']}**. DOOM was wrong on {e['doom_wrong']} maps. Conservative "
                f"rate (flagged maps DOOM ran exactly, of all flagged): {rate} ({t.get('conservative', 0)} of "
                f"{e['flagged']}). Not scored: {other or 'none'}.", ""]
        if e["flag_reasons"]:
            out += ["Why the checker flagged (hazard = two plane writes that can share a tic; the other names are the "
                    "timing margins below their minimum):", "", "| flagged for | DOOM wrong (TP) | DOOM exact (conservative) |",
                    "|---|---|---|"]
            for k, c in e["flag_reasons"].items():
                out.append(f"| {k} | {c.get('TP', 0)} | {c.get('conservative', 0)} |")
            out.append("")
        if "by_constant_role" in e:
            out += ["| constant, role | " + " | ".join(("FN", "TP", "conservative", "TN")) + " |", "|---|---|---|---|---|"]
            for k, c in e["by_constant_role"].items():
                out.append(f"| {k} | " + " | ".join(str(c.get(v, 0)) for v in ("FN", "TP", "conservative", "TN")) + " |")
            out.append("")
    fn = [r for r in rows if r["verdict"] == "FN"]
    if fn:
        out += ["## False negatives", ""]
        for r in fn:
            out.append(f"- seed {r['seed']} net {r['index']} opt {r['opt']} {_mode(r)} constants {r['constants']}: DOOM "
                       f"wrong on {r['doom_fail']}/{r['inputs']} inputs, values {r['doom_first']} ({r['note']})")
        out.append("")
    return "\n".join(out)


def write_summary(out: Path, **meta) -> dict:
    """Summary of everything in out/lintfuzz.jsonl (random and edge rows, the last row of a job winning), written to
    summary.json and summary.md; commands come from out/commands.txt."""
    out = Path(out)
    rows = {}
    for line in (out / "lintfuzz.jsonl").read_text().splitlines():
        r = json.loads(line)
        rows[_key(r)] = r
    rows = [rows[k] for k in sorted(rows, key=lambda k: tuple(str(v) for v in k))]
    cmds = (out / "commands.txt").read_text().splitlines() if (out / "commands.txt").is_file() else []
    nets = sorted({r["index"] for r in rows})
    s = summarize(rows, seed=rows[0]["seed"], start=nets[0], nets_requested=len(nets), command=cmds[-1] if cmds else "",
                  commands=cmds, opts=list(OPTS), **meta)
    (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    (out / "summary.md").write_text(markdown(s, rows))
    return s


def _key(r: dict) -> tuple:
    return (r["index"], r["opt"], _mode(r), r.get("const"), r.get("role"))


def run(nets: int = 1000, seed: int = 0, start: int = 0, workers: int = 4, opts=OPTS, out: Path | None = None,
        command: str = "", echo=print, only: list | None = None, edge: bool = False) -> dict:
    """Run nets x opts jobs in `workers` games (edge: the edge probes instead of the random constants). out: a
    directory that receives lintfuzz.jsonl (one row per map, appended as jobs finish; a job whose rows are all in it
    is skipped, so an interrupted campaign resumes), summary.json and summary.md. only: a list of (index, opt) to run
    instead of the whole grid (reproduction)."""
    from . import _mp
    out = Path(out) if out else None
    rows: dict[tuple, dict] = {}
    log = None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        log = out / "lintfuzz.jsonl"
        if command:
            with open(out / "commands.txt", "a") as c:
                c.write(command + "\n")
        if log.exists():
            for line in log.read_text().splitlines():
                r = json.loads(line)
                if r["seed"] == seed:
                    rows[_key(r)] = r
    mode = "edge" if edge else "random"
    finished = {(k[0], k[1]) for k in rows if k[2] == mode}
    grid = list(only) if only else [(i, o) for i in range(start, start + nets) for o in opts]
    todo = [(seed, i, o, edge) for i, o in grid if (i, o) not in finished]
    t0 = time.time()
    f = open(log, "a") if log else None
    try:
        if workers <= 1:
            results = (_job(a) for a in todo)
            pool = None
        else:
            ctx = _mp.detached_main()
            ctx.__enter__()
            pool = cf.ProcessPoolExecutor(workers, mp_context=_mp.context(), initializer=_init)
            results = (fu.result() for fu in cf.as_completed([pool.submit(_job, a) for a in todo]))
        for n, recs in enumerate(results, 1):
            for r in recs:
                rows[_key(r)] = r
                if f:
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                if r["verdict"] == "FN":
                    echo("FALSE NEGATIVE " + json.dumps({k: r.get(k) for k in ("seed", "index", "opt", "mode", "const",
                                                                              "constants")}))
            if n % 25 == 0 or n == len(todo):
                c = Counter(x["verdict"] for x in rows.values() if _mode(x) == mode)
                echo(f"[{n}/{len(todo)} jobs {time.time() - t0:.0f}s] {dict(c)}")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
            ctx.__exit__(None, None, None)
        if f:
            f.close()
    allrows = [rows[k] for k in sorted(rows, key=lambda k: tuple(str(v) for v in k))]
    s = summarize(allrows, seed=seed, start=start, nets_requested=nets, command=command, opts=list(opts),
                  workers=workers, minutes=round((time.time() - t0) / 60, 1))
    if out and not only:
        (out / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
        (out / "summary.md").write_text(markdown(s, allrows))
    return s


def add_arguments(p) -> None:
    g = p.add_argument_group("lint --fuzz: soundness campaign of the checker (every map also runs in DOOM)")
    g.add_argument("--nets", type=int, default=1000, help="random nets (each at opt 1, 2 and 3; default 1000)")
    g.add_argument("--seed", type=int, default=0, help="campaign seed (default 0)")
    g.add_argument("--start", type=int, default=0, help="first net index (default 0)")
    g.add_argument("--workers", type=int, default=4, help="DOOM games at once (default 4)")
    g.add_argument("--fuzz-out", metavar="DIR", help="write lintfuzz.jsonl, summary.json and summary.md here "
                                                     "(resumes a log already there)")
    g.add_argument("--edge", action="store_true", help="probe the checker's boundary instead of random constants: per "
                                                       "net, opt and constant (serial, clock_margin) the smallest "
                                                       "value it calls clean, and one below")
    g.add_argument("--only", metavar="INDEX:OPT[,INDEX:OPT]", help="run just these jobs (reproduce a row)")


def run_cli(args) -> int:
    import sys
    only = [tuple(int(v) for v in t.split(":")) for t in args.only.split(",")] if args.only else None
    cmd = "doomformer " + " ".join(sys.argv[1:])
    s = run(args.nets, args.seed, args.start, args.workers, out=args.fuzz_out, command=cmd, only=only, edge=args.edge)
    print(json.dumps({k: s[k] for k in ("random", "edge", "false_negatives")}))
    return 1 if s["false_negatives"] else 0
