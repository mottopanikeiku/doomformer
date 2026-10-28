"""Circuit analysis of the induction model, causal tests executed by voodoo dolls.

    python -m demos.circuit                       # reference tables only, about a minute
    python -m demos.circuit --doom-rows 6 --doom-circuits 2 --workers 3
    python -m demos.circuit models/induction --pairs 400 --target 0.95

The task is `doomformer.induction`: after a prefix of distinct tokens a run of it repeats, and at the last token the model
must give the token that followed its first occurrence. A pair of sequences differs in two swapped
positions and in the answer, either the answer token and another prefix token ("answer" pairs: what is copied) or the
first occurrence of the query token and another prefix token ("source" pairs: how the source is found); see
`doomformer.circuit`. Steps:
1. patching recovery, one cell at a time, clean into corrupt, mean over the pairs: per head (output, q, k, v) and per
   role of the position (query, src, ans, dist or src2 and ans2, bos, rest), and whole x1, m, x2 sites;
2. circuit search: every subset of the heads and MLP layers is run with all other head outputs and MLP hidden units
   replaced by their dataset means; the smallest subset whose logit difference reaches --target of the full model's is
   reported;
3. with --doom-rows / --doom-circuits, the chosen cells and the circuit-only runs also run in tile maps: the dolls do
   every product, sum and staircase, Python patches the bus (docs/interventions.md), and every exposed value, patched cell
   and the logits are compared with the equally intervened reference at tolerance 0.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from doomformer import circuit as C
from doomformer import shipped
from doomformer.intref import QModel

from . import say


def components(m: QModel) -> list[tuple]:
    return [("head", l, h) for l in range(m.L) for h in range(m.H)] + [("mlp", l, None) for l in range(m.L)]


def name(c: tuple) -> str:
    return f"L{c[1]}.H{c[2]}" if c[0] == "head" else f"L{c[1]}.MLP"


def search(m: QModel, pairs, data, target: float, cache: dict, baseline: str = "mean") -> tuple[list[dict], dict]:
    """Faithfulness of every subset of components; returns all rows and the smallest one reaching `target`. baseline:
    "mean" (dataset means, as in the doc) or "corrupt" (the corrupt run's activations, `counterfactual_faithfulness`)."""
    comps = components(m)
    rows = []
    for size in range(len(comps) + 1):
        for subset in itertools.combinations(comps, size):
            heads = [(c[1], c[2]) for c in subset if c[0] == "head"]
            mlps = [c[1] for c in subset if c[0] == "mlp"]
            f = (C.faithfulness(m, pairs, heads, mlps, data, cache=cache) if baseline == "mean" else
                 C.counterfactual_faithfulness(m, pairs, heads, mlps, cache=cache))
            rows.append(dict(subset=subset, heads=heads, mlps=mlps, **{k: v for k, v in f.items() if k != "doom"}))
    reach = [r for r in rows if r["faithfulness"] >= target]
    best = min(reach, key=lambda r: (len(r["subset"]), -r["faithfulness"])) if reach else None
    return rows, best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m demos.circuit", description=__doc__.split("\n\n")[0])
    ap.add_argument("model", nargs="?", default="induction", help="shipped model name or directory")
    ap.add_argument("--pairs", type=int, default=200, help="clean/corrupt pairs per kind from the held-out sequences")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=float, default=0.9, help="faithfulness the searched circuit must reach")
    ap.add_argument("--min-recovery", type=float, default=0.05, help="print cells whose recovery reaches this")
    ap.add_argument("--doom-rows", type=int, default=0, help="patched cells per kind to run in DOOM (largest first; "
                                                              "tautological and whole-residual cells are skipped)")
    ap.add_argument("--doom-kind", choices=sorted(C.ROLES), help="run the --doom-rows cells for this kind only")
    ap.add_argument("--doom-pairs", type=int, default=1, help="pairs per kind each --doom-rows cell runs on in DOOM")
    ap.add_argument("--doom-circuits", type=int, default=0, help="pairs whose circuit-only run also runs in DOOM")
    ap.add_argument("--workers", type=int, default=3, help="DOOM games")
    ap.add_argument("--baseline", choices=("mean", "corrupt"), default="mean",
                    help="what the components that are not kept take: their dataset mean at every position (default) or "
                         "the corrupt run's activations; with 'corrupt' only the circuit search runs (reference only)")
    args = ap.parse_args(argv)
    m = QModel.load(shipped.resolve(args.model))
    if args.baseline == "corrupt":
        for kind in C.ROLES:
            pairs = C.induction_pairs(args.pairs, seed=args.seed, kind=kind)
            rows, best = search(m, pairs, None, args.target, {}, baseline="corrupt")
            say(f"\nkind '{kind}', {args.pairs} pairs: components not kept take the corrupt run's activations; LD full "
                f"{rows[-1]['ld_full']:.1f}, all components from the corrupt run {rows[0]['ld_none']:.1f}")
            say(f"{'size':>4s} {'best subset':44s} {'faithfulness':>12s} {'top-1':>6s}")
            for size in range(len(components(m)) + 1):
                top = max((r for r in rows if len(r["subset"]) == size), key=lambda r: r["faithfulness"])
                say(f"{size:4d} {', '.join(map(name, top['subset'])) or '(nothing)':44s} {top['faithfulness']:12.3f} "
                    f"{top['top1_circuit']:6.2f}")
            say(f"smallest subset reaching {args.target:.0%}: {', '.join(map(name, best['subset']))} "
                f"({best['faithfulness']:.3f}, top-1 {best['top1_circuit']:.2f})" if best else "no subset reaches it")
        return 0
    cell_list = C.cells(m)
    kinds, results = {}, {}
    say(f"{m.L} layers, {m.H} heads, d={m.d}, attention {m.attn}; {args.pairs} pairs per kind (seed {args.seed})")
    for kind in C.ROLES:
        pairs = kinds[kind] = C.induction_pairs(args.pairs, seed=args.seed, kind=kind)
        clean_ok = np.mean([int(np.argmax(C.forward(m, p.clean)[-1])) == p.answer for p in pairs])
        corrupt_ok = np.mean([int(np.argmax(C.forward(m, p.corrupt)[-1])) == p.distractor for p in pairs])
        t0 = time.perf_counter()
        res = results[kind] = C.role_recovery(m, pairs, cell_list)
        say(f"\nkind '{kind}': example clean {pairs[0].clean}, corrupt {pairs[0].corrupt}, answer {pairs[0].answer}, "
            f"distractor {pairs[0].distractor}; the model gives the answer on {clean_ok:.1%} of the clean and the "
            f"distractor on {corrupt_ok:.1%} of the corrupt sequences")
        say(f"patching recovery, clean into corrupt, one cell at a time, mean over {len(pairs)} pairs "
            f"({time.perf_counter() - t0:.0f}s); cells with a role at or above {args.min_recovery}")
        say(f"{'cell':12s}" + "".join(f"{r:>7s}" for r in res["roles"]))
        for ci, (site, head) in enumerate(cell_list):
            row = res["recovery"][ci]
            if row.max() >= args.min_recovery:
                label = site + (f" h{head}" if head is not None else "")
                mark = {"tautological": "T", "whole residual": "W", "whole site": "S", "": " "}
                say(f"{label:12s}" + "".join(f"{v:7.2f}" + mark[res["flags"][ci][ri]] for ri, v in enumerate(row)))
        say("T tautological (the whole final residual at the last position: 1 by construction), W whole residual "
            "(every unit of the stream at that position), S whole site (every unit of a site at the last position); "
            "none is a finding")

    pairs, data = kinds["answer"], C.mean_data(kinds["answer"])
    t0 = time.perf_counter()
    cache: dict = {}
    rows, best = search(m, pairs, data, args.target, cache)
    say(f"\ncircuit search ('answer' pairs): all {len(rows)} subsets of {len(components(m))} components (heads and MLP "
        f"layers) kept, everything else mean-ablated at every position ({time.perf_counter() - t0:.0f}s)")
    say(f"{'size':>4s} {'best subset':44s} {'faithfulness':>12s} {'top-1':>6s}")
    for size in range(len(components(m)) + 1):
        top = max((r for r in rows if len(r["subset"]) == size), key=lambda r: r["faithfulness"])
        say(f"{size:4d} {', '.join(map(name, top['subset'])) or '(nothing)':44s} {top['faithfulness']:12.3f} "
            f"{top['top1_circuit']:6.2f}")
    if best is None:
        say(f"no subset reaches {args.target:.0%}")
    else:
        say(f"smallest circuit reaching {args.target:.0%}: {', '.join(map(name, best['subset'])) or '(nothing)'} "
            f"restores {best['faithfulness']:.1%} of the logit difference (top-1 {best['top1_circuit']:.2f}; full model "
            f"{best['top1_full']:.2f})")
    n_comp = len(components(m))
    show = best
    if best is None or len(best["subset"]) == n_comp:                 # the whole model: show the best proper subset
        show = max((r for r in rows if len(r["subset"]) == n_comp - 1), key=lambda r: r["faithfulness"])
        say(f"no proper subset reaches {args.target:.0%}; the best one has {len(show['subset'])} components "
            f"({', '.join(map(name, show['subset']))}) and restores {show['faithfulness']:.1%}")

    if not (args.doom_rows or args.doom_circuits):
        return 0
    from doomformer.patching import DoomRuns
    exact = total = cells_patched = values = 0
    with DoomRuns(m, workers=args.workers) as runs:
        for kind, res in results.items():
            if not args.doom_rows:
                break
            if args.doom_kind and kind != args.doom_kind:
                continue
            order = sorted(((ci, ri) for ci in range(len(cell_list)) for ri in range(len(res["roles"]))
                            if not res["flags"][ci][ri]), key=lambda x: -res["recovery"][x])[: args.doom_rows]
            picks = [(ci, res["roles"][ri], pi) for ci, ri in order for pi in range(args.doom_pairs)]
            out = C.role_recovery(m, kinds[kind][:args.doom_pairs], cell_list, doom_rows=picks, runs=runs)
            say(f"\nthe {len(order)} largest unflagged cells of kind '{kind}' patched in DOOM, pairs "
                f"0..{args.doom_pairs - 1}")
            for r in out["doom"]:
                site, head = r["cell"]
                say(f"  pair {r['pair']} {site}{'' if head is None else f' h{head}'} at {r['role']:5s} recovery {r['recovery']:6.3f}  "
                    f"{'EXACT' if r['exact'] else 'MISMATCH'} ({r['values']} values, {r['patched'][1]} patched cells, "
                    f"{r['seconds']:.0f}s)")
                exact, total = exact + r["exact"], total + 1
                cells_patched, values = cells_patched + r["patched"][1], values + r["values"]
        if args.doom_circuits:
            out = C.faithfulness(m, pairs, show["heads"], show["mlps"], data, runs=runs,
                                 doom_pairs=range(args.doom_circuits), cache=cache)
            say(f"\ncircuit-only ({', '.join(map(name, show['subset']))}) and all-ablated runs in DOOM, "
                f"{args.doom_circuits} pairs")
            for r in out["doom"]:
                say(f"  pair {r['pair']} {r['run']:7s} {'EXACT' if r['exact'] else 'MISMATCH'} ({r['values']} values, "
                    f"{r['patched'][1]} patched cells, {r['seconds']:.0f}s)")
                exact, total = exact + r["exact"], total + 1
                cells_patched, values = cells_patched + r["patched"][1], values + r["values"]
    say(f"\n{exact}/{total} intervened runs EXACT in DOOM: {cells_patched} patched cells, {values} values compared at "
        "tolerance 0")
    return 0 if exact == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
