"""A compiled ground-truth circuit: a one-layer integer transformer whose weights are set by construction.

    python -m doomformer.groundtruth                    # recovery table against the planted structure, reference only
    python -m doomformer.groundtruth --doom 3 --workers 3

The algorithm (the model is built, not trained): at the last position t of [BOS, a, b, c] the logits one-hot the token
(b + c) mod 4. Vocabulary "$0123"; the residual has 16 dimensions in four blocks of four:

    dims 0..3    CUR   one-hot of the token at this position (the embedding)
    dims 4..7    POS   one-hot of the position (the embedding); after attention they carry head 1's output (unread)
    dims 8..11   PREV  one-hot of the token at t - 1, written by head 0
    dims 12..15  OUT   one-hot of the answer, written by the MLP and read by the unembedding

    head 0 (units 0..7)   attends to position t - 1 (q one-hots t - 1, k one-hots s; the score is 1 iff s = t - 1) and
                          copies the token one-hot of that position; Wo writes it to PREV
    head 1 (units 8..15)  a decoy that attends to t - 2 and copies that token one-hot; Wo writes it to the POS block of
                          x1, which no later weight reads (the MLP reads CUR and PREV, the residual weight r2 is 0, the
                          unembedding reads OUT): it is active and has no causal effect on any logit
    MLP (16 units)        unit 4a + b fires iff PREV = a and CUR = b (z = PREV_a + CUR_b - 1 >= 1) and adds 1 to OUT
                          dimension (a + b) mod 4
    residual weights      r1 = 1 on CUR (the current token reaches the MLP through the residual stream), 0 elsewhere

So the planted structure is: the token at t - 1 reaches position t only through head 0 (not head 1), the token at t
only through the residual stream (no head), and every combination happens in the MLP. `planted_cells` derives, from this
construction alone, the cells (site, position, unit) a clean-into-corrupt patch can recover for a pair of contexts;
`check_recovery` runs every single-unit and single-head patch of every pair on the integer reference and compares the set
of cells with nonzero recovery with it. All weights are 0 or 1, thresholds [1] (one level per sign), so the model runs in
tile maps like any other (`python -m doomformer.groundtruth --doom N`).
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

from . import intervene as iv
from .intref import QModel, check_limits, forward

VOCAB = "$0123"
N = 4                                  # digits
T = 4                                  # BOS, a, b, c
CUR, POS, PREV, OUT = 0, 4, 8, 12      # first dimension of each block
D = 16
HEADS = ("L0.H0 (previous token)", "L0.H1 (decoy: t - 2)")


def planted_model(backup: bool = False, negative: bool = False, need_both: bool = False) -> QModel:
    """The model of the module docstring. backup=True: head 1 is not a decoy but a backup of head 0: it also attends to
    t - 1, copies that token and writes it to PREV as well (the staircase saturates, so either head alone gives the same
    x1). need_both=True (with backup): the same two heads, but x1 needs both of them (threshold 2 on PREV, CUR weight
    2): an AND where backup is an OR. negative=True: head 1 attends to the position itself and writes -1 on OUT[token]
    (the MLP writes 2 and x2 has two levels), so it suppresses the logit of the current token: it lowers or raises the
    logit difference by 1 in half of the contexts and never changes the top-1 token. Nothing else changes."""
    if negative and (backup or need_both) or need_both and not backup:
        raise ValueError("negative stands alone; need_both needs backup")
    z = lambda *shape: np.zeros(shape, dtype=np.int64)
    p = dict(tok_emb=z(len(VOCAB), D), pos_emb=z(T, D), thr_x0=np.array([1]), Wun=z(len(VOCAB), D), bun=z(len(VOCAB)))
    for a in range(N):
        p["tok_emb"][1 + a, CUR + a] = 1
        p["Wun"][1 + a, OUT + a] = 1
    for t in range(T):
        p["pos_emb"][t, POS + t] = 1
    L = {n: z(D, D) for n in ("Wq", "Wk", "Wv", "Wo")}
    for h, lag in ((0, 1), (1, 1 if backup else 2)):
        for j in range(T):
            L["Wk"][8 * h + j, POS + j] = 1                 # k_j = [s == j]
            if j + lag < T:
                L["Wq"][8 * h + j, POS + j + lag] = 1       # q_j = [t == j + lag]: score = [s == t - lag]
            if j < N:
                L["Wv"][8 * h + j, CUR + j] = 1             # v = token one-hot of position s
    for j in range(N):
        L["Wo"][PREV + j, j] = 1                            # head 0 -> PREV
        L["Wo"][(PREV if backup else POS) + j, 8 + j] = 1   # head 1 -> the unread POS block of x1 (backup: PREV too)
    up, dn = z(N * N, D), z(D, N * N)
    for a, b in itertools.product(range(N), repeat=2):
        up[N * a + b, PREV + a] = up[N * a + b, CUR + b] = 1
        dn[OUT + (a + b) % N, N * a + b] = 1
    r1 = z(D)
    r1[CUR:CUR + N] = 1
    layer = {**L, **{"b" + k: z(D) for k in "qkv"}, **{"thr_" + k: np.array([1]) for k in "qkv"},
             "thr_a": np.array([1]), "thr_o": np.ones((T, 1), dtype=np.int64), "r1": r1, "bo": z(D),
             "thr_x1": np.array([1]), "Wup": up, "bup": -np.ones(N * N, dtype=np.int64), "thr_m": np.array([1]),
             "Wdn": dn, "r2": z(D), "bdn": z(D), "thr_x2": np.array([1])}
    p.update({f"L0.{k}": v for k, v in layer.items()})
    K = 1
    if need_both:
        p["L0.thr_x1"] = np.array([2])
        p["L0.r1"] = 2 * p["L0.r1"]
    if negative:
        K = 2
        for j in range(T):
            p["L0.Wq"][8 + j, :] = 0
            p["L0.Wq"][8 + j, POS + j] = 1                  # q_j = [t == j], k_j = [s == j]: head 1 attends to itself
        for j in range(N):
            p["L0.Wo"][POS + j, 8 + j] = 0
            p["L0.Wo"][OUT + j, 8 + j] = -1                 # and writes -1 on OUT[token]
        p["L0.Wdn"] = 2 * p["L0.Wdn"]
        p["L0.r2"] = z(D)
        p["L0.r2"][OUT:OUT + N] = 1                         # x1's OUT dimensions reach x2
        p["L0.thr_x2"] = np.array([1, 2])
        for name in ("thr_x1", "thr_q", "thr_k", "thr_v"):
            p["L0." + name] = np.array([1, 50])
        p["thr_x0"] = np.array([1, 50])                     # second levels no value reaches: the staircases keep K = 2
    return QModel(VOCAB, T, D, 2, N * N, K, K, K, 1, p)


def contexts() -> list[list[int]]:
    """Every context [BOS, a, b, c] (64)."""
    return [[0, 1 + a, 1 + b, 1 + c] for a, b, c in itertools.product(range(N), repeat=3)]


def answer(tokens: list[int]) -> int:
    """The token the algorithm outputs at the last position: (b + c) mod 4."""
    return 1 + ((tokens[2] - 1) + (tokens[3] - 1)) % N


def top1(m: QModel) -> float:
    """Share of the 64 contexts whose last-position argmax is the algorithm's answer."""
    return float(np.mean([int(np.argmax(forward(m, c)[-1])) == answer(c) for c in contexts()]))


def make_pairs(kind: str) -> list[tuple[list[int], list[int]]]:
    """(clean, corrupt) pairs that differ in one token: kind "prev" changes b (position 2), "cur" changes c (position 3),
    "decoy" changes a (position 1: read only by the decoy head, so the logits cannot change)."""
    pos = {"prev": 2, "cur": 3, "decoy": 1}[kind]
    out = []
    for clean in contexts():
        for v in range(1, N + 1):
            if v != clean[pos]:
                corrupt = list(clean)
                corrupt[pos] = v
                out.append((clean, corrupt))
    return out


def planted_cells(kind: str, clean: list[int], corrupt: list[int], backup: bool = False,
                  need_both: bool = False) -> dict[tuple[str, int], set[int]]:
    """{(site kind, position): units} a clean-into-corrupt patch can recover, from the construction alone.

    prev: the token b -> b' at position 2 sits in x0 (CUR units b, b'), head 0 copies it from the value of position 2 into
    its output at position 3 (units b, b' of its first four), x1 at 3 holds it in PREV, the MLP units 4b + c and 4b' + c
    fire, x2 at 3 holds the two answers in OUT. cur: the token c -> c' at position 3 sits in x0 and in x1 (CUR, through
    the residual weight), the MLP units 4b + c and 4b + c', x2 as before; no head. decoy: nothing.

    backup=True (both heads copy t - 1 into PREV): a single patch into one head's v or ao leaves the other head writing
    the corrupt token b', so x1 at 3 holds both tokens and the logits tie (recovery 0.5); only the unit of the new token b
    in that head (b in head 0, 8 + b in head 1) changes anything, the unit of b' (patched to 0) is covered by the other head
    (recovery 0). The cells shared by both heads (x0 at 2, x1, m, x2 at 3) are as before. Each head alone reaches 0.5,
    both together 1.0: single-cell recovery splits the credit instead of finding the redundancy.

    need_both=True (an AND of the two heads): the roles of b and b' swap. Writing b into one head leaves b' in both heads
    at the threshold, so nothing changes (0); patching the b' unit of one head to 0 takes b' below the threshold and
    leaves nothing above it, a tie (0.5). The unit of b' in each head is planted, the unit of b is not."""
    cells: dict[tuple[str, int], set[int]] = {}
    b, c, b2, c2 = clean[2] - 1, clean[3] - 1, corrupt[2] - 1, corrupt[3] - 1
    if kind == "prev":
        head = {b2, 8 + b2} if need_both else {b, 8 + b} if backup else {b, b2}
        cells = {("x0", 2): {CUR + b, CUR + b2}, ("v", 2): head, ("ao", 3): head,
                 ("x1", 3): {PREV + b, PREV + b2}, ("m", 3): {N * b + c, N * b2 + c},
                 ("x2", 3): {OUT + (b + c) % N, OUT + (b2 + c) % N}}
    elif kind == "cur":
        cells = {("x0", 3): {CUR + c, CUR + c2}, ("x1", 3): {CUR + c, CUR + c2}, ("m", 3): {N * b + c, N * b + c2},
                 ("x2", 3): {OUT + (b + c) % N, OUT + (b + c2) % N}}
    return cells


def _head_of(unit: int) -> int:
    return unit // 8


def _expected_nonzero(kind: str, spec: dict, cells: dict) -> bool:
    key = (spec["site"].split(".")[-1], spec["pos"][0])
    units = cells.get(key)
    if not units:
        return False
    if spec.get("heads") is not None:
        return any(_head_of(u) in spec["heads"] for u in units)
    return bool(units & set(spec["units"]))


def cell_specs(m: QModel) -> list[dict]:
    """Every single-unit patch cell (site x position x unit) and every single-head cell of ao, q, k, v."""
    from .patching import grid
    return grid(m, per_unit=True) + grid(m, kinds=["ao", "q", "k", "v"], per_head=True)


def check_recovery(m: QModel, kinds=("prev", "cur", "decoy"), pairs_per_kind: int | None = None, out=print,
                   backup: bool = False, need_both: bool = False) -> dict:
    """Patch every cell of every pair (clean into corrupt) and compare the cells with nonzero recovery with
    `planted_cells`. Returns per kind: pairs, cells tested, disagreements, the recovery table averaged over the pairs
    and the planted-cell counts."""
    from .patching import patch_recovery
    specs = cell_specs(m)
    result = {}
    for kind in kinds:
        pairs = make_pairs(kind)
        if pairs_per_kind:
            pairs = pairs[::max(1, len(pairs) // pairs_per_kind)][:pairs_per_kind]
        wrong, tested, planted, nonzero, active = [], 0, 0, 0, 0
        agg: dict[tuple, list] = {}          # (site, position, "head h" or "units") -> [cells, planted, nonzero, recovery sum]
        t0 = time.perf_counter()
        for clean, corrupt in pairs:
            cells = planted_cells(kind, clean, corrupt, backup, need_both)
            if kind == "decoy":                 # the logits cannot change: nothing to recover; measure that instead
                tc, tk = {}, {}
                lc, lk = forward(m, clean, tc)[-1], forward(m, corrupt, tk)[-1]
                active += int((tc["L0.ao"][3, 8:] != tk["L0.ao"][3, 8:]).any())
                if not np.array_equal(lc, lk):
                    wrong.append(("logits differ", clean, corrupt))
                continue
            ac, ak = answer(clean), answer(corrupt)
            tab = patch_recovery(m, clean, corrupt, specs, answer=(ac, ak))
            for spec, row in zip(specs, tab.rows):
                want = _expected_nonzero(kind, spec, cells)
                got = row["recovery"] != 0
                tested += 1
                planted += want
                nonzero += got
                a = agg.setdefault((spec["site"], spec["pos"][0], row["cells"] if spec.get("heads") is not None else "units"),
                                   [0, 0, 0, 0.0])
                a[0] += 1
                a[1] += want
                a[2] += got
                a[3] += row["recovery"] if got else 0.0
                if want != got:
                    wrong.append((spec["site"], spec["pos"], row["cells"], row["recovery"], clean, corrupt))
        result[kind] = dict(pairs=len(pairs), decoy_active=active, tested=tested, planted=planted, nonzero=nonzero, wrong=wrong, agg=agg,
                            seconds=time.perf_counter() - t0)
        if kind == "decoy":
            out(f"decoy: {len(pairs)} pairs; head 1's output at the last position changes in {active} of them and the "
                f"logits in {len(wrong)}")
            continue
        out(f"{kind}: {len(pairs)} pairs, {tested} cells tested, {planted} planted, {nonzero} with nonzero recovery, "
            f"{len(wrong)} disagreements ({time.perf_counter() - t0:.0f}s)")
    return result


def recovery_table(result: dict, kind: str) -> str:
    """Per site, position and level (every single unit, or one head), over all pairs of `kind`: cells tested, cells the
    construction plants, cells with nonzero recovery and the mean recovery of those. Rows with neither are omitted."""
    rows = [(k, v) for k, v in sorted(result[kind]["agg"].items()) if v[1] or v[2]]
    lines = [f"{'site':7s} {'pos':>3s} {'level':8s} {'tested':>7s} {'planted':>8s} {'nonzero':>8s} {'mean recovery of nonzero':>25s}"]
    lines += [f"{site:7s} {pos:>3d} {lvl:8s} {n:7d} {pl:8d} {nz:8d} {(rec / nz if nz else 0):25.2f}"
              for (site, pos, lvl), (n, pl, nz, rec) in rows]
    return "\n".join(lines)


def circuit_check(m: QModel, doom_contexts: int = 0, workers: int = 2, out=print,
                  doom_keep=(("H0", "MLP"),)) -> dict:
    """Mean-ablation circuit search over the three components (L0.H0, L0.H1, the MLP) on the 64 contexts: every subset
    keeps its components and replaces the rest at every position by the mean over the contexts. Returns the faithfulness
    (share of the logit difference restored) and top-1 accuracy of each subset. doom_contexts > 0: the circuit-only run
    (H1 mean-ablated) and the all-ablated run of the first doom_contexts contexts also run in DOOM, for each subset of
    doom_keep (component names); their results are "doom" in the row of the subset."""
    from . import circuit as C
    pairs = [C.Pair("answer", c, c, answer(c), 1 + (answer(c) % N), {}) for c in contexts()]   # distractor: the next digit
    data = [c for c in contexts()]
    comps = [("H0", (0, 0)), ("H1", (0, 1)), ("MLP", None)]
    rows = {}
    cache: dict = {}
    for r in range(len(comps) + 1):
        for subset in itertools.combinations(comps, r):
            heads = [h for _, h in subset if h is not None]
            mlp = [0] if any(h is None for _, h in subset) else []
            res = C.faithfulness(m, pairs, heads, mlp, data=data, cache=cache)
            rows[tuple(n for n, _ in subset)] = res
            out(f"  keep {'+'.join(n for n, _ in subset) or 'nothing':10s} faithfulness {res['faithfulness']:.3f}  "
                f"top-1 {res['top1_circuit']:.3f}  per context: model's top-1 kept {res['agree_top1']:.3f}, "
                f"logit difference equal {res['agree_ld']:.3f}, worst ratio {res['ratio_min']:.2f}")
    if doom_contexts:
        from .patching import DoomRuns
        by_name = dict(comps)
        with DoomRuns(m, workers=workers) as runs:
            for names in doom_keep:
                heads = [by_name[n] for n in names if by_name[n] is not None]
                mlp = [0] if "MLP" in names else []
                res = C.faithfulness(m, pairs, heads, mlp, data=data, runs=runs, doom_pairs=range(doom_contexts),
                                     cache=cache)
                ok = sum(bool(r["exact"]) for r in res["doom"])
                out(f"  DOOM, circuit {'+'.join(names)} (the rest mean-ablated) and all-ablated runs on {doom_contexts} "
                    f"contexts: {ok}/{len(res['doom'])} EXACT at tolerance 0, {sum(r['values'] for r in res['doom'])} "
                    f"values, {sum(r['patched'][1] for r in res['doom'])} patched cells")
                rows[names]["doom"] = res["doom"]
    return rows


def negative_report(m: QModel, args) -> int:
    """The negative-interference model: head 1 attends to the position itself and subtracts 1 from the logit of the
    current token. There is no planted-cell set for it: the sign of a cell's recovery depends on the context (the
    suppressed token is the answer, the distractor or neither). What is reported: the recovery of head 1's cells over all
    "cur" pairs, the circuit search with per-context numbers next to the ratio of means, and, with --doom, DOOM rows."""
    from .patching import DoomRuns, patch_recovery
    specs = [dict(site="L0.ao", pos=(3,), heads=(0,)), dict(site="L0.ao", pos=(3,), heads=(1,)),
             dict(site="L0.v", pos=(3,), heads=(1,)), dict(site="L0.x2", pos=(3,), units=tuple(range(OUT, OUT + N)))]
    rec = {k: [] for k in range(len(specs))}
    pairs = make_pairs("cur")
    for clean, corrupt in pairs:
        tab = patch_recovery(m, clean, corrupt, specs, answer=(answer(clean), answer(corrupt)))
        for k, r in enumerate(tab.rows):
            rec[k].append(r["recovery"])
    print(f"\nrecovery over {len(pairs)} \"cur\" pairs (clean into corrupt): min, mean, max")
    for k, spec in enumerate(specs):
        v = np.array(rec[k])
        print(f"  {spec['site']} position 3 {'head ' + str(spec['heads'][0]) if 'heads' in spec else 'OUT units'}: "
              f"{v.min():.2f}, {v.mean():.2f}, {v.max():.2f}")
    print("\ncircuit search (mean ablation of everything not kept):")
    rows = circuit_check(m, args.doom_circuit, args.workers)
    h0 = rows[("H0", "MLP")]
    print(f"\nH0 + MLP keeps the model's top-1 token in {h0['agree_top1']:.0%} of the contexts and its logit difference in "
          f"{h0['agree_ld']:.0%}; the ratio of means reads {h0['faithfulness']:.3f}; the worst per-context ratio is "
          f"{h0['ratio_min']:.2f}. Head 1 looks irrelevant to the ratio of means and to zero-ablation (top-1 kept).")
    bad = 0
    if args.doom:
        rng = np.random.default_rng(0)
        total = exact = 0
        with DoomRuns(m, workers=args.workers) as runs:
            for i in rng.choice(len(pairs), args.doom, replace=False):
                clean, corrupt = pairs[int(i)]
                extra = [dict(site=s_, pos=(p_,), units=(int(u),)) for s_, p_, u in
                         [(a, b, c) for a in ("L0.x1", "L0.x2") for b in (2, 3) for c in rng.choice(D, 2, replace=False)]]
                allspecs = specs + extra
                tab = patch_recovery(m, clean, corrupt, allspecs, answer=(answer(clean), answer(corrupt)),
                                     doom=range(len(allspecs)), runs=runs)
                for r in tab.rows:
                    total += 1
                    exact += bool(r["doom"])
                    print(f"cur {clean}->{corrupt} {r['site']} pos {r['pos'][0]} {r['cells']}: recovery "
                          f"{r['recovery']:.2f} {'EXACT' if r['doom'] else 'MISMATCH'} ({r['values']} values)")
        print(f"DOOM: {exact}/{total} rows exact at tolerance 0")
        bad = total - exact
    if args.doom_circuit:
        bad += sum(not r["exact"] for r in h0["doom"])
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m doomformer.groundtruth", description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=int, default=0, help="pairs per kind for the reference table (0: all)")
    ap.add_argument("--doom", type=int, default=0, help="pairs per kind whose rows run in DOOM (all planted rows and "
                                                        "--doom-off unplanted rows of each)")
    ap.add_argument("--doom-off", type=int, default=12, help="unplanted rows run in DOOM per pair")
    ap.add_argument("--doom-circuit", type=int, default=0, help="contexts (of 64) whose circuit-only and all-ablated runs "
                                                                "also run in DOOM")
    ap.add_argument("--backup", action="store_true", help="the redundant model: head 1 is a backup of head 0 (an OR)")
    ap.add_argument("--need-both", action="store_true", help="the same two heads as an AND: x1 needs both")
    ap.add_argument("--negative", action="store_true",
                    help="head 1 writes -1 on the current token's logit (negative interference); no planted-cell rule")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save", metavar="DIR", help="write the model (config.json, params.npz) to DIR")
    args = ap.parse_args(argv)
    redundant = args.backup or args.need_both
    m = planted_model(redundant, args.negative, args.need_both)
    errs = check_limits(m, contexts(), planes=False)
    print(f"planted model: d={m.d}, {m.H} heads, MLP {m.dm}, T={m.T}; DOOM limits: {errs or 'ok'}; top-1 of "
          f"(b + c) mod 4 on all 64 contexts: {top1(m):.3f}")
    if args.save:
        m.save(Path(args.save))
    if args.negative:
        return negative_report(m, args)
    result = check_recovery(m, kinds=("prev", "cur") if redundant else ("prev", "cur", "decoy"),
                            pairs_per_kind=args.pairs or None, backup=redundant, need_both=args.need_both)
    for kind in ("prev", "cur"):
        print(f"\nkind {kind} ({result[kind]['pairs']} pairs; every unit and every head of every site and position "
              f"patched singly into the corrupt run)\n{recovery_table(result, kind)}")
    bad = sum(len(r["wrong"]) for r in result.values())
    print(f"\ncells where nonzero recovery differs from the planted set: {bad}")
    from .patching import grid, patch_recovery
    clean, corrupt = make_pairs("prev")[0]
    print(f"\nwhole-site and per-head rows of one pair, clean {clean} corrupt {corrupt}:")
    print(patch_recovery(m, clean, corrupt, grid(m, positions=[2, 3], per_head=True),
                         answer=(answer(clean), answer(corrupt))).text())
    if redundant:
        single = patch_recovery(m, clean, corrupt, [dict(site="L0.ao", pos=(3,), heads=(0,)),
                                                    dict(site="L0.ao", pos=(3,), heads=(1,))],
                                answer=(answer(clean), answer(corrupt)))
        print("\nredundancy: head 0 alone, head 1 alone patched at position 3: recovery "
              + ", ".join(f"{r['recovery']:.2f}" for r in single.rows)
              + " (both heads together rewrite the whole site at the read position: flagged, not evidence)")
        knock = [np.mean([int(np.argmax(forward(m, c, interventions=[iv.zero("L0.ao", heads=[h])])[-1])) == answer(c)
                          for c in contexts()]) for h in (0, 1)]
        for h, t1 in enumerate(knock):
            print(f"zero-ablating head {h} alone: top-1 {t1:.3f} on the 64 contexts")
        print("-> " + ("neither head is necessary (an OR)" if min(knock) == 1.0 else
                       "each head is necessary (an AND)" if max(knock) == 0.0 else "mixed"))
    print("\ncircuit search (mean ablation of everything not kept):")
    keep = ((("H0", "H1", "MLP"),) if args.need_both else (("H0", "MLP"), ("H1", "MLP")) if args.backup
            else (("H0", "MLP"),))
    circuit = circuit_check(m, args.doom_circuit, args.workers, doom_keep=keep)
    if args.doom_circuit and not all(r["exact"] for k in keep for r in circuit[k]["doom"]):
        return 1
    if args.doom:
        from .patching import DoomRuns, patch_recovery
        rng = np.random.default_rng(0)
        specs = cell_specs(m)
        total = exact = 0
        with DoomRuns(m, workers=args.workers) as runs:
            for kind in ("prev", "cur"):
                pairs = make_pairs(kind)
                for clean, corrupt in [pairs[i] for i in rng.choice(len(pairs), args.doom, replace=False)]:
                    cells = planted_cells(kind, clean, corrupt, redundant, args.need_both)
                    on = [i for i, s in enumerate(specs) if _expected_nonzero(kind, s, cells)]
                    off = [i for i in range(len(specs)) if i not in set(on)]
                    pick = on + [int(i) for i in rng.choice(off, args.doom_off, replace=False)]
                    tab = patch_recovery(m, clean, corrupt, specs, answer=(answer(clean), answer(corrupt)),
                                         doom=pick, runs=runs)
                    for i in pick:
                        r = tab.rows[i]
                        total += 1
                        exact += bool(r["doom"])
                        print(f"{kind} {clean}->{corrupt} {r['site']} pos {r['pos'][0]} {r['cells']}: recovery "
                              f"{r['recovery']:.2f} planted={i in set(on)} {'EXACT' if r['doom'] else 'MISMATCH'} "
                              f"({r['values']} values)")
        print(f"DOOM: {exact}/{total} rows exact at tolerance 0")
        return 0 if exact == total else 1
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
