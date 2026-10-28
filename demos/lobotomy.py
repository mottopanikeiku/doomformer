"""Lobotomy: shoot a few neurons of the marine's brain and see what marine is left.

The policy reads one of 86 observations (17 bearings x 5 distances, plus "no enemy"), so every single-unit ablation and
every random shotgun blast can be scanned exactly on all of them with the integer reference (`intref`). A blast is
`pellets` neurons of one activation; closing a neuron's doors in every tile that consumes it is what
`DoomTransformer(ablate=...)` does in DOOM. The picked lesion is then played for real (the reference chooses, ViZDoom
simulates `defend_the_center`) and a few of its decisions run on voodoo dolls with the doors closed and are compared,
tensor by tensor, with the equally lesioned reference.

`--minimal [--null]` and `--names` replace the scan by exact lesion statistics, computed with the batch evaluator
`doomformer.lesion` (equal to `intref.forward(..., ablate=)`): every subset size per site, all small subsets of all
neurons, the effect of a lesion as logit KL, a weight-shuffled null, and the single neurons of the name model.
Numbers and commands: docs/DEMOS.md, "Lesion statistics".
"""
from __future__ import annotations

import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from doomformer import intref, lesion, policy, shipped
from doomformer.intref import QModel

from . import say
from .shotgun import ablation_keys

DOCUMENTED = {"x0": [0, 2, 3, 5, 7, 11, 14]}      # the README's blast on models/doomplayer: a pacifist


def action_table(m: QModel, contexts, ablate=None) -> list[int]:
    return [policy.act_index(intref.forward(m, c, ablate=ablate)[-1]) for c in contexts]


def personality(table: list[int], base: list[int], contexts) -> str:
    c = Counter(table)
    n = len(table)
    aligned = [i for i, ctx in enumerate(contexts) if policy.VOCAB[ctx[1]] in "hij"]
    fire_aligned = sum(table[i] == 2 for i in aligned)
    if c[2] == 0:
        return "PACIFIST (never fires)"
    if c[0] == n or c[1] == n:
        return f"SPINS {'LEFT' if c[0] == n else 'RIGHT'} FOREVER"
    if c[2] >= 0.8 * n:
        return "TRIGGER HAPPY (fires at everything)"
    if c[0] == 0 or c[1] == 0:
        return f"CAN'T TURN {'LEFT' if c[0] == 0 else 'RIGHT'}"
    if fire_aligned < sum(base[i] == 2 for i in aligned):
        return "WON'T SHOOT WHAT'S IN FRONT OF HIM"
    return "SUBTLY DIFFERENT"


def scan(m: QModel, contexts, base: list[int], blasts: int, pellets: int, seed: int = 0) -> dict:
    """Every single neuron, then `blasts` random blasts of `pellets` neurons of one activation, each run on all
    contexts. Returns counts and the personality of every blast that changes at least one action."""
    keys = ablation_keys(m)
    names = list(keys)
    rng = np.random.default_rng(seed)
    singles = [{k: [u]} for k, n in keys.items() for u in range(n)]
    shots = []
    for _ in range(blasts):
        k = names[rng.integers(len(names))]
        shots.append({k: sorted(rng.choice(keys[k], pellets, replace=False).tolist())})
    found = {"singles": len(singles), "single_changed": 0, "blasts": blasts, "blast_changed": 0,
             "personalities": Counter(), "pacifists": []}
    for group, ablations in (("single", singles), ("blast", shots)):
        for ab in ablations:
            table = action_table(m, contexts, ab)
            if table == base:
                continue
            found[f"{group}_changed"] += 1
            who = personality(table, base, contexts)
            if group == "blast":
                found["personalities"][who] += 1
                if who.startswith("PACIFIST"):
                    found["pacifists"].append(ab)
    return found


def play(m: QModel, ablate, episodes: int, seed0: int) -> dict:
    """Mean kills and survival tics over `episodes` defend_the_center episodes; the reference chooses each action."""
    game = policy.make_game(render=False)
    kills, tics = [], []
    try:
        for e in range(episodes):
            choose = lambda state, g: policy.act_index(intref.forward(m, policy.observe(state, g), ablate=ablate)[-1])
            r = policy.play_episode(game, choose, seed=seed0 + e)
            kills.append(r["kills"]), tics.append(r["tics"])
    finally:
        game.close()
    return dict(kills=float(np.mean(kills)), survival=float(np.mean(tics)), episodes=episodes)


def doom_check(m: QModel, ablate: dict, contexts: list[list[int]], workers: int = 1, out=say) -> dict:
    """Run each context through DOOM with the ablated neurons' doors closed, in verified tiles; compare every exposed
    tensor, its negation twin, the logits and the action with the equally ablated reference."""
    import tempfile
    import time
    import doomformer as df
    from doomformer.orchestrator import DoomTransformer

    done = dict(decisions=0, exact=0, values=0, changed=0)
    with tempfile.TemporaryDirectory(prefix="doomformer-lobotomy-") as tmp:
        t0 = time.perf_counter()
        with df.compile(m, tmp + "/player.wad", mode="tiles", workers=workers) as brain:
            out(f"compiled the policy into verified tile maps in {time.perf_counter() - t0:.1f}s")
            eng = DoomTransformer(m, brain.wad, workers=workers, max_live=1, ablate=ablate)
            try:
                for i, tokens in enumerate(contexts):
                    t1 = time.perf_counter()
                    logits = eng.forward(tokens)
                    check = eng.check_tokens(tokens)
                    want = intref.forward(m, tokens, ablate=ablate)[-1]
                    healthy = intref.forward(m, tokens)[-1]
                    same = bool(check["exact"] and np.array_equal(logits, want))
                    n_values = sum(v for _, v in check["tensors"].values())
                    act, well = policy.ACTIONS[policy.act_index(logits)], policy.ACTIONS[policy.act_index(healthy)]
                    done["decisions"] += 1
                    done["exact"] += same
                    done["values"] += n_values
                    done["changed"] += act != well
                    out(f"decision {i + 1:2d}/{len(contexts)}  sees {', '.join(policy.describe(tokens)).lower()}  "
                        f"healthy brain: {well}, lesioned brain in DOOM: {act}  "
                        f"{'EXACT' if same else 'MISMATCH'} ({n_values} values, tensors, negation twins, logits, "
                        f"action; {time.perf_counter() - t1:.1f}s)")
            finally:
                eng.close()
    return done


# ---------------------------------------------------------------------------- lesion statistics: --minimal, --null, --names
FIRE = policy.ACTIONS.index("F")
NAME_CORPUS = Path(__file__).resolve().parent.parent / "training" / "data" / "levels.json"
NAME_SPLIT_SEED = 12345           # training/train_brain.py: SPLIT_SEED
DEMON_DEATH = {"L0.x1": 20, "L1.x1": 29, "L1.m": 33}      # the single neurons whose lesion writes DEMON DEATH at T=150, seed 2


def site_masks(key: str, n: int, subsets: np.ndarray) -> dict:
    mk = np.zeros((len(subsets), n), dtype=bool)
    mk[np.arange(len(subsets))[:, None], subsets] = True
    return {key: mk}


def random_subsets(rng, n: int, k: int, count: int) -> np.ndarray:
    """`count` uniform random k-subsets of range(n) (drawn independently, so a repeat is possible), rows sorted."""
    return np.sort(np.argsort(rng.random((count, n)), axis=1)[:, :k], axis=1)


def enumerate_site(m, tokens, base, key, n, max_size, sample, rng, changed=True, stop=False, keep_sets=5000) -> list[dict]:
    """Per subset size k = 1..n of one activation site: pacifist count and 'changes at least one action' count.
    Sizes up to `max_size` (every size when the site has at most 16 units) run over all C(n, k) subsets; larger
    sizes run over `sample` random subsets (skipped when `sample` is 0) unless C(n, k) <= `sample`. `stop`: end at the first size with a
    pacifist."""
    rows = []
    for k in range(1, n + 1):
        total = math.comb(n, k)
        exact = n <= 16 or k <= max_size or total <= sample
        if not exact and not sample:
            break
        chunks = lesion.subset_chunks(n, k, 1 << 14) if exact else [random_subsets(rng, n, k, sample)]
        probe = random_subsets(rng, n, k, min(total, 1500))
        orders = lesion.context_orders(m, tokens, base, FIRE, policy.ACTION_IDS, site_masks(key, n, probe))
        row = dict(site=key, size=k, total=total, exact=exact, tested=0, pacifists=0, changed=0, sets=[])
        for sub in chunks:
            mk = site_masks(key, n, sub)
            if changed:
                pac, ch = lesion.classify(m, tokens, base, FIRE, policy.ACTION_IDS, mk, orders)
                row["changed"] += int(ch.sum())
            else:
                pac = lesion.survivors(m, tokens[orders[0]], mk, lambda a, rows: a == FIRE, policy.ACTION_IDS)
            row["tested"] += len(sub)
            row["pacifists"] += int(pac.sum())
            if pac.any() and len(row["sets"]) < keep_sets:
                row["sets"] += sub[pac].tolist()
        rows.append(row)
        if stop and row["pacifists"]:
            break
    return rows


def enumerate_pool(m, tokens, base, kmax, rng, out=say) -> list[dict]:
    """All k-subsets of all neurons of the model (any mix of sites), k = 1..kmax: pacifist counts and sets."""
    N = sum(lesion.sites(m).values())
    rows = []
    for k in range(1, kmax + 1):
        t0 = time.perf_counter()
        total = math.comb(N, k)
        probe = random_subsets(rng, N, k, 1500)
        orders = lesion.context_orders(m, tokens, base, FIRE, policy.ACTION_IDS, lesion.masks_from_global(m, probe))
        row = dict(size=k, total=total, pacifists=0, sets=[])
        done = 0
        for sub in lesion.subset_chunks(N, k, 1 << 15):
            pac = lesion.survivors(m, tokens[orders[0]], lesion.masks_from_global(m, sub), lambda a, rows: a == FIRE,
                                   policy.ACTION_IDS)
            row["pacifists"] += int(pac.sum())
            row["sets"] += sub[pac].tolist()
            done += len(sub)
            if done % (1 << 22) < (1 << 15):
                out(f"    pooled size {k}: {done:,} of {total:,} subsets, {row['pacifists']} pacifists, "
                    f"{time.perf_counter() - t0:.0f}s")
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
    return rows


def effect(m, tokens, healthy, masks: dict, tau: float, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Mean logit KL over the contexts for every lesion: (over the three action tokens, over the full vocabulary)."""
    S = next(iter(masks.values())).shape[0]
    act, full = np.zeros(S), np.zeros(S)
    for s0 in range(0, S, block):
        lg = lesion.forward_lesions(m, tokens, {k: mk[s0:s0 + block] for k, mk in masks.items()}, S=min(block, S - s0))
        act[s0:s0 + block] = lesion.mean_kl(healthy, lg, tau, policy.ACTION_IDS)
        full[s0:s0 + block] = lesion.mean_kl(healthy, lg, tau)
    return act, full


def pick_decisions(contexts, base, table, n: int) -> list:
    """The contexts to run in DOOM: those where the lesion changes the action first, spread evenly."""
    diff = [c for c, b, t in zip(contexts, base, table) if b != t]
    rest = [c for c in contexts if c not in diff]
    return diff[:: len(diff) // n or 1][:n] if n < len(diff) else diff + rest[: n - len(diff)]


def _ab(site: str, units) -> dict:
    return {site: sorted(int(u) for u in units)}


def _pct(a: np.ndarray, qs=(0, 5, 25, 50, 75, 95, 100)) -> str:
    return " ".join(f"p{q}={v:.3g}" for q, v in zip(qs, np.percentile(a, qs)))


def _rate(row: dict) -> str:
    if row["exact"]:
        return f"{row['pacifists'] / row['total']:.4%}"
    lo, hi = lesion.wilson(row["pacifists"], row["tested"])
    return f"{row['pacifists'] / row['tested']:.4%} (sampled, Wilson 95% {lo:.4%}..{hi:.4%})"


def run_minimal(args, out=say) -> dict:
    m = QModel.load(shipped.resolve(args.model))
    contexts = policy.all_contexts()
    tokens = np.array(contexts)
    ids = policy.ACTION_IDS
    base = lesion.action_tables(m, tokens, {}, ids, S=1)[0]
    healthy = lesion.healthy_logits(m, tokens)
    rng = np.random.default_rng(args.stat_seed)
    keys = lesion.sites(m)
    res: dict = dict(model=args.model, tau=args.tau, max_size=args.max_size, sample_size=args.sample_size,
                     healthy_actions=dict(Counter(policy.ACTIONS[a] for a in base)))
    act_logits = healthy[:, ids]
    gap = np.sort(act_logits, 1)[:, -1] - np.sort(act_logits, 1)[:, -2]
    scale = json.loads((shipped.resolve(args.model) / "train_info.json").read_text()).get("int_logit_scale") \
        if (shipped.resolve(args.model) / "train_info.json").exists() else None
    out(f"model {args.model}: {m.L} layer(s), d={m.d}, H={m.H}, dm={m.dm}; {len(contexts)} contexts; healthy actions "
        f"{res['healthy_actions']}")
    out(f"logit scale: action-token gap between the top two logits, median {np.median(gap):.0f}, max "
        f"{gap.max():.0f} integer units"
        + (f"; the training run's float scale is {scale:.5f} per unit (1/scale = {1 / scale:.0f})" if scale else "")
        + f"; KL below uses tau = {args.tau:g} integer units")
    if (base == FIRE).sum() == 0:
        raise SystemExit("the healthy policy never fires; there is no pacifist to look for")

    # (i) per site, every subset size
    out("")
    out("(i) per activation site: pacifists (no action 'F' on any of the 86 contexts) and lesions that change at "
        "least one action, by subset size")
    t0 = time.perf_counter()
    per_site = {}
    for key, n in keys.items():
        t1 = time.perf_counter()
        rows = enumerate_site(m, tokens, base, key, n, args.max_size, args.sample_size, rng)
        per_site[key] = rows
        first = next((r for r in rows if r["pacifists"]), None)
        out(f"  site {key} ({n} units, {sum(r['total'] for r in rows if r['exact']):,} subsets enumerated"
            f"{'' if n <= 16 else f' for sizes 1..{args.max_size}, {args.sample_size:,} random subsets per larger size'}"
            f", {time.perf_counter() - t1:.0f}s): smallest pacifist size "
            f"{first['size'] if first else 'none'}"
            + (" (exhaustive)" if first and first["exact"] else " (sampled)" if first else ""))
        out("    size  subsets C(n,k)  tested      pacifists  pacifist rate                changes >=1 action  rate")
        for r in rows:
            ch = r["changed"]
            lo, hi = lesion.wilson(ch, r["tested"])
            rate_ch = f"{ch / r['tested']:.4%}" + ("" if r["exact"] else f" (sampled, Wilson 95% {lo:.4%}..{hi:.4%})")
            out(f"    {r['size']:4d}  {r['total']:>14,}  {r['tested']:>10,}  {r['pacifists']:>9,}  "
                f"{_rate(r):<27s}  {ch:>18,}  {rate_ch}")
    res["per_site"] = {k: [{x: y for x, y in r.items() if x != "sets"} | dict(sets=r["sets"][:200]) for r in rows]
                       for k, rows in per_site.items()}
    out(f"  (i) done in {time.perf_counter() - t0:.0f}s")

    # (ii) pooled over all neurons
    out("")
    N = sum(keys.values())
    out(f"(ii) pooled over all {N} neurons (any mix of sites): all subsets of size 1..{args.pool_size}")
    names = lesion.global_names(m)
    pool = enumerate_pool(m, tokens, base, args.pool_size, rng, out)
    for r in pool:
        out(f"  size {r['size']}: {r['total']:,} subsets, {r['pacifists']} pacifists ({r['seconds']:.0f}s)")
    res["pooled"] = [dict(size=r["size"], total=r["total"], pacifists=r["pacifists"], seconds=r["seconds"],
                          sets=[lesion.ablate_of(names, s) for s in r["sets"][:200]]) for r in pool]
    smallest = next((r for r in pool if r["pacifists"]), None)
    if smallest is None:
        out(f"  no pacifist with at most {args.pool_size} neurons; the global minimum is above {args.pool_size}")
        minimal_sets = []
    else:
        below = smallest["size"] - 1
        out(f"  no pacifist with at most {below} neurons"
            + (f" (all {sum(r['total'] for r in pool if r['size'] <= below):,} subsets of size 1..{below} tested)"
               if below else "")
            + f"; the global minimum is {smallest['size']} neurons, {smallest['pacifists']} sets:")
        minimal_sets = [lesion.ablate_of(names, s) for s in smallest["sets"]]
        for s in minimal_sets:
            out(f"    {s}")
    res["minimal_sets"] = minimal_sets

    # (iii) the documented blast
    out("")
    doc_site, doc_units = next(iter(DOCUMENTED.items()))
    k_doc = len(doc_units)
    x0_rows = {r["size"]: r for r in per_site[doc_site]}
    pac_doc = [tuple(s) for s in x0_rows[k_doc]["sets"]]
    x0_min = next((r for r in per_site[doc_site] if r["pacifists"]), None)
    out(f"(iii) the documented {k_doc}-neuron blast {DOCUMENTED}")
    out(f"  {x0_rows[k_doc]['pacifists']} of {x0_rows[k_doc]['total']:,} {k_doc}-subsets of {doc_site} are pacifists "
        f"({x0_rows[k_doc]['pacifists'] / x0_rows[k_doc]['total']:.4%}); the documented blast is one of them: "
        f"{tuple(doc_units) in pac_doc}")
    if pac_doc:
        common = sorted(set.intersection(*(set(s) for s in pac_doc)))
        out(f"  units in every one of them: {common}")
    inside = [c for k in range(1, k_doc + 1) for c in itertools.combinations(doc_units, k)]
    pac_in, _ = lesion.classify(m, tokens, base, FIRE, ids, site_masks(doc_site, keys[doc_site], np.array(
        [c + (c[0],) * (k_doc - len(c)) for c in inside])))             # pad with a repeat: same unit set
    cores = [c for c, p in zip(inside, pac_in) if p]
    res["documented"] = dict(units=doc_units, pacifist_sets=[list(s) for s in pac_doc], pacifist_subsets=[list(c) for c in cores])
    out(f"  pacifist subsets of the documented blast (all {len(inside)} non-empty subsets tested): {cores}")
    if x0_min:
        out(f"  smallest pacifist size on {doc_site}: {x0_min['size']}, {x0_min['pacifists']} sets: "
            f"{[tuple(s) for s in x0_min['sets']]}")
        extra = k_doc - x0_min["size"]
        for c in (tuple(t) for t in x0_min["sets"]):
            sup = math.comb(keys[doc_site] - len(c), extra)
            have = sum(set(c) <= set(s) for s in pac_doc)
            out(f"  minimal set {list(c)}: {sup} {k_doc}-subsets of {doc_site} contain it, {have} of them are pacifists")
        covered = sum(any(set(c) <= set(s) for c in x0_min["sets"]) for s in pac_doc)
        out(f"  of the {len(pac_doc)} pacifist {k_doc}-sets, {covered} contain a minimal set; the documented blast "
            f"contains {'one' if any(set(c) <= set(doc_units) for c in x0_min['sets']) else 'none'}")

    # effect measure
    out("")
    out(f"effect: mean logit KL(softmax(healthy/tau) || softmax(lesioned/tau)) over the {len(contexts)} contexts, "
        f"tau = {args.tau:g}; headline = the 3 action tokens, in brackets the full vocabulary")
    res["effect"] = {}
    cases = []
    if x0_min:
        cases.append(("minimal", doc_site, x0_min["size"], [tuple(s) for s in x0_min["sets"]]))
    cases.append(("documented", doc_site, k_doc, [tuple(doc_units)]))
    for label, site, k, sets in cases:
        n = keys[site]
        total = math.comb(n, k)
        if total <= args.sample_size:
            pop = np.array(list(itertools.combinations(range(n), k)))
            kind = f"all {total:,} {k}-subsets of {site}"
        else:
            pop = random_subsets(rng, n, k, args.sample_size)
            kind = f"{args.sample_size:,} random {k}-subsets of {site} (sampled)"
        pa, pf = effect(m, tokens, healthy, site_masks(site, n, pop), args.tau)
        sa, sf = effect(m, tokens, healthy, site_masks(site, n, np.array(sets)), args.tau)
        out(f"  {label} ({k} units on {site}), reference population: {kind}")
        out(f"    population percentiles, action tokens: {_pct(pa)}; full vocabulary: {_pct(pf)}")
        ent = []
        for s, a, f in zip(sets, sa, sf):
            above = int((pa > a).sum())
            ent.append(dict(set=list(s), kl_action=float(a), kl_full=float(f), rank=above + 1, of=len(pa)))
            out(f"    {list(s)}: KL {a:.3f} [{f:.3f}], rank {above + 1} of {len(pa)} from the top "
                f"({(pa < a).mean():.1%} of the population below)")
        res["effect"][label] = dict(population=kind, pct_action={q: float(v) for q, v in zip((0, 5, 25, 50, 75, 95, 100), np.percentile(pa, (0, 5, 25, 50, 75, 95, 100)))},
                                     lesions=ent)

    # (iv) the minimal lesion in DOOM
    ablate = _ab(doc_site, x0_min["sets"][0]) if x0_min else None
    if ablate is not None and (args.decisions or args.episodes):
        out("")
        table = lesion.action_tables(m, tokens, lesion.masks_from_ablate(m, [ablate]), ids)[0]
        out(f"(iv) the minimal lesion {ablate} (first of the {x0_min['pacifists']} in lexicographic order): "
            f"{personality(list(table), list(base), contexts)}; actions {dict(Counter(policy.ACTIONS[a] for a in table))} "
            f"(healthy {dict(Counter(policy.ACTIONS[a] for a in base))})")
        if args.episodes:
            t1 = time.perf_counter()
            healthy_play = play(m, None, args.episodes, args.seed)
            les_play = play(m, ablate, args.episodes, args.seed)
            out(f"  {args.episodes} episodes (the integer reference picks the actions, seeds {args.seed}..): healthy "
                f"{healthy_play['kills']:.2f} kills, lesioned {les_play['kills']:.2f} kills "
                f"({time.perf_counter() - t1:.0f}s)")
            res["play"] = dict(healthy=healthy_play, lesioned=les_play)
        if args.decisions:
            done = doom_check(m, ablate, pick_decisions(contexts, list(base), list(table), args.decisions),
                              workers=args.workers)
            out(f"  {done['exact']}/{done['decisions']} lesioned decisions EXACT in DOOM ({done['values']:,} values, "
                f"tolerance 0); the doors changed the chosen action in {done['changed']} of them")
            res["doom"] = done
    if args.null:
        res["null"] = run_null(args, m, per_site, out)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))
        out(f"results written to {args.json}")
    return res


def run_null(args, m, per_site, out=say) -> dict:
    """Weight-shuffled copies: minimal pacifist size per site, and the random-blast null of the 7-pellet scan."""
    contexts = policy.all_contexts()
    tokens = np.array(contexts)
    ids = policy.ACTION_IDS
    keys = lesion.sites(m)
    res = {}
    out("")
    out(f"--null (a): {args.null_seeds} copies of {args.model} with the entries of every weight matrix permuted at "
        f"random (seeds 0..{args.null_seeds - 1}); embeddings, biases, r1/r2 and thresholds kept. Smallest pacifist "
        f"size per site by the same enumeration (sites with more than 16 units: exhaustive up to size {args.max_size})")
    t0 = time.perf_counter()
    skipped, table = 0, []
    for seed in range(args.null_seeds):
        s = lesion.shuffled_copy(m, seed)
        base = lesion.action_tables(s, tokens, {}, ids, S=1)[0]
        if not (base == FIRE).any():
            skipped += 1
            out(f"  seed {seed}: the healthy shuffled policy never fires; skipped")
            continue
        rng = np.random.default_rng(1000 + seed)
        mins = {}
        for key, n in keys.items():
            rows = enumerate_site(s, tokens, base, key, n, args.max_size, 0, rng, changed=False, stop=True)
            hit = next((r["size"] for r in rows if r["pacifists"]), None)
            mins[key] = hit
        table.append(dict(seed=seed, fires=int((base == FIRE).sum()), min=mins))
        out(f"  seed {seed}: fires on {(base == FIRE).sum()} of 86 contexts; smallest pacifist size per site "
            f"{ {k: (v if v else 'none') for k, v in mins.items()} } ({time.perf_counter() - t0:.0f}s)")
    out(f"  skipped copies (healthy policy never fires): {skipped} of {args.null_seeds}")
    real = {k: next((r["size"] for r in rows if r["pacifists"]), None) for k, rows in per_site.items()}
    out(f"  real model, smallest pacifist size per site: { {k: (v if v else 'none') for k, v in real.items()} }")
    r_fires = int((lesion.action_tables(m, tokens, {}, ids, S=1)[0] == FIRE).sum())
    near = [t for t in table if r_fires / 2 <= t["fires"] <= 2 * r_fires]
    out(f"  the real model fires on {r_fires} of 86 contexts; copies that fire on {r_fires / 2:g} to {2 * r_fires} contexts "
        f"(0.5x to 2x): {len(near)} of {len(table)}, seeds {[t['seed'] for t in near]}")
    for key in keys:
        for label, group in (("all copies", table), ("copies near the real fire rate", near)):
            found = sorted(t["min"][key] for t in group if t["min"][key])
            line = f"  {key}, {label}: {len(found)} of {len(group)} have a pacifist"
            if found:
                line += f"; smallest size min {found[0]}, median {np.median(found):g}, max {found[-1]}"
                if real[key]:
                    line += f"; {sum(v <= real[key] for v in found)} at or below the real model's {real[key]}"
            if keys[key] > 16:
                line += f" (sizes above {args.max_size} not searched)"
            out(line)
    res["shuffled"] = dict(skipped=skipped, copies=table, real=real)

    out("")
    out("--null (b): the original blast scan; random 7-pellet blasts (a site uniformly, then 7 of its units), "
        "replayed with the scan's own random stream")
    keys_n = ablation_keys(m)
    names = list(keys_n)
    rng = np.random.default_rng(0)
    shots = []
    for _ in range(args.blasts):
        k = names[rng.integers(len(names))]
        shots.append((k, sorted(rng.choice(keys_n[k], args.pellets, replace=False).tolist())))
    base = lesion.action_tables(m, tokens, {}, ids, S=1)[0]
    rate = {k: next((r for r in rows if r["size"] == args.pellets), None) for k, rows in per_site.items()}
    by_site, total_pac, total_changed, exp_total = {}, 0, 0, 0.0
    for key in names:
        sel = np.array([u for k, u in shots if k == key], dtype=int).reshape(-1, args.pellets)
        if len(sel):
            pac, ch = lesion.classify(m, tokens, base, FIRE, ids, site_masks(key, keys_n[key], sel))
        else:
            pac = ch = np.zeros(0, dtype=bool)
        r = rate[key]
        exp = len(sel) * r["pacifists"] / r["tested"] if r else float("nan")
        exp_total += exp
        by_site[key] = dict(blasts=len(sel), changed=int(ch.sum()), pacifists=int(pac.sum()), expected_pacifists=exp,
                            rate=(r["pacifists"] / r["tested"]) if r else None, rate_exact=bool(r and r["exact"]))
        total_pac += int(pac.sum())
        total_changed += int(ch.sum())
        out(f"  {key:6s}: {len(sel):4d} blasts, {int(ch.sum()):4d} change an action, {int(pac.sum())} pacifists; "
            f"per-blast pacifist rate {by_site[key]['rate']:.4%} ({'exact' if by_site[key]['rate_exact'] else 'sampled'}) "
            f"=> expected {exp:.2f}")
    out(f"  all sites: {args.blasts} blasts, {total_changed} change an action, {total_pac} pacifists; expected "
        f"{exp_total:.2f} pacifists from the per-site rates")
    pool = random_subsets(np.random.default_rng(args.stat_seed + 1), sum(keys.values()), args.pellets, args.blasts)
    pac_pool = lesion.pacifist(lesion.action_tables(m, tokens, lesion.masks_from_global(m, pool), ids), FIRE)
    out(f"  {args.blasts} uniform {args.pellets}-subsets of all {sum(keys.values())} neurons (any mix of sites): "
        f"{int(pac_pool.sum())} pacifists")
    res["blasts"] = dict(by_site=by_site, pacifists=total_pac, changed=total_changed, expected=exp_total,
                         pooled_pacifists=int(pac_pool.sum()))
    return res


def name_split(path: Path = NAME_CORPUS, seed: int = NAME_SPLIT_SEED) -> dict:
    """The fixed train/dev/val split of training/train_brain.py:load_corpus, without importing torch."""
    import random
    rows = json.loads(Path(path).read_text())
    names = sorted({r["clean"] for r in rows if r.get("clean")})
    order = names[:]
    random.Random(seed).shuffle(order)
    nval = round(0.1 * len(order))
    val, rest = order[:nval], order[nval:]
    dev = min(300, len(rest) // 8)
    return dict(names=names, val=val, dev=rest[:dev], train=rest[dev:])


def run_names(args, out=say) -> dict:
    m = QModel.load(shipped.resolve(args.names_model))
    split = name_split()
    use = [n for n in split["val"] if len(n) + 1 <= m.T and all(c in m.vocab for c in n)]
    if args.names_limit:
        use = use[: args.names_limit]
    groups = {}
    for n in use:
        groups.setdefault(len(n) + 1, []).append([0] + [m.vocab.index(c) for c in n])
    groups = [np.array(g) for _, g in sorted(groups.items())]
    npos = sum(g.size for g in groups)
    keys = lesion.sites(m)
    names = lesion.global_names(m)
    N = len(names)
    out(f"name model {args.names_model}: {m.L} layers, d={m.d}, H={m.H}, dm={m.dm}, T={m.T}; {N} single neurons "
        f"({' + '.join(f'{n} {k}' for k, n in keys.items())})")
    out(f"teacher forcing: {len(use)} of the {len(split['val'])} validation names (fixed split of training/train_brain.py, "
        f"seed {NAME_SPLIT_SEED}), {npos:,} positions; tau = {args.tau:g}")
    t0 = time.perf_counter()
    masks = lesion.masks_from_global(m, np.arange(N)[:, None])
    kl_full, kl_noeos = lesion.mean_kl_batches(m, groups, masks, args.tau, drop=(0,))
    out(f"  KL scan of {N} single neurons over {npos:,} positions ({time.perf_counter() - t0:.0f}s)")

    runs = [("T=150 seed 2", 150.0, 2), ("T=150 seed 3", 150.0, 3), ("T=150 seed 7", 150.0, 7), ("greedy", 0.0, 0)]
    gen = lambda ab, temp, seed: intref.generate(m, "", temperature=temp, seed=seed, ablate=ab)
    t1 = time.perf_counter()
    healthy_txt = {lab: gen(None, temp, seed) for lab, temp, seed in runs}
    texts = {lab: [] for lab, _, _ in runs}
    for i, (k, u) in enumerate(names):
        for lab, temp, seed in runs:
            texts[lab].append(gen({k: [u]}, temp, seed))
    out(f"  {N} x {len(runs)} generations ({time.perf_counter() - t1:.0f}s)")
    out("healthy text: " + ", ".join(f"{lab}: {healthy_txt[lab]!r}" for lab, _, _ in runs))
    changed = {}
    for lab, _, _ in runs:
        ch = np.array([t != healthy_txt[lab] for t in texts[lab]])
        changed[lab] = ch
        out(f"  {lab}: {int(ch.sum())} of {N} single neurons change the text; {len(set(texts[lab]))} distinct texts")
    anych = np.any([changed[lab] for lab, _, _ in runs], axis=0)
    allch = np.all([changed[lab] for lab, _, _ in runs], axis=0)
    out(f"  changes the text in at least one run: {int(anych.sum())}; in all four: {int(allch.sum())}; in none: {int((~anych).sum())}")
    ids_dd = {f"{k}:{u}": names.index((k, u)) for k, u in DEMON_DEATH.items()}
    deaths = [i for i, t in enumerate(texts["T=150 seed 2"]) if t == "DEMON DEATH"]
    out(f"  neurons whose lesion writes 'DEMON DEATH' at T=150 seed 2: {[f'{names[i][0]}:{names[i][1]}' for i in deaths]}")
    out(f"  most common lesioned texts at T=150 seed 2: {Counter(texts['T=150 seed 2']).most_common(5)}")
    pcts = (0, 5, 25, 50, 75, 95, 100)
    out(f"single-neuron teacher-forced KL(healthy || lesioned), full vocabulary, tau {args.tau:g}: {_pct(kl_full, pcts)}")
    out(f"  without the end token: {_pct(kl_noeos, pcts)}")
    out(f"  neurons with KL exactly 0 on every position: {int((kl_full == 0).sum())}; of them change the text in some run: "
        f"{int(((kl_full == 0) & anych).sum())}")
    order_full = np.argsort(-kl_full, kind="stable")
    order_noeos = np.argsort(-kl_noeos, kind="stable")
    rank = lambda order, i: int(np.where(order == i)[0][0]) + 1
    rows = []
    for lab, i in ids_dd.items():
        rows.append(dict(neuron=lab, kl_full=float(kl_full[i]), kl_noeos=float(kl_noeos[i]), rank_full=rank(order_full, i),
                         rank_noeos=rank(order_noeos, i), text_seed2=texts["T=150 seed 2"][i]))
        out(f"  {lab}: KL {kl_full[i]:.4f} (rank {rank(order_full, i)} of {N}), without end token {kl_noeos[i]:.4f} "
            f"(rank {rank(order_noeos, i)}); text at seed 2 {texts['T=150 seed 2'][i]!r}")
    out(f"  largest KL: {[(f'{names[i][0]}:{names[i][1]}', round(float(kl_full[i]), 3)) for i in order_full[:5]]}")
    out(f"  mean KL of the neurons that change the seed-2 text {kl_full[changed['T=150 seed 2']].mean():.4f}, of those "
        f"that do not {kl_full[~changed['T=150 seed 2']].mean() if (~changed['T=150 seed 2']).any() else float('nan'):.4f}")
    res = dict(model=args.names_model, tau=args.tau, contexts=len(use), positions=npos, healthy=healthy_txt,
               changed={lab: int(c.sum()) for lab, c in changed.items()}, anychange=int(anych.sum()),
               kl_percentiles={q: float(v) for q, v in zip(pcts, np.percentile(kl_full, pcts))},
               demon_death=rows, death_neurons=[f"{names[i][0]}:{names[i][1]}" for i in deaths],
               neurons=[dict(neuron=f"{k}:{u}", kl_full=float(a), kl_noeos=float(b),
                             **{lab: texts[lab][i] for lab, _, _ in runs})
                        for i, ((k, u), a, b) in enumerate(zip(names, kl_full, kl_noeos))])
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))
        out(f"results written to {args.json}")
    return res


def main(argv=None) -> int:
    import argparse
    import time

    from doomformer import shipped

    ap = argparse.ArgumentParser(
        prog="python -m demos.lobotomy",
        description="Shoot a few neurons of the marine's DOOM brain. Scans every single neuron and N random "
                    "shotgun blasts on all 86 observations (integer reference), plays the README's pacifist "
                    "lesion against the healthy brain, and runs some of its decisions on voodoo dolls with the "
                    "doors closed, compared exactly with the lesioned reference.")
    ap.add_argument("--model", default="doomplayer", help="shipped model name or directory (the README's blast is for doomplayer)")
    ap.add_argument("--blasts", type=int, default=3000, help="random blasts to scan")
    ap.add_argument("--pellets", type=int, default=7, help="neurons one blast closes (one activation)")
    ap.add_argument("--seed", type=int, default=9000, help="first episode seed of the kill counts")
    ap.add_argument("--episodes", type=int, default=20, help="episodes for the kill counts (healthy and lesioned)")
    ap.add_argument("--decisions", type=int, default=3,
                    help="observations to run on voodoo dolls with the doors closed, those where the lesion changes "
                         "the action first (0: skip; up to 86)")
    ap.add_argument("--workers", type=int, default=1, help="DOOM games for the tile run")
    ap.add_argument("--minimal", action="store_true",
                    help="lesion statistics instead of the scan: every subset size per site, the pooled search over all "
                         "neurons, where the documented blast sits, the minimal pacifist in DOOM (see docs/DEMOS.md)")
    ap.add_argument("--null", action="store_true",
                    help="with --minimal: weight-shuffled copies of the model and the random-blast null of the scan")
    ap.add_argument("--names", action="store_true",
                    help="single-neuron lesions of the name model: teacher-forced logit KL and the changed sampled text")
    ap.add_argument("--names-model", default="doombrain", help="--names: the name model (default doombrain)")
    ap.add_argument("--names-limit", type=int, default=0, help="--names: use the first N validation names (0: all)")
    ap.add_argument("--max-size", type=int, default=6,
                    help="--minimal: largest subset size enumerated exhaustively on a site of more than 16 units (default 6)")
    ap.add_argument("--sample-size", type=int, default=20000,
                    help="--minimal: random subsets per larger size, and the size of the KL reference population "
                         "when a size has more subsets than this (default 20000)")
    ap.add_argument("--pool-size", type=int, default=5,
                    help="--minimal: pooled search over all neurons up to this subset size (default 5; 4 takes seconds)")
    ap.add_argument("--null-seeds", type=int, default=20, help="--null: weight-shuffled copies (default 20)")
    ap.add_argument("--tau", type=float, default=150.0,
                    help="temperature of the logit KL in integer logit units (default 150, the name model's sampling temperature)")
    ap.add_argument("--stat-seed", type=int, default=0, help="seed of the random subsets of --minimal / --null")
    ap.add_argument("--json", default="", help="write the numbers of --minimal / --names to this file")
    args = ap.parse_args(argv)
    if args.null and not args.minimal:
        ap.error("--null needs --minimal")
    if args.names:
        run_names(args)
        return 0
    if args.minimal:
        run_minimal(args)
        return 0
    m = QModel.load(shipped.resolve(args.model))
    contexts = policy.all_contexts()
    base = action_table(m, contexts)
    say(f"scanning every single neuron and {args.blasts:,} random {args.pellets}-pellet blasts on all {len(contexts)} "
        "observations, about 30 s ...")
    t0 = time.perf_counter()
    found = scan(m, contexts, base, args.blasts, args.pellets)
    say(f"scan of {len(contexts)} observations ({time.perf_counter() - t0:.0f}s): {found['single_changed']} of "
          f"{found['singles']} single neurons and {found['blast_changed']} of {found['blasts']} {args.pellets}-pellet "
          f"blasts change at least one action; {len(found['pacifists'])} of the blasts make a pacifist")
    say(f"personalities of the blasts: {dict(found['personalities'])}")
    ablate = DOCUMENTED if args.model == "doomplayer" and args.pellets == 7 else (found["pacifists"] or [None])[0]
    if ablate is None:
        say("no pacifist among these blasts; nothing to play")
        return 1
    table = action_table(m, contexts, ablate)
    who = personality(table, base, contexts)
    say(f"the lesion: {ablate} -> {who}; actions on the 86 observations {dict(Counter(policy.ACTIONS[a] for a in table))} "
          f"(healthy {dict(Counter(policy.ACTIONS[a] for a in base))})")
    t1 = time.perf_counter()
    healthy = play(m, None, args.episodes, args.seed)
    lesioned = play(m, ablate, args.episodes, args.seed)
    say(f"{args.episodes} episodes (the integer reference picks the actions, seeds {args.seed}..): healthy "
          f"{healthy['kills']:.2f} kills, lesioned {lesioned['kills']:.2f} kills ({time.perf_counter() - t1:.0f}s)")
    ok = True
    if args.decisions:
        diff = [c for c, b, t in zip(contexts, base, table) if b != t]
        rest = [c for c in contexts if c not in diff]
        picks = (diff[:: len(diff) // args.decisions or 1][: args.decisions] if args.decisions < len(diff)
                 else diff + rest[: args.decisions - len(diff)])
        done = doom_check(m, ablate, picks, workers=args.workers)
        ok = done["exact"] == done["decisions"]
        say(f"{done['exact']}/{done['decisions']} lesioned decisions EXACT in DOOM ({done['values']:,} values, "
              f"tolerance 0); the doors changed the chosen action in {done['changed']} of them")
        say("Python's part: closed the doors of the lesioned neurons in every tile that consumes them, copied "
              "tile outputs to the next tile, and compared. The dolls did every product, sum and activation.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
