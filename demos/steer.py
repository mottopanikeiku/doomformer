"""Rank-1 steering of the marine's policy, checked on voodoo dolls.

    python -m demos.steer                         # reference tables only, seconds
    python -m demos.steer --doom 4 --workers 3    # also run steered contexts in DOOM (tolerance 0)

`doomplayer` sees [BOS, bearing, distance] and picks turn left, turn right or fire. The probe is the difference of the mean
integer activations of the contexts whose bearing is 'i' (in the sights) and 'm' (12 to 20 degrees left), 5 distances each,
at one site and position (`doomformer.steer.probe`). Steering adds alpha times that integer direction to the same site and
position of a context (`doomformer.intervene.add_direction`). Measured on the 86 contexts of `policy.all_contexts()`:
which contexts change their action and where they end; the same with random directions of the same size as a control; the
probe fitted on distances 0 to 2 and steered on distances 3 and 4 as a held-out check; and `project_out` of the direction
from the 'i' contexts. With --doom the chosen steered contexts also run in tile maps: the dolls compute, Python patches the
bus, and every exposed tensor, patched cell and the logits are compared with the steered reference at tolerance 0.
"""
from __future__ import annotations

import argparse

import numpy as np

from doomformer import intervene as iv
from doomformer import policy, shipped, steer
from doomformer.intref import QModel, forward

SITES = (("x0", 1), ("L0.ao", 2), ("L0.x1", 2))
NAMES = "LRF"
MIN_DISTINCT = 10          # a control with fewer distinct directions than this is not a null


def acts(m, contexts, ivs_of=None):
    return np.array([policy.act_index(forward(m, c, interventions=ivs_of(c) if ivs_of else None)[-1]) for c in contexts])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m demos.steer", description=__doc__.split("\n\n")[0])
    ap.add_argument("model", nargs="?", default="doomplayer")
    ap.add_argument("--alpha", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--random", type=int, default=200, help="random directions per control")
    ap.add_argument("--doom", type=int, default=0, help="steered contexts per site to run in DOOM (the 'm' contexts first)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    m = QModel.load(shipped.resolve(args.model))
    C = policy.all_contexts()
    bearing = [policy.VOCAB[c[1]] for c in C]
    dist = [policy.VOCAB[c[2]] for c in C]
    base = acts(m, C)
    mi = [i for i, b in enumerate(bearing) if b == "m"]
    ii = [i for i, b in enumerate(bearing) if b == "i"]
    movable = [i for i in mi if base[i] != 2]
    print(f"{args.model}: {len(C)} contexts; healthy actions L/R/F = {np.bincount(base, minlength=3).tolist()}; "
          f"'m' contexts {[NAMES[base[i]] for i in mi]} by distance 0..4 ({len(movable)} do not fire)")
    fit = mi + ii
    rng = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)        # the controls added later: the draws above keep their numbers
    runs_wanted = []
    for site, pos in SITES:
        pr = steer.probe(m, site, [C[i] for i in fit], [bearing[i] == "i" for i in fit], pos=pos)
        held = steer.probe(m, site, [C[i] for i in fit if dist[i] in "012"],
                           [bearing[i] == "i" for i in fit if dist[i] in "012"], pos=pos)
        print(f"\nsite {site} position {pos}: direction {pr.direction.tolist()} (largest entry {pr.scale}), "
              f"fit accuracy {pr.accuracy:.2f} on {len(fit)} contexts")
        print(f"{'alpha':>5s} {'m contexts that fire':>21s} {'changed (of 86)':>16s} {'end in F':>9s} "
              f"{'random: m fire':>15s} {'random: changed':>16s} {'held-out m (dist 3,4)':>22s}")
        for alpha in args.alpha:
            new = acts(m, C, lambda c: [pr.add(alpha, c)])
            fire_m = sum(new[i] == 2 for i in movable)
            ch = int((new != base).sum())
            to_f = int(((new == 2) & (base != 2)).sum())
            rf, rc = [], []
            for _ in range(args.random):
                u = np.zeros_like(pr.direction)
                while not u.any():
                    u = rng.integers(-pr.scale, pr.scale + 1, pr.direction.shape)
                u[np.argmax(np.abs(u))] = np.sign(u[np.argmax(np.abs(u))]) * pr.scale
                a2 = acts(m, C, lambda c: [iv.add_direction(site, u, alpha, pos=[pos])])
                rf.append(sum(a2[i] == 2 for i in movable))
                rc.append(int((a2 != base).sum()))
            hn = acts(m, C, lambda c: [held.add(alpha, c)])
            hm = [i for i in movable if dist[i] in "34"]
            print(f"{alpha:5d} {fire_m:>14d} of {len(movable)} {ch:16d} {to_f:9d} {np.mean(rf):10.2f} of {len(movable)} "
                  f"{np.mean(rc):16.1f} {sum(hn[i] == 2 for i in hm):>17d} of {len(hm)}")
        alpha = 2 if 2 in args.alpha else args.alpha[0]
        real = acts(m, C, lambda c: [pr.add(alpha, c)])
        real_fire = int(sum(real[i] == 2 for i in movable))
        print(f"  controls at alpha {alpha} ({args.random} draws each, same site, position and alpha; 'm' fire = of the "
              f"{len(movable)} 'm' contexts that do not fire; P(>= real) = share of draws that fire at least as many):")
        print(f"  {'direction':52s} {'m fire':>7s} {'P(>= real)':>11s} {'changed':>8s} {'end in F':>9s} {'distinct':>9s}")

        def row(name, fires, changed, to_f, distinct=None):
            fires = np.asarray(fires)
            note = ""
            if distinct is not None and distinct < MIN_DISTINCT:
                note = f"  not a null: only {distinct} distinct direction(s) (the draws are the real direction, its negation or smaller)"
            print(f"  {name:52s} {fires.mean():7.2f} {(fires >= real_fire).mean():11.3f} {np.mean(changed):8.1f} "
                  f"{np.mean(to_f):9.1f} {'' if distinct is None else distinct:>9}{note}")

        def draw(make):
            fires, changed, to_f, seen = [], [], [], set()
            for _ in range(args.random):
                u = make()
                seen.add(tuple(int(v) for v in u))
                a2 = acts(m, C, lambda c: [iv.add_direction(site, u, alpha, pos=[pos])])
                fires.append(sum(a2[i] == 2 for i in movable))
                changed.append(int((a2 != base).sum()))
                to_f.append(int(((a2 == 2) & (base != 2)).sum()))
            return fires, changed, to_f, len(seen)

        row("real probe ('i' vs 'm' contexts)", [real_fire], [int((real != base).sum())],
            [int(((real == 2) & (base != 2)).sum())])

        def random_dir():
            u = np.zeros_like(pr.direction)
            while not u.any():
                u = rng.integers(-pr.scale, pr.scale + 1, pr.direction.shape)
            u[np.argmax(np.abs(u))] = np.sign(u[np.argmax(np.abs(u))]) * pr.scale
            return u

        row("random direction, largest entry 4", *draw(random_dir))
        row("same entries as the probe, permuted, random signs", *draw(
            lambda: rng.permutation(pr.direction) * rng.choice([-1, 1], len(pr.direction))))

        centred = steer._activations(m, C, site, pos)
        centred = centred - centred.mean(0)

        def span_dir():                     # a random combination of the 86 contexts' activations, rounded to the probe's L2 norm
            while True:
                u = rng2.normal(size=len(C)) @ centred
                if np.abs(u).max() > 0:
                    u = np.rint(u * np.linalg.norm(pr.direction) / np.linalg.norm(u)).astype(np.int64)
                    if u.any():
                        return u

        row("random in the span of the data, probe's L2 norm", *draw(span_dir))

        def other_contexts():               # the same recipe (difference of class means) on 5 against 5 random contexts
            pick = [int(i) for i in rng2.permutation(len(C))[:10]]
            labels = [k < 5 for k in range(10)]
            try:
                return steer.probe(m, site, [C[i] for i in pick], labels, pos=pos).direction
            except ValueError:
                return other_contexts()

        row("probe on 5 against 5 random contexts", *draw(other_contexts))

        def shuffled():
            while True:
                labels = list(rng.permutation([bool(bearing[i] == "i") for i in fit]))
                try:
                    return steer.probe(m, site, [C[i] for i in fit], labels, pos=pos).direction
                except ValueError:
                    continue

        row("shuffled-label probe ('i' vs 'm' labels)", *draw(shuffled))
        fire_probe = steer.probe(m, site, C, [bool(a == 2) for a in base], pos=pos)
        a2 = acts(m, C, lambda c: [fire_probe.add(alpha, c)])
        row("'fire vs not fire' probe (all 86 contexts)", [int(sum(a2[i] == 2 for i in movable))],
            [int((a2 != base).sum())], [int(((a2 == 2) & (base != 2)).sum())])
        new = acts(m, C, lambda c: [pr.add(alpha, c)])
        print(f"  alpha {alpha}: 'm' contexts now {[NAMES[new[i]] for i in mi]}; the action counts L/R/F go from "
              f"{np.bincount(base, minlength=3).tolist()} to {np.bincount(new, minlength=3).tolist()}")
        neg = acts(m, C, lambda c: [pr.add(-alpha, c)])
        print(f"  alpha {-alpha}: 'i' contexts {[NAMES[base[i]] for i in ii]} -> {[NAMES[neg[i]] for i in ii]}")
        proj = acts(m, C, lambda c: [pr.project(c)])
        print(f"  project_out: 'i' contexts {[NAMES[base[i]] for i in ii]} -> {[NAMES[proj[i]] for i in ii]}; "
              f"{int((proj != base).sum())} of 86 change")
        order = movable + [i for i in range(len(C)) if i not in movable]
        runs_wanted.append((site, pos, pr, alpha, order[:args.doom] if args.doom else []))
    if not args.doom:
        return 0
    from doomformer.patching import DoomRuns
    total = exact = values = cells = 0
    with DoomRuns(m, workers=args.workers) as runs:
        for site, pos, pr, alpha, chosen in runs_wanted:
            for i in chosen:
                plan = [pr.add(alpha, C[i])]
                rep = runs.run(C[i], plan)
                total += 1
                exact += bool(rep["exact"])
                values += rep["values"]
                cells += rep["patches"][1]
                want = policy.act_index(forward(m, C[i], interventions=plan)[-1])
                print(f"DOOM {site}@{pos} alpha {alpha} context {policy.VOCAB[C[i][1]]}{policy.VOCAB[C[i][2]]}: healthy "
                      f"{NAMES[base[i]]} steered {NAMES[want]} (DOOM logits give {NAMES[policy.act_index(rep['logits'])]}) "
                      f"{'EXACT' if rep['exact'] else 'MISMATCH'} ({rep['values']} values, {rep['patches'][1]} patched cells, "
                      f"{rep['seconds']:.0f}s)")
            i = ii[0]
            plan = [pr.project(C[i])]
            rep = runs.run(C[i], plan)
            total += 1
            exact += bool(rep["exact"])
            values += rep["values"]
            cells += rep["patches"][1]
            print(f"DOOM {site}@{pos} project_out context {policy.VOCAB[C[i][1]]}{policy.VOCAB[C[i][2]]}: "
                  f"{'EXACT' if rep['exact'] else 'MISMATCH'} ({rep['values']} values, {rep['patches'][1]} patched cells)")
    print(f"\n{exact}/{total} steered runs EXACT in DOOM: {cells} patched cells, {values} values at tolerance 0")
    return 0 if exact == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
