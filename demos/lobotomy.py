"""Lobotomy: shoot a few neurons of the marine's brain and see what marine is left.

The policy reads one of 86 observations (17 bearings x 5 distances, plus "no enemy"), so every single-unit ablation and
every random shotgun blast can be scanned exactly on all of them with the integer reference (`intref`). A blast is
`pellets` neurons of one activation; closing a neuron's doors in every tile that consumes it is what
`DoomTransformer(ablate=...)` does in DOOM. The picked lesion is then played for real (the reference chooses, ViZDoom
simulates `defend_the_center`) and a few of its decisions run on voodoo dolls with the doors closed and are compared,
tensor by tensor, with the equally lesioned reference.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from doomformer import intref, policy
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
    args = ap.parse_args(argv)
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
