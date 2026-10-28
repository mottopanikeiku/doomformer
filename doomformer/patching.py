"""Activation patching with recovery, run on the integer reference and, row by row, on voodoo dolls.

    import doomformer as df
    table = df.patch_recovery(model, clean, corrupt, sites, doom=True)
    print(table.text())

Denoising (the default): run the corrupt tokens and write the clean run's activation of one site (a position and units
or heads) into the run. The metric is the logit difference logit[a] - logit[b] at the last position (answer = (a, b);
default: the clean run's argmax and the corrupt run's argmax). The recovery of a row is
    (metric(patched) - metric(corrupt)) / (metric(clean) - metric(corrupt)),
1 when the patch alone restores the clean answer, 0 when it changes nothing. direction="noising" runs the clean tokens
and writes the corrupt activations: recovery = (metric(clean) - metric(patched)) / (metric(clean) - metric(corrupt)),
the damage of the cell. Rows are independent single-cell patches (doomformer.intervene `resample`), not path patches.

Rows whose patch is the whole residual stream carry a `flag` and stay out of the headline rows (`PatchTable.headline`):
    "tautological"    every unit of the final layer's x1 or x2 at the last position (the metric's read position): the
                      logits are a function of that vector alone, so the patch restores the donor's logits and the
                      recovery (denoising) or damage (noising) is 1 by construction
    "whole residual"  every unit of x0, x1 or x2 of any other layer or position: the patch writes everything that
                      position holds at that point, so it bounds what any smaller patch there can recover; it says
                      where the information is, not which units carry it

With doom=True (or row indices, or N for the N rows with the largest |recovery|) each selected row also runs in tile
maps with DOOM doing all arithmetic and the host patching the bus (see doomformer.intervene); the row is `exact`
when every exposed tensor, negation twin, patched cell and the logits equal the reference at tolerance 0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import intervene as iv
from .intref import QModel, forward


def tokens_of(m: QModel, x) -> list[int]:
    """Token ids of `x`: a list of ids, or a string of vocabulary characters (BOS is prepended when missing)."""
    if isinstance(x, str):
        ids = [m.vocab.index(c) for c in x]
        return ids if ids and ids[0] == 0 else [0] + ids
    return [int(t) for t in x]


def row_flag(m: QModel, key: str, pos, units, n: int) -> str:
    """"tautological", "whole residual" or "" for a patch of `units` of site `key` at positions `pos` in a run of n
    tokens (see the module docstring). Only x0, x1 and x2 are residual-stream sites."""
    if key.split(".")[-1] not in ("x0", "x1", "x2") or sorted(units) != list(range(m.d)):
        return ""
    if key in (f"L{m.L - 1}.x1", f"L{m.L - 1}.x2") and (n - 1) in pos:
        return "tautological"
    return "whole residual"


def grid(m: QModel, kinds=None, positions=None, per_head: bool = False, per_unit: bool = False) -> list[dict]:
    """Single-cell patch specs, one per (site, position[, head or unit]). kinds: site names without layer
    ("x0", "q", "k", "v", "ao", "x1", "m", "x2"; default all) applied to every layer."""
    kinds = list(kinds) if kinds is not None else ["x0", "q", "k", "v", "ao", "x1", "m", "x2"]
    positions = list(range(m.T)) if positions is None else list(positions)
    out = []
    for key in iv.all_sites(m):
        kind = key.split(".")[-1]
        if kind not in kinds:
            continue
        for p in positions:
            if per_unit:
                out += [dict(site=key, pos=(p,), units=(u,)) for u in range(iv.width(m, key))]
            elif per_head and kind in iv.HEAD_SITES:
                out += [dict(site=key, pos=(p,), heads=(h,)) for h in range(m.H)]
            else:
                out.append(dict(site=key, pos=(p,)))
    return out


def _spec(m: QModel, s) -> dict:
    """A site spec: a site name (every position, one row), (site, pos), or a dict site/pos/units/heads."""
    if isinstance(s, str):
        return dict(site=s)
    if isinstance(s, (tuple, list)):
        return dict(site=s[0], pos=s[1])
    return dict(s)


def _label(spec: dict) -> str:
    part = ""
    if spec.get("heads") is not None:
        part = "head " + ",".join(map(str, spec["heads"]))
    elif spec.get("units") is not None:
        part = "unit " + ",".join(map(str, spec["units"]))
    return part or "all units"


@dataclass
class PatchTable:
    rows: list[dict]
    clean: list[int]
    corrupt: list[int]
    answer: tuple[int, int]
    direction: str
    metric_clean: int
    metric_corrupt: int
    vocab: str = ""
    seconds: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def headline(self) -> list[dict]:
        """The rows without a flag: tautological and whole-residual rows are not findings and stay out of counts and
        headline numbers."""
        return [r for r in self.rows if not r["flag"]]

    @property
    def doom_rows(self) -> list[dict]:
        return [r for r in self.rows if r["doom"] is not None]

    @property
    def exact(self) -> bool:
        """Every row run in DOOM was exact (False when none ran)."""
        run = self.doom_rows
        return bool(run) and all(r["doom"] for r in run)

    def text(self) -> str:
        a, b = (self.vocab[i] if self.vocab else str(i) for i in self.answer)
        head = (f"patch recovery ({self.direction}): clean {self.clean} corrupt {self.corrupt}, metric logit[{a!r}] - "
                f"logit[{b!r}] = {self.metric_clean} (clean) / {self.metric_corrupt} (corrupt)")
        lines = [head, f"{'site':9s} {'pos':>3s} {'cells':<14s} {'changed':>7s} {'metric':>7s} {'recovery':>9s}  doom"]
        for r in self.rows:
            doom = "-" if r["doom"] is None else (f"EXACT ({r['values']} values, {r['seconds']:.0f}s)" if r["doom"]
                                                  else "MISMATCH")
            pos = ",".join(map(str, r["pos"])) if r["pos"] is not None else "all"
            lines.append(f"{r['site']:9s} {pos:>3s} {r['cells']:<14s} {r['changed']:7d} {r['metric']:7d} "
                         f"{r['recovery']:9.3f}  {doom}" + (f"  [{r['flag']}]" if r["flag"] else ""))
        flagged = len(self.rows) - len(self.headline)
        if flagged:
            lines.append(f"{flagged} row(s) patch the whole residual stream and are flagged: tautological rows recover 1 "
                         "by construction; neither kind is a finding; headline rows exclude them")
        if self.doom_rows:
            ok = sum(1 for r in self.doom_rows if r["doom"])
            lines.append(f"{ok}/{len(self.doom_rows)} rows exact in DOOM (tolerance 0)")
        return "\n".join(lines)


class DoomRuns:
    """Tile maps of a model and an engine that runs intervened contexts on them, with the exactness check.

    tiles: a tile directory or a compiled tile Brain (df.compile(..., mode="tiles")); None compiles (or reuses the
    cached) tile maps of the model. Use as a context manager."""

    def __init__(self, m: QModel, tiles=None, workers: int = 2, max_live: int = 1):
        from .orchestrator import DoomTransformer
        from .tiled import compile_tiles
        self._brain = None
        if tiles is None:
            tiles = self._brain = compile_tiles(m, workers=workers)
        directory = getattr(tiles, "wad", tiles)
        self.engine = DoomTransformer(m, Path(directory), workers=workers, max_live=max_live)
        self.model = m

    def run(self, tokens: list[int], interventions=None, check_against=None) -> dict:
        """Run `tokens` with `interventions` (a list or Plan) from the longest held prefix; returns logits, the check
        (exact, tensors, patches, values) and seconds. check_against: another intervention list for the reference
        (the tamper test)."""
        eng = self.engine
        t0 = time.perf_counter()
        eng.use(interventions)
        logits = eng.forward(tokens, reuse=True)
        rep = eng.check_tokens(tokens, interventions=check_against)
        want = forward(self.model, tokens, interventions=check_against if check_against is not None
                       else interventions)[-1]
        rep["exact"] = bool(rep["exact"] and np.array_equal(logits, want))
        rep["values"] = sum(v for _, v in rep["tensors"].values()) + rep["patches"][1]
        rep.update(logits=logits, seconds=time.perf_counter() - t0, patched_cells=len(eng.patches))
        return rep

    def close(self):
        self.engine.close()
        if self._brain is not None:
            self._brain.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def patch_recovery(m: QModel, clean, corrupt, sites=None, *, answer=None, direction: str = "denoising", doom=False,
                   workers: int = 2, tiles=None, runs: DoomRuns | None = None, out=None) -> PatchTable:
    """Logit-difference recovery per site (see the module docstring).

    sites: specs from `grid(...)` or hand-written ((site, pos), a dict with site/pos/units/heads, a site name);
    default grid(m). doom: False, True (every row), an iterable of row indices, or an int N (the N rows with the
    largest |recovery|). runs: an open DoomRuns to reuse (else one is created for the call). out: callable that
    receives one line per DOOM row as it finishes."""
    if direction not in ("denoising", "noising"):
        raise ValueError("direction must be 'denoising' or 'noising'")
    clean, corrupt = tokens_of(m, clean), tokens_of(m, corrupt)
    if len(clean) != len(corrupt):
        raise ValueError("clean and corrupt must have the same length (positions are aligned)")
    lc, lk = forward(m, clean)[-1], forward(m, corrupt)[-1]
    if answer is None:
        answer = (int(np.argmax(lc)), int(np.argmax(lk)))
    answer = tuple(m.vocab.index(a) if isinstance(a, str) else int(a) for a in answer)
    if answer[0] == answer[1]:
        raise ValueError("answer needs two different tokens (the clean and the corrupt run have the same argmax)")
    metric = lambda lg: int(lg[answer[0]] - lg[answer[1]])
    mc, mk = metric(lc), metric(lk)
    if mc == mk:
        raise ValueError("the clean and the corrupt run have the same metric: nothing to recover")
    base, donor = (corrupt, clean) if direction == "denoising" else (clean, corrupt)
    specs = [_spec(m, s) for s in (sites if sites is not None else grid(m))]
    base_trace: dict = {}
    forward(m, base, base_trace)
    rows = []
    for s in specs:
        ivs = [iv.resample(s["site"], donor, units=s.get("units"), heads=s.get("heads"), pos=s.get("pos"))]
        plan = iv.compile_plan(m, None, ivs)
        (op,) = plan.ops
        lg = forward(m, base, interventions=plan)[-1]
        mp = metric(lg)
        rec = (mp - mk) / (mc - mk) if direction == "denoising" else (mc - mp) / (mc - mk)
        have = base_trace[op.key][np.ix_(list(op.pos), list(op.units))]
        rows.append(dict(site=s["site"], pos=s.get("pos"), cells=_label(s), changed=int((have != op.values).sum()),
                         metric=mp, recovery=rec, doom=None, logits=lg, plan=plan,
                         flag=row_flag(m, op.key, op.pos, op.units, len(base))))
    chosen = set()
    if doom is True:
        chosen = set(range(len(rows)))
    elif isinstance(doom, (int, np.integer)) and not isinstance(doom, bool) and doom:
        order = sorted(range(len(rows)), key=lambda i: (bool(rows[i]["flag"]), -abs(rows[i]["recovery"])))
        chosen = set(order[:int(doom)])
    elif doom:
        chosen = {int(i) for i in doom}
    t0 = time.perf_counter()
    if chosen:
        own = runs is None
        runs = runs or DoomRuns(m, tiles, workers=workers)
        try:
            for i in sorted(chosen):
                r = rows[i]
                rep = runs.run(base, r["plan"])
                r.update(doom=bool(rep["exact"] and np.array_equal(rep["logits"], r["logits"])), values=rep["values"],
                         seconds=rep["seconds"], patched=rep["patches"])
                if out:
                    out(f"row {i}: {r['site']} pos {r['pos']} {r['cells']}: recovery {r['recovery']:.3f}, "
                        f"{'EXACT' if r['doom'] else 'MISMATCH'} ({rep['values']} values, {rep['seconds']:.1f}s)")
        finally:
            if own:
                runs.close()
    for r in rows:
        r.pop("plan"), r.pop("logits")
    return PatchTable(rows, clean, corrupt, answer, direction, mc, mk, m.vocab, time.perf_counter() - t0)


def main(argv=None) -> int:
    import argparse

    from . import shipped

    ap = argparse.ArgumentParser(
        prog="python -m doomformer.patching",
        description="Activation patching with logit-difference recovery per site, on the integer reference and, for "
                    "the rows chosen by --doom, on voodoo dolls checked at tolerance 0.")
    ap.add_argument("model", help="shipped model name or directory")
    ap.add_argument("--clean", required=True, help="clean tokens: vocabulary characters, e.g. i1 (BOS is added)")
    ap.add_argument("--corrupt", required=True, help="corrupt tokens, same length as --clean")
    ap.add_argument("--answer", help="two vocabulary characters 'ab': metric = logit[a] - logit[b] (default: the "
                                     "argmax of the clean run and of the corrupt run)")
    ap.add_argument("--site", action="append", help="site to patch, e.g. x0 or L0.ao (repeatable; default: every site)")
    ap.add_argument("--pos", help="positions, comma separated (default: all)")
    ap.add_argument("--per-head", action="store_true", help="ao, q, k, v: one row per head")
    ap.add_argument("--per-unit", action="store_true", help="one row per unit")
    ap.add_argument("--direction", default="denoising", choices=["denoising", "noising"])
    ap.add_argument("--doom", default="0", help="rows to run in DOOM: 'all', a number N (the N rows with the largest "
                                                "|recovery|), or 0 (default: reference only)")
    ap.add_argument("--workers", type=int, default=2, help="DOOM games")
    args = ap.parse_args(argv)
    m = QModel.load(shipped.resolve(args.model))
    positions = [int(p) for p in args.pos.split(",")] if args.pos else None
    kinds = None
    if args.site:
        kinds = {s.split(".")[-1] for s in args.site}
    specs = [s for s in grid(m, kinds, positions, args.per_head, args.per_unit)
             if not args.site or s["site"] in args.site or (s["site"] == "x0" and "x0" in args.site)]
    doom = True if args.doom == "all" else int(args.doom)
    table = patch_recovery(m, args.clean, args.corrupt, specs, answer=tuple(args.answer) if args.answer else None,
                           direction=args.direction, doom=doom, workers=args.workers, out=print)
    print(table.text())
    return 0 if not table.doom_rows or table.exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
