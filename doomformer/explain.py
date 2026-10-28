"""`doomformer plan --explain`: where the one-map settle time goes, layer by layer, and what each schedule pass saves.

Every number comes from `pure.compile_pure(layout_only=True)`, the same geometry counter `df.plan` uses (no ACS, no
game). `estimated_settle_tics` of a layer is `(waves * S + corridor_width + sync_wait + clock_margin + 15) // 16 + 16`
(docs/scaling.md); its parts are shown separately so the dominant one is visible. The pass table builds the net once per
cumulative pass set: baseline (opt 0), + exclusive, + lights, + colour (opt 1), + tuned (opt 2), + copy (opt 3).
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .pure import TUNING_FILE, Timing

PASS_ORDER = ("exclusive", "lights", "colour", "tuned", "copy")
PASS_LABEL = {"exclusive": "+ exclusive dolls share waves", "lights": "+ light targets never conflict",
              "colour": "+ DSATUR / circle-method colouring", "tuned": "+ tuned timing constants",
              "copy": "+ copy bus (one wave per layer)"}


def _meta(net, passes):
    from .kit import _build
    return _build(net, Path("explain_layout.wad"), layout_only=True, opt=frozenset(passes))


def _timing(layout: dict) -> Timing:
    return Timing(**layout["timing"]) if "timing" in layout else Timing()


def layer_rows(layout: dict) -> list[dict]:
    """Per layer: dolls, waves, S, sectors, tics and the parts the tics are made of (all in game tics)."""
    t = _timing(layout)
    rows = []
    for s in layout["stage_stats"]:
        copy = t.copy_settle if s.get("copy") else 0
        sync = s.get("sync_wait", t.sync_wait)
        parts = dict(waves=s["waves"] * s["wave_spacing"] / 16, corridor=s["corridor_width"] / 16,
                     sync=sync / 16, margin=t.clock_margin / 16, copy=copy / 16, fixed=16 + 15 / 16)
        tics = (s["waves"] * s["wave_spacing"] + s["corridor_width"] + sync + t.clock_margin + copy + 15) // 16 + 16
        top = max(parts, key=parts.get)
        rows.append(dict(name=s["name"], dolls=s["dolls"], waves=s["waves"], lower_bound=s.get("lower_bound"),
                         wave_spacing=s["wave_spacing"], sectors=s.get("sectors"), tics=tics, parts=parts,
                         dominant=top, share=parts[top] / sum(parts.values())))
    return rows


def explain(net, opt=1) -> dict:
    """The per-layer table of `opt` and the cumulative pass table (baseline, then each pass added in turn; the tuned
    rows only when doomformer/tuning.json exists). Layout-only builds, no game."""
    from .pure import opt_passes
    chosen = opt_passes(opt)
    sets = [("baseline (opt 0)", frozenset())]
    acc: set[str] = set()
    for p in PASS_ORDER:
        if p == "tuned" and not TUNING_FILE.is_file():
            continue
        acc.add(p)
        sets.append((PASS_LABEL[p], frozenset(acc)))
    table, prev = [], None
    for label, ps in sets:
        meta = _meta(net, ps)
        lay = meta.layout
        row = dict(label=label, passes=sorted(ps), tics=lay["estimated_settle_tics"], sectors=meta.n_sectors,
                   waves=sum(s["waves"] for s in lay["stage_stats"]), chosen=ps == chosen)
        row["d_tics"] = None if prev is None else row["tics"] - prev["tics"]
        row["d_sectors"] = None if prev is None else row["sectors"] - prev["sectors"]
        table.append(row)
        prev = row
    meta = _meta(net, chosen)
    lay = meta.layout
    return dict(opt=sorted(chosen), tics=lay["estimated_settle_tics"], sectors=meta.n_sectors, layers=layer_rows(lay),
                passes=table, timing=lay.get("timing") or asdict(Timing()))


def explain_text(net, opt=1) -> str:
    """The `plan --explain` report as text."""
    r = explain(net, opt)
    names = {"waves": "waves", "corridor": "corridor length", "sync": "sync wait", "margin": "clock margin",
             "copy": "copy settle", "fixed": "fixed 16"}
    out = [f"schedule of the one-map build, opt {r['opt']}: {r['tics']:,} estimated game tics "
           f"(timing {r['timing']})", ""]
    out.append(f"{'layer':<22}{'dolls':>7}{'waves':>7}{'bound':>7}{'S':>5}{'tics':>8}{'sectors':>10}  dominated by")
    for l in r["layers"]:
        bound = "" if l["lower_bound"] is None else str(l["lower_bound"])
        out.append(f"{l['name']:<22}{l['dolls']:>7,}{l['waves']:>7,}{bound:>7}{l['wave_spacing']:>5}{l['tics']:>8,}"
                   f"{l['sectors'] or 0:>10,}  {names[l['dominant']]} ({l['share']:.0%})")
    out += ["", "passes, cumulative (estimated settle tics, geometry counter):",
            f"{'':<36}{'tics':>8}{'delta':>8}{'sectors':>10}{'delta':>9}{'waves':>8}"]
    for row in r["passes"]:
        d = "" if row["d_tics"] is None else f"{row['d_tics']:+,}"
        ds = "" if row["d_sectors"] is None else f"{row['d_sectors']:+,}"
        out.append(f"{row['label'] + (' <-' if row['chosen'] else ''):<36}{row['tics']:>8,}{d:>8}{row['sectors']:>10,}"
                   f"{ds:>9}{row['waves']:>8,}")
    out.append("bound: clique lower bound of the layer's conflict graph (waves cannot go below it); empty at opt 0.")
    return "\n".join(out)
