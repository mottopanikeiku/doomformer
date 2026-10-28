"""The survivor rules of the mutation harness (`mutate.reason`) and the aggregate controls behind them: a single-site
change to a mechanism the compiler builds several times survives, the same mechanism changed everywhere fails."""
from __future__ import annotations

import pytest

from doomformer import mutate
from doomformer.mutate import Mutant

CTX = dict(n_dolls=100, scrollers=4, raised={64021, 64022, 64023})


def acs(where, old, new, call="Floor_MoveToValue", arg=0):
    return Mutant(0, "acs-reset", where, "acs", 0, old, new, dict(script=3, call=call, arg=arg))


def line(special, arg, cls):
    return Mutant(0, cls, f"linedef 1 special {special} arg{arg} 0 -> 1", "textmap", 1, "", "",
                  dict(special=special, arg=arg))


@pytest.mark.parametrize("m, start", [
    (line(223, 2, "scroller"), "redundancy: the carry belt"),
    (line(37, 0, "move-to-tag"), "inert in forward nets"),
    (line(37, 3, "line-other"), "inert in forward nets"),
    (acs("script 3 Floor_MoveToValue arg0: literal 64009 -> 64010 (Floor_MoveToValue(64009, 150000, 13000, 0);)",
         "64009", "64010"), "redundancy: one wave barrier"),
    (acs("script 3 Floor_MoveToValue arg3: literal 0 -> 1 (Floor_MoveToValue(64018, 150000, 13000, 0);)", "0", "1",
         arg=3), "redundancy: one wave barrier"),
    (acs("script 3 Floor_MoveToValue arg3: literal 0 -> -1 (Floor_MoveToValue(2, 150000, 13000, 0);)", "0", "-1",
         arg=3), "redundancy: script 7"),
    (acs("script 3 Floor_MoveToValue arg3: literal 1 -> 2 (Floor_MoveToValue(64021, 150000, 26, 1);)", "1", "2",
         arg=3), "the negate flag"),
    (acs("script 3 Floor_MoveToValue arg2: literal 26 -> 25 (Floor_MoveToValue(64021, 150000, 26, 1);)", "26", "25",
         arg=2), "freeze walkway height"),
    (acs("script 3 Floor_MoveToValue arg0: literal 64036 -> 64035 (Floor_MoveToValue(64036, 150000, 26, 1);)",
         "64036", "64035"), "inert in forward nets: no line"),
])
def test_survivor_rules(m, start):
    assert (mutate.reason(m, CTX) or "").startswith(start)


def test_rules_do_not_explain_what_they_should_not():
    # a raised freeze barrier whose reset is retargeted is a real fault: no rule
    bad = acs("script 3 Floor_MoveToValue arg0: literal 64021 -> 64022 (Floor_MoveToValue(64021, 150000, 26, 1);)",
              "64021", "64022")
    assert mutate.reason(bad, CTX) is None
    assert mutate.reason(line(223, 2, "scroller"), dict(CTX, scrollers=1)) is None       # a lone scroller is the belt
    assert mutate.reason(line(110, 1, "light-weight"), CTX) is None                       # a weight line


def test_aggregate_controls_on_the_demo_net():
    """One scroller of four, one wave barrier reset, and every freeze line are redundant in a forward pass; three of
    four scrollers and the wave resets as a whole are not."""
    want = {"scrollers removed: 1 of 4": True, "scrollers removed: 3 of 4": False, "freeze lines removed": True,
            "wave barrier resets removed: the first": True, "wave barrier resets removed: all": False,
            "wave barrier reset below the walkway: the first": True,
            "wave barrier resets below the walkway: all": False}
    rows = mutate.survivor_controls(mutate.demo_net(), n_inputs=12, only=set(want))
    got = {r["name"].split(": all ")[0] if r["name"].startswith("freeze") else r["name"]: r["exact"] for r in rows}
    assert got == want, rows


def test_walking_floor_mutants_are_not_equivalent():
    """The walking-floor finding: lowering the walkway floor of a lane by 1 moves that lane's gate threshold by one. On the demo
    net the mutant 'sector 231 (tag 1) heightfloor -26 -> -27' is exact at x = 0 and x = 3 and wrong at x = 2, the
    threshold of the first lane (DOOM h1_0 12, reference 15). So the class is not an equivalent-mutant class, and
    `reason` gives it no rule once any site of it dies on the deep input set."""
    import tempfile
    from dataclasses import asdict
    from pathlib import Path

    from doomformer.fuzz import spec_of
    from doomformer.tile import TileMeta
    net = mutate.demo_net()
    with tempfile.TemporaryDirectory(prefix="dfwalk-") as d:
        base = Path(d) / "b.wad"
        source = mutate.capture_acs(net, base)
        text = dict(mutate.read_lumps(base))["TEXTMAP"].decode()
        sites = mutate.textmap_mutants(text, TileMeta.load(base.with_suffix(".json")))
        m = next(m for m in sites if m.cls.startswith("sector-floor-walk") and "sector 231 (tag 1) heightfloor -26 -> -27" in m.where)

        def run(x):
            return mutate._run_mutant(dict(mutant=asdict(m), spec=spec_of(net), base=str(base), source=source, inputs=[x],
                                           tile=None, max_tics=2000))
        assert not run([0, 0, 0])["killed"] and not run([3, 3, 3])["killed"]
        hit = run([2, 2, 2])
        assert hit["killed"] and "h1_0: DOOM 12 != reference 15" in hit["detail"] + hit.get("how", ""), hit
    # without a deep kill the old label stands; with one the family has no equivalence rule left
    assert "24-unit" in mutate.reason(m, CTX)
    why = mutate.reason(m, dict(CTX, demoted={"sector-floor-walk"}))
    assert why.startswith("input gap: not equivalent")
