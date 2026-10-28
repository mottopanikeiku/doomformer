"""Does a frozen world depend on how many tics it stays frozen? The frozen integer-policy control (demos/brainlive.py)
with N extra tics of dormancy after each action is chosen, seeds 4, 5, 6, first life: the first decision whose action
differs from N = 0. Run from the repository root: python models/live_demons/dormant_tics.py"""
import tempfile
from pathlib import Path

from demos import brainlive as b
from doomformer import shipped
from doomformer.intref import QModel

m = QModel.load(shipped.resolve("doomplayer_onemap"))
orig = b.control_clock_acs


def run(seed, extra):
    def patched(lives, delay, frozen):
        s = orig(lives, delay, frozen)
        return s.replace("while (bl_action < 0) Delay(1); ", f"while (bl_action < 0) Delay(1); Delay({extra}); ") if extra else s
    b.control_clock_acs = patched
    try:
        with tempfile.TemporaryDirectory() as t:
            r = b.ControlRun(b.build_control_wad(Path(t) / "c.wad", lives=1, delay=0, frozen=True), seed)
            res = r.play("model", m, 1)
            r.close()
    finally:
        b.control_clock_acs = orig
    return "".join("LRF"[a] for a in res["actions"]), res["rows"]


for seed in (6, 4, 5):
    base, rows = run(seed, 0)
    print(seed, rows)
    for extra in (1, 3, 700):
        a, rows = run(seed, extra)
        n = next((i for i, (x, y) in enumerate(zip(base, a)) if x != y), None)
        print("  extra dormant tics", extra, "first difference at decision", None if n is None else n + 1)
