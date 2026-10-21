"""The d=16 marine policy with his arena as its world still packs into the 32,000-unit span (demos/arena.py)."""
from pathlib import Path

from demos import arena
from doomformer import pure
from doomformer.intref import QModel
from doomformer.onemap import build_onemap

MODELS = Path(__file__).resolve().parents[2] / "models"


def test_policy_with_arena_stays_within_the_span_limit():
    layout = build_onemap(QModel.load(MODELS / "doomplayer"), Path("unused.wad"), layout_only=True,
                          world=arena.build).layout                   # d=16: 528,366 sectors, was 57,978 wide
    x0, y0, x1, y1 = layout["bounds"]
    assert layout["fits"] and layout["span"] == [x1 - x0, y1 - y0]
    assert max(layout["span"]) <= pure.MAX_SPAN == 32000
    assert pure.X_ORIGIN <= x0 and x1 <= pure.X_END
    assert layout["world_area"] is not None
