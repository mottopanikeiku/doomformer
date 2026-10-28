"""A Python-free `hello` brain for other ZDoom-family ports.

`doomformer facts --export DIR` writes `DIR/hello/hello.wad` and `DIR/hello/expected.txt`. The WAD is the brain of
`doomformer hello` (MLP 4-6-5-3, seed 0) with a driver script added: an ACS OPEN script writes the input ports, drops
the start barriers, waits until every voodoo doll has stood still for 8 tics, and logs one line with every value the
Python runner reads (3 outputs, hidden preactivations and gate floors), once per input, rewinding the map between
inputs with the map's own script 3. `expected.txt` holds the same lines computed by the NumPy reference
(`MLP.reference`), not by any engine. A port that prints identical lines runs the doll arithmetic exactly.

The driver replaces what `TileRunner` does from Python (`start`, `settle`, `read`). The settle test is the map's own
motion script (`tile.motion_acs`) in a loop, without the player.
"""
from __future__ import annotations

import contextlib
import re
from pathlib import Path

import numpy as np

from .tile import DOLL_TID0, MOVE_SPEED, START_TAG

LOG_TAG = "FACT"
SEED = 0
RUNS = 5                    # the default of `doomformer hello`
SETTLE_LIMIT = 20000        # tics; the driver logs `settle=timeout` instead of values beyond it


def hello_net(seed: int = SEED, runs: int = RUNS):
    """The network and the inputs of `doomformer hello --seed SEED --cases RUNS` (same draws, same order)."""
    import doomformer as df
    rng = np.random.default_rng(seed)
    sizes = [4, 6, 5, 3]
    net = df.MLP([rng.integers(-20, 21, (sizes[l + 1], sizes[l])) for l in range(3)],
                 [rng.integers(-30, 31, sizes[l + 1]) for l in range(3)],
                 [np.cumsum(rng.integers(15, 40, 3)) for _ in range(3)])
    return net, [rng.integers(-150, 151, net.n_in) for _ in range(runs)]


def expected_lines(net, inputs, names: list[str]) -> list[str]:
    lines = []
    for k, x in enumerate(inputs, 1):
        ref = net.reference(net._x(x))
        lines.append(f"{LOG_TAG} hello run={k}" + "".join(f" {n}={net._expected(n, ref)}" for n in names))
    return lines


def driver_acs(meta, net, inputs) -> str:
    """ACS of the driver: `meta` is the TileMeta of the brain built without it (its tags and readout expressions)."""
    n = meta.n_dolls
    names = list(meta.outputs)
    reads = meta.layout["read_expressions"]
    body = []
    for k, x in enumerate(inputs, 1):
        ports = []
        for key, tag in meta.ports.items():
            kind, name = key.split(":", 1)
            v = int(net._x(x)[int(name.lstrip("-")[1:])])
            v = -v if name.startswith("-") else v
            fn = "Floor_MoveToValue" if kind == "floor" else "Ceiling_MoveToValue"
            ports.append(f"  {fn}({tag}, {MOVE_SPEED}, {abs(v)}, {int(v < 0)});")
        log = f's:"{LOG_TAG} hello run={k}"' + "".join(f', s:" {nm}=", d:{reads[nm]}' for nm in names)
        body.append(f"""  // run {k}
  {"ACS_Execute(3, 0); Delay(3);" if k > 1 else ""}
{chr(10).join(ports)}
  Delay(3);
  Floor_LowerToLowest({START_TAG}, {MOVE_SPEED});
  tics = 0;
  moved = 1;
  while (moved > 0 && tics < {SETTLE_LIMIT}) {{
    Delay(8);
    tics += 8;
    moved = 0;
    for (i = 0; i < {n}; i++) {{
      tid = {DOLL_TID0} + i;
      if (!IsTIDUsed(tid)) {{ moved++; continue; }}
      px = GetActorX(tid);
      py = GetActorY(tid);
      pz = GetActorZ(tid);
      if (px != hd_x[i] || py != hd_y[i] || pz != hd_z[i]) {{ moved++; hd_x[i] = px; hd_y[i] = py; hd_z[i] = pz; }}
    }}
  }}
  if (moved > 0) {{ Log(s:"{LOG_TAG} hello run={k} settle=timeout"); }}
  else {{ Log({log}); }}""")
    return f"""int hd_x[{n}];
int hd_y[{n}];
int hd_z[{n}];
script 20 OPEN {{
  int tics;
  int moved;
  int i;
  int tid;
  int px;
  int py;
  int pz;
  Delay(2);
{chr(10).join(body)}
  Delay(2);
  ConsoleCommand("quit");
}}"""


@contextlib.contextmanager
def _tags_below_2_15():
    """Build with the stage and freeze tag bases moved below 32,768 (the compiler's defaults are 60,000 and 64,000).
    Zandronum 3.2.1 reads sector tags as signed 16 bits, so it cannot find the sectors of a default brain (see
    docs/engine_limits.md, "Other ports"). Diagnostic only: the default build is untouched."""
    from . import pure
    saved = pure.STAGE_TAG0, pure.FREEZE_TAG0
    pure.STAGE_TAG0, pure.FREEZE_TAG0 = 25000, 30000
    try:
        yield
    finally:
        pure.STAGE_TAG0, pure.FREEZE_TAG0 = saved


def build_hello(out: Path, seed: int = SEED, runs: int = RUNS, opt: int = 0, low_tags: bool = False) -> list[str]:
    """Write hello.wad and expected.txt into `out`; returns the expected lines. low_tags: every sector tag below 32,768."""
    with _tags_below_2_15() if low_tags else contextlib.nullcontext():
        return _build_hello(Path(out), seed, runs, opt, low_tags)


def _build_hello(out: Path, seed: int, runs: int, opt: int, low_tags: bool) -> list[str]:
    from .pure import compile_pure
    out.mkdir(parents=True, exist_ok=True)
    net, inputs = hello_net(seed, runs)
    stages = net._stages()
    consumed = {key.split(":", 1)[1] for s in stages for key in s.ports}
    readout = [name for s in stages for name, _ in s.outputs if name in consumed]
    first = out / "_driverless.wad"
    meta = compile_pure(stages, first, "mlp", readout=readout, opt=opt)
    acs = driver_acs(meta, net, inputs)

    def world(m) -> str:                                     # the default closet of compile_pure, plus the driver
        m.rect(-768, -512, -512, -256, floor=0, ceil=128)
        m.thing(-640, -384, 1)
        return acs

    meta2 = compile_pure(stages, out / "hello.wad", "mlp", readout=readout, world=world, opt=opt)
    assert meta2.ports == meta.ports and meta2.outputs == meta.outputs, "the driver changed the layout"
    if low_tags:
        wad = (out / "hello.wad").read_bytes()
        ids = [int(v) for v in re.findall(rb"\bid = (\d+);", wad)]
        assert ids and max(ids) < 32768, f"a sector tag of {max(ids)} remains"
    for p in (first, first.with_suffix(".json"), (out / "hello.json")):
        p.unlink(missing_ok=True)
    lines = expected_lines(net, inputs, list(meta.outputs))
    (out / "expected.txt").write_text("\n".join(lines) + "\n")
    (out / "inputs.txt").write_text("".join(f"run={k} x={x.tolist()}\n" for k, x in enumerate(inputs, 1)))
    return lines
