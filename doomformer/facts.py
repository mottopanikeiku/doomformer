"""The engine facts of docs/HOW_IT_WORKS.md and docs/engine_limits.md as executable probes.

`doomformer facts` builds one tiny map per case with `udmf.MapBuilder`, runs it in ViZDoom and checks what the engine did
against the expected values, one PASS or FAIL row per fact. Most probes are ACS-only: script 1 sets up the situation,
waits, reads the result into ACS globals (USER1..) and writes one `FACT <case> key=value ...` line with `Log`. Python
only advances tics and reads the globals. `doomformer facts --export DIR` writes those maps as WADs with an
`expected.txt`, so another ZDoom-family port can be run on them and diffed (see `export`).

Probes that need the Python side (the `puke` console command, ViZDoom's action handling, ViZDoom's object list) are
marked `python` and are not exported.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .udmf import MapBuilder, compile_acs, write_wad

DONE_SLOT = 60               # ACS global 60 (USER60) is set to 1 when a case has logged its line
LOG_TAG = "FACT"
ROOM_CEIL = 256


def engine_version() -> str:
    """ViZDoom's version and the ZDoom it embeds (read from the engine binary's version string)."""
    import vizdoom as vzd
    zdoom = "ZDoom version unknown"
    exe = Path(vzd.__file__).parent / "vizdoom"
    try:
        m = re.search(rb"\(ZDOOM ([0-9.]+)\+?\)", exe.read_bytes())
        if m:
            zdoom = f"ZDoom {m.group(1).decode()}"
    except OSError:
        pass
    return f"ViZDoom {vzd.__version__} ({zdoom})"


@dataclass
class Case:
    """One map and what it must show. `keys` are the observed values, in the order of the ACS globals r_<key>."""
    name: str
    m: MapBuilder
    keys: list[str]
    script: str                                  # ACS statements of script 1; assign r_<key>; may Delay()
    expected: dict[str, int]
    extra: str = ""                              # ACS declarations and further scripts
    max_tics: int = 3000
    drive: Callable | None = None                # python-driven case: drive(game) -> observed dict
    configure: Callable | None = None            # python-driven case: configure(game) before the engine starts

    def acs(self, quit_after: bool = False) -> str:
        decl = "\n".join(f"global int {i + 1}:r_{k};" for i, k in enumerate(self.keys))
        log = f'Log(s:"{LOG_TAG} {self.name}"' + "".join(f', s:" {k}=", d:r_{k}' for k in self.keys) + ");"
        return (f'#include "zcommon.acs"\n{decl}\nglobal int {DONE_SLOT}:fact_done;\n{self.extra}\n'
                f"script 1 OPEN {{\n  Delay(2);\n{self.script}\n  {log}\n  fact_done = 1;\n"
                f'{"  Delay(2);" + chr(10) + "  ConsoleCommand(" + chr(34) + "quit" + chr(34) + ");" if quit_after else ""}\n}}\n')

    def expected_line(self) -> str:
        return f"{LOG_TAG} {self.name}" + "".join(f" {k}={self.expected[k]}" for k in self.keys)

    def write(self, path: Path, quit_after: bool = False) -> Path:
        write_wad(path, self.m.textmap(), compile_acs(self.acs(quit_after)))
        return path


@dataclass
class Probe:
    name: str                                    # `doomformer facts --only NAME`
    claim: str                                   # the engine fact in one line
    doc: str                                     # where the docs state it
    cases: Callable[[], list[Case]]
    kind: str = "acs"                            # acs: exportable; python: needs the Python side
    note: str = ""


PROBES: list[Probe] = []


def probe(name: str, claim: str, doc: str, kind: str = "acs", note: str = ""):
    def deco(fn):
        PROBES.append(Probe(name, claim, doc, fn, kind, note))
        return fn
    return deco


# ---- map helpers ---------------------------------------------------------------------------------------------------

@dataclass
class ProbeMap(MapBuilder):
    """A MapBuilder that writes the real player's start last: the last player-1 start in a map's thing list is the
    real player, every earlier one is a voodoo doll."""
    player: tuple[int, int] | None = None

    def textmap(self) -> str:
        saved = list(self.things)
        if self.player:
            self.thing(*self.player, 1)
        try:
            return super().textmap()
        finally:
            self.things[:] = saved


def base_map(player_at=(-1024, 1024)) -> ProbeMap:
    """A 512 x 512 room (floor 0) well away from the probe sectors; the real player starts in it."""
    m = ProbeMap()
    x, y = player_at
    m.rect(x, y, x + 512, y + 512, floor=0, ceil=ROOM_CEIL)
    m.player = (x + 256, y + 256)
    return m


def box(m: MapBuilder, x0: int, y0: int, w: int = 128, h: int = 128, floor: int = 0, ceil: int = ROOM_CEIL, **kw) -> int:
    return m.rect(x0, y0, x0 + w, y0 + h, floor=floor, ceil=ceil, **kw)


# ---- running -------------------------------------------------------------------------------------------------------

def _new_game(wad: Path, console: bool = False, configure: Callable | None = None):
    import vizdoom as vzd
    from .runtime import FREEDOOM2, USER, init_game
    g = vzd.DoomGame()
    g.set_doom_game_path(FREEDOOM2)
    g.set_doom_scenario_path(str(wad))
    g.set_doom_map("MAP01")
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    g.set_render_hud(False)
    g.set_objects_info_enabled(True)
    g.set_available_game_variables(USER)
    g.set_episode_timeout(10 ** 9)
    g.set_mode(vzd.Mode.PLAYER)
    if console:
        g.set_console_enabled(True)
    if configure:
        configure(g)
    init_game(g)
    return g


def _user(g, slot: int) -> int:
    from .runtime import USER
    return int(g.get_game_variable(USER[slot - 1]))


def observe(case: Case, workdir: Path) -> tuple[dict[str, int], int]:
    """Run a case in ViZDoom: (observed values, tics advanced)."""
    wad = case.write(Path(workdir) / f"{case.name.replace('/', '_')}.wad")
    g = _new_game(wad, configure=case.configure)
    try:
        if case.drive is not None:
            return case.drive(g), 0
        tics = 0
        while tics < case.max_tics and not _user(g, DONE_SLOT):
            g.advance_action(5)
            tics += 5
        if not _user(g, DONE_SLOT):
            raise RuntimeError(f"{case.name}: the map's script did not finish within {case.max_tics} tics")
        return {k: _user(g, i + 1) for i, k in enumerate(case.keys)}, tics
    finally:
        g.close()


@dataclass
class Row:
    probe: str
    case: str
    ok: bool
    observed: dict
    expected: dict
    seconds: float
    error: str = ""


def run_probe(p: Probe, workdir: Path) -> list[Row]:
    rows = []
    for case in p.cases():
        t0 = time.time()
        try:
            got, _ = observe(case, workdir)
            err = ""
        except Exception as e:                    # a probe that cannot run is a FAIL row with the reason
            got, err = {}, f"{type(e).__name__}: {e}"
        rows.append(Row(p.name, case.name, got == case.expected and not err, got, case.expected,
                        time.time() - t0, err))
    return rows


def run(only: list[str] | None = None, workdir: Path | None = None, progress=None) -> list[Row]:
    chosen = [p for p in PROBES if not only or p.name in only]
    unknown = set(only or ()) - {p.name for p in PROBES}
    if unknown:
        raise KeyError(f"unknown probe {sorted(unknown)}; known: {', '.join(p.name for p in PROBES)}")
    rows: list[Row] = []
    with tempfile.TemporaryDirectory() as tmp:
        for p in chosen:
            new = run_probe(p, Path(workdir or tmp))
            rows += new
            if progress:
                progress(p, new)
    return rows


# ---- export --------------------------------------------------------------------------------------------------------

def export(out: Path, only: list[str] | None = None) -> list[str]:
    """Write every ACS-only case as `<case>.wad` plus `expected.txt` (the FACT lines) and `RUN.txt` to `out`.
    Returns the case names written."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names, lines, skipped = [], [], []
    for p in PROBES:
        if only and p.name not in only:
            continue
        if p.kind != "acs":
            skipped.append(p.name)
            continue
        for case in p.cases():
            case.write(out / f"{case.name}.wad", quit_after=True)
            names.append(case.name)
            lines.append(case.expected_line())
    (out / "expected.txt").write_text("\n".join(lines) + "\n")
    (out / "RUN.txt").write_text(_run_text(names, skipped, hello=not only))
    if not only:
        from .ports import build_hello
        build_hello(out / "hello")                # a brain with its own driver script; expected.txt from NumPy
        from .ports_machine import export_same_tic
        export_same_tic(out / "same_tic")            # the doll clock's same-tic line specials (models/dollclock)
        build_hello(out / "hello_tags15", low_tags=True)       # the same with every sector tag below 32,768 (Zandronum)
    return names


def _run_text(names: list[str], skipped: list[str], hello: bool = True) -> str:
    brain = ("""hello/expected.txt is the NumPy integer reference of `doomformer hello`: hello.wad runs that brain on 5 inputs (a driver
script writes the inputs, waits for the dolls to stop, logs every readout) and must print the 5 lines byte for byte.
hello_tags15 is the same brain with every sector tag below 32,768.
""" if hello else "")
    return f"""Doomformer engine facts, exported maps ({len(names)} WADs{'; hello/ and hello_tags15/ hold a whole brain' if hello else ''}). Each WAD is
one UDMF map (MAP01, namespace "zdoom") with an ACS OPEN script. The script builds the situation, waits, and logs one line:
    FACT <case> <key>=<value> ...
expected.txt holds the line the case must print. The probe values are what ViZDoom 1.3.2 (ZDoom 2.8.1) printed when
`doomformer facts --export --check` ran these WADs; the claims they check are in docs/engine_limits.md.
{brain}
The maps were run in GZDoom 4.14.2, LZDoom 3.88b, UZDoom 5.0.3, ZDoom 2.8.1 and Zandronum 3.2.1 on x86_64 Linux (GitHub
Actions): results, differences and their causes are in docs/engine_limits.md, "Other ports"; logs in models/ports/.

One case (Linux; an IWAD such as freedoom2.wad from the vizdoom package; stdbuf because the engines buffer a pipe):
    stdbuf -oL gzdoom -iwad freedoom2.wad -file CASE.wad +map MAP01 -stdout -nosound -nomusic -noautoload -nojoy | grep '^{LOG_TAG} '
Headless: xvfb-run, LIBGL_ALWAYS_SOFTWARE=1 (and +vid_preferbackend 0 or 1 for GZDoom). None of the ports above ended
itself after the script's ConsoleCommand("quit"): ZDoom, GZDoom, LZDoom and UZDoom print "... doesn't support execution of
console commands from scripts" (the GZDoom family does not start the 5 maps whose coordinates exceed 32,768), Zandronum
prints nothing and keeps running; ViZDoom's engine refuses it too ("ViZDoom doesn't support execution of console commands from
scripts"). Stop the engine after the line has appeared. The runner of the repository does that, diffs every case against
expected.txt and prints a PASS / DIFF / ERROR table (.github/scripts/ports_run.py, used by .github/workflows/ports.yml):
    python .github/scripts/ports_run.py --name gzdoom --wads DIR --iwad freedoom2.wad --out logs --jobs 3 --cmd \\
      "stdbuf -oL -eL gzdoom -iwad {{iwad}} -file {{wad}} +map MAP01 -stdout -nosound -nomusic -noautoload -nojoy"
    python .github/scripts/ports_run.py --name gzdoom-hello --wads DIR/hello --timeout 300 ... (about 15 s per input)
same_tic/ holds four maps of the doll clock's same-tic probe (models/dollclock): they log how many of 24 dolls were not teleported
home by tic 500. The whole doll-clock counter machine is a separate export: `doomformer machine multiply 7 9 --clock dolls
--export DIR` (also `subleq add 3 4`, and `--low-tags` for Zandronum) writes machine.wad with a logger script and expected.txt
from the interpreter; it takes 235,567 tics, 1.9 hours at 35 tics per second, so the repository's workflow runs it behind
.github/scripts/ports_timeshim.c (an LD_PRELOAD shim that runs the engine's clock 200 times faster; 1 to 6 minutes per port).
The ports run at 35 tics per second, so a probe takes 4 to 15 s. A difference is a finding about the port, not necessarily a
bug in the map. The wrap rows' expected.txt lines are the Linux ViZDoom values; the macOS arm64 wheel of ViZDoom
finds every room in these rows, so an export made on macOS holds the found values. Facts that need the Python side are not exported: {', '.join(skipped) if skipped else 'none'}.
Cases: {', '.join(names)}
"""


# ---- probes: sector light and plane movers (ACS only) ----------------------------------------------------------------

@probe("light_saturates", "sector light is a saturating int16: +30,000 twice from 160 reads 32,767; -30,000 three times "
       "reads -32,768", "HOW_IT_WORKS Engine facts: sector light")
def _light_saturates():
    m = base_map()
    box(m, 0, 0, tag=1, light=160)
    script = """  Light_RaiseByValue(1, 30000);
  r_up1 = GetSectorLightLevel(1);
  Light_RaiseByValue(1, 30000);
  r_up2 = GetSectorLightLevel(1);
  Light_LowerByValue(1, 30000);
  Light_LowerByValue(1, 30000);
  Light_LowerByValue(1, 30000);
  r_down = GetSectorLightLevel(1);"""
    return [Case("light_saturates", m, ["up1", "up2", "down"], script, dict(up1=30160, up2=32767, down=-32768))]


@probe("light_same_tic", "Light_RaiseByValue twice in one tic: both apply (160 + 200 + 200 = 560)",
       "HOW_IT_WORKS Engine facts: the same-tic loss applies to plane movers only")
def _light_same_tic():
    m = base_map()
    box(m, 0, 0, tag=1, light=160)
    script = """  Light_RaiseByValue(1, 200);
  Light_RaiseByValue(1, 200);
  r_light = GetSectorLightLevel(1);"""
    return [Case("light_same_tic", m, ["light"], script, dict(light=560))]


def _mover_case(name: str, script: str, keys: list[str], expected: dict, tags=(2,), floors=0) -> Case:
    m = base_map()
    for i, t in enumerate(tags):
        box(m, 256 * i, 0, tag=t, floor=0, ceil=2000)
    return Case(name, m, keys, script, expected)


@probe("mover_same_tic", "two Floor_MoveToValue requests on one sector in the same tic: the first runs, the second is lost",
       "HOW_IT_WORKS Engine facts: two plane moves, same tic")
def _mover_same_tic():
    s = """  Floor_MoveToValue(2, 64, 100, 0);
  Floor_MoveToValue(2, 64, 50, 0);
  Floor_MoveToValue(3, 64, 50, 0);
  Floor_MoveToValue(3, 64, 100, 0);
  Delay(60);
  r_first100 = GetSectorFloorZ(2, 0, 0) >> 16;
  r_first50 = GetSectorFloorZ(3, 0, 0) >> 16;"""
    return [_mover_case("mover_same_tic", s, ["first100", "first50"], dict(first100=100, first50=50), tags=(2, 3))]


@probe("mover_active_lock", "a sector's floor mover refuses a second request while the first is still moving (not only in "
       "the same tic); once it has stopped a new one is accepted; a floor and a ceiling mover on one sector do not "
       "block each other", "HOW_IT_WORKS Engine facts: two plane moves, same tic (the rule is one active mover "
       "per plane)")
def _mover_active_lock():
    # speed 64 = 8 units per tic: 0 -> 100 takes 13 tics
    s = """  Floor_MoveToValue(2, 64, 100, 0);
  Delay(5);
  Floor_MoveToValue(2, 64, 50, 0);
  Delay(60);
  r_busy = GetSectorFloorZ(2, 0, 0) >> 16;
  Floor_MoveToValue(3, 64, 100, 0);
  Delay(40);
  Floor_MoveToValue(3, 64, 50, 0);
  Delay(40);
  r_after = GetSectorFloorZ(3, 0, 0) >> 16;
  Floor_MoveToValue(4, 64, 100, 0);
  Ceiling_MoveToValue(4, 64, 1800, 0);
  Delay(60);
  r_floor = GetSectorFloorZ(4, 0, 0) >> 16;
  r_ceil = GetSectorCeilingZ(4, 0, 0) >> 16;"""
    return [_mover_case("mover_active_lock", s, ["busy", "after", "floor", "ceil"],
                        dict(busy=100, after=50, floor=100, ceil=1800), tags=(2, 3, 4))]


@probe("mover_speed", "a plane mover's per-tic step is speed/8 units in 16.16: speeds up to 262,144 reach the target in one tic, "
       "300,000 and above pass through wrong heights (the compiler's writes use 150,000)",
       "HOW_IT_WORKS Engine facts: mover height + speed overflows")
def _mover_speed():
    speeds = [150000, 262144, 300000, 500000]
    m = base_map()
    for i in range(len(speeds)):
        box(m, 256 * i, 0, tag=10 + i, ceil=16000)
    s = "".join(f"  Floor_MoveToValue({10 + i}, {v}, 12000, 0);\n" for i, v in enumerate(speeds)) + "  Delay(1);\n"
    keys, exp = [], {}
    # tic 1 and tic 3 after the request; values wrap like 32-bit 16.16 arithmetic
    want = {1: [12000, 12000, 9464, -6072], 3: [12000, 12000, 12000, -12144]}
    for t in (1, 2, 3):
        for i, v in enumerate(speeds):
            if t in want:
                s += f"  r_t{t}_{v} = GetSectorFloorZ({10 + i}, 0, 0) >> 16;\n"
                keys.append(f"t{t}_{v}")
                exp[f"t{t}_{v}"] = want[t][i]
        s += "  Delay(1);\n"
    s += "  Delay(100);\n" + "".join(f"  r_end_{v} = GetSectorFloorZ({10 + i}, 0, 0) >> 16;\n" for i, v in enumerate(speeds))
    for v in speeds:
        keys.append(f"end_{v}")
        exp[f"end_{v}"] = 12000
    return [Case("mover_speed", m, keys, s, exp)]


@probe("mover_busy", "a plane move that starts above 14,018 (floor raise; 14,017 for a ceiling raise) or below -14,017 (floor "
       "lower; -14,018 for a ceiling lower) keeps its mover busy for 3 tics, so a request on that plane in tics 1 and 2 "
       "is dropped (the compiler's speed 150,000 = 18,750 units per tic, 32,768 - 18,750 = 14,018)",
       "docs/verification.md plane-move bound (doomformer/planes.py)")
def _mover_busy():
    # (key, plane, first move: +1 raise / -1 lower, start height, tic of the second, opposite move)
    rows = [("fr_14018", "floor", 1, 14018, 1), ("fr_14019", "floor", 1, 14019, 1),
            ("fl_m14017", "floor", -1, -14017, 1), ("fl_m14018", "floor", -1, -14018, 1),
            ("cr_14017", "ceil", 1, 14017, 1), ("cr_14018", "ceil", 1, 14018, 1),
            ("cl_m14018", "ceil", -1, -14018, 1), ("cl_m14019", "ceil", -1, -14019, 1),
            ("fr_14019_tic3", "floor", 1, 14019, 3)]
    fn = {("floor", 1): "Floor_RaiseByValue", ("floor", -1): "Floor_LowerByValue",
          ("ceil", 1): "Ceiling_RaiseByValue", ("ceil", -1): "Ceiling_LowerByValue"}
    m = base_map()
    for i, (_, plane, _, p, _) in enumerate(rows):
        if plane == "floor":
            box(m, 256 * i, 0, floor=p, ceil=16000, tag=10 + i)
        else:
            box(m, 256 * i, 0, floor=-15000, ceil=p, tag=10 + i)
    s = "".join(f"  {fn[plane, d]}({10 + i}, 150000, 12);\n" for i, (_, plane, d, _, _) in enumerate(rows))
    tic = 0
    for at in sorted({r[4] for r in rows}):
        s += f"  Delay({at - tic});\n"
        tic = at
        s += "".join(f"  {fn[plane, -d]}({10 + i}, 150000, 255);\n" for i, (_, plane, d, _, a) in enumerate(rows) if a == at)
    s += "  Delay(10);\n"
    keys, exp = [], {}
    for i, (key, plane, d, p, _) in enumerate(rows):
        get = "GetSectorFloorZ" if plane == "floor" else "GetSectorCeilingZ"
        s += f"  r_{key} = ({get}({10 + i}, 0, 0) >> 16) - ({p});\n"
        keys.append(key)
        busy = key in ("fr_14019", "fl_m14018", "cr_14018", "cl_m14019")
        exp[key] = 12 * d if busy else (-243 * d)
    return [Case("mover_busy", m, keys, s, exp)]


# ---- probes: plane span, TIDs, conveyors (ACS only) ----------------------------------------------------------------

PLAYER_TID = 77
TAG_SCRIPTS = (f"script 2 ENTER {{ Thing_ChangeTID(0, {PLAYER_TID}); }}\n"      # the real player names himself
               "script 4 (int tid) { Thing_ChangeTID(0, tid); }")                # a doll names itself on a line


@probe("plane_span", "a sector whose floor-to-ceiling span is 32,768 or more freezes the player (16.16 overflow); 32,767 "
       "does not", "HOW_IT_WORKS Engine facts: heights are 16.16, spans >= 32768 freeze dolls")
def _plane_span():
    cases = []
    for floor, ceil in [(0, 32767), (0, 32768), (-16000, 16000), (-16500, 16500)]:
        m = MapBuilder()
        m.rect(-256, -256, 256, 256, floor=floor, ceil=ceil)
        m.thing(0, 0, 1)
        s = f"""  r_span = {ceil - floor};
  int x0 = GetActorX({PLAYER_TID});
  ThrustThing(0, 20, 0, {PLAYER_TID});
  Delay(35);
  r_moved = GetActorX({PLAYER_TID}) > x0;"""
        cases.append(Case(f"plane_span_{ceil - floor}", m, ["span", "moved"], s,
                          dict(span=ceil - floor, moved=int(ceil - floor < 32768)), extra=TAG_SCRIPTS))
    return cases


@probe("voodoo_tid", "player-1 start things do not pass their TID on: the real player and every voodoo doll spawn with TID 0",
       "HOW_IT_WORKS Engine facts: voodoo dolls do not inherit player-start TIDs")
def _voodoo_tid():
    m = MapBuilder()
    m.rect(-512, -512, 512, 512, floor=0, ceil=ROOM_CEIL)
    for i, (x, y) in enumerate([(-300, -300), (-100, -300), (100, -300), (300, -300)]):
        m.thing(x, y, 1, id=101 + i)
    s = """  r_used101 = IsTIDUsed(101);
  r_used102 = IsTIDUsed(102);
  r_used103 = IsTIDUsed(103);
  r_used104 = IsTIDUsed(104);"""
    return [Case("voodoo_tid", m, ["used101", "used102", "used103", "used104"], s,
                 dict(used101=0, used102=0, used103=0, used104=0))]


def _lane(m: MapBuilder, x0: int, y0: int, tid: int, segs: list[tuple], stub_tag: int = 0, w: int = 40,
          tag_x: int = 24, start: int | None = None) -> list[int]:
    """A walled corridor east from (x0, y0): a 24-wide tagging stub (sector tag `stub_tag`), then `segs` =
    [(width, floor, ceil, tag)] in order. A voodoo doll starts 2 units west of the stub's end; the line at the stub's
    end gives it TID `tid` (ACS_ExecuteAlways script 4). Returns the rect indices of segs."""
    a = m.rect(x0, y0, x0 + tag_x, y0 + w, floor=0, ceil=ROOM_CEIL, tag=stub_tag)
    x = x0 + tag_x
    ids = []
    for width, floor, ceil, tag in segs:
        ids.append(m.rect(x, y0, x + width, y0 + w, floor=floor, ceil=ceil, tag=tag))
        x += width
    m.link(a, ids[0], special=226, arg0=4, arg1=0, arg2=tid, playercross=True, repeatspecial=True)
    m.thing(x0 + tag_x - 2 if start is None else start, y0 + w // 2, 1)
    return ids


CARRY = dict(xmove=255, ymove=128)       # the compiler's conveyor strength: x +127, y 0, per scroller (tile.CONVEYOR)


@probe("carry_stack", "stacked Boom carry scrollers (Scroll_Floor arg2 = 1) add up: a doll moves about 4 units per tic per "
       "scroller (78, 157, 236, 315 units in 20 tics for 1 to 4)", "HOW_IT_WORKS Engine facts: conveyors are stacked carry scrollers")
def _carry_stack():
    m = base_map()
    stacks = [1, 2, 3, 4]
    for k, st in enumerate(stacks):
        _lane(m, 0, 1024 + 256 * k, 500 + k, [(3000, 0, ROOM_CEIL, 100 + k)], stub_tag=100 + k)
        m.scroll_carry(100 + k, stack=st, **CARRY)
    s = "  Delay(40);\n"
    keys = []
    for k in range(4):
        s += f"  int a{k} = GetActorX({500 + k});\n"
    s += "  Delay(20);\n"
    for k in range(4):
        s += f"  r_d{stacks[k]} = (GetActorX({500 + k}) - a{k}) >> 16;\n"
        keys.append(f"d{stacks[k]}")
    return [Case("carry_stack", m, keys, s, dict(d1=78, d2=157, d3=236, d4=315), extra=TAG_SCRIPTS)]


@probe("setcurrent", "Sector_SetCurrent called from ACS moves no voodoo doll, no real player and no barrel in its sector (it "
       "is not doll-specific); a carry scroller on the same kind of sector moves the doll and the player", "HOW_IT_WORKS Engine facts: Sector_SetCurrent")
def _setcurrent():
    cases = []
    # a doll: a 24-unit stub with a carry scroller pushes it over its tagging line into the test sector (tag 31x)
    m = base_map()
    for k in range(2):
        _lane(m, 0, 1024 + 256 * k, 500 + k, [(3000, 0, ROOM_CEIL, 310 + k)], stub_tag=300)
    m.scroll_carry(300, stack=4, **CARRY)
    m.scroll_carry(311, stack=4, **CARRY)                       # lane 1 is the control: its test sector is a conveyor
    s = """  Delay(40);
  Sector_SetCurrent(310, 255, 0, 0);
  int a0 = GetActorX(500);
  int a1 = GetActorX(501);
  Delay(35);
  r_doll_current = (GetActorX(500) - a0) >> 16 > 0;
  r_doll_carry = (GetActorX(501) - a1) >> 16 > 0;"""
    cases.append(Case("setcurrent_doll", m, ["doll_current", "doll_carry"], s, dict(doll_current=0, doll_carry=1),
                      extra=TAG_SCRIPTS))
    # the real player in a room with Sector_SetCurrent, and in one with a carry scroller
    for name, tag, special in [("player_current", 320, True), ("player_carry", 321, False)]:
        m = MapBuilder()
        m.rect(-512, -512, 512, 512, floor=0, ceil=ROOM_CEIL, tag=tag)
        m.thing(0, 0, 1)
        if not special:
            m.scroll_carry(tag, stack=4, **CARRY)
        s = f"""  int x0 = GetActorX({PLAYER_TID});
  {"Sector_SetCurrent(" + str(tag) + ", 255, 0, 0);" if special else ""}
  Delay(20);
  r_{name} = (GetActorX({PLAYER_TID}) - x0) >> 16 > 0;"""
        cases.append(Case(f"setcurrent_{name}", m, [name], s, {name: int(not special)}, extra=TAG_SCRIPTS))
    # a barrel
    m = base_map()
    box(m, 0, 0, w=512, h=512, tag=330)
    m.thing(256, 256, 2035, id=88)
    s = """  int x0 = GetActorX(88);
  Sector_SetCurrent(330, 255, 0, 0);
  Delay(35);
  r_barrel = (GetActorX(88) - x0) >> 16 != 0;"""
    cases.append(Case("setcurrent_barrel", m, ["barrel"], s, dict(barrel=0), extra=TAG_SCRIPTS))
    return cases


@probe("doll_steps", "a voodoo doll on a conveyor climbs a 24-unit step and is stopped by a 25-unit one; it passes a 56-unit "
       "gap between floor and ceiling and is stopped by a 55-unit one", "HOW_IT_WORKS Engine facts: thermometer gates "
       "(step up <= 24, ceiling gates 56 units clear)")
def _doll_steps():
    m = base_map()
    lanes = [("step24", 24, ROOM_CEIL + 24), ("step25", 25, ROOM_CEIL + 25), ("gap56", 0, 56), ("gap55", 0, 55)]
    for k, (name, floor, ceil) in enumerate(lanes):
        _lane(m, 0, 1024 + 256 * k, 500 + k, [(64, 0, ROOM_CEIL, 300), (400, floor, ceil, 300)], stub_tag=300)
    m.scroll_carry(300, stack=4, **CARRY)
    s = "  Delay(80);\n"
    for k, (name, _, _) in enumerate(lanes):
        s += f"  r_{name} = GetActorX({500 + k}) > {(24 + 64 + 100) << 16};\n"      # past the first 100 units of the step/gap
    return [Case("doll_steps", m, [n for n, _, _ in lanes], s, dict(step24=1, step25=0, gap56=1, gap55=0),
                 extra=TAG_SCRIPTS)]


@probe("gate_opens", "a doll pushing against a closed floor gate walks on when the gate drops to its own floor (step 0), "
       "but not when it drops to a 1..24-unit step; a doll that arrives at a gate already at a 10- or 24-unit step climbs "
       "it", "HOW_IT_WORKS Engine facts: a doll already pushing against a gate")
def _gate_opens():
    m = base_map()
    steps = [0, 1, 10, 24]
    for k, st in enumerate(steps):                          # lanes 0..3: gate closed (100) while the doll waits
        _lane(m, 0, 1024 + 256 * k, 500 + k, [(64, 0, ROOM_CEIL, 300), (8, 100, ROOM_CEIL, 310 + k),
                                              (400, 0, ROOM_CEIL, 300)], stub_tag=300)
    for k, st in enumerate([10, 24]):                       # lanes 4, 5: the gate is already at the step
        _lane(m, 0, 1024 + 256 * (4 + k), 504 + k, [(64, 0, ROOM_CEIL, 300), (8, st, ROOM_CEIL, 320 + k),
                                                    (400, 0, ROOM_CEIL, 300)], stub_tag=300)
    m.scroll_carry(300, stack=4, **CARRY)
    s = "  Delay(60);\n"
    s += "".join(f"  Floor_MoveToValue({310 + k}, 150000, {st}, 0);\n" for k, st in enumerate(steps))
    s += "  Delay(80);\n"
    s += "".join(f"  r_opens{st} = GetActorX({500 + k}) > {(24 + 64 + 8 + 50) << 16};\n" for k, st in enumerate(steps))
    s += "".join(f"  r_arrives{st} = GetActorX({504 + k}) > {(24 + 64 + 8 + 50) << 16};\n" for k, st in enumerate([10, 24]))
    keys = [f"opens{st}" for st in steps] + ["arrives10", "arrives24"]
    return [Case("gate_opens", m, keys, s, dict(opens0=1, opens1=0, opens10=0, opens24=0, arrives10=1, arrives24=1),
                 extra=TAG_SCRIPTS)]


@probe("self_tag", "a voodoo doll that crosses an ACS_ExecuteAlways line can give itself a TID (Thing_ChangeTID on the "
       "activator); a doll that never crosses has none", "HOW_IT_WORKS Engine facts: dolls tag themselves")
def _self_tag():
    m = base_map()
    _lane(m, 0, 1024, 500, [(3000, 0, ROOM_CEIL, 300)], stub_tag=300)           # carried over its line
    _lane(m, 0, 1280, 501, [(3000, 0, ROOM_CEIL, 301)], stub_tag=301)           # no conveyor: never crosses
    m.scroll_carry(300, stack=4, **CARRY)
    s = """  Delay(30);
  r_crossed = IsTIDUsed(500);
  r_idle = IsTIDUsed(501);
  r_east = GetActorX(500) > 24 << 16;"""
    return [Case("self_tag", m, ["crossed", "idle", "east"], s, dict(crossed=1, idle=0, east=1), extra=TAG_SCRIPTS)]


@probe("line_special_tid", "a line special with a nonzero TID argument, crossed by a voodoo doll, acts on the thing with "
       "that TID (here ThrustThing on the real player); with TID 0 it acts on the doll that crossed",
       "HOW_IT_WORKS Engine facts: line special with a TID acts on that thing")
def _line_special_tid():
    cases = []
    for name, thing_arg in [("player", PLAYER_TID), ("doll", 0)]:
        m = base_map()
        for k, special in enumerate([True, False]):         # lane 0 carries the ThrustThing line, lane 1 does not
            ids = _lane(m, 0, 1024 + 256 * k, 500 + k, [(32, 0, ROOM_CEIL, 300), (600, 0, ROOM_CEIL, 0)], stub_tag=300)
            if special:
                m.link(ids[0], ids[1], special=72, arg0=0, arg1=20, arg2=0, arg3=thing_arg, playercross=True,
                       repeatspecial=True)
        m.scroll_carry(300, stack=4, **CARRY)
        s = f"""  int p0 = GetActorX({PLAYER_TID});
  Delay(80);
  r_player_moved = (GetActorX({PLAYER_TID}) - p0) >> 16 > 0;
  r_doll_pushed = (GetActorX(500) - GetActorX(501)) >> 16 > 50;"""
        cases.append(Case(f"line_special_tid_{name}", m, ["player_moved", "doll_pushed"], s,
                          dict(player_moved=int(name == "player"), doll_pushed=int(name == "doll")), extra=TAG_SCRIPTS))
    return cases


def _order_case(order: str, spacing: int, lanes: int = 16) -> Case:
    """Two lines `spacing` units apart on a conveyor lane, crossed in order by the doll. Each of `lanes` lane pairs
    shifts the pair 1 unit further east, so the doll (about 16 units per tic) meets both lines in one tic for some of
    them. Control lane: both lines run a script that records Timer(); test lane: one line is Teleport_NoFog, the other
    records. `order` "teleport_last": the recorder is crossed first; "teleport_first": the teleport is."""
    m = base_map()
    for i in range(lanes):
        for k, control in enumerate([True, False]):
            row = 2 * i + k
            y = 1024 + 64 * row
            xa = 124 + i
            a = m.rect(0, y, 24, y + 40, floor=0, ceil=ROOM_CEIL, tag=300)
            r1 = m.rect(24, y, xa, y + 40, floor=0, ceil=ROOM_CEIL, tag=300)
            r2 = m.rect(xa, y, xa + spacing, y + 40, floor=0, ceil=ROOM_CEIL, tag=300)
            r3 = m.rect(xa + spacing, y, xa + spacing + 400, y + 40, floor=0, ceil=ROOM_CEIL, tag=300)
            m.link(a, r1, special=226, arg0=4, arg1=0, arg2=500 + row, playercross=True, repeatspecial=True)
            m.thing(22, y + 20, 1)
            recorder = lambda which: dict(special=226, arg0=10, arg1=0, arg2=i, arg3=which, arg4=int(control),
                                          playercross=True, repeatspecial=True)
            if control:
                m.link(r1, r2, **recorder(0))
                m.link(r2, r3, **recorder(1))
            else:
                dest = dict(special=71, arg0=700 + i, playercross=True, repeatspecial=True)
                if order == "teleport_last":
                    m.link(r1, r2, **recorder(0))
                    m.link(r2, r3, **dest)
                else:
                    m.link(r1, r2, **dest)
                    m.link(r2, r3, **recorder(1))
                px = 8000 + 128 * i
                m.rect(px, y, px + 64, y + 64, floor=0, ceil=ROOM_CEIL)
                m.thing(px + 32, y + 32, 14, id=700 + i)
    m.scroll_carry(300, stack=4, **CARRY)
    rec_idx = 0 if order == "teleport_last" else 1
    extra = TAG_SCRIPTS + f"""
int rec_c[{2 * lanes}];
int rec_t[{2 * lanes}];
script 10 (int lane, int which, int control) {{
  if (control) {{ rec_c[lane * 2 + which] = Timer(); }}
  else {{ rec_t[lane * 2 + which] = Timer(); }}
}}"""
    # bad: lanes that break the rule. teleport_last: a same-tic recorder must not fire, an earlier one must.
    # teleport_first: a same-tic recorder must fire, and no other lane reaches the recorder (the doll is gone).
    s = f"""  for (int i = 0; i < {2 * lanes}; i++) {{ rec_c[i] = -1; rec_t[i] = -1; }}
  Delay(100);
  r_same = 0; r_bad = 0;
  for (i = 0; i < {lanes}; i++) {{
    int fired = rec_t[i * 2 + {rec_idx}] >= 0;
    int same = rec_c[i * 2] >= 0 && rec_c[i * 2] == rec_c[i * 2 + 1];
    if (same) {{ r_same++; }}
    {"if (fired != !same) { r_bad++; }" if order == "teleport_last" else "if (fired != same) { r_bad++; }"}
  }}"""
    return Case(f"line_order_{order}_{spacing}", m, ["same", "bad"], s, {}, extra=extra)


@probe("teleport_cancels", "a Teleport_NoFog line a doll crosses in the same tic after another line (4 or 8 units apart) "
       "runs first and stops that line: lines are processed last-crossed first; lines 64 units apart never share a tic",
       "HOW_IT_WORKS Engine facts: lines crossed in one tic are processed last-crossed first, a TeleportOther stops "
       "the rest")
def _teleport_cancels():
    cases = []
    for sp in (4, 8, 64):
        c = _order_case("teleport_last", sp)
        c.expected = dict(same=_SAME["teleport_last", sp], bad=0)
        cases.append(c)
    return cases


@probe("teleport_first", "a recorder line crossed after a teleport line in the same tic still runs (the teleport is "
       "processed last, so it stops nothing); a doll teleported away never reaches the recorder in a later tic",
       "HOW_IT_WORKS Engine facts: a floor line sharing a tic with a teleport fired 17/17 when crossed last")
def _teleport_first():
    cases = []
    for sp in (4, 8, 64):
        c = _order_case("teleport_first", sp)
        c.expected = dict(same=_SAME["teleport_first", sp], bad=0)
        cases.append(c)
    return cases


_SAME = {("teleport_last", 4): 10, ("teleport_last", 8): 4, ("teleport_last", 64): 0,
         ("teleport_first", 4): 10, ("teleport_first", 8): 4, ("teleport_first", 64): 0}


# ---- probes: the 16.16 coordinate wrap ---------------------------------------------------------------------------

def _lone_room(x: int) -> Case:
    """150 rooms (floor 0) in 0..6,000 and one lone room (floor 64) at x; a MapSpot spawned at the lone room's centre
    reports the floor of the sector the engine files it under: 64 if right, 0 if under a neighbour."""
    m = ProbeMap()
    for i in range(15):
        for j in range(10):
            m.rect(i * 400, j * 400, i * 400 + 256, j * 400 + 256, floor=0, ceil=ROOM_CEIL)
    m.player = (128, 128)
    m.rect(x, 0, x + 256, 256, floor=64, ceil=ROOM_CEIL)
    s = f"""  Spawn("MapSpot", {x + 128}.0, 128.0, 64.0, 900);
  r_floor = GetActorFloorZ(900) >> 16;"""
    return Case(f"wrap_lone_room_{x}", m, ["floor"], s, dict(floor=64 if x not in _LONE_WRONG or ON_MACOS else 0))


_LONE_WRONG = (-32000, 32800, 33000, 34000)
ON_MACOS = sys.platform == "darwin"          # wrap rows: ViZDoom 1.3.2's macOS arm64 build files every thing correctly
MACOS_NOTE = (" Platform dependent: the Linux wheel (x86_64, arm64) misfiles the rows named; the macOS arm64 wheel found "
              "every room in its own sector in GitHub Actions run 40690058123 (macos-14), so on macOS these rows expect "
              "the found value (floor 64, bad 0); the cause was not investigated.")


@probe("wrap_lone_room", "node differences above 32,767 units can misfile a thing, in a way that depends on the BSP: with 150 "
       "rooms in 0..6,000 and one lone room, a lone room starting at x = 32,700 is found in its own sector, at 32,800, "
       "33,000 and 34,000 it is filed under a neighbour, at 36,000 it is found again by wrap luck; to the west -30,000 "
       "is found and -32,000 is not." + MACOS_NOTE, "engine_limits.md: 16.16 coordinate wrap; HOW_IT_WORKS Engine facts")
def _wrap_lone_room():
    return [_lone_room(x) for x in (-30000, -32000, 28000, 32000, 32700, 32800, 33000, 34000, 36000)]


def _balanced(width: int, vertical: bool, n: int = 100) -> Case:
    """`n` rooms of floors 16..80 evenly over `width` units on one axis (a balanced BSP, its root near the middle); the
    script spawns a MapSpot at each room's centre and counts the rooms the engine files under another sector."""
    m = ProbeMap()
    step = width // n
    for i in range(n):
        lo = -width // 2 + i * step
        f = 16 * (1 + i % 5)
        m.rect(*((0, lo, 128, lo + 100) if vertical else (lo, 0, lo + 100, 128)), floor=f, ceil=ROOM_CEIL)
    first = -width // 2 + 50
    m.player = (64, first) if vertical else (first, 64)
    centre = f"({-width // 2} + i * {step}) * 1.0 + 50.0"
    pos = f"64.0, {centre}" if vertical else f"{centre}, 64.0"
    s = f"""  r_bad = 0;
  for (int i = 0; i < {n}; i++) {{
    int f = 16 * (1 + i % 5);
    Spawn("MapSpot", {pos}, f * 1.0, 900 + i);
    if (GetActorFloorZ(900 + i) >> 16 != f) {{ r_bad++; }}
  }}"""
    axis = "y" if vertical else "x"
    return Case(f"wrap_balanced_{axis}_{width}", m, ["bad"], s, dict(bad=int(vertical and width == 65000 and not ON_MACOS)))


@probe("wrap_balanced", "in a balanced BSP (100 rooms in a row) no room is misfiled up to 64,000 units wide on either axis "
       "(every node difference stays below 32,768); the first miss appears at 65,000 on the y axis." + MACOS_NOTE,
       "engine_limits.md: 16.16 coordinate wrap")
def _wrap_balanced():
    return [_balanced(w, v) for w in (60000, 64000) for v in (False, True)] + [_balanced(65000, True)]


@probe("doll_fall", "a voodoo doll carried off a 70-unit ledge falls for 10 tics (2, 5, 9, 14, 20, 27 ... units below the "
       "ledge in the first tics) before it stands on the lower floor", "HOW_IT_WORKS Engine facts: a doll that crosses "
       "a gate falls back for several tics")
def _doll_fall():
    m = base_map()
    _lane(m, 0, 1024, 500, [(40, 0, ROOM_CEIL, 300), (600, -70, ROOM_CEIL, 300)], stub_tag=300)
    m.scroll_carry(300, stack=4, **CARRY)
    extra = TAG_SCRIPTS + "\nint zz[60];"
    s = """  for (int t = 0; t < 60; t++) { zz[t] = GetActorZ(500) >> 16; Delay(1); }
  int t0 = 0;
  while (zz[t0] >= 0) { t0++; }
  r_below1 = zz[t0];
  r_below3 = zz[t0 + 2];
  r_below6 = zz[t0 + 5];
  r_fall_tics = 0;
  for (t = 0; t < 60; t++) { if (zz[t] < 0 && zz[t] > -70) { r_fall_tics++; } }
  r_landed = zz[59];"""
    return [Case("doll_fall", m, ["below1", "below3", "below6", "fall_tics", "landed"], s,
                 dict(below1=-2, below3=-9, below6=-27, fall_tics=10, landed=-70), extra=extra)]


@probe("dormant", "a Thing_Deactivate'd (dormant) monster takes no Thing_Damage2; a monster that starts dormant takes none "
       "until Thing_Activate", "HOW_IT_WORKS Engine facts: dormant monsters take no damage (here ACS damage; the docs' "
       "pistol shots are not repeated)")
def _dormant():
    m = base_map()
    box(m, 0, 0, w=512, h=512)
    m.thing(100, 100, 3001, id=900, dormant=True)
    m.thing(300, 100, 3001, id=901)
    s = """  Thing_Damage2(900, 20, "Normal");
  Thing_Damage2(901, 20, "Normal");
  Delay(5);
  r_start_dormant = GetActorProperty(900, APROP_Health);
  r_awake = GetActorProperty(901, APROP_Health);
  Thing_Activate(900);
  Thing_Damage2(900, 20, "Normal");
  Delay(5);
  r_activated = GetActorProperty(900, APROP_Health);
  Thing_Deactivate(901);
  Thing_Damage2(901, 20, "Normal");
  Delay(5);
  r_deactivated = GetActorProperty(901, APROP_Health);"""
    return [Case("dormant", m, ["start_dormant", "awake", "activated", "deactivated"], s,
                 dict(start_dormant=60, awake=40, activated=40, deactivated=40))]


# ---- probes that need the Python side ------------------------------------------------------------------------------

def _drive_puke(g) -> dict:
    g.advance_action(3)
    out = {}
    for label, cmd in (("plain", "puke 5"), ("always", "puke -6")):
        before = _user(g, 1 if label == "plain" else 2)
        g.send_game_command(cmd)
        g.send_game_command(cmd)
        g.advance_action(3)
        out[f"{label}_twice_one_tic"] = _user(g, 1 if label == "plain" else 2) - before
        before = _user(g, 1 if label == "plain" else 2)
        g.send_game_command(cmd)
        g.advance_action(1)
        g.send_game_command(cmd)
        g.advance_action(3)
        out[f"{label}_two_tics"] = _user(g, 1 if label == "plain" else 2) - before
    return out


@probe("puke", "`puke N` starts script N once even when sent twice in one tic (the second start is dropped); `puke -N` "
       "(ACS_ExecuteAlways) starts it both times", "HOW_IT_WORKS Engine facts: puke N drops a second start", kind="python")
def _puke():
    m = base_map()
    extra = "script 5 (void) { r_plain++; }\nscript 6 (void) { r_always++; }"
    return [Case("puke", m, ["plain", "always"], "", dict(plain_twice_one_tic=1, plain_two_tics=2,
                                                          always_twice_one_tic=2, always_two_tics=2),
                 extra=extra, drive=_drive_puke)]


def _drive_object_list(g) -> dict:
    g.advance_action(5)
    objs = g.get_state().objects
    return dict(listed=len(objs), doomplayers=sum(o.name == "DoomPlayer" for o in objs))


@probe("object_list", "ViZDoom's game state lists at most 4,096 objects (VIZ_MAX_OBJECTS): a map with 4,225 voodoo dolls "
       "and the player returns 4,096", "engine_limits.md: object list", kind="python")
def _object_list():
    m = base_map()
    box(m, 0, 0, w=2700, h=2700)
    n = 0
    for i in range(65):
        for j in range(65):
            m.thing(20 + 40 * i, 20 + 40 * j, 1)
            n += 1
    return [Case("object_list", m, [], "", dict(listed=4096, doomplayers=4096), drive=_drive_object_list)]


def _configure_buttons(g):
    import vizdoom as vzd
    g.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.TURN_LEFT])


def _drive_actions(g) -> dict:
    import vizdoom as vzd
    px = lambda: g.get_game_variable(vzd.GameVariable.POSITION_X)
    g.advance_action(3)
    g.set_action([1, 0, 0])
    g.advance_action(20)                      # set_action once; advance_action keeps pressing MOVE_FORWARD
    p0 = px()
    g.advance_action(1)
    held = px() - p0
    g.set_action([0, 0, 0])
    g.advance_action(60)
    p1 = px()
    g.advance_action(1)
    released = px() - p1
    return dict(repeats=int(held > 5), stops=int(released < 0.5))


@probe("advance_action", "ViZDoom's advance_action repeats the last action every tic until set_action changes it",
       "HOW_IT_WORKS Engine facts: advance_action repeats the last action", kind="python")
def _advance_action():
    m = ProbeMap()
    m.rect(-4000, -4000, 4000, 4000, floor=0, ceil=ROOM_CEIL)
    m.player = (0, 0)
    return [Case("advance_action", m, [], "", dict(repeats=1, stops=1), drive=_drive_actions,
                 configure=_configure_buttons)]


def _drive_turn(g) -> dict:
    import vizdoom as vzd
    angle = lambda: g.get_game_variable(vzd.GameVariable.ANGLE)
    g.advance_action(3)
    start = angle()
    g.set_action([0, 0, 1])                    # TURN_LEFT key held
    out = {}
    for k in range(1, 11):
        g.advance_action(1)
        if k in (1, 2, 6, 10):
            out[f"key_{k}"] = round(((angle() - start + 180) % 360 - 180) * 1e4)
    g.set_action([0, 0, 0])
    g.advance_action(3)
    start = angle()
    g.set_action([0, 3.515625, 0])             # TURN_LEFT_RIGHT_DELTA, one tic
    g.advance_action(1)
    out["delta_1"] = round(((angle() - start + 180) % 360 - 180) * 1e4)
    return out


@probe("turn_key", "a freshly pressed turn key turns slowly for its first tics (Doom's slow-turn ramp); "
       "TURN_LEFT_RIGHT_DELTA turns exactly the angle it is given in one tic (values in 1e-4 degrees: 0 and 0 after "
       "1 and 2 tics of the key, 5.2734 after 6, 19.3359 after 10, against 3.515625 per tic by delta)",
       "HOW_IT_WORKS Engine facts: a freshly pressed turn key", kind="python")
def _turn_key():
    m = base_map()
    return [Case("turn_key", m, [], "", dict(key_1=0, key_2=0, key_6=52734, key_10=193359, delta_1=-35156),
                 drive=_drive_turn, configure=_configure_buttons)]


# ---- command line --------------------------------------------------------------------------------------------------

def _wrap(prefix: str, text: str, width: int = 118) -> str:
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=" " * len(prefix))


def _run_wad(wad: Path) -> int:
    """Child of `--check`: run an exported WAD with the engine's console on stdout until it quits or has logged."""
    g = _new_game(wad, console=True)
    try:
        for _ in range(2000):
            g.advance_action(5)
            if _user(g, DONE_SLOT):
                break
    except Exception:                     # the map's own `quit` ends the engine process: expected
        pass
    return 0


def check_export(out: Path, timeout: int = 120) -> list[tuple[str, bool, str]]:
    """Run every exported WAD in a fresh ViZDoom process with the console on stdout, collect its FACT line and compare
    with expected.txt. Returns (case, ok, observed line)."""
    expected = {line.split()[1]: line for line in (Path(out) / "expected.txt").read_text().splitlines() if line}
    results = []
    for name, want in expected.items():
        r = subprocess.run([sys.executable, "-m", "doomformer.facts", "--run-wad", str(Path(out) / f"{name}.wad")],
                           capture_output=True, text=True, timeout=timeout, env=dict(os.environ, OMP_NUM_THREADS="1"))
        lines = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if ln.strip().startswith(LOG_TAG + " ")]
        got = lines[0] if lines else ""
        results.append((name, got == want, got))
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="doomformer facts", description="Engine facts as executable probes: one tiny map "
                                 "per case, run in ViZDoom, one PASS or FAIL row per fact.")
    ap.add_argument("--only", action="append", metavar="NAME", help="run only this probe (repeatable); --list shows names")
    ap.add_argument("--list", action="store_true", help="list the probes and exit")
    ap.add_argument("--json", action="store_true", help="print the rows as JSON")
    ap.add_argument("--export", metavar="DIR", help="write the ACS-only probes as WADs with expected.txt and RUN.txt "
                    "(command lines for GZDoom and Zandronum) to DIR instead of running them")
    ap.add_argument("--check", action="store_true", help="with --export: run every exported WAD in ViZDoom with the "
                    "console on stdout and compare its FACT line with expected.txt (the export path, not another port)")
    ap.add_argument("--run-wad", metavar="WAD", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.run_wad:
        return _run_wad(Path(args.run_wad))
    if args.list:
        for p in PROBES:
            print(f"{p.name:20s} {p.kind:7s} {p.claim}")
        return 0
    if args.export:
        names = export(Path(args.export), args.only)
        print(f"{len(names)} WADs, expected.txt and RUN.txt written to {args.export}")
        if args.check:
            res = check_export(Path(args.export))
            for name, ok, got in res:
                print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n      got: {got or '(no FACT line)'}"))
            print(f"{sum(ok for _, ok, _ in res)}/{len(res)} exported WADs print their expected line in ViZDoom")
            return 0 if all(ok for _, ok, _ in res) else 1
        return 0
    t0 = time.time()
    version = engine_version()

    def progress(p: Probe, rows: list[Row]) -> None:
        if not args.json:
            ok = all(r.ok for r in rows)
            print(f"{'PASS' if ok else 'FAIL'}  {p.name:18s} {sum(r.ok for r in rows)}/{len(rows)} cases  "
                  f"{sum(r.seconds for r in rows):5.1f}s  [{p.kind}]", flush=True)

    if not args.json:
        print(f"engine: {version}\n")
    try:
        rows = run(args.only, progress=progress)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    by_probe = {p.name: p for p in PROBES}
    if args.json:
        print(json.dumps(dict(engine=version, seconds=round(time.time() - t0, 1),
                              rows=[dict(probe=r.probe, case=r.case, ok=r.ok, observed=r.observed, expected=r.expected,
                                         seconds=round(r.seconds, 2), error=r.error, kind=by_probe[r.probe].kind,
                                         claim=by_probe[r.probe].claim) for r in rows]), indent=1))
        return 0 if all(r.ok for r in rows) else 1
    bad = [r for r in rows if not r.ok]
    for r in bad:
        print(f"\nFAIL {r.case}: expected {r.expected}, observed {r.observed} {r.error}")
    names = {r.probe for r in rows}
    n_ok = sum(all(r.ok for r in rows if r.probe == n) for n in names)
    print(f"\n{n_ok}/{len(names)} facts hold ({sum(r.ok for r in rows)}/{len(rows)} cases) on {version} in "
          f"{time.time() - t0:.0f} s")
    print("\nWhat each row checks:")
    for p in PROBES:
        if p.name in names:
            print(_wrap(f"  {p.name:18s} ", p.claim))
    return 0 if not bad else 1


if __name__ != "__main__":
    from . import facts_ports  # noqa: E402,F401  (registers the follow-up probes of docs/engine_limits.md)

if __name__ == "__main__":
    sys.exit(main())
