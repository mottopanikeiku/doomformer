"""THE FLY IS ALIVE: a closed sensorimotor loop inside ONE DOOM game, with no Python decision in the loop.

One map holds the spiking circuit (doomformer.spiking: a neuron's potential is a floor height, a spike is a voodoo doll
getting through a gate, a synapse is a line it crosses) AND the marine's level: a walkway (Boom carry scrollers) that
carries him from station to station. An ACS clock in the map (script CLOCK, an OPEN script) runs every 1 ms model step:

  1. REWIND   pure.py's script 3: dolls home, barriers up, transient values back to their biases (potentials persist)
  2. SENSE    from the game state: standing at a food station while its health bonus (on a counter) still
              exists -> sugar; in the nukage station -> bitter; in the dust station while dust is left -> Johnston's
              organ (jo); the demon visible (CheckSight), within LOOM_R and +-45 degrees of where he faces -> LC4.
              Then one Bernoulli draw per input line from an in-map integer PRNG (Park-Miller, Schrage's method: no
              intermediate leaves int32); a line spikes iff its group is on and its draw is <= the group's
              threshold. Every line draws every step, so the PRNG stream does not depend on what he senses.
  3. WRITE    the input ports (pure.py's script 1, as TileRunner.start does), then START: script 7 drops the barriers
  4. COMPUTE  PERIOD tics (calibrate(): worst-case settle of the map's clock-free twin + margin). Output neurons'
              spike dolls cross extra line specials aimed at the player's TID (build_spiking(actions=...)): MN9 ->
              ThrustThing toward the food; Giant Fiber -> ThrustThing away from the demon + ThrustThingZ (takeoff);
              aDN1 -> Radius_Quake on him (the grooming twitch) + Thing_Remove of the dust motes + Sector_SetColor/
              Sector_SetFade clearing the dusty patch's haze. Each also bumps an in-map action counter.
  5. HOLD     HOLD tics in which the step's results stay readable
  6. WORLD    WORLD tics of "world time": the demon is awake only now (Thing_Activate ... Thing_Deactivate); the
              station logic runs once per step (a station's gate opens after `linger` steps without its stimulus
              source -- food eaten, dust gone, he is in the pit -- or after `max_steps`), dust puffs are spawned.

The walkway conveyors run all the time (level design); his body moves at spike time (thrusts) and on the walkway.
Python builds the map, advances the game clock (advance_action, no buttons) and runs read-only readout scripts that
copy values into ACS globals nothing in the map reads (ViZDoom USER1..60). It writes no port, potential or input and
moves nothing: its game handle is a doomformer.readout.ReadoutOnly, which raises on any other command or game call.
Verification (LoopRun.read): the reference IntNet.step is fed the Python replica of the in-map PRNG gated by the
published group mask, and every neuron's spike and potential must agree, every step.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import vizdoom as vzd

from doomformer.readout import ReadoutOnly, readout_commands
from doomformer.runtime import USER, TileRunner
from doomformer.spiking import IntNet, build_spiking, read_step
from doomformer.tile import PAGE

from . import say

ROOT = Path(__file__).resolve().parents[1]

# line specials (ZDoom numbering)
THRUST, THRUST_Z, QUAKE, THING_REMOVE, EXEC_ALWAYS, SET_COLOR, SET_FADE = 72, 128, 120, 132, 226, 212, 213

# Bernoulli input spikes: Park-Miller "minimal standard" Lehmer generator, x <- 16807 x mod (2^31 - 1), evaluated
# with Schrage's method so that no intermediate value leaves the signed 32-bit range (ACS ints are 32-bit).
M31, LEHMER_A, SCHRAGE_Q, SCHRAGE_R = 2147483647, 16807, 127773, 2836


def lehmer(x: int) -> int:
    hi, lo = divmod(x, SCHRAGE_Q)
    x = LEHMER_A * lo - SCHRAGE_R * hi
    return x if x > 0 else x + M31


def threshold(p: float) -> int:
    """A line with spike probability p spikes iff its draw (uniform on 1..M31-1) is <= this."""
    return int(p * M31)


# --- ids: tags 7000-7099 and TIDs 900-999 are free in pure.py maps; scripts >= 200 belong to the world --------
PLAYER_TID, FOOD_TID0, DUST_TID, DEMON_TID = 900, 910, 920, 930
WALK_TAG, ZONE_TAG0, GATE_TAG0, PIT_TAG = 7000, 7010, 7030, 7050
CLOCK, STATUS, ENTER, COUNT = 210, 211, 212, 213
# --- clock (tics) --------------------------------------------------------------------------------------------
BOOT, REWIND, WRITE, HOLD, WORLD = 8, 4, 3, 16, 16
PH_START, PH_COMPUTE, PH_HOLD, PH_WORLD = 1, 2, 3, 4
# HOLD is world time (the walkway carries him through it) and the documented runs were made with it. Guarded reads of
# the 255-neuron circuit's 9 pages take 18 tics, more than HOLD, so the loop keeps the unguarded readback
# (docs/verification.md). Every spike and potential is still compared with the reference every step.
GUARD = False
# --- level -----------------------------------------------------------------------------------------------------
X0, Y0, HW, CEIL_H = -2048, -6144, 80, 256     # walkway start, centre line, half width, ceiling
APPROACH, STATION_LEN, GATE_W, GATE_H = 448, 256, 16, 40
WALK = dict(xmove=255, ymove=88, stack=2)      # transit: ~7.9 units/tic east, ~2.5 units/tic south (he ends up on
#                                                the south wall, so every station starts him at the same distance)
PUSH = dict(xmove=255, ymove=128, stack=1)     # station floors and gates: ~4 units/tic east, against the gate
COUNTER_W, COUNTER_D, COUNTER_H = 64, 44, 32  # the food sits on a counter in the gate's north corner: too high to
#   step onto (> 24), so he only presses against it, and an item above a scrolling floor is not carried. Pressed
#   against the gate on the south wall, he reaches the food after 2*HW - COUNTER_D - 36 = 80 units north
MOAT, PEN_LEN, PEN_EXTRA, DEMON_AT = 64, 1024, 256, 704   # the pen lies north of the loom platform, behind a grate
PIT_DEPTH, PIT_W = 192, 384                     # the loom platform's south edge is a ledge into a pit (behind him)
FACING = 90                                     # he faces north (degrees) the whole time
LOOM_R, HALF_FOV = 512, 8192                    # LC4: demon within 512 units and +-45 degrees (8192/65536 turn)
# --- body (line specials on his TID, crossed by the output neurons' spike dolls) ------------------------------
EAT_ANGLE, EAT_FORCE = 64, 4                    # north = forward, toward the food counter (~31 units per lunge)
ESC_ANGLE, ESC_FORCE, ESC_Z = 192, 8, 48        # south = backward, away from the demon (off the ledge), and up
QUAKE_ARGS = (3, 12, 0, 2)                      # intensity, duration (tics), damage radius 0, tremor radius 2*64
DUST_LIGHT, DUST_FADE = 0xB09060, 0x604828      # the dusty patch's brown haze (UDMF light/fade colour)
OUTPUTS = {"MN9_left": "eat", "MN9_right": "eat", "GF_1": "escape", "GF_2": "escape", "aDN1_right": "groom",
           "aDN1_left": "groom"}


@dataclass(frozen=True)
class Station:
    kind: str           # "sugar" | "bitter" | "dust" | "loom"
    max_steps: int      # his stay ends after this many steps at the latest ...
    linger: int         # ... or once its stimulus source has been gone this many steps (food eaten, dust gone, in pit)


DAY = (Station("sugar", 90, 4), Station("dust", 120, 4), Station("bitter", 60, 4), Station("loom", 90, 6))
STATION_FLAT = {"sugar": "FLAT5_4", "bitter": "NUKAGE1", "dust": "FLOOR6_2", "loom": "FLOOR0_1"}


@dataclass
class LoopSpec:
    """Everything Python needs to drive and check a compiled loop map (saved next to the WAD as .loop.json)."""
    wad: str                              # repo-relative (absolute outside the repository)
    seed: int
    period: int
    calibration: dict
    stations: list[dict]
    groups: list[str]                     # group g <-> mask bit g
    line_group: list[int]                 # input line -> group index (-1: none)
    thresholds: list[int]                 # per group
    outputs: dict[str, str]               # output neuron -> action kind
    status: list[str]                     # USER1.. order of the STATUS readout
    schedule: dict = field(default_factory=dict)
    n_dolls: int = 0
    n_sectors: int = 0

    @property
    def step_tics(self) -> int:
        return REWIND + WRITE + self.period + HOLD + WORLD

    @property
    def wad_path(self) -> Path:
        return ROOT / self.wad

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> "LoopSpec":
        return cls(**json.loads(Path(path).read_text()))


def groups_of(net: IntNet) -> tuple[list[str], list[int], list[int]]:
    names = list(net.meta["input_groups"])
    line_group = [-1] * len(net.inputs)
    for g, name in enumerate(names):
        for m in net.meta["input_groups"][name]:
            line_group[m] = g
    p = net.meta["input_p_per_step"]
    thr = [threshold(p[name] if isinstance(p, dict) else p) for name in names]
    return names, line_group, thr


def actions_for(net: IntNet, stations) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Output neuron -> line specials its spike doll crosses (aimed at PLAYER_TID or the dust), + the action counter.
    Grooming removes the dust motes and clears the dusty patch's haze (so Johnston's organ input stops)."""
    clear = [s for k, st in enumerate(stations) if st.kind == "dust"
             for s in (dict(special=SET_COLOR, arg0=ZONE_TAG0 + k, arg1=255, arg2=255, arg3=255, arg4=0),
                       dict(special=SET_FADE, arg0=ZONE_TAG0 + k, arg1=0, arg2=0, arg3=0))]
    acts, outs = {}, {}
    for k, name in enumerate(n for n in OUTPUTS if n in net.names):
        kind = OUTPUTS[name]
        body = {"eat": [dict(special=THRUST, arg0=EAT_ANGLE, arg1=EAT_FORCE, arg2=0, arg3=PLAYER_TID)],
                "escape": [dict(special=THRUST, arg0=ESC_ANGLE, arg1=ESC_FORCE, arg2=0, arg3=PLAYER_TID),
                           dict(special=THRUST_Z, arg0=PLAYER_TID, arg1=ESC_Z, arg2=0, arg3=0)],
                "groom": [dict(special=QUAKE, arg0=QUAKE_ARGS[0], arg1=QUAKE_ARGS[1], arg2=QUAKE_ARGS[2],
                               arg3=QUAKE_ARGS[3], arg4=PLAYER_TID),
                          dict(special=THING_REMOVE, arg0=DUST_TID)] + clear}[kind]
        acts[name] = body + [dict(special=EXEC_ALWAYS, arg0=COUNT, arg1=0, arg2=k)]
        outs[name] = kind
    return acts, outs


# --- the level ----------------------------------------------------------------------------------------------------
def build_level(m, stations) -> dict:
    """Walkway, stations, demon pen and pit at y ~ Y0 (the circuit is at y >= 0). He faces north throughout: the food
    counters and the demon pen are in front of him, the pit behind him; the walkway carries him sideways (east)."""
    room = dict(ceil=CEIL_H, light=192, ctex="CEIL3_5", wall="STARTAN3")
    ys, yn = Y0 - HW, Y0 + HW
    x = X0
    m.rect(x, ys, x + APPROACH, yn, floor=0, tag=WALK_TAG, ftex="FLOOR5_1", **room)
    x += APPROACH
    info = dict(stations=[])
    for k, st in enumerate(stations):
        zone = ZONE_TAG0 + k
        if st.kind == "loom":         # end of the line: a platform behind a grate, a pen in front, a pit behind
            xe = x + STATION_LEN
            plat = m.rect(x, ys, xe, yn, floor=0, tag=zone, ftex=STATION_FLAT["loom"], **room)
            moat = m.rect(x, yn, xe, yn + MOAT, floor=0, ftex="FLAT20", **room)
            pen = m.rect(x - PEN_EXTRA, yn + MOAT, xe + PEN_EXTRA, yn + MOAT + PEN_LEN, floor=0, ftex="FLOOR7_1",
                         **room)
            for r, side in ((plat, "n"), (moat, "s"), (moat, "n"), (pen, "s")):
                m.edge(r, side, line=dict(blocking=True, dontpegbottom=True), sidedef=dict(texturemiddle="MIDGRATE"))
            m.rect(x, ys - PIT_W, xe, ys, floor=-PIT_DEPTH, tag=PIT_TAG, ftex="RROCK09", **room)   # off the ledge
            m.thing((x + xe) // 2, yn + MOAT + DEMON_AT, 3002, angle=270, id=DEMON_TID, dormant=True)
            info["stations"].append(dict(kind=st.kind, x0=x, x1=xe, grate=yn, pit=[x, ys - PIT_W, xe, ys]))
            break
        xg = x + STATION_LEN
        flat = STATION_FLAT[st.kind]
        if st.kind in ("sugar", "bitter"):
            yc = yn - COUNTER_D
            m.rect(x, ys, xg, yc, floor=0, tag=zone, ftex=flat, **room)
            m.rect(x, yc, xg - COUNTER_W, yn, floor=0, tag=zone, ftex=flat, **room)
            m.rect(xg - COUNTER_W, yc, xg, yn, floor=COUNTER_H, ftex="FLAT1", facing="WOOD1", **room)
            m.thing(xg - COUNTER_W // 2, yc + 16, 2014, id=FOOD_TID0 + k)       # health bonus = the food
        else:
            haze = dict(sector=dict(lightcolor=DUST_LIGHT, fadecolor=DUST_FADE)) if st.kind == "dust" else {}
            m.rect(x, ys, xg, yn, floor=0, tag=zone, ftex=flat, **haze, **room)
        m.rect(xg, ys, xg + GATE_W, yn, floor=GATE_H, tag=GATE_TAG0 + k, ftex="FLAT23", facing="SUPPORT3", **room)
        if st.kind == "dust":
            for dx, dy, h in [(-40, 40, 40), (-80, 10, 56), (-120, 60, 30), (-60, -50, 64), (-150, -20, 48),
                              (-30, 60, 72), (-100, -60, 36), (-180, 30, 60)]:
                m.thing(xg + dx, Y0 + dy, 9001, id=DUST_TID, height=h)            # dust in the air (MapSpots)
        info["stations"].append(dict(kind=st.kind, x0=x, gate=xg))
        x = xg + GATE_W
        m.rect(x, ys, x + APPROACH, yn, floor=0, tag=WALK_TAG, ftex="FLOOR5_1", **room)
        x += APPROACH
    else:
        m.rect(x, ys, x + 128, yn, floor=0, ftex="FLOOR5_1", **room)       # end of the line (no loom station)
    m.scroll_carry(WALK_TAG, **WALK)
    for k, st in enumerate(stations):
        m.scroll_carry(ZONE_TAG0 + k, **PUSH)
        if st.kind != "loom":
            m.scroll_carry(GATE_TAG0 + k, **PUSH)
    m.thing(X0 + 48, Y0, 1, angle=FACING)  # LAST player-1 start: the marine; every earlier one is a voodoo doll of him
    return info


def loop_acs(stations, groups: list[str], line_group: list[int], thr: list[int], port_tags: list[int],
             n_out: int, seed: int, period: int | None) -> tuple[str, list[str]]:
    """World ACS: ENTER (his TID), the action counter, the STATUS readout and (period given) the CLOCK."""
    bit = {g: 1 << i for i, g in enumerate(groups)}
    kinds = [st.kind for st in stations]
    code = {"sugar": 0, "bitter": 1, "dust": 2, "loom": 3}
    nst = len(stations)

    def arr(name, vals):
        return f"int {name}[{max(1, len(vals))}] = {{ {', '.join(str(int(v)) for v in vals) or '0'} }};"

    zone_sense, food_sense = [], ""
    if "sugar" in bit:                     # at a food station while the food still exists
        food_sense = (f"if (kind == 0 || kind == 1) {{ if (ThingCount(T_NONE, {FOOD_TID0} + j) > 0) {{ "
                      f"if (fl_in({ZONE_TAG0} + j)) mask |= {bit['sugar']}; }} }}")
    if "bitter" in bit:                    # standing in the nukage pool
        zone_sense.append(f"if (kind == 1) mask |= {bit['bitter']};")
    if "jo" in bit:                        # standing in the dusty patch while there is dust left
        zone_sense.append(f"if (kind == 2) {{ if (ThingCount(T_NONE, {DUST_TID}) > 0) mask |= {bit['jo']}; }}")
    lc4 = f"if (fl_demon) {{ if (fl_looming()) mask |= {bit['lc4']}; }}" if "lc4" in bit else ""
    status = (["step", "phase", "mask", "seed", "rng", "station", "wait", "gone", "done", "demon"]
              + [f"act{k}" for k in range(n_out)] + [f"food{k}" for k in range(nst)]
              + ["dust", "demon_hp", "demon_dx", "demon_dy", "sight", "pit", "zone"])
    exprs = (["fl_step", "fl_phase", "fl_mask", "fl_seed", "fl_rng", "fl_station", "fl_wait", "fl_gone", "fl_done",
              "fl_demon"] + [f"fl_act[{k}]" for k in range(n_out)]
             + [f"ThingCount(T_NONE, {FOOD_TID0 + k})" for k in range(nst)]
             + [f"ThingCount(T_NONE, {DUST_TID})", f"GetActorProperty({DEMON_TID}, APROP_Health)",
                f"(GetActorX({DEMON_TID}) - GetActorX({PLAYER_TID})) >> 16",
                f"(GetActorY({DEMON_TID}) - GetActorY({PLAYER_TID})) >> 16",
                f"CheckSight({PLAYER_TID}, {DEMON_TID}, 0)", f"fl_in({PIT_TAG})", "fl_zone()"])
    if len(status) > PAGE:
        raise ValueError("status readout longer than one page")
    acs = f"""
int fl_step; int fl_phase; int fl_mask; int fl_seed; int fl_rng = {int(seed)};
int fl_station; int fl_wait; int fl_gone; int fl_done; int fl_demon; int fl_arrived;
{arr('fl_act', [0] * n_out)}
{arr('fl_tag', port_tags)}
{arr('fl_grp', line_group)}
{arr('fl_thr', thr)}
{arr('fl_kind', [code[k] for k in kinds])}
{arr('fl_max', [st.max_steps for st in stations])}
{arr('fl_linger', [st.linger for st in stations])}

function int fl_lehmer (int x) {{
  int hi = x / {SCHRAGE_Q};
  int lo = x % {SCHRAGE_Q};
  x = {LEHMER_A} * lo - {SCHRAGE_R} * hi;
  if (x <= 0) x += {M31};
  return x;
}}
function int fl_in (int tag) {{ return ThingCountSector(T_NONE, {PLAYER_TID}, tag) > 0; }}
function int fl_zone (void) {{
  for (int j = 0; j < {nst}; j++) {{ if (fl_in({ZONE_TAG0} + j)) return j + 1; }}
  return 0;
}}
function int fl_looming (void) {{
  if (ThingCount(T_NONE, {DEMON_TID}) == 0) return 0;
  if (GetActorProperty({DEMON_TID}, APROP_Health) <= 0) return 0;
  if (!CheckSight({PLAYER_TID}, {DEMON_TID}, 0)) return 0;
  int dx = (GetActorX({DEMON_TID}) - GetActorX({PLAYER_TID})) >> 16;
  int dy = (GetActorY({DEMON_TID}) - GetActorY({PLAYER_TID})) >> 16;
  if (dx >= {LOOM_R} || -dx >= {LOOM_R} || dy >= {LOOM_R} || -dy >= {LOOM_R}) return 0;
  if (dx * dx + dy * dy >= {LOOM_R * LOOM_R}) return 0;
  int a = (VectorAngle(dx << 16, dy << 16) - GetActorAngle({PLAYER_TID})) & 65535;
  return a < {HALF_FOV} || a > {65536 - HALF_FOV};
}}
function int fl_sense (void) {{
  int mask = 0;
  for (int j = 0; j < {nst}; j++) {{
    int kind = fl_kind[j];
    {food_sense}
    if (fl_in({ZONE_TAG0} + j)) {{ {' '.join(zone_sense)} }}
  }}
  {lc4}
  return mask;
}}
function int fl_source_gone (int k) {{
  int kind = fl_kind[k];
  if (kind == 0 || kind == 1) return ThingCount(T_NONE, {FOOD_TID0} + k) == 0;
  if (kind == 2) return ThingCount(T_NONE, {DUST_TID}) == 0;
  return fl_in({PIT_TAG});
}}
function void fl_world_start (void) {{
  if (fl_station < {nst}) {{
    int k = fl_station;
    if (!fl_arrived && fl_in({ZONE_TAG0} + k)) fl_arrived = 1;
    if (fl_arrived) {{
      if (fl_kind[k] == 3) fl_demon = 1;
      fl_wait++;
      if (fl_source_gone(k)) fl_gone++;
      if (fl_wait >= fl_max[k] || fl_gone >= fl_linger[k]) {{
        if (fl_kind[k] != 3) Floor_MoveToValue({GATE_TAG0} + k, 64, 0, 0);
        fl_station++;
        fl_arrived = 0; fl_wait = 0; fl_gone = 0;
        if (fl_station == {nst}) fl_done = 1;
      }}
    }}
  }}
  if (fl_demon) Thing_Activate({DEMON_TID});
  if (ThingCount(T_NONE, {DUST_TID}) > 0) SpawnSpot("BulletPuff", {DUST_TID}, 0, 0);
}}
function void fl_world_end (void) {{ if (fl_demon) Thing_Deactivate({DEMON_TID}); }}

script {ENTER} ENTER {{ Thing_ChangeTID(0, {PLAYER_TID}); }}
script {COUNT} (int k) {{ fl_act[k]++; }}
script {STATUS} (void) {{ {' '.join(f'o{i + 1} = {e};' for i, e in enumerate(exprs))} }}
"""
    if period is not None:
        acs += f"""
script {CLOCK} OPEN {{
  Delay({BOOT});
  while (TRUE) {{
    fl_phase = {PH_START};
    ACS_ExecuteAlways(3, 0);
    Delay({REWIND});
    fl_mask = fl_sense();
    fl_seed = fl_rng;
    for (int m = 0; m < {len(line_group)}; m++) {{
      fl_rng = fl_lehmer(fl_rng);
      int b = 0;
      int g = fl_grp[m];
      if (g >= 0) {{ if (fl_mask & (1 << g)) {{ if (fl_rng <= fl_thr[g]) b = 1; }} }}
      if (fl_tag[m]) ACS_ExecuteAlways(1, 0, fl_tag[m], b, b);
    }}
    Delay({WRITE});
    fl_phase = {PH_COMPUTE};
    ACS_ExecuteAlways(7, 0);
    Delay({period});
    fl_step++;
    fl_phase = {PH_HOLD};
    Delay({HOLD});
    fl_phase = {PH_WORLD};
    fl_world_start();
    Delay({WORLD});
    fl_world_end();
  }}
}}
"""
    return acs, status


def _compile(net, wad, stations, seed, period, port_tags):
    groups, line_group, thr = groups_of(net)
    acts, outs = actions_for(net, stations)
    holder = {}

    def world(m):
        holder["level"] = build_level(m, stations)
        acs, holder["status"] = loop_acs(stations, groups, line_group, thr, port_tags, len(outs), seed, period)
        return acs

    meta = build_spiking(net, wad, name=Path(wad).stem, world=world, actions=acts, guard=GUARD)
    return meta, holder


def input_port_tags(net: IntNet, meta) -> list[int]:
    """Port tag of input line m (0 if the map does not read it); only floor ports exist for inputs."""
    tags = []
    for m in range(len(net.inputs)):
        key = f"floor:-x{m}"
        if f"ceil:-x{m}" in meta.ports:
            raise ValueError("input ports are expected to be floor ports")
        tags.append(meta.ports.get(key, 0))
    return tags


def calibrate(net: IntNet, wad: Path, margin: float = 0.12, min_margin: int = 48) -> dict:
    """Worst-case settle of the compiled circuit (clock-free twin map): every neuron at theta (all spike, all
    positive-leak lanes pass) and every input line on, so every doll that can pass a gate does. Returns the tic of the
    last doll motion after the start barriers drop, and PERIOD = that + margin. Calibration only: this separate game's
    state is written by Python; the loop run never is."""
    r = TileRunner(wad)
    g = r.game
    try:
        tags = r.meta.layout["tags"]
        for i in range(net.n):
            for key, h in ((f"floor:v{i}", net.theta), (f"floor:-v{i}", -net.theta)):
                if key in tags:
                    g.send_game_command(f"puke -1 {tags[key]} {abs(h)} {int(h < 0)}")
        g.advance_action(3)
        ports = {key.split(":", 1)[1]: -1 for key in r.meta.ports}
        r.start(ports)
        last, prev, t = 0, None, 0
        while t - last < 96:
            g.advance_action(1)
            t += 1
            cur = tuple((o.position_x, o.position_y, o.position_z) for o in g.get_state().objects
                        if o.name == "DoomPlayer" and o.position_y > -1024)
            if cur != prev:
                last = t
            prev = cur
            if t > 50000:
                raise RuntimeError("calibration: dolls never stopped")
        _, s = read_step(r.read(), net.n)
    finally:
        r.close()
    return dict(worst_settle=last, spikes=int(s.sum()), period=last + max(min_margin, int(np.ceil(margin * last))))


def compile_flyloop(net: IntNet, wad: Path, stations=DAY, seed: int = 1, period: int | None = None) -> LoopSpec:
    """Two compiles of the same map: a clock-free twin (to learn the input port tags and to calibrate PERIOD), then
    the loop map with the ACS clock. The circuit's geometry is identical in both (same net, same actions)."""
    if not 1 <= seed < M31:
        raise ValueError("seed must be in 1..2^31-2")
    stations = tuple(stations)
    wad = Path(wad)
    twin = wad.with_name(wad.stem + "_cal.wad")
    meta0, _ = _compile(net, twin, stations, seed, None, [0] * len(net.inputs))
    tags = input_port_tags(net, meta0)
    cal = calibrate(net, twin) if period is None else dict(worst_settle=None, period=int(period))
    meta, holder = _compile(net, wad, stations, seed, cal["period"], tags)
    if meta.ports != meta0.ports:
        raise RuntimeError("the clock-free twin and the loop map disagree on port tags")
    if meta.read_tics + 2 > HOLD:                 # a status read before the pages and one after them
        raise ValueError(f"{meta.read_tics} readout tics and two status reads do not fit the {HOLD}-tic HOLD window")
    groups, line_group, thr = groups_of(net)
    _, outs = actions_for(net, stations)
    path = wad.resolve()
    spec = LoopSpec(wad=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path), seed=seed,
                    period=cal["period"], calibration=cal,
                    stations=[asdict(s) for s in stations], groups=groups, line_group=line_group, thresholds=thr,
                    outputs=outs, status=holder["status"], n_dolls=meta.n_dolls, n_sectors=meta.n_sectors,
                    schedule=dict(boot=BOOT, rewind=REWIND, write=WRITE, period=cal["period"], hold=HOLD,
                                  world=WORLD, level=holder["level"]))
    spec.save(wad.with_suffix(".loop.json"))
    return spec


# --- Python's side: the replica (reference) and the read-only driver --------------------------------------------
class Replica:
    """The in-map PRNG and input gating, replicated, feeding IntNet.step."""

    def __init__(self, net: IntNet, spec: LoopSpec):
        self.net = net
        self.line_group = np.array(spec.line_group)
        self.thr = list(spec.thresholds)
        self.state = spec.seed
        self.v = np.zeros(net.n, dtype=np.int64)

    def inputs(self, mask: int) -> tuple[np.ndarray, int, int]:
        """(x, PRNG state before, after) for one step with the given group mask."""
        before = self.state
        x = np.zeros(len(self.line_group), dtype=np.int64)
        for m, g in enumerate(self.line_group):
            self.state = lehmer(self.state)
            if g >= 0 and mask >> g & 1 and self.state <= self.thr[g]:
                x[m] = 1
        return x, before, self.state

    def step(self, mask: int):
        x, before, after = self.inputs(mask)
        self.v, s = self.net.step(self.v, x)
        return x, s, before, after


class LoopRun:
    """Drives a compiled loop map. Python's whole job: advance tics, run readout scripts, compare."""

    def __init__(self, spec: LoopSpec, net: IntNet, configure=None, resolution=None, automap: bool = False):
        self.spec, self.net = spec, net
        self.r = TileRunner(spec.wad_path, configure=configure, automap=automap,
                            resolution=resolution or vzd.ScreenResolution.RES_160X120)
        self.r.game = self.g = ReadoutOnly(self.r.game, readout_commands(len(self.r.meta.pages), f"puke -{STATUS}"))
        self.replica = Replica(net, spec)
        self.tic = 0                    # tics advanced since init
        self.k = 0                      # steps verified
        self.window = "compute"         # which window the clock is in while Python advances (for filming)
        self.stats = dict(steps=0, exact=0, prng_ok=0, settled=0, mismatches=[])
        self._hold: tuple[dict, int] | None = None     # (status, tic before it) of the HOLD window reached
        self._used: int | None = None                  # tics of the last HOLD window spent reading

    def advance(self, n: int, on_tic=None, every: int = 0):
        """Advance n tics; with on_tic, call it every `every` tics (filming)."""
        if n <= 0:
            return
        if on_tic is None or every <= 0:
            self.g.advance_action(n)
            self.tic += n
            return
        while n > 0:
            c = min(every, n)
            self.g.advance_action(c)
            self.tic += c
            n -= c
            on_tic(self)

    def status(self) -> dict:
        self.g.send_game_command(f"puke -{STATUS}")
        self.g.advance_action(1)
        self.tic += 1
        return {name: int(self.g.get_game_variable(USER[i])) for i, name in enumerate(self.spec.status)}

    def dolls(self):
        return tuple((o.position_x, o.position_y, o.position_z) for o in self.g.get_state().objects
                     if o.name == "DoomPlayer" and o.position_y > -1024)

    def to_hold(self, on_tic=None, every: int = 0, world_every: int = 0):
        """Advance to the next HOLD window, calling on_tic every `every` tics (every `world_every` tics in the WORLD
        window). The first call advances to the predicted first HOLD window and polls the status readout there."""
        if self._used is None:
            self.advance(BOOT + REWIND + WRITE + self.spec.period - 4, on_tic, every)
            for _ in range(64):
                st = self.status()
                if st["phase"] == PH_HOLD:
                    break
            else:
                raise RuntimeError("never reached the first HOLD window")
        else:
            rest = self.spec.step_tics - self._used
            world_from = HOLD - self._used                           # tics until the WORLD window opens
            if on_tic is not None and world_every:
                self.advance(world_from, on_tic, every)
                self.window = "world"
                self.advance(WORLD, on_tic, world_every)
                self.window = "compute"
                self.advance(rest - world_from - WORLD, on_tic, every)
            else:
                self.advance(rest, on_tic, every)
            st = self.status()
        self._hold = (st, self.tic - 1)

    def read(self, observe=None) -> dict:
        """Read the HOLD window reached by to_hold (results of step k) and check them against the reference.
        observe(self) runs first (read-only: game variables, objects) -> rec["obs"]."""
        st, t_hold = self._hold
        if st["phase"] != PH_HOLD or st["step"] != self.k + 1:
            raise RuntimeError(f"step {self.k}: expected HOLD of step {self.k + 1}, got {st}")
        obs = observe(self) if observe else None
        d0 = self.dolls()
        out = self.r.read()
        self.tic += self.r.meta.read_tics
        st2 = self.status()
        d1 = self.dolls()
        if st2["phase"] != PH_HOLD or st2["step"] != st["step"]:
            raise RuntimeError(f"step {self.k}: readout left the HOLD window ({st2})")
        v, s = read_step(out, self.net.n)
        x, s_ref, before, after = self.replica.step(st["mask"])
        prng_ok = before == st["seed"] and after == st["rng"]
        ok = bool(prng_ok and (s == s_ref).all() and (v == self.replica.v).all())
        settled = d0 == d1
        self.stats["steps"] += 1
        self.stats["exact"] += ok
        self.stats["prng_ok"] += prng_ok
        self.stats["settled"] += settled
        if not ok:
            bad_s = np.nonzero(s != s_ref)[0].tolist()
            bad_v = np.nonzero(v != self.replica.v)[0].tolist()
            self.stats["mismatches"].append(dict(step=self.k, prng_ok=prng_ok, spikes=bad_s[:10], potentials=bad_v[:10]))
        rec = dict(k=self.k, status=st, x=x, s=s, v=v, ok=ok, prng_ok=prng_ok, settled=settled, tic=t_hold, obs=obs)
        self.k += 1
        self._used = self.tic - t_hold
        return rec

    def step(self, on_tic=None, every: int = 0, world_every: int = 0, observe=None) -> dict:
        """to_hold + read: the next step's verified record."""
        self.to_hold(on_tic, every, world_every)
        return self.read(observe)

    def audit(self) -> dict:
        """Every console command Python sent; any outside the readout scripts raised at the call (forbidden)."""
        return self.g.audit()

    def close(self):
        self.r.close()


# --- the command-line run ------------------------------------------------------------------------------------------
_OBS = {"x": vzd.GameVariable.POSITION_X, "y": vzd.GameVariable.POSITION_Y, "z": vzd.GameVariable.POSITION_Z,
        "vx": vzd.GameVariable.VELOCITY_X, "vy": vzd.GameVariable.VELOCITY_Y, "vz": vzd.GameVariable.VELOCITY_Z,
        "health": vzd.GameVariable.HEALTH}
GAME_SEED = 1       # the engine's own RNG (the demon's AI): fixed, so every run is the same game


def _configure(g: vzd.DoomGame):
    for v in _OBS.values():
        g.add_available_game_variable(v)
    g.set_seed(GAME_SEED)
    g.set_render_hud(True)
    g.set_screen_format(vzd.ScreenFormat.CRCGCB)


def _observe(run) -> dict:
    return {k: round(run.g.get_game_variable(v), 2) for k, v in _OBS.items()}


def station_summary(rows: list[dict], spec: LoopSpec) -> list[str]:
    """One line per station: what he sensed, which output neurons spiked, what his body did."""
    lines = []
    names = list(spec.outputs)
    for j, st in enumerate(spec.stations):
        rs = [r for r in rows if r["station"] == j]
        here = [r for r in rs if r["zone"] == j + 1]
        if not here:
            lines.append(f"station {j + 1} {st['kind']:6s} not reached")
            continue
        sensed = {g: sum(g in r["senses"] for r in rs) for g in spec.groups}
        sensed = ", ".join(f"{g} {n}" for g, n in sensed.items() if n) or "-"
        spikes = {o: sum(r["out_spikes"].get(o, 0) for r in rs) for o in names}
        spikes = ", ".join(f"{o} {n}" for o, n in spikes.items() if n) or "none"
        after = [r for r in rs if r["k"] >= here[0]["k"]]
        north = max(r["obs"]["y"] for r in after) - here[0]["obs"]["y"]
        z_peak = max(r["z_peak"] for r in after)
        kind, over = st["kind"], rows[-1]["station"] > j       # over: he has moved on to the next station
        if kind in ("sugar", "bitter"):
            ate = any(r[f"food{j}"] == 0 for r in rs)
            did = f"{north:.0f} units north, " + ("ATE the health bonus" if ate else
                                                  "REFUSED (the bonus stays on its counter)" if over else
                                                  "the bonus is still on its counter")
        elif kind == "dust":
            did = "dust removed" if any(r["dust"] == 0 for r in rs) else "dust still there"
        else:
            did = f"peak height {z_peak:.0f}, " + ("ESCAPED into the pit" if any(r["pit"] for r in rs) else
                                                   "did not escape" if over else "not in the pit yet")
        lines.append(f"station {j + 1} {kind:6s} senses (steps) {sensed} | output spikes {spikes} | {did} | "
                     f"{len(rs)} step{'s' * (len(rs) != 1)}{'' if over else ' so far'}")
    return lines


def run_fly(*, seed: int = 1, steps: int = 40, net: IntNet | None = None, stations=DAY, period: int | None = None,
            gif=None, workdir=None, every: int = 10, out=say) -> dict:
    """The fly is alive: a 255-neuron FlyWire circuit, an in-map clock and the marine's level in one DOOM map.

    Each 1 ms model step the map senses, draws the input spikes with an in-map PRNG, runs the circuit and holds;
    output spikes thrust him. Python advances tics and runs readout scripts; after each step it compares every
    neuron's spike and potential with the integer reference fed by a replica of the PRNG (tolerance 0).
    net: an IntNet (default: the shipped FlyWire circuit, models/fly/brain.json); stations: the level's stations.
    Returns the totals."""
    import tempfile
    import time

    from doomformer import shipped

    net = net if net is not None else IntNet.load(shipped.data("fly/brain.json"))
    tmp = tempfile.TemporaryDirectory(prefix="doomformer-fly-") if workdir is None else None
    wad_dir = Path(workdir) if workdir else Path(tmp.name)
    wad_dir.mkdir(parents=True, exist_ok=True)
    wad = wad_dir / f"fly_s{seed}.wad"
    out("building the map and calibrating the step period on a clock-free twin map (about 10 s) ...")
    t0 = time.perf_counter()
    spec = compile_flyloop(net, wad, stations=stations, seed=seed, period=period)
    t_build = time.perf_counter() - t0
    out(f"one map: his level + {net.n} fly neurons ({spec.n_dolls:,} voodoo dolls of him, {spec.n_sectors:,} sectors); "
        f"calibration {spec.calibration}; step = {REWIND} rewind + {WRITE} write + {spec.period} compute + "
        f"{HOLD} hold + {WORLD} world = {spec.step_tics} tics; built in {t_build:.1f}s")
    trace = None
    if gif:
        from .gifout import GifTrace
        trace = GifTrace(wad, gif)
    run = LoopRun(spec, net, configure=_configure,
                  resolution=vzd.ScreenResolution.RES_320X240 if gif else vzd.ScreenResolution.RES_160X120)
    caption = {"text": ""}
    peak = {"z": -1e9, "n": 0}

    def tracker(r):
        """Read-only sampling while the clock runs: his peak height this step, and a GIF frame (thinned while he
        stands still)."""
        gv = vzd.GameVariable
        peak["z"] = max(peak["z"], r.g.get_game_variable(gv.POSITION_Z))
        peak["n"] += 1
        moving = abs(r.g.get_game_variable(gv.VELOCITY_X)) + abs(r.g.get_game_variable(gv.VELOCITY_Y)) > 0.6 \
            or r.window == "world"
        if trace is not None and (moving or peak["n"] % 4 == 0):
            st = r.g.get_state()
            trace.add(st.screen_buffer, [(o.position_x, o.position_y) for o in st.objects
                                         if o.name == "DoomPlayer" and o.position_y > -1024], caption["text"])

    rows = []
    acts_prev = {o: 0 for o in spec.outputs}
    action_mismatch = 0
    out_idx = {o: net.names.index(o) for o in spec.outputs}
    t_run = time.perf_counter()
    try:
        for _ in range(steps):
            peak["z"] = -1e9
            run.to_hold(on_tic=tracker, every=16 if trace else 8, world_every=4 if trace else 8)
            rec = run.read(_observe)
            st = rec["status"]
            senses = [g for i, g in enumerate(spec.groups) if st["mask"] >> i & 1]
            acts = {o: st[f"act{i}"] for i, o in enumerate(spec.outputs)}
            out_spikes = {o: int(rec["s"][i]) for o, i in out_idx.items() if rec["s"][i]}
            rows.append(dict(k=rec["k"], station=st["station"], zone=st["zone"], senses=senses,
                             out_spikes=out_spikes, obs=rec["obs"], dust=st["dust"], pit=st["pit"],
                             z_peak=peak["z"], **{f"food{j}": st[f"food{j}"] for j in range(len(spec.stations))}))
            for o in spec.outputs:
                action_mismatch += (acts[o] - acts_prev[o]) != out_spikes.get(o, 0)
            acts_prev = acts
            n_in = int(rec["x"].sum())
            caption["text"] = f"step {rec['k'] + 1}  {run.stats['exact']}/{run.stats['steps']} exact"
            if rec["k"] % every == 0 or out_spikes or not rec["ok"]:
                out(f"step {rec['k'] + 1:4d}  station {min(st['station'] + 1, len(spec.stations))}  senses "
                    f"{'+'.join(senses) or '-':12s} {n_in:3d} input spikes, {int(rec['s'].sum()):3d} neurons spike  "
                    f"output {out_spikes or '-'}  pos ({rec['obs']['x']:.0f},{rec['obs']['y']:.0f},"
                    f"{rec['obs']['z']:.0f}) health {rec['obs']['health']:.0f}  "
                    f"{'EXACT' if rec['ok'] else 'MISMATCH'} ({net.n} potentials + {net.n} spikes + PRNG state; "
                    f"{run.stats['exact']}/{run.stats['steps']}, {time.perf_counter() - t_run:.0f}s)")
            if st["done"]:
                break
    finally:
        audit = run.audit()
        run.close()
    wall = time.perf_counter() - t_run
    s = run.stats
    for line in station_summary(rows, spec):
        out(line)
    done = bool(rows) and rows[-1]["station"] == len(spec.stations)
    out(f"{s['exact']}/{s['steps']} steps EXACT ({2 * net.n + 1} values each: every spike, every potential, the PRNG "
        f"state; tolerance 0), PRNG {s['prng_ok']}/{s['steps']}, dolls at rest in the hold window "
        f"{s['settled']}/{s['steps']}, action lines vs output spikes: {action_mismatch} differ; "
        f"scenario complete: {done}")
    n_cmd = sum(audit["commands"].values())
    out(f"Python decisions in the loop: 0. Console commands Python sent: {n_cmd}, all readout scripts (non-readout: "
        f"{audit['non_readout_commands']}); it wrote no port, potential or input and moved nothing.")
    out(f"step = {spec.step_tics} tics (compute {spec.period}, calibrated worst settle "
        f"{spec.calibration.get('worst_settle')}); wall {wall:.0f}s, {wall / max(1, s['steps']):.2f}s per step")
    if trace is not None:
        path, size = trace.close()
        out(f"gif: {path} ({size / 1e6:.2f} MB, {len(trace.frames)} frames)")
    if tmp is not None:
        tmp.cleanup()
    return dict(steps=s["steps"], exact=s["exact"], prng_ok=s["prng_ok"], settled=s["settled"],
                action_mismatch=action_mismatch, audit=audit, complete=done, wall_s=wall,
                period=spec.period, step_tics=spec.step_tics, rows=rows)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m demos.flyloop",
        description="The fly is alive: a 255-neuron circuit derived from the FlyWire connectome and the marine's level in "
                    "one DOOM map. The map's clock senses, draws the input spikes, runs the circuit and holds; "
                    "output spikes thrust him. Python advances tics and reads, and checks every neuron's spike "
                    "and potential every step (tolerance 0).")
    ap.add_argument("--seed", type=int, default=1, help="seed of the in-map spike-train PRNG; 1 is the documented run")
    ap.add_argument("--steps", type=int, default=40,
                    help="1 ms model steps to run (default 40; the documented run is --steps 900 --seed 1, "
                         "which ends by itself after 169 steps)")
    ap.add_argument("--period", type=int, default=None, help="compute window in tics (default: measured on a twin map)")
    ap.add_argument("--gif", metavar="PATH", help="write a small looping GIF of the run (the game's frames and the dolls)")
    ap.add_argument("--workdir", help="where to keep the compiled maps (default: a temporary directory)")
    ap.add_argument("--every", type=int, default=10, help="print every N-th step (and every step with an output spike)")
    args = ap.parse_args(argv)
    if args.steps < 1 or args.every < 1 or (args.period is not None and args.period < 1):
        ap.error("--steps, --every and --period must be positive")
    r = run_fly(seed=args.seed, steps=args.steps, period=args.period, gif=args.gif, workdir=args.workdir,
                every=args.every)
    ok = (r["exact"] == r["steps"] == r["prng_ok"] == r["settled"] > 0 and not r["audit"]["non_readout_commands"]
          and r["action_mismatch"] == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
