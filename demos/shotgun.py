"""Shoot a neuron: the "shotgun" film.

A real first-person player stands in a viewing room attached to one compute tile (the room borders the
tile's lane area and the entry of the neuron's corridor through impassable windows, the player never
crosses a machine line). Shooting a switch on the room's wall runs ACS script SWITCH_SCRIPT, which slams
the zone doors of the ablated neuron's blocks. The same doors are what `DoomTransformer(ablate=...)`
closes in every consuming tile, so the filmed tile run must equal the logged ablated run bit for bit.

Pieces: ablation scan (`scan`), where a neuron is consumed (`tile_consumers`), picking the logged tile
execution to film (`choose_scenario`), the decorated tile map (`compile_film_tile`), replay proofs
(`replay`) and the camera/weapon driver (`Director`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from doomformer import intref
from doomformer.intref import QModel
from doomformer.model_tiles import tile_specs
from doomformer.runtime import USER, TileRunner
from doomformer.tile import DOOR_TAG0, LANE_W, MOVE_SPEED, TileMeta, TileSpec, compile_tile

SWITCH_SCRIPT = 210      # impact script on the switch: slams the neuron's doors
ENTER_SCRIPT = 211       # gives the real player the shotgun
POLL_SCRIPT = 212        # camera crew: o60 := ceiling height of the primary block's door
SWITCH_LINE_ID = 210     # linedef id of the switch (its texture flips when it is shot)
SLAM_DELAY = 44          # tics between the shot and the slam
SLAM_FROM = 160          # the door drops instantly from its open height to this ceiling, then slams
SLAM_SPEED = 120         # 1/8 unit per tic (15 units/tic)
ROOM_W = 72              # gap between a block's lane area and its zone entry
ROOM_H = 128             # ceiling above the room floor
NICHE_W, NICHE_D = 64, 16
OPEN_DOOR_TEX, TARGET_DOOR_TEX = "BIGDOOR2", "DOORRED2"
SWITCH_TEX, SWITCH_ON_TEX = "SW1COMP", "SW2COMP"


# ---------------------------------------------------------------------------- ablation bookkeeping
def tile_consumers(m: QModel, key: str) -> list[tuple[str, str]]:
    """(tile name, tile-local port prefix) of every tile whose gates read the ablated activation `key`;
    mirrors the `consumed` maps in DoomTransformer.step."""
    if key == "x0" or key.endswith(".x2"):
        l = 0 if key == "x0" else int(key[1:key.index(".")]) + 1
        if l >= m.L:
            return [("unembed", "in")]
        return [(f"L{l}.qkv", "in")] + [(f"L{l}.out_{t}", "in") for t in range(m.T)]
    l, kind = int(key[1:key.index(".")]), key.split(".", 1)[1]
    return {"ao": [(f"L{l}.out_{t}", "zo") for t in range(m.T)],
            "x1": [(f"L{l}.mlp_up", "z1"), (f"L{l}.mlp_dn", "z1")],
            "m": [(f"L{l}.mlp_dn", "zm")]}[kind]


def ablation_keys(m: QModel) -> dict[str, int]:
    """Every ablatable activation key -> number of units."""
    keys = {"x0": m.d}
    for l in range(m.L):
        keys |= {f"L{l}.ao": m.d, f"L{l}.x1": m.d, f"L{l}.m": m.dm, f"L{l}.x2": m.d}
    return keys


def scan(m: QModel, prompt: str = "", temperature: float = 0.0,
         seed: int = 0) -> tuple[str, list[tuple[str, int, str]]]:
    """Text without ablation and with every single-unit ablation (same temperature/seed):
    (base, [(key, unit, text)])."""
    gen = lambda ab: intref.generate(m, prompt, temperature=temperature, seed=seed, ablate=ab)
    base = gen(None)
    rows = [(k, u, gen({k: [u]})) for k, n in ablation_keys(m).items() for u in range(n)]
    return base, rows


def closed_ports(m: QModel, ablate: dict, tile: str) -> list[str]:
    """Tile-local port names of the ablated units that `tile` consumes (what the orchestrator closes)."""
    out = []
    for key, units in ablate.items():
        for t, local in tile_consumers(m, key):
            if t == tile:
                out += [f"{local}_{u}" for u in units]
    return out


def neuron_blocks(layout: dict, ports: list[str]) -> list[dict]:
    """Layout blocks whose lane port is one of the local port names (either sign)."""
    want = set(ports)
    return [b for b in layout["blocks"] if b["lane_port"].split(":", 1)[1].lstrip("-") in want]


def strip(inputs: dict) -> dict[str, int]:
    return {k: int(v) for k, v in inputs.items() if k != "__closed__"}


def dolls_released(spec: TileSpec, block: int, inputs: dict) -> int:
    """Dolls of one block that pass their ADC gate for these inputs (they reach the zone door)."""
    n = 0
    for g in spec.blocks[block].lanes:
        v = inputs[spec.ports[g.port].split(":", 1)[1]]
        n += int(v <= g.limit) if g.kind == "floor" else int(v >= g.limit)
    return n


@dataclass
class Scenario:
    tile: str
    entry: dict                   # logged execution of the ablated run: tile, inputs, outputs
    ports: list[str]              # ablated tile-local ports, e.g. ['in_15']
    blocks: list[int]             # block indices of the neuron (all signs)
    primary: int                  # block that releases the most dolls
    dolls: int                    # dolls released by the primary block
    base_entry: dict | None       # logged execution of the unablated run with identical inputs, if any
    position: int                 # index of the execution among that tile's executions in the run
    spec: TileSpec = field(repr=False, default=None)


def choose_scenario(m: QModel, ablate: dict, base_log: list[dict], abl_log: list[dict],
                    tile_dir: Path, tile: str | None = None, position: int | None = None) -> Scenario:
    """Pick the logged tile execution (of the ablated run) to film: the one releasing the most dolls into the
    neuron's corridors, preferring executions whose inputs also occur unchanged in the unablated run (then
    the same inputs are filmed with the door open and with the door closed)."""
    specs = {s.name: s for s in tile_specs(m)}
    consumers = {t for key in ablate for t, _ in tile_consumers(m, key)}
    base_by = {(e["tile"], tuple(sorted(strip(e["inputs"]).items()))): e for e in base_log}
    best, count = None, {}
    for e in abl_log:
        if e["tile"] not in consumers or (tile and e["tile"] != tile):
            continue
        t = e["tile"]
        pos = count[t] = count.get(t, -1) + 1
        if position is not None and pos != position:
            continue
        ports = closed_ports(m, ablate, t)
        meta = TileMeta.load(Path(tile_dir) / f"{t}.json")
        blocks = neuron_blocks(meta.layout, ports)
        ins = strip(e["inputs"])
        per = {b["index"]: dolls_released(specs[t], b["index"], ins) for b in blocks}
        if not per or not any(per.values()):
            continue
        prim = max(per, key=lambda i: (per[i], -i))
        twin = base_by.get((t, tuple(sorted(ins.items()))))
        score = (twin is not None, per[prim], -pos)
        if best is None or score > best[0]:
            best = (score, Scenario(t, e, ports, sorted(per), prim, per[prim], twin, pos, specs[t]))
    if best is None:
        raise RuntimeError(f"no logged execution releases dolls into the ablated neuron(s) {ablate}")
    return best[1]


# ---------------------------------------------------------------------------- the decorated tile
def _chain(blocks: list[dict], primary: dict) -> list[dict]:
    """Neuron blocks stacked directly above/below `primary` in the same column (one room spans them)."""
    col = [b for b in blocks if b["lanes"][0] == primary["lanes"][0]]
    chain, cur = [primary], primary
    while (nxt := next((b for b in col if b["lanes"][1] == cur["lanes"][3]), None)):
        chain.append(nxt)
        cur = nxt
    cur = primary
    while (prv := next((b for b in col if b["lanes"][3] == cur["lanes"][1]), None)):
        chain.insert(0, prv)
        cur = prv
    return chain


def make_decorate(sc: Scenario, info: dict):
    """decorate(builder, layout) -> ACS for compile_tile. Only adds rects/things/lines/scripts next to the
    machine and sets cosmetic textures; `info` receives the geometry (player, switch, door tags)."""
    def decorate(m, layout):
        blocks = {b["index"]: b for b in layout["blocks"]}
        prim = blocks[sc.primary]
        chain = _chain([blocks[i] for i in sc.blocks if i in blocks], prim)
        x_col, zone_x = prim["lanes"][2], prim["zone"][0]
        assert zone_x - x_col == ROOM_W, (zone_x, x_col)
        y0, y1 = chain[0]["lanes"][1], chain[-1]["lanes"][3]
        floor = prim["entry_floor"]
        m.rect(x_col, y0, zone_x, y1, floor=floor, ceil=floor + ROOM_H, light=208, ftex="FLOOR4_8",
               ctex="FLAT23", wall="COMPBLUE", blocking=True)
        # stand level with the dolls that will queue at the door (lanes that pass their ADC gate)
        px, py = x_col + 18, (prim["lanes"][1] + prim["lanes"][3]) / 2
        ins = strip(sc.entry["inputs"])
        lane_y = [prim["lanes"][1] + i * LANE_W + LANE_W / 2 for i, g in enumerate(sc.spec.blocks[sc.primary].lanes)
                  if ins[sc.spec.ports[g.port].split(":", 1)[1]] <= g.limit]
        if lane_y:
            py = min(max(sum(lane_y) / len(lane_y), y0 + 40), y1 - 40)
        # switch niche in the wall nearest to the player: shooting its back wall runs the script
        north = (y1 - py) <= (py - y0)
        nx0 = x_col + (ROOM_W - NICHE_W) // 2
        ny0, ny1 = (y1, y1 + NICHE_D) if north else (y0 - NICHE_D, y0)
        niche = m.rect(nx0, ny0, nx0 + NICHE_W, ny1, floor=floor, ceil=floor + ROOM_H, light=255, ftex="FLOOR4_8",
                       ctex="FLAT23", wall="COMPBLUE", blocking=True)
        m.edge(niche, "n" if north else "s",
               line=dict(special=226, arg0=SWITCH_SCRIPT, arg1=0, impact=True, id=SWITCH_LINE_ID),
               sidedef=dict(texturemiddle=SWITCH_TEX))
        m.thing(px, py, 1, angle=0)                       # last player-1 start = the real player
        m.thing(px, py, 2001)                             # shotgun
        tags = [b["door_tag"] for b in (blocks[i] for i in sc.blocks if i in blocks)]
        for r in m.rects:                                 # cosmetic: doors look like doors, the target is red
            if DOOR_TAG0 <= r.tag < DOOR_TAG0 + 10**4:
                r.facing = TARGET_DOOR_TEX if r.tag in tags else OPEN_DOOR_TEX
                r.ctex = "FLAT23"      # a sky ceiling would hide the slab: sky hack skips the upper wall
        info.update(player=(px, py), switch=((nx0 + NICHE_W / 2), ny1 if north else ny0), door_tags=tags,
                    room=(x_col, y0, zone_x, y1), floor=floor, primary=sc.primary)
        slam = "\n  ".join(f"Ceiling_MoveToValue({t}, {MOVE_SPEED}, {SLAM_FROM}, 0);" for t in tags)
        close = "\n  ".join(f"Ceiling_LowerToFloor({t}, {SLAM_SPEED});" for t in tags)
        return f"""
script {SWITCH_SCRIPT} (void) {{
  SetLineTexture({SWITCH_LINE_ID}, SIDE_FRONT, TEXTURE_MIDDLE, "{SWITCH_ON_TEX}");
  Delay({SLAM_DELAY});
  {slam}
  Delay(1);
  {close}
}}
script {POLL_SCRIPT} (void) {{ o60 = GetSectorCeilingZ({prim["door_tag"]}, 0, 0) >> 16; }}
script {ENTER_SCRIPT} ENTER {{
  GiveInventory("Shotgun", 1);
  GiveInventory("Shell", 40);
  SetWeapon("Shotgun");
}}
"""
    return decorate


def compile_film_tile(sc: Scenario, wad: Path) -> tuple[TileMeta, dict]:
    info: dict = {}
    meta = compile_tile(sc.spec, wad, decorate=make_decorate(sc, info))
    return meta, info


# ---------------------------------------------------------------------------- proofs
def replay(wad: Path, entries: list[dict], closed: set[int]) -> dict:
    """Re-run logged executions through a (decorated) tile map headless with the given doors shut and
    compare every output integer with the log."""
    r = TileRunner(Path(wad))
    try:
        if closed:
            r.set_doors(closed)
        bad, total = 0, 0
        for e in entries:
            out, _, _, _ = r.run(strip(e["inputs"]))
            total += len(out)
            bad += sum(1 for k, v in out.items() if e["outputs"][k] != v) + len(set(e["outputs"]) ^ set(out))
        return dict(entries=len(entries), values=total, mismatches=bad)
    finally:
        r.close()


# ---------------------------------------------------------------------------- the camera
def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class Director:
    """Drives the real player of a decorated tile: per-tic frames, turning, shooting, tile runs."""

    def __init__(self, wad: Path, emit, resolution=None, fov: float = 100.0, seed: int = 7):
        import vizdoom as vzd
        self.vzd = vzd
        self.emit = emit               # emit(frame HxWx3 uint8)
        self.buttons = [vzd.Button.ATTACK, vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.LOOK_UP_DOWN_DELTA]
        gv = vzd.GameVariable

        def configure(g):
            g.set_render_hud(True)
            g.set_render_weapon(True)
            g.set_render_crosshair(True)
            g.set_render_decals(True)
            g.set_render_particles(True)
            g.set_available_buttons(self.buttons)
            g.set_available_game_variables(USER + [gv.ANGLE, gv.POSITION_X, gv.POSITION_Y, gv.SELECTED_WEAPON,
                                                   gv.SELECTED_WEAPON_AMMO])
            g.set_seed(seed)

        self.runner = TileRunner(Path(wad), resolution=resolution or vzd.ScreenResolution.RES_1280X720,
                                 configure=configure)
        self.g = self.runner.game
        self.g.send_game_command(f"fov {fov:g}")
        self.tics = 0
        self.poll = False              # when set, every tic also polls the door ceiling into self.ceiling
        self.ceiling = None
        self.bounds = None             # (xmin, ymin, xmax, ymax) the real player has ever stood on
        for _ in range(12):            # the engine ignores mouse deltas during the first ~8 tics
            self.g.make_action([0, 0, 0], 1)
        self.turn_scale = 1.0

    # -- primitives
    def var(self, v):
        return self.g.get_game_variable(v)

    def tic(self, attack=0, turn=0.0, look=0.0):
        if self.poll:
            self.g.send_game_command(f"puke -{POLL_SCRIPT}")
        self.g.make_action([attack, turn, look], 1)
        self.tics += 1
        x, y = self.position()
        b = self.bounds or (x, y, x, y)
        self.bounds = (min(b[0], x), min(b[1], y), max(b[2], x), max(b[3], y))
        if self.poll:
            self.ceiling = int(self.var(USER[59]))
        frame = np.transpose(self.g.get_state().screen_buffer, (1, 2, 0))
        self.emit(frame)
        return frame

    def idle(self, n: int):
        for _ in range(n):
            self.tic()

    def angle(self) -> float:
        return float(self.var(self.vzd.GameVariable.ANGLE))

    def position(self) -> tuple[float, float]:
        gv = self.vzd.GameVariable
        return float(self.var(gv.POSITION_X)), float(self.var(gv.POSITION_Y))

    def calibrate(self):
        """How many degrees one unit of TURN_LEFT_RIGHT_DELTA turns (measured, not assumed)."""
        a0 = self.angle()
        self.tic(turn=10.0)
        da = wrap180(self.angle() - a0)
        if not da:
            raise RuntimeError("TURN_LEFT_RIGHT_DELTA has no effect")
        self.turn_scale = da / 10.0
        return self.turn_scale

    def turn_to(self, target: float, tics: int):
        """Ease the view to `target` degrees in `tics` tics (closed loop on the measured angle)."""
        start = self.angle()
        total = wrap180(target - start)
        for i in range(1, tics + 1):
            f = (1 - math.cos(math.pi * i / tics)) / 2
            want = start + total * f
            self.tic(turn=wrap180(want - self.angle()) / self.turn_scale)
        # settle residual error (quantisation)
        for _ in range(3):
            err = wrap180(target - self.angle())
            if abs(err) < 0.05:
                break
            self.tic(turn=err / self.turn_scale)

    def aim_at(self, x: float, y: float, tics: int):
        px, py = self.position()
        self.turn_to(math.degrees(math.atan2(y - py, x - px)) % 360, tics)

    def shoot(self, tics_after: int = 0):
        self.tic(attack=1)
        for _ in range(tics_after):
            self.tic()

    # -- tile runs with frames: TileRunner.start's state transitions, filmed tic by tic
    def start(self, inputs: dict[str, int]):
        self.runner.step = self.idle
        self.runner.start(inputs, reset=True)   # always reset: the map has already run for a while

    def settle(self, check_every: int = 8, max_tics: int | None = None, on_check=None) -> int:
        """TileRunner.settle, filmed tic by tic."""
        return self.runner.settle(check_every, max_tics, on_tick=on_check, step=self.idle)

    def run(self, inputs: dict[str, int], on_check=None) -> tuple[dict[str, int], int]:
        self.start(inputs)
        tics = self.settle(on_check=on_check)
        self.g.set_action([0] * len(self.buttons))
        return self.runner.read(), tics

    def doll_positions(self) -> list[tuple[float, float, float]]:
        return list(self.runner._dolls())

    def close(self):
        self.runner.close()
