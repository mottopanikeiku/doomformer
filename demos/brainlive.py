"""Live demons: the marine thinks while the demons keep coming, and what that costs him.

`python -m demos.brainlevel` freezes every demon (Thing_Deactivate) while his dolls compute, so a decision costs the
world nothing. This module removes the freeze and measures the result. The level's own clock samples the sensor once,
the dolls compute for `period` tics and hold for `hold` tics with every demon awake, and the action is applied to the
world as it is then (the sensing is stale by period + hold + WRITE tics).

Four ways to play the same arena, all through the same ACS (sensor, turn, pistol, life rules), differing only in who
chooses the action and how long the world runs while it is chosen:

  brain    his dolls compute the transformer and the argmax (the Stage-B brain of demos/brainlevel.py), demons awake;
           Python is readout-only (audit: 0 non-readout commands) and checks every exposed value at tolerance 0
  model    control: the same integer transformer in NumPy, in Python, with `delay` tics of latency, no dolls
  aimbot   control: the scripted expert (exact positions) with `delay` tics of latency, no dolls
  frozen   the model or aimbot control with the world stopped while it chooses (delay 0, as the frozen brain)

Lives are judged identically in every mode, from the per-tic record of the game (`Life`): a life ends at the first tic
at which health is 1 (buddha mode: he cannot die, so this is his death), the pistol is empty, or he has been awake
2100 tics. Kills are KILLCOUNT at that tic minus at the life's first tic; survival is awake tics until then.

Usage: python -m demos.brainlive brain --lives 20      (the dolls, live demons)
       python -m demos.brainlive model --delay 200 --lives 20
       python -m demos.brainlive sweep                 (the control sweep over delays and the references)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import vizdoom as vzd

from doomformer import policy
from doomformer.brainacs import PLAYER_TID, SCRIPT_PISTOL, TURN_UNITS, sensor_acs
from doomformer.intref import forward
from doomformer.onemap import (argmax_from_readout, argmax_reference, build_onemap, compare_trace,
                               logits_from_readout)
from doomformer.readout import ReadoutOnly, readout_commands
from doomformer.runtime import FREEDOOM2, USER, init_game
from doomformer.tile import MOVE_SPEED
from doomformer.udmf import MapBuilder, compile_acs, write_wad

from . import arena, say
from .brainlevel import (BOOT, CHOOSE, CLOCK, ENTER, NO_ACTION, PH_DONE, PH_HOLD, REWIND, STATUS, WRITE,
                         BrainRun, calibrate_period, stage_b_brain)

SETACT = 214
PH_COMPUTE, PH_WORLD, PH_SENSED = 2, 4, 6
GV = vzd.GameVariable
TIME_OFFSET = 1          # ViZDoom's episode time after a tic = the ACS Timer() of that tic + 1 (checked at every decision)


# --- the level's ACS -----------------------------------------------------------------------------------------------
def _world_phase() -> str:
    """What the level does with the chosen action: 4 tics of turning, or the trigger and hands off until the bullet
    is out. The same text as the frozen brain's clock (demos/brainlevel.clock_acs)."""
    return f"""
    bl_phase={PH_WORLD};
    int ammo0=CheckActorInventory({PLAYER_TID}, "Clip");
    if (bl_action == 2) GiveActorInventory({PLAYER_TID}, "ScriptFireToken", 1);
    int t=0;
    while (t < {policy.TICS_PER_DECISION} ||
           (bl_action == 2 && t < {arena.FOLLOW_MAX} &&
            CheckActorInventory({PLAYER_TID}, "Clip") == ammo0 && ammo0 > 0)) {{
      if (t < {policy.TICS_PER_DECISION}) {{
        if (bl_action == 0) SetActorAngle({PLAYER_TID}, (GetActorAngle({PLAYER_TID})+{TURN_UNITS})&65535);
        if (bl_action == 1) SetActorAngle({PLAYER_TID}, (GetActorAngle({PLAYER_TID})+65536-{TURN_UNITS})&65535);
      }}
      Delay(1); t++; bl_awake++;
    }}"""


def _life_check(lives: int, awake: str) -> str:
    return f"""
    if (GetActorProperty({PLAYER_TID}, APROP_Health) <= 1 ||
        CheckActorInventory({PLAYER_TID}, "Clip") <= 0 || {awake} >= {arena.LIFE_TICS}) {{
      if (bl_life >= {lives}) {{ bl_phase={PH_DONE}; terminate; }}
      ACS_ExecuteAlways({arena.NEW_LIFE}, 0);
      Delay(1);
      bl_awake=0; bl_life++; bl_life_t0=Timer();
      continue;
    }}"""


_PRELUDE = f"""
int bl_phase; int bl_step; int bl_action={NO_ACTION}; int bl_hold_start; int bl_awake;
int bl_life=1; int bl_compute_start; int bl_sample_t; int bl_life_t0;
script {ENTER} ENTER {{
  Thing_ChangeTID(0, {PLAYER_TID});
  GiveInventory("ScriptPistol", 1);
  SetWeapon("ScriptPistol");
}}
"""


def live_clock_acs(meta, period: int, hold: int, lives: int) -> str:
    """The brain's clock with the demons awake the whole time (compare demos/brainlevel.clock_acs, which deactivates
    them at the top of every cycle and wakes them after the hold). It samples the sensor once per cycle."""
    writes = []
    for key, tag in meta.ports.items():
        kind, name = key.split(":", 1)
        negative = name.startswith("-")
        t = int(name.lstrip("-").split(":")[1])
        expression = ("0", "bl_bearing", "bl_distance")[t]
        fn = "Floor_MoveToValue" if kind == "floor" else "Ceiling_MoveToValue"
        writes.append(f"{fn}({tag}, {MOVE_SPEED}, {expression}, {int(negative)});")
    return _PRELUDE + f"""
script {CHOOSE} (int action) {{ bl_action=action; }}
script {STATUS} (void) {{
  o1=bl_step; o2=bl_phase; o3=bl_bearing; o4=bl_distance; o5=bl_action;
  o6=bl_hold_start; o7=Timer(); o8=bl_life_t0;
  o9=bl_px; o10=bl_py; o11=bl_pa; o12=bl_life;
  o13=bl_compute_start; o14=bl_sample_t;
}}
script {CLOCK} OPEN {{
  Delay({BOOT});
  SetActivator({PLAYER_TID});
  ACS_ExecuteAlways({arena.NEW_LIFE}, 0);
  Delay(1);
  bl_life_t0=Timer(); arena_awake=1;
  while (TRUE) {{
    Delay(2);
    {_life_check(lives, "(Timer() - bl_life_t0)")}
    bl_phase={PH_COMPUTE};
    ACS_ExecuteAlways(3, 0);
    Delay({REWIND});
    bl_sense(); bl_sample_t=Timer();
    {' '.join(writes)}
    Delay({WRITE});
    bl_compute_start=Timer();
    bl_action={NO_ACTION};
    ACS_ExecuteAlways(7, 0);
    Delay({period});
    bl_step++; bl_phase={PH_HOLD}; bl_hold_start=Timer();
    Delay({hold});
    {_world_phase()}
  }}
}}
"""


def control_clock_acs(lives: int, delay: int, frozen: bool) -> str:
    """The same level without dolls: the clock samples the sensor, Python (script SETACT) names the action, and the
    level applies it once `delay` tics have passed since the sample (live: the demons keep moving meanwhile) or at
    once (frozen: the demons are dormant until the action arrives)."""
    awake = "bl_awake" if frozen else "(Timer() - bl_life_t0)"
    wait = (f"arena_awake=0; Thing_Deactivate({arena.DEMON_TID}); while (bl_action < 0) Delay(1); "
            f"Thing_Activate({arena.DEMON_TID}); arena_awake=1;" if frozen else
            f"while (bl_action < 0 || Timer() < bl_sample_t + {delay}) Delay(1);")
    top = f"arena_awake=0; Thing_Deactivate({arena.DEMON_TID});" if frozen else ""
    return _PRELUDE + f"""
script {SETACT} (int a) {{ if (bl_phase == {PH_SENSED}) bl_action = a; }}
script {STATUS} (void) {{
  o1=bl_step; o2=bl_phase; o3=bl_bearing; o4=bl_distance; o5=bl_action;
  o6=bl_life_t0; o7=Timer(); o8={awake};
  o9=bl_px; o10=bl_py; o11=bl_pa; o12=bl_life;
  o13=bl_sample_t;
}}
script {CLOCK} OPEN {{
  Delay({BOOT});
  SetActivator({PLAYER_TID});
  ACS_ExecuteAlways({arena.NEW_LIFE}, 0);
  Delay(1);
  bl_life_t0=Timer(); arena_awake={0 if frozen else 1};
  while (TRUE) {{
    {top}
    Delay(2);
    {_life_check(lives, awake)}
    bl_action={NO_ACTION};
    bl_sense(); bl_sample_t=Timer(); bl_step++; bl_phase={PH_SENSED};
    {wait}
    {_world_phase()}
  }}
}}
"""


SETQ = 215


def pipelined_clock_acs(lives: int, delay: int) -> str:
    """Control with a pure reaction delay and the frozen run's rate of actions: the level samples the sensor every 4
    tics, Python names the action of each sample (script SETQ), and the actuator starts, whenever it is free, the
    newest action whose sample is at least `delay` tics old (older ones are dropped). The marine acts as often as the
    frozen marine but on a world `delay` tics old; the brain cannot do this (one computation at a time)."""
    return _PRELUDE + f"""
int bl_qt[256]; int bl_qa[256]; int bl_consumed; int bl_last_sample = -100;
int bl_act_left; int bl_act_t; int bl_ammo0;
script {SETQ} (int slot, int a) {{ bl_qa[slot & 255] = a; }}
script {STATUS} (void) {{
  o1=bl_step; o2=bl_phase; o3=bl_bearing; o4=bl_distance; o5=bl_action;
  o6=bl_life_t0; o7=Timer(); o8=Timer() - bl_life_t0;
  o9=bl_px; o10=bl_py; o11=bl_pa; o12=bl_life;
  o13=bl_last_sample;
}}
script {CLOCK} OPEN {{
  Delay({BOOT});
  SetActivator({PLAYER_TID});
  ACS_ExecuteAlways({arena.NEW_LIFE}, 0);
  Delay(1);
  bl_life_t0=Timer(); arena_awake=1;
  while (TRUE) {{
    Delay(1);
    if (GetActorProperty({PLAYER_TID}, APROP_Health) <= 1 ||
        CheckActorInventory({PLAYER_TID}, "Clip") <= 0 || Timer() - bl_life_t0 >= {arena.LIFE_TICS}) {{
      if (bl_life >= {lives}) {{ bl_phase={PH_DONE}; terminate; }}
      ACS_ExecuteAlways({arena.NEW_LIFE}, 0);
      Delay(1);
      bl_life++; bl_life_t0=Timer(); bl_consumed=bl_step; bl_last_sample=-100; bl_act_left=0;
      continue;
    }}
    int now = Timer();
    if (bl_act_left > 0) {{
      if (bl_action == 0 && bl_act_t < {policy.TICS_PER_DECISION})
        SetActorAngle({PLAYER_TID}, (GetActorAngle({PLAYER_TID})+{TURN_UNITS})&65535);
      if (bl_action == 1 && bl_act_t < {policy.TICS_PER_DECISION})
        SetActorAngle({PLAYER_TID}, (GetActorAngle({PLAYER_TID})+65536-{TURN_UNITS})&65535);
      bl_act_t++;
      if (bl_act_t >= {policy.TICS_PER_DECISION} && (bl_action != 2 || bl_act_t >= {arena.FOLLOW_MAX} ||
          CheckActorInventory({PLAYER_TID}, "Clip") != bl_ammo0 || bl_ammo0 <= 0)) bl_act_left = 0;
    }}
    if (now - bl_last_sample >= {policy.TICS_PER_DECISION}) {{
      bl_sense();
      bl_qt[bl_step & 255] = Timer(); bl_qa[bl_step & 255] = {NO_ACTION};
      bl_sample_t = Timer(); bl_last_sample = Timer(); bl_step++;
    }}
    if (bl_act_left == 0) {{
      int i = bl_step - 1;
      while (i >= bl_consumed && i >= bl_step - 120) {{
        if (bl_qa[i & 255] >= 0 && bl_qt[i & 255] + {delay} <= now) {{
          bl_action = bl_qa[i & 255]; bl_consumed = i + 1; bl_act_left = 1; bl_act_t = 0;
          bl_ammo0 = CheckActorInventory({PLAYER_TID}, "Clip");
          if (bl_action == 2) GiveActorInventory({PLAYER_TID}, "ScriptFireToken", 1);
          i = -1;
        }}
        i--;
      }}
    }}
  }}
}}
"""


def build_control_wad(wad: Path, *, lives: int, delay: int, frozen: bool, pipelined: bool = False) -> Path:
    m = MapBuilder()
    acs = '#include "zcommon.acs"\n' + "\n".join(f"global int {i}:o{i};" for i in range(1, 61)) + arena.build(m)
    acs += sensor_acs(arena.DEMON_TID) + (
        pipelined_clock_acs(lives, delay) if pipelined else control_clock_acs(lives, delay, frozen))
    write_wad(wad, m.textmap(), compile_acs(acs), extra_lumps=[("DECORATE", SCRIPT_PISTOL.encode())])
    return wad


def build_live_wad(model, wad: Path, *, period: int, lives: int, opt: int):
    brain = stage_b_brain(model)
    planned = build_onemap(model, wad, world=arena.build, layout_only=True, opt=opt, **brain)
    hold = max(32, len(planned.pages) + 8)

    def world(builder):
        return arena.build(builder) + sensor_acs(arena.DEMON_TID) + live_clock_acs(planned, period, hold, lives)

    meta = build_onemap(model, wad, world=world, opt=opt, extra_lumps=[("DECORATE", SCRIPT_PISTOL.encode())], **brain)
    if (meta.ports, meta.pages, meta.layout["read_expressions"]) != (
            planned.ports, planned.pages, planned.layout["read_expressions"]):
        raise AssertionError("clock plan and emitted transformer differ")
    wad.with_suffix(".loop.json").write_text(json.dumps(dict(period=period, hold=hold, lives=lives, live=True),
                                                        indent=2))
    return meta


# --- the per-tic record and the life rules -------------------------------------------------------------------------
class Record:
    """Everything the host sees of every tic: (health, ammo, kills) and the enemies' positions. The host only reads."""

    def __init__(self):
        self.hist: dict[int, tuple[float, float, float, float]] = {}     # episode time -> health, ammo, kills, awake
        self.snaps: dict[int, tuple[tuple, list]] = {}                    # episode time -> pose, enemies
        self.keep = 1500

    def tic(self, g, awake: float = float("nan"), objects: bool = True):
        t = g.get_episode_time()
        self.hist[t] = (g.get_game_variable(GV.HEALTH), g.get_game_variable(GV.AMMO2),
                        g.get_game_variable(GV.KILLCOUNT), awake)
        if objects:
            state = g.get_state()
            pose = (g.get_game_variable(GV.POSITION_X), g.get_game_variable(GV.POSITION_Y),
                    g.get_game_variable(GV.ANGLE))
            foes = [(o.name, o.position_x, o.position_y) for o in state.objects if o.name in policy.ENEMIES] \
                if state is not None else []
            self.snaps[t] = (pose, foes)
            self.snaps.pop(t - self.keep, None)
        return t


def life_rows(rec: Record, starts: list[int], end_time: int | None = None, cap: int = arena.LIFE_TICS,
              awake_from_time: bool = False) -> list[dict]:
    """One row per complete life. starts: the episode time of each life's first tic (from the level's own life
    counter). A life ends at the first tic with health <= 1 ('dead'), ammo 0 ('ammo') or awake >= cap ('cap')."""
    rows = []
    times = sorted(rec.hist)
    for i, s in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else end_time
        if stop is None:
            break                                      # the last life has no recorded end
        k0 = rec.hist[s][2]
        for t in times:
            if t < s or t >= stop:
                continue
            health, ammo, kills, awake = rec.hist[t]
            alive = (t - s) if awake_from_time else awake
            reason = ("dead" if health <= 1 else "ammo" if ammo <= 0 else "cap" if alive >= cap else None)
            if reason:
                rows.append(dict(life=i + 1, kills=int(kills - k0), alive=int(alive), reason=reason))
                break
        else:
            rows.append(dict(life=i + 1, kills=int(rec.hist[max(t for t in times if t < stop)][2] - k0),
                             alive=int(stop - s), reason="open"))
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return dict(lives=0)
    kills = np.array([r["kills"] for r in rows])
    alive = np.array([r["alive"] for r in rows])
    reasons = {k: sum(r["reason"] == k for r in rows) for k in ("dead", "ammo", "cap", "open")}
    return dict(lives=n, kills_per_life=round(float(kills.mean()), 2), kills_sd=round(float(kills.std()), 2),
                kills_total=int(kills.sum()), alive_median=int(np.median(alive)), alive_mean=round(float(alive.mean()), 1),
                alive_min=int(alive.min()), alive_max=int(alive.max()), ended=reasons)


# --- controls: the integer policy or the aimbot, in Python, in the same arena ----------------------------------------
class ControlRun:
    def __init__(self, wad: Path, seed: int, resolution=vzd.ScreenResolution.RES_160X120):
        g = self.g = vzd.DoomGame()
        g.set_doom_game_path(FREEDOOM2)
        g.set_doom_scenario_path(str(wad))
        g.set_doom_map("MAP01")
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_screen_resolution(resolution)
        g.set_render_hud(False)
        g.set_objects_info_enabled(True)
        g.set_available_game_variables(USER)
        g.set_episode_timeout(10**9)
        g.set_mode(vzd.Mode.PLAYER)
        arena.configure(g)
        g.set_seed(seed)
        init_game(g)
        self.rec = Record()
        self.status_names = ("step", "phase", "bearing", "distance", "action", "life_t0", "timer", "awake",
                             "px", "py", "pa", "life", "sample_t")

    def poll(self, send: str | None = None) -> dict:
        """One tic: the commands (the choice made by Python, then the status readout) and the readout."""
        if send:
            self.g.send_game_command(send)
        self.g.send_game_command(f"puke -{STATUS}")
        self.g.advance_action(1)
        st = dict(zip(self.status_names, (int(self.g.get_game_variable(v)) for v in USER[:len(self.status_names)])))
        self.rec.tic(self.g, awake=st["awake"], objects=False)
        return st

    def play(self, kind: str, model, lives: int, max_tics: int = 10**7, pipelined: bool = False) -> dict:
        """Run until the clock has played `lives` lives. kind: 'model' (integer transformer on the sensed tokens) or
        'aimbot' (exact positions at the moment of the sample, a tic or two stale)."""
        st = self.poll()
        starts, answered, decisions, actions = [], -1, 0, []
        t_run = time.perf_counter()
        while st["phase"] != PH_DONE and self.g.get_episode_time() < max_tics:
            send = None
            if (st["step"] != answered and st["step"] > 0) if pipelined else \
                    (st["phase"] == PH_SENSED and st["step"] != answered):
                answered = st["step"]
                if kind == "model":
                    idx = policy.act_index(forward(model, [0, st["bearing"], st["distance"]])[-1])
                else:
                    idx = policy.expert_index(self.g.get_state(), self.g)
                actions.append(idx)
                send = f"puke -{SETQ} {st['step'] - 1} {idx}" if pipelined else f"puke -{SETACT} {idx}"
                decisions += 1
            st = self.poll(send)
            while len(starts) < st["life"] and st["life_t0"] > 0:
                starts.append(st["life_t0"] + TIME_OFFSET)
        end = self.g.get_episode_time()
        rows = life_rows(self.rec, starts, end_time=end + 1)
        rows = [r for r in rows if r["reason"] != "open"]
        return dict(rows=rows, decisions=decisions, actions=actions, tics=end, wall=time.perf_counter() - t_run)

    def close(self):
        self.g.close()


# --- the dolls, live ----------------------------------------------------------------------------------------------
class Observed(ReadoutOnly):
    """ReadoutOnly that steps every advance one tic at a time and records the tic (health, ammo, kills, enemies)."""

    def __init__(self, game, allowed):
        super().__init__(game, allowed)
        self.rec = Record()

    def advance_action(self, tics: int = 1, *rest):
        for _ in range(int(tics)):
            self._game.advance_action(1)
            self.rec.tic(self._game)


class LiveRun(BrainRun):
    """Readout-only host of the live clock; every decision is checked as in demos/brainlevel.ClockRun, and the
    sensed tokens are checked against the enemies' recorded positions at the sample tic."""

    NAMES = ("step", "phase", "bearing", "distance", "action", "hold_start", "timer", "life_t0",
             "px", "py", "pa", "life", "compute_start", "sample_t")

    def __init__(self, model, wad, *, seed, resolution):
        super().__init__(model, wad, seed=seed, resolution=resolution)
        self.r.game = self.g = Observed(self.g, readout_commands(len(self.r.meta.pages), f"puke -{STATUS}"))
        self.rec = self.g.rec
        self.schedule = json.loads(Path(wad).with_suffix(".loop.json").read_text())
        self.last_hold = None
        self.decisions = 0
        self.starts: list[int] = []
        self.holds: list[tuple[int, int]] = []         # (life number, episode time at which the action starts)
        self.behind = 0                                  # samples with the nearest demon exactly behind him

    def audit(self):
        return self.g.audit()

    def dolls(self):
        """The voodoo dolls' positions (the brain lies at y >= 0, the arena at y < 0: the marine, who slides when a
        demon bites him, is not a doll)."""
        return tuple((o.position_x, o.position_y, o.position_z) for o in self.g.get_state().objects
                     if o.name == "DoomPlayer" and o.position_y >= 0)

    def status(self) -> dict:
        self.g.send_game_command(f"puke -{STATUS}")
        self.g.advance_action(1)
        st = dict(zip(self.NAMES, (int(self.g.get_game_variable(v)) for v in USER[:len(self.NAMES)])))
        while len(self.starts) < st["life"] and st["life_t0"] > 0:
            self.starts.append(st["life_t0"] + TIME_OFFSET)
        return st

    def check_sample(self, st: dict):
        """The sensed tokens equal policy.observe on the enemies recorded at the sample tic, and the pose the level
        exported is the pose recorded at that tic (so the tic is the right one)."""
        t = st["sample_t"] + TIME_OFFSET
        if t not in self.rec.snaps:
            raise AssertionError(f"no record of the sample tic {t}")
        (px, py, pa), foes = self.rec.snaps[t]
        ex, ey, ea = st["px"] / 65536, st["py"] / 65536, st["pa"] * 360 / 65536
        if abs(px - ex) > 0.01 or abs(py - ey) > 0.01 or abs(((pa - ea) + 180) % 360 - 180) > 0.01:
            raise AssertionError(f"pose at tic {t}: recorded {(px, py, pa)}, level exported {(ex, ey, ea)}")
        state = SimpleNamespace(objects=[SimpleNamespace(name=n, position_x=x, position_y=y) for n, x, y in foes])
        coordinates = {GV.POSITION_X: ex, GV.POSITION_Y: ey, GV.ANGLE: ea}
        game = SimpleNamespace(get_game_variable=coordinates.__getitem__)
        observed = policy.observe(state, game)
        tokens = [0, st["bearing"], st["distance"]]
        if tokens != observed:
            near = policy.enemies(state, game)[:3]
            if near and abs(abs(near[0][1]) - 180) < 1e-6:
                # A demon exactly behind the marine is -180 degrees ('a') to policy.observe and +180 ('q') to the
                # level's sensor: the same direction. Accepted only for that case, and counted.
                other = policy.BEARING_CHARS[0 if near[0][1] > 0 else -1]
                if tokens == [0, policy.VOCAB.index(other), observed[2]]:
                    self.behind += 1
                    return len(foes)
            raise AssertionError(f"ACS tokens {tokens}, policy.observe on the enemies at tic {t}: {observed}; nearest "
                                 f"(distance, bearing) there: {[(round(d, 5), round(b, 4)) for d, b in near]}")
        return len(foes)

    def decide(self):
        period, hold = self.schedule["period"], self.schedule["hold"]
        target = (BOOT + period if self.last_hold is None else
                  self.last_hold + hold + policy.TICS_PER_DECISION + 2 + REWIND + WRITE + period)
        self.g.advance_action(max(1, target - self.g.get_episode_time()))
        for _ in range(256):
            status = self.status()
            if status["phase"] == PH_DONE:
                return None
            if status["phase"] == PH_HOLD and status["step"] == self.decisions + 1:
                break
        else:
            raise RuntimeError(f"missed ACS hold window: {status}")
        tokens = [0, status["bearing"], status["distance"]]
        n_foes = self.check_sample(status)
        before = self.dolls()
        got = self.r.read()
        after = self.status()
        if after["phase"] != PH_HOLD or after["step"] != status["step"] or self.dolls() != before:
            raise AssertionError(f"transformer readout was not a stable hold window: status {status} -> {after}, "
                                 f"dolls moved {self.dolls() != before}")
        trace = {}
        expected = forward(self.model, tokens, trace)
        best = policy.act_index(expected[-1])
        if argmax_reference(self.model, policy.ACTION_IDS, expected, trace) != best:
            raise AssertionError("comparator reference differs from np.argmax")
        if set(got) != set(self.r.meta.outputs):
            raise AssertionError("incomplete transformer readout")
        checked = compare_trace(got, trace, len(tokens))
        logits = logits_from_readout(self.model, got, len(tokens))
        np.testing.assert_array_equal(logits, expected)
        t, k = self.model.T - 1, len(policy.ACTION_IDS)
        winner = argmax_from_readout(got, k, t)
        action = status["action"]
        if not action == winner == best:
            raise AssertionError(f"level action {action}, dolls' winner {winner}, argmax {best}")
        self.decisions += 1
        self.last_hold = status["hold_start"]
        self.holds.append((status["life"], status["hold_start"] + hold + TIME_OFFSET))
        return dict(tokens=tokens, logits=logits[-1].tolist(), action=action, scalars=checked, status=status,
                    enemies=n_foes)

    def play(self):
        remaining = self.last_hold + self.schedule["hold"] - self.g.get_episode_time()
        if remaining > 0:
            self.g.advance_action(remaining)
        for _ in range(policy.TICS_PER_DECISION):
            self.g.advance_action(1)


def run_brain(*, seed: int = 4, lives: int = 5, opt: int = 3, model=None, workdir=None, out=say, period=None) -> dict:
    import tempfile

    from doomformer import shipped
    from doomformer.intref import QModel
    model = model if model is not None else QModel.load(shipped.resolve("doomplayer_onemap"))
    tmp = tempfile.TemporaryDirectory(prefix="doomformer-live-") if workdir is None else None
    wad_dir = Path(workdir) if workdir else Path(tmp.name)
    wad_dir.mkdir(parents=True, exist_ok=True)
    if period is None:
        settled, period = calibrate_period(model, wad_dir / "twin.wad", opt=opt)
        out(f"calibration on a clock-free twin (opt {opt}): the dolls settle in {settled} tics; compute budget {period}")
    wad = wad_dir / "live.wad"
    meta = build_live_wad(model, wad, period=period, lives=lives, opt=opt)
    hold = json.loads(wad.with_suffix(".loop.json").read_text())["hold"]
    out(f"one map, demons awake while his dolls compute: {meta.n_dolls:,} voodoo dolls, {meta.n_sectors:,} sectors; "
        f"the action lands {WRITE + period + hold} tics after the sensing (opt {opt}, period {period}, hold {hold})")
    run = LiveRun(model, wad, seed=seed, resolution=vzd.ScreenResolution.RES_160X120)
    t0 = time.perf_counter()
    exact = scalars = 0
    log = []
    try:
        while True:
            t_dec = time.perf_counter()
            d = run.decide()
            if d is None:
                break
            exact += 1
            scalars += d["scalars"]
            run.play()
            log.append(dict(life=d["status"]["life"], tokens=d["tokens"], action=d["action"]))
            out(f"life {d['status']['life']:2d} decision {exact:3d}  sensed {', '.join(policy.describe(d['tokens'])).lower()}"
                f" ({d['enemies']} enemies alive then)  his dolls chose {policy.ACTION_NAMES[d['action']]}  EXACT "
                f"({d['scalars']} values, tokens checked against the recorded enemies)  {time.perf_counter() - t_dec:.1f}s  "
                f"kills {int(run.g.get_game_variable(GV.KILLCOUNT))}")
    finally:
        audit = run.audit()
        end = run.g.get_episode_time()
        rec = run.rec
        run.close()
    rows = life_rows(rec, run.starts, end_time=end + 1, awake_from_time=True)
    rows = [r for r in rows if r["reason"] != "open"]
    for r in rows:                                    # actions that started while the life was still on
        r["actions"] = sum(1 for life, tic in run.holds
                           if life == r["life"] and tic - run.starts[r["life"] - 1] <= r["alive"])
    if run.behind:
        out(f"{run.behind} of the {exact} samples had the nearest demon exactly behind him (+180 degrees to the level, "
            "-180 to policy.observe: the same direction, accepted for that case only)")
    out(f"{exact} decisions EXACT ({scalars:,} values, tolerance 0); Python sent {sum(audit['commands'].values())} "
        f"commands, non-readout: {audit['non_readout_commands']}")
    if tmp is not None:
        tmp.cleanup()
    return dict(mode="brain", behind=run.behind, opt=opt, seed=seed, period=period, hold=hold, decisions=exact, exact=exact, scalars=scalars,
                audit=audit, rows=rows, summary=summarise(rows), wall=time.perf_counter() - t0, log=log)


def run_control(kind: str, *, seed: int, lives: int, delay: int = 0, frozen: bool = False, pipelined: bool = False,
                model=None, workdir=None) -> dict:
    import tempfile

    from doomformer import shipped
    from doomformer.intref import QModel
    if kind == "model" and model is None:
        model = QModel.load(shipped.resolve("doomplayer_onemap"))
    with tempfile.TemporaryDirectory(prefix="doomformer-ctl-") as tmp:
        wad = build_control_wad(Path(workdir or tmp) / f"ctl_{kind}_{delay}_{int(frozen)}{int(pipelined)}.wad",
                                lives=lives, delay=delay, frozen=frozen, pipelined=pipelined)
        run = ControlRun(wad, seed)
        try:
            res = run.play(kind, model, lives, pipelined=pipelined)
        finally:
            run.close()
    res.update(mode=kind, delay=delay, frozen=frozen, pipelined=pipelined, seed=seed, summary=summarise(res["rows"]))
    return res


SWEEP = ([("aimbot", dict(frozen=True), "scripted aimbot, world frozen while he decides")]
         + [("model", dict(frozen=True), "integer policy (what the frozen dolls compute), world frozen")]
         + [("aimbot", dict(delay=3), "aimbot, one decision at a time, latency 3")]
         + [("model", dict(delay=d), f"integer policy, one decision at a time, latency {d}")
            for d in (3, 50, 100, 200, 400, 773)]
         + [("model", dict(delay=d, pipelined=True), f"integer policy, an action every 4 tics on a sensing {d} tics old")
            for d in (3, 6, 9, 12, 16, 20, 25, 35, 50, 100, 200)]
         + [("aimbot", dict(delay=d, pipelined=True), f"aimbot, an action every 4 tics on a sensing {d} tics old")
            for d in (3, 9, 16, 25)])


def sweep(seeds, lives: int, out=say) -> list[dict]:
    table = []
    for kind, kw, label in SWEEP:
        rows = []
        for seed in seeds:
            rows += run_control(kind, seed=seed, lives=lives, **kw)["rows"]
        table.append(dict(label=label, kind=kind, **kw, seeds=list(seeds), summary=summarise(rows), rows=rows))
        sm = table[-1]["summary"]
        out(f"{label:75s} kills/life {sm['kills_per_life']:5.2f}  alive median {sm['alive_median']:4d} tics  "
            f"ended {sm['ended']}")
    return table


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m demos.brainlive", description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("brain", "model", "aimbot", "sweep"))
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 6], help="sweep: the seeds (lives each)")
    ap.add_argument("--lives", type=int, default=5)
    ap.add_argument("--opt", type=int, choices=(0, 1, 2, 3), default=3)
    ap.add_argument("--delay", type=int, default=3, help="control: tics between the sensing and the action")
    ap.add_argument("--frozen", action="store_true", help="control: the world stands still while the action is chosen")
    ap.add_argument("--pipelined", action="store_true",
                    help="control: an action every 4 tics, each chosen on a sensing `delay` tics old")
    ap.add_argument("--period", type=int, default=None)
    ap.add_argument("--workdir")
    ap.add_argument("--json", metavar="PATH", help="write the per-life rows and the summary")
    args = ap.parse_args(argv)
    if args.mode == "sweep":
        table = sweep(args.seeds, args.lives)
        if args.json:
            Path(args.json).write_text(json.dumps(table, indent=1))
        return 0
    if args.mode == "brain":
        res = run_brain(seed=args.seed, lives=args.lives, opt=args.opt, workdir=args.workdir, period=args.period)
    else:
        res = run_control(args.mode, seed=args.seed, lives=args.lives, delay=args.delay, frozen=args.frozen,
                          pipelined=args.pipelined, workdir=args.workdir)
    say(json.dumps(res["summary"]))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
