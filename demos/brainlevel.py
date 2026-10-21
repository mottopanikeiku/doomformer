"""A complete integer policy transformer and its marine's arena in one map.

Stage A leaves sensing, argmax, scheduling and buttons on the host. Stage B uses
the level's ACS clock and ScriptPistol, and the dolls also choose the action: after
the logits, comparator layers compute the exact argmax over L/R/F (onemap `choose`)
and the winner's doll runs ACS_ExecuteAlways(CHOOSE, 0, action), which only stores
it; ACS compares no logits. Python only advances tics and reads/checks (ReadoutOnly
raises on any other game call). Only token IDs enter the transformer; every exposed
scalar, including the comparator's, is checked independently.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import vizdoom as vzd
from PIL import Image, ImageDraw

from doomformer import policy

from . import arena
from doomformer.brainacs import PLAYER_TID, SCRIPT_PISTOL, TURN_UNITS, sensor_acs
from doomformer.intref import forward
from .learnviz import Board, FONTS, GAME_BOX, H, W
from doomformer.onemap import (argmax_from_readout, argmax_reference, build_onemap, compare_trace,
                               logits_from_readout, token_ports, value)
from doomformer.pure import EXEC_ALWAYS
from doomformer.readout import ReadoutOnly, readout_commands
from doomformer.runtime import TileRunner, USER
from doomformer.tile import MOVE_SPEED

BOOT, REWIND, WRITE = 70, 4, 3
CLOCK, STATUS, ENTER, CHOOSE = 210, 211, 212, 213
PH_COMPUTE, PH_HOLD, PH_WORLD, PH_DONE = 2, 3, 4, 5
NO_ACTION = -1


def choice_actions(model):
    """The winner's doll (one-hot output argz:T-1:a) hands action a to the level's ACS."""
    return {value("argz", model.T - 1, a): [dict(special=EXEC_ALWAYS, arg0=CHOOSE, arg1=0, arg2=a)]
            for a in range(len(policy.ACTION_IDS))}


def stage_b_brain(model):
    """build_onemap arguments of the Stage-B brain: the transformer plus the dolls' action argmax."""
    return dict(choose=policy.ACTION_IDS, actions=choice_actions(model))


def build_twin(model, wad: Path):
    """Clock-free calibration twin of the Stage-B map: the same brain in the bare arena (the winner lines call the
    absent script CHOOSE); TileRunner writes its ports and measures the settle tics."""
    return build_onemap(model, wad, world=arena.build, **stage_b_brain(model))


def build_brainlevel(model, wad: Path, *, layout_only=False, period=None, lives=1):
    if period is None:
        return build_onemap(model, wad, world=arena.build, layout_only=layout_only)
    if period <= 0:
        raise ValueError("the ACS compute period must be positive")
    brain = stage_b_brain(model)
    planned = build_onemap(model, wad, world=arena.build, layout_only=True, **brain)
    hold = max(32, len(planned.pages) + 8)

    def world(builder):
        return arena.build(builder) + sensor_acs(arena.DEMON_TID) + clock_acs(planned, period, hold, lives)

    meta = build_onemap(model, wad, world=world, layout_only=layout_only,
                        extra_lumps=[("DECORATE", SCRIPT_PISTOL.encode())], **brain)
    if (meta.ports, meta.pages, meta.layout["read_expressions"]) != (
            planned.ports, planned.pages, planned.layout["read_expressions"]):
        raise AssertionError("clock plan and emitted transformer differ")
    if not layout_only:
        wad.with_suffix(".loop.json").write_text(json.dumps(dict(period=period, hold=hold, lives=lives), indent=2))
    return meta


def clock_acs(meta, period, hold, lives):
    """The clock performs no transformer arithmetic and no comparison of logits: it senses, writes the token ports,
    releases the dolls, waits, and acts on the action the winner doll stored via script CHOOSE (NO_ACTION if none).
    It never reads the USER variables that Python's readout scripts fill."""
    writes = []
    for key, tag in meta.ports.items():
        kind, name = key.split(":", 1)
        negative = name.startswith("-")
        t = int(name.lstrip("-").split(":")[1])
        expression = ("0", "bl_bearing", "bl_distance")[t]
        fn = "Floor_MoveToValue" if kind == "floor" else "Ceiling_MoveToValue"
        writes.append(f"{fn}({tag}, {MOVE_SPEED}, {expression}, {int(negative)});")
    return f"""
int bl_phase; int bl_step; int bl_action={NO_ACTION}; int bl_hold_start; int bl_awake;
int bl_life=1; int bl_compute_start;
script {CHOOSE} (int action) {{ bl_action=action; }}
script {ENTER} ENTER {{
  Thing_ChangeTID(0, {PLAYER_TID});
  GiveInventory("ScriptPistol", 1);
  SetWeapon("ScriptPistol");
}}
script {STATUS} (void) {{
  o1=bl_step; o2=bl_phase; o3=bl_bearing; o4=bl_distance; o5=bl_action;
  o6=bl_hold_start; o7=Timer(); o8=bl_awake;
  o9=bl_px; o10=bl_py; o11=bl_pa; o12=bl_life;
  o13=bl_compute_start;
}}
script {CLOCK} OPEN {{
  Delay({BOOT});
  SetActivator({PLAYER_TID});
  ACS_ExecuteAlways({arena.NEW_LIFE}, 0);
  Delay(1);
  while (TRUE) {{
    arena_awake=0; Thing_Deactivate({arena.DEMON_TID});
    Delay(2);
    if (GetActorProperty({PLAYER_TID}, APROP_Health) <= 1 ||
        CheckActorInventory({PLAYER_TID}, "Clip") <= 0 || bl_awake >= {arena.LIFE_TICS}) {{
      if (bl_life >= {lives}) {{ bl_phase={PH_DONE}; terminate; }}
      ACS_ExecuteAlways({arena.NEW_LIFE}, 0);
      Delay(1);
      bl_awake=0; bl_life++;
      continue;
    }}
    bl_phase={PH_COMPUTE};
    ACS_ExecuteAlways(3, 0);
    Delay({REWIND});
    bl_sense();
    {' '.join(writes)}
    Delay({WRITE});
    bl_compute_start=Timer();
    bl_action={NO_ACTION};
    ACS_ExecuteAlways(7, 0);
    Delay({period});
    bl_step++; bl_phase={PH_HOLD}; bl_hold_start=Timer();
    Delay({hold});
    bl_phase={PH_WORLD};
    Thing_Activate({arena.DEMON_TID}); arena_awake=1;
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
    }}
  }}
}}
"""


class BrainRun:
    """Host-directed Stage A; the integer reference never supplies a circuit value."""

    def __init__(self, model, wad, *, seed, resolution):
        self.model = model

        def configure(game):
            arena.configure(game)
            game.set_seed(seed)

        self.r = TileRunner(wad, configure=configure, resolution=resolution)
        self.g = self.r.game

    def decide(self, on_think=None, think_every=0):
        self.g.set_action(arena.HANDS_OFF)
        self.g.send_game_command(f"puke -{arena.FREEZE}")
        self.g.advance_action(1)
        tokens = policy.observe(self.g.get_state(), self.g)
        trace = {}
        expected = forward(self.model, tokens, trace)
        if on_think is not None and think_every > 0:
            self.r.start(token_ports(self.model, tokens))
            on_think(tokens, 0, None)
            settled = self.r.settle(check_every=think_every,
                                    on_tick=lambda tic: on_think(tokens, tic, None))
            got = self.r.read()
        else:
            got, settled, _, _ = self.r.run(token_ports(self.model, tokens))
        if set(got) != set(self.r.meta.outputs):
            raise AssertionError("incomplete transformer readout")
        checked = compare_trace(got, trace, len(tokens))
        logits = logits_from_readout(self.model, got, len(tokens))
        np.testing.assert_array_equal(logits, expected)
        action = policy.act_index(logits[-1])
        if action != policy.act_index(expected[-1]):
            raise AssertionError("action differs from integer reference")
        return dict(tokens=tokens, logits=logits[-1].tolist(), action=action,
                    scalars=checked, settled_tics=settled)

    def close(self):
        self.r.close()


class ClockRun(BrainRun):
    """Readout-only host for the ACS loop, checked against policy.observe and intref."""

    def __init__(self, model, wad, *, seed, resolution):
        super().__init__(model, wad, seed=seed, resolution=resolution)
        self.r.game = self.g = ReadoutOnly(self.g, readout_commands(len(self.r.meta.pages), f"puke -{STATUS}"))
        self.schedule = json.loads(Path(wad).with_suffix(".loop.json").read_text())
        self.last_hold = None
        self.decisions = 0

    def audit(self):
        return self.g.audit()

    def status(self):
        self.g.send_game_command(f"puke -{STATUS}")
        self.g.advance_action(1)
        names = ("step", "phase", "bearing", "distance", "action", "hold_start", "timer", "awake",
                 "px", "py", "pa", "life", "compute_start")
        return dict(zip(names, (int(self.g.get_game_variable(v)) for v in USER[:len(names)])))

    def decide(self, on_think=None, think_every=0):
        period, hold = self.schedule["period"], self.schedule["hold"]
        if on_think is not None and think_every > 0:
            # Observe the new sensor sample before the dolls start moving. These
            # readouts consume tics inside the existing clock budget, never extend it.
            earliest = (BOOT + 1 + 2 + REWIND + WRITE if self.last_hold is None else
                        self.last_hold + hold + policy.TICS_PER_DECISION + 2 + REWIND + WRITE)
            self.g.advance_action(max(1, earliest - self.g.get_episode_time()))
            for _ in range(128):
                sample = self.status()
                if sample["phase"] == PH_DONE:
                    return None
                if sample["phase"] == PH_COMPUTE and sample["compute_start"] > (self.last_hold or 0):
                    break
            else:
                raise RuntimeError(f"missed ACS compute window: {sample}")
            start = sample["compute_start"]
            tokens = [0, sample["bearing"], sample["distance"]]
            on_think(tokens, sample["timer"] - start, period)
            for tic in range(think_every, period, think_every):
                # Timer() is one behind ViZDoom's episode time.
                remaining = start + tic + 1 - self.g.get_episode_time()
                if remaining > 0:
                    self.g.advance_action(remaining)
                    on_think(tokens, tic, period)
        target = (BOOT + period if self.last_hold is None else
                  self.last_hold + hold + policy.TICS_PER_DECISION + 2 + REWIND + WRITE + period)
        self.g.advance_action(max(1, target - self.g.get_episode_time()))
        for _ in range(128):
            status = self.status()
            if status["phase"] == PH_DONE:
                return None
            if status["phase"] == PH_HOLD and status["step"] == self.decisions + 1:
                break
        else:
            raise RuntimeError(f"missed ACS hold window: {status}")
        tokens = [0, status["bearing"], status["distance"]]
        # Damage thrust can move the player during compute. Verify the observation at
        # its actual sampling position, not the later HOLD position. Demons are dormant.
        coordinates = {vzd.GameVariable.POSITION_X: status["px"] / 65536,
                       vzd.GameVariable.POSITION_Y: status["py"] / 65536,
                       vzd.GameVariable.ANGLE: status["pa"] * 360 / 65536}
        snapshot = SimpleNamespace(get_game_variable=coordinates.__getitem__)
        observed = policy.observe(self.g.get_state(), snapshot)
        if tokens != observed:
            raise AssertionError(f"ACS tokens {tokens}, policy.observe {observed}")
        before = self.r._dolls()
        got = self.r.read()
        after = self.status()
        if after["phase"] != PH_HOLD or after["step"] != status["step"] or self.r._dolls() != before:
            raise AssertionError("transformer readout was not a stable hold window")
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
            scores = [int(expected[-1][i]) for i in policy.ACTION_IDS]
            raise AssertionError(f"level action {action}, dolls' winner {winner}, argmax of the L/R/F logits "
                                 f"{scores} is {best}")
        comparator = dict(d=[got[value("argd", t, a, b)] for a in range(k) for b in range(a + 1, k)],
                          wins=[got[value("argw", t, a)] for a in range(k)],
                          onehot=[got[value("argz", t, a)] for a in range(k)])
        self.decisions += 1
        self.last_hold = status["hold_start"]
        return dict(tokens=tokens, logits=logits[-1].tolist(), action=action, comparator=comparator,
                    scalars=checked, compute_budget_tics=period, status=status)

    def play(self, on_tic):
        """Advance through HOLD and the four action tics; never send a button or write."""
        remaining = self.last_hold + self.schedule["hold"] - self.g.get_episode_time()
        if remaining > 0:
            self.g.advance_action(remaining)
        for _ in range(policy.TICS_PER_DECISION):
            self.g.advance_action(1)
            on_tic()


def compose(view, board: Board, dolls, decision, stats, *, moving=()):
    """First-person view plus actual map linedefs and instrumented doll positions."""
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[GAME_BOX[1]:GAME_BOX[3], GAME_BOX[0]:GAME_BOX[2]] = np.asarray(
        Image.fromarray(view).resize((960, 720), Image.Resampling.NEAREST))
    board.draw(canvas, dolls)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for x, y in moving:
        point = board.pt(x, y)
        if point is not None:
            draw.rectangle([point[0] - 2, point[1] - 2, point[0] + 1, point[1] + 1], fill=(255, 255, 255))
    draw.text((40, 25), "HIS BRAIN IS THE LEVEL", font=FONTS["big"], fill=(255, 64, 64))
    draw.text((40, 85), f"one DOOM map | full policy transformer | {stats['dolls']:,} voodoo dolls of himself",
              font=FONTS["mid"], fill=(230, 230, 230))
    draw.text((1060, 145), "His transformer, in this same level", font=FONTS["small"], fill=(255, 176, 32))
    draw.text((1060, 180), "Red: actual dolls | white: moving now (since previous sample)",
              font=FONTS["tiny"], fill=(220, 220, 220))
    draw.text((40, 900), "SENSED TOKENS: " + " / ".join(policy.describe(decision["tokens"])),
              font=FONTS["mid"], fill=(220, 220, 220))
    if "thinking_tic" in decision:
        total = decision["thinking_total"]
        budget = f" / ~{total:,}" if total is not None else " / awaiting settle"
        draw.text((40, 945), f"THINKING: tic {decision['thinking_tic']}{budget} "
                  f"(time-lapse, 1 frame per {stats['think_every']} tics)",
                  font=FONTS["mid"], fill=(255, 176, 32))
    else:
        scores = [decision["logits"][i] for i in policy.ACTION_IDS]
        chooser = "HIS DOLLS CHOSE" if stats["stage"] == "B" else "CHOSEN"
        draw.text((40, 945), f"{chooser}: {policy.ACTION_NAMES[decision['action']]}     L / R / F logits: {scores}",
                  font=FONTS["mid"], fill=(255, 176, 32))
    role = ("Stage B: dolls compute the transformer and choose the action; ACS senses and acts; "
            "Python only advances / reads / verifies"
            if stats["stage"] == "B" else
            "Stage A: Python senses, selects argmax and presses buttons; DOOM computes the transformer")
    draw.text((40, 990), role, font=FONTS["small"], fill=(190, 190, 190))
    draw.text((40, 1035), f"seed {stats['seed']} | kills {stats['kills']} | "
              f"{stats['exact']}/{stats['decisions']} decisions bit-exact (all exposed intermediates + logits + action)",
              font=FONTS["small"], fill=(90, 220, 110))
    return np.asarray(image)
