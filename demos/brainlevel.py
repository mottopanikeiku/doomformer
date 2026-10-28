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

from . import arena, say
from doomformer.brainacs import PLAYER_TID, SCRIPT_PISTOL, TURN_UNITS, sensor_acs
from doomformer.intref import forward
from .learnviz import Board, FONTS, GAME_BOX, H, W
from doomformer.onemap import (argmax_from_readout, argmax_reference, build_onemap, compare_trace,
                               logits_from_readout, token_ports, value)
from doomformer.pure import DOOR_OPEN, EXEC_ALWAYS
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


def build_twin(model, wad: Path, opt: int = 0):
    """Clock-free calibration twin of the Stage-B map: the same brain in the bare arena (the winner lines call the
    absent script CHOOSE); TileRunner writes its ports and measures the settle tics."""
    return build_onemap(model, wad, world=arena.build, opt=opt, **stage_b_brain(model))


def build_brainlevel(model, wad: Path, *, layout_only=False, period=None, lives=1, opt=0, doors=None, lesion_at=None):
    """doors: {value name: tag} lane doors (doomformer.intervene.lane_doors); lesion_at: the decision (1-based) from
    which the level's own clock closes them, compiled into the clock script. Both or neither."""
    if (doors is None) != (lesion_at is None):
        raise ValueError("doors and lesion_at go together")
    if period is None:
        return build_onemap(model, wad, world=arena.build, layout_only=layout_only)
    if period <= 0:
        raise ValueError("the ACS compute period must be positive")
    brain = stage_b_brain(model)
    planned = build_onemap(model, wad, world=arena.build, layout_only=True, opt=opt, doors=doors, **brain)
    hold = max(32, len(planned.pages) + 8)

    def world(builder):
        return (arena.build(builder) + sensor_acs(arena.DEMON_TID)
                + clock_acs(planned, period, hold, lives, sorted(set((doors or {}).values())), lesion_at))

    meta = build_onemap(model, wad, world=world, layout_only=layout_only, opt=opt, doors=doors,
                        extra_lumps=[("DECORATE", SCRIPT_PISTOL.encode())], **brain)
    if (meta.ports, meta.pages, meta.layout["read_expressions"]) != (
            planned.ports, planned.pages, planned.layout["read_expressions"]):
        raise AssertionError("clock plan and emitted transformer differ")
    if not layout_only:
        wad.with_suffix(".loop.json").write_text(json.dumps(
            dict(period=period, hold=hold, lives=lives, **(dict(lesion_at=lesion_at) if doors else {})), indent=2))
    return meta


def clock_acs(meta, period, hold, lives, door_tags=(), lesion_at=None):
    """The clock performs no transformer arithmetic and no comparison of logits: it senses, writes the token ports,
    releases the dolls, waits, and acts on the action the winner doll stored via script CHOOSE (NO_ACTION if none).
    It never reads the USER variables that Python's readout scripts fill. door_tags, lesion_at: when the clock's own
    decision counter reaches lesion_at - 1 (the next decision is the lesion_at-th), right after the rewind has taken
    every doll home, it lowers the ceilings of these lane doors (closing a door over a doll that stands in it
    would stop at the doll); the trigger number is a constant of the script and Python never sends it."""
    writes = []
    for key, tag in meta.ports.items():
        kind, name = key.split(":", 1)
        negative = name.startswith("-")
        t = int(name.lstrip("-").split(":")[1])
        expression = ("0", "bl_bearing", "bl_distance")[t]
        fn = "Floor_MoveToValue" if kind == "floor" else "Ceiling_MoveToValue"
        writes.append(f"{fn}({tag}, {MOVE_SPEED}, {expression}, {int(negative)});")
    close = ""
    if door_tags:
        lower = " ".join(f"Ceiling_LowerByValue({t}, {MOVE_SPEED}, {DOOR_OPEN});" for t in door_tags)
        close = f"if (bl_doors == 0 && bl_step + 1 >= {lesion_at}) {{ bl_doors = 1; {lower} }}"
    return f"""
int bl_phase; int bl_step; int bl_action={NO_ACTION}; int bl_hold_start; int bl_awake;
int bl_life=1; int bl_compute_start; int bl_doors;
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
  o13=bl_compute_start; o14=bl_doors;
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
    {close}
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

    def __init__(self, model, wad, *, seed, resolution, lesion=None):
        super().__init__(model, wad, seed=seed, resolution=resolution)
        self.r.game = self.g = ReadoutOnly(self.g, readout_commands(len(self.r.meta.pages), f"puke -{STATUS}"))
        self.schedule = json.loads(Path(wad).with_suffix(".loop.json").read_text())
        self.last_hold = None
        self.lesion = lesion          # the units whose lane doors the level closes at schedule["lesion_at"] (or None)
        if bool(lesion) != ("lesion_at" in self.schedule):
            raise ValueError("lesion and the map's lesion_at go together")
        self.decisions = 0

    def audit(self):
        return self.g.audit()

    def status(self):
        self.g.send_game_command(f"puke -{STATUS}")
        self.g.advance_action(1)
        names = ("step", "phase", "bearing", "distance", "action", "hold_start", "timer", "awake",
                 "px", "py", "pa", "life", "compute_start", "doors")
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
        if self.lesion:     # the level's clock says whether it has closed the doors; that must match its own schedule
            if bool(status["doors"]) != (status["step"] >= self.schedule["lesion_at"]):
                raise AssertionError(f"doors flag {status['doors']} at decision {status['step']} (lesion_at "
                                     f"{self.schedule['lesion_at']})")
        expected = forward(self.model, tokens, trace, ablate=self.lesion if status["doors"] else None)
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
                    scalars=checked, compute_budget_tics=period, status=status, doors=bool(status["doors"]))

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


def calibrate_period(model, wad: Path, opt: int = 0) -> tuple[int, int]:
    """(settle tics, compute budget) of the Stage-B brain, measured on the clock-free twin map.

    The twin holds the same transformer and comparator dolls in the bare arena; Python writes one token context and
    waits for the dolls to stand still. The settle depth does not depend on the tokens (a few contexts give the same
    count), so one context suffices. The budget adds a margin of max(64, settle // 8) tics; the clock run raises if a
    decision's dolls are not at rest in the hold window, so a short budget cannot pass unnoticed."""
    build_twin(model, wad, opt)
    runner = TileRunner(wad, configure=arena.configure)
    try:
        settled = runner.run(token_ports(model, policy.all_contexts()[0]))[1]
    finally:
        runner.close()
    return settled, settled + max(64, settled // 8)


def run_brainlevel(*, seed: int = 4, decisions: int = 5, lives: int = 5, period: int | None = None,
                   model=None, gif=None, workdir=None, out=say, lesion: dict | None = None, opt: int = 0,
                   lesion_at: int | None = None) -> dict:
    """Stage B of the README's "his brain is the level": the level senses and acts, the dolls decide.

    Builds the policy transformer and the arena into one map (the dolls also take the argmax of the three action
    logits), then lets the map's own ACS clock run `decisions` decisions. Python advances tics and reads: every
    decision it compares the ACS-sensed tokens with `policy.observe`, every transformer scalar and logit with the
    integer reference, the comparator values and the dolls' winner with the reference argmax, and the action the level
    stored with both. model: a QModel with the marine policy vocabulary (default: the shipped doomplayer_onemap).
    Returns the totals. lesion with lesion_at=N: the lesion is a runtime event. The map holds the healthy brain plus a
    door in every lane that reads a lesioned unit; the level's own clock closes the doors before decision N (the number
    is compiled into the clock script), and from then on the reference is the masked model (doomformer.intervene
    lane_doors). Without lesion_at the lesion is folded into the weights (a different map)."""
    import tempfile
    import time

    from doomformer import shipped
    from doomformer.intref import QModel

    GV = vzd.GameVariable
    model = model if model is not None else QModel.load(shipped.resolve("doomplayer_onemap"))
    healthy_model = model
    if lesion and lesion_at is None:
        from doomformer.intervene import fold
        model = fold(model, ablate=lesion)
        out(f"lesion {lesion} compiled into the map's weights (doomformer.intervene.fold): the map computes the lesioned "
            "brain, the reference is the lesioned model")
    if lesion_at is not None and (not lesion or lesion_at < 1):
        raise ValueError("lesion_at needs a lesion and a decision number >= 1")
    doors = None
    if lesion_at is not None:
        from doomformer.intervene import lane_doors
        doors, door_tags = lane_doors(model, lesion)
    if model.vocab != policy.VOCAB or model.T != policy.CONTEXT_LEN:
        raise ValueError("model must use the marine policy vocabulary and three-token context")
    tmp = tempfile.TemporaryDirectory(prefix="doomformer-brainlevel-") if workdir is None else None
    wad_dir = Path(workdir) if workdir else Path(tmp.name)
    wad_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if period is None:
        out("calibrating the tic budget on a clock-free twin map (one transformer pass, about 40 s) ...")
        settled, period = calibrate_period(model, wad_dir / "twin.wad", opt=opt)
        out(f"calibration (a clock-free twin of the map): the dolls settle in {settled} tics; compute budget "
            f"{period} tics; {time.perf_counter() - t0:.0f}s")
    wad = wad_dir / "brainlevel_B.wad"
    t1 = time.perf_counter()
    meta = build_brainlevel(model, wad, period=period, lives=lives, opt=opt, doors=doors, lesion_at=lesion_at)
    if doors:
        lanes = sum(meta.layout["doors"].values())
        out(f"runtime doors: {len(door_tags)} units, {lanes} lanes with a door in the map; the level's clock closes them "
            f"before decision {lesion_at} (the number is compiled into the clock script)")
    out(f"one map: his arena + his transformer + the action argmax ({meta.n_dolls:,} voodoo dolls of him, "
        f"{meta.n_sectors:,} sectors); built in {time.perf_counter() - t1:.1f}s")
    run = ClockRun(model, wad, seed=seed, lesion=lesion if lesion_at is not None else None,
                   resolution=vzd.ScreenResolution.RES_320X240 if gif else vzd.ScreenResolution.RES_160X120)
    g = run.g
    trace = None
    if gif:
        from .gifout import GifTrace
        trace = GifTrace(wad, gif)

    def positions():
        return [(o.position_x, o.position_y) for o in g.get_state().objects
                if o.name == "DoomPlayer" and o.position_y >= 0]

    stats = dict(seed=seed, decisions=0, exact=0, scalars=0, kills=0, shots=0, period=period,
                 dolls=meta.n_dolls, sectors=meta.n_sectors, actions=[])
    t_run = time.perf_counter()
    caption = {"text": ""}
    try:
        while stats["decisions"] < decisions:
            t_dec = time.perf_counter()

            def thinking(tokens, tic, budget):
                if trace is not None:
                    trace.add(g.get_state().screen_buffer, positions(),
                              f"decision {stats['decisions'] + 1}  the dolls are computing: tic {tic}"
                              + (f" of {budget}" if budget else "") + f"  {caption['text']}")

            decision = run.decide(on_think=thinking if trace else None,
                                  think_every=max(64, period // 12) if trace else 0)
            if decision is None:
                out("the level's clock finished its lives")
                break
            stats["decisions"] += 1
            stats["exact"] += 1                     # decide() raises on the first value that differs
            stats["scalars"] += decision["scalars"]
            ammo0 = g.get_game_variable(GV.AMMO2)
            act = policy.ACTIONS[decision["action"]]
            dolls = positions()
            run.play(lambda: None)
            kills, ammo = int(g.get_game_variable(GV.KILLCOUNT)), g.get_game_variable(GV.AMMO2)
            stats["kills"] = kills
            stats["shots"] += int(ammo0 - ammo)
            stats["actions"].append(act)
            caption["text"] = f"{stats['exact']}/{stats['decisions']} exact  kills {kills}"
            n = stats["decisions"]
            healthy = ""
            if lesion:                                 # what the unlesioned reference brain chooses on these tokens
                was = policy.ACTIONS[policy.act_index(forward(healthy_model, decision["tokens"])[-1])]
                stats["changed"] = stats.get("changed", 0) + (was != act)
                stats.setdefault("healthy_actions", []).append(was)
                stats.setdefault("doors", []).append(decision["doors"])
                if was != act:
                    stats.setdefault("changed_at", []).append(n)
                healthy = f"  (healthy brain: {policy.ACTION_NAMES[policy.ACTIONS.index(was)]})"
            if lesion_at is not None:
                healthy += f"  doors {'CLOSED' if decision['doors'] else 'open'}"
            out(f"decision {n:3d}/{decisions}  sees {', '.join(policy.describe(decision['tokens'])).lower()}  "
                f"his dolls chose {policy.ACTION_NAMES[decision['action']]}{healthy} (logits L/R/F "
                f"{[decision['logits'][i] for i in policy.ACTION_IDS]})  EXACT ({decision['scalars']} values: "
                f"tokens, every transformer scalar, logits, comparator, action; {period} compute tics; "
                f"{time.perf_counter() - t_dec:.1f}s)  kills {kills}, bullets {int(ammo)}")
            if trace is not None:
                trace.add(g.get_state().screen_buffer, dolls,
                          f"decision {n}  his dolls chose {act}  {caption['text']}")
    finally:
        stats["audit"] = run.audit()
        run.close()
    stats["wall_s"] = time.perf_counter() - t_run
    n = stats["decisions"]
    out(f"{stats['exact']}/{n} decisions EXACT ({stats['scalars']:,} values checked, tolerance 0); the level "
        f"fired {stats['shots']} shots and {stats['kills']} demons died"
        + (f"; the lesion changed the chosen action in {stats.get('changed', 0)} of them"
           + (f" (decisions {stats['changed_at']})" if stats.get("changed_at") else "") if lesion else ""))
    audit = stats["audit"]
    out(f"Python sent {sum(audit['commands'].values())} console commands, all readout scripts (non-readout: "
        f"{audit['non_readout_commands']}); it wrote no token, port or action and pressed no button. The level "
        f"sensed the demon, the dolls computed and chose, the level turned him or fired.")
    out(f"wall {stats['wall_s']:.0f}s, {stats['wall_s'] / max(1, n):.1f}s per decision ({period} compute tics + "
        f"{run.schedule['hold']} hold + 4 action tics each)")
    if trace is not None:
        path, size = trace.close()
        out(f"gif: {path} ({size / 1e6:.2f} MB, {len(trace.frames)} frames)")
    if tmp is not None:
        tmp.cleanup()
    return stats


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m demos.brainlevel",
        description="His brain is the level: the marine's policy transformer and the action argmax run on voodoo "
                    "dolls in the same DOOM map as the arena he fights in. The map's own clock senses the demon, "
                    "the dolls decide, the level turns him or fires. Python only advances tics and reads, and "
                    "checks every value against an integer reference (tolerance 0).")
    ap.add_argument("--seed", type=int, default=4, help="the game's own RNG; 4 is the documented run")
    ap.add_argument("--decisions", type=int, default=5,
                    help="decisions to run (default 5; the documented run is --decisions 50 --seed 4)")
    ap.add_argument("--lives", type=int, default=5, help="lives the level's clock plays before it stops")
    ap.add_argument("--period", type=int, default=None,
                    help="compute budget per decision in tics (default: measured on a twin map, plus a margin)")
    ap.add_argument("--gif", metavar="PATH", help="write a small looping GIF of the run (the game's frames and the dolls)")
    ap.add_argument("--workdir", help="where to keep the compiled maps (default: a temporary directory)")
    ap.add_argument("--opt", type=int, choices=(0, 1, 2, 3), default=0,
                    help="schedule of the brain's map (doomformer.compile opt): 0 the original, 3 runs every layer as "
                         "one wave with an ACS copy between layers; the compute budget is measured on the twin at the "
                         "same level")
    ap.add_argument("--lesion", metavar="SPEC",
                    help="lesion units (site:units;...), for example 'x0:0;L0.ao:7;L0.x1:7;L0.m:0;L0.x2:0'. With "
                         "--lesion-at N the level closes the lesioned units' lane doors itself before decision N; "
                         "without it the lesion is compiled into the map's weights (doomformer.intervene.fold), a "
                         "different map, and the reference is the lesioned model")
    ap.add_argument("--lesion-at", type=int, metavar="N", default=None,
                    help="close the --lesion doors from decision N (1-based): the level's own clock does it, the "
                         "number is compiled into the clock script, Python only reads. Every value up to N is "
                         "checked against the healthy reference, from N on against the masked reference")
    args = ap.parse_args(argv)
    if args.decisions < 1 or args.lives < 1 or (args.period is not None and args.period < 1):
        ap.error("--decisions, --lives and --period must be positive")
    if args.lesion_at is not None and (not args.lesion or args.lesion_at < 1):
        ap.error("--lesion-at needs --lesion and a decision number >= 1")
    lesion = None
    if args.lesion:
        try:
            lesion = {site: [int(u) for u in units.split(",")]
                      for site, units in (part.split(":") for part in args.lesion.split(";"))}
        except ValueError:
            ap.error("--lesion: use 'site:unit,unit;site:unit', e.g. x0:0,3;L0.m:5")
    stats = run_brainlevel(seed=args.seed, decisions=args.decisions, lives=args.lives, period=args.period,
                           gif=args.gif, workdir=args.workdir, lesion=lesion, opt=args.opt, lesion_at=args.lesion_at)
    return 0 if stats["exact"] == stats["decisions"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
