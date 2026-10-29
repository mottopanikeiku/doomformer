"""An ACS-clocked sensor/network/actuator loop, with a readout-only Python host.

Dolls compute network arithmetic. ACS samples engine queries, writes input gates,
resets/releases dolls, selects argmax and applies actions. Python builds/calibrates
an isolated twin, then only advances tics and reads/verifies the live map.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import vizdoom as vzd

from .brainacs import SCRIPT_PISTOL, sensor_acs
from .intref import QModel, forward
from .kit import MLP, MismatchError, RangeError, _world_hook
from .onemap import build_onemap
from .pure import compile_pure
from .readout import ReadoutOnly, readout_commands
from .runtime import TileRunner, USER
from .sense import Sensor, nearest_replica
from .tile import MOVE_SPEED

CLOCK, ENTER, STATUS, SAMPLE = 30000, 30001, 30002, 30003
BOOT, REWIND, WRITE = 70, 4, 3


@dataclass(frozen=True)
class Actuator:
    """A single selected action, executed once per decision by ACS."""
    source: str

    @staticmethod
    def turn(delta):
        """Turn by signed integer 1/65536-circle units; positive is left."""
        delta = _integer(delta, "turn delta", -65535, 65535)
        return Actuator(f"SetActorAngle(AG_PLAYER, (GetActorAngle(AG_PLAYER)+({delta}))&65535);")

    @staticmethod
    def fire():
        """Trigger ScriptPistol from current facing, using one Clip and no horizontal spread."""
        return Actuator('GiveActorInventory(AG_PLAYER, "ScriptFireToken", 1);')

    @staticmethod
    def thrust(angle, force):
        """Thrust the player: integer degrees quantized to 256 directions, force 0..255."""
        angle = _integer(angle, "thrust angle", 0, 359)
        force = _integer(force, "thrust force", 0, 255)
        return Actuator(f"ThrustThing({round(angle * 256 / 360) % 256}, {force}, 0, AG_PLAYER);")

    @staticmethod
    def door(tag, speed=16):
        """Open a world door (sector tag 3..999), leaving it open."""
        tag = _integer(tag, "door tag", 3, 999)
        speed = _integer(speed, "door speed", 1, 255)
        return Actuator(f"Door_Open({tag}, {speed}, 0);")

    @staticmethod
    def script(number, *args):
        """Execute a world script once, with the real player as activator."""
        number = _integer(number, "world script", 200, 29999)
        if len(args) > 3:
            raise ValueError("a script takes at most three arguments")
        args = [_integer(v, "script argument", -2147483648, 2147483647) for v in args]
        return Actuator(f"ACS_ExecuteAlways({number}, 0, {', '.join(map(str, args + [0] * (3-len(args))))});")


def _integer(value, name, lo, hi):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


class Agent:
    """Compile an MLP or token-input Transformer into a reusable autonomous map.

    sensors: one Sensor per MLP input, or 1..T token sensors. Transformer unused
    positions are padded with token 0 and ALSO verified. actuators: insertion-
    ordered {output: Actuator or sequence}; output is yN / logits:t:j, or an
    integer output index (Transformer indexes the last sensor position). A
    Transformer vocabulary character is also accepted. Argmax is over these
    outputs ONLY; equal scores choose the first mapping entry.

    world(builder) must place the final player start; monsters share monster_tid.
    Reserved: scripts 30000..30003, identifiers ag_*, AG_PLAYER, player_tid,
    and temporary TIDs 50000..50000+max_monsters-1. World scripts must not write
    network ports or USER variables during autonomous execution.

    freeze_monsters makes tagged monsters dormant during compute/readout; they
    wake during world_tics. This is optional, and does not freeze the player.
    period=None measures settling on a clock-free twin plus max(64,20%) margin.
    Explicit periods are checked by exact readback, not assumed sufficient.
    """
    def __init__(self, net, *, sensors, actuators, world, period=None,
                 freeze_monsters=True, monster_tid=901, player_tid=900,
                 max_monsters=64, world_tics=20, wad=None):
        if not isinstance(net, (MLP, QModel)):
            raise TypeError("Agent takes an MLP or Transformer")
        self.net = net
        self.sensors = tuple(sensors)
        if not self.sensors or not all(isinstance(s, Sensor) for s in self.sensors):
            raise TypeError("sensors must be a nonempty sequence of sense.Sensor")
        if isinstance(net, MLP) and len(self.sensors) != net.n_in:
            raise ValueError("one sensor is required per MLP input")
        if isinstance(net, QModel) and len(self.sensors) > net.T:
            raise ValueError("token sensors exceed Transformer context")
        self.player_tid = _integer(player_tid, "player_tid", 1, 9999)
        self.monster_tid = _integer(monster_tid, "monster_tid", 1, 9999)
        if self.player_tid == self.monster_tid:
            raise ValueError("player and monsters need different TIDs")
        self.capacity = _integer(max_monsters, "max_monsters", 1, 512)
        self.world_tics = _integer(world_tics, "world_tics", 1, 10000)
        self.freeze = bool(freeze_monsters)
        self.world = _world_hook(world)
        self.action_keys, self.output_names, self.actions = [], [], []
        for key, value in actuators.items():
            name = self._output_name(key)
            group = (value,) if isinstance(value, Actuator) else tuple(value)
            if not group or not all(isinstance(a, Actuator) for a in group):
                raise TypeError("each actuator value must be an Actuator or a nonempty sequence")
            self.action_keys.append(key)
            self.output_names.append(name)
            self.actions.append(group)
        if not self.actions:
            raise ValueError("at least one action output is required")
        self._temporary = TemporaryDirectory(prefix="doomformer-agent-")
        self.wad = Path(wad) if wad is not None else Path(self._temporary.name) / "agent.wad"
        self.wad.parent.mkdir(parents=True, exist_ok=True)
        self.runner = None
        self.game = None
        self.steps = 0
        self.calibration = None
        try:
            twin = Path(self._temporary.name) / "twin.wad"
            planned = self._build(twin, clock=None)
            if any(name not in planned.outputs for name in self.output_names):
                raise ValueError("actuators must select exposed final network outputs")
            if len(planned.pages) >= CLOCK - 100:
                raise ValueError("network readout scripts overlap Agent scripts")
            if period is None:
                r = TileRunner(twin)
                try:
                    # Both input polarities open: conservative circuit timing, not a
                    # reference inference. Internal layer clocks run all their waves.
                    r.start({key.split(':', 1)[1]: 12000 for key in planned.ports})
                    settled = r.settle()
                finally:
                    r.close()
                period = settled + max(64, math.ceil(settled * 0.2))
                self.calibration = dict(settled_tics=settled, margin=period-settled)
            self.period = _integer(period, "period", 1, 1000000)
            self.sample_size = 2 + 2 * len(self.sensors) + 6 * self.capacity
            self.sample_pages = math.ceil(self.sample_size / len(USER))
            # the samples, the network pages (a poison tic before each on a guarded map), two status reads; the
            # unguarded window (pages + 12) is kept wherever the guarded readout fits it with 4 tics to spare
            self.hold = self.sample_pages + max(len(planned.pages) + 12, planned.read_tics + 4)
            self.meta = self._build(self.wad, clock=planned)
        except BaseException:
            self._temporary.cleanup()
            raise

    def _output_name(self, key):
        if isinstance(self.net, MLP):
            name = f"y{key}" if isinstance(key, int) else str(key)
            if name not in {f"y{i}" for i in range(self.net.n_out)}:
                raise ValueError(f"invalid MLP action output {key!r}")
            return name
        if isinstance(key, int):
            j = key
        elif key in self.net.vocab:
            j = self.net.vocab.index(key)
        elif isinstance(key, str) and key.startswith("logits:"):
            _, t, j = key.split(":")
            if not 0 <= int(t) < len(self.sensors) or not 0 <= int(j) < len(self.net.vocab):
                raise ValueError(f"invalid Transformer action output {key!r}")
            return key
        else:
            raise ValueError(f"invalid Transformer action output {key!r}")
        if not 0 <= j < len(self.net.vocab):
            raise ValueError(f"invalid Transformer action output {key!r}")
        return f"logits:{len(self.sensors)-1}:{j}"

    def _build(self, wad, clock):
        def world(builder):
            source = self.world(builder)
            source += f"\n#define AG_PLAYER {self.player_tid}\n"
            source += f'''script {ENTER} ENTER {{
              Thing_ChangeTID(0, AG_PLAYER);
              GiveInventory("ScriptPistol", 1); SetWeapon("ScriptPistol");
              {f'Thing_Deactivate({self.monster_tid});' if self.freeze or clock is None else ''}
            }}\n'''
            if clock is not None:
                source += self._clock_source(clock)
            return source
        kwargs = dict(world=world, extra_lumps=[("DECORATE", SCRIPT_PISTOL.encode())])
        if isinstance(self.net, QModel):
            return build_onemap(self.net, wad, **kwargs)
        stages = self.net._stages()
        consumed = {key.split(":", 1)[1] for stage in stages for key in stage.ports}
        exposed = [name for stage in stages for name, _ in stage.outputs if name in consumed]
        return compile_pure(stages, wad, "agent_mlp", readout=exposed, **kwargs)

    def _clock_source(self, meta):
        n, cap = len(self.sensors), self.capacity
        primitives = "\n".join(f"ag_raw[{i}]={s.primitive(self.player_tid)};" for i, s in enumerate(self.sensors))
        sensed = "\n".join(f"ag_input[{i}]={s.expression(f'ag_raw[{i}]')};" for i, s in enumerate(self.sensors))
        writes = []
        for key, tag in meta.ports.items():
            kind, name = key.split(":", 1)
            base = name.lstrip("-")
            idx = int(base[1:]) if isinstance(self.net, MLP) else int(base.split(":")[1])
            value = f"ag_input[{idx}]" if idx < n else "0"
            if name.startswith("-"):
                value = f"-({value})"
            fn = "Floor_MoveToValue" if kind == "floor" else "Ceiling_MoveToValue"
            writes.append(f"v={value}; {fn}({tag}, {MOVE_SPEED}, v<0 ? -v : v, v<0);")
        score = "\n".join(f"ag_score[{i}]={meta.layout['read_expressions'][name]};" for i, name in enumerate(self.output_names))
        actions = "\n".join(f"if (ag_action=={i}) {{ {' '.join(a.source for a in group)} }}" for i, group in enumerate(self.actions))
        flat = ["ag_n", "ag_pa"] + [f"ag_raw[{i}]" for i in range(n)] + [f"ag_input[{i}]" for i in range(n)] + [f"ag_c[{i}]" for i in range(6*cap)]
        pages = []
        for page in range(self.sample_pages):
            expressions = flat[page*len(USER):(page+1)*len(USER)]
            pages.append(f"if (page=={page}) {{ " + " ".join(f"o{i+1}={expr};" for i, expr in enumerate(expressions)) + " }")
        freeze = f"Thing_Deactivate({self.monster_tid});" if self.freeze else ""
        wake = f"Thing_Activate({self.monster_tid});" if self.freeze else ""
        token_guard = f"ag_input[i]<0 || ag_input[i]>={len(self.net.vocab)}" if isinstance(self.net, QModel) else "ag_input[i]<-12000 || ag_input[i]>12000"
        return sensor_acs(self.monster_tid, prefix="ag_base", player_tid=self.player_tid, tmp_tid=50000) + f'''
int ag_step; int ag_phase; int ag_action; int ag_error; int ag_hold_start;
int ag_n; int ag_pa; int ag_alive; int ag_bearing; int ag_distance; int ag_sight;
int ag_raw[{n}]; int ag_input[{n}]; int ag_c[{6*cap}]; int ag_score[{len(self.actions)}];
function void ag_sample(void) {{
  int i; int h; int dx; int dy; int d; int b; int sight; int best=-1;
  int px=GetActorX(AG_PLAYER); int py=GetActorY(AG_PLAYER);
  int c; int s; int rx; int ry;
  ag_pa=GetActorAngle(AG_PLAYER); ag_n=0; ag_alive=0;
  {primitives}
  while (SetActivator({self.monster_tid})) {{
    if (ag_n>={cap}) {{ ag_error=1; break; }}
    h=GetActorProperty(0, APROP_Health);
    dx=GetActorX(0)-px; dy=GetActorY(0)-py;
    d=VectorLength(dx,dy); b=(VectorAngle(dx,dy)-ag_pa)&65535;
    if (b>=32768) b-=65536;
    Thing_ChangeTID(0, 50000+ag_n);
    sight=CheckSight(AG_PLAYER, 50000+ag_n, 0);
    ag_c[6*ag_n]=dx; ag_c[6*ag_n+1]=dy; ag_c[6*ag_n+2]=d;
    ag_c[6*ag_n+3]=b; ag_c[6*ag_n+4]=sight; ag_c[6*ag_n+5]=h;
    if (h>0) {{
      if (best<0 || d<ag_c[6*best+2] || (d==ag_c[6*best+2] && b<ag_c[6*best+3])) best=ag_n;
      ag_alive++;
    }}
    ag_n++;
  }}
  for (i=0;i<ag_n;i++) Thing_ChangeTID(50000+i, {self.monster_tid});
  SetActivator(AG_PLAYER);
  ag_sight=0; ag_bearing=ag_base_bearing; ag_distance=ag_base_distance;
  if (best>=0) {{
    c=ag_base_fcos(ag_pa); s=ag_base_fsin(ag_pa);
    dx=ag_c[6*best]; dy=ag_c[6*best+1];
    rx=FixedMul(dx,c)+FixedMul(dy,s); ry=FixedMul(dy,c)-FixedMul(dx,s);
    ag_bearing=ag_base_btok(ag_c[6*best+3],rx,ry);
    ag_distance=ag_base_dtok(ag_c[6*best+2]); ag_sight=ag_c[6*best+4];
  }}
  {sensed}
  for(i=0;i<{n};i++) if ({token_guard}) ag_error=2;
}}
script {STATUS} (void) {{
  o1=ag_step; o2=ag_phase; o3=ag_action; o4=ag_error;
  o5=ag_hold_start; o6=GetLevelInfo(LEVELINFO_KILLED_MONSTERS);
}}
script {SAMPLE} (int page) {{ {' '.join(pages)} }}
script {CLOCK} OPEN {{
  int v; int i; Delay({BOOT});
  while(TRUE) {{
    ag_phase=1; {freeze}
    ACS_ExecuteAlways(3,0); Delay({REWIND});
    ag_sample();
    if(ag_error) {{ ag_phase=3; terminate; }}
    {' '.join(writes)}
    Delay({WRITE}); ACS_ExecuteAlways(7,0); Delay({self.period});
    {score}
    ag_action=0;
    for(i=1;i<{len(self.actions)};i++) if(ag_score[i]>ag_score[ag_action]) ag_action=i;
    {wake} SetActivator(AG_PLAYER);
    {actions}
    Delay({self.world_tics}); {freeze}
    ag_step++; ag_phase=2; ag_hold_start=Timer();
    Delay({self.hold});
  }}
}}
'''

    def start(self):
        """Open the live map lazily, installing the audit before any host command."""
        if self.runner is None:
            def configure(game):
                game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
            self.runner = TileRunner(self.wad, configure=configure)
            allowed = readout_commands(len(self.meta.pages), f"puke -{STATUS}",
                                       *(f"puke -{SAMPLE} {i}" for i in range(self.sample_pages)))
            self.game = ReadoutOnly(self.runner.game, allowed)
            self.runner.game = self.game
        return self

    def _status(self):
        self.game.send_game_command(f"puke -{STATUS}")
        self.game.advance_action(1)
        return [int(self.game.get_game_variable(v)) for v in USER[:6]]

    def _step(self):
        # Polling is readout-only. ACS uses direct sector queries for argmax, so
        # its decisions never race Python over the shared USER readout slots.
        limit = BOOT + REWIND + WRITE + self.period + self.world_tics + self.hold + 16
        for _ in range(limit):
            if self.game.is_episode_finished():
                raise RuntimeError("Agent episode ended before the next complete decision")
            status = self._status()
            step, phase, action, error, _, kills = status
            if error:
                raise RangeError("sample exceeds max_monsters" if error == 1 else "sensor value outside network input range")
            if phase == 2 and step == self.steps+1:
                break
            if step > self.steps+1:
                raise RuntimeError("an autonomous decision was missed; external tic advancement is unsupported")
        else:
            raise RuntimeError("Agent clock did not enter its readout window")
        flat = []
        for page in range(self.sample_pages):
            self.game.send_game_command(f"puke -{SAMPLE} {page}")
            self.game.advance_action(1)
            flat.extend(int(self.game.get_game_variable(v)) for v in USER)
        n = len(self.sensors)
        count, angle = flat[:2]
        raw, inputs = flat[2:2+n], flat[2+n:2+2*n]
        rows = flat[2+2*n:2+2*n+6*count]
        candidates = [rows[i:i+6] for i in range(0, len(rows), 6)]
        nearest = nearest_replica(angle, candidates)
        expected_inputs = [s.replica(v, nearest) for s, v in zip(self.sensors, raw)]
        bad = [(f"sensor:{i}", a, b) for i, (a,b) in enumerate(zip(inputs, expected_inputs)) if a != b]
        got = self.runner.read()
        after = self._status()
        if after[:2] != status[:2]:
            raise RuntimeError("readout escaped the stable hold window")
        if set(got) != set(self.meta.outputs):
            raise RuntimeError("incomplete network readout")
        if isinstance(self.net, MLP):
            self.net.check(expected_inputs)
            ref = self.net.reference(expected_inputs)
            expected = {name: self.net._expected(name, ref) for name in got}
        else:
            # All exposed positions, including padding, are checked.
            tokens = expected_inputs + [0] * (self.net.T-len(expected_inputs))
            trace = {}
            forward(self.net, tokens, trace)
            expected = {}
            for name in got:
                key, *indices = name.lstrip('-').split(':')
                expected[name] = int(trace[key][tuple(map(int, indices))]) * (-1 if name.startswith('-') else 1)
        bad.extend((name, v, expected[name]) for name, v in got.items() if v != expected[name])
        scores = [expected[name] for name in self.output_names]
        chosen = max(range(len(scores)), key=scores.__getitem__)
        if action != chosen:
            bad.append(("action", action, chosen))
        if bad:
            raise MismatchError(f"Agent step {step}", bad)
        self.steps = step
        return dict(step=step, inputs=inputs, outputs=got, action=self.action_keys[action],
                    action_index=action, scores=scores, checked=len(got)+n+1,
                    exact=True, kills=kills, sample=dict(raw=raw, angle=angle, monsters=candidates),
                    period=self.period)

    def run(self, steps):
        """Advance and verify exactly `steps` additional decisions; return per-step records."""
        steps = _integer(steps, "steps", 0, 1000000000)
        self.start()
        return [self._step() for _ in range(steps)]

    def audit(self):
        """Counts of live-map readout commands and rejected host writes (calibration excluded)."""
        return self.game.audit() if self.game else dict(commands={}, forbidden={}, non_readout_commands=0)

    def close(self):
        if self.runner is not None:
            self.runner.close()
            self.runner = None
        self._temporary.cleanup()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
