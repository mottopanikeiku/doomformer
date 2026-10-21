"""DOOM marine that WALKS A GENERATED LEVEL: the "ouroboros" finale (DOOM named a level, DOOM built it, now DOOM plays it).

Scenario: a ``demos.levelgen`` level (PWAD for freedoom2.wad, MAP01): rooms, corridors, stairs, lifts, doors,
monsters, items and an exit switch (USE on it ends the level).  The marine starts with a pistol, plays on
``SKILL`` (no god mode, no notarget) and must walk start -> exit, open doors, fight what attacks it and press the
exit switch.  Score: ``reached_exit``, ``progress`` (0..1 along the route), ``kills``, ``died``.

One decision = one transformer call, exactly like ``doomformer.policy``:
    observe(state, game) -> [0, route, enemy, distance, flag]     5 tokens, BOS first
    logits = model(tokens)[-1]                                    any model; here the integer QModel
    buttons = act(logits)                                         argmax over the ACTION tokens only
    game.make_action(buttons, 1) x TICS_PER_DECISION              (or make_action(buttons, TICS_PER_DECISION))

Tokens (every token is one character of ``VOCAB``; index 0 is '$' = BOS):
    actions   'L' 'R' big turn left/right, 'l' 'r' small turn left/right (turn in place), 'W' walk forward,
              'F' fire, 'U' walk forward while pressing USE (doors, the exit switch)          (``ACTIONS``)
    route     (position 1) 17 characters 'a'..'q': angle of the next route waypoint (a look-ahead point
              ``LOOKAHEAD`` map units further along the route) relative to the marine's aim, in degrees, positive =
              to the left.  'a' = [-180,-100)  ...  'i' = [-2,2)  ...  'q' = [100,180]
    enemy     (position 2) the same 17 characters for the NEAREST VISIBLE LIVE MONSTER within ``ENGAGE_RANGE``;
              '?' = none
    distance  (position 3) '0'..'4': distance to that monster (edges 128/256/400/600 map units); '4' when none
    flag      (position 4) '0' nothing special; '1' a door (or the exit switch) is within reach, ``U`` is
              the way through; '2' the same, and the previous decision was already ``U`` (USE is edge-triggered:
              holding it opens nothing, so the marine must release it -> walk once, then press again);
              '3' blocked: the last ``STALL_DECISIONS`` decisions were all W/U and the marine did not move

COMPASS: the route bearing is HOST-PROVIDED NAVIGATION INFO, like a compass or a quest marker.  The harness calls
``set_level(info)`` (info = ``levelgen.build_level``'s return value) before an episode; ``observe`` reads the
route, the door points and the marine's position/aim from it.  The brain never sees the map: it gets the bearing
of the next waypoint and chooses every action (turn, walk, fire, use) itself.  Enemy bearing/distance come from
ViZDoom's object list restricted to monsters ViZDoom's label buffer reports as VISIBLE on screen.

Host restriction: only the logits of the ACTION tokens are meaningful; ``act`` takes the argmax over them only
(lowest index on ties).  Hidden host state: the module remembers the marine's progress along the route and its last
action (both are inputs the observation already encodes, never extra decision logic); both reset at a new episode.

Expert: ``expert_index`` has exact info (objects, route): engage the nearest visible threat (turn, fire when inside
the hit cone), otherwise follow the route, USE at doors and at the exit.  The learned policy sees only the bins.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from pathlib import Path

import numpy as np
import vizdoom as vzd

from doomformer.runtime import FREEDOOM2, init_game

ACTIONS: list[str] = ["L", "R", "l", "r", "W", "F", "U"]
ACTION_NAMES: list[str] = ["TURN LEFT", "TURN RIGHT", "VEER LEFT", "VEER RIGHT", "WALK", "FIRE", "USE"]
# buttons of make_game: [MOVE_FORWARD_BACKWARD_DELTA, TURN_LEFT_RIGHT_DELTA, ATTACK, USE]; a positive turn delta
# turns RIGHT.  The same vector is held for every tic of a decision.
WALK_SPEED = 50.0
BIG_TURN = 6.0           # degrees per tic (x TICS_PER_DECISION per decision)
SMALL_TURN = 1.0
BUTTONS: dict[str, list[float]] = {
    "L": [0, -BIG_TURN, 0, 0], "R": [0, BIG_TURN, 0, 0],
    "l": [WALK_SPEED, -SMALL_TURN, 0, 0], "r": [WALK_SPEED, SMALL_TURN, 0, 0],
    "W": [WALK_SPEED, 0, 0, 0], "F": [0, 0, 1, 0], "U": [WALK_SPEED, 0, 0, 1],
}
BEARING_EDGES = [-100, -60, -35, -24, -12, -7, -4, -2, 2, 4, 7, 12, 24, 35, 60, 100]   # degrees, 17 bins
BEARING_CHARS = "abcdefghijklmnopq"
NO_ENEMY = "?"
DIST_EDGES = [64, 160, 300, 500]                          # map units, 5 bins
DIST_CHARS = "01234"
VOCAB: str = "$" + "".join(ACTIONS) + BEARING_CHARS + NO_ENEMY + DIST_CHARS
assert len(VOCAB) <= 32 and VOCAB[0] == "$" and len(ACTIONS) <= 8

ACTION_IDS: list[int] = [VOCAB.index(c) for c in ACTIONS]
TICS_PER_DECISION = 6                                      # game tics the chosen buttons are held
CONTEXT_LEN = 5
SKILL = 2                                                  # doom_skill 1..5: 2 = "hey, not too rough"
ENGAGE_RANGE = 700.0                                       # monsters further away are not reported
PICKUP_CATEGORIES = ("Weapon", "Health", "Armor", "Ammo")
WEAPON_RANGE = 700.0                                       # ... weapons from further away
PICK_RANGE = 120.0                                         # visible useful pickups this close become the compass target
PICK_END = 300.0                                           # ... and not on the last stretch of the route
PICK_SAFE = 300.0                                          # ... and not while a visible monster is this close
PICK_STUCK = 4                                             # ... or has not moved 24 units in this many decisions
PICK_DECISIONS = 25                                        # ... until the marine has chased one this long
LOOKAHEAD = 150.0                                          # map units along the route
USE_REACH = 140.0                                          # door point distance that raises the use flag
EXIT_REACH = 110.0                                         # distance to the end of the route that raises it
FIRE_TOL = 4.0                                             # expert fires inside this aim error (degrees) ...
FIRE_TOL_NEAR = 7.0                                        # ... and inside this one at point blank (< 64 units)
TURN_BIG_AT = 24.0                                         # expert: |bearing| >= this -> big turn
TURN_SMALL_AT = 7.0                                        # expert: |bearing| >= this -> small turn
STALL_DECISIONS = 3                                        # walking this many decisions ...
STALL_DIST = 12.0                                          # ... and moving less than this = blocked (flag '3')
EXTRA = 120.0                                              # the route is extended past the exit switch
_GV = vzd.GameVariable
GAME_VARIABLES = [_GV.POSITION_X, _GV.POSITION_Y, _GV.ANGLE, _GV.HEALTH, _GV.KILLCOUNT, _GV.ARMOR]
EPISODE_TICS = 36000


# ----------------------------------------------------------------------------- level / route
class Route:
    def __init__(self, pts):
        self.pts = [tuple(map(float, p)) for p in pts]
        self.cum = [0.0]
        for a, b in zip(self.pts, self.pts[1:]):
            self.cum.append(self.cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.total = self.cum[-1]

    def at(self, s):
        s = max(0.0, min(self.total, s))
        for i in range(len(self.pts) - 1):
            if s <= self.cum[i + 1]:
                seg = self.cum[i + 1] - self.cum[i] or 1.0
                t = (s - self.cum[i]) / seg
                a, b = self.pts[i], self.pts[i + 1]
                return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        return self.pts[-1]

    def project(self, x, y, lo, hi):
        """Arc length of the closest route point with s in [lo, hi]."""
        best, bs = 1e18, lo
        for i in range(len(self.pts) - 1):
            if self.cum[i + 1] < lo or self.cum[i] > hi:
                continue
            (ax, ay), (bx, by) = self.pts[i], self.pts[i + 1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy or 1.0
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
            s = max(lo, min(hi, self.cum[i] + t * math.sqrt(L2)))
            px, py = self.at(s)
            d = math.hypot(x - px, y - py)
            if d < best:
                best, bs = d, s
        return bs


class _Level:
    def __init__(self, info: dict):
        pts = [tuple(map(float, p)) for p in info["route"]]
        (ax, ay), (bx, by) = pts[-2], pts[-1]
        seg = math.hypot(bx - ax, by - ay) or 1.0
        self.route = Route(pts + [(bx + (bx - ax) / seg * EXTRA, by + (by - ay) / seg * EXTRA)])
        self.total = self.route.total - EXTRA            # arc length of the exit approach point
        self.doors = [tuple(map(float, p)) for p in info.get("door_points", [])]
        self.end = pts[-1]


class _Track:
    """Host memory for one episode: progress along the route and the previous action."""

    def __init__(self):
        self.s = 0.0
        self.t = -1
        self.last = -1
        self.pos: list[tuple[float, float]] = []
        self.acts: list[int] = []         # actions taken (one per decision)
        self.goal = None                  # id of the pickup the compass points at
        self.goal_n = 0                   # decisions spent on it
        self.goal_t = -1
        self.skip: set[int] = set()       # pickups given up (unreachable or already full)
        self.health = self.armor = 0.0


_LEVEL: _Level | None = None
_TR = _Track()


def set_level(info: dict) -> None:
    """Host-provided navigation: the level's route (a compass), door points; call before every episode."""
    global _LEVEL
    _LEVEL = _Level(info)
    _TR.__init__()


def level_total() -> float:
    return _LEVEL.total


def prepare_level(name: str, out_dir: Path | str) -> tuple[dict, Path]:
    """Generate the levelgen level for `name` into out_dir, register it as the compass (`set_level`) and return
    (build_level's info, wad path)."""
    from .levelgen import build_level, slugify
    wad = Path(out_dir) / f"{slugify(name)}.wad"
    info = build_level(name, wad)
    set_level(info)
    return info, wad


def note_action(idx: int) -> None:
    """Remember the action chosen for the current decision (idempotent within a decision)."""
    _TR.last = idx
    n = len(_TR.pos)
    if len(_TR.acts) >= n:
        _TR.acts[n - 1:] = [idx]
    else:
        _TR.acts += [idx]


def make_game(render: bool = False, resolution=None, wad: Path | str | None = None, seed: int | None = None,
              skill: int | None = None) -> vzd.DoomGame:
    """A levelgen WAD with objects + labels info (the marine's senses).  No god mode, no notarget.

    render=False: headless 160x120, no HUD.  render=True: 320x240 (or `resolution`) with the HUD so the host can
    grab frames.  Buttons: [forward delta, turn delta, ATTACK, USE]; ``new_episode()`` restarts the level."""
    assert wad is not None, "make_game needs the level WAD"
    g = vzd.DoomGame()
    g.set_doom_game_path(FREEDOOM2)
    g.set_doom_scenario_path(str(wad))
    g.set_doom_map("MAP01")
    g.set_doom_skill(SKILL if skill is None else skill)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_objects_info_enabled(True)
    g.set_labels_buffer_enabled(True)
    g.set_available_buttons([vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, vzd.Button.TURN_LEFT_RIGHT_DELTA,
                             vzd.Button.ATTACK, vzd.Button.USE])
    g.set_available_game_variables(GAME_VARIABLES)
    if resolution is None:
        resolution = vzd.ScreenResolution.RES_320X240 if render else vzd.ScreenResolution.RES_160X120
    g.set_screen_resolution(resolution)
    g.set_render_hud(bool(render))
    g.set_render_crosshair(False)
    g.set_screen_format(vzd.ScreenFormat.CRCGCB)            # planar RGB, CHW
    g.set_episode_timeout(EPISODE_TICS)
    g.set_mode(vzd.Mode.PLAYER)
    if seed is not None:
        g.set_seed(int(seed))
    init_game(g)
    return g


# ----------------------------------------------------------------------------- perception
def _norm(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _sync(game):
    """Marine pose; updates the host memory (new episode detection, route progress, position history)."""
    x, y = game.get_game_variable(_GV.POSITION_X), game.get_game_variable(_GV.POSITION_Y)
    ang = game.get_game_variable(_GV.ANGLE)
    t = game.get_episode_time()
    if t < _TR.t:
        _TR.__init__()
    _TR.health, _TR.armor = game.get_game_variable(_GV.HEALTH), game.get_game_variable(_GV.ARMOR)
    if t != _TR.t:
        _TR.t = t
        lv = _LEVEL
        _TR.s = max(_TR.s, lv.route.project(x, y, _TR.s - 40, _TR.s + 260))
        _TR.pos.append((x, y))
    return x, y, ang


def threats(state, game) -> list[tuple[float, float]]:
    """[(distance, bearing in degrees, positive = left of the aim)] of every VISIBLE live monster within
    ENGAGE_RANGE, nearest first.  Visible = ViZDoom's label buffer lists the object (it is drawn on screen)."""
    x, y, ang = _sync(game)
    seen = {l.object_id for l in state.labels if l.object_category == "Monster"} if state.labels else set()
    out = []
    for o in state.objects:
        if o.category == "Monster" and o.id in seen:
            dx, dy = o.position_x - x, o.position_y - y
            d = math.hypot(dx, dy)
            if d <= ENGAGE_RANGE:
                out.append((d, _norm(math.degrees(math.atan2(dy, dx)) - ang)))
    return sorted(out)


def _wanted(name: str, health: float, armor: float) -> bool:
    if name in ("Medikit", "Stimpack"):
        return health < 100
    if name == "HealthBonus":
        return health < 200
    if name == "GreenArmor":
        return armor < 100
    if name in ("BlueArmor", "ArmorBonus"):
        return armor < 200
    return True


def _pickup(state, x, y):
    """The nearest visible, useful pickup within PICK_RANGE (id, x, y) or None; gives up on one after PICK_TICS."""
    if _TR.goal_t != _TR.t:
        _TR.goal_t = _TR.t
        if _TR.goal is not None:
            _TR.goal_n += 1
            if _TR.goal_n > PICK_DECISIONS:
                _TR.skip.add(_TR.goal)
                _TR.goal = None
    if _TR.goal is not None and len(_TR.pos) > PICK_STUCK and _TR.goal_n >= PICK_STUCK:
        px, py = _TR.pos[-PICK_STUCK - 1]
        if math.hypot(x - px, y - py) < 24.0:         # walking into a wall / a drop: not reachable that way
            _TR.skip.add(_TR.goal)
            _TR.goal = None
    hp, ar = _TR.health, _TR.armor
    best = None
    for l in (state.labels if state is not None else ()):
        if l.object_category in PICKUP_CATEGORIES and l.object_id not in _TR.skip and _wanted(l.object_name, hp, ar):
            d = math.hypot(l.object_position_x - x, l.object_position_y - y)
            if d <= (WEAPON_RANGE if l.object_category == "Weapon" else PICK_RANGE) and (best is None or d < best[0]):
                best = (d, l.object_id, l.object_position_x, l.object_position_y)
    if best is None:
        _TR.goal = None
        return None
    if best[1] != _TR.goal:
        _TR.goal, _TR.goal_n = best[1], 0
    return best[1:]


def compass(state, game) -> float:
    """Bearing (degrees, positive = left) the marine should head for, relative to its aim: a visible useful pickup
    within PICK_RANGE if there is one, else the look-ahead waypoint of the route."""
    x, y, ang = _sync(game)
    p = _pickup(state, x, y) if _LEVEL.total - _TR.s > PICK_END else None
    if p is not None and PICK_SAFE and any(d < PICK_SAFE for d, _ in threats(state, game)):
        p = None                                           # no detours under fire
    if p is not None:
        tx, ty = p[1], p[2]
    else:
        lv = _LEVEL
        remaining = lv.total - _TR.s
        tx, ty = lv.route.at(_TR.s + (LOOKAHEAD if remaining > LOOKAHEAD else 60.0))
    return _norm(math.degrees(math.atan2(ty - y, tx - x)) - ang)


def use_zone(game) -> bool:
    x, y, _ = _sync(game)
    lv = _LEVEL
    if math.hypot(x - lv.end[0], y - lv.end[1]) < EXIT_REACH and lv.total - _TR.s < EXIT_REACH:
        return True
    return any(math.hypot(x - dx, y - dy) < USE_REACH for dx, dy in lv.doors)


def progress(game) -> float:
    _sync(game)
    return min(1.0, _TR.s / _LEVEL.total)


def bearing_token(bearing: float) -> int:
    """Token of the bearing bin; the aligned bin 'i' is [-2, 2)."""
    return VOCAB.index(BEARING_CHARS[bisect_right(BEARING_EDGES, bearing)])


def distance_token(dist: float) -> int:
    return VOCAB.index(DIST_CHARS[bisect_right(DIST_EDGES, dist)])


def stalled(game) -> bool:
    """The marine pushed forward for STALL_DECISIONS decisions and did not move (a monster or a wall is in the way)."""
    _sync(game)
    n = STALL_DECISIONS
    if len(_TR.pos) <= n or len(_TR.acts) < n:
        return False
    walking = {ACTIONS.index(c) for c in "WU"}
    (px, py), (x, y) = _TR.pos[-n - 1], _TR.pos[-1]
    return all(a in walking for a in _TR.acts[-n:]) and math.hypot(x - px, y - py) < STALL_DIST


def flag_value(game) -> int:
    if use_zone(game):
        return 2 if _TR.last == ACTIONS.index("U") else 1
    return 3 if stalled(game) else 0


def observe(state, game) -> list[int]:
    """Token context for one decision: [BOS, route bearing, enemy bearing, enemy distance, flag]."""
    en = threats(state, game)
    rb = bearing_token(compass(state, game))
    flag = VOCAB.index(DIST_CHARS[flag_value(game)])
    if not en:
        return [0, rb, VOCAB.index(NO_ENEMY), VOCAB.index(DIST_CHARS[-1]), flag]
    dist, bearing = en[0]
    return [0, rb, bearing_token(bearing), distance_token(dist), flag]


def _bin_words(char: str, noun: str) -> str:
    b = BEARING_CHARS.index(char)
    lo = ([-180] + BEARING_EDGES)[b]
    hi = (BEARING_EDGES + [180])[b]
    if lo == -2:
        return f"{noun} AHEAD"
    if lo < 0:
        return f"{noun} {-hi}-{-lo} DEG RIGHT"
    return f"{noun} {lo}-{hi} DEG LEFT"


def describe(tokens: list[int]) -> list[str]:
    """Short human-readable words for an observation context, e.g. ['EXIT 12-20 DEG LEFT', 'IMP 7-12 DEG LEFT', 'NEAR']."""
    chars = [VOCAB[t] for t in tokens if t != 0]
    if len(chars) < 4:
        return ["..."]
    words = [_bin_words(chars[0], "ROUTE")]
    if chars[1] != NO_ENEMY:
        words.append(_bin_words(chars[1], "ENEMY").replace("AHEAD", "IN SIGHTS"))
        words.append(["POINT BLANK", "NEAR", "MID RANGE", "FAR", "DISTANT"][DIST_CHARS.index(chars[2])])
    if chars[3] in "12":
        words.append("DOOR / SWITCH IN REACH")
    elif chars[3] == "3":
        words.append("BLOCKED")
    return words


def all_contexts() -> list[list[int]]:
    """Every context observe() can return: 17 route bearings x (no enemy | 17 bearings x 5 distances) x 4 flags."""
    ctx = []
    en = [(NO_ENEMY, DIST_CHARS[-1])] + [(b, d) for b in BEARING_CHARS for d in DIST_CHARS]
    for rb in BEARING_CHARS:
        for eb, ed in en:
            for fl in "0123":
                ctx.append([0, VOCAB.index(rb), VOCAB.index(eb), VOCAB.index(ed), VOCAB.index(fl)])
    return ctx


# ----------------------------------------------------------------------------- policy interface
def act_index(logits) -> int:
    """Index into ACTIONS of the largest ACTION-token logit (lowest index on ties); remembered as the last action."""
    idx = int(np.argmax(np.asarray(logits)[ACTION_IDS]))
    note_action(idx)
    return idx


def act(logits) -> list[float]:
    """Button vector [forward delta, turn delta, ATTACK, USE] from the model's logits at the last position.

    Only the logits of the ACTION tokens are compared; the other vocabulary entries are ignored."""
    return list(BUTTONS[ACTIONS[act_index(logits)]])


def _turn_toward(bearing: float) -> int:
    a = abs(bearing)
    return ACTIONS.index(("L" if bearing > 0 else "R") if a >= TURN_BIG_AT else ("l" if bearing > 0 else "r"))


def expert_index(state, game) -> int:
    """Scripted player with exact info (objects, route): index into ACTIONS.

    A speed-runner (gunfire wakes every monster in earshot, so he never shoots unless he must): follow the compass,
    turning only as far as needed and walking while he veers; press USE in the use zone (door / exit switch, released
    every other decision because USE is edge-triggered); when blocked (flag '3': pushing forward without moving, a
    monster is in the way) turn to the nearest visible monster and kill it.  Every threshold sits on a token bin
    edge, so the policy is a function of the tokens alone (the learner can match it exactly)."""
    en = threats(state, game)
    rb = compass(state, game)
    flag = flag_value(game)
    if en and flag == 3:
        dist, bearing = en[0]
        if abs(bearing) < (FIRE_TOL_NEAR if dist < DIST_EDGES[0] else FIRE_TOL):
            return ACTIONS.index("F")
        return _turn_toward(bearing)
    if flag in (1, 2) and abs(rb) < TURN_BIG_AT:
        return ACTIONS.index("W" if flag == 2 else "U")
    a = abs(rb)
    if a >= TURN_SMALL_AT:
        return _turn_toward(rb)
    return ACTIONS.index("W")


def play_episode(game, choose, seed: int | None = None, on_decision=None, max_decisions: int = 900) -> dict:
    """Run one episode.  choose(state, game) -> index into ACTIONS.  on_decision(state, game, idx) is called
    before every action.  Returns kills, progress (0..1 along the route), reached_exit, died, decisions."""
    if seed is not None:
        game.set_seed(seed)
    game.new_episode()
    decisions, prog = 0, 0.0
    while not game.is_episode_finished() and decisions < max_decisions:
        state = game.get_state()
        idx = choose(state, game)
        note_action(idx)
        prog = max(prog, progress(game))
        if on_decision is not None:
            on_decision(state, game, idx)
        buttons = list(BUTTONS[ACTIONS[idx]])
        for _ in range(TICS_PER_DECISION):                  # one tic at a time, exactly like the ouroboros harness
            game.make_action(buttons, 1)
            if game.is_episode_finished():
                break
        decisions += 1
    died = bool(game.is_player_dead())
    reached = bool(game.is_episode_finished() and not died and game.get_episode_time() < EPISODE_TICS - 1)
    return dict(kills=int(game.get_game_variable(_GV.KILLCOUNT)), progress=1.0 if reached else float(prog),
                reached_exit=reached, died=died, decisions=decisions, health=int(game.get_game_variable(_GV.HEALTH)))
