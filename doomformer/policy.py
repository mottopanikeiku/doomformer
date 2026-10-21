"""DOOM marine policy interface for the "ouroboros" demo: a transformer (computed by DOOM games) plays DOOM.

Scenario: ViZDoom's built-in ``defend_the_center`` (cfg/wad from the vizdoom package ``scenarios/`` dir).
The marine stands in the middle of a round room with a pistol and 26 bullets; it can only turn left, turn
right and shoot while demons ("Demon") and melee zombies ("MarineChainsawVzd") walk in from every side.
Score = kills (an episode lasts at most 2100 game tics and ends when the marine dies).

One decision = one transformer call:
    observe(state, game) -> [0, bearing, distance]        3 tokens, BOS first (DOOM cost grows with positions)
    logits = model(tokens)[-1]                             any model; here the integer QModel
    buttons = act(logits)                                  argmax over the ACTION tokens only
    game.make_action(buttons, TICS_PER_DECISION)

Tokens (every token is one character of ``VOCAB``; index 0 is '$' = BOS):
    actions    'L' turn left, 'R' turn right, 'F' fire                      (``ACTIONS``)
    bearing    17 characters 'a'..'q': angle of the NEAREST enemy relative to the marine's aim, in degrees,
               positive = to the left.  'a' = [-180,-100)  ...  'i' = [-2,2] (aligned)  ...  'q' = (100,180];
               '?' = no enemy alive
    distance   5 characters '0'..'4': distance to that enemy, bin edges 100/200/350/550 map units
The most threatening enemy is the nearest one (it is the one that bites first).

Host restriction: the unembedding has one logit per vocab token, but only the three action tokens are
meaningful.  ``act`` takes the argmax over the action tokens' logits only (lowest index on ties); the host
(an evaluation loop or a self-running demo) must never sample or argmax over the whole vocabulary.

UI helpers: ``ACTION_NAMES`` (labels aligned with ACTIONS) and ``describe(tokens)`` (words for a context).

Expert: ``expert_action`` is a scripted aimbot with exact positions (turn toward the nearest enemy, fire when the
aim error is inside the hit cone atan(18/dist), at least 2 degrees).  The learned policy only sees the binned
tokens, which is what limits its agreement with the expert.
"""
from __future__ import annotations

import math
import os
from bisect import bisect_right

import numpy as np
import vizdoom as vzd

from .runtime import init_game

ACTIONS: list[str] = ["L", "R", "F"]                    # token chars; button order of the scenario cfg
BUTTONS: dict[str, list[int]] = {"L": [1, 0, 0], "R": [0, 1, 0], "F": [0, 0, 1]}   # TURN_LEFT, TURN_RIGHT, ATTACK
ACTION_NAMES: list[str] = ["TURN LEFT", "TURN RIGHT", "FIRE"]      # human labels aligned with ACTIONS

BEARING_EDGES = [-100, -60, -35, -20, -12, -7, -4, -2, 2, 4, 7, 12, 20, 35, 60, 100]   # degrees, 17 bins
BEARING_CHARS = "abcdefghijklmnopq"
NO_ENEMY = "?"
DIST_EDGES = [100, 200, 350, 550]                       # map units, 5 bins
DIST_CHARS = "01234"
VOCAB: str = "$" + "".join(ACTIONS) + BEARING_CHARS + NO_ENEMY + DIST_CHARS
assert len(VOCAB) <= 32 and VOCAB[0] == "$"

ACTION_IDS: list[int] = [VOCAB.index(c) for c in ACTIONS]
TICS_PER_DECISION = 4                                   # game tics the chosen buttons are held
CONTEXT_LEN = 3
ENEMIES = ("Demon", "MarineChainsawVzd")
HIT_RADIUS = 18.0                                       # expert hit cone: atan(HIT_RADIUS / dist)
MIN_TOL = 2.0                                           # degrees
SCENARIO_CFG = os.path.join(vzd.scenarios_path, "defend_the_center.cfg")
SCENARIO_WAD = os.path.join(vzd.scenarios_path, "defend_the_center.wad")
_GV = vzd.GameVariable
GAME_VARIABLES = [_GV.ANGLE, _GV.POSITION_X, _GV.POSITION_Y, _GV.AMMO2, _GV.HEALTH, _GV.KILLCOUNT]


def make_game(render: bool = False, resolution=None) -> vzd.DoomGame:
    """defend_the_center with objects info enabled.

    render=False: fastest headless setup (160x120, no HUD).  render=True: 320x240 (or `resolution`) screen
    buffer with the HUD so the host can grab frames (`game.get_state().screen_buffer`).  The window itself is
    never opened.  Seed with ``game.set_seed(s)`` before ``game.new_episode()``.
    """
    g = vzd.DoomGame()
    g.load_config(SCENARIO_CFG)
    g.set_doom_scenario_path(SCENARIO_WAD)
    g.set_window_visible(False)
    g.set_sound_enabled(False)
    g.set_objects_info_enabled(True)
    g.set_available_game_variables(GAME_VARIABLES)
    if resolution is None:
        resolution = vzd.ScreenResolution.RES_320X240 if render else vzd.ScreenResolution.RES_160X120
    g.set_screen_resolution(resolution)
    g.set_render_hud(bool(render))
    g.set_screen_format(vzd.ScreenFormat.CRCGCB)            # planar RGB, CHW
    init_game(g)
    return g


def _norm(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def enemies(state, game) -> list[tuple[float, float]]:
    """[(distance, bearing in degrees, positive = left of the aim)] of every live enemy, nearest first."""
    px, py = game.get_game_variable(_GV.POSITION_X), game.get_game_variable(_GV.POSITION_Y)
    ang = game.get_game_variable(_GV.ANGLE)
    out = []
    for o in state.objects:
        if o.name in ENEMIES:
            dx, dy = o.position_x - px, o.position_y - py
            out.append((math.hypot(dx, dy), _norm(math.degrees(math.atan2(dy, dx)) - ang)))
    return sorted(out)


def bearing_token(bearing: float) -> int:
    """Token of the bearing bin; the aligned bin 'i' is [-2, 2)."""
    return VOCAB.index(BEARING_CHARS[bisect_right(BEARING_EDGES, bearing)])


def distance_token(dist: float) -> int:
    return VOCAB.index(DIST_CHARS[bisect_right(DIST_EDGES, dist)])


def observe(state, game) -> list[int]:
    """Token context for one decision: [BOS, bearing token, distance token] of the nearest enemy."""
    en = enemies(state, game)
    if not en:
        return [0, VOCAB.index(NO_ENEMY), VOCAB.index(DIST_CHARS[-1])]
    dist, bearing = en[0]
    return [0, bearing_token(bearing), distance_token(dist)]


def describe(tokens: list[int]) -> list[str]:
    """Short human-readable words for an observation context, e.g. ['ENEMY 7-12 DEG LEFT', 'NEAR']."""
    chars = [VOCAB[t] for t in tokens if t != 0]
    if not chars or chars[0] == NO_ENEMY:
        return ["NO ENEMY"]
    b, dist = BEARING_CHARS.index(chars[0]), DIST_CHARS.index(chars[1])
    lo = ([-180] + BEARING_EDGES)[b]
    hi = (BEARING_EDGES + [180])[b]
    if lo == -2:
        side = "ENEMY IN SIGHTS"
    elif lo < 0:
        side = f"ENEMY {-hi}-{-lo} DEG RIGHT"
    else:
        side = f"ENEMY {lo}-{hi} DEG LEFT"
    return [side, ["POINT BLANK", "NEAR", "MID RANGE", "FAR", "DISTANT"][dist]]


def all_contexts() -> list[list[int]]:
    """Every context observe() can return (17 bearings x 5 distances, plus the no-enemy context)."""
    ctx = [[0, VOCAB.index(b), VOCAB.index(d)] for b in BEARING_CHARS for d in DIST_CHARS]
    return ctx + [[0, VOCAB.index(NO_ENEMY), VOCAB.index(DIST_CHARS[-1])]]


def act_index(logits) -> int:
    """Index into ACTIONS of the largest ACTION-token logit (lowest index on ties)."""
    return int(np.argmax(np.asarray(logits)[ACTION_IDS]))


def act(logits) -> list[int]:
    """Button vector [TURN_LEFT, TURN_RIGHT, ATTACK] from the model's logits at the last position.

    Only the logits of the ACTION tokens ('L','R','F') are compared; the other vocabulary entries are ignored
    (the host restricts the choice to action tokens)."""
    return list(BUTTONS[ACTIONS[act_index(logits)]])


def expert_index(state, game) -> int:
    """Scripted aimbot with exact positions: index into ACTIONS."""
    en = enemies(state, game)
    if not en:
        return ACTIONS.index("L")                      # nothing to shoot: scan
    dist, bearing = en[0]
    if abs(bearing) <= max(MIN_TOL, math.degrees(math.atan2(HIT_RADIUS, dist))):
        return ACTIONS.index("F")
    return ACTIONS.index("L" if bearing > 0 else "R")


def expert_action(state, game) -> list[int]:
    return list(BUTTONS[ACTIONS[expert_index(state, game)]])


def play_episode(game, choose, seed: int | None = None, on_decision=None) -> dict:
    """Run one episode.  choose(state, game) -> index into ACTIONS.  on_decision(state, game, idx) is called
    before every action.  Returns kills, survival tics, remaining ammo/health, whether the marine died."""
    if seed is not None:
        game.set_seed(seed)
    game.new_episode()
    decisions = 0
    while not game.is_episode_finished():
        state = game.get_state()
        idx = choose(state, game)
        if on_decision is not None:
            on_decision(state, game, idx)
        game.make_action(list(BUTTONS[ACTIONS[idx]]), TICS_PER_DECISION)
        decisions += 1
    return dict(kills=int(game.get_game_variable(_GV.KILLCOUNT)), tics=decisions * TICS_PER_DECISION,
                decisions=decisions, ammo=int(game.get_game_variable(_GV.AMMO2)),
                health=int(game.get_game_variable(_GV.HEALTH)), dead=bool(game.is_player_dead()))
