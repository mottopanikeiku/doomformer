"""The marine's arena, built into the SAME map as his brain (compile_pure(world=...)).

One map, one DOOM game: the real player stands in this room, and every voodoo doll computing his decisions is a
copy of him (Doom spawns the last player-1 start as the player; the earlier ones become his voodoo dolls).

Rules, close to ViZDoom's defend_the_center: a square room, the marine fixed in the centre (he can only turn and
fire), pistol with 26 bullets, one-hit demons spawn on a ring and keep coming (ARENA_DEMONS alive at once).
Time stands still while he thinks: script FREEZE makes every demon dormant (Thing_Deactivate) and pauses the
spawner; UNFREEZE wakes them. Deaths are lives: he is in buddha mode (damage cannot take him below 1 health); a
life ends when health reaches 1, when the 26 bullets are spent (buddha would otherwise let him punch on forever),
or after LIFE_TICS awake tics. NEW_LIFE clears the demons and restores health and ammo -- the episode, and so the
brain, is never reloaded.
"""
from __future__ import annotations

import math

import vizdoom as vzd

from doomformer.udmf import MapBuilder

CX, CY, HALF = 6144, -6144, 640        # room centre and half-size (well away from the brain at y >= 0)
DEMON_TID, NEW_TID, SPOT_TID0 = 40000, 40001, 41000
DEMON_HEALTH = 1                       # one pistol hit, like defend_the_center's demons
N_SPOTS, ARENA_DEMONS, RESPAWN_TICS = 16, 3, 35      # aimbot, seed 4, 5 lives: 9, 26, 26, 25, 26 kills
SPOT_R = 560
AMMO, LIFE_TICS = 26, 2100
FOLLOW_MAX = 20                        # after pulling the trigger: awake tics (hands off) until the bullet is out
FREEZE, UNFREEZE, NEW_LIFE = 200, 201, 202
_GV = vzd.GameVariable
GAME_VARIABLES = [_GV.ANGLE, _GV.POSITION_X, _GV.POSITION_Y, _GV.AMMO2, _GV.HEALTH, _GV.KILLCOUNT]


def build(m: MapBuilder) -> str:
    """Room, demon spawn spots and the marine (placed last, so he is the player); returns the arena ACS."""
    m.rect(CX - HALF, CY - HALF, CX + HALF, CY + HALF, floor=0, ceil=256, light=200, ftex="FLOOR5_1",
           ctex="CEIL3_5", wall="STARTAN3")
    for k in range(N_SPOTS):
        a = 2 * math.pi * k / N_SPOTS
        x, y = CX + SPOT_R * math.cos(a), CY + SPOT_R * math.sin(a)
        facing = int(round(math.degrees(a) + 180)) % 360
        m.thing(round(x), round(y), 9001, angle=facing, id=SPOT_TID0 + k)          # MapSpot
    m.thing(CX, CY, 1, angle=90)                                                    # LAST player-1 start: him
    return f"""
int arena_awake = 0;
int arena_cooldown = 0;
function int spawn_demon (int spot) {{
  if (!SpawnSpotFacing("Demon", spot, {NEW_TID})) return 0;
  SetActorProperty({NEW_TID}, APROP_Health, {DEMON_HEALTH});
  Thing_ChangeTID({NEW_TID}, {DEMON_TID});
  return 1;
}}
script {FREEZE} (void) {{ arena_awake = 0; Thing_Deactivate({DEMON_TID}); }}
script {UNFREEZE} (void) {{ Thing_Activate({DEMON_TID}); arena_awake = 1; }}
script {NEW_LIFE} (void) {{
  Thing_Remove({DEMON_TID});
  SetActorPosition(0, {CX} << 16, {CY} << 16, 0, 0);
  SetActorAngle(0, 0.25);
  SetActorProperty(0, APROP_Health, 100);
  TakeInventory("Clip", 999);
  GiveInventory("Clip", {AMMO});
  int quarter = {N_SPOTS // ARENA_DEMONS};
  for (int k = 0; k < {ARENA_DEMONS}; k++)
    spawn_demon({SPOT_TID0} + k * quarter + random(0, quarter - 1));
  arena_cooldown = {RESPAWN_TICS};
}}
script 203 OPEN {{
  while (TRUE) {{
    if (arena_awake) {{
      if (arena_cooldown > 0) arena_cooldown--;
      if (arena_cooldown == 0 && ThingCount(T_NONE, {DEMON_TID}) < {ARENA_DEMONS}) {{
        if (spawn_demon({SPOT_TID0} + random(0, {N_SPOTS - 1})))
          arena_cooldown = {RESPAWN_TICS};
      }}
    }}
    Delay(1);
  }}
}}
script 204 ENTER {{ SetPlayerProperty(0, 1, PROP_BUDDHA); }}
"""


def life_over(g: vzd.DoomGame, awake_tics: int) -> bool:
    return (g.get_game_variable(_GV.HEALTH) <= 1 or g.get_game_variable(_GV.AMMO2) <= 0
            or awake_tics >= LIFE_TICS)


# DOOM eases into turns: a freshly pressed turn key turns at half speed for its first tics, and he lets go of every
# key while he thinks, so held-key turning would always be slow. Exact per-tic turning instead, at the held-key
# speed (640/65536 of a circle = 3.515625 degrees per tic); positive delta turns right.
TURN_DEG = 640 * 360 / 65536
HANDS_OFF = [0.0, 0]


def buttons(action: str) -> list:
    """[TURN_LEFT_RIGHT_DELTA, ATTACK] for "L", "R" or "F"."""
    return {"L": [-TURN_DEG, 0], "R": [TURN_DEG, 0], "F": [0.0, 1]}[action]


def configure(g: vzd.DoomGame):
    """TileRunner configure hook: the arena's buttons, variables and HUD on top of the brain's setup."""
    g.set_available_buttons([vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.ATTACK])
    for v in GAME_VARIABLES:
        g.add_available_game_variable(v)
    g.set_doom_skill(3)
    g.set_render_hud(True)
    g.set_screen_format(vzd.ScreenFormat.CRCGCB)
