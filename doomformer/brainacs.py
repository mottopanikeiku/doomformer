"""ACS building blocks that let the MAP see and act for the marine (no Python, no host token injection).

Used by the level builders. Probe maps measured every claim below.

SCRIPT_PISTOL  DECORATE lump text: weapon "ScriptPistol" + inventory item "ScriptFireToken". The weapon is a pistol (Clip
               ammo, 1 per shot, FBF_NORANDOM = exact aim along the player's CURRENT angle). Give the player the weapon in
               his ENTER script (PISTOL_ENTER), then GiveInventory("ScriptFireToken", 1) pulls the trigger: one bullet on
               the next tic, ~16 tics until the next. Demons must be awake (Thing_Activate): dormant ones are immune.
sensor_acs()   ACS source of `bl_sense()`: policy.observe's two tokens of the nearest live demon (VOCAB indexes, so the
               same numbers a Python host would feed the model) into globals bl_bearing / bl_distance.
DECORATE lump: pass `extra_lumps=[("DECORATE", SCRIPT_PISTOL.encode())]` to udmf.write_wad (it goes in front of the map).

Measured limits of the sensor (3000 natural seed-4 snapshots with 0..22 live demons, 1664
bearing-edge, 264 distance-edge and 300 tie cases):
  - distance token: exact (VectorLength is exact to 1/65536; edges compared in fixed point; 0 mismatches incl. +-1/65536).
  - bearing token: 0 mismatches unless the true bearing is within 1e-4 degrees of a bin edge (72/128 there; 0/1536 beyond
    0.002 deg). VectorAngle alone is only good to ~0.032 deg, so within MARGIN units (0.066 deg) of an edge an exact
    cross-product test decides, using Taylor sin/cos (ACS Sin/Cos are good to only ~50/65536, unusable).
  - ties: nearest by (distance, signed bearing) like sorted((dist, bearing)); equal distances whose bearings differ by
    < ~0.03 deg could pick the other demon (not seen in 291 exact-distance ties).
  - the player is NOT stopped and may be sliding (a bite thrusts the marine for ~20 tics). The token describes the world at
    the sense tic; if he slides afterwards, policy.observe on a later state differs (film decision 31: sampled bearing
    20.70 deg, token q, vs 12.92 deg rested, token p). So the sensor exports the player's raw pose at the sample
    ({prefix}_px, _py = GetActorX/Y, _pa = GetActorAngle) and a verifier must evaluate policy.observe on the frozen demons
    with the player's coordinates / angle overridden by these (the demons are dormant and do not move).
  - activator: the sensor walks the demons with SetActivator, so it LEAVES THE ACTIVATOR = the player (TID player_tid).
    Call it from a script whose activator is the player, or do any activator-dependent work first.
  - temp TIDs tmp_tid .. tmp_tid + n - 1 (n = demons sharing demon_tid, dead included) exist during the call only; all are
    restored to demon_tid (0 leftovers in every probe snapshot). Dead demons keep demon_tid; they are skipped by health.
"""
from __future__ import annotations

import math

from . import policy

PLAYER_TID = 900          # the real player's TID (ENTER script: Thing_ChangeTID(0, PLAYER_TID))
TMP_TID = 60000           # first temporary TID used while the sensor iterates the demons
TURN_UNITS = 640          # one held-key turn tic, in 1/65536 of a circle (policy.L = +640 left, R = -640)
MARGIN = 12               # units (0.066 deg) around a bearing edge where the exact cross-product test decides

SCRIPT_PISTOL = """
ACTOR ScriptFireToken : Inventory { Inventory.MaxAmount 1 Inventory.InterHubAmount 0 }
ACTOR ScriptPistol : Weapon
{
  Weapon.SelectionOrder 1900
  Weapon.AmmoUse 1
  Weapon.AmmoGive 0
  Weapon.AmmoType "Clip"
  States
  {
  Ready:
    PISG A 0 A_JumpIfInventory("ScriptFireToken", 1, "Fire")
    PISG A 1 A_WeaponReady(WRF_NOFIRE)
    Loop
  Select:
    PISG A 1 A_Raise
    Loop
  Deselect:
    PISG A 1 A_Lower
    Loop
  Fire:
    PISG A 0 A_TakeInventory("ScriptFireToken", 1)
    PISG A 4 A_FireBullets(0, 0, 1, 5, "BulletPuff", FBF_USEAMMO|FBF_NORANDOM)
    Goto Ready
  }
}
"""
PISTOL_ENTER = 'GiveInventory("ScriptPistol", 1); SetWeapon("ScriptPistol");'    # statements for the ENTER script

_BO = policy.VOCAB.index(policy.BEARING_CHARS[0])
_DO = policy.VOCAB.index(policy.DIST_CHARS[0])
NO_ENEMY_TOKENS = (policy.VOCAB.index(policy.NO_ENEMY), policy.VOCAB.index(policy.DIST_CHARS[-1]))   # (bearing, dist)


def sensor_acs(demon_tid: int, prefix: str = "bl", player_tid: int = PLAYER_TID, tmp_tid: int = TMP_TID) -> str:
    """ACS (after `#include "zcommon.acs"`) defining, with names starting `{prefix}_`:

      int {prefix}_bearing, {prefix}_distance   policy.VOCAB token indexes of the nearest live demon (as policy.observe's
                                                [BOS, bearing, distance][1:]); no live demon: NO_ENEMY_TOKENS
      int {prefix}_px, {prefix}_py, {prefix}_pa   the player's GetActorX / GetActorY (16.16) / GetActorAngle (16-bit) at the
                                                sample, captured before any selection
      int {prefix}_alive    live demons;  {prefix}_iter  demons visited (dead included, <= ThingCount(demon_tid) at entry)
      int {prefix}_dist_fx  nearest distance, 16.16;  {prefix}_bear16  its signed bearing, 16-bit angle (+ = left)
      function void {prefix}_sense (void)       refreshes all of the above (one tic, no Delay); see the module doc for
                                                the activator and temp-TID rules
    Names {prefix}_tsin/_tcos/_fsin/_fcos/_btok/_dtok are helpers."""
    p = prefix
    b_thr = [math.ceil(e * 65536 / 360) for e in policy.BEARING_EDGES]            # signed 16-bit angle units
    d_thr = [e << 16 for e in policy.DIST_EDGES]                                  # 16.16
    ce = [round(math.cos(math.radians(e)) * 65536) for e in policy.BEARING_EDGES]
    se = [round(math.sin(math.radians(e)) * 65536) for e in policy.BEARING_EDGES]
    dtok = " + ".join([str(_DO)] + [f"(d >= {t})" for t in d_thr])
    bt = "".join(f"  if (b > {t + MARGIN}) tok++;\n"
                 f"  else if (b >= {t - MARGIN}) {{ if (FixedMul(ry, {c}) - FixedMul(rx, {s}) >= 0) tok++; }}\n"
                 for t, c, s in zip(b_thr, ce, se))
    return f"""
int {p}_bearing = {NO_ENEMY_TOKENS[0]};  int {p}_distance = {NO_ENEMY_TOKENS[1]};
int {p}_alive = 0;  int {p}_iter = 0;  int {p}_dist_fx = 0;  int {p}_bear16 = 0;
int {p}_px = 0;  int {p}_py = 0;  int {p}_pa = 0;

// ACS Sin / Cos are only good to ~50/65536 (the engine's 8192-step table), so: Taylor series on [-45, 45] deg
// (r = signed 16-bit angle, x = r in radians as 16.16) and quadrant folding; good to ~3/65536
function int {p}_tsin (int r) {{
  int x = r * 6 + ((r * 18559) >> 16);  int x2 = FixedMul(x, x);
  return FixedMul(x, 65536 - FixedMul(x2, 10923 - FixedMul(x2, 546 - FixedMul(x2, 13))));
}}
function int {p}_tcos (int r) {{
  int x = r * 6 + ((r * 18559) >> 16);  int x2 = FixedMul(x, x);
  return 65536 - FixedMul(x2, 32768 - FixedMul(x2, 2731 - FixedMul(x2, 91)));
}}
function int {p}_fsin (int a) {{
  int k = ((a + 8192) >> 14) & 3;  int r = ((a + 8192) & 16383) - 8192;
  if (k == 0) return {p}_tsin(r);
  if (k == 1) return {p}_tcos(r);
  if (k == 2) return -{p}_tsin(r);
  return -{p}_tcos(r);
}}
function int {p}_fcos (int a) {{
  int k = ((a + 8192) >> 14) & 3;  int r = ((a + 8192) & 16383) - 8192;
  if (k == 0) return {p}_tcos(r);
  if (k == 1) return -{p}_tsin(r);
  if (k == 2) return -{p}_tcos(r);
  return {p}_tsin(r);
}}
// bearing token of the vector (rx, ry) (the demon in the player's frame, x ahead, y left); b = its VectorAngle-based
// signed 16-bit bearing, good to ~0.033 deg, which decides every bin except within {MARGIN} units of an edge, where the
// exact sign of the cross product with the edge direction decides
function int {p}_btok (int b, int rx, int ry) {{
  int tok = {_BO};
  if (b >= {32768 - MARGIN} || b <= {-32768 + MARGIN}) {{ if (ry >= 0) return {_BO + len(b_thr)}; return {_BO}; }}
{bt}  return tok;
}}
function int {p}_dtok (int d) {{ return {dtok}; }}

function void {p}_sense (void) {{
  int px = GetActorX({player_tid});  int py = GetActorY({player_tid});  int pa = GetActorAngle({player_tid});
  {p}_px = px;  {p}_py = py;  {p}_pa = pa;
  int n = 0;  int alive = 0;  int bd = 0;  int bb = 0;  int bdx = 0;  int bdy = 0;
  int dx;  int dy;  int d;  int b;  int i;  int c;  int sn;  int rx;  int ry;
  while (SetActivator({demon_tid})) {{
    if (GetActorProperty(0, APROP_Health) > 0) {{
      dx = GetActorX(0) - px;  dy = GetActorY(0) - py;
      d = VectorLength(dx, dy);
      b = (VectorAngle(dx, dy) - pa) & 65535;
      if (b >= 32768) b -= 65536;
      if (alive == 0 || d < bd || (d == bd && b < bb)) {{ bd = d;  bb = b;  bdx = dx;  bdy = dy; }}
      alive++;
    }}
    Thing_ChangeTID(0, {tmp_tid} + n);
    n++;
  }}
  for (i = 0; i < n; i++) Thing_ChangeTID({tmp_tid} + i, {demon_tid});
  SetActivator({player_tid});
  {p}_alive = alive;  {p}_iter = n;  {p}_dist_fx = bd;  {p}_bear16 = bb;
  if (alive == 0) {{ {p}_bearing = {NO_ENEMY_TOKENS[0]};  {p}_distance = {NO_ENEMY_TOKENS[1]}; return; }}
  c = {p}_fcos(pa);  sn = {p}_fsin(pa);
  rx = FixedMul(bdx, c) + FixedMul(bdy, sn);  ry = FixedMul(bdy, c) - FixedMul(bdx, sn);
  {p}_bearing = {p}_btok(bb, rx, ry);  {p}_distance = {p}_dtok(bd);
}}
"""

