"""Line specials for `df.compile(..., actions={output_name: [specials]})`: let a brain's arithmetic touch the world.

Semantics (exactly those of `pure.compile_pure`): every voodoo doll that adds a POSITIVE weight to the named
output also crosses these specials. An action therefore fires once per positive contribution unit (a doll of an
input unit whose weight onto that output is > 0 and that passed its staircase gate), as the sum is being formed;
it does not look at the output's final value, its bias, or negative contributions. Example: y0 = 1 + 2*a0 - 5*a1
with a0 = 3, a1 = 1 fires y0's actions 3 times (y0 = 2).

Each helper returns a plain dict (special = the ZDoom line special number from zspecial.acs, arg0..arg4), so any
other ZDoom special can be written the same way by hand. The activator of every special is a voodoo doll, never
the player, so specials that act on "the activator" (tid 0, tag 0) would hit the computing machinery; the helpers
therefore insist on explicit world tags/TIDs.

Numbers a world may use without colliding with the brain (see `kit` for the world hook):
  sector tags 3..999 (the brain uses 1, 2 and >= 1000), thing TIDs 1..9999 (dolls and their reset spots use
  >= 10000), ACS scripts >= 200 (the brain uses 1..7 and 100 + readout page).
"""
from __future__ import annotations

THRUST_THING, DOOR_OPEN, LIGHT_CHANGE_TO_VALUE, ACS_EXECUTE_ALWAYS = 72, 11, 112, 226
WORLD_TAGS = range(3, 1000)
WORLD_TIDS = range(1, 10000)
WORLD_SCRIPT0 = 200


def _within(value: int, allowed: range, what: str) -> int:
    value = int(value)
    if value not in allowed:
        raise ValueError(f"{what} {value} outside {allowed.start}..{allowed.stop - 1} (other numbers belong to the "
                         "brain's machinery)")
    return value


def thrust_player(angle: float, force: int, tid: int, nolimit: bool = False) -> dict:
    """ThrustThing (72) on the thing with TID `tid` (give the player a TID in an ENTER script of your world, e.g.
    ``script 200 ENTER { Thing_ChangeTID(0, 900); }``). `angle` in degrees (0 = east, 90 = north), converted to
    ZDoom's byte angle (256 per turn); `force` in map units per tic; ZDoom caps the result at 30 units/tic
    unless `nolimit`. Thrusts add up: N firings push N times."""
    return dict(special=THRUST_THING, arg0=round(angle * 256 / 360) % 256, arg1=int(force), arg2=int(bool(nolimit)),
                arg3=_within(tid, WORLD_TIDS, "TID"))


def door_open(tag: int, speed: int = 16) -> dict:
    """Door_Open (11): raise the ceiling of every sector tagged `tag` to 4 below its lowest neighbouring ceiling
    and leave it open. `speed` in 1/8 units per tic (16 = DOOM's normal door speed)."""
    return dict(special=DOOR_OPEN, arg0=_within(tag, WORLD_TAGS, "sector tag"), arg1=int(speed), arg2=0)


def light(tag: int, value: int) -> dict:
    """Light_ChangeToValue (112): set the light level of every sector tagged `tag` to `value` (idempotent)."""
    return dict(special=LIGHT_CHANGE_TO_VALUE, arg0=_within(tag, WORLD_TAGS, "sector tag"), arg1=int(value))


def script(n: int, *args: int) -> dict:
    """ACS_ExecuteAlways (226): start a new instance of world script `n` (>= 200) with up to 3 int arguments,
    once per firing. Its activator is the voodoo doll that crossed the line."""
    if len(args) > 3:
        raise ValueError("ACS_ExecuteAlways passes at most 3 script arguments")
    if int(n) < WORLD_SCRIPT0:
        raise ValueError(f"world scripts are numbered >= {WORLD_SCRIPT0} (lower numbers belong to the brain)")
    a = [int(v) for v in args] + [0] * (3 - len(args))
    return dict(special=ACS_EXECUTE_ALWAYS, arg0=int(n), arg1=0, arg2=a[0], arg3=a[1], arg4=a[2])
