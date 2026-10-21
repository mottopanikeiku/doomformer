"""06 - A tiny neural network shoves the marine.

A 4-output integer MLP reads a two-component "wind" (wx, wy), each an integer in -2..2, and shoves the marine the way
the wind blows. The `world` hook builds a room with the real player in it; the `actions` hook attaches a
ThrustThing line special (df.act.thrust_player) to each output: every voodoo doll that adds a positive weight to
"east" / "north" / "west" / "south" crosses a line that gives the player a kick of fixed force in that direction.

    layer 0:  a = adc_signed(wind, [1, 2])                   in -2..2   (a 4-step signed staircase: gates in DOOM)
    layer 1:  h = [ax, ay, -ax, -ay]; a1 = adc_relu(h, [1, 2])  in 0..2 (two relu stairs: up to two active lanes)
    layer 2:  y = I @ a1                                      y_east, y_north, y_west, y_south

Each active lane of a hidden unit is one doll, one kick: a wind of +2 gives two kicks east. (Kicks fire per positive
contribution unit, not on the final output value; see docs/BUILD_YOUR_OWN.md.) The map's ENTER script gives the
player the TID that the thrust lines aim at; a second script puts him back in the middle between runs.

DOOM computes the network (verified exactly against the integer reference) AND the physics (ZDoom's own friction and
thrust). There is no integer reference for the physics, so only the direction is checked and the displacement printed.

    python examples/06_shove_the_marine.py

Writes examples/out/06_shove_the_marine.png. Exits non-zero on a mismatch or if he moved the wrong way.
"""
import sys
from pathlib import Path

import numpy as np
import vizdoom as vzd
from PIL import Image, ImageDraw

import doomformer as df
from doomformer.udmf import MapBuilder

OUT = Path(__file__).resolve().parent / "out"
PLAYER_TID = 900
CX, CY = -640, -2688                     # the middle of the room, where he is put before each run
FORCE = 6                                # ThrustThing force of one kick

# angles are in degrees: 0 east, 90 north, 180 west, 270 south
net = df.MLP(
    weights=[np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), np.eye(4, dtype=np.int64)],
    biases=[np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)],
    thresholds=[np.array([1, 2]), np.array([1, 2])],
    signed_input=True,
)
DIRECTIONS = [("east", 0, (1, 0)), ("north", 90, (0, 1)), ("west", 180, (-1, 0)), ("south", 270, (0, -1))]
actions = {f"y{i}": [df.act.thrust_player(angle, FORCE, PLAYER_TID)] for i, (_, angle, _) in enumerate(DIRECTIONS)}


def world(m: MapBuilder) -> str:
    m.rect(CX - 512, CY - 512, CX + 512, CY + 512, floor=0, ceil=192, light=210, ctex="CEIL5_1", ftex="FLOOR4_8",
           wall="STARTAN3")
    for gx in range(-3, 4):                                  # a grid of columns, so motion is visible on screen
        for gy in range(-3, 4):
            if (gx, gy) != (0, 0) and (gx + gy) % 2 == 0:
                m.thing(CX + gx * 128 + 40, CY + gy * 128 + 40, 48)
    m.thing(CX, CY, 1, angle=0)                              # last player-1 start = the real player
    return f"""
script 200 ENTER {{ Thing_ChangeTID(0, {PLAYER_TID}); }}
script 202 (void) {{ SetActorPosition({PLAYER_TID}, {CX}.0, {CY}.0, 0.0, 0); SetActorVelocity({PLAYER_TID}, 0.0, 0.0, 0.0, 0, 0); }}
script 203 (void) {{ o1 = GetActorX({PLAYER_TID}) >> 16; o2 = GetActorY({PLAYER_TID}) >> 16; }}
"""


def where(brain) -> tuple[int, int]:
    x, y = brain.run_script(203, read=2)                     # world script 203 -> USER1, USER2
    return x, y


def snapshot(game: vzd.DoomGame) -> Image.Image:
    return Image.fromarray(np.transpose(game.get_state().screen_buffer, (1, 2, 0)))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    ok, shots = True, []
    with df.compile(net, world=world, actions=actions, resolution=vzd.ScreenResolution.RES_320X240) as brain:
        game = brain.runner.game
        game.advance_action(3)                                # ENTER script runs: the player gets his TID
        print(f"map: {brain.info['dolls']} dolls, {brain.info['sectors']} sectors")
        for wind in [(0, 0), (2, 0), (0, -1), (-2, 2)]:
            brain.run_script(202)                              # put him back in the middle, at rest
            game.advance_action(1)
            x0 = where(brain)
            before = snapshot(game)
            y = brain(np.array(wind))                          # DOOM runs the net; the thrust lines fire inside it
            game.advance_action(70)                            # let him slide to a stop (friction is ZDoom's)
            x1 = where(brain)
            d = (x1[0] - x0[0], x1[1] - x0[1])
            kicks = {name: int(y[i]) for i, (name, _, _) in enumerate(DIRECTIONS)}
            good = True
            for i, (name, _, (ux, uy)) in enumerate(DIRECTIONS):
                if kicks[name] > 0:
                    good &= d[0] * ux + d[1] * uy > 8       # moved clearly toward each direction that got a kick
            if not any(kicks.values()):
                good &= abs(d[0]) + abs(d[1]) <= 1            # no kick, no movement
            if kicks["east"] and not kicks["west"]:
                good &= d[0] > 0
            ok &= good
            print(f"wind {wind}: DOOM y={y.tolist()} (EXACT vs reference) kicks={kicks}  player ({x0[0]},{x0[1]}) -> "
                  f"({x1[0]},{x1[1]})  delta {d}  {'moved as the net decided' if good else 'WRONG MOVE'}")
            shots.append((wind, d, before, snapshot(game)))
    sheet = Image.new("RGB", (320 * 4 + 12, 240 * 2 + 4), (0, 0, 0))
    for i, (wind, d, before, after) in enumerate(shots):
        for row, im in enumerate((before, after)):
            im = im.copy()
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 319, 12], fill=(0, 0, 0))
            dr.text((4, 1), f"wind={wind}  " + ("before" if row == 0 else f"after: moved {d}"), fill=(255, 255, 255))
            sheet.paste(im, (i * 324, row * 244))
    sheet.save(OUT / "06_shove_the_marine.png")
    print(f"wrote {OUT / '06_shove_the_marine.png'}")
    print("OK: every pass exact; the marine moved as the network decided" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
