"""03 - A neural network that opens a door.

An integer MLP lives in the same DOOM map as the player. Two "switches" a and b are its inputs:

    hidden 0 = a AND b          output "y0" (door)  <- hidden 0
    hidden 1 = a OR  b          output "y1" (light) <- hidden 1

The `world` hook adds a dim room with the player, a closed door and a chamber with a BFG behind it. The `actions` hook
attaches extra line specials to the outputs: every voodoo doll that adds a POSITIVE weight to "y0" also crosses a
Door_Open line, and every doll that adds a positive weight to "y1" crosses a Light_ChangeToValue line. So the network's
own arithmetic opens the door and switches on the light; Python never touches either.

Semantics worth knowing (docs/BUILD_YOUR_OWN.md): an action fires once per positive contribution unit that actually
passes its gate, NOT on the final value of the output. Here both weights are +1 and both actions are idempotent, so
that is the same thing as "fires when the output is positive".

The player's view is saved after each of three runs: (a, b) = (0,0), (1,0), (1,1).

    python examples/03_brain_opens_doors.py

Writes examples/out/03_brain_opens_doors.png (+ 03_frame_*.png). Exits non-zero on any mismatch or if the door/light
did not end up where the reference says.
"""
import sys
from pathlib import Path

import numpy as np
import vizdoom as vzd
from PIL import Image, ImageDraw

import doomformer as df
from doomformer.udmf import MapBuilder

OUT = Path(__file__).resolve().parent / "out"
LIGHT_TAG, DOOR_TAG = 101, 102          # sector tags 3..999 are free for your world
DIM, BRIGHT = 64, 255

net = df.MLP(
    weights=[np.array([[1, 1], [1, 1]]), np.array([[1, 0], [0, 1]])],
    biases=[np.array([-1, 0]), np.array([0, 0])],       # hidden 0: a+b-1 >= 1 (AND), hidden 1: a+b >= 1 (OR)
    thresholds=[np.array([1]), np.array([1])],
    signed_input=False,
)


def world(m: MapBuilder) -> str:
    """Everything the player sees. MapBuilder rects are axis-aligned sectors; shared edges become lines."""
    # The brain's machinery occupies y >= 0; the player's rooms must be at y < -1024. Last player-1 start = the player.
    y0, y1 = -2880, -2496
    m.rect(-896, y0, -640, y1, floor=0, ceil=128, light=DIM, tag=LIGHT_TAG, ctex="CEIL5_1", ftex="FLOOR4_8",
           wall="STARTAN3")
    # the door: a sector whose ceiling sits on its floor (closed). Door_Open raises it.
    m.rect(-640, -2784, -608, -2592, floor=0, ceil=0, tag=DOOR_TAG, ctex="CEIL5_1", ftex="FLOOR4_8",
           wall="DOORTRAK", facing="DOOR1")
    m.rect(-608, y0, -288, y1, floor=0, ceil=128, light=200, ctex="CEIL5_1", ftex="FLAT5_5", wall="COMPBLUE")
    m.thing(-400, -2688, 2006)                                 # BFG9000 on the floor of the chamber
    m.thing(-420, -2790, 48)                                   # tall techno columns
    m.thing(-420, -2586, 48)
    m.thing(-880, -2688, 1, angle=0)                           # the player, facing east, toward the door
    # World ACS: script 201 copies the door ceiling height and the room light into ViZDoom USER1/USER2, so Python can
    # print what the game says without parsing pixels.
    return f"""
script 201 (void) {{ o1 = GetSectorCeilingZ({DOOR_TAG}, 0, 0) >> 16; o2 = GetSectorLightLevel({LIGHT_TAG}); }}
"""


actions = {
    "y0": [df.act.door_open(DOOR_TAG, 32)],
    "y1": [df.act.light(LIGHT_TAG, BRIGHT)],
}


def snapshot(game: vzd.DoomGame) -> np.ndarray:
    buf = game.get_state().screen_buffer
    return np.transpose(buf, (1, 2, 0)) if buf.shape[0] == 3 else buf


def main() -> int:
    OUT.mkdir(exist_ok=True)
    frames, ok = [], True
    with df.compile(net, world=world, actions=actions, resolution=vzd.ScreenResolution.RES_320X240) as brain:
        game = brain.runner.game
        print(f"map: {brain.info['dolls']} dolls, {brain.info['sectors']} sectors, wad {brain.info['wad']}")
        for ab in [(0, 0), (1, 0), (1, 1)]:
            x = np.array(ab)
            y = brain(x)                                   # one forward pass in DOOM, verified against net.reference
            ref = net.reference(x)
            game.advance_action(80)                        # let the door finish sliding up
            door_h, light = brain.run_script(201, read=2)   # world script 201 -> USER1, USER2
            want_door, want_light = int(ref["y"][0] > 0), int(ref["y"][1] > 0)
            good = (door_h > 0) == bool(want_door) and (light == BRIGHT) == bool(want_light)
            ok &= good
            print(f"a,b={ab}: DOOM y={y.tolist()} (reference {ref['y'].tolist()}, EXACT) | door ceiling "
                  f"{door_h:3d} ({'open' if door_h else 'closed'}) | room light {light:3d} | "
                  f"{'as expected' if good else 'UNEXPECTED'}")
            frames.append((ab, snapshot(game).copy(), door_h, light))
    cells = []
    for i, (ab, img, door_h, light) in enumerate(frames):
        im = Image.fromarray(img)
        im.save(OUT / f"03_frame_{i}.png")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 319, 24], fill=(0, 0, 0))
        d.text((6, 3), f"a={ab[0]} b={ab[1]}  AND(door)={int(ab[0] and ab[1])}  OR(light)={int(ab[0] or ab[1])}",
               fill=(255, 255, 255))
        d.text((6, 13), f"read from the game: door ceiling={door_h} room light={light}", fill=(255, 220, 120))
        cells.append(im)
    sheet = Image.new("RGB", (320 * 3 + 8, 240), (0, 0, 0))
    for i, im in enumerate(cells):
        sheet.paste(im, (i * 324, 0))
    sheet.save(OUT / "03_brain_opens_doors.png")
    print(f"wrote {OUT / '03_brain_opens_doors.png'}")
    print("OK: every pass exact; door and light followed the network" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
