"""04 - Spiking disco: a ring oscillator that flashes the lights of a hall.

Eight integer leaky-integrate-and-fire neurons are wired in a ring (neuron i excites neuron i+1). One time step is
one run of the DOOM map; the membrane potentials persist inside the map between steps. Two external "kicks" start two
spikes that then chase each other around the ring forever.

What happens in the game: when a neuron spikes, its spike doll also crosses two light-changing lines (the `actions`
hook): its own hall panel goes to 255 and the previous panel goes back to the dim level (80). The player stands at the end of the hall
and watches the lights run. The hall's panels are plain sectors placed by the `world` hook.

Python does: write the kick inputs, read back potentials and spikes, and verify them against IntNet.step. DOOM does:
all of the integrate/fire/leak/reset arithmetic, and the light changes. The script also reads the real sector light
levels back out of the game and checks that exactly the neurons that spiked are lit.

    python examples/04_spiking_disco.py [--steps 24]

Writes examples/out/04_spiking_disco.gif. Exits non-zero on any mismatch.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import vizdoom as vzd
from PIL import Image, ImageDraw

import doomformer as df
from doomformer.udmf import MapBuilder

OUT = Path(__file__).resolve().parent / "out"
N, THETA = 8, 10
TAG0 = 110                                   # panel i has sector tag TAG0 + i
DIM, BRIGHT = 80, 255

names = [f"n{i}" for i in range(N)]
W = np.zeros((N, N), dtype=np.int64)
for i in range(N):
    W[(i + 1) % N, i] = THETA                # a spike of i pushes i+1 over the threshold on the next step
U = np.zeros((N, 2), dtype=np.int64)
U[0, 0] = THETA                              # kick A starts a spike at neuron 0
U[4, 1] = THETA                              # kick B starts a spike at neuron 4
net = df.SpikingNet(names, ["kickA", "kickB"], W, U, theta=THETA, leak=0)

actions = {names[i]: [df.act.light(TAG0 + i, BRIGHT), df.act.light(TAG0 + (i - 1) % N, DIM)] for i in range(N)}


def world(m: MapBuilder) -> str:
    # an entrance alcove with the player, then eight 128-unit panels down a hall. Everything at y < -1024.
    y0, y1 = -2816, -2560
    m.rect(-1664, y0, -1536, y1, floor=0, ceil=128, light=140, ctex="CEIL5_1", ftex="FLOOR4_8", wall="STARTAN3")
    for i in range(N):
        x0 = -1536 + 128 * i
        m.rect(x0, y0, x0 + 128, y1, floor=0, ceil=128, light=DIM, tag=TAG0 + i, ctex="CEIL5_1", ftex="FLAT14",
               wall=["COMPBLUE", "STARTAN3"][i % 2])
    m.thing(-1650, -2688, 1, angle=0)        # the player, looking down the hall
    reads = " ".join(f"o{i + 1} = GetSectorLightLevel({TAG0 + i});" for i in range(N))
    return f"script 201 (void) {{ {reads} }}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    X = np.zeros((args.steps, 2), dtype=np.int64)
    X[0] = [1, 1]
    frames, ok = [], True
    ref_v = np.zeros(N, dtype=np.int64)
    with df.compile(net, world=world, actions=actions, resolution=vzd.ScreenResolution.RES_320X240) as brain:
        game = brain.runner.game
        print(f"map: {brain.info['dolls']} dolls, {brain.info['sectors']} sectors")
        for t, x in enumerate(X):
            v, s = brain.step(x)                      # DOOM step, verified against IntNet.step (raises on mismatch)
            ref_v, ref_s = net.step(ref_v, x)
            assert (v == ref_v).all() and (s == ref_s).all()
            lights = brain.run_script(201, read=N)     # world script 201 writes the panel light levels to USER1..8
            lit = [i for i in range(N) if lights[i] == BRIGHT]
            good = lit == [i for i in range(N) if s[i]]
            ok &= good
            print(f"t={t:2d} kick={x.tolist()} spikes={''.join(str(int(b)) for b in s)} lit panels={lit} "
                  f"v={v.tolist()} EXACT {'lights match spikes' if good else 'LIGHTS DIFFER'}")
            buf = game.get_state().screen_buffer
            im = Image.fromarray(np.transpose(buf, (1, 2, 0))).convert("RGB")
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, 319, 14], fill=(0, 0, 0))
            d.text((4, 2), f"t={t:2d} spikes {''.join(str(int(b)) for b in s)}  (DOOM computed, exact)",
                   fill=(255, 255, 255))
            frames.append(im.resize((480, 360), Image.NEAREST))
    gif = OUT / "04_spiking_disco.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=300, loop=0)
    print(f"wrote {gif} ({len(frames)} frames)")
    print("OK: every step exact, lights match spikes" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
