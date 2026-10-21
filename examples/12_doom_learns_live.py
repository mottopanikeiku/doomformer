"""12 - DOOM learns live: a perceptron whose weights are ceiling heights, taught by labels, one example at a time.

A demon stands somewhere on a line of 7 spots in front of the player (spot -3 far left ... +3 far right). The rule
to learn is a toy: "shoot iff the demon is left of centre". Features (8 binary values): a bias and a one-hot of the spot.
Labels come from a teacher that is plain Python (it knows the rule); the learner never sees the rule, only labels.

    df.Perceptron(8, heads=1)  ->  .compile(world=..., actions=...)  ->  learner.step(x, y)

What DOOM does in each step (docs/learning.md): voodoo dolls add up the score s = sum_i x_i * clamp(w_i, -3, 3) from
the weights, which are the ceiling heights of 16 sectors (w and -w); they choose the action ("shoot" iff s >= 1);
they decide right/wrong against the label (right: y=+1 and s >= 1, or y=-1 and s <= 0) and cross a line special that
turns the GREEN lamp on (right) or the RED lamp on (wrong); and for a mistake (y*s <= 0) they raise or lower the ceilings
of the active features' weights by 1. The weights persist in the map from step to step.
What Python does: draws the spot, writes x and y into the map, moves the demon (scenery), reads everything back and
compares it with the integer twin (PerceptronRef) with tolerance 0. Every step must be exact, including every weight
and both lamps (their light levels are read from the game by a world script); otherwise the script exits nonzero.

Halfway through, the teacher flips the rule ("shoot iff the demon is RIGHT of centre"). Accuracy drops and DOOM relearns:
the dolls are still adjusting ceilings, nothing is reset.

    python examples/12_doom_learns_live.py            # ~10 s, writes examples/out/12_doom_learns_live.png
"""
import sys
import time
from pathlib import Path

import numpy as np
import vizdoom as vzd
from PIL import Image, ImageDraw

import doomformer as df
from doomformer.learnkit import Perceptron
from doomformer.udmf import MapBuilder

OUT = Path(__file__).resolve().parent / "out"
GREEN_TAG, RED_TAG, DEMON_TID = 101, 102, 901
DIM, LIT = 40, 255
SPOTS = range(-3, 4)                     # -3 far left ... +3 far right of the player (he faces east, left is north)
PHASES = [("shoot iff LEFT of centre", lambda p: p < 0, 4), ("shoot iff RIGHT of centre", lambda p: p > 0, 6)]
SEED = 1


def world(m: MapBuilder) -> str:
    """A hall; its east half is a traffic light: the south half is lit green or dim, the north half red or dim."""
    kw = dict(floor=0, ceil=160, ctex="CEIL5_1", ftex="FLAT23", wall="STARTAN3")
    m.rect(-1024, -3072, -288, -2048, light=150, **kw)
    m.rect(-288, -3072, 512, -2560, light=DIM, tag=GREEN_TAG, sector=dict(lightcolor=0x00FF00), **kw)
    m.rect(-288, -2560, 512, -2048, light=DIM, tag=RED_TAG, sector=dict(lightcolor=0xFF0000), **kw)
    m.thing(-420, -2560, 3002, angle=180, id=DEMON_TID, dormant=True)        # a demon that stands where it is put
    m.thing(-768, -2560, 1, angle=0)                                         # the player, facing east
    return f"""
script 210 (int spot) {{ SetActorPosition({DEMON_TID}, -420.0, -2560.0 - spot * 96.0, 0.0, 0); }}
script 211 (void) {{ o1 = GetSectorLightLevel({GREEN_TAG}); o2 = GetSectorLightLevel({RED_TAG}); }}
"""


def lamps(green: bool) -> list[dict]:
    on, off = (GREEN_TAG, RED_TAG) if green else (RED_TAG, GREEN_TAG)
    return [df.act.light(on, LIT), df.act.light(off, DIM)]


ACTIONS = {"right0": lamps(True), "wrong0": lamps(False)}


def features(spot: int) -> np.ndarray:
    x = np.zeros(8, dtype=np.int64)
    x[0] = 1                                                  # bias
    x[1 + spot + 3] = 1
    return x


def view(game: vzd.DoomGame) -> Image.Image:
    return Image.fromarray(np.transpose(game.get_state().screen_buffer, (1, 2, 0)))


def curve(acc: list[float], boundaries: list[int], size=(320, 240)) -> Image.Image:
    im = Image.new("RGB", size, (16, 16, 24))
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = 34, 28, size[0] - 12, size[1] - 34
    for t in (0, 0.5, 1):
        y = y1 - t * (y1 - y0)
        d.line([x0, y, x1, y], fill=(60, 60, 80))
        d.text((6, y - 5), f"{int(t * 100)}%", fill=(180, 180, 200))
    step = (x1 - x0) / max(len(acc) - 1, 1)
    for b in boundaries:
        d.line([x0 + (b - 1) * step - step / 2, y0, x0 + (b - 1) * step - step / 2, y1], fill=(220, 160, 60))
    pts = [(x0 + i * step, y1 - a * (y1 - y0)) for i, a in enumerate(acc)]
    d.line(pts, fill=(90, 220, 120), width=2)
    for p in pts:
        d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=(90, 220, 120))
    d.text((6, 6), "accuracy per epoch (orange: teacher flips the rule)", fill=(230, 230, 240))
    d.text((x0, y1 + 8), "epoch 1 ... %d  (each = the 7 spots, shuffled)" % len(acc), fill=(180, 180, 200))
    return im


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    spec = Perceptron(n_features=8, heads=1)
    print(f"{spec}: plan {spec.plan()}")
    lamp_ok = total = 0
    acc, boundaries, shots = [], [], {}
    t0 = time.perf_counter()
    with spec.compile(world=world, actions=ACTIONS, resolution=vzd.ScreenResolution.RES_320X240) as learner:
        game = learner.runner.game
        game.advance_action(3)
        print(f"map: {learner.info['dolls']} dolls, {learner.info['sectors']} sectors | start: weights read from "
              f"DOOM {learner.weights.tolist()}")
        epoch = 0
        for name, rule, n_epochs in PHASES:
            print(f"--- teacher: {name}")
            if epoch:
                boundaries.append(epoch + 1)
            for _ in range(n_epochs):
                epoch += 1
                right = 0
                for spot in rng.permutation(list(SPOTS)):
                    spot = int(spot)
                    x, label = features(spot), 1 if rule(spot) else -1
                    learner.run_script(210, spot)                    # world script: put the demon on his spot
                    try:
                        scores, action = learner.step(x, [label])   # DOOM: scores, action, lamp, update; checked
                    except df.MismatchError as e:
                        print(f"MISMATCH: {e}")
                        return 1
                    total += 1
                    # DOOM's verdict, from the scores it computed: right iff the action equals the label
                    was_right = (action == 1) == (label == 1)
                    right += was_right
                    green, red = learner.run_script(211, read=2)    # lamp light levels, read from the game
                    lamps_ok = (green, red) == ((LIT, DIM) if was_right else (DIM, LIT))
                    lamp_ok += lamps_ok
                    if not lamps_ok:
                        print(f"LAMP MISMATCH: spot {spot}: green {green} red {red}, right={was_right}")
                    game.advance_action(4)
                    if epoch == 1 and not was_right and "wrong" not in shots:
                        shots["wrong"] = view(game)
                    if epoch == sum(p[2] for p in PHASES) and was_right and "right" not in shots:
                        shots["right"] = view(game)
                acc.append(right / len(SPOTS))
                print(f"epoch {epoch:2d}: accuracy {acc[-1]:4.0%} | weights read from DOOM "
                      f"{learner.weights[0].tolist()} | steps so far {total}, all exact, lamps matched the verdict {lamp_ok}/{total}")
        # a last look at everything DOOM holds, against the twin
        w = learner.weights
        weights_ok = np.array_equal(w, learner.twin.w)
        print(f"final weights from DOOM {w.tolist()} == twin: {weights_ok} | lessons {learner.info['lessons']} of "
              f"{learner.info['steps']} steps | {(time.perf_counter() - t0) / total * 1000:.0f} ms per step")
    sheet = Image.new("RGB", (320 * 3 + 8, 240), (0, 0, 0))
    sheet.paste(curve(acc, boundaries), (0, 0))
    for i, (key, label) in enumerate([("wrong", "epoch 1: wrong -> RED"), ("right", "last epoch: right -> GREEN")]):
        if key in shots:
            im = shots[key]
            ImageDraw.Draw(im).text((6, 4), label, fill=(255, 255, 255))
            sheet.paste(im, ((i + 1) * 324, 0))
    sheet.save(OUT / "12_doom_learns_live.png")
    print(f"wrote {OUT / '12_doom_learns_live.png'}")
    first_rule_end, second_start = PHASES[0][2], PHASES[0][2] + 1
    learned = acc[first_rule_end - 1] == 1.0 and acc[-1] == 1.0
    rose = acc[first_rule_end - 1] > acc[0] and acc[-1] > acc[second_start - 1]
    dropped = acc[second_start - 1] < acc[first_rule_end - 1]
    ok = lamp_ok == total and weights_ok and learned and rose and dropped
    print(f"{total} steps; scores, action, picks, verdicts and all weights exact in every step (a mismatch aborts "
          f"the run); lamps matched the verdicts {lamp_ok}/{total}")
    print("OK: DOOM learned the rule, lost it when the teacher flipped it, and learned the new one" if ok else
          "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
