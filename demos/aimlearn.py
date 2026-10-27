"""DOOM learns to aim: two online perceptrons on arena features, taught by the aimbot every decision.

features (19, binary): one-hot bearing bin of the nearest demon (17), "no enemy" (1), bias (1)
head 0 "fire?":  label +1 if the aimbot fires, else -1
head 1 "turn":   label +1 if the aimbot turns left, -1 if right, 0 (no lesson) when it fires
action, chosen by dolls in the same map: FIRE if s_fire > 0, else TURN LEFT if s_turn >= 0, else TURN RIGHT
(a blank brain spins left and never shoots). Two act dolls read the score floors and add to the "act" light:
+2 if s_fire >= 1; +1 if s_turn <= -1 and s_fire <= 0. So act is the index into policy.ACTIONS ("L", "R", "F").
The scores and act come from the map BEFORE it learns from this decision's label: the marine acts on what it knew.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from doomformer import policy
from doomformer.learn import perceptron_specs, ports_for
from doomformer.ops import adc_lanes, fx, output_single
from doomformer.pure import compile_pure
from doomformer.tile import Block, Gate, Stage, TileMeta

from . import say

N_FEAT = len(policy.BEARING_CHARS) + 2
HEADS = 2
K = 3


def build_learner(wad: Path, name: str = "aim_learner", world=None) -> TileMeta:
    """world: compile_pure world hook, e.g. arena.build to put his arena and his brain in one map."""
    fwd, lrn, state, readout = perceptron_specs(N_FEAT, HEADS, K)
    act = output_single(lrn, "act")
    fire = Gate(lrn.port("-s0", "floor"), -1, "floor")                       # s_fire >= 1
    lrn.blocks.append(Block(adc_lanes(lrn, "go", [1], 1), [Stage(fire, fx(act, 2))]))
    right = Gate(lrn.port("s1", "floor"), -1, "floor")                       # s_turn <= -1
    hold = Gate(lrn.port("s0", "floor"), 0, "floor")                         # s_fire <= 0
    lrn.blocks.append(Block(adc_lanes(lrn, "go", [1], 1), [Stage(right, []), Stage(hold, fx(act, 1))]))
    return compile_pure([fwd, lrn], wad, name, state=state, readout=readout, world=world)


def ports(x: np.ndarray, y: np.ndarray) -> dict[str, int]:
    return ports_for(x, y) | {"go": 1, "-go": -1}


def features(state, game) -> np.ndarray:
    x = np.zeros(N_FEAT, dtype=np.int64)
    en = policy.enemies(state, game)
    if en:
        x[policy.BEARING_CHARS.index(policy.VOCAB[policy.bearing_token(en[0][1])])] = 1
    else:
        x[N_FEAT - 2] = 1
    x[N_FEAT - 1] = 1
    return x


def labels(expert_idx: int) -> np.ndarray:
    a = policy.ACTIONS[expert_idx]
    return np.array([1 if a == "F" else -1, {"L": 1, "R": -1, "F": 0}[a]], dtype=np.int64)


def action(s: np.ndarray) -> int:
    """Reference for the act dolls."""
    if s[0] > 0:
        return policy.ACTIONS.index("F")
    return policy.ACTIONS.index("L" if s[1] >= 0 else "R")


def _bin_text(x: np.ndarray) -> str:
    """Words for the feature vector: the bearing bin of the nearest demon, or none."""
    if x[N_FEAT - 2]:
        return "no enemy"
    return policy.describe([0, policy.VOCAB.index(policy.BEARING_CHARS[int(np.argmax(x[:N_FEAT - 2]))]),
                            policy.VOCAB.index("0")])[0].lower()


def run_learner(*, seed: int = 4, lives: int = 2, decisions: int | None = None, gif: str | None = None,
                workdir: str | None = None, show: int = 10, every: int = 25, resolution=None, driver: str = "brain",
                out=say) -> dict:
    """Play `lives` lives (or stop after `decisions` decisions) of the one-game learner and check every decision.

    One map holds his arena and the perceptron that learns to aim. Each decision: time stands still, Python bins what he
    sees into 19 features and supplies the aimbot's label, the dolls compute both scores, choose the action and rewrite
    the weights, and Python compares the 2 scores, the action and all 38 weights with an integer NumPy perceptron
    (tolerance 0). The marine then presses the dolls' action for 4 tics.
    driver: who presses the buttons: "brain" (the dolls' action), "aimbot" or "blank" (baselines; the brain still
    runs and is checked every decision). Returns the totals."""
    import tempfile
    import time

    import vizdoom as vzd

    from doomformer.learn import PerceptronRef
    from doomformer.runtime import TileRunner

    from . import arena

    GV = vzd.GameVariable
    tmp = tempfile.TemporaryDirectory(prefix="doomformer-aimlearn-") if workdir is None else None
    wad_dir = Path(workdir) if workdir else Path(tmp.name)
    wad_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    wad = wad_dir / "one_game.wad"
    meta = build_learner(wad, name="one_game", world=arena.build)
    out(f"one map: his arena + a perceptron that learns to aim ({meta.n_dolls} voodoo dolls of him, "
        f"{meta.n_sectors} sectors); built in {time.perf_counter() - t0:.1f}s")
    r = TileRunner(wad, configure=arena.configure, resolution=resolution or (
        vzd.ScreenResolution.RES_320X240 if gif else vzd.ScreenResolution.RES_160X120))
    g = r.game
    g.set_seed(seed)
    g.new_episode()
    trace = None
    if gif:
        from .gifout import GifTrace
        trace = GifTrace(wad, gif)
    ref = PerceptronRef(N_FEAT, HEADS, K)
    w_doom = np.zeros((HEADS, N_FEAT), dtype=np.int64)
    values = HEADS + 1 + HEADS * N_FEAT
    stats = dict(decisions=0, exact=0, lessons=0, kills=[], dolls=meta.n_dolls, sectors=meta.n_sectors,
                 values_per_decision=values)
    t_run = time.perf_counter()
    limit_hit = False
    try:
        for life in range(lives):
            g.send_game_command(f"puke -{arena.NEW_LIFE}")
            g.advance_action(1)
            k0, awake, dec, agree = g.get_game_variable(GV.KILLCOUNT), 0, 0, 0
            while not arena.life_over(g, awake):
                if decisions is not None and stats["decisions"] >= decisions:
                    limit_hit = True
                    break
                g.set_action(arena.HANDS_OFF)            # advance_action repeats the last action
                g.send_game_command(f"puke -{arena.FREEZE}")        # time stands still while he thinks
                g.advance_action(1)
                state = g.get_state()
                x, e = features(state, g), policy.expert_index(state, g)
                y = labels(e)
                t_dec = time.perf_counter()
                got, settled, _, _ = r.run(ports(x, y))
                s = np.array([got[f"s{h}"] for h in range(HEADS)])
                w_new = np.array([[got[f"w{h}_{i}"] for i in range(N_FEAT)] for h in range(HEADS)])
                a = int(got["act"])
                s_ref = ref.step(x, y)
                exact = bool((s == s_ref).all() and (w_new == ref.w).all() and a == action(s_ref))
                changed, w_doom = w_new != w_doom, w_new
                stats["decisions"] += 1
                stats["exact"] += exact
                stats["lessons"] += int(changed.any())
                dec += 1
                agree += a == e
                dolls = [(o.position_x, o.position_y) for o in g.get_state().objects if o.name == "DoomPlayer"]
                n = stats["decisions"]
                if not exact:
                    out(f"  MISMATCH decision {n}: scores {s.tolist()} reference {s_ref.tolist()}, "
                        f"action {a} reference {action(s_ref)}")
                if n <= show or n % every == 0 or not exact:
                    out(f"decision {n:4d}  life {life + 1}  sees {_bin_text(x)}  aimbot says {policy.ACTIONS[e]}  "
                        f"his dolls chose {policy.ACTIONS[a]}  scores {s.tolist()}  "
                        f"{int(changed.any()) * 'learned: weights changed' or 'no change'}  "
                        f"{'EXACT' if exact else 'MISMATCH'} ({values} values: {HEADS} scores, the action, "
                        f"{HEADS * N_FEAT} weights; {settled} tics, {time.perf_counter() - t_dec:.1f}s)")
                g.send_game_command(f"puke -{arena.UNFREEZE}")
                played = {"brain": a, "aimbot": e, "blank": action(np.zeros(HEADS, int))}[driver]
                buttons = arena.buttons(policy.ACTIONS[played])
                t, ammo0 = 0, g.get_game_variable(GV.AMMO2)
                # 4 tics of his buttons; after the trigger, hands off but awake until the bullet is out
                while t < policy.TICS_PER_DECISION or (policy.ACTIONS[played] == "F" and t < arena.FOLLOW_MAX
                                                       and g.get_game_variable(GV.AMMO2) == ammo0 > 0):
                    g.make_action(buttons if t < policy.TICS_PER_DECISION else arena.HANDS_OFF, 1)
                    awake += 1
                    t += 1
                if trace is not None:
                    kills = int(g.get_game_variable(GV.KILLCOUNT) - k0)
                    trace.add(g.get_state().screen_buffer, dolls,
                              f"life {life + 1}  decision {dec}  kills {kills}  dolls chose {policy.ACTIONS[a]}  "
                              f"{stats['exact']}/{stats['decisions']} exact")
            kills = int(g.get_game_variable(GV.KILLCOUNT) - k0)
            if dec:
                why = ("killed" if g.get_game_variable(GV.HEALTH) <= 1 else
                       "out of bullets" if g.get_game_variable(GV.AMMO2) <= 0 else
                       "time" if awake >= arena.LIFE_TICS else "stopped at the decision limit")
                stats["kills"].append(kills)
                out(f"life {life + 1}: {kills} kills ({why}), {dec} decisions, agrees with the aimbot "
                    f"{agree / dec:.0%}, {stats['exact']}/{stats['decisions']} exact so far, "
                    f"{time.perf_counter() - t_run:.0f}s")
            if limit_hit:
                break
    finally:
        r.close()
    stats["wall_s"] = time.perf_counter() - t_run
    stats["final_w"] = w_doom.tolist()
    n = stats["decisions"]
    out(f"{stats['exact']}/{n} decisions EXACT ({values} values each, {values * n} in all; tolerance 0); "
        f"weights changed in {stats['lessons']} of them; kills per life {stats['kills']}")
    pressed = {"brain": "the action the dolls chose", "aimbot": "the aimbot's action (a baseline run)",
               "blank": "a blank brain's action, turn left (a baseline run)"}[driver]
    out(f"Python's part: binned what he sees into features, supplied the aimbot's labels, froze and woke the demons, "
        f"pressed {pressed}. The dolls computed the scores, the action and every weight update.")
    out(f"wall {stats['wall_s']:.0f}s, {stats['wall_s'] / max(1, n):.2f}s per decision (dolls thinking, "
        f"checking and playing the action)")
    if trace is not None:
        path, size = trace.close()
        out(f"gif: {path} ({size / 1e6:.2f} MB, {len(trace.frames)} frames)")
    if tmp is not None:
        tmp.cleanup()
    return stats


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m demos.aimlearn",
        description="DOOM learns to aim: a blank perceptron in his own level is taught by an aimbot, with its "
                    "weights as ceiling heights. Every decision is checked against an integer NumPy perceptron "
                    "(scores, action, all 38 weights; tolerance 0).")
    ap.add_argument("--seed", type=int, default=4, help="the game's own RNG (demon spawns); 4 is the documented run")
    ap.add_argument("--lives", type=int, default=2,
                    help="lives to play (default 2; the documented run is --lives 5 --seed 4)")
    ap.add_argument("--decisions", type=int, default=None, help="stop after this many decisions (default: play the lives out)")
    ap.add_argument("--driver", choices=("brain", "aimbot", "blank"), default="brain",
                    help="who presses the buttons: the dolls' action (default) or a baseline; the brain still "
                         "learns and is checked every decision")
    ap.add_argument("--gif", metavar="PATH", help="write a small looping GIF of the run (the game's frames and the dolls)")
    ap.add_argument("--workdir", help="where to keep the compiled map (default: a temporary directory)")
    ap.add_argument("--show", type=int, default=10, help="print every decision up to this one, then every --every-th")
    ap.add_argument("--every", type=int, default=25)
    args = ap.parse_args(argv)
    if args.lives < 1 or (args.decisions is not None and args.decisions < 1):
        ap.error("--lives and --decisions must be positive")
    stats = run_learner(seed=args.seed, lives=args.lives, decisions=args.decisions, gif=args.gif,
                        workdir=args.workdir, show=args.show, every=args.every, driver=args.driver)
    return 0 if stats["exact"] == stats["decisions"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
