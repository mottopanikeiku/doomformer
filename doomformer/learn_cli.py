"""`doomformer learn digits`: a blank DOOM brain learns to read digits, one example per run of one map (docs/learning.md)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import learn_digits as ld


def add_parser(sub) -> None:
    p = sub.add_parser("learn", help="DOOM learns: an online multiclass perceptron whose 370 weights are ceiling "
                                     "heights of ONE map; dolls compute the 10 scores, take the argmax and apply the "
                                     "perceptron update; scores, prediction and all weights are checked after every "
                                     "step (tolerance 0)",
                       description="Trains a blank brain on 8x8 handwritten digits (models/digits, CC BY 4.0) inside one "
                                   "DOOM map. Python writes each example's pixels and its label (input writes), advances "
                                   "the clock and reads. The dolls compute the scores, the argmax and the update. After "
                                   "every step the scores, the 45 pair differences, the win counts, the winner, the "
                                   "update signal and ALL weights are compared with a NumPy integer twin; any difference "
                                   "stops the run. The update rule and the labels are Python's design (docs/learning.md).")
    p.add_argument("what", nargs="?", default="digits", choices=["digits"], help="what to learn (digits)")
    p.add_argument("--steps", type=int, default=300, help="training examples, one DOOM run each (default 300; 2 to 3 s "
                                                           "each with 10 classes on a loaded 16-core machine)")
    p.add_argument("--seed", type=int, default=0, help="seed of the example order (default 0)")
    p.add_argument("--every", type=int, default=100, help="held-out accuracy of the weights DOOM holds every N steps "
                                                           "(default 100)")
    p.add_argument("--classes", type=int, default=10, help="learn only digits 0..C-1 (default 10; a smaller C is a "
                                                            "faster, smaller map)")
    p.add_argument("--opt", type=int, choices=[0, 1, 2, 3], default=3,
                   help="schedule level (docs/scheduling.md). Default 3: an ACS script copies each layer's light "
                        "levels into the next layer's gate heights (it adds nothing); 0 to 2 keep the dolls' planes "
                        "but need one wave per forward doll, tens of thousands of tics per step")
    p.add_argument("--doom-eval", type=int, default=100, metavar="N",
                   help="after training, run the first N held-out digits through the learned weights in DOOM with "
                        "the ordinary forward path (df.MLP + with_argmax), checked exactly (default 100; 0 skips)")
    p.add_argument("--tamper", nargs=3, metavar=("CLASS", "FEATURE", "LINE"),
                   help="negative control: build one update line of one weight (LINE: lower, keep or raise) to move its "
                        "ceiling by 2; the run must diverge at the first step where the twin says a doll crosses that "
                        "line (lower: feature on; keep: on and q >= 1; raise: on and the true class was not predicted)")
    p.add_argument("--quiet", action="store_true", help="print only the checkpoint lines and the summary")
    p.add_argument("--log", metavar="PATH", help="also write the printed lines to this file")
    p.add_argument("--json", metavar="PATH", help="write the learning curve and totals to this JSON file")
    p.add_argument("--wad", metavar="PATH", help="keep the compiled map here")


class _Out:
    def __init__(self, path):
        self.f = open(path, "w") if path else None

    def __call__(self, text=""):
        print(text, flush=True)
        if self.f:
            self.f.write(text + "\n")
            self.f.flush()


def run(args) -> int:
    out = _Out(args.log)
    if args.steps < 1 or args.every < 1 or not 2 <= args.classes <= 10:
        print("doomformer learn: need --steps >= 1, --every >= 1 and 2 <= --classes <= 10", file=sys.stderr)
        return 2
    if args.steps > ld.STEP_MAX:
        print(f"doomformer learn: at most {ld.STEP_MAX} steps (weights are ceiling heights and must stay in the "
              f"exact range)", file=sys.stderr)
        return 2
    data = ld.load_digits()
    C = args.classes
    if C < 10:                                       # the first C digits only
        keep_tr, keep_te = data.y_train < C, data.y_test < C
        data = ld.DigitsData(data.x_train[keep_tr], data.y_train[keep_tr], data.x_test[keep_te], data.y_test[keep_te])
    spec = ld.DigitPerceptron(data.n_features, C)
    tamper = None
    if args.tamper:
        try:
            tamper = (int(args.tamper[0]), int(args.tamper[1]), args.tamper[2])
        except ValueError:
            tamper = None
        if tamper is None or tamper[2] not in ld.TAMPER_LINES or not (0 <= tamper[0] < C and 0 <= tamper[1] < spec.n_features):
            print(f"doomformer learn: --tamper CLASS FEATURE LINE with LINE in {'|'.join(ld.TAMPER_LINES)}, CLASS in "
                  f"0..{C - 1} and FEATURE in 0..{spec.n_features - 1}", file=sys.stderr)
            return 2
    plan = spec.plan(args.opt)
    out(f"digits: {len(data.x_train)} training and {len(data.x_test)} held-out digits, {data.n_features} binary "
        f"features (centre 6x6 pixels >= {ld.THRESHOLD}, and a bias), {C} classes, weights start at 0 "
        f"({C * data.n_features} weights)")
    out(f"map: {plan['dolls']:,} dolls, {plan['sectors']:,} sectors, opt {args.opt}, estimated {plan['tics']} tics per "
        f"step; {plan['values']} values read and checked per step: {plan['values'] - C * data.n_features} outputs and "
        f"all {C * data.n_features} weights")
    expect = None
    if tamper:
        out(f"TAMPER: the {tamper[2]} update line of weight (class {tamper[0]}, feature {tamper[1]}) moves its ceiling by 2")
        order = ld.stream(len(data.x_train), args.steps, args.seed)
        expect = ld.first_crossing(((data.x_train[i], data.y_train[i]) for i in order), spec.n_features, C, *tamper)
        out(f"the twin first crosses that line at step {expect + 1}" if expect is not None else
            "the twin never crosses that line in this run: the run should stay exact")
    out("per step: DOOM = the winner light of the dolls, margin = best score minus runner-up, updated = weights that "
        "changed, running = online accuracy so far, EXACT n/n = values (outputs and ALL weights) equal to the twin")
    t0 = time.time()
    learner = spec.compile(args.wad, opt=args.opt, tamper=tamper)
    import hashlib
    import platform

    import vizdoom
    out(f"built in {time.time() - t0:.1f} s: map sha256 {hashlib.sha256(Path(learner.wad).read_bytes()).hexdigest()[:16]}; "
        f"python {platform.python_version()}, numpy {np.__version__}, vizdoom {vizdoom.__version__}; seed {args.seed}")

    def on_step(r, running):
        what = "right" if r.correct else "WRONG"
        upd = f"updated {r.changed:3d}" if r.changed else "no update "
        if not args.quiet:
            out(f"step {r.index + 1:5d}  label {r.label}  DOOM {r.predicted}  margin {r.margin:3d}  {what:5s}  {upd}  "
                f"running {100 * running:5.1f}%  EXACT {r.checked}/{r.checked}  {r.tics} tics  {r.seconds:.2f} s")

    def on_checkpoint(k, acc):
        out(f"-- after {k:5d} examples: held-out accuracy of the weights DOOM holds {100 * acc:5.1f}% "
            f"({int(round(acc * len(data.y_test)))}/{len(data.y_test)}, NumPy forward over the read-back ceilings)")

    try:
        with learner:
            try:
                res = ld.train(learner, data, args.steps, args.seed, args.every, on_step, on_checkpoint)
            except Exception as e:
                from .kit import MismatchError
                if not isinstance(e, MismatchError):
                    raise
                step = learner.info["steps"] - 1
                out(f"MISMATCH at step {step + 1}: {e}")
                for name, got, want in e.mismatches[:8]:
                    out(f"    {name}: DOOM {got}, reference {want}")
                if tamper:
                    ok = expect is not None and step == expect
                    out(f"tamper test: DOOM diverged at step {step + 1}, the twin predicted step "
                        f"{'none' if expect is None else expect + 1}: {'CAUGHT' if ok else 'NOT as predicted'}")
                    return 0 if ok else 1
                return 1
            if tamper:
                out(f"tamper test: the run stayed exact (the twin predicted "
                    f"{'no divergence' if expect is None else f'step {expect + 1}'}): "
                    f"{'as predicted' if expect is None else 'NOT as predicted'}")
                return 0 if expect is None else 1
            audit = learner.audit()
            final = res["weights"]
            acc = ld.accuracy(final, data.x_test, data.y_test)
            plain = ld.plain_perceptron(data, res["order"], {args.steps}, clip=None, classes=C)
            clipped = ld.plain_perceptron(data, res["order"], {args.steps}, clip=ld.CLIP, classes=C)
            marks = set(res["curve"]) - {0}
            curve_plain = ld.plain_perceptron(data, res["order"], marks, clip=None, classes=C)["curve"]
            curve_clip = ld.plain_perceptron(data, res["order"], marks, clip=ld.CLIP, classes=C)["curve"]
            same = all(abs(curve_clip[k] - res["curve"][k]) == 0 for k in marks)
            out()
            out(f"EXACT: {args.steps} steps; after each, {learner.info['checked']} exposed values and all "
                f"{C * data.n_features} weights equal the NumPy integer twin (tolerance 0)")
            out(f"DOOM weights after {args.steps} examples: held-out accuracy {100 * acc:.1f}% "
                f"({int(round(acc * len(data.y_test)))}/{len(data.y_test)}); online accuracy over the stream "
                f"{100 * res['online_accuracy']:.1f}% ({res['mistakes']} updates)")
            out(f"NumPy perceptron, same stream, same clamp (independent code): held-out {100 * clipped['curve'][args.steps]:.1f}%, "
                f"curve identical to DOOM's: {same}, {clipped['mistakes']} updates")
            out(f"NumPy perceptron, same stream, no clamp (textbook; not DOOM): held-out {100 * plain['curve'][args.steps]:.1f}%, "
                f"{plain['mistakes']} updates")
            out("learning curve (held-out accuracy by examples seen): "
                + ", ".join(f"{k}: {100 * v:.1f}%" for k, v in res["curve"].items()))
            out(f"Python sent: {audit['input_writes']:,} input writes (pixels and label), {audit['rewinds']} rewinds, "
                f"{audit['starts']} starts, {audit['motion_checks']} motion checks, {audit['readouts']:,} readout commands, "
                f"{audit['other']} other (a write to a weight would be rejected)")
            out(f"total: {res['tics']:,} game tics, {res['seconds']:.0f} s for {args.steps} steps "
                f"({res['seconds'] / args.steps:.2f} s and {res['tics'] / args.steps:.0f} tics per step; timing on a "
                f"loaded shared machine unless stated)")
            doom_eval = None
            if args.doom_eval:
                n = min(args.doom_eval, len(data.y_test))
                t1 = time.time()
                winners = ld.doom_forward(final, data.x_test[:n])
                want = (np.clip(final, -ld.CLIP, ld.CLIP) + ld.CLIP) @ data.x_test[:n].T
                agree = int((np.array(winners) == want.argmax(0)).sum())
                right = int((np.array(winners) == data.y_test[:n]).sum())
                doom_eval = dict(n=n, agree=agree, correct=right, seconds=time.time() - t1)
                out(f"DOOM forward pass (df.MLP + with_argmax, every value exact) on the first {n} held-out digits with the "
                    f"learned weights: {agree}/{n} winners equal NumPy's argmax, {right}/{n} correct "
                    f"({time.time() - t1:.0f} s)")
        if args.json:
            Path(args.json).write_text(json.dumps(dict(
                steps=args.steps, seed=args.seed, classes=C, features=data.n_features, opt=args.opt,
                dolls=plan["dolls"], sectors=plan["sectors"], curve={str(k): v for k, v in res["curve"].items()},
                curve_numpy_clamped={str(k): v for k, v in curve_clip.items()},
                curve_numpy_plain={str(k): v for k, v in curve_plain.items()},
                online_accuracy=res["online_accuracy"], updates=res["mistakes"], heldout=acc,
                tics=res["tics"], seconds=res["seconds"], audit=audit, doom_forward=doom_eval,
                weights=final.tolist()), indent=1))
    finally:
        if out.f:
            out.f.close()
    return 0
