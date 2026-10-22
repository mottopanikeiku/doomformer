"""`doomformer` command line: demo, mnist, draw, ask, hello, plan, play, doctor."""
from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np


def _note_acc_build() -> None:
    from doomformer import acc
    if acc.find()[0] is None:
        print("building the ACS compiler (first use, about a minute; needs git, cmake, a C compiler and network "
              "access) ...", flush=True)


class _Heartbeat:
    """Prints `what()` every `every` seconds on a daemon thread while a long DOOM run is in progress."""

    def __init__(self, what, every: float = 15):
        self.what, self.every, self.done = what, every, threading.Event()
        self.t0 = time.time()

    def __enter__(self):
        def beat():
            while not self.done.wait(self.every):
                print(f"  ... {time.time() - self.t0:.0f} s{self.what()}", flush=True)
        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.done.set()


def _attention(net) -> str:
    if net.attn == "softmax":
        return "softmax attention (exponentials and the division by their sum computed by the dolls)"
    return "ReLU attention"


def _demo_args(args, net) -> int:
    """Check the prompt against the model; return how many characters may be generated."""
    bad = [c for c in args.prompt if c not in net.vocab]
    if bad:
        raise SystemExit(f"demo: prompt characters {bad} are not in the model's vocabulary {net.vocab[1:]!r}")
    room = net.T - 1 - len(args.prompt)
    if room < 1:
        raise SystemExit(f"demo: nothing to write: the context is {net.T - 1} characters, the prompt has {len(args.prompt)}")
    return room


def _demo_tiles(args) -> int:
    """The default demo: a trained model writes a whole name, its tile maps spread over a pool of DOOM games."""
    import doomformer as df
    from doomformer import intref, shipped
    from doomformer.record import topdown

    net = df.Transformer.load(args.model)
    corpus = shipped.corpus(args.model)
    passes = _demo_args(args, net) if args.letters is None else min(args.letters, _demo_args(args, net))
    workers = args.workers or min(os.cpu_count() or 1, 8)
    print(f"A transformer writes a DOOM level name; every number it computes is computed by DOOM.\n"
          f"model: the shipped {args.model}: {net.L} layer{'s' if net.L > 1 else ''}, d={net.d}, {net.H} heads, "
          f"context {net.T}, {_attention(net)}" + (f", trained on {len(corpus):,} DOOM level names" if corpus else "")
          + "\n"
          f"sampling: temperature {args.temperature:g}, seed {args.seed}"
          + (f", prompt {args.prompt!r}" if args.prompt else ""), flush=True)
    _note_acc_build()
    t0 = time.time()
    with df.compile(net, mode="tiles", workers=workers) as brain:
        info = brain.info
        print(f"{info['tiles']} tile maps ({info['dolls']:,} voodoo dolls, {info['sectors']:,} sectors) built in "
              f"{time.time() - t0:.0f} s; running them in {workers} DOOM game processes ...", flush=True)
        t1 = time.time()
        with _Heartbeat(lambda: f", {info['tile_runs']} tile runs"):
            try:
                text = brain.generate(args.prompt, temperature=args.temperature, seed=args.seed, max_new=passes)
            except df.MismatchError as e:
                print(f"MISMATCH: {e}")
                return 1
        elapsed = time.time() - t1
        if args.png:
            import json
            manifest = json.loads((brain.wad / "manifest.json").read_text())
            big = max(manifest, key=lambda k: manifest[k]["dolls"])
            topdown(brain.wad / f"{big}.wad", size=(1000, 520),
                    title=f"tile map {big} of {len(manifest)} from above ({manifest[big]['dolls']:,} voodoo dolls, "
                          f"orange = weight lines); DOOM wrote {text!r}").save(args.png)
        runs, checked = info["tile_runs"], info["checked_total"]
    ref = intref.generate(net, args.prompt, max_new=passes, temperature=args.temperature, seed=args.seed)
    exact = ref == text
    novelty = "" if corpus is None else (
        "   (one of its training names)" if text in corpus else "   (NOT in its training corpus: the model made it up)")
    print(f"\nDOOM wrote: {text!r}{novelty}\n"
          f"  {runs:,} tile runs in {workers} parallel DOOM games | {info['dolls']:,} voodoo dolls "
          f"in the tile maps | {checked:,} values read back and compared with the NumPy integer reference "
          f"(tolerance 0) | {elapsed:.0f} s\n"
          f"{'EXACT' if exact else 'MISMATCH'}: " + ("every value DOOM exposed equals the integer reference, and the "
                                                     "reference writes the same name" if exact else f"the reference wrote {ref!r}"))
    if args.png:
        print(f"top-down picture of the biggest tile map: {args.png}")
    return 0 if exact else 1


def _demo_map(args) -> int:
    """--one-map / --fast: the whole forward pass inside ONE DOOM map (a letter at a time)."""
    import doomformer as df
    from doomformer import intref, onemap
    from doomformer.record import topdown

    net = onemap.tiny_model() if args.fast else df.Transformer.load("doomnames_onemap")
    what = ("a random-weight toy transformer (writes nonsense; the checking is the same)" if args.fast else
            f"the shipped doomnames_onemap: {net.L} layer, d={net.d}, {net.H} head, context {net.T}, "
            "trained on DOOM level names of up to 5 letters")
    passes = min(args.letters or 1, _demo_args(args, net))
    print(f"A transformer's forward pass, computed inside ONE DOOM map.\nmodel: {what}", flush=True)
    _note_acc_build()
    t0 = time.time()
    with df.compile(net) as brain:
        dolls, sectors = brain.info["dolls"], brain.info["sectors"]
        print(f"map built in {time.time() - t0:.0f} s: {dolls:,} voodoo dolls cross lava lines in {sectors:,} sectors",
              flush=True)
        if not args.fast:
            print(f"DOOM is running {passes} forward pass(es), about 85 s each on one core (3,500 game tics of dolls "
                  "walking) ...", flush=True)
        t1 = time.time()
        with _Heartbeat(lambda: "", every=20):
            try:
                text = brain.generate(args.prompt, temperature=args.temperature, seed=args.seed, max_new=passes)
            except df.MismatchError as e:
                print(f"MISMATCH: {e}")
                return 1
        elapsed, checked, tics = time.time() - t1, brain.info["checked_total"], brain.info["tics"]
        if args.png:
            topdown(brain.wad, size=(1000, 520),
                    title=f"a transformer in one DOOM map ({dolls:,} voodoo dolls, {sectors:,} sectors, orange = weight "
                          f"lines); DOOM wrote {text!r}").save(args.png)
    ref = intref.generate(net, args.prompt, max_new=passes, temperature=args.temperature, seed=args.seed)
    exact = ref == text
    print(f"\nDOOM wrote: {text!r}\n"
          f"  {dolls:,} voodoo dolls | {sectors:,} sectors | {checked:,} values read back and compared with the NumPy "
          f"integer reference (tolerance 0) | {tics:,} game tics per pass | {elapsed:.0f} s\n"
          f"{'EXACT' if exact else 'MISMATCH'}: " + ("every value DOOM exposed equals the integer reference, and the "
                                                     "reference writes the same text" if exact else f"the reference wrote {ref!r}"))
    if args.png:
        print(f"top-down picture of the map: {args.png}")
    return 0 if exact else 1


def _demo(args) -> int:
    if args.fast and args.one_map:
        raise SystemExit("demo: --fast and --one-map are alternatives, not a pair")
    return _demo_map(args) if args.fast or args.one_map else _demo_tiles(args)


# --- doomformer mnist: DOOM reads handwritten digits -----------------------------------------------------------------
MNIST_SECONDS = 90          # measured: 76-99 s of wall time per digit per game, six games at once (docs/drawpad.md)
def _ascii(bits: np.ndarray) -> list[str]:
    """A 28x28 0/1 image as 14 text lines of half blocks (two pixel rows per line)."""
    img = np.asarray(bits).reshape(28, 28).astype(int)
    return ["".join(" ▀▄█"[img[r, c] + 2 * img[r + 1, c]] for c in range(28)) for r in range(0, 28, 2)]


def digit_from_image(path) -> np.ndarray:
    """A picture of one digit -> the 784 0/1 pixels the MNIST model reads, prepared the way MNIST was: ink bright on
    dark (light paper is inverted), the digit scaled into a 20x20 box, centred by its centre of mass in 28x28."""
    from PIL import Image
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255
    if g.mean() > 0.5:
        g = 1 - g
    ink = g > 0.5
    if not ink.any():
        raise ValueError(f"no ink found in {path}: draw a dark digit on light paper (or light on dark)")
    rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
    crop = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    scale = 20 / max(crop.shape)
    w, h = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
    a = np.asarray(Image.fromarray(np.uint8(crop * 255)).resize((w, h), Image.LANCZOS), dtype=np.float64) / 255
    cy, cx = (np.indices(a.shape) * a).reshape(2, -1).sum(1) / a.sum()
    top, left = (int(np.clip(round(14 - c), 0, 28 - n)) for c, n in ((cy, h), (cx, w)))
    canvas = np.zeros((28, 28))
    canvas[top:top + h, left:left + w] = a
    return (canvas > 0.5).astype(np.uint8).reshape(-1)


def doom_digit(y) -> int:
    """The digit DOOM decided: the index of the single 1 in the winner net's one-hot output (DOOM's dolls compared
    the scores and counted wins; this only reads which output light they set)."""
    y = [int(v) for v in y]
    if sorted(y) != [0] * (len(y) - 1) + [1]:
        raise ValueError(f"not a one-hot output: {y}")
    return y.index(1)


MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_TEST = {"t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
              "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"}
MNIST_ART = 4               # runs of more than twice this many digits draw only the first MNIST_ART


def mnist_test_set() -> tuple[np.ndarray, np.ndarray]:
    """The 10,000 MNIST test digits as 784 0/1 pixels (> 127 of 255, as the model was trained) and their labels.
    Downloaded once into <cache>/mnist (DOOMFORMER_CACHE overrides the root) and checked against sha256 digests."""
    import gzip
    import hashlib
    import urllib.request

    from doomformer.acc import cache_dir
    d = cache_dir() / "mnist"
    d.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()           # noqa: E731
    raw = {}
    for name, digest in MNIST_TEST.items():
        path = d / name
        if not path.is_file() or sha(path) != digest:
            print(f"downloading {MNIST_URL}{name} into {d} ...", flush=True)
            tmp = path.with_name(f".{name}.{os.getpid()}")
            try:
                with urllib.request.urlopen(MNIST_URL + name, timeout=120) as r:
                    tmp.write_bytes(r.read())
                if sha(tmp) != digest:
                    raise OSError(f"{MNIST_URL}{name}: sha256 {sha(tmp)} differs from the expected {digest}")
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        raw[name] = gzip.decompress(path.read_bytes())
    images = np.frombuffer(raw["t10k-images-idx3-ubyte.gz"], np.uint8, offset=16).reshape(-1, 784)
    labels = np.frombuffer(raw["t10k-labels-idx1-ubyte.gz"], np.uint8, offset=8)
    return (images > 127).astype(np.uint8), labels.astype(np.int64)


def _at_least(lo: int):
    """argparse type: an integer >= lo."""
    def integer(text: str) -> int:
        v = int(text)
        if v < lo:
            raise argparse.ArgumentTypeError(f"must be at least {lo}, not {v}")
        return v
    return integer


def _mnist_digits(args):
    """(pixels, labels, names, test indices) of the digits to read."""
    from doomformer import shipped
    if args.image:
        try:
            return np.array([digit_from_image(args.image)]), [None], [Path(args.image).name], [None]
        except (OSError, ValueError) as e:
            raise SystemExit(f"mnist: {e}") from None
    d = np.load(shipped.data("mnist/digits.npz"))
    if args.all or args.digits > len(d["labels"]):
        try:
            bits, labels = mnist_test_set()
        except OSError as e:
            raise SystemExit(f"mnist: the MNIST test set is not available: {e}") from None
        index = np.arange(len(bits))
    else:
        bits, labels, index = np.unpackbits(d["bits"], axis=1)[:, :784], d["labels"], d["test_index"]
    if args.all:
        pick = np.arange(len(bits))
    elif args.digits > len(bits):
        raise SystemExit(f"mnist: --digits must be at most {len(bits)}")
    else:
        pick = np.random.default_rng(args.seed).choice(len(bits), size=args.digits, replace=False)
    return bits[pick], [int(v) for v in labels[pick]], [f"test #{int(index[i])}" for i in pick], \
        [int(index[i]) for i in pick]


def _mnist(args) -> int:
    import json

    import doomformer as df
    from doomformer import shipped

    path = shipped.data("mnist/mlp.npz")
    net = df.MLP.load(path)
    digits, labels, names, test_index = _mnist_digits(args)
    n, ray = len(digits), args.ray is not None
    if ray:
        try:
            df.fleet.ray_module()
        except ImportError as e:
            raise SystemExit(f"mnist: {e}") from None
    workers = min(args.workers, n) if args.workers else (None if ray else min(os.cpu_count() or 1, 8, n))
    where = (f"in {workers} DOOM game{'s' if workers > 1 else ''} at once{' (Ray)' if ray else ''}" if workers else
             "on Ray, one DOOM game per CPU of the cluster")
    p = df.plan(net, mode="map")
    print(f"DOOM reads handwritten digits. The whole network (784 pixels -> {net.weights[0].shape[0]} -> "
          f"{net.weights[1].shape[0]} score comparisons -> 10 win counts -> one-hot) is ONE DOOM map:\n"
          f"{p['dolls']:,} voodoo dolls, {p['sectors']:,} sectors; every ink pixel releases a doll that crosses the "
          f"weight lines of the hidden layer, and dolls compare every pair of digit scores, so DOOM itself picks the "
          f"digit.\nreading {n} digit{'s' if n > 1 else ''} {where} (about {MNIST_SECONDS} s per digit per game) ...",
          flush=True)
    _note_acc_build()
    large = n > 2 * MNIST_ART
    results: dict[int, dict] = {}
    t0 = last = time.time()

    def line():
        done = len(results)
        right = sum(doom_digit(v["y"]) == labels[i] for i, v in results.items() if labels[i] is not None)
        games = len({v["game"] for v in results.values()})
        per_digit = sum(v["seconds"] for v in results.values()) / done          # one game's seconds per digit
        left = (n - done) * per_digit / max(games, workers or 0) / 60
        return (f"  [{done:>{len(str(n))}}/{n}] {right} right | {done} exact | "
                f"{done / (time.time() - t0) * 60:.1f} digits/min | {games} games | about {left:.0f} min left")

    def progress(e):
        nonlocal last
        results[e["index"]] = dict(y=[int(v) for v in e["output"]], checked=e["checked"], tics=e["tics"],
                                   seconds=round(e["seconds"], 3), game=e["game"], retried=e["retried"])
        if large and (time.time() - last >= 30 or len(results) == n):
            last = time.time()
            print(line(), flush=True)

    error, bad = None, None
    try:
        df.run_many(net, net.encode(digits.astype(np.float64)), workers=workers, backend="ray" if ray else "local",
                    address=args.ray or None, progress=progress)
    except (df.MismatchError, RuntimeError) as e:
        bad = getattr(e, "index", None)
        error = f"{names[bad]}: {e}" if bad is not None else str(e)
    wall = time.time() - t0
    shown = range(min(n, MNIST_ART if large else n))
    for start in range(0, len(shown), 4):
        block = shown[start:start + 4]
        print()
        for row in zip(*[_ascii(digits[i]) for i in block]):
            print("    ".join(row))
        cells = []
        for i in block:
            if i not in results:
                cells.append("DOOM: MISMATCH" if i == bad else "DOOM: not read")
                continue
            guess = doom_digit(results[i]["y"])
            verdict = "" if labels[i] is None else ("  right" if guess == labels[i] else f"  (it is a {labels[i]})")
            cells.append(f"DOOM reads: {guess}{verdict}")
        print("    ".join(f"{c:<28}" for c in cells).rstrip())
        print("    ".join(f"{names[i]:<28}" for i in block).rstrip())
    wrong = [i for i in sorted(results) if labels[i] is not None and doom_digit(results[i]["y"]) != labels[i]]
    if large and wrong:
        listed = ", ".join(f"{names[i]} read {doom_digit(results[i]['y'])} (is {labels[i]})" for i in wrong[:20])
        print(f"\nwrong: {listed}{f' (+{len(wrong) - 20} more)' if len(wrong) > 20 else ''}")
    done = [results[i] for i in sorted(results)]
    right = sum(1 for i in results if labels[i] is not None) - len(wrong)
    games = len({r["game"] for r in done})
    score = f"{right}/{n} read correctly | " if labels[0] is not None else ""
    tics = round(sum(r["tics"] for r in done) / len(done)) if done else 0
    print(f"\n{score}{len(done)}/{n} EXACT: every value DOOM computed ({done[0]['checked'] if done else 0} per "
          f"digit: hidden layer, comparisons, win counts and the one-hot answer) equals the NumPy integer reference "
          f"(tolerance 0); Python only compared | {tics:,} game tics per digit{' (mean)' if len(done) > 1 else ''} | "
          f"{wall:.0f} s | {len(done) / wall * 60:.1f} digits per minute in {games} DOOM "
          f"game{'s' if games != 1 else ''}")
    if error:
        print(f"MISMATCH: {error}" if bad is not None else f"FAILED: {error}")
    if args.record:
        import hashlib
        rec = dict(model="mnist/mlp.npz", model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   backend="ray" if ray else "local", workers=workers, games=games, digits=n, done=len(done),
                   correct=right if labels[0] is not None else None, exact=len(done), seconds=round(wall, 1),
                   digits_per_minute=round(len(done) / wall * 60, 2), error=error)
        rows = [json.dumps(dict(name=names[i], test_index=test_index[i], label=labels[i], doom=doom_digit(r["y"]), **r))
                for i, r in sorted(results.items())]                      # one line per digit
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(rec, indent=1)[:-2] + ',\n "results": [\n  '
                                     + ",\n  ".join(rows) + "\n ]\n}\n")
        print(f"wrote {args.record}")
    return 0 if error is None else 1


# --- doomformer draw: walk a digit onto a DOOM floor, the level reads it ---------------------------------------------
def _draw_report(pad, r, walked=None) -> None:
    from doomformer import drawpad
    print()
    for line in _ascii(r.pixels):
        print("    " + line)
    verdict = "" if walked is None else "  (right)" if r.digit == walked else f"  (wrong: the marine walked a {walked})"
    print(f"\nDOOM reads: {r.digit}{verdict}   -- the winner's doll lit it on the 7-segment display")
    print(f"EXACT: all {r.checked} values DOOM computed (hidden layer, {pad.net.weights[-3].shape[0]} score "
          f"comparisons and 10 win counts, each with its twin, and the one-hot answer) equal the NumPy integer "
          f"reference computed from the 784 pixel ports DOOM holds (tolerance 0); one display firing")
    print(f"pixels: the 784 ports {'equal' if r.canvas else 'DIFFER from'} the level's canvas (the tiles the pen inked)")
    if r.rasterised is not None:
        print(f"pixels: the 784 ports {'equal' if r.rasterised else 'DIFFER from'} the 3x3 brush stamped at every "
              f"position the marine had with FIRE held ({r.edge_positions} of them within {drawpad.EDGE} units of a "
              f"tile edge, counted for either tile)")
    n = sum(r.audit["commands"].values())
    print(f"READ pass: {r.tics:,} game tics = {r.tics / 35:.0f} s of real-time DOOM at 35 tics/s, {r.wall:.0f} s wall "
          f"| Python sent {n} commands, {r.readouts} readout scripts each followed by the ack script it waits for "
          f"({r.readout_waits} waited more than one advance), and nothing else ({r.audit['non_readout_commands']} "
          "rejected)")


DRAW_HELP = """You play in the DOOM window; click it so it has the keyboard and the mouse.
  walk   arrow keys or W/S; turn with Left/Right or the mouse; step sideways with A/D; hold Shift to run
  draw   hold Ctrl or the left mouse button while you walk on the canvas: the tiles you walk over turn blue
  READ   the right-hand switch south of the canvas: walk up to it and press Space or E. DOOM reads the digit in
         about {seconds:.0f} s (35 tics/s) and lights it on the display north of the canvas. The check prints here
         once every doll and you stand still.
  CLEAR  the left-hand switch wipes the canvas. While DOOM reads (until the display lights), the pen, READ and
         CLEAR are off.
  quit   Esc frees the mouse (DOOM's menu; the game pauses), then close the window; or Ctrl+C here"""


def _draw(args) -> int:
    import doomformer as df
    from doomformer import drawpad
    headless = args.digit is not None
    if not headless and sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or
                                                                   os.environ.get("WAYLAND_DISPLAY")):
        raise SystemExit("draw: you play it in a window and there is no display; `doomformer draw --digit 7` runs "
                         "headless (the marine walks a built-in 7)")
    _note_acc_build()
    t0 = time.time()
    if headless:
        pad = drawpad.DrawPad()
    else:
        import vizdoom as vzd
        pad = drawpad.DrawPad(spectator=True, resolution=vzd.ScreenResolution.RES_800X600)
    p = pad.brain.meta.layout["span"]
    print(f"DrawPad: the MNIST MLP and a 28x28 floor canvas in ONE DOOM map: {pad.info['dolls']:,} voodoo dolls, "
          f"{pad.info['sectors']:,} sectors, {p[0]:,} x {p[1]:,} map units (built in {time.time() - t0:.0f} s).")
    try:
        with pad:
            if headless:
                print(f"Python stands in for a human player: it presses the marine's movement keys and FIRE to walk a "
                      f"{args.digit} (built-in strokes, seed {args.seed}), then walks to the READ switch and presses "
                      "USE. From then on it only advances tics and runs readout scripts.", flush=True)
                strokes = drawpad.walked_strokes(args.digit, np.random.default_rng([args.seed, args.digit]))
                pad.draw(strokes)
                reads = pad.status()["reads"] + 1
                walk = len(pad.path)
                pad.press("read")
                print(f"walked {walk} tics, {len(pad.path) - walk} more to READ; DOOM is thinking ...", flush=True)
                with _Heartbeat(lambda: "", every=15):
                    r = pad.verify(reads)
                _draw_report(pad, r, args.digit)
                return 0
            pad.game                                       # opens the window; the level is running when it returns
            print(DRAW_HELP.format(seconds=round(pad.brain.meta.layout["estimated_settle_tics"] / 35, -1)), flush=True)

            def started(st):
                print(f"\nREAD {st['reads']}: DOOM is reading your digit; the check prints here once every doll and "
                      "you stand still", flush=True)

            def done(r):
                if isinstance(r, drawpad.ReadInterrupted):
                    print(r, flush=True)
                else:
                    _draw_report(pad, r)
                    print("\nCLEAR to draw another digit, or close the window to quit.", flush=True)

            pad.watch(done, started)
            print("the DOOM window closed")
            return 0
    except KeyboardInterrupt:
        print("stopped")
        return 130
    except df.MismatchError as e:
        print(f"MISMATCH: {e}")
        return 1


# --- doomformer ask: question answering, every number computed by DOOM ----------------------------------------------
def ask_prompt(question: str, vocab: str) -> tuple[str, str]:
    """(prompt, dropped characters): upper case, only the oracle's characters, one space between words, no trailing
    '?', then the '>' the oracle was trained to answer after."""
    keep = set(vocab) - {"$", ">"}
    q = question.upper()
    dropped = "".join(sorted({c for c in q if c not in keep and not c.isspace()}))
    q = " ".join("".join(c if c in keep else " " for c in q).split()).rstrip("?").strip()
    return (q + ">" if q else ""), dropped


def _ask(args) -> int:
    import doomformer as df
    from doomformer import intref

    net = df.Transformer.load("doomoracle")
    prompt, dropped = ask_prompt(" ".join(args.question), net.vocab)
    if not prompt:
        raise SystemExit("ask: ask DOOM something, e.g. doomformer ask \"what does iddqd do\"")
    if len(prompt) > net.T - 9:
        raise SystemExit(f"ask: DOOM remembers {net.T - 1} characters in all; keep the question under {net.T - 9}")
    workers = args.workers or min(os.cpu_count() or 1, 8)
    print(f"You ask DOOM: {prompt[:-1]}?" + (f"   (left out {dropped!r}: DOOM knows A-Z, space, ' ! ?)" if dropped else
                                             "") + "\n"
          f"DOOM answers with a {net.L}-layer transformer (d={net.d}, {net.H} heads) trained on 189 facts about DOOM, "
          f"split into tile maps over {workers} DOOM games.\nEvery letter is about {len(prompt)} positions of "
          f"voodoo-doll arithmetic: the first one takes a few minutes, the rest about 20 s each.", flush=True)
    _note_acc_build()
    t0 = time.time()
    text = prompt
    with df.compile(net, mode="tiles", workers=workers) as brain:
        print("DOOM is reading your question ", end="", flush=True)
        with _Dots():
            try:
                text = brain.generate(text, max_new=1)
            except df.MismatchError as e:
                print(f"\nMISMATCH: {e}")
                return 1
        print("\nDOOM says: " + text[len(prompt):], end="", flush=True)
        while len(text) > len(prompt) and len(text) < net.T - 1:
            try:
                longer = brain.generate(text, max_new=1)
            except df.MismatchError as e:
                print(f"\nMISMATCH: {e}")
                return 1
            if longer == text:
                break
            print(longer[len(text):], end="", flush=True)
            text = longer
        info = brain.info
    answer = text[len(prompt):]
    ref = intref.generate(net, prompt)
    exact = ref == text
    print(f"\n\n  {info['tile_runs']:,} tile runs in {workers} DOOM games | {info['checked_total']:,} values read back "
          f"and compared with the NumPy integer reference (tolerance 0) | {time.time() - t0:.0f} s\n"
          f"{'EXACT' if exact else 'MISMATCH'}: " + ("every value DOOM exposed equals the integer reference, and the "
                                                     "reference gives the same answer" if exact else
                                                     f"the reference answers {ref[len(prompt):]!r}"))
    return 0 if exact and answer else 1


class _Dots:
    """Prints a dot every few seconds on a daemon thread until the block ends."""

    def __init__(self, every: float = 5):
        self.every, self.done = every, threading.Event()

    def __enter__(self):
        def beat():
            while not self.done.wait(self.every):
                print(".", end="", flush=True)
        threading.Thread(target=beat, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.done.set()


def _hello(args) -> int:
    import doomformer as df

    rng = np.random.default_rng(args.seed)
    sizes = [4, 6, 5, 3]
    net = df.MLP([rng.integers(-20, 21, (sizes[l + 1], sizes[l])) for l in range(3)],
                 [rng.integers(-30, 31, sizes[l + 1]) for l in range(3)],
                 [np.cumsum(rng.integers(15, 40, 3)) for _ in range(3)])
    _note_acc_build()
    brain = df.compile(net)
    print(f"tiny integer MLP {'-'.join(map(str, sizes))} compiled into one DOOM map: {brain.info['dolls']} voodoo "
          f"dolls, {brain.info['sectors']} sectors ({brain.wad})")
    exact = 0
    with brain:
        for case in range(args.cases):
            x = rng.integers(-150, 151, net.n_in)
            try:
                y = brain(x)
            except df.MismatchError as e:
                print(f"input {case + 1}/{args.cases}  x={x.tolist()}  MISMATCH: {e}")
                continue
            exact += 1
            n_hidden = brain.info["checked"] - net.n_out
            print(f"input {case + 1}/{args.cases}  x={x.tolist()} -> y={y.tolist()}  EXACT ({net.n_out} outputs + "
                  f"{n_hidden} hidden readouts equal the integer reference; {brain.info['tics']} game tics)")
    print(f"{exact}/{args.cases} EXACT")
    return 0 if exact == args.cases else 1


def _load(path: str):
    import doomformer as df
    p = Path(path)
    if p.suffix == ".npz" and p.is_file():
        return df.MLP.load(p)
    if p.suffix == ".json" and p.is_file():
        return df.SpikingNet.load(p)
    try:
        return df.Transformer.load(path)
    except FileNotFoundError:
        raise SystemExit(f"plan: {path} is not a transformer model (a directory with config.json + params.npz, or a "
                         f"shipped name: {', '.join(df.available_models())}), an MLP .npz (MLP.save) or a spiking net "
                         ".json (SpikingNet.save)") from None


def _plan(args) -> int:
    import doomformer as df
    net = _load(args.model)
    print("counting dolls and sectors without building the map (large models take a few minutes) ...", flush=True)
    p = df.plan(net)
    kind = {"MLP": "MLP", "IntNet": "SpikingNet", "QModel": "Transformer"}[type(net).__name__]
    span = f" (span {p['span'][0]:,} x {p['span'][1]:,} units, limit 32,000 per axis)" if p["span"] else ""
    print(f"{kind} {args.model}: {p['dolls']} voodoo dolls, {p['sectors']} sectors, "
          f"fits in one map: {'yes' if p['fits'] else 'no'}{span}")
    est = p["map"].get("estimated_settle_tics") if p.get("map") else None
    if est:
        print(f"  one map: about {est:,} game tics per pass (estimate)")
    if p.get("tiles"):
        t = p["tiles"]
        print(f"  tiles: {t.get('tiles')} tile maps, {t.get('dolls')} dolls in total"
              f"{'' if t.get('fits', True) else ' (does not fit)'}")
    print(f"  df.compile(mode='auto') would use: {p['mode']}")
    return 0


def _play(args) -> int:
    import doomformer as df
    try:
        df.play(args.wad)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"play: {e}", file=sys.stderr)
        return 1
    return 0


def _doctor(args) -> int:
    from importlib import metadata

    from doomformer import acc
    ok = True

    def line(name, value, good=True):
        nonlocal ok
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {name:<10} {value}")

    line("python", f"{sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    try:
        line("doomformer", metadata.version("doomformer"))
    except metadata.PackageNotFoundError:
        line("doomformer", "not installed as a package (running from a checkout)")
    line("numpy", np.__version__)
    try:
        import vizdoom as vzd
        line("vizdoom", vzd.__version__, vzd.__version__ == "1.3.2")
        from doomformer.runtime import FREEDOOM2
        iwad = Path(FREEDOOM2)
        line("freedoom2", iwad if iwad.is_file() else f"missing ({iwad})", iwad.is_file())
    except ImportError as e:
        line("vizdoom", f"cannot import ({e})", False)
    path, source = acc.find()
    if path is None:
        missing = acc.missing_tools()
        target = acc.acc_cache() / acc.REV / "build" / "acc"
        line("acc", f"not built yet; built on first use into {target}" if not missing else
             f"not found and cannot be built: missing {', '.join(missing)}", not missing)
    else:
        try:
            from doomformer.udmf import compile_acs
            compile_acs('#include "zcommon.acs"\nscript 1 OPEN { }\n')
            line("acc", f"{path} ({source}; includes {acc.include_dir(path)}; compiles)")
        except Exception as e:                                 # report any failure of the compile probe
            line("acc", f"{path} ({source}) does not work: {e}", False)
    try:
        import torch
        line("torch", f"{torch.__version__} (optional: MLP.from_torch)")
    except ImportError:
        line("torch", "not installed (optional: MLP.from_torch)")
    line("cache", acc.cache_dir())
    line("display", "available (doomformer play)" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
         or not sys.platform.startswith("linux") else "none (doomformer play needs one; everything else is headless)")
    print("all good" if ok else "problems found (BAD lines)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="doomformer", description="Integer neural networks computed by DOOM.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="a trained transformer writes a DOOM level name, every number computed by DOOM and "
                                    "checked exactly (default: tile maps in a pool of DOOM games, under a minute)")
    d.add_argument("--one-map", action="store_true",
                   help="instead: a tiny trained model's forward pass inside ONE DOOM map, a letter at a time "
                        "(about 85 s per letter)")
    d.add_argument("--fast", action="store_true",
                   help="instead: a random-weight toy transformer inside ONE DOOM map, about 6 s in total")
    d.add_argument("--workers", type=int, default=0, help="DOOM game processes for the default demo (default: "
                                                          "min(CPU count, 8))")
    d.add_argument("--letters", type=int, default=None,
                   help="at most this many letters (default: the whole name; with --one-map or --fast: 1)")
    d.add_argument("--prompt", default="", help="characters to continue from (letters, space, ' and !)")
    d.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature in logit units (default 180 for the default demo, 0 = greedy otherwise)")
    d.add_argument("--seed", type=int, default=None, help="sampling seed (default 20 for the default demo, 0 otherwise)")
    d.add_argument("--model", default="doomnames",
                   help="shipped transformer for the default demo (default doomnames)")
    d.add_argument("--nightmare", action="store_true",
                   help="the big brain: doombrain (2 layers, 106 tile maps, 56,390 voodoo dolls) with temperature 150 "
                        "and seed 2, which write DEMON CANYON (a few minutes)")
    d.add_argument("--png", metavar="PATH", help="save a top-down picture of the map")
    d.add_argument("--softmax", action="store_true",
                   help="the same name model trained with softmax attention (doomnames_softmax): the dolls compute "
                        "the exponentials and divide by their sum")
    mn = sub.add_parser("mnist", help="DOOM reads handwritten digits: an MLP trained on MNIST, the whole network in ONE "
                                      "DOOM map, every value checked exactly")
    mn.add_argument("--image", metavar="PATH", help="read your own digit (a photo or drawing of one digit) instead")
    mn.add_argument("--digits", type=_at_least(1), default=4,
                    help="how many MNIST test digits (default 4; up to 100 from the shipped ones, more from the "
                         "10,000-digit test set, downloaded on first use)")
    mn.add_argument("--all", action="store_true", help="all 10,000 MNIST test digits, in order (downloaded on first "
                                                       "use)")
    mn.add_argument("--seed", type=int, default=None, help="which digits (default: different ones every run)")
    mn.add_argument("--workers", type=_at_least(0), default=0,
                    help="DOOM games at once (default: min(CPU count, 8); with --ray: the cluster's CPUs)")
    mn.add_argument("--ray", nargs="?", const="", default=None, metavar="ADDRESS",
                    help="run the games as Ray actors: on the cluster at ADDRESS, or a running cluster, or a local "
                         "Ray (needs pip install \"doomformer[ray]\")")
    mn.add_argument("--record", metavar="PATH", help="write every digit's result (DOOM's one-hot, label, tics, "
                                                      "seconds, game) to this JSON file")
    dw = sub.add_parser("draw", help="DrawPad: draw a digit by walking on a DOOM floor; a READ switch makes the level "
                                     "read it with voodoo dolls (needs a display; --digit runs headless)")
    dw.add_argument("--digit", type=int, choices=range(10), help="headless: the marine walks this digit (Python "
                                                                 "presses his keys) and presses READ")
    dw.add_argument("--seed", type=int, default=0, help="which built-in stroke variant, size, slant and wobble (random "
                                                        "draws seeded by [seed, digit]; default 0)")
    a = sub.add_parser("ask", help="ask DOOM a question about DOOM; a 2-layer transformer answers, every number "
                                   "computed by DOOM and checked exactly (several minutes)")
    a.add_argument("question", nargs="+", help='e.g. "what does iddqd do"')
    a.add_argument("--workers", type=int, default=0, help="DOOM game processes (default: min(CPU count, 8))")
    h = sub.add_parser("hello", help="compile a tiny MLP, run random inputs in DOOM, check every value exactly")
    h.add_argument("--cases", type=int, default=5)
    h.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("plan", help="count dolls/sectors of a model without building it")
    p.add_argument("model", help="transformer model dir, MLP .npz or spiking net .json")
    pl = sub.add_parser("play", help="walk around a compiled map in a ViZDoom window (needs a display)")
    pl.add_argument("wad")
    sub.add_parser("doctor", help="check python, vizdoom, ACC and the Freedoom IWAD")
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        tiles = not (args.fast or args.one_map)
        if args.softmax:
            if not tiles or args.nightmare:
                raise SystemExit("demo: --softmax runs the tile-map demo with doomnames_softmax; leave out "
                                 "--one-map/--fast/--nightmare")
            args.model = "doomnames_softmax"
        if args.nightmare:
            if not tiles:
                raise SystemExit("demo: --nightmare runs the tile-map demo; leave out --one-map/--fast")
            args.model = "doombrain"
            args.temperature = 150.0 if args.temperature is None else args.temperature
            args.seed = 2 if args.seed is None else args.seed
        args.temperature = args.temperature if args.temperature is not None else (180.0 if tiles else 0.0)
        args.seed = args.seed if args.seed is not None else (20 if tiles else 0)
    import signal

    from doomformer.acc import AccError
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # `kill`/`timeout`: unwind, close every DOOM game
    try:
        return {"demo": _demo, "mnist": _mnist, "draw": _draw, "ask": _ask, "hello": _hello, "plan": _plan,
                "play": _play, "doctor": _doctor}[args.cmd](args)
    except (AccError, FileNotFoundError) as e:                    # expected setup problems: one message, no traceback
        print(f"doomformer {args.cmd}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
