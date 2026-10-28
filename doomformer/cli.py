"""`doomformer` command line: demo, mnist, draw, ask, hello, plan, play, doctor."""
from __future__ import annotations

import argparse
import contextlib
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
        acc.require_tools()                      # a missing tool is reported before the build line, with an install hint
        print("building the ACS compiler (first use, a few seconds to a minute; needs git, cmake, a C compiler and "
              "network access) ...", flush=True)


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


def _check_writable(path: str, cmd: str, parents: bool = False) -> None:
    """Exit 2 with 'cannot write PATH: reason' if `path` cannot be written, before any DOOM work starts. `parents`:
    create the missing directories, as the writer later does."""
    p = Path(path)
    existed = p.exists()
    try:
        if parents:
            p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("ab"):
            pass
    except OSError as e:
        print(f"doomformer {cmd}: cannot write {path}: {e.strerror or e}", file=sys.stderr)
        raise SystemExit(2) from None
    if not existed:
        p.unlink(missing_ok=True)


def _version() -> str:
    from importlib import metadata
    try:
        return metadata.version("doomformer")
    except metadata.PackageNotFoundError:
        return "unknown (running from a checkout)"


def _attention(net) -> str:
    """The block options of a transformer, for the demo's model line."""
    parts = (["RMSNorm"] if net.norm == "rms" else []) + [
        "softmax attention (exponentials and the division by their sum computed by the dolls)"
        if net.attn == "softmax" else "ReLU attention"] + (["a gated (ReGLU) MLP"] if net.mlp == "reglu" else [])
    return ", ".join(parts)


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
        "   (one of its training names)" if text in corpus else "   (not one of its training names)")
    print(f"\nDOOM wrote: {text!r}{novelty}\n"
          f"  {runs:,} tile runs in {workers} parallel DOOM games | {info['dolls']:,} voodoo dolls "
          f"in the tile maps | {checked:,} values read back and compared with the NumPy integer reference "
          f"(tolerance 0) | {elapsed:.0f} s\n"
          f"{'EXACT' if exact else 'MISMATCH'}: " + ("every value DOOM exposed equals the integer reference, and the "
                                                     "reference writes the same name" if exact else f"the reference wrote {ref!r}"))
    if args.png:
        print(f"top-down picture of the biggest tile map: {args.png}")
    return 0 if exact else 1


ONE_MAP_PASS = {"doomnames_onemap": (85, 3500),            # about this many seconds and game tics per one-map pass
                "doomnames_onemap_softmax": (120, 4100)}   # (measured: 3,488 and 4,144 tics)


def _demo_map(args) -> int:
    """--one-map / --fast: the whole forward pass inside ONE DOOM map (a letter at a time)."""
    import doomformer as df
    from doomformer import intref, onemap
    from doomformer.record import topdown

    name = "doomnames_onemap_softmax" if args.softmax else "doomnames_onemap"
    net = onemap.tiny_model() if args.fast else df.Transformer.load(name)
    what = ("a random-weight toy transformer (writes nonsense; the checking is the same)" if args.fast else
            f"the shipped {name}: {net.L} layer, d={net.d}, {net.H} head, context {net.T}, {_attention(net)}, "
            "trained on DOOM level names of up to 5 letters")
    passes = min(args.letters or 1, _demo_args(args, net))
    print(f"A transformer's forward pass, computed inside ONE DOOM map.\nmodel: {what}", flush=True)
    _note_acc_build()
    t0 = time.time()
    with df.compile(net, opt=args.opt) as brain:
        dolls, sectors = brain.info["dolls"], brain.info["sectors"]
        print(f"map built in {time.time() - t0:.0f} s: {dolls:,} voodoo dolls cross lava lines in {sectors:,} sectors",
              flush=True)
        if not args.fast and not args.opt:                # measured at opt 0
            secs, tics = ONE_MAP_PASS[name]
            print(f"DOOM is running {passes} forward pass(es), about {secs} s each on one core ({tics:,} game tics of "
                  "dolls walking) ...", flush=True)
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


LOOP_PASS = {"doomnames_onemap": (190, 4000),              # about this many seconds and game tics per --loop pass
             "doomnames_onemap_softmax": (245, 4656)}      # (measured: 4,000 tics in 186-190 s, 4,656 in 237-259 s)


def _demo_loop(args) -> int:
    """--one-map --loop: ONE map writes the whole text; the level repeats its passes, Python only advances the clock
    and reads."""
    import tempfile

    from doomformer import intref, onemap, selfgen
    from doomformer.kit import MismatchError
    from doomformer.record import topdown

    name = "doomnames_onemap_softmax" if args.softmax else "doomnames_onemap"
    net = onemap.tiny_model() if args.fast else intref.QModel.load(name)
    what = ("a random-weight toy transformer (writes nonsense; the checking is the same)" if args.fast else
            f"the shipped {name}: {net.L} layer, d={net.d}, {net.H} head, context {net.T}, {_attention(net)}, "
            "trained on DOOM level names of up to 5 letters")
    _demo_args(args, net)
    show = lambda toks: "".join(net.vocab[t] for t in toks)
    print(f"A transformer writes a whole text inside ONE DOOM map, no Python in the loop.\nmodel: {what}", flush=True)
    _note_acc_build()
    with tempfile.TemporaryDirectory(prefix="doomformer-loop-") as tmp:
        wad = Path(tmp) / "selfgen.wad"
        t0 = time.time()
        meta = selfgen.build(net, wad, args.prompt, opt=args.opt)
        sched = meta.layout["selfgen"]
        print(f"map built in {time.time() - t0:.0f} s: {meta.n_dolls:,} voodoo dolls in {meta.n_sectors:,} sectors; "
              f"the dolls compare all {len(net.vocab)} logits at positions {sched['positions']} and write the winner "
              f"into the next position; the level runs {sched['passes']} passes on its own", flush=True)
        if not args.fast and not args.opt:                # measured at opt 0
            secs, tics = LOOP_PASS[name]
            print(f"about {secs} s and {tics:,} game tics per pass on one core ...", flush=True)
        print(f"context before pass 1: {show(sched['tokens'])!r} ($ = BOS / placeholder)", flush=True)
        now = {"k": 1}
        t1 = time.time()

        def on_pass(p):
            now["k"] = p["index"] + 1
            print(f"pass {p['index']}: {p['tics']:,} tics, {p['checked']:,} values exact; the dolls wrote "
                  f"{show(p['tokens'])!r}", flush=True)

        try:
            with selfgen.LoopRun(net, wad) as run, _Heartbeat(lambda: f" (pass {now['k']})", every=60):
                wrote = run.run(on_pass)
                audit = run.audit()
        except (MismatchError, AssertionError, RuntimeError) as e:
            print(f"MISMATCH: {e}")
            return 1
        if args.png:
            topdown(wad, size=(1000, 520), title=f"a transformer that writes on its own in one DOOM map "
                                                 f"({meta.n_dolls:,} voodoo dolls); DOOM wrote {wrote!r}").save(args.png)
    elapsed = time.time() - t1
    checked = sum(p["checked"] for p in run.passes)
    tics = sorted({p["tics"] for p in run.passes})
    tics = f"{tics[0]:,}" if len(tics) == 1 else f"{tics[0]:,} to {tics[-1]:,}"
    print(f"\nDOOM wrote: {wrote!r}\n"
          f"  {len(run.passes)} passes run by the level | {meta.n_dolls:,} voodoo dolls | {checked:,} values read back "
          f"and compared with the NumPy integer reference (tolerance 0) | {tics} game tics per pass | {elapsed:.0f} s\n"
          f"  audit: {sum(audit['commands'].values()):,} readout commands, {audit['non_readout_commands']} other "
          f"commands or game calls\n"
          f"EXACT: every value of every pass equals the integer reference, and intref.generate writes the same text")
    if args.png:
        print(f"top-down picture of the map: {args.png}")
    return 0


# --- demo --until-eos (doomformer/selfhalt.py) ------------------------------------------------------------------------
def _demo_until_eos(args) -> int:
    """--one-map --loop --until-eos: the level leaves its loop after the pass whose newest token is EOS; the map is the
    same for every prompt of one length, the prompt enters as initial state."""
    import hashlib
    import tempfile

    from doomformer import intref, onemap, selfhalt
    from doomformer.kit import MismatchError
    from doomformer.record import topdown

    name = "doomnames_onemap_softmax" if args.softmax else "doomnames_onemap"
    net = onemap.tiny_model() if args.fast else intref.QModel.load(name)
    what = ("a random-weight toy transformer (writes nonsense; the checking is the same)" if args.fast else
            f"the shipped {name}: {net.L} layer, d={net.d}, {net.H} head, context {net.T}, {_attention(net)}, "
            "trained on DOOM level names of up to 5 letters")
    _demo_args(args, net)
    show = lambda toks: "".join(net.vocab[t] for t in toks)
    print("A transformer writes a text inside ONE DOOM map and the map stops at its end-of-text token.\n"
          f"model: {what}", flush=True)
    _note_acc_build()
    with tempfile.TemporaryDirectory(prefix="doomformer-until-") as tmp:
        wad = Path(tmp) / "selfhalt.wad"
        t0 = time.time()
        meta = selfhalt.build_until(net, wad, len(args.prompt))
        sched = meta.layout["selfhalt"]
        sha = hashlib.sha256(wad.read_bytes()).hexdigest()
        n, bound = len(args.prompt), sched["passes"]
        print(f"map built in {time.time() - t0:.0f} s: {meta.n_dolls:,} voodoo dolls in {meta.n_sectors:,} sectors, "
              f"the same file for every prompt of {n} character{'s' * (n != 1)} (sha256 {sha}). The winner at each "
              f"position also sets that position's halt latch (1 for EOS); the level runs at most {bound} "
              f"pass{'es' * (bound != 1)} and leaves its loop after the pass whose newest token is EOS", flush=True)
        if not args.fast:
            secs, tics = LOOP_PASS[name]
            print(f"about {secs} s and {tics:,} game tics per pass on one core (as for --loop) ...", flush=True)
        now = {"k": 1}
        t1 = time.time()

        def on_pass(p):
            now["k"] = p["index"] + 1
            print(f"pass {p['index']}: {p['tics']:,} tics, {p['checked']:,} values exact; the dolls wrote "
                  f"{show(p['tokens'])!r}; the level read latch {p['latch']}", flush=True)

        try:
            with selfhalt.UntilEosRun(net, wad, args.prompt) as run, _Heartbeat(lambda: f" (pass {now['k']})",
                                                                                every=60):
                boot = run.load()
                print(f"prompt {args.prompt!r} loaded as initial state: {boot['loads']} port writes, then sealed; "
                      f"{boot['checked']} state values read back exact before pass 1 "
                      f"({show(run.reference[0].tokens)!r}, $ = BOS / placeholder)", flush=True)
                wrote = run.run(on_pass)
                audit = run.audit()
        except (MismatchError, AssertionError, RuntimeError) as e:
            print(f"MISMATCH: {e}")
            return 1
        if args.png:
            topdown(wad, size=(1000, 520), title=f"a transformer that stops at its end-of-text token in one DOOM "
                                                 f"map ({meta.n_dolls:,} voodoo dolls); DOOM wrote {wrote!r}"
                    ).save(args.png)
    elapsed = time.time() - t1
    checked = sum(p["checked"] for p in run.passes)
    tics = sorted({p["tics"] for p in run.passes})
    tics = f"{tics[0]:,}" if len(tics) == 1 else f"{tics[0]:,} to {tics[-1]:,}"
    why = "its newest token is EOS" if run.reference[run.stop - 1].winners[run.stop - 1] == 0 else "the bound"
    print(f"\nDOOM wrote: {wrote!r}\n"
          f"  the level stopped after pass {len(run.passes)} of at most {sched['passes']} ({why}) | {meta.n_dolls:,} "
          f"voodoo dolls | {checked:,} values read back and compared with the NumPy integer reference (tolerance 0) | "
          f"{tics} game tics per pass | {elapsed:.0f} s\n"
          f"  audit: {audit['loaded']} prompt loads before the seal, then {sum(audit['commands'].values()):,} readout "
          f"commands, {audit['non_readout_commands']} other commands or game calls\n"
          f"EXACT: every value of every pass equals the integer reference, the level stopped where greedy generation "
          f"stops, and intref.generate writes the same text")
    if args.png:
        print(f"top-down picture of the map: {args.png}")
    return 0
# --- end demo --until-eos ---------------------------------------------------------------------------------------------



def _demo(args) -> int:
    if args.fast and args.one_map:
        raise SystemExit("demo: --fast and --one-map are alternatives, not a pair")
    if args.until_eos and not args.loop:                      # --until-eos: doomformer/selfhalt.py
        raise SystemExit("demo: --until-eos ends a --loop early; add --loop")
    if args.loop:
        if not (args.fast or args.one_map):
            raise SystemExit("demo: --loop runs in one map; add --one-map (or --fast)")
        if args.temperature or args.letters is not None:
            raise SystemExit("demo: --loop decodes greedily and writes every position; leave out --temperature/--letters")
        return _demo_until_eos(args) if args.until_eos else _demo_loop(args)
    return _demo_map(args) if args.fast or args.one_map else _demo_tiles(args)


# --- doomformer mnist: DOOM reads handwritten digits -----------------------------------------------------------------
MNIST_TILES_SECONDS = 1.5   # measured: see docs/scaling.md (one worker, tile games resident)
MNIST_SECONDS = 90          # about: 75 s of engine CPU on a quiet machine, 76-104 s of wall time per digit under load (docs/scaling.md)
def _ascii(bits: np.ndarray) -> list[str]:
    """A 28x28 0/1 image as 14 text lines of half blocks (two pixel rows per line)."""
    img = np.asarray(bits).reshape(28, 28).astype(int)
    return ["".join(" ▀▄█"[img[r, c] + 2 * img[r + 1, c]] for c in range(28)) for r in range(0, 28, 2)]


DIGIT_WARNING = "this does not look like a single centred digit; DOOM will still read it"
MIN_INK, MAX_INK, MAX_BLOBS = 15, 0.35, 3       # ink pixels, ink share of the 28x28, separate blobs (see tests)


def _blobs(mask: np.ndarray) -> int:
    """Number of 8-connected groups of True cells."""
    m = np.pad(np.asarray(mask, bool), 1)
    seen = np.zeros_like(m)
    n = 0
    for r, c in zip(*np.nonzero(m)):
        if seen[r, c]:
            continue
        n += 1
        seen[r, c] = True
        stack = [(r, c)]
        while stack:
            y, x = stack.pop()
            for yy in (y - 1, y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if m[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
    return n


def prepare_digit(path) -> tuple[np.ndarray, str | None]:
    """A picture of one digit -> (the 784 0/1 pixels the MNIST model reads, a warning or None). Prepared the way MNIST
    was: ink bright on dark (a light picture is inverted), the ink cropped, scaled into a 20x20 box, centred by its
    centre of mass in 28x28 and thresholded at 0.5. The warning says the result does not look like one digit: too few
    or too many ink pixels, or more than MAX_BLOBS separate blobs in the 28x28 or in the source picture."""
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
    bits = (canvas > 0.5).astype(np.uint8).reshape(-1)
    if not bits.any():
        raise ValueError("nothing to read: after scaling to 28x28 no ink is left; draw one thick dark digit on light "
                         "paper")
    k = max(1, -(-max(ink.shape) // 96))                 # source blobs are counted on a picture of at most ~96 pixels
    pad = [(0, -s % k) for s in ink.shape]
    small = np.pad(ink, pad).reshape((ink.shape[0] + pad[0][1]) // k, k, (ink.shape[1] + pad[1][1]) // k, k).any((1, 3))
    why = []
    if bits.sum() < MIN_INK:
        why.append(f"the 28x28 keeps {int(bits.sum())} ink pixels, fewer than {MIN_INK}: thin strokes or small text")
    if bits.mean() > MAX_INK:
        why.append(f"ink covers {bits.mean():.0%} of the 28x28")
    if _blobs(bits.reshape(28, 28)) > MAX_BLOBS or _blobs(small) > MAX_BLOBS:
        why.append(f"more than {MAX_BLOBS} separate blobs")
    return bits, f"{DIGIT_WARNING} ({'; '.join(why)})" if why else None


def digit_from_image(path) -> np.ndarray:
    """The 784 0/1 pixels of `prepare_digit`."""
    return prepare_digit(path)[0]


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
        from PIL import UnidentifiedImageError
        try:
            bits, warning = prepare_digit(args.image)
        except UnidentifiedImageError:
            print(f"doomformer mnist: {args.image} is not an image PIL can read", file=sys.stderr)
            raise SystemExit(2) from None
        except (OSError, ValueError) as e:
            print(f"doomformer mnist: {e}", file=sys.stderr)
            raise SystemExit(2) from None
        print(f"{Path(args.image).name} as the 28x28 picture DOOM will read:")
        for line in _ascii(bits):
            print("    " + line)
        if warning:
            print(f"doomformer mnist: warning: {warning}", file=sys.stderr)
        sys.stdout.flush()
        return np.array([bits]), [None], [Path(args.image).name], [None]
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
    n, ray, tiles = len(digits), args.ray is not None, args.tiles
    if ray and tiles:
        raise SystemExit("mnist: --tiles runs on this machine; leave out --ray")
    if tiles and args.opt:
        raise SystemExit("mnist: --opt sets the one-map schedule (tiles have no waves); leave out --tiles")
    if ray:
        try:
            df.fleet.ray_module()
        except ImportError as e:
            raise SystemExit(f"mnist: {e}") from None
    workers = min(args.workers, n) if args.workers else (None if ray else min(os.cpu_count() or 1, 8, n))
    if tiles:
        L = net.n_layers
        where = (f"with {workers} worker{'s' if workers > 1 else ''}; each holds the {L} tile games of the network "
                 f"({L * workers} engine processes, {workers} computing at once)")
    else:
        where = (f"in {workers} DOOM game{'s' if workers > 1 else ''} at once{' (Ray)' if ray else ''}" if workers
                 else "on Ray, one DOOM game per CPU of the cluster")
    if tiles:
        p = df.plan(net, mode="tiles")
        print(f"DOOM reads handwritten digits. The network (784 pixels -> {net.weights[0].shape[0]} -> "
              f"{net.weights[1].shape[0]} score comparisons -> 10 win counts -> one-hot) runs as {p['tiles']['tiles']} "
              f"TILE maps, one per layer: {p['tiles']['dolls']:,} voodoo dolls, {p['tiles']['sectors']:,} sectors in "
              f"all. This is the multi-map design: DOOM computes every layer's sums, comparisons and win counts, "
              f"and PYTHON COPIES the integer values of each layer into the next tile's game (the one-map design, "
              f"without `--tiles`, has no Python between layers and is much slower).\nreading {n} "
              f"digit{'s' if n > 1 else ''} {where} (about {MNIST_TILES_SECONDS} s per digit per worker) ...",
              flush=True)
    else:
        p = df.plan(net, mode="map", opt=args.opt)
        per_digit = MNIST_SECONDS                    # measured at opt 0; other levels: tics x cost.py's per-tic cost
        if args.opt:
            from doomformer.cost import map_seconds
            per_digit = max(1, round(map_seconds(p["sectors"], p["map"]["estimated_settle_tics"], p["dolls"])))
        print(f"DOOM reads handwritten digits. The whole network (784 pixels -> {net.weights[0].shape[0]} -> "
              f"{net.weights[1].shape[0]} score comparisons -> 10 win counts -> one-hot) is ONE DOOM map:\n"
              f"{p['dolls']:,} voodoo dolls, {p['sectors']:,} sectors; every ink pixel releases a doll that crosses "
              f"the weight lines of the hidden layer, and dolls compare every pair of digit scores, so DOOM itself "
              f"picks the digit.\nreading {n} digit{'s' if n > 1 else ''} {where} (about {per_digit} s per digit "
              f"per game) ...", flush=True)
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
                f"{done / (time.time() - t0) * 60:.1f} digits/min | {games} {'workers' if tiles else 'games'} | "
                f"about {left:.0f} min left")

    journal = open(f"{args.record}.partial", "w") if args.record else None   # one line per digit as it finishes

    def progress(e):
        nonlocal last
        results[e["index"]] = dict(y=[int(v) for v in e["output"]], checked=e["checked"], tics=e["tics"],
                                   seconds=round(e["seconds"], 3), game=e["game"], retried=e["retried"])
        if journal:
            journal.write(json.dumps(dict(index=e["index"], **results[e["index"]])) + "\n")
            journal.flush()
        if large and (time.time() - last >= 30 or len(results) == n):
            last = time.time()
            print(line(), flush=True)

    error, bad = None, None
    try:
        with _Heartbeat(lambda: f", {len(results)}/{n} digits read", every=30) if not large else contextlib.nullcontext():
            df.run_many(net, net.encode(digits.astype(np.float64)), workers=workers, backend="ray" if ray else "local",
                        address=args.ray or None, progress=progress, mode="tiles" if tiles else "map", opt=args.opt)
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
    steps = (f"Python wrote the 784 pixels, copied the layer values between the {net.n_layers} tile games and "
             "compared" if tiles else "Python wrote the 784 pixels, released the dolls and compared")
    print(f"\n{score}{len(done)}/{n} EXACT: every value DOOM computed ({done[0]['checked'] if done else 0} per "
          f"digit: hidden layer, comparisons, win counts and the one-hot answer) equals the NumPy integer reference "
          f"(tolerance 0); {steps} | {tics:,} game tics per "
          f"digit{' (mean)' if len(done) > 1 else ''}{' summed over the tiles' if tiles else ''} | "
          f"{wall:.0f} s | {len(done) / wall * 60:.1f} digits per minute in {games} "
          f"{'tile worker' if tiles else 'DOOM game'}{'s' if games != 1 else ''}")
    matches = sum(int(np.array_equal(r["y"], net.reference(net.encode(digits[i].astype(np.float64)))["y"]))
                  for i, r in results.items())
    if large:
        print(f"reference: DOOM's one-hot output equals NumPy's integer output on {matches}/{len(results)} digits")
    if error:
        print(f"MISMATCH: {error}" if bad is not None else f"FAILED: {error}")
    if args.record:
        import hashlib
        rec = dict(model="mnist/mlp.npz", model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   backend="ray" if ray else "local", mode="tiles" if tiles else "map",
                   tiles=net.n_layers if tiles else 1, workers=workers, games=games, digits=n, done=len(done),
                   correct=right if labels[0] is not None else None, exact=len(done), reference_match=matches,
                   accuracy_percent=round(100 * right / n, 2) if labels[0] is not None else None,
                   seconds=round(wall, 1), digits_per_minute=round(len(done) / wall * 60, 2), error=error)
        if args.opt:
            rec["opt"] = args.opt
        rows = [json.dumps(dict(name=names[i], test_index=test_index[i], label=labels[i], doom=doom_digit(r["y"]), **r))
                for i, r in sorted(results.items())]                      # one line per digit
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(rec, indent=1)[:-2] + ',\n "results": [\n  '
                                     + ",\n  ".join(rows) + "\n ]\n}\n")
        print(f"wrote {args.record}")
        journal.close()
        Path(f"{args.record}.partial").unlink(missing_ok=True)
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


DRAW_OPT_NOTE = {
    0: "The original schedule; --opt 3 reads in about 8 s.",
    1: "Dolls that can never hit the same plane share a wave; --opt 3 reads in about 8 s.",
    2: "As --opt 1 with tuned timing; --opt 3 reads in about 8 s.",
    3: "An ACS script copies each layer's light levels into the next layer's gate heights, so every layer runs as "
       "one wave (the original --opt 0 needs about 163 s).",
}

DRAW_HELP = """You play in the DOOM window; click it so it has the keyboard and the mouse.
  walk   arrow keys or W/S; turn with Left/Right or the mouse; step sideways with A/D; hold Shift to run
  draw   hold Ctrl or the left mouse button while you walk on the canvas: the tiles you walk over turn blue
  READ   the right-hand switch south of the canvas: walk up to it and press Space or E. DOOM reads the digit in
         about {seconds} s of game time (35 tics/s) and lights it on the display north of the canvas. The check
         prints here once every doll and you stand still.
  --opt {opt}  {opt_note}
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
        pad = drawpad.DrawPad(opt=args.opt)
    else:
        import vizdoom as vzd
        pad = drawpad.DrawPad(spectator=True, resolution=vzd.ScreenResolution.RES_800X600, opt=args.opt)
    p = pad.brain.meta.layout["span"]
    print(f"DrawPad: the MNIST MLP and a 28x28 floor canvas in ONE DOOM map: {pad.info['dolls']:,} voodoo dolls, "
          f"{pad.info['sectors']:,} sectors, {p[0]:,} x {p[1]:,} map units (built in {time.time() - t0:.0f} s).")
    try:
        with pad:
            if headless:
                print(f"Python stands in for a human player: it presses the marine's movement keys and FIRE to walk a "
                      f"{args.digit} (built-in strokes, seed {args.seed}), then walks to the READ switch and presses "
                      f"USE. From then on it only advances tics and runs readout scripts. --opt {args.opt}: "
                      f"{DRAW_OPT_NOTE[args.opt]}",
                      flush=True)
                strokes = drawpad.walked_strokes(args.digit, np.random.default_rng([args.seed, args.digit]))
                with _Heartbeat(lambda: "", every=30):
                    pad.draw(strokes)
                    reads = pad.status()["reads"] + 1
                    walk = len(pad.path)
                    pad.press("read")
                    print(f"walked {walk} tics, {len(pad.path) - walk} more to READ; DOOM is thinking (about "
                          f"{pad.read_s} s of game time at --opt {args.opt}) ...", flush=True)
                    r = pad.verify(reads)
                _draw_report(pad, r, args.digit)
                return 0
            pad.game                                       # opens the window; the level is running when it returns
            print(DRAW_HELP.format(seconds=pad.read_s, opt=args.opt, opt_note=DRAW_OPT_NOTE[args.opt]), flush=True)

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
    first, later, first_runs = ask_estimate(net, len(prompt), workers)
    games = f"{workers} DOOM game{'s' if workers != 1 else ''}"
    print(f"You ask DOOM: {prompt[:-1]}?" + (f"   (left out {dropped!r}: DOOM knows A-Z, space, ' ! ?)" if dropped else
                                             "") + "\n"
          f"DOOM answers with a {net.L}-layer transformer (d={net.d}, {net.H} heads) trained on 189 facts about DOOM, "
          f"split into tile maps over {games}.\nThe first letter runs all {len(prompt) + 1} positions of "
          f"the question through the network ({first_runs:,} tile runs): about {first / 60:.0f} min with {games}. "
          f"Each later letter adds one position: about {later:.0f} s. (Fitted to two timed runs on a loaded "
          f"machine; a quiet one was up to twice as fast.)", flush=True)
    _note_acc_build()
    t0 = time.time()
    text = prompt
    with df.compile(net, mode="tiles", workers=workers) as brain:
        print("DOOM is reading your question ...", flush=True)
        with _Heartbeat(lambda: f", {len(brain.pool.log)} of about {first_runs:,} tile runs for the first letter",
                        every=30):
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
    print(f"\n\n  {info['tile_runs']:,} tile runs in {games} | {info['checked_total']:,} values read back "
          f"and compared with the NumPy integer reference (tolerance 0) | {time.time() - t0:.0f} s\n"
          f"{'EXACT' if exact else 'MISMATCH'}: " + ("every value DOOM exposed equals the integer reference, and the "
                                                     "reference gives the same answer" if exact else
                                                     f"the reference answers {ref[len(prompt):]!r}"))
    return 0 if exact and answer else 1


def ask_runs(net, t: int) -> int:
    """Tile runs of one step at position t of a ReLU-attention transformer: embed, unembed, and per layer qkv, out,
    mlp_up, mlp_dn, d wsum and H*(t+1) score runs. (28 steps of the oracle: 5,320 runs, the logged ask.)"""
    return 2 + net.L * (4 + net.d + net.H * (t + 1))


ASK_SETUP_SECONDS, ASK_RUN_SECONDS = 96, 0.63       # fitted to the two logged runs of the same question (5,320 tile
                                                    # runs): 653 s on 6 games and 1,210 s on 3, on a shared machine


def ask_estimate(net, prompt_len: int, workers: int) -> tuple[float, float, int]:
    """(seconds of the first letter, seconds of a later letter, tile runs of the first letter) for `workers` games:
    a fixed setup plus game-seconds per tile run divided by the games."""
    runs = sum(ask_runs(net, t) for t in range(prompt_len + 1))
    return (ASK_SETUP_SECONDS + ASK_RUN_SECONDS * runs / workers,
            ASK_RUN_SECONDS * ask_runs(net, prompt_len + 4) / workers, runs)


def _hello(args) -> int:
    import doomformer as df

    rng = np.random.default_rng(args.seed)
    sizes = [4, 6, 5, 3]
    net = df.MLP([rng.integers(-20, 21, (sizes[l + 1], sizes[l])) for l in range(3)],
                 [rng.integers(-30, 31, sizes[l + 1]) for l in range(3)],
                 [np.cumsum(rng.integers(15, 40, 3)) for _ in range(3)])
    _note_acc_build()
    brain = df.compile(net, opt=args.opt)
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
    """A network from a name or path: an MLP .npz, a model directory holding mlp.npz (models/mnist), `mnist` (the shipped
    MNIST MLP), a spiking net .json, a transformer model directory or a shipped transformer name."""
    import doomformer as df
    from doomformer import shipped
    p = Path(path)
    if p.suffix == ".npz" and p.is_file():
        return df.MLP.load(p)
    if p.suffix == ".json" and p.is_file():
        return df.SpikingNet.load(p)
    if p.suffix in (".npz", ".json"):
        raise SystemExit(f"no such file: {path}")
    if p.is_dir() and (p / "mlp.npz").is_file() and not (p / "config.json").is_file():
        return df.MLP.load(p / "mlp.npz")
    if not p.exists() and p.name == "mnist" and p.parent in (Path("."), Path("models")):
        return df.MLP.load(shipped.data("mnist/mlp.npz"))
    try:
        return df.Transformer.load(path)
    except FileNotFoundError:
        raise SystemExit(f"{path} is not a model: give a shipped name ({', '.join(df.available_models())}, mnist), a "
                         "model directory (config.json + params.npz for a transformer, mlp.npz for an MLP such as "
                         "models/mnist), an MLP .npz (MLP.save) or a spiking net .json (SpikingNet.save)") from None


def _seconds(s: float) -> str:
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.1f} s" if s < 100 else f"{s / 60:.1f} min" if s < 7200 \
        else f"{s / 3600:.1f} h"


def _plan(args) -> int:
    import doomformer as df
    net = _load(args.model)
    print("counting dolls and sectors without building the map (large models take a few minutes) ...", flush=True)
    p = df.plan(net, opt=args.opt)
    kind = {"MLP": "MLP", "IntNet": "SpikingNet", "QModel": "Transformer"}[type(net).__name__]
    span = f" (span {p['span'][0]:,} x {p['span'][1]:,} units, limit 32,000 per axis)" if p["span"] else ""
    print(f"{kind} {args.model}: {p['dolls']} voodoo dolls, {p['sectors']} sectors, "
          f"fits in one map: {'yes' if p['fits'] else 'no'}{span}")
    m, t = p.get("map") or {}, p.get("tiles")
    if m.get("estimated_settle_tics"):
        if "calibrated_tics" in m:
            print(f"  one map: about {m['calibrated_tics']:,} game tics per pass, +-{m['tics_error_pct']:g} % "
                  f"(+-{m['tics_band']} tics = max(20 tics, 1.5 %); all 68 measured settles of the calibration lay inside, "
                  f"the largest error was 40 tics, 15 on nets under 400 tics, so below 400 tics the band in percent is "
                  f"wide; docs/scheduling.md; the geometry bound "
                  f"the runner's timeout uses is {m['estimated_settle_tics']:,})"
                  + (f"\n  one map: predicted {_seconds(m['predicted_seconds'])} per pass (cost model on the "
                     f"{m['estimated_settle_tics']:,}-tic bound)" if "predicted_seconds" in m else ""))
        else:
            print(f"  one map: about {m['estimated_settle_tics']:,} game tics per pass (estimate)"
                  + (f", predicted {_seconds(m['predicted_seconds'])} per pass" if "predicted_seconds" in m else ""))
    if t:
        print(f"  tiles: {t.get('tiles')} tile maps, {t.get('dolls')} dolls in total"
              f"{'' if t.get('fits', True) else ' (does not fit)'}"
              + (f", about {t['predicted_tics']:,} game tics summed, predicted {_seconds(t['predicted_seconds'])} per "
                 "pass" if "predicted_seconds" in t else ""))
    if "predicted_seconds" in m or (t and "predicted_seconds" in t):
        from doomformer import cost
        print(f"  predicted seconds: engine time of one pass with the games loaded (about {cost.MAP_NS_PER_SECTOR_TIC:.0f}"
              " ns per sector per tic, fitted under machine load of 10 to 40, error about +-20%, docs/scaling.md; the shipped MNIST "
              "at opt 0 takes 75 s of engine CPU on a quiet machine against 102 s predicted); a loaded or slower machine "
              "takes longer")
    if p.get("recommended_ratio") and p["recommended_ratio"] >= 1.5:
        fast = p["recommended"]
        how = ("df.run_many(net, xs, mode='tiles'); `doomformer mnist --tiles` for the MNIST net" if fast == "tiles"
               else "the default one map")
        print(f"  recommendation: {fast} is faster by about {p['recommended_ratio']:g}x ({how})"
              + ("; one map keeps Python out of the loop between layers" if fast == "tiles" else ""))
    elif p.get("recommended_ratio"):
        print(f"  recommendation: map and tiles are within {p['recommended_ratio']:g}x of each other")
    print(f"  df.compile(mode='auto') would use: {p['mode']}")
    if args.explain:
        if kind == "SpikingNet" and args.opt:
            print("plan: --opt > 0 is not available for spiking nets (their potentials are state the schedule "
                  "analysis does not model)", file=sys.stderr)
            return 1
        from doomformer.explain import explain_text
        print()
        print(explain_text(net, args.opt))
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


def _facts(rest: list[str]) -> int:
    from doomformer import facts
    return facts.main(rest)


# --- doomformer tune: failure edges of the one-map timing constants (doomformer/tune.py) -------------------------------
def _tune(args) -> int:
    from doomformer import tune
    return tune.run(args)
# --- end tune ----------------------------------------------------------------------------------------------------------


# --- doomformer lint: static schedule check of a compiled one-map brain (doomformer/schedule.py) -----------------------
def _lint(args) -> int:
    import json

    if args.fuzz:
        from doomformer import lintfuzz
        return lintfuzz.run_cli(args)
    if not args.wad:
        raise SystemExit("doomformer lint: give a .wad, or --fuzz")

    from doomformer.schedule import check_schedule
    net = _load(args.net) if args.net else None
    try:
        report = check_schedule(args.wad, net=net, witnesses=args.witnesses)
    except ValueError as e:
        print(f"doomformer lint: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict()) if args.json else report)
    return 0 if report.ok else 1
# --- end lint ----------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="doomformer", description="Integer neural networks computed by DOOM.")
    ap.add_argument("--version", action="version", version=f"doomformer {_version()}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    opt_help = ("schedule of the one-map brain: 0 the original (default), 1 lets dolls that can never hit the same "
                "plane together share a wave, 2 also uses the tuned timing of doomformer/tuning.json, 3 also runs "
                "every layer as one wave into light accumulators that an ACS script copies into the next layer's gates")
    opts = (0, 1, 2, 3)
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
                   help="the same name model trained with softmax attention (doomnames_softmax; with --one-map "
                        "doomnames_onemap_softmax): the dolls compute the exponentials and divide by their sum")
    d.add_argument("--loop", action="store_true",
                   help="with --one-map (or --fast): the map writes the whole text by itself; the winner doll of each "
                        "position writes the next token, the level repeats its passes, Python only advances the "
                        "clock, reads and checks")
    d.add_argument("--until-eos", action="store_true",                   # doomformer/selfhalt.py
                   help="with --loop: the winner doll also sets a halt latch and the level stops after the pass whose "
                        "newest token is EOS; one map for every prompt of the same length, the prompt is written as "
                        "initial state before the first pass")
    d.add_argument("--opt", type=int, choices=opts, default=0, help=f"with --one-map or --fast: the {opt_help}")
    mn = sub.add_parser("mnist", help="DOOM reads handwritten digits: an MLP trained on MNIST, the whole network in ONE "
                                      "DOOM map, every value checked exactly")
    mn.add_argument("--image", metavar="PATH",
                    help="read your own digit (a photo or drawing of one digit) instead. Prepared like MNIST: "
                         "inverted if the paper is light, cropped to the ink, scaled into 20x20, centred by centre "
                         "of mass in 28x28, thresholded at 0.5. The 28x28 picture prints before the run, with a "
                         "warning if it does not look like one digit")
    mn.add_argument("--digits", type=_at_least(1), default=4,
                    help="how many MNIST test digits (default 4; up to 100 from the shipped ones, more from the "
                         "10,000-digit test set, downloaded on first use)")
    mn.add_argument("--all", action="store_true", help="all 10,000 MNIST test digits, in order (downloaded on first "
                                                       "use)")
    mn.add_argument("--seed", type=int, default=None, help="which digits (default: different ones every run)")
    mn.add_argument("--workers", type=_at_least(0), default=0,
                    help="DOOM games at once (default: min(CPU count, 8); with --ray: the cluster's CPUs); with "
                         "--tiles: worker processes, each holding the 4 tile games")
    mn.add_argument("--tiles", action="store_true",
                    help="run the network as one tile map per layer in resident games, Python copying the integer "
                         "layer values between them (the multi-map design; about 1 s per digit instead of 90, same "
                         "248 values checked exactly)")
    mn.add_argument("--ray", nargs="?", const="", default=None, metavar="ADDRESS",
                    help="run the games as Ray actors: on the cluster at ADDRESS, or a running cluster, or a local "
                         "Ray (needs pip install \"doomformer[ray]\")")
    mn.add_argument("--record", metavar="PATH", help="write every digit's result (DOOM's one-hot, label, tics, "
                                                      "seconds, game) to this JSON file")
    mn.add_argument("--opt", type=int, choices=opts, default=0, help=f"the {opt_help}")
    dw = sub.add_parser("draw", help="DrawPad: draw a digit by walking on a DOOM floor; a READ switch makes the level "
                                     "read it with voodoo dolls (needs a display; --digit runs headless)",
                        description="Without --digit this opens a DOOM window and needs a display. With --digit the "
                                    "marine walks a built-in digit headless (about 15 s at the default --opt 3, about two "
                                    "minutes at --opt 0).")
    dw.add_argument("--digit", type=int, choices=range(10), help="headless: the marine walks this digit (Python "
                                                                 "presses his keys) and presses READ")
    dw.add_argument("--seed", type=int, default=0, help="which built-in stroke variant, size, slant and wobble (random "
                                                        "draws seeded by [seed, digit]; default 0)")
    dw.add_argument("--opt", type=int, choices=opts, default=3,
                    help="the " + opt_help.replace("0 the original (default)", "0 the original") + " (default 3 here; "
                         "READ takes about 8 s of game time at 3 and about 163 s at 0)")
    a = sub.add_parser("ask", help="ask DOOM a question about DOOM; a 2-layer transformer answers, every number "
                                   "computed by DOOM and checked exactly (several minutes)")
    a.add_argument("question", nargs="+", help='e.g. "what does iddqd do"')
    a.add_argument("--workers", type=int, default=0, help="DOOM game processes (default: min(CPU count, 8))")
    h = sub.add_parser("hello", help="compile a tiny MLP, run random inputs in DOOM, check every value exactly")
    h.add_argument("--cases", type=_at_least(1), default=5, help="how many random inputs to run (default 5)")
    h.add_argument("--seed", type=int, default=0, help="seed of the network and of the inputs (default 0)")
    h.add_argument("--opt", type=int, choices=opts, default=0, help=f"the {opt_help}")
    p = sub.add_parser("plan", help="count dolls/sectors of a model without building it")
    from doomformer import shipped
    p.add_argument("model", help=f"a shipped model name ({', '.join(shipped.names())}, mnist), a model directory "
                                 "(models/mnist too), an MLP .npz or a spiking net .json")
    p.add_argument("--opt", type=int, choices=opts, default=0, help=f"the {opt_help} (it sets the tics estimate, printed with its "
                                                          "error band)")
    p.add_argument("--explain", action="store_true", help="also print the one-map schedule per layer (what dominates "
                                                          "its game tics) and the cumulative effect of each pass")
    pl = sub.add_parser("play", help="walk around a compiled map in a ViZDoom window (needs a display)")
    pl.add_argument("wad", help="a compiled map (.wad) to walk around in")
    sub.add_parser("doctor", help="check python, vizdoom, ACC and the Freedoom IWAD")
    sub.add_parser("facts", help="run the engine facts of docs/engine_limits.md as probe maps in ViZDoom (PASS/FAIL "
                   "per fact, about 15 s; options: --json, --only NAME, --list, --export DIR [--check]; "
                   "`doomformer facts --help`)")
    from doomformer import fuzz as _fuzz
    from doomformer import verify as _verify
    _fuzz.add_arguments(sub.add_parser("fuzz", help="differential fuzzing: random nets compiled and run in DOOM, every "
                                       "value compared with the integer reference (--mutate: mutation kill rates; "
                                       "--plant: self-test with a planted compiler bug)"))
    _verify.add_arguments(sub.add_parser("verify", help="verify a small MLP over its whole input domain: --mode all "
                                         "enumerates every input, --mode classes one run per input class"))
    from doomformer import machine_cli as _machine
    _machine.add_parser(sub)
    from doomformer import learn_cli as _learn
    _learn.add_parser(sub)
    # --- tune (doomformer/tune.py) ---
    from doomformer import tune as _tune_mod
    _tune_mod.add_arguments(sub.add_parser(
        "tune", help="measure where each timing constant of the one-map schedule starts to fail (every value checked "
                     "against the integer reference) and write doomformer/tuning.json, which --opt 2 and 3 build "
                     "with (hours); --check re-runs the committed values"))
    # --- end tune ---
    # --- lint (doomformer/schedule.py) ---
    li = sub.add_parser("lint", help="check the wave schedule of a compiled one-map brain against the engine model of "
                                     "doomformer/schedule.py: plane writes that can share a tic, timing margins below "
                                     "their measured minimum (exit 1), with a witness input per hazard")
    li.add_argument("wad", nargs="?", help="a compiled one-map brain: the .wad, with its .json next to it")
    li.add_argument("--fuzz", action="store_true", help="soundness campaign instead: random nets x opt 1, 2, 3 x random "
                    "timing constants, the checker's verdict against DOOM on boundary inputs (false negatives, "
                    "conservative rate; hours; options below)")
    li.add_argument("--net", help="its network (a shipped model name, a model directory or an MLP .npz), used to "
                                  "search an input for hazards whose gates read hidden values")
    li.add_argument("--witnesses", type=_at_least(0), default=8, help="hazards to find a witness input for (default 8)")
    li.add_argument("--json", action="store_true", help="print the report as JSON")
    from doomformer import lintfuzz as _lintfuzz
    _lintfuzz.add_arguments(li)
    # --- end lint ---
    if (sys.argv[1:] if argv is None else argv)[:1] == ["facts"]:     # facts has its own parser (facts.main)
        return _facts((sys.argv[1:] if argv is None else argv)[1:])
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        tiles = not (args.fast or args.one_map)
        if args.opt and (tiles or args.until_eos):
            raise SystemExit("demo: --opt sets the schedule of the one-map brain of --one-map/--fast (and --loop); "
                             "it does not apply to tile maps or --until-eos")
        if args.softmax:
            if args.fast or args.nightmare:
                raise SystemExit("demo: --softmax runs doomnames_softmax in tile maps, or doomnames_onemap_softmax with "
                                 "--one-map; leave out --fast/--nightmare")
            if tiles:
                args.model = "doomnames_softmax"
        if args.nightmare:
            if not tiles:
                raise SystemExit("demo: --nightmare runs the tile-map demo; leave out --one-map/--fast")
            args.model = "doombrain"
            args.temperature = 150.0 if args.temperature is None else args.temperature
            args.seed = 2 if args.seed is None else args.seed
        args.temperature = args.temperature if args.temperature is not None else (180.0 if tiles else 0.0)
        args.seed = args.seed if args.seed is not None else (20 if tiles else 0)
    for flag, parents in (("png", False), ("record", True)):
        if getattr(args, flag, None):
            _check_writable(getattr(args, flag), args.cmd, parents)
    import signal

    from doomformer.acc import AccError
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # `kill`/`timeout`: unwind, close every DOOM game
    try:
        return {"demo": _demo, "mnist": _mnist, "draw": _draw, "ask": _ask, "hello": _hello, "plan": _plan,
                "play": _play, "doctor": _doctor, "fuzz": _fuzz.run_cli, "verify": _verify.run_cli,
                "machine": _machine.run,
                "learn": _learn.run,
                "tune": _tune,
                "lint": _lint}[args.cmd](args)
    except (AccError, FileNotFoundError) as e:                    # expected setup problems: one message, no traceback
        print(f"doomformer {args.cmd}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
