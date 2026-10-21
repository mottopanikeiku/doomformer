"""Film a verified DOOM run: `brain.record(inputs, "run.gif")`.

What is filmed, and by whom:

* DOOM runs the pass (or passes) exactly as `brain(x)` / `brain.step(x)` / `brain.logits(t)` / `brain.generate(p)`
  would, including the exact comparison with the independent integer reference. `record` calls those same methods.
* While a pass runs, `record` samples the game every `every` tics (a multiple of 8: that is how often
  `TileRunner.settle` looks at the dolls anyway, so the tics of the pass do not change). A sample only READS state:
  the positions of every voodoo doll from ViZDoom's object list (and, for the first-person view, the screen buffer).
  Nothing is written into the game while sampling.
* After a pass returned (the check passed) the frames of that pass are drawn with the input, the values read back
  from DOOM, and the word EXACT. If the check fails (`df.MismatchError`) the file is still written, with a red MISMATCH
  banner on the failing pass, and the error is re-raised. The EXACT label never appears on a pass that was not verified.
* Python draws the frames (Pillow) and writes the file (imageio). The picture is not DOOM's own rendering except
  for the first-person view, which is ViZDoom's screen buffer.

The top-down view is the same drawing as `record.topdown` (linedefs of the compiled WAD, lines with a special in
orange, shapes to scale, a very wide map cut into bands stacked top to bottom) with every voodoo doll drawn as a dot
coloured by its layer; white = moved since the previous frame.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIEWS = ("topdown", "firstperson", "both")
STAGE_COLOURS = [(255, 190, 40), (70, 170, 255), (255, 70, 190), (90, 230, 90),
                 (255, 110, 60), (180, 120, 255), (40, 230, 210), (255, 60, 60)]
TRAIL = (150, 170, 230)
BG, WALL, SPECIAL = (8, 10, 18), (70, 90, 140), (255, 160, 40)
TEXT, GREY, GREEN, RED = (235, 235, 235), (150, 150, 160), (90, 220, 100), (255, 80, 80)
MOVE_EPS = 0.5              # map units: dolls that wait still drift by ~0.1 (floor settling); walking is faster
MAX_DRAWN_LINES = 40_000    # more linedefs than this are drawn as a density image instead of one by one


@dataclass
class Recording:
    """What `brain.record` returns. `verified` is True only if every pass in the file passed the exact check."""
    path: Path
    frames: int
    bytes: int
    passes: int
    tics: int                       # game tics of the recorded passes (all passes together)
    checked: int                    # values compared with the integer reference (all passes together)
    verified: bool
    results: list = field(default_factory=list)   # what the brain call returned, one entry per call

    def __repr__(self):
        return (f"<Recording {self.path.name}: {self.frames} frames, {self.bytes / 1e6:.2f} MB, {self.passes} pass(es), "
                f"{self.tics} tics, {self.checked} values checked, verified={self.verified}>")


# --- static geometry ---------------------------------------------------------------------------------------------
def _textmap(wad: Path) -> str:
    data = Path(wad).read_bytes()
    n, dofs = struct.unpack("<ii", data[4:12])
    for i in range(n):
        o, sz, nm = struct.unpack("<ii8s", data[dofs + 16 * i: dofs + 16 * i + 16])
        if nm.rstrip(b"\0") == b"TEXTMAP":
            return data[o:o + sz].decode()
    raise ValueError(f"{wad}: no TEXTMAP lump")


def load_lines(wad: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linedefs of a compiled WAD as (a, b, special): endpoints (n, 2) and the special numbers. Only lines with
    both ends at y >= 0 (the brain; the player's room, the output lights and your world are below)."""
    tm = _textmap(wad)
    verts = np.array([(float(a), float(b)) for a, b in re.findall(r"vertex \{ x = (-?[\d.]+); y = (-?[\d.]+); \}", tm)])
    v1, v2, sp = [], [], []
    for body in re.findall(r"linedef \{([^}]*)\}", tm):
        v1.append(int(re.search(r"v1 = (\d+)", body).group(1)))
        v2.append(int(re.search(r"v2 = (\d+)", body).group(1)))
        s = re.search(r"special = (\d+)", body)
        sp.append(int(s.group(1)) if s else 0)
    a, b, sp = verts[v1], verts[v2], np.array(sp)
    keep = (a[:, 1] >= 0) & (b[:, 1] >= 0)
    if not keep.any():
        raise ValueError(f"{wad}: no lines at y >= 0")
    return a[keep], b[keep], sp[keep]


class _Map:
    """The brain's lines to scale in an (w, h) panel; a wide map is cut into bands stacked top to bottom."""

    def __init__(self, wad: Path, size: tuple[int, int], max_lines: int = MAX_DRAWN_LINES):
        self.max_lines = max_lines
        a, b, sp = load_lines(wad)
        self.w, self.h, pad = size[0], size[1], 4
        self.lo = np.minimum(a.min(0), b.min(0))
        hi = np.maximum(a.max(0), b.max(0))
        wm, hm = max(hi[0] - self.lo[0], 1.0), max(hi[1] - self.lo[1], 1.0)
        aw, ah = self.w - 2 * pad, self.h - 2 * pad
        best = (0.0, 1)
        for nb in range(1, 65):                                   # the band count that gives the biggest scale
            sc = min(aw * nb / wm, ah / (nb * hm + (nb - 1) * 0.15 * hm))
            best = max(best, (sc, nb))
        self.scale, self.nb = best
        self.band_w, self.band_h, self.pad, self.top = wm / self.nb, hm * self.scale, pad, hi[1]
        self.base = self._draw(a, b, sp)
        self.n_lines = len(sp)

    def px(self, x, y, band=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if band is None:
            band = np.clip(((x - self.lo[0]) // self.band_w).astype(int), 0, self.nb - 1)
        return ((x - self.lo[0] - band * self.band_w) * self.scale + self.pad,
                self.pad + band * self.band_h * 1.15 + (self.top - y) * self.scale)

    def _draw(self, a, b, sp) -> np.ndarray:
        img = Image.new("RGB", (self.w, self.h), BG)
        band = np.clip((((a[:, 0] + b[:, 0]) / 2 - self.lo[0]) // self.band_w).astype(int), 0, self.nb - 1)
        ax, ay = self.px(a[:, 0], a[:, 1], band)
        bx, by = self.px(b[:, 0], b[:, 1], band)
        if len(sp) <= self.max_lines:
            d = ImageDraw.Draw(img)
            for sel, colour in ((sp == 0, WALL), (sp != 0, SPECIAL)):
                for i in np.nonzero(sel)[0]:
                    d.line([(ax[i], ay[i]), (bx[i], by[i])], fill=colour, width=1)
            return np.asarray(img).copy()
        # a huge map: count line endpoints and midpoints per pixel, show log density (film_onemap's trick)
        dens, eff = np.zeros((self.h, self.w)), np.zeros((self.h, self.w))
        for px_, py_ in ((ax, ay), (bx, by), ((ax + bx) / 2, (ay + by) / 2)):
            ix = np.clip(px_.astype(int), 0, self.w - 1)
            iy = np.clip(py_.astype(int), 0, self.h - 1)
            for grid, sel in ((dens, sp == 0), (eff, sp != 0)):
                np.add.at(grid, (iy[sel], ix[sel]), 1)
        dens, eff = np.log1p(dens), np.log1p(eff)
        out = np.zeros((self.h, self.w, 3))
        out += (np.clip(dens / max(np.percentile(dens, 99.5), 1e-9), 0, 1) ** 0.6)[..., None] * np.array(WALL)
        out += (np.clip(eff / max(np.percentile(eff, 99.5), 1e-9), 0, 1) ** 0.6)[..., None] * np.array(SPECIAL) * 0.6
        return np.clip(out + np.array(BG), 0, 255).astype(np.uint8)


def topdown(wad: Path, size: tuple[int, int] = (900, 420), title: str = "") -> Image.Image:
    """A still picture of a compiled WAD from above (geometry only, no game needed): walls grey-blue, lines that carry
    a special (the weights: each one a line a voodoo doll crosses to add or subtract a number) orange, shapes to scale,
    a very wide map cut into bands stacked top to bottom. Every line is drawn, however many there are."""
    head = 22 if title else 0
    img = Image.new("RGB", size, BG)
    img.paste(Image.fromarray(_Map(Path(wad), (size[0], size[1] - head), max_lines=1 << 62).base), (0, head))
    if title:
        ImageDraw.Draw(img).text((10, 5), title, fill=(235, 235, 235))
    return img


def _put(canvas: np.ndarray, xs, ys, colour, size: int):
    """Square dots of side `size` centred on (xs, ys), straight into the canvas."""
    h, w = canvas.shape[:2]
    xs, ys = xs.astype(int) - size // 2, ys.astype(int) - size // 2
    for dx in range(size):
        for dy in range(size):
            px, py = xs + dx, ys + dy
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            canvas[py[ok], px[ok]] = colour


def _dim(c, f):
    return tuple(int(v * f) for v in c)


# --- sampling (reads state, writes nothing) ----------------------------------------------------------------------
@dataclass
class _Sample:
    tic: int
    xy: np.ndarray | None           # (n_dolls, 2) float32, voodoo dolls in object-id order (the real player excluded)
    screen: np.ndarray | None       # HxWx3 uint8 screen buffer


@dataclass
class _Pass:
    samples: list = field(default_factory=list)
    write: str = ""                 # what Python wrote into the map before the pass
    result: list = field(default_factory=list)   # lines read back from DOOM (shown only once verified)
    status: str = "running"         # running | exact | mismatch
    checked: int = 0
    note: str = ""                  # mismatch description
    tics: int = 0


class _Tap:
    """Samples every `every` tics of each pass by wrapping the runner's settle loop; restores it on exit."""

    def __init__(self, brain, every: int, want_xy: bool, want_screen: bool):
        self.brain, self.every, self.want_xy, self.want_screen = brain, every, want_xy, want_screen
        self.runner = brain.runner
        self.passes: list[_Pass] = []
        self.n_dolls = brain.meta.n_dolls

    def sample(self, used: int) -> _Sample:
        g = self.runner.game
        st = g.get_state()
        xy = screen = None
        if self.want_xy:
            objs = sorted((o for o in st.objects if o.name == "DoomPlayer"), key=lambda o: o.id)[:-1]   # last = the player
            xy = np.array([(o.position_x, o.position_y) for o in objs], dtype=np.float32).reshape(-1, 2)
        if self.want_screen:
            buf = np.asarray(st.screen_buffer)
            screen = np.transpose(buf, (1, 2, 0)) if buf.ndim == 3 and buf.shape[0] in (1, 3) else buf
            screen = np.ascontiguousarray(screen if screen.ndim == 3 else np.stack([screen] * 3, -1))
            if screen.shape[2] == 1:
                screen = np.repeat(screen, 3, 2)
        return _Sample(used, xy, screen)

    def __enter__(self):
        runner, orig, tap = self.runner, self.runner.settle, self

        def settle(check_every: int = 8, max_tics: int | None = None, frames=None, on_tick=None):
            p = _Pass()
            tap.passes.append(p)
            p.samples.append(tap.sample(0))

            def tick(used):
                if used % tap.every == 0:
                    p.samples.append(tap.sample(used))
                if on_tick is not None:
                    on_tick(used)
            used = orig(check_every=check_every, max_tics=max_tics, frames=frames, on_tick=tick)
            if p.samples[-1].tic != used:
                p.samples.append(tap.sample(used))
            p.tics = used
            return used
        runner.settle = settle                         # instance attribute shadows the method; removed in __exit__
        return self

    def __exit__(self, *exc):
        self.runner.__dict__.pop("settle", None)


# --- text helpers ------------------------------------------------------------------------------------------------
def _vec(a, limit: int = 12) -> str:
    a = np.asarray(a).reshape(-1).tolist()
    return str(a) if len(a) <= limit else "[" + ", ".join(map(str, a[:limit])) + f", ... ] (n={len(a)})"


def _stage_info(brain) -> list[tuple[str, int]]:
    """[(label, n_voodoo_dolls incl. the layer's clock doll)] per layer, from the compile-time layout."""
    stats = (brain.meta.layout or {}).get("stage_stats") or []
    out = []
    for i, s in enumerate(stats):
        name = str(s.get("name", f"layer {i}"))
        mm = re.fullmatch(r"mlp_l(\d+)", name)
        out.append((f"layer {int(mm.group(1)) + 1}" if mm else name, int(s["dolls"]) + 1))
    return out


def _fit(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "...", font=font) > width:
        text = text[:-1]
    return text + "..."


# --- frame composer ----------------------------------------------------------------------------------------------
class _Composer:
    def __init__(self, brain, view: str, width: int, screen_size: tuple[int, int] | None, title: str):
        self.view, self.brain, self.title = view, brain, title
        fs = max(11, width // 48)
        self.font = ImageFont.load_default(size=fs)
        self.fs = fs
        self.stages = _stage_info(brain)
        if sum(n for _, n in self.stages) != brain.meta.n_dolls:  # the layout does not account for every doll: one colour
            self.stages = []
        n = brain.meta.n_dolls                                     # fewer dolls -> bigger dots
        self.dot = max(2, width // (60 if n <= 40 else 120 if n <= 400 else 300))
        sw, sh = screen_size or (4, 3)
        if view == "firstperson":
            self.pw_t, self.pw_f, self.ph = 0, width, round(width * sh / sw)
        elif view == "topdown":
            self.pw_t, self.pw_f, self.ph = width, 0, round(width / 2)
        else:                                                      # both: first-person gets the same height as the map
            self.ph = round(width * 0.3)
            self.pw_f = round(self.ph * sw / sh)
            self.pw_t = width - self.pw_f
        self.map = _Map(Path(brain.wad), (self.pw_t, self.ph)) if self.pw_t else None
        probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
        self.legend_rows = self._legend_layout(probe, width)
        self.head_h = fs + 8 + len(self.legend_rows) * (fs + 3)
        self.foot_h = 4 * (fs + 4) + 6
        self.W = width + width % 2
        self.H = self.head_h + self.ph + self.foot_h
        self.H += self.H % 2
        # stage index of each doll (object-id order): per layer its lane dolls, then its clock doll
        self.stage_of = np.zeros(0, int)
        if self.stages:
            self.stage_of = np.concatenate([np.full(n, i) for i, (_, n) in enumerate(self.stages)])

    def _legend_layout(self, draw, width):
        rows, row, x = [], [], 4
        items = [(STAGE_COLOURS[i % len(STAGE_COLOURS)], n) for i, (n, _) in enumerate(self.stages)] \
            if self.map else []
        for colour, name in items:
            w = int(draw.textlength(name, font=self.font)) + self.fs + 12
            if row and x + w > width:
                rows.append(row)
                row, x = [], 4
            row.append((x, colour, name))
            x += w
        if row:
            rows.append(row)
        return rows

    # -- panels -----------------------------------------------------------------------------------------------------
    def _march(self, p: _Pass):
        """Per sample: which dolls are moving (moved since the previous sample) and which have ever left home."""
        out, home, prev, ever = [], None, None, None
        for s in p.samples:
            n = len(s.xy)
            if home is None:                                  # the first sample is the dolls at home
                home = prev = last = s.xy
                ever = np.zeros(n, bool)
                moved = np.zeros(n, bool)
            else:
                last = prev
                moved = np.abs(s.xy - prev).sum(1) > MOVE_EPS
                ever = ever | (np.abs(s.xy - home).sum(1) > MOVE_EPS)
                prev = s.xy
            out.append((moved, ever.copy(), last))
        return out

    def _map_panel(self, s: _Sample, moved: np.ndarray, ever: np.ndarray, before: np.ndarray) -> np.ndarray:
        canvas = self.map.base.copy()
        sx, sy = self.map.px(s.xy[:, 0], s.xy[:, 1])
        inside = (s.xy[:, 1] >= 0)
        stage = self.stage_of if len(self.stage_of) == len(s.xy) else np.zeros(len(s.xy), int)
        for i in np.unique(stage):
            c = STAGE_COLOURS[int(i) % len(STAGE_COLOURS)]
            m = (stage == i) & inside
            for sel, col, size in ((m & ~ever & ~moved, _dim(c, 0.6), self.dot), (m & ever & ~moved, c, self.dot + 1)):
                _put(canvas, sx[sel], sy[sel], col, size)
        sel = moved & inside
        if sel.any() and sel.sum() <= 4000:                       # a short tail from where the doll was one frame ago
            bx, by = self.map.px(before[sel, 0], before[sel, 1])
            im = Image.fromarray(canvas)
            d = ImageDraw.Draw(im)
            for x0, y0, x1, y1 in zip(bx, by, sx[sel], sy[sel]):
                if abs(x1 - x0) < self.map.w / 2 and abs(y1 - y0) < self.map.band_h:     # not across a band wrap
                    d.line([(x0, y0), (x1, y1)], fill=TRAIL, width=max(1, self.dot // 2))
            canvas = np.asarray(im).copy()
        _put(canvas, sx[sel], sy[sel], (255, 255, 255), self.dot + 2)
        return canvas

    def _fp_panel(self, s: _Sample) -> np.ndarray:
        return np.asarray(Image.fromarray(s.screen).resize((self.pw_f, self.ph), Image.BILINEAR))

    # -- one frame --------------------------------------------------------------------------------------------------
    def frame(self, p: _Pass, k: int, n_passes: int, idx: int, state, final: bool) -> Image.Image:
        s = p.samples[idx]
        im = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(im)
        f, fs = self.font, self.fs
        d.text((4, 3), _fit(d, self.title, f, self.W - 150), font=f, fill=TEXT)
        right = f"pass {k + 1}/{n_passes}   tic {s.tic}"
        d.text((self.W - 4 - d.textlength(right, font=f), 3), right, font=f, fill=GREY)
        for r, row in enumerate(self.legend_rows):
            y = fs + 8 + r * (fs + 3)
            for x, colour, name in row:
                d.rectangle([x, y + 2, x + fs - 4, y + fs - 2], fill=colour)
                d.text((x + fs, y), name, font=f, fill=GREY)
        x0 = 0
        if self.map:
            moved, ever, before = state
            panel = self._map_panel(s, moved, ever, before)
            im.paste(Image.fromarray(panel), (0, self.head_h))
            x0 = self.pw_t
        if self.pw_f:
            im.paste(Image.fromarray(self._fp_panel(s)), (x0, self.head_h))
        y = self.head_h + self.ph + 4
        lines = [(p.write, TEXT)]
        if final and p.status == "exact":
            lines += [(r, TEXT) for r in p.result[:2]]
            lines.append((f"EXACT: {p.checked} values read back from DOOM == the integer reference (tolerance 0)", GREEN))
        elif final and p.status == "mismatch":
            lines += [(p.note, RED), ("MISMATCH: this pass did NOT match the integer reference", RED)]
        else:
            lines += [(f"DOOM is computing: tic {s.tic} (dolls walk, lines fire; white = moving)" if s.tic else
                       "dolls at home; the run is about to start", GREY), ("result: pending (read back and checked after the run)", GREY)]
        for text, colour in lines[:4]:
            d.text((4, y), _fit(d, text, f, self.W - 8), font=f, fill=colour)
            y += fs + 4
        return im


# --- driving the brain -------------------------------------------------------------------------------------------
def _drive(brain, inputs, gen: dict, tap: _Tap) -> list:
    """Run the brain's own verified entry point(s) on `inputs`; label the passes they create. Returns the results."""
    import doomformer.kit as kit
    kind = brain.kind
    results: list = []

    def item(write: str, call, describe, keep: bool = True):
        n0 = len(tap.passes)
        try:
            res = call()
        except kit.MismatchError as e:
            if len(tap.passes) > n0:                           # the pass ran; only its comparison failed
                last = tap.passes[-1]
                last.write, last.status = write, "mismatch"
                bad = e.mismatches[0] if getattr(e, "mismatches", None) else None
                last.note = (f"{len(e.mismatches)} value(s) differ, e.g. {bad[0]}: DOOM {bad[1]}, reference {bad[2]}"
                             if bad else str(e))
            raise
        passes = tap.passes[n0:]
        if not passes:
            raise RuntimeError("the brain call did not run a pass in DOOM; nothing to record")
        for p in passes:
            p.write, p.status = write, "exact"
            p.checked = int(brain.info["checked"] or 0)
        passes[-1].result = describe(res)
        if keep:
            results.append(res)
        return res

    if kind == "mlp":
        a = np.asarray(inputs)
        for x in (a[None] if a.ndim == 1 else list(a)):
            item(f"Python writes x = {_vec(x)} into the {brain.net.n_in} input ports",
                 lambda x=x: brain(x), lambda y: [f"DOOM readback  y = {_vec(y)}"])
    elif kind == "spiking":
        a = np.asarray(inputs)
        steps = a[None] if a.ndim == 1 else list(a)
        for t, x in enumerate(steps):
            item(f"step {t + 1}/{len(steps)}: Python writes input spikes {_vec(x)}",
                 lambda x=x: brain.step(x),
                 lambda vs: [f"DOOM readback  spikes = {_vec(vs[1])}   v = {_vec(vs[0])}"])
    elif kind == "transformer":
        m = brain.net
        if isinstance(inputs, str):
            orig = brain.logits

            def logits(tokens):
                ctx = "$" + "".join(m.vocab[t] for t in tokens[1:])
                return item(f"Python writes token ids {list(map(int, tokens))} ({ctx!r})", lambda: orig(tokens),
                            lambda lg: [_top_logits(m, lg[-1])], keep=False)
            brain.logits = logits                              # generate() calls self.logits: each step is recorded
            try:
                results.append(brain.generate(inputs, **gen))
            finally:
                brain.__dict__.pop("logits", None)
            if tap.passes:
                tap.passes[-1].result.append(f"generated {results[-1]!r} (Python picks each letter from DOOM's logits)")
        else:
            toks = [int(t) for t in inputs]
            ctx = "$" + "".join(m.vocab[t] for t in toks[1:])
            item(f"Python writes token ids {toks} ({ctx!r})", lambda: brain.logits(toks), lambda lg: [_top_logits(m, lg[-1])])
    else:
        raise TypeError(f"brain.record does not know how to drive a {kind!r} brain")
    return results


def _top_logits(m, row) -> str:
    order = np.argsort(-np.asarray(row), kind="stable")[:4]
    return "DOOM logits (last position), top: " + "  ".join(f"{m.vocab[i] if i else '$'}={int(row[i])}" for i in order)


# --- writing -----------------------------------------------------------------------------------------------------
def _palette_of(frames: list[Image.Image], colors: int) -> Image.Image:
    """One shared palette for the whole file (smaller, no flicker), from a sample of the frames."""
    step = max(1, len(frames) // 12)
    pick = [frames[i] for i in range(0, len(frames), step)] + [frames[-1]]
    sheet = Image.new("RGB", (pick[0].width, pick[0].height * len(pick)))
    for i, fr in enumerate(pick):
        sheet.paste(fr, (0, i * fr.height))
    return sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def _write(path: Path, frames: list[Image.Image], durations: list[float], fps: int, colors: int):
    import imageio.v2 as imageio
    suffix = path.suffix.lower()
    if suffix == ".gif":
        pal = _palette_of(frames, colors)
        arrays = [np.asarray(fr.quantize(palette=pal, dither=Image.Dither.NONE).convert("RGB")) for fr in frames]
        with imageio.get_writer(path, mode="I", duration=[max(0.02, d) for d in durations], loop=0,
                                palettesize=colors, subrectangles=True) as w:
            for a in arrays:
                w.append_data(a)
    elif suffix == ".mp4":
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=2,
                                ffmpeg_params=["-crf", "24", "-preset", "medium",
                                               "-movflags", "+faststart", "-threads", "1"], pixelformat="yuv420p") as w:
            for fr, dur in zip(frames, durations):
                a = np.asarray(fr)
                for _ in range(max(1, round(dur * fps))):
                    w.append_data(a)
    else:
        raise ValueError(f"record writes .gif or .mp4, not {path.suffix!r}")


# --- the public function -----------------------------------------------------------------------------------------
def record(brain, inputs, path, view: str = "topdown", every: int = 8, fps: int = 30, *, width: int = 640,
           colors: int | None = None, hold: float = 1.5, camera_script: int | None = None, title: str | None = None,
           **gen) -> Recording:
    """Run `inputs` through `brain` (one verified pass per input, as the brain's own call does) and film it.

    inputs: MLP: one integer vector or a list/2-D array of them (one pass each); spiking: one spike vector or a
        (steps, n_in) array (one DOOM step each, state carries over); transformer: a prompt `str` (generate; pass
        temperature=, seed=, max_new= as keywords) or a list of token ids (one forward pass).
    path: `.gif` (shared palette, only changed regions stored) or `.mp4` (H.264, needs imageio-ffmpeg).
    view: 'topdown' (the compiled map's lines, every voodoo doll a dot), 'firstperson' (ViZDoom's screen buffer from
        the real player's eyes) or 'both' side by side.
    every: DOOM tics between frames; a positive multiple of 8. fps: playback frame rate of the file.
    width: frame width in pixels (smaller = smaller file). colors: GIF palette size (default 64 top-down, 128 with
        first-person). hold: seconds each finished pass stays on screen with its result.
    camera_script: number (>= 200) of an ACS script in YOUR world that points the player's view at a camera thing
        (ChangeCamera); record runs it with argument 1 before the first pass and 0 after the last one. For
        first-person only; without it you see what the player sees.

    Returns a `Recording`. Recording only reads game state, so the results equal an unrecorded run; a failed check
    still writes the file (red MISMATCH banner) and then raises df.MismatchError."""
    from .kit import MismatchError
    view = view.lower()
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
    if not isinstance(every, (int, np.integer)) or every < 8 or every % 8:
        raise ValueError(f"every must be a positive multiple of 8 DOOM tics (the dolls are inspected every 8), got {every!r}")
    if fps < 1 or width < 160 or hold < 0:
        raise ValueError("fps >= 1, width >= 160 and hold >= 0 are required")
    path = Path(path)
    if path.suffix.lower() not in (".gif", ".mp4"):
        raise ValueError(f"record writes .gif or .mp4, not {path.suffix!r}")
    if camera_script is not None and view == "topdown":
        raise ValueError("camera_script changes the first-person view; use view='firstperson' or 'both'")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runner = brain.runner                                   # starts the game if it is not running yet
    except TypeError as e:                                      # tile-mode brains have many games, no single runner
        raise TypeError("brain.record films one game: compile with mode='map' (tile mode runs one game per tile)") from e
    if view != "topdown":
        sw, sh = runner.game.get_screen_width(), runner.game.get_screen_height()
    else:
        sw = sh = None
    err = None
    results: list = []
    with _Tap(brain, every, want_xy=view != "firstperson", want_screen=view != "topdown") as tap:
        try:
            if camera_script is not None:
                brain.run_script(camera_script, 1)
            try:
                results = _drive(brain, inputs, gen, tap)
            except MismatchError as e:
                err = e
        finally:
            if camera_script is not None:
                brain.run_script(camera_script, 0)
    passes = tap.passes
    if not passes:
        raise RuntimeError("nothing was recorded")
    if any(p.status == "running" for p in passes):              # a pass that raised something else than a mismatch
        raise RuntimeError("a recorded pass did not complete; no file written")
    ncol = colors or (64 if view == "topdown" else 128)
    comp = _Composer(brain, view, width, (sw, sh) if sw else None,
                     title or f"doomformer: {brain.kind}, {brain.meta.n_dolls} voodoo dolls, {brain.meta.n_sectors} sectors")
    frames: list[Image.Image] = []
    durations: list[float] = []
    dt = 1.0 / fps
    for k, p in enumerate(passes):
        state = comp._march(p) if comp.map else [None] * len(p.samples)
        for i in range(len(p.samples)):
            frames.append(comp.frame(p, k, len(passes), i, state[i], final=False))
            durations.append(max(dt * 8, 0.4) if i == 0 else dt)
        last = len(p.samples) - 1
        frames.append(comp.frame(p, k, len(passes), last, state[last], final=True))
        durations.append(max(hold, dt))
    _write(path, frames, durations, fps, ncol)
    rec = Recording(path=path, frames=len(frames), bytes=path.stat().st_size, passes=len(passes),
                    tics=sum(p.tics for p in passes), checked=sum(p.checked for p in passes if p.status == "exact"),
                    verified=err is None and all(p.status == "exact" for p in passes), results=results)
    if err is not None:
        raise err
    return rec
