"""Drawing for the DOOM-learns demos: the marine's view, DOOM's memory (the weight ceilings), the learning map."""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from doomformer import policy

from . import aimlearn as A
from doomformer.pure import CEIL_LOWER, CEIL_RAISE, FLOOR_LOWER, FLOOR_RAISE

W, H = 1920, 1080
GAME_BOX = (40, 150, 1000, 870)
NAMES = {"L": "TURN LEFT", "R": "TURN RIGHT", "F": "FIRE"}
# display columns left-to-right as the marine sees them: far left ... dead ahead ... far right, none, bias
ORDER = list(range(len(policy.BEARING_CHARS) - 1, -1, -1)) + [A.N_FEAT - 2, A.N_FEAT - 1]


def feat_label(i: int) -> str:
    if i == A.N_FEAT - 1:
        return "bias"
    if i == A.N_FEAT - 2:
        return "no demon"
    lo, hi = ([-180] + policy.BEARING_EDGES)[i], (policy.BEARING_EDGES + [180])[i]
    if lo == -2:
        return "dead ahead"
    return f"{lo}-{hi} deg left" if lo >= 0 else f"{-hi}-{-lo} deg right"


class Board:
    """The learner map drawn as a circuit board: its linedefs, weight/score lines coloured, dolls on top.
    Each layer strip (separated by gaps in x) gets its own row, scaled to the full width."""

    def __init__(self, wad: Path, box):
        data = wad.read_bytes()
        n, dofs = struct.unpack("<ii", data[4:12])
        for i in range(n):
            o, sz, nm = struct.unpack("<ii8s", data[dofs + 16 * i: dofs + 16 * i + 16])
            if nm.rstrip(b"\0") == b"TEXTMAP":
                tm = data[o:o + sz].decode()
        verts = [(float(a), float(b)) for a, b in re.findall(r"vertex \{ x = (-?[\d.]+); y = (-?[\d.]+); \}", tm)]
        lines = []
        for body in re.findall(r"linedef \{([^}]*)\}", tm):
            v1, v2 = int(re.search(r"v1 = (\d+)", body).group(1)), int(re.search(r"v2 = (\d+)", body).group(1))
            sp = re.search(r"special = (\d+)", body)
            lines.append((verts[v1], verts[v2], int(sp.group(1)) if sp else 0))
        lines = [ln for ln in lines if ln[0][1] >= 0 and ln[1][1] >= 0]
        spans = sorted((min(a[0], b[0]), max(a[0], b[0])) for a, b, _ in lines)
        self.segs: list[list[float]] = []
        for lo, hi in spans:                               # merge x coverage into layer strips
            if self.segs and lo <= self.segs[-1][1] + 100:
                self.segs[-1][1] = max(self.segs[-1][1], hi)
            else:
                self.segs.append([lo, hi])
        ys = [p[1] for ln in lines for p in ln[:2]]
        self.y0, self.y1 = min(ys), max(ys)
        self.box = box
        bx0, by0, bx1, by1 = box
        self.row_h = (by1 - by0) / len(self.segs)
        im = Image.new("RGB", (bx1 - bx0, by1 - by0), (8, 8, 12))
        d = ImageDraw.Draw(im)
        colour = {CEIL_RAISE: (60, 230, 90), CEIL_LOWER: (240, 60, 60), FLOOR_RAISE: (255, 150, 30),
                  FLOOR_LOWER: (255, 150, 30)}
        for special in (0, 1):
            for a, b, sp in lines:
                c = colour.get(sp)
                if (c is not None) != bool(special):
                    continue
                d.line([self.pt(*a, local=True), self.pt(*b, local=True)], fill=c or (46, 46, 60), width=1)
        for k in range(1, len(self.segs)):
            d.line([(0, k * self.row_h), (bx1 - bx0, k * self.row_h)], fill=(120, 120, 120), width=1)
        self.bg = np.asarray(im)

    def pt(self, x, y, local=False):
        k = next((i for i, (lo, hi) in enumerate(self.segs) if lo - 1 <= x <= hi + 1), None)
        if k is None:
            return None
        lo, hi = self.segs[k]
        px = (x - lo) / (hi - lo) * (self.box[2] - self.box[0] - 1)
        py = k * self.row_h + 2 + (self.y1 - y) / (self.y1 - self.y0) * (self.row_h - 4)
        return (px, py) if local else (px + self.box[0], py + self.box[1])

    def draw(self, canvas: np.ndarray, dolls):
        bx0, by0, bx1, by1 = self.box
        canvas[by0:by1, bx0:bx1] = self.bg
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        for x, y in dolls:
            p = self.pt(x, y) if self.y0 <= y <= self.y1 else None
            if p is not None:
                d.ellipse([p[0] - 1.5, p[1] - 1.5, p[0] + 1.5, p[1] + 1.5], fill=(255, 40, 40))
        canvas[:] = np.asarray(im)


def draw_memory(d: ImageDraw.ImageDraw, w: np.ndarray, changed: np.ndarray, x: np.ndarray, fonts, K: int):
    small, tiny = fonts
    left, top, colw, half = 1060, 170, 40, 70
    for h, title in enumerate(["FIRE?  (head 1)", "TURN LEFT (+) / RIGHT (-)  (head 2)"]):
        base = top + h * 250 + 30 + half
        d.text((left, top + h * 250), title, font=small, fill=(235, 235, 235))
        d.line([(left, base), (left + colw * len(ORDER), base)], fill=(90, 90, 90))
        for c, i in enumerate(ORDER):
            v = int(w[h, i])
            cx = left + c * colw
            if x[i]:
                d.rectangle([cx + 2, base - half - 6, cx + colw - 2, base + half + 6], outline=(255, 210, 0), width=2)
            hpx = int(round(np.clip(v, -K, K) / K * (half - 4)))
            fill = (70, 220, 90) if v > 0 else (235, 70, 60)
            if v:
                y0, y1 = sorted((base, base - hpx))
                d.rectangle([cx + 8, y0, cx + colw - 8, y1], fill=fill,
                            outline=(255, 255, 255) if changed[h, i] else None, width=3 if changed[h, i] else 0)
            d.text((cx + colw // 2, base + half + 14), str(v), font=tiny, fill=(200, 200, 200), anchor="mm")
    y = top + 2 * 250 + 8
    d.text((left, y), "<- demon far left", font=tiny, fill=(160, 160, 160))
    d.text((left + colw * 8.5, y), "dead ahead", font=tiny, fill=(160, 160, 160), anchor="ma")
    d.text((left + colw * 17 - 6, y), "far right ->", font=tiny, fill=(160, 160, 160), anchor="ra")
    d.text((left + colw * 17.5, y + 22), "none", font=tiny, fill=(160, 160, 160), anchor="ma")
    d.text((left + colw * 18.5, y + 22), "bias", font=tiny, fill=(160, 160, 160), anchor="ma")


FONTS = {k: ImageFont.load_default(size=s) for k, s in dict(big=44, mid=26, small=22, tiny=16).items()}


def lesson_text(w_new: np.ndarray, changed: np.ndarray) -> str:
    if not changed.any():
        return "no lesson needed"
    return "lesson: dolls move ceilings: " + ", ".join(
        f"{'fire' if h == 0 else 'turn'}[{feat_label(i)}] -> {int(w_new[h, i])}" for h, i in zip(*np.nonzero(changed)))


def compose(view: np.ndarray, board: Board, dolls, w: np.ndarray, changed: np.ndarray, x: np.ndarray, *,
            title: str, subtitle: str, obs: str, a: int, e: int, s: np.ndarray, footer: str, ok: bool,
            board_label: str = "the learning map (dolls = red dots, green/red = weight lines)") -> np.ndarray:
    """One 1920x1080 frame: the marine's view, what he saw and chose, the aimbot's lesson, DOOM's memory, the map."""
    f = FONTS
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[GAME_BOX[1]:GAME_BOX[3], GAME_BOX[0]:GAME_BOX[2]] = np.asarray(
        Image.fromarray(view).resize((960, 720), Image.NEAREST))
    board.draw(canvas, dolls)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.text((40, 30), title, font=f["big"], fill=(255, 64, 64))
    d.text((40, 88), subtitle, font=f["mid"], fill=(225, 225, 225))
    d.text((1060, 100), f"DOOM's memory: {A.HEADS * A.N_FEAT} ceilings (read back every decision)", font=f["small"],
           fill=(255, 176, 32))
    draw_memory(d, w, changed, x, (f["small"], f["tiny"]), A.K)
    d.text((1060, 770), board_label, font=f["tiny"], fill=(160, 160, 160))
    d.text((40, 885), f"sees: {obs}", font=f["mid"], fill=(200, 200, 200))
    d.text((40, 925), f"DOOM decides: {NAMES[policy.ACTIONS[a]]}   (fire {s[0]:+d}, turn {s[1]:+d})", font=f["mid"],
           fill=(255, 64, 64))
    d.text((40, 962), f"aimbot says: {NAMES[policy.ACTIONS[e]]}  {'(agrees)' if a == e else '(disagrees)'}",
           font=f["mid"], fill=(90, 200, 90) if a == e else (255, 176, 32))
    d.text((40, 999), lesson_text(w, changed), font=f["small"], fill=(200, 200, 200))
    d.text((40, H - 40), footer, font=f["small"], fill=(90, 200, 90) if ok else (255, 0, 0))
    return np.asarray(im)
