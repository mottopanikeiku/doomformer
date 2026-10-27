"""A small looping GIF of a demo run: the game's own frame beside the compiled map with every voodoo doll a dot.

Built from `doomformer.record` pieces (the same top-down drawing of the compiled WAD, the same dot colours and the
same shared-palette GIF writer). It only reads what the run already read: the screen buffer and the doll positions.
No titles, no overlays; one caption line under the pictures.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from doomformer import record as R

MAX_BYTES = 3_000_000


class GifTrace:
    """Collects frames while a demo runs and writes one looping GIF of at most MAX_BYTES."""

    def __init__(self, wad: Path, path: str | Path, *, map_size=(480, 360), view_size=(360, 270), fps: int = 10,
                 colors: int = 96, max_frames: int = 400):
        self.path, self.fps, self.colors, self.max_frames = Path(path), fps, colors, max_frames
        self.map = R._Map(Path(wad), map_size, max_lines=1 << 62)
        self.size, self.view_size = map_size, view_size
        self.frames: list[Image.Image] = []
        self.durations: list[float] = []
        self._previous: np.ndarray | None = None
        self._font = ImageFont.load_default()

    def add(self, screen: np.ndarray | None, dolls, caption: str = "", hold: float = 0.0):
        """screen: ViZDoom screen buffer (3, H, W) or (H, W, 3) or None; dolls: (x, y) pairs in map units."""
        w, h = self.size
        panel = self.map.base.copy()
        pts = np.array([d[:2] for d in dolls if d[1] >= 0], dtype=float).reshape(-1, 2)
        if len(pts):
            px, py = self.map.px(pts[:, 0], pts[:, 1])
            moved = np.zeros(len(pts), bool)
            if self._previous is not None and len(self._previous) == len(pts):
                moved = np.abs(pts - self._previous).max(1) > R.MOVE_EPS
            self._previous = pts
            R._put(panel, px[~moved], py[~moved], R.STAGE_COLOURS[7], 3)
            R._put(panel, px[moved], py[moved], (255, 255, 255), 3)
        views = []
        if screen is not None:
            view = np.asarray(screen)
            if view.shape[0] == 3:
                view = np.transpose(view, (1, 2, 0))
            views.append(Image.fromarray(np.ascontiguousarray(view)).resize(self.view_size, Image.Resampling.NEAREST))
        height = max(h, self.view_size[1] if views else 0)
        canvas = Image.new("RGB", (w + (self.view_size[0] if views else 0), height + 16), R.BG)
        x0 = 0
        for view in views:
            canvas.paste(view, (0, 0))
            x0 = view.width
        canvas.paste(Image.fromarray(panel), (x0, 0))
        ImageDraw.Draw(canvas).text((4, height + 3), caption, fill=R.TEXT, font=self._font)
        self.frames.append(canvas)
        self.durations.append(1.0 / self.fps + hold)
        if len(self.frames) > self.max_frames:          # long run: keep every second frame, the same total time
            self.frames, self.durations = _thin(self.frames, self.durations)

    def close(self) -> tuple[Path, int]:
        """Write the GIF; thin the frames by half until it is under MAX_BYTES."""
        if not self.frames:
            raise ValueError("no frames were collected")
        frames, durations = self.frames, self.durations
        while True:
            R._write(self.path, frames, durations, self.fps, self.colors)
            size = self.path.stat().st_size
            if size <= MAX_BYTES or len(frames) <= 8:
                return self.path, size
            frames, durations = _thin(frames, durations)


def _thin(frames: list, durations: list) -> tuple[list, list]:
    """Every second frame (and the last), each lasting as long as itself and the frame dropped after it."""
    keep = list(range(0, len(frames), 2))
    out_d = [durations[i] + (durations[i + 1] if i + 1 < len(durations) else 0.0) for i in keep]
    out_f = [frames[i] for i in keep]
    if keep[-1] != len(frames) - 1:
        out_f.append(frames[-1])
        out_d.append(durations[-1])
    return out_f, out_d
