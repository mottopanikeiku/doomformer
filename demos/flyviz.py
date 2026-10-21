"""Spike raster for the fly demos: sensory input lines on top, the circuit's neurons below, outputs highlighted."""
from __future__ import annotations

import numpy as np
from PIL import ImageDraw

from .learnviz import FONTS
from doomformer.spiking import IntNet

GROUP_COLOURS = {"sugar": (90, 200, 255), "bitter": (120, 230, 60), "lc4": (255, 120, 220), "jo": (240, 240, 240),
                 "p9": (255, 160, 60)}


class Raster:
    def __init__(self, net: IntNet, box, T: int, outputs: list[str]):
        self.net, self.box, self.T = net, box, T
        self.groups = net.meta["input_groups"]
        self.out = {name: net.names.index(name) for name in outputs if name in net.names}
        self.hist_x: list[np.ndarray] = []
        self.hist_s: list[np.ndarray] = []

    def clear(self):
        self.hist_x, self.hist_s = [], []

    def push(self, x, s):
        self.hist_x = (self.hist_x + [x])[-self.T:]
        self.hist_s = (self.hist_s + [s])[-self.T:]

    def draw(self, d: ImageDraw.ImageDraw):
        x0, y0, x1, y1 = self.box
        f = FONTS
        cw = (x1 - x0 - 110) / self.T
        rows_in = len(self.net.inputs)
        h_in = min(1.5, 180 / rows_in)
        h_n = (y1 - y0 - rows_in * h_in - 30) / self.net.n
        left = x0 + 110
        d.text((x0, y0), "sensory\nneurons", font=f["tiny"], fill=(200, 200, 200))
        top_n = y0 + rows_in * h_in + 30
        d.text((x0, top_n), f"fly brain\n({self.net.n} neurons\nof FlyWire)", font=f["tiny"], fill=(200, 200, 200))
        for name, i in self.out.items():
            yy = top_n + i * h_n
            d.rectangle([left - 4, yy - 1, x1, yy + h_n + 1], outline=(255, 210, 0))
            d.text((left - 8, yy - 7), name, font=f["tiny"], fill=(255, 210, 0), anchor="ra")
        for t, (x, s) in enumerate(zip(self.hist_x, self.hist_s)):
            cx = left + t * cw
            for grp, cols in self.groups.items():
                colour = GROUP_COLOURS.get(grp, (200, 200, 200))
                for m in cols:
                    if x[m]:
                        d.rectangle([cx, y0 + m * h_in, cx + max(cw - 1, 1), y0 + (m + 1) * h_in], fill=colour)
            for i in np.nonzero(s)[0]:
                colour = (255, 210, 0) if i in self.out.values() else (255, 60, 60)
                d.rectangle([cx, top_n + i * h_n, cx + max(cw - 1, 1), top_n + (i + 1) * h_n], fill=colour)
