"""First-person cinematography for compute tiles (ViZDoom screen buffer, not the automap).

`compile_tile(spec, wad, decorate=fn)` hands `fn(builder, layout)` the map under construction. The decorators
in this module only (a) add a camera actor and an ACS script that places it, (b) restyle cosmetic fields
(textures, light) of NON-output rects, (c) add isolated decoration rects. They never touch heights, tags,
specials, links or output sectors, so the computation is unchanged. Replaying the logged inputs on the
decorated WAD reproduces the outputs bit for bit.

Camera: a MapSpot actor (tid CAM_TID) shown with ChangeCamera. ACS script 200 moves it; the host sends
one `puke -200 x4 y4 z4 ang` per tic (4 args is the ACS limit, so yaw and pitch share one argument).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vizdoom as vzd

from doomformer.runtime import TileRunner
from doomformer.tile import CONV_TAG, DOOR_TAG0, OUT_TAG0, PORT_TAG0

CAM_TID = 9000
CAM_SCRIPT = 200
OFS = 32768                      # puke args are offset to stay non-negative
W, H = 1280, 720
CAM_ACS = f"""
script {CAM_SCRIPT} (int x4, int y4, int z4, int ang) {{
  int yaw10 = ang / 2048;
  int pitch10 = (ang % 2048) - 1024;
  Thing_Remove({CAM_TID});
  SpawnForced("MapSpot", (x4 - {OFS}) << 14, (y4 - {OFS}) << 14, (z4 - {OFS}) << 14, {CAM_TID}, 0);
  SetActorAngle({CAM_TID}, yaw10 * 65536 / 3600);
  SetActorPitch({CAM_TID}, pitch10 * 65536 / 3600);
  ChangeCamera({CAM_TID}, 1, 0);
}}
script {CAM_SCRIPT + 1} OPEN {{ ChangeCamera({CAM_TID}, 1, 0); }}
"""


def add_camera(b) -> str:
    """Adds the camera actor (parked in the player's closet) and returns its ACS."""
    b.thing(-256, -128, 9001, id=CAM_TID)
    return CAM_ACS


# --------------------------------------------------------------------------------------------- terrain
@dataclass
class Terrain:
    """Static snapshot of the map's rects, for placing cameras (floors/ceilings as compiled)."""
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    floor: np.ndarray
    ceil: np.ndarray
    tag: np.ndarray

    @classmethod
    def of(cls, b) -> "Terrain":
        a = lambda f: np.array([getattr(r, f) for r in b.rects])
        return cls(a("x0"), a("y0"), a("x1"), a("y1"), a("floor"), a("ceil"), a("tag"))

    def at(self, x: float, y: float) -> int:
        m = (self.x0 <= x) & (x < self.x1) & (self.y0 <= y) & (y < self.y1)
        i = np.flatnonzero(m)
        return int(i[0]) if len(i) else -1

    def floor_at(self, x, y, default=0.0):
        i = self.at(x, y)
        return float(self.floor[i]) if i >= 0 else default

    def ceil_at(self, x, y, default=1e9):
        i = self.at(x, y)
        return float(self.ceil[i]) if i >= 0 else default

    def gates(self, zone, lo=PORT_TAG0, hi=OUT_TAG0):
        """Port-gate rects inside a block zone [x0,y0,x1,y1], ordered by x: (x0, x1, floor, tag)."""
        zx0, zy0, zx1, zy1 = zone
        m = (self.tag >= lo) & (self.tag < hi) & (self.x0 >= zx0) & (self.x1 <= zx1) & (self.y0 >= zy0) & (self.y1 <= zy1)
        idx = np.flatnonzero(m)
        idx = idx[np.argsort(self.x0[idx])]
        return [(int(self.x0[i]), int(self.x1[i]), int(self.floor[i]), int(self.tag[i])) for i in idx]

    def floor_profile(self, y, xs, smooth=0.0):
        """Floor height along a line y = const; `smooth` (units) low-passes the staircase."""
        f = np.array([self.floor_at(x, y) for x in xs])
        if smooth > 0 and len(xs) > 2:
            step = float(xs[1] - xs[0])
            k = max(1, int(round(smooth / step)))
            ker = np.ones(2 * k + 1) / (2 * k + 1)
            pad = np.pad(f, k, mode="edge")
            f = np.convolve(pad, ker, mode="valid")
        return f


# ------------------------------------------------------------------------------------------ cosmetics
def restyle(b, *, zone_floor=None, zone_wall=None, zone_ceil=None, zone_light=None, lane_floor=None,
            lane_wall=None, gate_wall=None, gate_floor=None, gate_ceil=None, gate_light=None, fx_floor=None,
            fx_light=None, door_wall=None, step_light=None, out_floor=None, out_ceil=None, out_wall=None, fx_alt_light=None, gate_face=None):
    """Cosmetic restyle by role. Only ftex/ctex/wall/light are written; output accumulators get textures only
    (their light level IS the accumulator and is never touched)."""
    if gate_face:
        # the wall a doll (or the camera) faces when it walks up to a gate comes from the sidedef of the rect
        # in front of it: give those rects a distinctive texture so closed gates read as doors
        gates = [g for g in b.rects if PORT_TAG0 <= g.tag < OUT_TAG0]
        by_x1, by_x0 = {}, {}
        for r in b.rects:
            if r.tag == CONV_TAG:
                by_x1.setdefault((r.x1, r.y0, r.y1), []).append(r)
                by_x0.setdefault((r.x0, r.y0, r.y1), []).append(r)
        for g in gates:
            for r in by_x1.get((g.x0, g.y0, g.y1), []) + by_x0.get((g.x1, g.y0, g.y1), []):
                r.wall = gate_face
    for r in b.rects:
        if OUT_TAG0 <= r.tag < OUT_TAG0 + 10000:
            _set(r, out_floor, out_ceil, out_wall, None)
            continue
        is_gate = PORT_TAG0 <= r.tag < OUT_TAG0
        is_door = r.tag >= DOOR_TAG0
        is_fx = r.ftex == "FLAT1_1" and r.light == 200
        lane = r.tag == CONV_TAG and r.light in (192, 128)
        if is_gate:
            _set(r, gate_floor, gate_ceil, gate_wall, gate_light)
        elif is_door:
            _set(r, zone_floor, zone_ceil, door_wall, None)
        elif is_fx:
            odd = fx_alt_light is not None and (r.x0 // 4) % 2
            _set(r, fx_floor, zone_ceil, zone_wall, fx_alt_light if odd else fx_light)
        elif lane:
            _set(r, lane_floor, zone_ceil, lane_wall, None)
        elif r.tag == CONV_TAG:
            _set(r, zone_floor, zone_ceil, None if gate_face and r.wall == gate_face else zone_wall,
                 step_light if step_light is not None else zone_light)


def _set(r, ftex, ctex, wall, light):
    if ftex:
        r.ftex = ftex
    if ctex:
        r.ctex = ctex
    if wall:
        r.wall = wall
    if light is not None:
        r.light = light


def add_room(b, x0, y0, x1, y1, floor, ceil, *, light=200, ftex="FLOOR0_1", ctex="F_SKY1", wall="STARTAN3") -> int:
    """Isolated decoration rect (must not share an edge with any existing rect)."""
    return b.rect(x0, y0, x1, y1, floor=floor, ceil=ceil, light=light, ftex=ftex, ctex=ctex, wall=wall)


# ---------------------------------------------------------------------------------------------- paths
def smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _catmull(p0, p1, p2, p3, u):
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)


@dataclass
class Keys:
    """Camera keyframes (tic, x, y, z, yaw_deg, pitch_deg_down). Catmull-Rom through the points, yaw unwrapped,
    ease in/out per segment when `ease`."""
    k: list
    ease: bool = False

    def __post_init__(self):
        a = np.array(self.k, dtype=float)
        a[:, 4] = np.degrees(np.unwrap(np.radians(a[:, 4])))
        self.a = a

    def __call__(self, t: float):
        a = self.a
        if t <= a[0, 0]:
            return tuple(a[0, 1:])
        if t >= a[-1, 0]:
            return tuple(a[-1, 1:])
        i = int(np.searchsorted(a[:, 0], t, side="right") - 1)
        u = (t - a[i, 0]) / (a[i + 1, 0] - a[i, 0])
        if self.ease:
            u = smoothstep(u)
        p = lambda j: a[min(max(j, 0), len(a) - 1), 1:]
        return tuple(_catmull(p(i - 1), p(i), p(i + 1), p(i + 2), u))


def look_at(pos, target) -> tuple[float, float]:
    """(yaw_deg, pitch_deg_down) that aims a camera at `pos` towards `target`."""
    dx, dy, dz = (target[i] - pos[i] for i in range(3))
    return math.degrees(math.atan2(dy, dx)), math.degrees(math.atan2(-dz, math.hypot(dx, dy)))


def pose_command(x, y, z, yaw, pitch) -> str:
    ang = int(round((yaw % 360.0) * 10)) * 2048 + int(round(pitch * 10)) + 1024
    q = lambda v: int(round(v * 4)) + OFS
    return f"puke -{CAM_SCRIPT} {q(x)} {q(y)} {q(z)} {ang}"


# ------------------------------------------------------------------------------------------- session
def _configure():
    def cfg(g: vzd.DoomGame):
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_render_hud(False)
        g.set_render_minimal_hud(False)
        g.set_render_weapon(False)
        g.set_render_crosshair(False)
        g.set_render_messages(False)
        g.set_render_decals(False)
        g.set_render_particles(False)
        g.set_render_screen_flashes(False)
        g.set_render_effects_sprites(False)
        g.set_render_corpses(True)
    return cfg


class FilmSession:
    """One ViZDoom game on a decorated WAD at 1280x720 with the camera script available."""

    def __init__(self, wad: Path):
        self.runner = TileRunner(wad, resolution=vzd.ScreenResolution.RES_1280X720, configure=_configure(),
                                 recycle=False)
        self.game = self.runner.game
        self.meta = self.runner.meta

    def _doll_pos(self):
        return {o.id: (o.position_x, o.position_y, o.position_z) for o in self.game.get_state().objects
                if o.name == "DoomPlayer"}

    def run(self, inputs: dict, n_tics: int, pose=None, record_frames: bool = True, track: bool = False,
            watch: list[str] | None = None):
        """Write the ports exactly as the production runner does (so the dolls follow the logged timeline; any
        delay before the ports changes jam order and therefore results) and play `n_tics` tics, one frame per tic.
        `pose(t)` -> camera pose at tic t. `watch` output names are read from DOOM after every tic.
        Returns dict(frames, poses, dolls, watch, outputs)."""
        g = self.game
        frames, poses, dolls, series = [], [], [], []
        watch_idx = None
        if watch:
            names = [n for page in self.meta.pages for n in page]
            watch_idx = [(names.index(n) // 60, names.index(n) % 60) for n in watch]
        if pose is not None:
            g.send_game_command(pose_command(*pose(0)))
        self.runner.start(inputs)                  # advances one tic, then queues the port writes
        for t in range(n_tics):
            p = pose(t) if pose is not None else None
            if p is not None:
                g.send_game_command(pose_command(*p))
            if watch_idx:
                for pi in sorted({a for a, _ in watch_idx}):
                    g.send_game_command(f"puke -{100 + pi}")
            g.advance_action(1)
            if record_frames:
                frames.append(np.array(g.get_state().screen_buffer))
                poses.append(p)
            if track:
                dolls.append(self._doll_pos())
            if watch_idx:
                from doomformer.runtime import USER
                series.append({n: int(g.get_game_variable(USER[i])) for n, (pi, i) in zip(watch, watch_idx)})
        return dict(frames=frames, poses=poses, dolls=dolls, watch=series, outputs=self.runner.read())

    def close(self):
        self.runner.close()


# --------------------------------------------------------------------------------------- projection
def project(point, pose, fov_deg=90.0, size=(W, H)):
    """Screen position (px, py, depth) of a world point seen from pose=(x,y,z,yaw,pitch_down), or None if behind.
    ZDoom's horizontal FOV is `fov_deg` at the 4:3 reference and widens with the aspect ratio; callers calibrate
    `fov_deg` against a rendered frame."""
    x, y, z, yaw, pitch = pose
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    dx, dy, dz = point[0] - x, point[1] - y, point[2] - z
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    right = dx * math.sin(yaw) - dy * math.cos(yaw)
    f2 = fwd * math.cos(pitch) - dz * math.sin(pitch)
    up = dz * math.cos(pitch) + fwd * math.sin(pitch)
    if f2 <= 1:
        return None
    f = (size[0] / 2) / math.tan(math.radians(fov_deg) / 2)
    return size[0] / 2 + f * right / f2, size[1] / 2 - f * up / f2, f2


# --------------------------------------------------------------------------------------- video output
TECH = dict(zone_floor="FLAT5_1", zone_wall="STEP1", zone_ceil="CEIL5_1", lane_floor="FLAT5_1", lane_wall="STEP1",
            gate_floor="FLAT1_1", gate_ceil="CEIL5_1", gate_wall="COMPSTA1", gate_light=255, zone_light=215,
            fx_floor="FLAT14", fx_light=255, door_wall="DOOR3")


def caption(frame: np.ndarray, lines: list[str], big: str | None = None) -> np.ndarray:
    """Composite a small caption plate (bottom-left) and an optional big readout (top-left) onto a frame copy."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    small = ImageFont.load_default(size=22)
    if lines:
        w = max(d.textlength(s, font=small) for s in lines) + 28
        h = 12 + 28 * len(lines)
        d.rectangle([20, im.height - 20 - h, 20 + w, im.height - 20], fill=(0, 0, 0, 150))
        for i, s in enumerate(lines):
            d.text((34, im.height - 20 - h + 8 + 28 * i), s, font=small, fill=(255, 220, 120, 255))
    if big:
        f = ImageFont.load_default(size=44)
        w = d.textlength(big, font=f) + 36
        d.rectangle([20, 20, 20 + w, 20 + 66], fill=(0, 0, 0, 150))
        d.text((38, 30), big, font=f, fill=(120, 255, 140, 255))
    return np.array(Image.alpha_composite(im, ov).convert("RGB"))


def encode(frames, path: Path, fps: int = 35, slow: int = 1):
    """H.264 mp4. slow>1: motion-interpolated slow motion by that factor, still `fps` frames/s (the clip
    gets `slow` times longer than game time)."""
    import subprocess

    import imageio.v2 as imageio
    import imageio_ffmpeg
    path = Path(path)
    tmp = path.with_suffix(".raw.mp4") if slow > 1 else path
    w = imageio.get_writer(str(tmp), fps=fps, codec="libx264", quality=None, macro_block_size=16,
                           pixelformat="yuv420p", ffmpeg_params=["-crf", "21", "-preset", "medium"])
    for f in frames:
        w.append_data(f)
    w.close()
    if slow > 1:
        vf = f"setpts={slow}*PTS,minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp), "-vf", vf,
                        "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", str(path)], check=True)
        tmp.unlink()
