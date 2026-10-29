"""Run compiled tiles inside ViZDoom."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import vizdoom as vzd
from PIL import Image

from . import _mp
from .tile import GUARD_NONCE, PAGE, TileMeta

FREEDOOM2 = os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad")
USER = [getattr(vzd.GameVariable, f"USER{i}") for i in range(1, PAGE + 1)]
VIZ_MAX_OBJECTS = 4096   # ViZDoom 1.3 lists at most this many objects (VIZ_MAX_OBJECTS, viz_defines.h)
SETTLE_TICS = 20000      # default give-up point of settle; maps with a larger settle estimate get twice it
# "circuit board" automap: effect lines orange, gates blue, dolls red
AUTOMAP_COLORS = {"am_backcolor": "00 00 00", "am_wallcolor": "40 40 60", "am_fdwallcolor": "30 50 90",
                  "am_cdwallcolor": "90 60 20", "am_tswallcolor": "18 18 28", "am_specialwallcolor": "ff b0 20",
                  "am_thingcolor": "ff 30 30", "am_yourcolor": "00 00 00"}

_INIT_LOCK = threading.Lock()


def _engine_home() -> Path:
    """Directory for the engine's config and working files: <cache>/vizdoom, else the system temp dir."""
    from .acc import cache_dir
    home = cache_dir() / "vizdoom"
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        import tempfile
        home = Path(tempfile.gettempdir()) / "doomformer-vizdoom"
        home.mkdir(parents=True, exist_ok=True)
    return home


def init_game(g: vzd.DoomGame) -> None:
    """g.init() without littering the caller's directory.

    ViZDoom writes `_vizdoom.ini` and creates an empty `_vizdoom/` in the working directory of its engine process.
    Unless the caller chose a config path, the ini goes to <cache>/vizdoom, and the engine is started with that as
    its working directory (the process cwd is switched for the duration of init(), under a lock, so call this from
    one thread at a time per process; relative scenario/game paths are made absolute first)."""
    home = _engine_home()
    if not g.get_doom_config_path():
        g.set_doom_config_path(str(home / "_vizdoom.ini"))
    for get, put in ((g.get_doom_game_path, g.set_doom_game_path),
                     (g.get_doom_scenario_path, g.set_doom_scenario_path)):
        if get():
            put(os.path.abspath(get()))
    with _INIT_LOCK:
        here = os.getcwd()
        os.chdir(home)
        before = _mp.children()
        try:
            g.init()
        finally:
            os.chdir(here)
        for pid in _mp.children() - before:               # the engine: killed if this process dies without close()
            _mp.guard_engine(pid)


class TileRunner:
    def __init__(self, wad: Path, resolution=vzd.ScreenResolution.RES_160X120, configure=None,
                 recycle: bool = True, automap: bool = False):
        """recycle=True: reuse the loaded map between runs (ACS script 3 teleports every doll home,
        restores biases and closes gates) instead of reloading it with new_episode()."""
        self.recycle = recycle
        self.runs = 0
        wad = Path(wad)
        self.meta = TileMeta.load(wad.with_suffix(".json"))
        g = self.game = vzd.DoomGame()
        g.set_doom_game_path(FREEDOOM2)
        g.set_doom_scenario_path(str(wad))
        g.set_doom_map("MAP01")
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_screen_resolution(resolution)
        g.set_render_hud(False)
        g.set_objects_info_enabled(True)
        g.set_available_game_variables(USER)
        g.set_episode_timeout(10**9)
        g.set_mode(vzd.Mode.PLAYER)
        if automap:
            g.set_automap_buffer_enabled(True)
            g.set_automap_mode(vzd.AutomapMode.OBJECTS)
            g.set_automap_rotate(False)
            g.set_automap_render_textures(False)
            g.add_game_args("+viz_am_center 1 +am_colorset 0 +am_cheat 1")
        if configure:
            configure(g)
        init_game(g)
        self.step = g.advance_action             # how start() advances tics (films substitute a recorder)
        estimate = self.meta.layout.get("estimated_settle_tics") or 0
        self.max_tics = max(SETTLE_TICS, 2 * estimate)
        if automap:
            for k, v in AUTOMAP_COLORS.items():
                g.send_game_command(f'{k} "{v}"')

    COMMANDS_PER_TIC = 512   # ViZDoom crashes when about 880 commands wait for one tic (860 ran, 880 crashed in a tile of
    # 440 signed inputs); port writes beyond this many advance a tic and go on, so wide inputs run

    def _dolls(self):
        """Positions of the DoomPlayers ViZDoom lists: all of them only below VIZ_MAX_OBJECTS objects (see settle)."""
        return tuple((o.position_x, o.position_y, o.position_z) for o in self.game.get_state().objects
                     if o.name == "DoomPlayer")

    def start(self, inputs: dict[str, int], write_delay: int = 0, reset: bool | None = None):
        """Reset the tile and write every port verbatim (gate floor or ceiling := value).
        write_delay: extra tics before the ports are written (a slow bus); results must not depend on it.
        reset: force (True) or skip (False) the recycle reset; default: reset unless this is the first run.
        Tics advance through self.step(n), so callers that film every tic can substitute their own."""
        needed = {key.split(":", 1)[1] for key in self.meta.ports}
        missing = needed - set(inputs)
        if missing:
            raise KeyError(f"{self.meta.name}: unwritten ports {sorted(missing)[:5]}")
        if reset is None:
            reset = bool(self.runs) and self.recycle
        if reset:
            self.game.send_game_command("puke -3")
            self.step(3)                         # script tic + up to 2 tics of gate movers
        else:
            if self.runs:
                self.game.new_episode()
            self.step(1)
        self.runs += 1
        if write_delay:
            self.step(write_delay)
        for k, (key, tag) in enumerate(self.meta.ports.items(), 1):
            kind, name = key.split(":", 1)
            v = int(inputs[name])
            self.game.send_game_command(f"puke -{1 if kind == 'floor' else 2} {tag} {abs(v)} {int(v < 0)}")
            if k % self.COMMANDS_PER_TIC == 0:
                self.step(1)                     # see COMMANDS_PER_TIC
        self.step(3)                             # port scripts run, gate movers finish (<= 2 tics)
        self.game.send_game_command("puke -7")   # only now drop the start barriers: dolls meet settled gates

    def frame(self) -> np.ndarray:
        """Current automap as HxWx3 uint8."""
        return np.transpose(self.game.get_state().automap_buffer, (1, 2, 0))

    @staticmethod
    def shrink(img: np.ndarray, cell: tuple[int, int]) -> np.ndarray:
        """Downscale to cell=(w, h); block max-pooling keeps 1-pixel automap lines and dolls visible."""
        w, h = cell
        H, W = img.shape[:2]
        if H % h == 0 and W % w == 0:
            return img.reshape(h, H // h, w, W // w, 3).max(axis=(1, 3))
        return np.asarray(Image.fromarray(img).resize(cell, Image.BILINEAR))

    def settle(self, check_every: int = 8, max_tics: int | None = None, frames: list | None = None,
               on_tick=None, step=None) -> int:
        """Advance until every doll and the player have stood still for `check_every` tics; returns the tics used.

        ViZDoom lists at most VIZ_MAX_OBJECTS objects (later ones are cut off). When the map has more dolls than
        that, or the list comes back full, the list cannot prove that nothing moves. The map's motion script
        (tile.motion_acs) then runs on the last tic of every window and counts every doll and the player that moved
        since its previous run; settled then also needs a zero count for the last window (sampled at both its ends).
        max_tics: raise RuntimeError beyond it; default self.max_tics (twice the layout's settle estimate, >= 20,000).
        step(n): advances n tics (default game.advance_action; films pass a per-tic recorder)."""
        step = step or self.game.advance_action
        max_tics = self.max_tics if max_tics is None else max_tics
        scripted = self.meta.n_dolls + 1 >= VIZ_MAX_OBJECTS
        used, last, sampled, moved, untagged = 0, None, None, None, None
        while used < max_tics:
            if scripted:
                script = self._motion_script()
                if check_every > 1:
                    step(check_every - 1)
                self.game.send_game_command(f"puke -{script}")
                step(1)
            else:
                step(check_every)
            used += check_every
            if frames is not None:
                frames.append(self.frame())
            if on_tick is not None:
                on_tick(used)
            objects = self.game.get_state().objects
            cur = tuple((o.position_x, o.position_y, o.position_z) for o in objects if o.name == "DoomPlayer")
            if scripted:
                moved, untagged = (int(self.game.get_game_variable(USER[i])) for i in (0, 1))
                still = cur == last and sampled == used - check_every and moved == untagged == 0
                sampled = used
            else:
                still = cur == last
                if len(objects) >= VIZ_MAX_OBJECTS:      # the list is cut off and may hide moving dolls
                    scripted, still = True, False
            if still:
                return used
            last = cur
        detail = f" (last window: {moved} moved, {untagged} dolls without their TID)" if scripted else ""
        raise RuntimeError(f"{self.meta.name}: not settled after {max_tics} tics{detail}")

    def _motion_script(self) -> int:
        script = self.meta.layout.get("motion_script")
        if script is None:
            raise RuntimeError(f"{self.meta.name}: ViZDoom lists at most {VIZ_MAX_OBJECTS} objects, this map has "
                               f"{self.meta.n_dolls} dolls, and it was compiled without the motion script that "
                               "checks every doll; recompile it")
        return script

    def read(self) -> dict[str, int]:
        """Poison and read each page. A dropped guard or page raises, even for identical consecutive pages."""
        out = {}
        guard = self.meta.layout.get("guard_script")
        for pi, page in enumerate(self.meta.pages):
            if guard is not None:
                self.game.send_game_command(f"puke -{guard} {GUARD_NONCE}")
                self.game.advance_action(1)
                if any(int(self.game.get_game_variable(USER[i])) != GUARD_NONCE for i in range(len(page))):
                    raise RuntimeError(f"{self.meta.name}: readout guard for page {pi} did not run")
            self.game.send_game_command(f"puke -{100 + pi}")
            self.game.advance_action(1)
            for i, name in enumerate(page):
                out[name] = int(self.game.get_game_variable(USER[i]))
            if guard is not None and any(out[name] == GUARD_NONCE for name in page):
                raise RuntimeError(f"{self.meta.name}: readout page {pi} did not run (it returned the guard nonce "
                                   f"{GUARD_NONCE}); the values of this pass are not DOOM's")
        return out

    def set_doors(self, closed: set[int]):
        """Close the zone doors of the given block indices (ablation) and open all others.
        Doors are not touched by the per-run reset, so the setting persists until changed."""
        from .tile import DOOR_TAG0
        for b in self.meta.layout.get("blocks", []):
            script = 5 if b["index"] in closed else 6
            self.game.send_game_command(f"puke -{script} {DOOR_TAG0 + b['index']}")
        self.game.advance_action(2)

    def run(self, inputs: dict[str, int], record_every: int = 0, cell: tuple[int, int] | None = None,
            keep_full: bool = False, write_delay: int = 0):
        """Returns (outputs, tics, frames, full_frames). Frames are automap snapshots every `record_every`
        tics, shrunk to `cell`; full_frames keeps them at render resolution when keep_full."""
        self.start(inputs, write_delay)
        raw = [self.frame()] if record_every else None
        tics = self.settle(check_every=record_every or 8, frames=raw)
        raw = raw or []
        small = [self.shrink(f, cell) for f in raw] if cell else raw
        return self.read(), tics, small, (raw if keep_full else [])

    def close(self):
        self.game.close()

    def __enter__(self) -> "TileRunner":
        return self

    def __exit__(self, *exc):
        self.close()
