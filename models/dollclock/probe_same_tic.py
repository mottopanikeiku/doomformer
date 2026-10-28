"""Engine probe behind dollclock.CELL: line specials that one doll crosses in the same tic do not all act.

The loop doll of the doll-driven clock walks a chain of lines, one special per line: TeleportOther for every doll,
Floor_MoveToValue for every barrier and port, Light_ChangeToValue for every light. This probe builds the counter machine
`add 3 4` with the chain cells 4 units wide (the doll crosses about 4 lines per tic) and 16 units wide (one per tic) and
reads, with a debug ACS script added to the probe's own copy of the map, where every doll stands 500 tics after the
start (the first pass ends near tic 290, the chain starts near tic 430 and ends by tic 490): dolls that are far from their home spot were not
teleported by their TeleportOther line.

    python models/dollclock/probe_same_tic.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from doomformer import dollclock
from doomformer.machine import _wad_bytes, _wad_lumps
from doomformer.machine_programs import PROGRAMS
from doomformer.runtime import USER, TileRunner
from doomformer.tile import DOLL_TID0, PAGE, TileMeta, motion_acs
from doomformer.udmf import compile_acs


def far_from_home(cell: int, tics: int = 500) -> list[int]:
    dollclock.CELL = cell
    machine, *_ = PROGRAMS["add"].build(3, 4)
    with tempfile.TemporaryDirectory() as tmp:
        wad = Path(tmp) / "probe.wad"
        meta = machine.compile(wad, clock="dolls")
        lumps = _wad_lumps(wad.read_bytes())
        bh = next(i for i, (n, _) in enumerate(lumps) if n == "BEHAVIOR")
        reads = meta.layout["read_expressions"]
        pages = [f"script {100 + i} (void) {{ " + " ".join(f"o{j + 1} = {reads[n]};" for j, n in enumerate(p)) + " }"
                 for i, p in enumerate(TileMeta.load(wad.with_suffix(".json")).pages)]
        n = meta.n_dolls
        acs = ('#include "zcommon.acs"\n' + "\n".join(f"global int {i}:o{i};" for i in range(1, PAGE + 1)) + "\n"
               + "\n".join(pages) + "\n" + motion_acs(n) + "\nscript 50 (void) { "
               + " ".join(f"o{1 + j} = GetActorX({DOLL_TID0 + j}) >> 16;" for j in range(n)) + " }\n")
        lumps[bh] = ("BEHAVIOR", compile_acs(acs))
        wad.write_bytes(_wad_bytes(lumps))
        runner = TileRunner(wad)
        game = runner.game
        game.advance_action(2)

        def snap():
            game.send_game_command("puke -50")
            game.advance_action(1)
            return [int(game.get_game_variable(USER[j])) for j in range(n)]

        home = snap()
        game.advance_action(tics)
        now = snap()
        runner.close()
    return [i for i in range(n) if abs(home[i] - now[i]) > 60]


if __name__ == "__main__":
    for cell in (4, 16):
        print(f"chain cells {cell} units wide: dolls not teleported home by tic 500 (of 24): {far_from_home(cell)}")
