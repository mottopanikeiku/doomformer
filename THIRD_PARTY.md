# Third-party material

DOOMFORMER's own code and documentation are MIT-licensed (see `LICENSE`). The MIT licence does not cover the material
below: section A lists third-party **data committed in this repository**, which keeps its own licence; section B lists
software and assets **used but not distributed here** (installed or fetched at setup).

## A. In this repository, not covered by the MIT licence

| Files | Derived from | Licence | What that asks of you |
|---|---|---|---|
| `models/fly/feeding.json`, `models/fly/brain.json` (and `models/fly/NOTICE.md`) | the FlyWire connectome, v783 public release (Dorkenwald et al. 2024; Schlegel et al. 2024), as packaged by Shiu et al. 2024 / Eon Systems; selected neurons, pruned, integerized | **CC BY-NC 4.0** | non-commercial use only; attribution and an indication of changes. Citations: `models/fly/NOTICE.md`. Method and numbers: [`docs/fly_circuit.md`](docs/fly_circuit.md) |
| `training/data/levels.json` (15,088 entries), `training/data/SOURCES.md`, `models/doomnames_onemap/corpus.json` and its identical copy `models/doomnames_onemap_softmax/corpus.json` (416 entries, a subset of `levels.json`) | level names and WAD/slot labels read from the [Doom Wiki](https://doomwiki.org) through Internet Archive (web.archive.org) captures: 14,957 entries in `levels.json` (404 in `corpus.json`) cite a `doomwiki.org` capture. The other 131 (12 in `corpus.json`) are the level names of the Doom, Doom II, Plutonia and TNT games, parsed from `d_englsh.h` in id Software's released DOOM source | DoomWiki text: **CC BY-SA 4.0** ([Doom Wiki:Copyrights](https://doomwiki.org/wiki/Doom_Wiki:Copyrights)). id Software source strings: **GPL-2.0** ([id-Software/DOOM](https://github.com/id-Software/DOOM), `LICENSE.TXT`) | attribution to the Doom Wiki and its contributors (the `source_url` of every entry is its attribution link) and an indication of changes (names extracted and cleaned as described in `SOURCES.md`); redistribution of the files, or of adaptations of them, under CC BY-SA 4.0 (DoomWiki entries) or GPL-2.0 (id source entries) |
| `training/data/oracle.json` (189 facts), `training/data/oracle_pool.json` (349 facts), `training/data/ORACLE_SOURCES.md` | facts, evidence strings and source URLs from the same two sources: 93 and 166 facts cite DoomWiki captures, 96 and 183 cite files of id Software's DOOM source (`d_englsh.h`, `info.c`, `st_stuff.c`, `p_inter.c`, `p_pspr.c`, `d_items.c`, `doomdef.h`, `p_local.h`); no fact cites both | as above, per fact: CC BY-SA 4.0 for the DoomWiki facts, GPL-2.0 for the id source facts | as above |
| `training/data/make_oracle.py`, `training/data.py`, `models/doomnames/corpus.json` and its identical copies `models/doomnames_softmax/corpus.json` and `models/doomnames_modern/corpus.json` | the scripts are MIT; the level-name lists and the short evidence strings quoted inside them come from the sources above and, for Doom 64 and SIGIL, from level-name listings on wad-archive.com and classicdoom.com (`data.py` documents each source; the 136 official names of `data.py` are also listed in `models/doomnames/corpus.json`) | the excerpts keep the licence of their source | as above for the excerpts |
| `models/mnist/digits.npz` (the first 100 MNIST test digits, binarised and bit-packed, with their labels) | the [MNIST database](https://web.archive.org/web/20201231002723/http://yann.lecun.com/exdb/mnist/) by Yann LeCun and Corinna Cortes (a derivative of NIST data) | **CC BY-SA 3.0** | attribution and share-alike for these digits |
| the images and films on the GitHub release that the README and docs link to | screenshots and renders of maps and marines that use the Freedoom graphics | Freedoom: modified BSD | keep the Freedoom notice below |

How the level names and facts were collected: `training/data/SOURCES.md`, `training/data/ORACLE_SOURCES.md`. The Doom
Wiki material used is article titles, facts and short evidence excerpts. Every entry cites the exact capture it was read
from.

The trained weights in `models/*/params.npz` and `models/mnist/mlp.npz` are outputs of the training scripts in this
repository and are released under the MIT licence with the code. Several were trained on the corpora above (`doombrain`,
`doombrain_xl`, `doomnames`, `doomnames_softmax`, `doomnames_modern`, `doomnames_onemap` and
`doomnames_onemap_softmax` on level names, `doomoracle` on the oracle facts, `doomwalker` on level names for the maps it
plays). The MNIST MLP was trained on the 60,000 MNIST training digits (`training/train_mnist.py`).
Whether a trained model counts as an adaptation of its training text is not settled. The corpora are named so that you
can decide, and the training scripts are in `training/`.

DOOM is a trademark of id Software / ZeniMax Media. This project is not affiliated with or endorsed by id Software,
ZeniMax, the Freedoom project, the Doom Wiki, FlyWire, the MNIST authors or the ViZDoom authors.

## B. Used, not distributed in this repository

| Component | Used for | Where it comes from | Licence (as stated upstream) |
|---|---|---|---|
| [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) 1.3.2 | the game engine (a ZDoom fork) and its Python API | PyPI wheel (`vizdoom==1.3.2`), installed as a dependency | ViZDoom's own code: MIT; the bundled ZDoom engine contains code under several licences, see [ViZDoom's README](https://github.com/Farama-Foundation/ViZDoom#license) and [ZDoom's licence page](https://zdoom.org/wiki/license) |
| [Freedoom](https://freedoom.github.io/) Phase 1 and 2 (`freedoom1.wad`, `freedoom2.wad`) | textures, sprites, the marine (`freedoom2.wad` is the IWAD every map uses) | shipped inside the ViZDoom wheel; no WAD is committed here | modified BSD: see the notice below |
| [ACC](https://github.com/rheit/acc) (commit `bdb9bc4`) | compiling the ACS scripts embedded in every generated map | cloned and built on first use by `doomformer/acc.py` into `~/.cache/doomformer/acc` | the upstream repository has no licence file; its source files carry "Copyright (c) 1995-2023 Raven Software, Corp." Not redistributed here: check upstream before redistributing a build |
| FlyWire connectivity v783 (`2025_Connectivity_783.parquet`, `2025_Completeness_783.csv`) | rebuilding `models/fly/*.json` only | downloaded from the Eon Systems `fly-brain` repository (pinned commit, SHA-256 checked) into `build/fly` (not committed) | the data is **CC BY-NC 4.0** (FlyWire); the `fly-brain` repository's code is GPL-2.0-or-later, which does not relicense the data. `demos/fly.py` reimplements the published equations and copies no GPL code |
| NumPy, Pillow, imageio, imageio-ffmpeg; optionally PyTorch, pandas, pyarrow, SciPy | the Python toolkit, training and data scripts | PyPI, installed as dependencies | their own licences |

`docs/engine_limits.md` quotes one three-line function from ZDoom's `r_utility.h` (ViZDoom 1.3.0 source, linked) to show
the sector-lookup arithmetic.

## Freedoom notice

The Freedoom graphics visible in the images listed above are under the following licence
(<https://github.com/freedoom/freedoom/blob/master/COPYING.adoc>):

> Copyright © 2001-2024 Contributors to the Freedoom project. All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
> following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
>   disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
>   disclaimer in the documentation and/or other materials provided with the distribution.
> * Neither the name of the Freedoom project nor the names of its contributors may be used to endorse or promote products
>   derived from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
> INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> For a list of contributors to the Freedoom project, see the file CREDITS.
