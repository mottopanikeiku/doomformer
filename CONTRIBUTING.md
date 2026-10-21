# Contributing to DOOMFORMER

Made something cursed with it? Show it off.

## Show and tell

Open an issue with the **Show and tell** template: the network, what the dolls do in the map, a GIF or screenshot, and
the bit-exact check you ran. Good ones get linked from the README gallery.

## The one rule: it has to be exact

DOOM's game mechanics compute the numbers *exactly*. Anything that claims DOOM computed a result must read that result
back out of the running game and compare it, with tolerance 0, against an independent integer reference.
`df.compile(...)` brains do this on every call and raise `df.MismatchError`. If Python does part of the work instead of
the map (sampling, binning observations, copying values between games), say so.

## Setting up

```sh
git clone https://github.com/mottopanikeiku/doomformer && cd doomformer
python -m venv .venv && .venv/bin/pip install -e ".[test,train]"
.venv/bin/doomformer doctor             # checks ViZDoom, ACC and the Freedoom IWAD (ACC is built on first use)
.venv/bin/python -m pytest -q tests     # the suite; most tests boot real headless DOOM games
```

Start from [`docs/BUILD_YOUR_OWN.md`](docs/BUILD_YOUR_OWN.md).

## Pull requests

- One idea per PR. A new engine trick (a line special that does something useful, a cheaper gate) is welcome with a
  test that measures it in a real map.
- New compiler behaviour needs a test that runs it in DOOM, and ideally a negative control: a deliberately broken variant
  that the check must catch.
- `pytest -q tests` must pass. The engine is version-sensitive: ViZDoom is pinned to 1.3.2 and ACC to a fixed commit.
- Don't commit large media. Link it from your PR or issue.

By contributing you agree that your contribution is licensed under the repository's [MIT licence](LICENSE). Third-party
data keeps its own licence ([`THIRD_PARTY.md`](THIRD_PARTY.md)). If a contribution adds data from another source, state
where it comes from and under what licence.
