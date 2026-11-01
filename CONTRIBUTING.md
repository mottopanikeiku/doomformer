# Contributing to DOOMFORMER

Share a network you compiled into a DOOM map.

## Show and tell

To suggest a project for the README gallery, open an issue with the **Show and tell** template. Include the network,
what the dolls do in the map, a GIF or screenshot, and the bit-exact check you ran.

## Exact arithmetic

Read each result from the running game and compare it with an independent integer reference, with tolerance 0.
`df.compile(...)` brains do this on every call and raise `df.MismatchError` on any difference. Document work Python
does outside the map, such as sampling, binning observations or copying values between games.

## Setting up

```sh
git clone https://github.com/mottopanikeiku/doomformer && cd doomformer
python -m venv .venv && .venv/bin/pip install -e ".[test,train]"
.venv/bin/doomformer doctor             # checks ViZDoom, ACC and the Freedoom IWAD (ACC is built on first use)
.venv/bin/python -m pytest -q tests     # the suite; most tests boot real headless DOOM games
```

Start from [`docs/BUILD_YOUR_OWN.md`](docs/BUILD_YOUR_OWN.md).

## Pull requests

- Keep each PR focused on one change. Include a test in a real map for engine tricks such as useful line specials
  or cheaper gates.
- New compiler behaviour needs a test that runs it in DOOM. Include a negative control where useful: a deliberately
  broken variant that the check must catch.
- `pytest -q tests` must pass. The engine is version-sensitive: ViZDoom is pinned to 1.3.2 and ACC to a fixed commit.
- Don't commit large media. Link it from your PR or issue.

By contributing you agree that your contribution is licensed under the repository's [MIT licence](LICENSE). Third-party
data keeps its own licence ([`THIRD_PARTY.md`](THIRD_PARTY.md)). If a contribution adds data from another source, state
where it comes from and under what licence.
