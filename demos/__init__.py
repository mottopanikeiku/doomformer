"""Research demos and film code behind the README results (not part of the installed toolkit).

Run from a checkout, from the repository root: `python -m demos.brainlevel`, `demos.aimlearn`, `demos.flyloop` and
`demos.lobotomy` (see demos/README.md). pytest puts the root on sys.path through pyproject.toml.
"""


def say(line: str = "") -> None:
    """The command-line demos print through this so a line appears as soon as a decision or step is done."""
    print(line, flush=True)
