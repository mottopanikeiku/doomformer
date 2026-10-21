"""The trained transformers that ship with doomformer, found by name.

`Transformer.load("doomnames_onemap")` works from anywhere: a bare name is looked up first in the installed package
data (``doomformer/models/<name>/{config.json,params.npz}``, built from the repository's ``models/`` directory by
pyproject.toml), then in the ``models/`` directory of a source checkout. A path to an existing model directory
(``models/doomnames_onemap``, ``/data/mymodel``) is used as given.

A model directory may also hold ``corpus.json``, the names it was trained on (strings, or rows with a ``clean`` field);
`corpus` reads it. Other shipped files (the MNIST MLP and its sample digits under ``mnist/``) are found by `data`.
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOTS = (_HERE / "models", _HERE.parent / "models")       # installed package data, source checkout
_FILES = ("config.json", "params.npz")


def _is_model(d: Path) -> bool:
    return all((d / f).is_file() for f in _FILES)


def names() -> list[str]:
    """Names of the shipped transformer models, sorted."""
    return sorted({d.name for root in _ROOTS if root.is_dir() for d in root.iterdir() if _is_model(d)})


def resolve(model) -> Path:
    """Directory of a model given as a path or a shipped model name (``models/<name>`` is accepted as a name)."""
    path = Path(model)
    if path.is_dir():
        return path
    name = path.name if path.parent in (Path("."), Path("models")) else None
    for root in _ROOTS:
        if name and _is_model(root / name):
            return root / name
    raise FileNotFoundError(f"no model directory {str(model)!r} and no shipped model of that name; "
                            f"shipped models: {', '.join(names()) or 'none found'}")


def corpus(model) -> set[str] | None:
    """The training names listed in the model's ``corpus.json`` (upper case), or None if it has none."""
    f = resolve(model) / "corpus.json"
    if not f.is_file():
        return None
    return {(r if isinstance(r, str) else r["clean"]).upper() for r in json.loads(f.read_text())}


def data(relpath: str) -> Path:
    """A shipped data file such as ``mnist/mlp.npz``: installed package data first, then the checkout's ``models/``."""
    for root in _ROOTS:
        if (root / relpath).is_file():
            return root / relpath
    raise FileNotFoundError(f"shipped file {relpath!r} not found under {' or '.join(map(str, _ROOTS))}")
