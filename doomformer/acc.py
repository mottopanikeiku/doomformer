"""The ACS compiler (ACC) that turns each map's scripts into its BEHAVIOR lump.

`ensure()` returns an `acc` executable, looked up in this order:

1. the environment variable ``DOOMFORMER_ACC`` (path to an ``acc`` binary; its include files
   ``zcommon.acs`` etc. must sit next to it, one directory up, or in ``../share/acc``);
2. ``third_party/acc/build/acc`` in a source checkout, if one was built there (not looked for in an
   installed package); skipped when ``DOOMFORMER_ACC_IGNORE_REPO=1``;
3. ``<cache>/<REV>/build/acc``, where ``<cache>`` is ``DOOMFORMER_ACC_CACHE`` or ``~/.cache/doomformer/acc``.
   If it is missing it is built once, on first use: ``git clone`` https://github.com/rheit/acc, checkout of the
   pinned revision ``REV``, ``cmake`` configure + build with dynamic Linux runtimes (no glibc-static required).
   That needs git, cmake, a C compiler and network access to github.com (failures raise `AccError`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

URL = "https://github.com/rheit/acc"
REV = "bdb9bc4d2c5aee7ca3ff8da985d73aaf83af0557"      # rheit/acc, 2026-04-02 (the pinned revision)
_CHECKOUT = Path(__file__).resolve().parent.parent
REPO_ACC = _CHECKOUT / "third_party" / "acc" / "build" / "acc" if (_CHECKOUT / "pyproject.toml").is_file() else None
_BINARIES = ("acc", "acc.exe", "Release/acc.exe")      # cmake output names (single-/multi-config generators)


class AccError(RuntimeError):
    """ACC is missing and cannot be built here."""


def cache_dir() -> Path:
    """Root of doomformer's cache: DOOMFORMER_CACHE, else $XDG_CACHE_HOME/doomformer, else ~/.cache/doomformer."""
    if os.environ.get("DOOMFORMER_CACHE"):
        return Path(os.environ["DOOMFORMER_CACHE"]).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser() / "doomformer"


def acc_cache() -> Path:
    """Directory that holds per-revision ACC builds."""
    return Path(os.environ.get("DOOMFORMER_ACC_CACHE") or cache_dir() / "acc").expanduser()


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _built(src: Path) -> Path | None:
    return next((src / "build" / b for b in _BINARIES if _executable(src / "build" / b)), None)


def find() -> tuple[Path | None, str]:
    """(acc path or None, where it came from) without building anything."""
    env = os.environ.get("DOOMFORMER_ACC")
    if env:
        return Path(env).expanduser(), "DOOMFORMER_ACC"
    if REPO_ACC is not None and not os.environ.get("DOOMFORMER_ACC_IGNORE_REPO") and _executable(REPO_ACC):
        return REPO_ACC, "repository third_party/acc"
    built = _built(acc_cache() / REV)
    return (built, "cache") if built else (None, "")


_INSTALL = {"git": "sudo apt install git, or brew install git",
            "cmake": "sudo apt install cmake, or brew install cmake",
            "a C compiler (cc/gcc/clang)": "sudo apt install build-essential, or xcode-select --install on macOS"}


def missing_tools() -> list[str]:
    """Tools a first-use build needs that are not on PATH."""
    missing = [t for t in ("git", "cmake") if shutil.which(t) is None]
    cc = os.environ.get("CC")
    if not (cc and shutil.which(cc)) and not any(shutil.which(c) for c in ("cc", "gcc", "clang", "cl")):
        missing.append("a C compiler (cc/gcc/clang)")
    return missing


def require_tools() -> None:
    """Raise AccError naming each missing build tool with an install command, before any build step starts."""
    missing = missing_tools()
    if missing:
        raise AccError("ACC (the ACS compiler) is not available and cannot be built: missing "
                       + "; ".join(f"{t} (install: {_INSTALL[t]})" for t in missing)
                       + f". Or point DOOMFORMER_ACC at an acc binary built from {URL} at {REV}.")


def ensure() -> Path:
    """Path to a working acc executable, building the pinned revision into the cache on first use."""
    path, source = find()
    if source == "DOOMFORMER_ACC" and not _executable(path):
        raise AccError(f"DOOMFORMER_ACC={path} is not an executable file")
    return path if path is not None else build(acc_cache() / REV)


def include_dir(acc: Path) -> Path:
    """Directory holding zcommon.acs for this acc binary."""
    acc = Path(acc)
    for d in (acc.parent, acc.parent.parent, acc.parent.parent.parent, acc.parent.parent / "share" / "acc"):
        if (d / "zcommon.acs").is_file():
            return d
    raise AccError(f"no zcommon.acs next to {acc} (looked in its directory, its parents and ../share/acc)")


def _run(step: str, cmd: list[str], timeout: int, hint: str = ""):
    """Run one build step; any failure (non-zero exit, timeout) raises AccError with an actionable first line."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})      # never wait for a credential prompt
    except subprocess.TimeoutExpired:
        raise AccError(f"building ACC: `{step}` timed out after {timeout} s. {hint}".strip()) from None
    if r.returncode:
        detail = "\n".join(t for t in (r.stdout[-1500:].strip(), r.stderr[-1500:].strip()) if t)
        raise AccError(f"building ACC: `{step}` failed (exit {r.returncode}). {hint}\n{detail}".strip())


def build(src: Path) -> Path:
    """Clone + checkout REV + cmake build into `src` (atomic: built in a temp dir, then renamed)."""
    src = Path(src)
    require_tools()
    src.parent.mkdir(parents=True, exist_ok=True)
    lock = (src.parent / f".{src.name}.lock").open("w")
    try:
        try:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)              # concurrent first uses build once
        except ImportError:                               # no fcntl (Windows): no lock
            pass
        done = _built(src)
        if done:
            return done
        tmp = Path(tempfile.mkdtemp(prefix=f".{src.name}-", dir=src.parent))
        try:
            offline = (f"Is github.com reachable? Offline, build acc from {URL} at {REV} elsewhere and set "
                       "DOOMFORMER_ACC to the binary.")
            _run("git clone", ["git", "clone", "-q", URL, str(tmp)], timeout=300, hint=offline)
            _run("git checkout", ["git", "-C", str(tmp), "checkout", "-q", REV], timeout=120, hint=offline)
            # The pinned upstream CMakeLists forces static Linux/MinGW runtimes. Use the system's dynamic
            # runtimes instead; Fedora, for example, does not install static libc with its C compiler.
            cmake = tmp / "CMakeLists.txt"
            text = cmake.read_text()
            static = " -static -static-libgcc -static-libstdc++"
            if text.count(static) != 3:
                raise AccError("the pinned ACC CMakeLists changed: expected three static-runtime flag lists")
            cmake.write_text(text.replace(static, ""))
            _run("cmake configure", ["cmake", "-S", str(tmp), "-B", str(tmp / "build"), "-DCMAKE_BUILD_TYPE=Release"],
                 timeout=300)
            _run("cmake build", ["cmake", "--build", str(tmp / "build"), "--config", "Release", "-j", "2"], timeout=900)
            if _built(tmp) is None:
                raise AccError(f"cmake finished but no acc binary in {tmp / 'build'}")
            if src.exists():
                shutil.rmtree(src)
            os.replace(tmp, src)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return _built(src)
    finally:
        lock.close()
