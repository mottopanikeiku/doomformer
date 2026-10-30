"""Rewrite README.md in place for the PyPI project page (run in CI before the build; never committed).

PyPI renders README.md without the repository around it, so:
- relative links and image paths become absolute GitHub URLs at the release tag (GITHUB_REF_NAME, else main);
- GitHub alerts (> [!TIP]) become a bold label, which PyPI shows;
- mermaid blocks are dropped: PyPI shows them as raw code.
"""
import os
import re
import sys
from pathlib import Path

REPO = "https://github.com/mottopanikeiku/doomformer"


def absolute(target: str, ref: str) -> str:
    if re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|//)", target, re.I):
        return target
    path, _, anchor = target.partition("#")
    kind = "tree" if path.endswith("/") or "." not in path.rsplit("/", 1)[-1] else "blob"
    return f"{REPO}/{kind}/{ref}/{path.lstrip('./')}" + (f"#{anchor}" if anchor else "")


def rewrite(text: str, ref: str) -> str:
    text = re.sub(r"```mermaid\n.*?```\n?", "", text, flags=re.S)
    text = re.sub(r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$",
                  lambda m: f"> **{m.group(1).capitalize()}**", text, flags=re.M)
    text = re.sub(r"(\]\()([^)\s]+)", lambda m: m.group(1) + absolute(m.group(2), ref), text)
    text = re.sub(r'((?:href|src)=")([^"]+)', lambda m: m.group(1) + absolute(m.group(2), ref), text)
    return text


if __name__ == "__main__":
    readme = Path(sys.argv[1] if len(sys.argv) > 1 else "README.md")
    readme.write_text(rewrite(readme.read_text(), os.environ.get("GITHUB_REF_NAME") or "main"))
