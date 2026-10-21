"""Tiny polite fetcher with an on-disk cache (gzip) used by fetch_levels.py.

doomwiki.org answers every non-browser request with a Cloudflare 403, so DoomWiki pages are read
from the Internet Archive's copy: https://web.archive.org/web/<timestamp>/https://doomwiki.org/wiki/<Title>
(the `id_` flavour of the URL returns the original, un-rewritten HTML). The snapshot URL that was
actually opened is returned and is what gets recorded as `source_url`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache")
UA = "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36 doomformer-corpus/1.0"
DELAY = 0.15  # seconds between live requests

_last = 0.0
_lock = threading.Lock()


def _get(url: str, timeout: int = 45) -> tuple[bytes, str]:
    global _last
    with _lock:  # spaces the *start* of requests; transfers still overlap across threads
        wait = DELAY - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    data = gzip.decompress(data)
                return data, r.geturl()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 429:   # 403 here is usually a captured error page: deterministic
                raise
            time.sleep(2 + 3 * attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"giving up on {url}")


def fetch(url: str, fresh: bool = False) -> tuple[str, str]:
    """Return (text, final_url). Results are cached by request URL; fresh=True bypasses the cache."""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(url.encode()).hexdigest()
    path = os.path.join(CACHE, key + ".json.gz")
    if not fresh and os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("status"):
            raise urllib.error.HTTPError(url, d["status"], "cached error", None, None)  # type: ignore[arg-type]
        return d["text"], d["final_url"]
    try:
        data, final = _get(url)
    except urllib.error.HTTPError as e:
        if 400 <= e.code < 500 and e.code != 429:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                json.dump({"url": url, "status": e.code}, f)
        raise
    text = data.decode("utf-8", errors="replace")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"url": url, "final_url": final, "text": text}, f)
    return text, final


def latest_ok_capture(title: str) -> str | None:
    """Timestamp of the newest HTTP-200 capture of the DoomWiki article (via the Wayback CDX API).

    A bare /web/2025id_/ request redirects to the *nearest* capture even if that capture is a recorded
    403/5xx error page, so the usable capture is chosen explicitly.
    """
    q = urllib.parse.quote(title, safe=":/_,()!'*-.~")
    url = ("https://web.archive.org/cdx/search/cdx?url=doomwiki.org/wiki/" + urllib.parse.quote(q, safe="") +
           "&output=txt&fl=timestamp,length&filter=statuscode:200&limit=-4")
    txt, _ = fetch(url)
    best = None
    for line in txt.splitlines():
        p = line.split()
        if len(p) == 2 and p[1].isdigit() and int(p[1]) > 3000:   # skip stub/error captures
            best = p[0] if best is None or p[0] > best else best
    return best


TS_TRIES = ["20260601000000", "20251201000000", "20250601000000", "20241201000000", "20240601000000",
            "20230101000000", "20220101000000", "20200101000000"]


def _usable(html: str) -> bool:
    return 'id="mw-content-text"' in html and len(html) > 5000


def wiki(title: str, fresh: bool = False) -> tuple[str, str]:
    """Fetch DoomWiki article `title` (underscored, as in the /wiki/ URL) via the Wayback Machine.

    Returns (html, viewer_url) where viewer_url is https://web.archive.org/web/<exact ts>/https://doomwiki.org/wiki/<title>.
    `/web/<ts>id_/` redirects to the capture nearest to <ts>, which can be a recorded 403 page, so several
    timestamps are tried and the result must look like a rendered article; the CDX API (slow: ~1 min per
    lookup) is the last resort. Raises urllib.error.HTTPError(404) when the page was never captured.
    """
    q = urllib.parse.quote(title, safe=":/_,()!'*-.~")

    def get(ts: str):
        html, final = fetch(f"https://web.archive.org/web/{ts}id_/https://doomwiki.org/wiki/{q}", fresh=fresh)
        m = re.match(r"https://web\.archive\.org/web/(\d{14})id_/(.*)$", final)
        viewer = f"https://web.archive.org/web/{m.group(1)}/{m.group(2)}" if m else final
        return html, viewer

    for k, ts in enumerate(TS_TRIES):
        try:
            html, viewer = get(ts)
        except urllib.error.HTTPError as e:
            if e.code == 404 and k == 0:
                raise                      # never captured
            continue
        if _usable(html):
            return html, viewer
    best = latest_ok_capture(title)
    if best is not None:
        html, viewer = get(best)
        if _usable(html):
            return html, viewer
    raise urllib.error.HTTPError(title, 404, "no usable capture", None, None)  # type: ignore[arg-type]
