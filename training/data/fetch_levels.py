#!/usr/bin/env python3
"""Build levels.json: a verified corpus of real DOOM level / map names, every one tied to a source page.

    python fetch_levels.py                 # full run (needs network; everything is cached in ./_cache)
    python fetch_levels.py --no-pages      # quick interim run: skip the per-WAD DoomWiki page crawl
    python fetch_levels.py --out PATH      # default: levels.json next to this script

Sources (see SOURCES.md for the per-WAD table):
  1. id Software's own source, linuxdoom-1.10/d_englsh.h (HUSTR_*, PHUSTR_*, THUSTR_* automap strings):
     Doom / The Ultimate Doom (E1-E4), Doom II, The Plutonia Experiment, TNT: Evilution.
  2. DoomWiki (https://doomwiki.org) level articles. doomwiki.org answers non-browser clients with a
     Cloudflare 403, so the pages are read from the Internet Archive's captures. Two kinds of evidence:
       a. the WAD's own article ("map list page"): every link whose title has the shape
          `MAP07: Dead Simple (Doom II)` / `E2M3: Refinery (Doom)` is a level of that WAD, name = text between
          the colon and the final "(WAD)" suffix. These are cited with the snapshot URL of the WAD article.
       b. the Wayback CDX index of all archived doomwiki.org/wiki/MAP*, E?M* article URLs (status 200). The
          title of such an article is `<slot>: <Level name> (<WAD article title>)`. Levels that are not linked
          from the WAD page (or whose WAD page could not be fetched) are cited with the snapshot URL of the
          level article itself.
Nothing is invented or completed: a name is only emitted if it occurs verbatim in a cited page title/link.
"""
from __future__ import annotations

import argparse
import collections
import html as htmllib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wbfetch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CDX = "https://web.archive.org/cdx/search/cdx"
ID_URL = "https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_englsh.h"

MAX_CLEAN = 23
ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ '!")

# ----- WAD tiers: output order == dedupe priority ("keep first source") --------------------------------
OFFICIAL_WIKI = ["Doom 64", "SIGIL", "SIGIL II", "No Rest for the Living", "Legacy of Rust", "Master Levels"]
CURATED = [  # hand-picked well-known WADs, as DoomWiki article titles
    "Scythe", "Scythe 2", "Alien Vendetta", "Hell Revealed", "Hell Revealed II", "Requiem", "Memento Mori",
    "Memento Mori II", "Plutonia 2", "Eviternity", "Eviternity II", "Sunlust", "Ancient Aliens", "BTSX-E1",
    "BTSX-E2", "Valiant", "Going Down", "Sunder", "Speed of Doom", "Deus Vult", "Deus Vult II", "Kama Sutra",
    "Hell Ground", "Stardate 20X6", "Heartland", "Lost Civilization", "Interception", "Hellbound",
    "1000 Line Community Project", "1000 Lines 2 CP", "1000 Lines 3 CP", "Icarus: Alien Vanguard",
    "Resurgence", "Doom the Way id Did", "Doom 2 the Way id Did", "TNT: Revilution", "Doom 2 In Name Only",
]
# Other id/Raven/Rogue franchises on the same engine are not DOOM level names; skip them (and community
# WADs whose DoomWiki title says they are for them).
EXCLUDE_WAD = re.compile(r"\b(Heretic|Hexen|Strife)\b", re.I)
MEGAWAD_MIN_LEVELS = 9     # WADs with at least this many level articles count as megawads/episodes

# Names that are placeholders rather than names (matched after lower-casing / stripping punctuation).
PLACEHOLDER = re.compile(r"^(untitled|unnamed|no ?name|n/?a|tbd|none|unknown|nameless)( \d+)?$")
FILENAME = re.compile(r"\.(wad|pk3|pk7|zip|deh|bex|txt|lmp|wsc)\b", re.I)
SLOT_ONLY = re.compile(r"^(map|level|e\dm\d|lvl)\s*\d*$", re.I)
LEVEL_TITLE = re.compile(r"^(MAP[0-9A-Z]{1,3}|E\dM\d{1,2}): (.+) \((.+)\)$")


def clean_name(name: str) -> str | None:
    """Brief's rule: upper-case, A-Z space ' ! only, '.' dropped, '-' '/' -> space; else None."""
    s = name.replace("\u2019", "'").replace("\u2018", "'")
    if not s.isascii():          # accents, symbols, ligatures (str.upper() would turn 'ß' into 'SS')
        return None
    if FILENAME.search(s):
        return None
    s = s.upper().replace("-", " ").replace("/", " ").replace(".", "")
    if any(c not in ALLOWED for c in s):   # digits, , ? : & ( ) " _ # ...
        return None
    s = re.sub(r" +", " ", s).strip()
    if not s or not re.search(r"[A-Z]", s) or len(s) > MAX_CLEAN:
        return None
    return s


def is_placeholder(name: str) -> bool:
    k = re.sub(r"[^a-z0-9/ ]", "", name.lower()).strip()
    return bool(PLACEHOLDER.match(k) or SLOT_ONLY.match(k))


# ----- source 1: id Software d_englsh.h ----------------------------------------------------------------
def id_levels() -> list[dict]:
    text, _ = wbfetch.fetch(ID_URL)
    out = []
    for line in text.splitlines():
        m = re.match(r'#define\s+(HUSTR_E\dM\d|HUSTR_\d+|PHUSTR_\d+|THUSTR_\d+)\s+"(.*)"\s*$', line)
        if not m:
            continue
        sym, s = m.groups()
        body = re.sub(r"^(?:E\dM\d|level \d+): ", "", s)
        assert body != s, s
        if sym.startswith("HUSTR_E"):
            slot, wad = sym[6:], "Doom"
        else:
            n = int(re.search(r"\d+$", sym).group())
            slot = f"MAP{n:02d}"
            wad = {"HUSTR": "Doom II", "PHUSTR": "The Plutonia Experiment", "THUSTR": "TNT: Evilution"}[sym.split("_")[0]]
        out.append(dict(name=body, wad=wad, slot=slot, source_url=ID_URL, tier="official"))
    assert len(out) == 36 + 32 * 3, len(out)
    return out


# ----- source 2a: Wayback CDX index of DoomWiki level articles -----------------------------------------
def cdx_level_articles() -> dict[tuple[str, str], tuple[str, str, str, str]]:
    """{(wad, slot, name)->...}: returns {title: (timestamp, slot, name, wad, original_url)} keyed by decoded title."""
    recs: dict[str, tuple[str, str, str, str, str]] = {}
    prefixes = ["MAP"] + [f"E{i}M" for i in range(1, 10)]
    for p in prefixes:
        url = (f"{CDX}?url=doomwiki.org/wiki/{p}*&output=txt&fl=timestamp,original,statuscode"
               "&collapse=urlkey&filter=statuscode:200&limit=300000")
        txt, _ = wbfetch.fetch(url)
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) != 3:
                continue
            ts, orig, _st = parts
            m = re.match(r"https?://(?:www\.)?doomwiki\.org(?::80)?/wiki/(.+)$", orig)
            if not m or "?" in m.group(1):
                continue
            title = urllib.parse.unquote(m.group(1)).replace("_", " ")
            mm = LEVEL_TITLE.match(title)
            if not mm:
                continue
            slot, name, wad = mm.groups()
            if title not in recs or ts < recs[title][0]:
                recs[title] = (ts, slot, name.strip(), wad.strip(), orig)
    return recs


# ----- source 2b: the WAD article's own level links ---------------------------------------------------
def page_levels(wad: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Fetch the DoomWiki article of `wad`; return (snapshot_url, [(slot, name)...]) or None."""
    try:
        html, viewer = wbfetch.wiki(wad.replace(" ", "_"))
    except (urllib.error.HTTPError, RuntimeError):
        return None
    i = html.find('id="mw-content-text"')
    body = html[i:] if i >= 0 else html
    j = body.find("printfooter")
    body = body[:j] if j > 0 else body
    got: list[tuple[str, str]] = []
    for t in re.findall(r'<a\b[^>]*?\btitle="([^"]+)"', body):
        t = htmllib.unescape(t)
        t = re.sub(r" \(page does not exist\)$", "", t)
        mm = LEVEL_TITLE.match(t)
        if mm and mm.group(3).strip() == wad:
            got.append((mm.group(1), mm.group(2).strip()))
    return viewer, got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pages", action="store_true", help="skip the per-WAD article crawl (interim output)")
    ap.add_argument("--out", default=os.path.join(HERE, "levels.json"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--page-tiers", default="official,curated,cacoward",
                    help="tiers whose WAD article is crawled (Wayback latency is 5-30 s/page); other tiers are cited via level articles")
    args = ap.parse_args()
    page_tiers = set(args.page_tiers.split(","))

    rows: list[dict] = id_levels()
    id_wads = {r["wad"] for r in rows}  # Doom, Doom II, The Plutonia Experiment, TNT: Evilution

    arts = cdx_level_articles()
    by_wad: dict[str, dict[tuple[str, str], tuple[str, str]]] = collections.defaultdict(dict)  # wad->(slot,name)->(ts,orig)
    skipped: dict[str, int] = collections.Counter()
    for ts, slot, name, wad, orig in arts.values():
        if EXCLUDE_WAD.search(wad):
            skipped[wad] += 1
            continue
        by_wad[wad][(slot, name)] = (ts, orig)
    if skipped:
        print("excluded non-DOOM WADs:", dict(skipped), file=sys.stderr)
    print(f"CDX: {len(arts)} level articles in {len(by_wad)} WADs", file=sys.stderr)

    # ---- tiers -----------------------------------------------------------------------------------------
    cacoward = load_cacoward_wads(set(by_wad)) if not args.no_pages else set()
    sizes = {w: len(v) for w, v in by_wad.items()}

    def tier_of(w: str) -> tuple[int, str]:
        if w in OFFICIAL_WIKI:
            return 0, "official"
        if w in CURATED:
            return 1, "curated"
        if w in cacoward:
            return 2, "cacoward"
        if sizes[w] >= MEGAWAD_MIN_LEVELS:
            return 3, "megawad"
        return 4, "small"

    # ---- per-WAD article crawl ------------------------------------------------------------------------
    pages: dict[str, tuple[str, list[tuple[str, str]]] | None] = {}
    if not args.no_pages:
        todo = [w for w in by_wad if w not in id_wads and tier_of(w)[1] in page_tiers]
        with ThreadPoolExecutor(args.workers) as ex:
            for n, (w, res) in enumerate(zip(todo, ex.map(page_levels, todo)), 1):
                pages[w] = res
                if n % 50 == 0:
                    print(f"  pages {n}/{len(todo)}", file=sys.stderr)
        print(f"pages fetched: {sum(1 for v in pages.values() if v)}/{len(pages)}", file=sys.stderr)

    def wad_key(w: str) -> tuple:
        t = tier_of(w)[0]
        if t == 0:
            return (0, OFFICIAL_WIKI.index(w), "")
        if t == 1:
            return (1, CURATED.index(w), "")
        return (t, 0, w.casefold())

    def slot_key(s: str) -> tuple:
        return tuple(int(x) if x.isdigit() else x for x in re.findall(r"\d+|\D+", s))

    report: dict[str, dict] = {}
    for w in sorted((w for w in by_wad if w not in id_wads), key=wad_key):
        tname = tier_of(w)[1]
        page = pages.get(w)
        page_set = set(page[1]) if page else set()
        items = set(by_wad[w]) | page_set
        report[w] = dict(tier=tname, cdx_level_articles=len(by_wad[w]), page_url=page[0] if page else None,
                         page_levels=len(page_set), page_only=len(page_set - set(by_wad[w])),
                         cdx_not_on_page=len(set(by_wad[w]) - page_set) if page else None)
        for slot, name in sorted(items, key=lambda x: (slot_key(x[0]), x[1])):
            if (slot, name) in page_set:
                url = page[0]
            else:
                ts, orig = by_wad[w][(slot, name)]
                url = f"https://web.archive.org/web/{ts}/{orig}"
            rows.append(dict(name=name, wad=w, slot=slot, source_url=url, tier=tname))

    # ---- clean + dedupe by `clean` (first source wins) -------------------------------------------------
    out: list[dict] = []
    seen_clean: set[str] = set()
    seen_null: set[tuple] = set()
    for r in rows:
        c = None if is_placeholder(r["name"]) else clean_name(r["name"])
        if c is not None:
            if c in seen_clean:
                continue
            seen_clean.add(c)
        else:
            k = (r["name"].casefold(), r["wad"], r["slot"])
            if k in seen_null:
                continue
            seen_null.add(k)
        out.append(dict(name=r["name"], wad=r["wad"], slot=r["slot"], source_url=r["source_url"], clean=c, tier=r["tier"]))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n]\n")
    for w in id_wads:
        report[w] = dict(tier="official", source="id d_englsh.h", n_raw=sum(1 for r in rows if r["wad"] == w))
    kept = collections.Counter(o["wad"] for o in out if o["clean"])
    nulls = collections.Counter(o["wad"] for o in out if not o["clean"])
    rawn = collections.Counter(r["wad"] for r in rows)
    for w, d in report.items():
        d.update(raw_entries=rawn[w], kept_clean=kept[w], null_clean=nulls[w])
    with open(os.path.join(HERE, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(dict(excluded=dict(skipped), page_fetch_failed=sorted(w for w, v in pages.items() if v is None),
                       wads=report), f, ensure_ascii=False, indent=1)
    nn = sum(1 for o in out if o["clean"])
    print(f"wrote {args.out}: {len(out)} entries, {nn} unique non-null clean", file=sys.stderr)


def load_cacoward_wads(known: set[str]) -> set[str]:
    """WAD titles linked from the DoomWiki pages Cacowards_2004 ... Cacowards_2025 (winners, runners-up, mentions)."""
    found: set[str] = set()
    def get(y: int):
        try:
            return wbfetch.wiki(f"Cacowards_{y}")[0]
        except (urllib.error.HTTPError, RuntimeError):
            return ""
    with ThreadPoolExecutor(4) as ex:
        for h in ex.map(get, range(2004, 2026)):
            i = h.find('id="mw-content-text"')
            for t in re.findall(r'<a href="/wiki/[^"#]+" title="([^"]+)"', h[i:] if i >= 0 else h):
                t = htmllib.unescape(t)
                if t in known:
                    found.add(t)
    print(f"cacoward-linked WADs: {len(found)}", file=sys.stderr)
    return found


if __name__ == "__main__":
    main()
