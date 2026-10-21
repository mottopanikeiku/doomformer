"""Fetch the source pages used for oracle.json and write plain-text copies.

* id Software's released source (github.com/id-Software/DOOM, linuxdoom-1.10) via raw.githubusercontent.com
* DoomWiki articles via the Internet Archive (doomwiki.org itself answers 403 to this host); reuses wbfetch.wiki
  and its on-disk cache.

Output: training/data/_oracle_text/<key>.txt (+ urls.json: key -> exact URL that was opened).
Usage: python training/data/oracle_fetch.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wbfetch  # noqa: E402

# the default 2025/2026 timestamps often time out on the Wayback side; 2024 captures answer quickly
wbfetch.TS_TRIES = ["20240601000000", "20240101000000", "20230601000000", "20250101000000"]

OUT = HERE / "_oracle_text"
ID_RAW = "https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/"
ID_FILES = ["info.c", "info.h", "d_englsh.h", "st_stuff.c", "p_inter.c", "d_items.c", "doomdef.h", "p_pspr.c",
            "p_enemy.c", "p_user.c", "p_mobj.c", "g_game.c", "m_menu.c", "p_doors.c", "p_floor.c", "p_spec.c",
            "p_map.c", "d_main.c", "r_main.c", "doomstat.h", "sounds.c", "p_local.h", "m_cheat.c", "wi_stuff.c",
            "f_finale.c", "am_map.c", "p_setup.c"]

WIKI = """Imp Demon Spectre Lost_soul Cacodemon Pain_elemental Hell_knight Baron_of_Hell Revenant Mancubus Arachnotron
Arch-vile Cyberdemon Spider_Mastermind Zombieman Shotgun_guy Chaingunner Commander_Keen Icon_of_Sin Wolfenstein_SS
Pistol Fist Chainsaw Shotgun Super_shotgun Chaingun Rocket_launcher Plasma_rifle BFG9000 Backpack Soul_sphere Megasphere
Invulnerability Berserk Partial_invisibility Radiation_shielding_suit Computer_area_map Light_amplification_visor
Medikit Stimpack Armor Green_armor Blue_armor Cheating Doom id_Software John_Romero John_Carmack Tom_Hall
Adrian_Carmack Sandy_Petersen American_McGee Bobby_Prince Kevin_Cloud Dave_Taylor Jay_Wilbur Doomguy
Knee-Deep_in_the_Dead The_Shores_of_Hell Inferno Thy_Flesh_Consumed Final_Doom Doom_II:_Hell_on_Earth Doom_3
Doom_(2016) Doom_Eternal Phobos Deimos Union_Aerospace_Corporation Doom_engine The_Ultimate_Doom Doom_64
Skill_level Doom_comic Doom_(film) Quake Wolfenstein_3D Chocolate_Doom ZDoom Deathmatch Cooperative
Plutonia_Experiment TNT:_Evilution Master_Levels_for_Doom_II Barrel Blue_keycard Red_keycard Yellow_keycard
Blue_skull_key Red_skull_key Yellow_skull_key Health_bonus Armor_bonus Clip Shell Rocket Energy_cell Box_of_bullets
Box_of_shells Box_of_rockets Energy_cell_pack Spawn_cube Telefrag Infighting Hitscan Doom_(video_game)
Sigil No_Rest_for_the_Living Doomworld BFG_Division_9000 Shareware Nightmare! Hell Doom_Bible Doom_(series)
Deathmatch Shotgun_guy Former_human Former_human_sergeant Former_commando Demon_(Doom) Hell_on_Earth""".split()

SKIP_TAGS = ["script", "style", "noscript", "sup.reference", "div.printfooter", "div#toc", "table.navbox",
             "div.navbox", "span.mw-editsection", "div.mw-authority-control"]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # drop the Wayback toolbar / wombat noise and wiki chrome
    for sel in SKIP_TAGS + ["#wm-ipp-base", "#donato"]:
        for t in soup.select(sel):
            t.decompose()
    root = soup.select_one("#mw-content-text") or soup
    title = soup.select_one("#firstHeading")
    lines = [f"TITLE: {title.get_text(' ', strip=True)}"] if title else []
    for el in root.find_all(["h2", "h3", "h4", "p", "li", "tr", "dd", "dt"]):
        if el.name == "tr":
            cells = [c.get_text(" ", strip=True) for c in el.find_all(["th", "td"], recursive=False)]
            txt = " | ".join(c for c in cells if c)
        else:
            if el.find_parent("tr") is not None and el.name != "tr":
                continue
            txt = el.get_text(" ", strip=True)
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            lines.append(("## " if el.name.startswith("h") else "") + txt)
    return "\n".join(lines)


YEARS = ("2024", "2023", "2025", "2022")


def _cached(url: str):
    import gzip
    import hashlib
    p = Path(wbfetch.CACHE) / (hashlib.sha1(url.encode()).hexdigest() + ".json.gz")
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        d = json.load(f)
    return None if d.get("status") else (d["text"], d["final_url"])


def wiki(title: str) -> tuple[str, str]:
    """DoomWiki article via https://web.archive.org/web/<year>/https://doomwiki.org/wiki/<title> (redirects to the
    nearest capture of that year; the redirect target is returned and recorded as the source URL)."""
    q = urllib.parse.quote(title, safe=":/_,()!'*-.~")
    for year in YEARS:                      # a capture already in the on-disk cache always wins (reproducible text + URL)
        hit = _cached(f"https://web.archive.org/web/{year}/https://doomwiki.org/wiki/{q}")
        if hit and 'id="mw-content-text"' in hit[0] and len(hit[0]) > 5000:
            return hit
    last = None
    for year in YEARS:
        for _ in range(2):
            try:
                html, final = wbfetch.fetch(f"https://web.archive.org/web/{year}/https://doomwiki.org/wiki/{q}")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise
                last = e
                break
            except Exception as e:  # noqa: BLE001
                last = e
                continue
            if 'id="mw-content-text"' in html and len(html) > 5000:
                return html, final
            last = RuntimeError("not an article capture")
            break
    raise RuntimeError(f"no usable capture of {title}: {last}")


def fetch_one(kind: str, name: str):
    try:
        if kind == "id":
            text, final = wbfetch.fetch(ID_RAW + name)
            return f"id_{name}", text, ID_RAW + name
        html, viewer = wiki(name)
        return f"wiki_{name}", clean_html(html), viewer
    except Exception as e:  # noqa: BLE001
        return f"{kind}_{name}", None, f"FAILED {type(e).__name__}: {e}"


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [("id", f) for f in ID_FILES] + [("wiki", w) for w in dict.fromkeys(WIKI)]
    urls = {}
    with ThreadPoolExecutor(5) as ex:
        for key, text, url in ex.map(lambda j: fetch_one(*j), jobs):
            if text is None:
                print(key, url)
                continue
            (OUT / f"{re.sub(r'[^A-Za-z0-9_.()-]', '_', key)}.txt").write_text(text)
            urls[key] = url
    (OUT / "urls.json").write_text(json.dumps(urls, indent=1, sort_keys=True))
    print(len(urls), "pages ok")


if __name__ == "__main__":
    main()
