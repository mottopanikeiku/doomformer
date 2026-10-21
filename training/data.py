"""Official level names of the DOOM family -> character-level training set.

Provenance (names are copied verbatim, nothing invented):
* The Ultimate Doom E1-E4, Doom II, The Plutonia Experiment, TNT: Evilution: the in-game
  automap strings HUSTR_* / PHUSTR_* / THUSTR_* of id Software's Doom source
  (github.com/id-Software/DOOM linuxdoom-1.10/d_englsh.h; same text in chocolate-doom
  src/doom/d_englsh.h, both fetched and parsed).
* Doom 64 (32 original levels, MAP01-MAP32): map names listed in doom64.wad on wad-archive.com and
  cross-checked against classicdoom.com/d64maps.htm (in-game level index).
* SIGIL (E5M1-E5M9): map names listed in sigil_v1_21.wad on wad-archive.com; E5M1/E5M2/E5M3/E5M4/
  E5M5/E5M6/E5M9 also confirmed in the DoomWiki / StrategyWiki navigation boxes returned by search.

Cleaning (`clean`): upper-case; '-' and '/' become a space; every other character outside
A-Z, space, apostrophe, '!' is dropped ('.' only, in practice); repeated spaces collapse.
Names containing a digit are excluded (the digit has no vocabulary slot and deleting it would
make an invented name, e.g. "GO 2 IT" -> "GO IT"). Names longer than T-1 = 15 characters after
cleaning are excluded. Duplicates are removed (first occurrence wins).
"""
from __future__ import annotations

import re

T = 16
VOCAB = "$ ABCDEFGHIJKLMNOPQRSTUVWXYZ'!"   # index 0 = BOS/EOS marker
MAX_CHARS = T - 1

SOURCES: dict[str, list[str]] = {
    "doom1": [
        "Hangar", "Nuclear Plant", "Toxin Refinery", "Command Control", "Phobos Lab",
        "Central Processing", "Computer Station", "Phobos Anomaly", "Military Base",
        "Deimos Anomaly", "Containment Area", "Refinery", "Deimos Lab", "Command Center",
        "Halls of the Damned", "Spawning Vats", "Tower of Babel", "Fortress of Mystery",
        "Hell Keep", "Slough of Despair", "Pandemonium", "House of Pain", "Unholy Cathedral",
        "Mt. Erebus", "Limbo", "Dis", "Warrens",
        "Hell Beneath", "Perfect Hatred", "Sever The Wicked", "Unruly Evil", "They Will Repent",
        "Against Thee Wickedly", "And Hell Followed", "Unto The Cruel", "Fear",
    ],
    "doom2": [
        "entryway", "underhalls", "the gantlet", "the focus", "the waste tunnels", "the crusher",
        "dead simple", "tricks and traps", "the pit", "refueling base", "'o' of destruction!",
        "the factory", "downtown", "the inmost dens", "industrial zone", "suburbs", "tenements",
        "the courtyard", "the citadel", "gotcha!", "nirvana", "the catacombs", "barrels o' fun",
        "the chasm", "bloodfalls", "the abandoned mines", "monster condo", "the spirit world",
        "the living end", "icon of sin", "wolfenstein", "grosse",
    ],
    "plutonia": [
        "congo", "well of souls", "aztec", "caged", "ghost town", "baron's lair", "caughtyard",
        "realm", "abattoire", "onslaught", "hunted", "speed", "the crypt", "genesis",
        "the twilight", "the omen", "compound", "neurosphere", "nme", "the death domain",
        "slayer", "impossible mission", "tombstone", "the final frontier",
        "the temple of darkness", "bunker", "anti-christ", "the sewers", "odyssey of noises",
        "the gateway of hell", "cyberden", "go 2 it",
    ],
    "tnt": [
        "system control", "human bbq", "power control", "wormhole", "hanger", "open season",
        "prison", "metal", "stronghold", "redemption", "storage facility", "crater",
        "nukage processing", "steel works", "dead zone", "deepest reaches", "processing area",
        "mill", "shipping/respawning", "central processing", "administration center", "habitat",
        "lunar mining project", "quarry", "baron's den", "ballistyx", "mount pain", "heck",
        "river styx", "last call", "pharaoh", "caribbean",
    ],
    "doom64": [
        "Staging Area", "The Terraformer", "Main Engineering", "Holding Area", "Tech Center",
        "Alpha Quadrant", "Research Lab", "Final Outpost", "Even Simpler", "The Bleeding",
        "Terror Core", "Altar of Pain", "Dark Citadel", "Eye of the Storm", "Dark Entries",
        "Blood Keep", "Watch Your Step", "Spawned Fear", "The Spiral", "Breakdown", "Pitfalls",
        "Burnt Offerings", "Unholy Temple", "No Escape", "Cat And Mouse", "HardCore",
        "Playground", "The Absolution", "Outpost Omega", "The Lair", "In The Void", "Hectic",
    ],
    "sigil": [
        "Baphomet's Demesne", "Sheol", "Cages of the Damned", "Paths of Wretchedness",
        "Abaddon's Void", "Unspeakable Persecution", "Nightmare Underworld", "Halls of Perdition",
        "Realm of Iblis",
    ],
}


def clean(name: str) -> str:
    s = name.upper().replace("-", " ").replace("/", " ")
    s = "".join(c for c in s if c in VOCAB[1:])
    return re.sub(r" +", " ", s).strip()


def build() -> tuple[list[str], dict[str, list[str]]]:
    names: list[str] = []
    dropped: dict[str, list[str]] = {}
    for game, raw in SOURCES.items():
        for r in raw:
            c = clean(r)
            if any(ch.isdigit() for ch in r) or not c or len(c) > MAX_CHARS:
                dropped.setdefault(game, []).append(r)
            elif c not in names:
                names.append(c)
    return names, dropped


NAMES, DROPPED = build()


def encode(name: str) -> list[int]:
    """Model input tokens: BOS + chars."""
    return [0] + [VOCAB.index(c) for c in name]


def encode_target(name: str) -> list[int]:
    """Next-char targets for each input position; last target is the end marker 0."""
    return [VOCAB.index(c) for c in name] + [0]


if __name__ == "__main__":
    print(f"{len(NAMES)} names kept, vocab={VOCAB!r} ({len(VOCAB)} symbols)")
    print(f"{sum(len(n) + 1 for n in NAMES)} target tokens")
    for g, d in DROPPED.items():
        print(f"dropped {g}: {len(d)}")
    print(NAMES)
