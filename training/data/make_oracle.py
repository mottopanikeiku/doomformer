"""Build training/data/oracle.json + ORACLE_SOURCES.md: true DOOM facts with sources and question paraphrases.

Every fact carries (a) a source URL that was actually opened by oracle_fetch.py and (b) an `evidence` string that
build() re-checks against the fetched text of that source (case/whitespace-insensitive substring, or, for
id Software's info.c, a parsed mobjinfo field).  Nothing is added without evidence; build() aborts otherwise.

Format of an example: QUESTION + '>' + ANSWER, alphabet A-Z space ' ! ? (numbers spelled out), question <= 28 chars,
answer <= 18 chars.  Question templates that exceed 28 chars for a long entity name are dropped for that fact.

Usage: python training/data/make_oracle.py        (needs training/data/_oracle_text from oracle_fetch.py)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT = HERE / "_oracle_text"
ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ '!?")
QMAX, AMAX, MIN_Q = 28, 18, 3

ONES = "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN " \
       "SEVENTEEN EIGHTEEN NINETEEN".split()
TENS = "_ _ TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY".split()


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " HUNDRED" + (" " + words(n % 100) if n % 100 else "")
    return ONES[n // 1000] + " THOUSAND" + (" " + words(n % 1000) if n % 1000 else "")


FACTS: list[dict] = []


def fact(cat: str, answer: str, qs: list[str], src: str, ev, note: str = ""):
    """src: key in urls.json (e.g. 'id_info.c', 'wiki_Imp'); ev: substring of the source text, or ('mobj', MT, field, value)."""
    FACTS.append(dict(cat=cat, answer=answer, qs=qs, src=src, ev=ev, note=note))


def Q(*qs):
    return list(qs)


def expand(templates: list[str], **kw) -> list[str]:
    """Fill templates; drop results over the length limit; keep order, remove duplicates."""
    out = []
    for t in templates:
        q = t.format(**kw)
        if len(q) <= QMAX and q not in out:
            out.append(q)
    return out


# ----------------------------------------------------------------------------- id Software source: monsters
MOBJ = {}      # MT_name -> {field: value}


def load_mobjinfo():
    s = (TEXT / "id_info.c.txt").read_text()
    body = s[s.index("mobjinfo_t mobjinfo"):]
    for m in re.finditer(r"\{\t\t// (MT_\w+)\n(.*?)\n    \}", body, re.S):
        MOBJ[m.group(1)] = {b: a for a, b in re.findall(r"(-?\w+),\s*//\s*(\w+)", m.group(2))}


load_mobjinfo()

# (MT name, entity words, article form)
MONSTERS = [
    ("MT_POSSESSED", "ZOMBIEMAN"), ("MT_SHOTGUY", "SHOTGUN GUY"), ("MT_CHAINGUY", "CHAINGUNNER"),
    ("MT_TROOP", "IMP"), ("MT_SERGEANT", "DEMON"), ("MT_SHADOWS", "SPECTRE"), ("MT_SKULL", "LOST SOUL"),
    ("MT_HEAD", "CACODEMON"), ("MT_PAIN", "PAIN ELEMENTAL"), ("MT_KNIGHT", "HELL KNIGHT"),
    ("MT_BRUISER", "BARON OF HELL"), ("MT_UNDEAD", "REVENANT"), ("MT_FATSO", "MANCUBUS"),
    ("MT_BABY", "ARACHNOTRON"), ("MT_VILE", "ARCH VILE"), ("MT_CYBORG", "CYBERDEMON"),
    ("MT_SPIDER", "SPIDER MASTERMIND"), ("MT_KEEN", "COMMANDER KEEN"), ("MT_BARREL", "EXPLODING BARREL"),
]
HEALTH_T = ["{E} HEALTH?", "{E} HP?", "{E} HIT POINTS?", "HEALTH OF {E}?", "HP OF {E}?", "HOW MUCH HP HAS {E}?",
            "HOW MUCH HEALTH HAS {E}"]
SPEED_T = ["{E} SPEED?", "HOW FAST IS {E}?", "SPEED OF {E}?", "{E} MOVE SPEED?", "{E} WALKING SPEED?"]

for mt, ent in MONSTERS:
    h = int(MOBJ[mt]["spawnhealth"])
    if ent == "EXPLODING BARREL":
        qs = expand(["BARREL HEALTH?", "BARREL HIT POINTS?", "BARREL HP?", "HEALTH OF A BARREL?",
                     "HOW MUCH HP HAS A BARREL?"])
    else:
        qs = expand(HEALTH_T, E=ent)
    fact("health", words(h), qs, "id_info.c", ("mobj", mt, "spawnhealth", h))

for mt, ent in [("MT_TROOP", "IMP"), ("MT_SERGEANT", "DEMON"), ("MT_UNDEAD", "REVENANT"), ("MT_VILE", "ARCH VILE"),
                ("MT_CYBORG", "CYBERDEMON"), ("MT_BABY", "ARACHNOTRON"), ("MT_BRUISER", "BARON OF HELL"),
                ("MT_HEAD", "CACODEMON")]:
    v = int(MOBJ[mt]["speed"])
    fact("speed", words(v), expand(SPEED_T, E=ent), "id_info.c", ("mobj", mt, "speed", v))

# ----------------------------------------------------------------------------- id source: weapons, ammo, items
AMMO_T = ["{W} AMMO?", "AMMO FOR THE {W}?", "{W} USES WHAT AMMO?", "WHAT AMMO DOES THE {W} USE?",
          "WHAT FEEDS THE {W}?", "WHAT DOES THE {W} SHOOT?"]
for ent, ans, ev in [
        ("PISTOL", "BULLETS", "// pistol am_clip,"), ("SHOTGUN", "SHELLS", "// shotgun am_shell,"),
        ("CHAINGUN", "BULLETS", "// chaingun am_clip,"), ("ROCKET LAUNCHER", "ROCKETS", "// missile launcher am_misl,"),
        ("PLASMA RIFLE", "ENERGY CELLS", "// plasma rifle am_cell,"), ("BFG", "ENERGY CELLS", "// bfg 9000 am_cell,"),
        ("SUPER SHOTGUN", "SHELLS", "// super shotgun am_shell,"), ("CHAINSAW", "NO AMMO", "// chainsaw am_noammo,")]:
    fact("ammo", ans, expand(AMMO_T, W=ent), "id_d_items.c", ev)

fact("weapon", "SEVEN", ["SHOTGUN PELLETS PER SHOT?", "HOW MANY PELLETS IN A SHOTGUN SHOT", "PELLETS IN A SHOTGUN BLAST?",
                         "SHOTGUN PELLET COUNT?"], "id_p_pspr.c", "for (i=0 ; i<7 ; i++) P_GunShot (player->mo, false);")
fact("weapon", "TWENTY", ["SUPER SHOTGUN PELLETS?", "PELLETS PER SUPER SHOTGUN SHOT?", "HOW MANY PELLETS IN A SSG BLAST",
                          "SUPER SHOTGUN PELLET COUNT?"], "id_p_pspr.c", "for (i=0 ; i<20 ; i++) { damage = 5*(P_Random ()%3+1);")
fact("weapon", "TWO", ["SHELLS PER SUPER SHOTGUN SHOT?", "SUPER SHOTGUN SHELL USE?", "HOW MANY SHELLS DOES THE SSG USE",
                       "SHELLS USED BY SUPER SHOTGUN?"], "id_p_pspr.c", "ammo[weaponinfo[player->readyweapon].ammo]-=2;")
fact("weapon", "FORTY", ["CELLS PER BFG SHOT?", "HOW MANY CELLS DOES A BFG SHOT USE", "BFG CELL COST PER SHOT?",
                         "BFG SHOT USES HOW MANY CELLS?"], "id_p_pspr.c", "#define BFGCELLS 40")
fact("weapon", "FIVE TO FIFTEEN", ["PISTOL BULLET DAMAGE?", "DAMAGE OF A PISTOL SHOT?", "HOW MUCH DAMAGE DOES A BULLET DO",
                                   "BULLET DAMAGE?", "DAMAGE PER BULLET?"], "id_p_pspr.c", "damage = 5*(P_Random ()%3+1);")
fact("weapon", "TWO TO TWENTY", ["CHAINSAW DAMAGE?", "DAMAGE OF THE CHAINSAW?", "HOW MUCH DAMAGE DOES THE SAW DO",
                                 "CHAINSAW HIT DAMAGE?"], "id_p_pspr.c", "damage = 2*(P_Random ()%10+1);")
fact("weapon", "TWO TO TWENTY", ["FIST DAMAGE?", "PUNCH DAMAGE?", "DAMAGE OF A PUNCH?", "HOW MUCH DAMAGE DOES A PUNCH DO",
                                 "BARE FIST HIT DAMAGE?"], "id_p_pspr.c", "damage = (P_Random ()%10+1)<<1;")
fact("weapon", "TEN TIMES", ["BERSERK FIST DAMAGE?", "BERSERK PUNCH MULTIPLIER?", "HOW STRONG IS A BERSERK PUNCH",
                             "PUNCH DAMAGE WITH BERSERK?", "BERSERK MAKES PUNCHES HOW STRONG"], "id_p_pspr.c",
     "if (player->powers[pw_strength]) damage *= 10;")
fact("weapon", "NINE", ["HOW MANY WEAPONS IN DOOM?", "NUMBER OF DOOM WEAPONS?", "HOW MANY GUNS DOES DOOM HAVE",
                        "DOOM WEAPON COUNT?"], "id_doomdef.h", "wp_supershotgun, NUMWEAPONS")

for ent, n, ev in [("BULLETS", 200, "{200, 50, 300, 50}"), ("SHELLS", 50, "{200, 50, 300, 50}"),
                   ("CELLS", 300, "{200, 50, 300, 50}"), ("ROCKETS", 50, "{200, 50, 300, 50}")]:
    fact("ammo", words(n), expand(["MAX {E}?", "MAX {E} YOU CAN CARRY?", "{E} LIMIT?", "HOW MANY {E} CAN YOU HOLD",
                                   "{E} CARRY LIMIT?"], E=ent),
         "id_p_inter.c", "int maxammo[NUMAMMO] = " + ev)
fact("item", "DOUBLE AMMO LIMITS", ["WHAT DOES A BACKPACK DO?", "BACKPACK EFFECT?", "WHY GRAB A BACKPACK?",
                                    "WHAT IS A BACKPACK FOR?", "BACKPACK BONUS?"], "id_p_inter.c",
     "player->maxammo[i] *= 2;")
fact("ammo", "TEN", ["BULLETS IN A CLIP?", "HOW MANY BULLETS IN A CLIP", "CLIP SIZE?", "A CLIP GIVES HOW MANY BULLETS"],
     "id_p_inter.c", "int clipammo[NUMAMMO] = {10, 4, 20, 1};")
fact("ammo", "FIFTY", ["BULLETS IN A BOX?", "BOX OF BULLETS SIZE?", "HOW MANY BULLETS IN A BOX", "BULLET BOX HOLDS?"],
     "id_p_inter.c", ["int clipammo[NUMAMMO] = {10, 4, 20, 1};", "if (!P_GiveAmmo (player, am_clip,5))"])
fact("ammo", "FOUR", ["SHELLS IN A SHELL PICKUP?", "SMALL SHELL PICKUP SIZE?", "HOW MANY SHELLS IN ONE PICKUP",
                      "SHELL PICKUP GIVES?"], "id_d_englsh.h", '#define GOTSHELLS "Picked up 4 shotgun shells."')
fact("ammo", "TWENTY", ["SHELLS IN A BOX?", "BOX OF SHELLS SIZE?", "HOW MANY SHELLS IN A BOX", "SHELL BOX HOLDS?"],
     "id_p_inter.c", ["int clipammo[NUMAMMO] = {10, 4, 20, 1};", "if (!P_GiveAmmo (player, am_shell,5))"])
fact("ammo", "TWENTY", ["CELLS IN A CELL PICKUP?", "ENERGY CELL PICKUP SIZE?", "HOW MANY CELLS IN ONE CELL",
                        "ONE ENERGY CELL GIVES?"], "id_p_inter.c", ["int clipammo[NUMAMMO] = {10, 4, 20, 1};", "case SPR_CELL:"])
fact("ammo", "ONE HUNDRED", ["CELLS IN A CELL PACK?", "CELL PACK SIZE?", "HOW MANY CELLS IN A PACK", "CELL PACK HOLDS?"],
     "id_p_inter.c", ["int clipammo[NUMAMMO] = {10, 4, 20, 1};", "case SPR_CELP:"])
fact("ammo", "FIVE", ["ROCKETS IN A BOX?", "BOX OF ROCKETS SIZE?", "HOW MANY ROCKETS IN A BOX", "ROCKET BOX HOLDS?"],
     "id_p_inter.c", ["int clipammo[NUMAMMO] = {10, 4, 20, 1};", "if (!P_GiveAmmo (player, am_misl,5))"])

fact("item", "ONE", ["HEALTH BONUS GIVES?", "HOW MUCH HEALTH IN A BONUS", "HEALTH BONUS AMOUNT?"],
     "id_p_inter.c", "case SPR_BON1: player->health++; // can go over 100%")
fact("item", "TEN", ["STIMPACK HEALS HOW MUCH?", "STIMPACK HEALTH?", "HOW MUCH DOES A STIMPACK HEAL", "STIMPACK GIVES?"],
     "id_p_inter.c", "case SPR_STIM: if (!P_GiveBody (player, 10))")
fact("item", "TWENTY FIVE", ["MEDIKIT HEALS HOW MUCH?", "MEDIKIT HEALTH?", "HOW MUCH DOES A MEDIKIT HEAL", "MEDIKIT GIVES?"],
     "id_p_inter.c", "case SPR_MEDI: if (!P_GiveBody (player, 25))")
fact("item", "ONE HUNDRED", ["SOUL SPHERE GIVES HOW MUCH?", "SOULSPHERE HEALTH?", "HOW MUCH HEALTH IS A SOULSPHERE",
                             "SOUL SPHERE HEALTH BONUS?"], "id_p_inter.c", "case SPR_SOUL: player->health += 100;")
fact("item", "TWO HUNDRED", ["MAX HEALTH FROM ITEMS?", "HEALTH BONUS CAP?", "HIGHEST HEALTH POSSIBLE?",
                             "SOUL SPHERE HEALTH CAP?"], "id_p_inter.c", "if (player->health > 200) player->health = 200;")
fact("item", "HEALTH AND ARMOR", ["WHAT DOES A MEGASPHERE GIVE?", "MEGASPHERE GIVES?", "MEGASPHERE EFFECT?",
                                  "WHAT IS IN A MEGASPHERE?"], "id_p_inter.c",
     "case SPR_MEGA: if (gamemode != commercial) return; player->health = 200;")
fact("item", "DOOM TWO", ["WHICH GAME HAS THE MEGASPHERE?", "MEGASPHERE IS IN WHICH GAME?", "WHERE IS THE MEGASPHERE?",
                          "MEGASPHERE ONLY IN WHAT?"], "id_p_inter.c", "case SPR_MEGA: if (gamemode != commercial) return;")
fact("item", "ONE HUNDRED", ["GREEN ARMOR POINTS?", "HOW MUCH ARMOR IS GREEN ARMOR", "GREEN ARMOR GIVES?",
                             "ARMOR POINTS OF GREEN ARMOR?"], "id_p_inter.c", ["hits = armortype*100;", "case SPR_ARM1: if (!P_GiveArmor (player, 1))"])
fact("item", "TWO HUNDRED", ["BLUE ARMOR POINTS?", "HOW MUCH ARMOR IS BLUE ARMOR", "BLUE ARMOR GIVES?",
                             "ARMOR POINTS OF BLUE ARMOR?", "MEGAARMOR POINTS?"], "id_p_inter.c",
     ["hits = armortype*100;", "case SPR_ARM2: if (!P_GiveArmor (player, 2))"])
fact("item", "ONE THIRD", ["GREEN ARMOR ABSORBS?", "HOW MUCH DAMAGE DOES GREEN ARMOR TAKE", "GREEN ARMOR PROTECTION?",
                           "GREEN ARMOR SAVES HOW MUCH?"], "id_p_inter.c", "if (player->armortype == 1) saved = damage/3;")
fact("item", "ONE HALF", ["BLUE ARMOR ABSORBS?", "HOW MUCH DAMAGE DOES BLUE ARMOR TAKE", "BLUE ARMOR PROTECTION?",
                          "BLUE ARMOR SAVES HOW MUCH?"], "id_p_inter.c", "else saved = damage/2;")
fact("item", "ONE", ["ARMOR BONUS GIVES?", "HOW MUCH ARMOR IN A BONUS", "ARMOR BONUS AMOUNT?"],
     "id_p_inter.c", "case SPR_BON2: player->armorpoints++; // can go over 100%")
fact("item", "THIRTY SECONDS", ["INVULNERABILITY LASTS?", "HOW LONG IS INVULNERABILITY", "INVULNERABILITY DURATION?",
                                "HOW LONG DOES INVULN LAST"], "id_doomdef.h", "INVULNTICS = (30*TICRATE),")
fact("item", "SIXTY SECONDS", ["INVISIBILITY LASTS?", "HOW LONG IS INVISIBILITY", "PARTIAL INVISIBILITY TIME?",
                               "BLUR SPHERE DURATION?"], "id_doomdef.h", "INVISTICS = (60*TICRATE),")
fact("item", "SIXTY SECONDS", ["RADIATION SUIT LASTS?", "HOW LONG IS THE RADSUIT", "RAD SUIT DURATION?",
                               "RAD SUIT TIME?"], "id_doomdef.h", "IRONTICS = (60*TICRATE)")
fact("item", "TWO MINUTES", ["LIGHT AMP VISOR LASTS?", "HOW LONG IS THE LIGHT VISOR", "LIGHT AMPLIFICATION TIME?"], "id_doomdef.h", "INFRATICS = (120*TICRATE),")
fact("item", "ONE HUNDRED", ["BERSERK HEALS UP TO?", "BERSERK HEALTH EFFECT?", "HOW MUCH HEALTH DOES BERSERK GIVE",
                             "BERSERK HEALS TO WHAT?"], "id_p_inter.c", "if (power == pw_strength) { P_GiveBody (player, 100);")
fact("item", "ONE HUNDRED", ["NORMAL MAX HEALTH?", "PLAYER MAX HEALTH WITHOUT BONUS?", "HEALTH CAP FOR MEDIKITS?",
                             "MEDIKIT HEALTH LIMIT?"], "id_p_local.h", "#define MAXHEALTH 100")
fact("item", "SIX", ["HOW MANY KEY TYPES?", "NUMBER OF DOOM KEYS?", "HOW MANY KEYS ARE THERE", "KEY COUNT?"],
     "id_doomdef.h", "it_blueskull, it_yellowskull, it_redskull,")
fact("message", "FIND SOME MEAT!", ["CHAINSAW PICKUP MESSAGE?", "WHAT DOES THE CHAINSAW SAY?", "FOUND A CHAINSAW SAYS WHAT",
                                    "CHAINSAW MESSAGE?"], "id_d_englsh.h", '#define GOTCHAINSAW "A chainsaw! Find some meat!"')
fact("message", "OH YES", ["BFG PICKUP MESSAGE?", "WHAT DOES THE BFG MESSAGE END WITH", "BFG GET MESSAGE?",
                           "FOUND A BFG SAYS WHAT"], "id_d_englsh.h", '#define GOTBFG9000 "You got the BFG9000! Oh, yes."')
fact("message", "SUPERCHARGE!", ["SOUL SPHERE MESSAGE?", "SOULSPHERE PICKUP MESSAGE?", "WHAT DOES A SOULSPHERE SAY",
                                 "SOUL SPHERE SAYS WHAT"], "id_p_inter.c", "player->message = GOTSUPER;")
fact("message", "MEGAARMOR", ["BLUE ARMOR PICKUP MESSAGE?", "WHAT DOES BLUE ARMOR SAY", "BLUE ARMOR MESSAGE?",
                              "BLUE ARMOR SAYS WHAT"], "id_p_inter.c", ["case SPR_ARM2:", "player->message = GOTMEGA;"])
fact("engine", "THIRTY FIVE", ["TICS PER SECOND IN DOOM?", "DOOM TIC RATE?", "HOW MANY TICS A SECOND", "GAME TICS PER SECOND?"],
     "id_doomdef.h", "#define TICRATE 35")
fact("engine", "FOUR", ["MAX NETGAME PLAYERS?", "HOW MANY PLAYERS IN A NETGAME", "DOOM PLAYER LIMIT?", "MAX PLAYERS?"],
     "id_doomdef.h", "#define MAXPLAYERS 4")
fact("engine", "TWO HUNDRED", ["SCREEN HEIGHT?", "DOOM SCREEN HEIGHT IN PIXELS?", "HOW TALL IS THE SCREEN", "SCREEN PIXEL HEIGHT?"],
     "id_doomdef.h", "#define SCREENHEIGHT 200")

PAIN_T = ["{E} PAIN CHANCE?", "PAIN CHANCE OF {E}?", "{E} FLINCH CHANCE?", "{E} PAIN ODDS?", "HOW OFTEN DOES {E} FLINCH?"]
for mt, ent in [("MT_TROOP", "IMP"), ("MT_SERGEANT", "DEMON"), ("MT_UNDEAD", "REVENANT"), ("MT_FATSO", "MANCUBUS"),
                ("MT_KNIGHT", "HELL KNIGHT"), ("MT_BRUISER", "BARON OF HELL"), ("MT_SPIDER", "SPIDER MASTERMIND"),
                ("MT_CYBORG", "CYBERDEMON"), ("MT_VILE", "ARCH VILE")]:
    v = int(MOBJ[mt]["painchance"])
    fact("painchance", words(v), expand(PAIN_T, E=ent), "id_info.c", ("mobj", mt, "painchance", v))

# ----------------------------------------------------------------------------- id source: cheats
PW = ("id_doomdef.h", "pw_invulnerability, pw_strength, pw_invisibility, pw_ironfeet, pw_allmap, pw_infrared,")
CHEAT_T = ["WHAT DOES {C} DO?", "{C} CHEAT?", "WHAT IS {C}?", "WHAT HAPPENS WITH {C}?", "EFFECT OF {C}?",
           "{C} DOES WHAT?"]
for code, ans, ev in [
        ("IDDQD", "GOD MODE", ["// iddqd", "plyr->cheats ^= CF_GODMODE;"]),
        ("IDKFA", "WEAPONS AND KEYS", ["// idkfa", "plyr->cards[i] = true;", ("id_d_englsh.h", '#define STSTR_KFAADDED "Very Happy Ammo Added"')]),
        ("IDFA", "WEAPONS NO KEYS", ["// idfa", ("id_d_englsh.h", '#define STSTR_FAADDED "Ammo (no keys) Added"')]),
        ("IDCLIP", "NO CLIPPING", ["// idclip", ("id_d_englsh.h", '#define STSTR_NCON "No Clipping Mode ON"')]),
        ("IDSPISPOPD", "NO CLIPPING", ["// idspispopd", ("id_d_englsh.h", '#define STSTR_NCON "No Clipping Mode ON"')]),
        ("IDBEHOLD", "POWER UP MENU", ["// behold", ("id_d_englsh.h", '#define STSTR_BEHOLD "inVuln, Str, Inviso, Rad, Allmap, or Lite-amp"')]),
        ("IDBEHOLDV", "INVULNERABILITY", ["// beholdv", PW]),
        ("IDBEHOLDS", "BERSERK", ["// beholds", PW]),
        ("IDBEHOLDI", "INVISIBILITY", ["// beholdi", PW]),
        ("IDBEHOLDR", "RADIATION SUIT", ["// beholdr", PW]),
        ("IDBEHOLDA", "AUTOMAP", ["// beholda", PW]),
        ("IDBEHOLDL", "LIGHT AMP VISOR", ["// beholdl", PW]),
        ("IDMYPOS", "COORDINATES", ["// idmypos", 'sprintf(buf, "ang=0x%x;x,y=(0x%x,0x%x)"']),
        ("IDCLEV", "LEVEL WARP", ["// idclev", ("id_d_englsh.h", '#define STSTR_CLEV "Changing Level..."')])]:
    src = "id_st_stuff.c"
    fact("cheat", ans, expand(CHEAT_T, C=code), src, ev)

# ----------------------------------------------------------------------------- id source: level names
ENG = (TEXT / "id_d_englsh.h.txt").read_text()
D1_T = ["WHAT IS E {e} M {m}?", "NAME OF E {e} M {m}?", "EPISODE {e} MAP {m} NAME?", "E {e} M {m} LEVEL NAME?",
        "E {e} M {m} IS CALLED?", "WHAT IS E {e} M {m} CALLED"]
D2_T = ["WHAT IS MAP {n}?", "NAME OF MAP {n}?", "MAP {n} NAME?", "DOOM TWO MAP {n} NAME?", "WHAT IS MAP {n} CALLED",
        "MAP {n} IN DOOM TWO?", "LEVEL {n} OF DOOM TWO?"]


def level_ok(ans: str) -> bool:
    return len(ans) <= AMAX and set(ans) <= ALPHABET


LEVEL_ANS: dict[str, str] = {}
LEVEL_EV: dict[str, str] = {}
for m in re.finditer(r'#define HUSTR_E(\d)M(\d)\s+"E\d+M\d+: ([^"]+)"', ENG):
    e, mm, name = int(m.group(1)), int(m.group(2)), m.group(3).upper()
    if not level_ok(name):
        continue
    LEVEL_ANS[f"E{e}M{mm}"] = name
    LEVEL_EV[f"E{e}M{mm}"] = m.group(0)
    fact("level", name, expand(D1_T, e=words(e), m=words(mm)), "id_d_englsh.h", m.group(0))
for m in re.finditer(r'#define HUSTR_(\d+)\s+"level \d+: ([^"]+)"', ENG):
    n, name = int(m.group(1)), m.group(2).upper()
    if not level_ok(name):
        continue
    LEVEL_ANS[f"MAP{n:02d}"] = name
    LEVEL_EV[f"MAP{n:02d}"] = m.group(0)
    fact("level", name, expand(D2_T, n=words(n)), "id_d_englsh.h", m.group(0))


REV_T = ["WHICH MAP IS {N}?", "WHERE IS {N}?", "{N} IS WHICH LEVEL?", "{N} IS IN WHICH MAP?", "WHAT MAP HAS {N}?"]
for key, code in [("E1M1", "E ONE M ONE"), ("E2M8", "E TWO M EIGHT"), ("E3M8", "E THREE M EIGHT"),
                  ("MAP01", "MAP ONE"), ("MAP07", "MAP SEVEN"), ("MAP30", "MAP THIRTY"), ("MAP31", "MAP THIRTY ONE"),
                  ("MAP32", "MAP THIRTY TWO")]:
    fact("level-rev", code, expand(REV_T, N=LEVEL_ANS[key]), "id_d_englsh.h", LEVEL_EV[key])

# ----------------------------------------------------------------------------- id source: story texts (d_englsh.h)
fact("story", "DAISY", Q("DOOMGUY'S PET RABBIT?", "WHAT WAS THE RABBIT'S NAME?", "WHAT IS THE RABBIT CALLED?", "NAME OF THE PET RABBIT?", "THE MARINE'S RABBIT IS?"),
     "id_d_englsh.h", "happened to daisy, your pet rabbit.")
fact("story", "DEIMOS", Q("WHAT MOON FLOATS ABOVE HELL?", "WHICH MOON IS ABOVE HELL?", "WHAT FLOATS OVER HELL?", "WHAT HOVERS ABOVE HELL?"),
     "id_d_englsh.h", "Deimos floats above Hell itself!")
fact("story", "CYBERDEMON", Q("WHO RULED THE DEIMOS BASE?", "WHO RULES LOST DEIMOS?", "DEIMOS BASE RULER?", "WHICH LORD RULED DEIMOS?"),
     "id_d_englsh.h", "demon lord that ruled the lost Deimos")
fact("story", "SPIDERDEMON", Q("WHO MASTERMINDED THE INVASION?", "WHO PLANNED THE INVASION?", "INVASION MASTERMIND?", "WHO LED THE MOON INVASION?"),
     "id_d_englsh.h", "The loathsome spiderdemon that")
fact("story", "HUMANS", Q("WHO BUILT THE SECRET LEVEL?", "WHO MADE MAP THIRTY ONE?", "SECRET LEVEL BUILDERS?", "WHO BUILT WOLFENSTEIN LEVEL?"),
     "id_d_englsh.h", "LOOKS LIKE IT'S BEEN BUILT BY")
fact("story", "YOUR HOME CITY", Q("WHERE IS THE ALIEN BASE?", "ALIEN BASE LOCATION IN DOOM TWO?", "WHERE ARE THE ALIENS IN DOOM TWO", "ALIEN BASE IS WHERE?", "ALIEN BASE CITY?"),
     "id_d_englsh.h", "YOUR OWN HOME CITY, NOT FAR FROM THE")
fact("story", "STARPORT", Q("WHAT IS INFESTED IN DOOM TWO?", "WHICH PORT IS INFESTED?", "INFESTED PLACE IN DOOM TWO?", "WHAT PLACE IS INFESTED?"),
     "id_d_englsh.h", "YOU HAVE ENTERED DEEPLY INTO THE INFESTED")
fact("story", "GATEKEEPER", Q("PLUTONIA FINAL BOSS?", "WHO IS THE PLUTONIA BOSS?", "PLUTONIA BOSS NAME?", "WHO ENDS PLUTONIA?"),
     "id_d_englsh.h", "The Gatekeeper's evil face is splattered")

# ----------------------------------------------------------------------------- DoomWiki (via web.archive.org): monsters
fact("monster", "LOST SOULS", Q("WHAT DOES A PAIN ELEMENTAL SPIT?", "PAIN ELEMENTAL ATTACK?", "WHAT DOES THE PAIN ELEMENTAL FIRE", "PAIN ELEMENTAL SHOOTS WHAT?", "WHAT DOES A PE SPAWN?"),
     "wiki_Pain_elemental", "the pain elemental periodically spews lost souls at its opponents")
fact("monster", "THREE LOST SOULS", Q("WHAT HAPPENS WHEN A PE DIES?", "PAIN ELEMENTAL DEATH SPAWNS?", "WHAT COMES OUT OF A DEAD PE", "DYING PAIN ELEMENTAL RELEASES?"),
     "wiki_Pain_elemental", "When killed, the pain elemental collapses in on itself and explodes, spawning three lost souls")
fact("monster", "SPECTRE", Q("WHICH MONSTER IS PARTLY INVISIBLE?", "FUZZY INVISIBLE MONSTER?", "WHO IS PARTIALLY INVISIBLE?", "WHICH DEMON IS SEE THROUGH?"),
     "wiki_Spectre", "Spectres are partially invisible")
fact("monster", "DEMON", Q("WHAT IS A SPECTRE BASED ON?", "SPECTRE IS A FUZZY WHAT?", "SPECTRE IS THE SAME AS WHICH MONSTER", "SPECTRE SHARES SPRITES WITH?"),
     "wiki_Spectre", "exactly the same as the demon in behavior and attributes")
fact("monster", "PINKY", Q("DEMON NICKNAME?", "WHAT DO FANS CALL THE DEMON?", "ANOTHER NAME FOR THE DEMON?", "WHAT IS THE DEMON'S NICKNAME"),
     "wiki_Demon", "have often since referred to the monster as pinky")
fact("monster", "MELEE ONLY", Q("DEMON ATTACK TYPE?", "CAN A DEMON SHOOT?", "HOW DOES A DEMON ATTACK?", "DEMON ATTACKS HOW?"),
     "wiki_Demon", "unique in their exclusivity to melee attacks")
fact("monster", "BEHOLDER", Q("CACODEMON LOOKS LIKE WHAT?", "WHAT D AND D MONSTER IS A CACO", "CACODEMON D AND D LOOKALIKE?", "CACODEMON RESEMBLES WHAT?"),
     "wiki_Cacodemon", "similar to that of the Beholder")
fact("monster", "LIGHTNING BALL", Q("CACODEMON ATTACK?", "WHAT DOES A CACODEMON SPIT?", "WHAT DOES THE CACODEMON FIRE", "CACODEMON SHOOTS WHAT?"),
     "wiki_Cacodemon", "Cacodemons attack by spitting a ball of lightning")
fact("monster", "GREEK", Q("CACODEMON NAME COMES FROM?", "WHAT LANGUAGE IS CACODEMON?", "CACODEMON WORD ORIGIN?", "CACODEMON IS FROM WHICH TONGUE"),
     "wiki_Cacodemon", "derived from the Greek")
fact("monster", "ROCKETS", Q("CYBERDEMON ATTACK?", "WHAT DOES THE CYBERDEMON FIRE?", "CYBERDEMON SHOOTS WHAT?", "CYBERDEMON WEAPON?"),
     "wiki_Cyberdemon", "It attacks by firing barrages of three rockets")
fact("monster", "E TWO M EIGHT", Q("CYBERDEMON FIRST MAP?", "WHERE IS CYBERDEMON BOSS?", "EPISODE TWO BOSS MAP?", "CYBERDEMON BOSS LEVEL?", "WHERE IS THE E TWO BOSS?"),
     "wiki_Cyberdemon", "first appears as a boss at the end of Doom level E2M8")
fact("monster", "BLAST DAMAGE", Q("WHAT IS THE CYBERDEMON IMMUNE TO?", "CYBERDEMON IGNORES WHAT DAMAGE?", "WHAT DOES THE CYBERDEMON RESIST", "CYBERDEMON IMMUNITY?"),
     "wiki_Cyberdemon", "are immune to all blast damage")
fact("monster", "CYBERDEMON", Q("FASTEST MONSTER IN DOOM ONE?", "WHO IS THE FASTEST IN ORIGINAL DOOM", "FASTEST ORIGINAL DOOM MONSTER?", "WHICH MONSTER MOVES FASTEST IN DOOM"),
     "wiki_Cyberdemon", "It is also the fastest moving monster in the original game")
fact("monster", "SHORES OF HELL", Q("LOST SOUL FIRST EPISODE?", "WHICH EPISODE HAS LOST SOULS?", "LOST SOUL DEBUT EPISODE?", "WHERE DID LOST SOULS DEBUT?"),
     "wiki_Lost_soul", "first introduced in Doom's second episode, The Shores of Hell")
fact("monster", "LOST SOUL", Q("WHICH MONSTER IS A FLAMING SKULL?", "WHO IS THE FLAMING SKULL?", "FLAMING FLOATING SKULL IS WHAT?", "WHICH MONSTER IS A FLYING SKULL"),
     "wiki_Lost_soul", "The lost soul is a flaming, floating skull with horns")
fact("monster", "IT CHARGES", Q("HOW DOES A LOST SOUL ATTACK?", "LOST SOUL ATTACK?", "WHAT DOES A LOST SOUL DO?", "LOST SOULS ATTACK HOW?"),
     "wiki_Lost_soul", "it launches itself towards its target at high speed")
fact("monster", "BARON OF HELL", Q("HELL KNIGHT IS A WEAK WHAT?", "HELL KNIGHT IS A WEAKER WHO?", "WHAT IS THE HELL KNIGHT BASED ON", "STRONGER VERSION OF HELL KNIGHT?"),
     "wiki_Hell_knight", "a weaker version of the baron of Hell")
fact("monster", "DOOM TWO", Q("HELL KNIGHT FIRST APPEARS IN?", "WHICH GAME HAS THE HELL KNIGHT?", "HELL KNIGHT DEBUT GAME?", "WHERE DID THE HELL KNIGHT DEBUT"),
     "wiki_Hell_knight", "The Hell knight is an enemy introduced in Doom II")
fact("monster", "BRUISER BROTHERS", Q("BARON ORIGINAL CONCEPT?", "BARONS WERE FIRST CALLED?", "WHAT WERE BARONS FIRST NAMED", "BARON OF HELL FIRST NAME?"),
     "wiki_Baron_of_Hell", 'originally conceived as the "bruiser brothers" by Tom Hall')
fact("monster", "TOM HALL", Q("WHO CONCEIVED THE BRUISER BROTHERS?", "WHO INVENTED THE BARON?", "WHO CREATED THE BARON CONCEPT", "BARON CONCEPT AUTHOR?"),
     "wiki_Baron_of_Hell", 'originally conceived as the "bruiser brothers" by Tom Hall')
fact("monster", "HOMING MISSILES", Q("REVENANT ATTACK?", "WHICH MONSTER HAS HOMING MISSILES?", "WHAT DOES A REVENANT FIRE?", "REVENANT SHOOTS WHAT?"),
     "wiki_Revenant", "Revenants fire two varieties of missiles, unguided and homing")
fact("monster", "ONE WHO RETURNS", Q("WHAT DOES REVENANT MEAN?", "REVENANT MEANING?", "REVENANT NAME MEANS?", "MEANING OF THE WORD REVENANT"),
     "wiki_Revenant", "one who returns (from death)")
fact("monster", "A CHAINGUN", Q("WHAT DOES A CHAINGUNNER DROP?", "CHAINGUNNER LOOT?", "CHAINGUNNER DROPS WHAT?", "WHAT DOES THE HEAVY GUY DROP"),
     "wiki_Chaingunner", "It drops the chaingun, which contains 10 bullets")
fact("monster", "HEAVY WEAPON DUDE", Q("CHAINGUNNER OFFICIAL NAME?", "MANUAL NAME OF THE CHAINGUNNER?", "CHAINGUNNER IS ALSO CALLED?", "WHAT IS A CHAINGUNNER ALSO CALLED"),
     "wiki_Chaingunner", "Heavy weapon dude data")
fact("monster", "A SHOTGUN", Q("WHAT DOES A SHOTGUN GUY DROP?", "SHOTGUN GUY LOOT?", "SHOTGUN GUY DROPS WHAT?", "WHAT DROPS WHEN A SERGEANT DIES"),
     "wiki_Shotgun_guy", "the shotgun guy drops its weapon")
fact("monster", "SECRET LEVELS", Q("WHERE IS THE WOLFENSTEIN SS?", "WOLFENSTEIN SS LIVES WHERE?", "WHICH LEVELS HAVE THE SS OFFICER", "WHERE DO NAZIS APPEAR IN DOOM"),
     "wiki_Wolfenstein_SS", "makes an appearance in the secret levels of Doom II")
fact("monster", "DOOM TWO", Q("WHICH GAME HAS COMMANDER KEEN?", "COMMANDER KEEN IS IN WHICH DOOM", "KEEN APPEARS IN WHICH GAME?", "WHERE DOES KEEN APPEAR"),
     "wiki_Commander_Keen", "Appears in Doom II Hit points 100")
fact("monster", "OPENS DOORS", Q("WHAT DOES KILLING KEENS DO?", "WHAT HAPPENS IF ALL KEENS DIE?", "ALL KEENS DEAD DOES WHAT?", "KEEN DEATH EFFECT?"),
     "wiki_Commander_Keen", "When all the Keens in a map are killed, any sectors tagged 666 will open as doors")
fact("monster", "TWO HUNDRED FIFTY", Q("ICON OF SIN HEALTH?", "ICON OF SIN HIT POINTS?", "ICON OF SIN HP?", "HEALTH OF THE FINAL BOSS?", "HOW MUCH HP IS THE FINAL BOSS"),
     "wiki_Icon_of_Sin", "It starts with 250 hit points")
fact("monster", "MAP THIRTY", Q("WHERE IS THE DOOM TWO FINAL BOSS?", "WHICH MAP HAS THE ICON OF SIN?", "ICON OF SIN IS ON WHICH MAP", "WHICH LEVEL IS THE FINAL BOSS"),
     "wiki_Icon_of_Sin", "is the final boss of Doom II, in MAP30: Icon of Sin")
fact("monster", "ROMERO'S HEAD", Q("WHAT MUST YOU DESTROY IN MAP THIRTY?", "WHAT KILLS THE FINAL BOSS?", "ICON OF SIN WEAK POINT?", "WHAT DO YOU SHOOT TO WIN DOOM TWO"),
     "wiki_Icon_of_Sin", "Romero's head is the thing that must be destroyed")
fact("monster", "DEMONS", Q("WHAT DOES THE ICON OF SIN SPAWN?", "ICON OF SIN SPAWNS WHAT?", "FINAL BOSS SPAWNS WHAT?", "WHAT COMES OUT OF THE BRAIN"),
     "wiki_Icon_of_Sin", "allows it to spawn endless scores of demons")
for ent, mt_, ans in [("IMP", "MT_TROOP", "TROOP"), ("DEMON", "MT_SERGEANT", "SERGEANT"), ("REVENANT", "MT_UNDEAD", "UNDEAD"),
                      ("MANCUBUS", "MT_FATSO", "FATSO"), ("BARON OF HELL", "MT_BRUISER", "BRUISER"),
                      ("CYBERDEMON", "MT_CYBORG", "CYBORG"), ("CACODEMON", "MT_HEAD", "HEAD"),
                      ("LOST SOUL", "MT_SKULL", "SKULL"), ("HELL KNIGHT", "MT_KNIGHT", "KNIGHT"),
                      ("CHAINGUNNER", "MT_CHAINGUY", "CHAINGUY"), ("SHOTGUN GUY", "MT_SHOTGUY", "SHOTGUY")]:
    pg = {"IMP": "Imp", "DEMON": "Demon", "REVENANT": "Revenant", "MANCUBUS": "Mancubus", "BARON OF HELL": "Baron_of_Hell",
          "CYBERDEMON": "Cyberdemon", "CACODEMON": "Cacodemon", "LOST SOUL": "Lost_soul", "HELL KNIGHT": "Hell_knight",
          "CHAINGUNNER": "Chaingunner", "SHOTGUN GUY": "Shotgun_guy"}[ent]
    fact("code-name", ans, expand(["{E} CODE NAME?", "SOURCE CODE NAME OF {E}?", "WHAT IS {E} CALLED IN CODE?",
                                   "{E} INTERNAL NAME?", "ID SOURCE NAME FOR {E}?"], E=ent),
         "wiki_" + pg, "Enum " + mt_)


# ----------------------------------------------------------------------------- DoomWiki: more monsters, weapons
fact("monster", "ARCH VILE", Q("WHO RESURRECTS MONSTERS?", "WHICH MONSTER REVIVES THE DEAD?", "WHAT MONSTER RAISES CORPSES?", "WHO BRINGS MONSTERS BACK?", "RESURRECTING MONSTER?"),
     "wiki_Arch-vile", "The arch-vile has the power to resurrect other, lower-level monsters")
fact("monster", "SUMMONS FIRE", Q("ARCH VILE ATTACK?", "HOW DOES AN ARCH VILE ATTACK?", "WHAT DOES AN ARCH VILE DO?", "ARCH VILE SPECIAL ATTACK?"),
     "wiki_Arch-vile", "summoning fire upon the targeted foe")
fact("monster", "NINETY", Q("ARCH VILE BLAST DAMAGE?", "MAX ARCH VILE DAMAGE?", "HOW HARD DOES AN ARCH VILE HIT", "ARCH VILE MAX DAMAGE?"),
     "wiki_Arch-vile", "causing up to 90 points in damage")
fact("monster", "PLASMA CANNON", Q("ARACHNOTRON WEAPON?", "WHAT DOES AN ARACHNOTRON FIRE?", "ARACHNOTRON SHOOTS WHAT?", "ARACHNOTRON GUN?"),
     "wiki_Arachnotron", "it wields a plasma cannon")
fact("monster", "DEAD SIMPLE", Q("ARACHNOTRON DEBUT MAP?", "WHERE DO ARACHNOTRONS DEBUT?", "FIRST ARACHNOTRON MAP?", "WHERE IS THE FIRST ARACHNOTRON"),
     "wiki_Arachnotron", "they make their first appearance in Dead Simple")
fact("monster", "SPIDER MASTERMIND", Q("ARACHNOTRON IS A SMALL WHAT?", "WHO IS THE BIG ARACHNOTRON?", "ARACHNOTRON IS A WEAK WHO?", "ARACHNOTRON BASED ON?"),
     "wiki_Arachnotron", "It is structurally the same as the spiderdemon, but weaker, and much smaller in size")
fact("monster", "SPIDER MASTERMIND", Q("WHO IS THE INFERNO BOSS?", "EPISODE THREE BOSS?", "WHO BOSSES EPISODE THREE?", "FINAL BOSS OF INFERNO?"),
     "wiki_Spider_Mastermind", "is the boss of the final episode of Doom, Inferno")
fact("monster", "SUPER CHAINGUN", Q("SPIDER MASTERMIND GUN?", "SPIDER MASTERMIND WEAPON?", "WHAT DOES THE MASTERMIND WIELD", "MASTERMIND SHOOTS WITH WHAT?"),
     "wiki_Spider_Mastermind", "a super chaingun")
fact("monster", "A CLIP", Q("WHAT DOES A ZOMBIEMAN DROP?", "ZOMBIEMAN LOOT?", "ZOMBIEMAN DROPS WHAT?", "WHAT DROPS WHEN A ZOMBIE DIES"),
     "wiki_Zombieman", "the zombie will drop another clip each time it is killed again")
fact("monster", "A RIFLE", Q("ZOMBIEMAN WEAPON?", "ZOMBIEMAN GUN?", "WHAT DOES A ZOMBIEMAN WIELD?", "WHAT GUN HAS THE ZOMBIEMAN"),
     "wiki_Zombieman", "The zombieman is armed with a rifle")
for ent, pg, n in [("FIST", "Fist", 1), ("CHAINSAW", "Chainsaw", 1), ("PISTOL", "Pistol", 2), ("SHOTGUN", "Shotgun", 3),
                   ("CHAINGUN", "Chaingun", 4), ("ROCKET LAUNCHER", "Rocket_launcher", 5), ("PLASMA RIFLE", "Plasma_rifle", 6),
                   ("BFG", "BFG9000", 7)]:
    fact("weapon-slot", words(n), expand(["{W} KEY?", "WHICH KEY SELECTS THE {W}?", "{W} WEAPON SLOT?", "WHAT NUMBER IS THE {W}?",
                                          "WHICH SLOT IS THE {W}?"], W=ent), "wiki_" + pg, f"Weapon slot | {n}")
fact("weapon", "EVIL DEAD TWO", Q("CHAINSAW INSPIRATION?", "WHAT FILM INSPIRED THE CHAINSAW?", "WHICH MOVIE INSPIRED THE SAW?", "CHAINSAW CAME FROM WHICH MOVIE"),
     "wiki_Chainsaw", "was inspired by the 1987 comedy horror film Evil Dead II")
fact("weapon", "TOM HALL", Q("WHO BORROWED THE CHAINSAW?", "WHOSE CHAINSAW WAS SCANNED?", "WHO PROVIDED THE SAW SCAN?", "WHO SCANNED THE CHAINSAW"),
     "wiki_Chainsaw", "borrowed by Tom Hall from his then-girlfriend")
fact("weapon", "PLASMA RIFLE", Q("PLASMA GUN OTHER NAME?", "MANUAL NAME OF PLASMA GUN?", "PLASMA GUN IS ALSO CALLED?", "WHAT IS THE PLASMA GUN CALLED"),
     "wiki_Plasma_rifle", "also known as the plasma rifle in the game manuals")
fact("weapon", "BERETTA", Q("PISTOL IS BASED ON WHAT?", "WHAT GUN WAS THE PISTOL MODELED", "PISTOL IMAGES CAME FROM?", "REAL GUN BEHIND THE PISTOL?"),
     "wiki_Pistol", "created from the Beretta 92FS pistol")

# ----------------------------------------------------------------------------- DoomWiki: items
fact("item", "NUCLEAR PLANT", Q("WHERE IS THE FIRST SOUL SPHERE?", "SOUL SPHERE DEBUT MAP?", "FIRST SUPERCHARGE MAP?", "WHERE DOES THE SOUL SPHERE DEBUT"),
     "wiki_Soul_sphere", "first introduced in E1M2: Nuclear Plant")
fact("item", "TOXIN REFINERY", Q("WHERE IS THE FIRST BLUR SPHERE?", "BLUR SPHERE DEBUT MAP?", "FIRST INVISIBILITY MAP?", "WHERE DOES THE BLUR SPHERE DEBUT"),
     "wiki_Partial_invisibility", "first introduced in E1M3: Toxin Refinery")
fact("item", "SOUL SPHERE", Q("MANUAL NAME OF SUPERCHARGE?", "SUPERCHARGE IS ALSO CALLED?", "WHAT IS A SUPERCHARGE CALLED?", "SUPERCHARGE IN THE MANUAL?"),
     "wiki_Soul_sphere", "(called soul sphere in the manuals)")
fact("item", "BLUR SPHERE", Q("MANUAL NAME OF INVISIBILITY?", "INVISIBILITY IS ALSO CALLED?", "WHAT IS PARTIAL INVIS CALLED?", "INVISIBILITY IN THE MANUAL?"),
     "wiki_Partial_invisibility", "(called blur sphere in the manuals)")
fact("item", "BERSERK PACK", Q("MANUAL NAME OF BERSERK?", "BERSERK IS ALSO CALLED?", "WHAT IS BERSERK CALLED?", "BERSERK IN THE MANUAL?"),
     "wiki_Berserk", "called berserk pack in the manuals")
fact("item", "DAMAGING FLOORS", Q("RADIATION SUIT PROTECTS FROM?", "WHAT DOES THE RAD SUIT BLOCK?", "WHAT IS THE RAD SUIT FOR?", "WHY WEAR A RADIATION SUIT?"),
     "wiki_Radiation_shielding_suit", "Picking one up will make players immune to damaging floors")
fact("item", "GREEN", Q("RAD SUIT SCREEN TINT?", "WHAT COLOR IS THE RAD SUIT HAZE?", "RADIATION SUIT HAZE COLOR?", "RAD SUIT VISION COLOR?"),
     "wiki_Radiation_shielding_suit", "fill their vision with a green haze")
fact("item", "RED", Q("BERSERK SCREEN TINT?", "WHAT COLOR IS THE BERSERK HAZE?", "BERSERK HAZE COLOR?", "BERSERK VISION COLOR?"),
     "wiki_Berserk", "fills the screen with a gradually diminishing red haze")
fact("item", "INVERSE COLORS", Q("INVULNERABILITY SCREEN LOOK?", "WHAT HAPPENS TO COLORS IN INVULN?", "INVULNERABILITY VISION?", "INVULN SCREEN COLORS?"),
     "wiki_Invulnerability", "the colors in the player's view change to inverse monochrome")
fact("item", "GREEN SPHERE", Q("WHAT DOES INVULNERABILITY LOOK LIKE?", "INVULNERABILITY SPRITE?", "INVULNERABILITY LOOKS LIKE?", "INVULN ITEM SHAPE?"),
     "wiki_Invulnerability", "looks like a green sphere with a grinning red-eyed head")
fact("item", "BLUE SPHERE", Q("SOUL SPHERE COLOR?", "WHAT COLOR IS A SOUL SPHERE?", "SOUL SPHERE LOOKS LIKE?", "SUPERCHARGE COLOR?"),
     "wiki_Soul_sphere", "floating blue sphere with a glassy appearance")
fact("item", "GRAY MEDIKIT", Q("BERSERK LOOKS LIKE WHAT?", "WHAT DOES BERSERK LOOK LIKE?", "BERSERK SPRITE?", "BERSERK ITEM SHAPE?"),
     "wiki_Berserk", "an item that looks like a dark gray medikit")
fact("item", "UNSEEN MAP LINES", Q("COMPUTER MAP SHOWS WHAT?", "WHAT DOES THE AUTOMAP ITEM DO?", "AREA MAP PICKUP EFFECT?", "WHAT IS A COMPUTER MAP FOR?"),
     "wiki_Computer_area_map", "cause unexplored lines in the current level to be shown on the automap in gray")
fact("item", "NONE", Q("BACKPACK BEHOLD CODE?", "IS THERE A BACKPACK CHEAT?", "BACKPACK IDBEHOLD LETTER?", "BACKPACK POWERUP CHEAT?"),
     "wiki_Backpack", "The backpack is the only powerup which does not have a corresponding idbehold cheat code")
fact("item", "FOUR HUNDRED", Q("MAX BULLETS WITH BACKPACK?", "BULLET LIMIT WITH BACKPACK?", "BACKPACK BULLET LIMIT?", "BULLETS WITH A BACKPACK?"),
     "wiki_Backpack", "up to 400 bullets")
fact("item", "SIX HUNDRED", Q("MAX CELLS WITH BACKPACK?", "CELL LIMIT WITH BACKPACK?", "BACKPACK CELL LIMIT?", "CELLS WITH A BACKPACK?"),
     "wiki_Backpack", "600 energy cell units")
fact("item", "ONE HUNDRED", Q("MAX SHELLS WITH BACKPACK?", "SHELL LIMIT WITH BACKPACK?", "BACKPACK SHELL LIMIT?", "SHELLS WITH A BACKPACK?"),
     "wiki_Backpack", "100 shotgun shells")
fact("item", "ONE HUNDRED", Q("MAX ROCKETS WITH BACKPACK?", "ROCKET LIMIT WITH BACKPACK?", "BACKPACK ROCKET LIMIT?", "ROCKETS WITH A BACKPACK?"),
     "wiki_Backpack", "100 rockets")
fact("item", "AN EXTRA LIFE", Q("SOUL SPHERE ORIGINAL PLAN?", "PLANNED SOUL SPHERE EFFECT?", "WHAT SHOULD SOUL SPHERES GIVE?", "SOUL SPHERE WAS MEANT FOR?"),
     "wiki_Soul_sphere", "collecting it would award an extra life")
fact("item", "BLUE ARMOR", Q("MEGASPHERE ARMOR TYPE?", "MEGASPHERE ARMOR IS WHICH?", "WHICH ARMOR DOES A MEGA GIVE?", "MEGASPHERE GIVES WHICH ARMOR?"),
     "wiki_Megasphere", "The armor given by a megasphere is equivalent to blue armor")

# ----------------------------------------------------------------------------- DoomWiki: id Software, people, Doom history
fact("history", "MESQUITE TEXAS", Q("WHERE WAS ID FIRST BASED?", "ID SOFTWARE FIRST CITY?", "WHERE DID ID START?", "ID SOFTWARE'S FIRST HOME?"),
     "wiki_id_Software", "based first in Mesquite, Texas")
fact("history", "RICHARDSON TEXAS", Q("WHERE IS ID LATER BASED?", "ID SOFTWARE'S LATER HOME?", "WHERE DID ID MOVE TO?", "ID SOFTWARE'S NEW CITY?"),
     "wiki_id_Software", "and later in Richardson, Texas")
fact("history", "FEBRUARY", Q("ID FOUNDING MONTH?", "WHAT MONTH WAS ID FOUNDED?", "ID WAS FOUNDED IN WHICH MONTH?", "MONTH OF ID'S FOUNDING?"),
     "wiki_id_Software", "founded on February 1, 1991 by four members")
fact("history", "FOUR", Q("HOW MANY ID FOUNDERS?", "ID FOUNDER COUNT?", "HOW MANY PEOPLE FOUNDED ID?", "NUMBER OF ID FOUNDERS?"),
     "wiki_id_Software", "by four members of the computer company Softdisk")
fact("history", "SOFTDISK", Q("WHERE DID ID'S FOUNDERS WORK?", "ID FOUNDERS CAME FROM?", "ID FOUNDERS' OLD COMPANY?", "WHICH FIRM DID ID'S FOUNDERS LEAVE"),
     "wiki_id_Software", "by four members of the computer company Softdisk")
fact("history", "IN DEMAND", Q("WHAT DOES ID STAND FOR?", "WHERE DOES THE NAME ID COME FROM?", "ID NAME ORIGIN?", "ID MEANS WHAT?"),
     "wiki_id_Software", 'the name "id" came from the phrase "in demand"')
fact("history", "ID TECH ONE", Q("DOOM ENGINE IS WHICH ID TECH?", "WHAT IS THE DOOM ENGINE CALLED?", "DOOM ENGINE ID NAME?", "ID TECH NUMBER OF DOOM?"),
     "wiki_id_Software", "with the Doom engine as id Tech 1")
fact("people", "JOHN CARMACK", Q("WHO WROTE THE DOOM ENGINE?", "DOOM ENGINE PROGRAMMER?", "WHO CODED THE DOOM ENGINE?", "DOOM ENGINE AUTHOR?"),
     "wiki_Doom", "with John Carmack writing the new game engine")
fact("people", "JOHN CARMACK", Q("ID LEAD PROGRAMMER?", "WHO WAS ID'S LEAD CODER?", "LEAD PROGRAMMER AT ID?", "ID'S LEAD PROGRAMMER WAS?"),
     "wiki_John_Carmack", "was the lead programmer at id Software")
fact("people", "ARMADILLO", Q("CARMACK ROCKET COMPANY?", "CARMACK'S AEROSPACE FIRM?", "WHICH ROCKET FIRM DID CARMACK START", "CARMACK FOUNDED WHICH FIRM?"),
     "wiki_John_Carmack", "the founder of Armadillo Aerospace")
fact("people", "OCULUS", Q("WHICH VR FIRM HIRED CARMACK?", "CARMACK'S VR EMPLOYER?", "CARMACK CTO AT WHICH FIRM?", "CARMACK JOINED WHICH VR FIRM"),
     "wiki_John_Carmack", "Oculus VR announced John Carmack as its new Chief Technology Officer")
fact("people", "COLORADO", Q("WHERE WAS ROMERO BORN?", "ROMERO BIRTHPLACE?", "ROMERO WAS BORN IN?", "WHICH STATE BORE ROMERO?"),
     "wiki_John_Romero", "Born in Colorado, John Romero began programming")
fact("people", "ION STORM", Q("ROMERO'S POST ID COMPANY?", "WHAT FIRM DID ROMERO AND HALL FOUND", "ROMERO AND HALL'S FIRM?", "WHICH FIRM DID ROMERO CO FOUND"),
     "wiki_John_Romero", "Romero then co-founded Ion Storm in Dallas, Texas with id co-worker Tom Hall")
fact("people", "DAIKATANA", Q("ROMERO'S ION STORM GAME?", "ROMERO'S GAME AT ION STORM?", "WHAT DID ROMERO MAKE AT ION STORM", "FAMOUS ION STORM GAME?"),
     "wiki_John_Romero", "where he designed and produced Daikatana")
fact("people", "WISCONSIN", Q("WHERE WAS TOM HALL BORN?", "TOM HALL BIRTHPLACE?", "TOM HALL IS FROM WHICH STATE?", "TOM HALL WAS BORN IN?"),
     "wiki_Tom_Hall", "Tom Hall is a Wisconsin-born game designer")
fact("people", "TOM HALL", Q("WHO WROTE THE DOOM BIBLE?", "DOOM BIBLE AUTHOR?", "WHO AUTHORED THE BIBLE?", "WHO MADE THE DOOM BIBLE?"),
     "wiki_Tom_Hall", "is the author of the Doom Bible")
fact("people", "DOOM SIXTY FOUR", Q("WHICH GAME HAS THE UNMAKER?", "UNMAKER LATER APPEARED IN?", "WHERE WAS THE UNMAKER SEEN?", "UNMAKER LATER SEEN WHERE?"),
     "wiki_Tom_Hall", "later appeared in Doom 64")
fact("history", "DECEMBER", Q("DOOM RELEASE MONTH?", "WHAT MONTH WAS DOOM RELEASED?", "DOOM LAUNCHED IN WHICH MONTH?", "MONTH OF DOOM'S RELEASE?"),
     "wiki_Doom", "It was first released on December 10, 1993")
fact("history", "WISCONSIN", Q("WHERE WAS DOOM FIRST UPLOADED?", "DOOM FIRST UPLOAD SITE?", "FIRST DOOM FTP SERVER?", "DOOM FIRST APPEARED WHERE?"),
     "wiki_Doom", "uploaded to an FTP server at the University of Wisconsin")
fact("history", "SPEAR OF DESTINY", Q("ID FINISHED WHAT BEFORE DOOM?", "WHAT WAS ID FINISHING?", "WHAT GAME CAME BEFORE DOOM?", "GAME BEFORE DOOM AT ID?"),
     "wiki_Doom", "the rest of id Software was finishing Spear of Destiny")
fact("history", "EPISODE ONE", Q("WHAT IS IN THE SHAREWARE?", "SHAREWARE DOOM CONTAINS?", "WHICH EPISODE IS SHAREWARE?", "SHAREWARE EPISODE?"),
     "wiki_Doom", "The first-episode shareware format")
fact("history", "THREE", Q("EPISODES IN REGISTERED DOOM?", "ORIGINAL DOOM EPISODE COUNT?", "HOW MANY EPISODES IN DOOM?", "REGISTERED DOOM EPISODES?"),
     "wiki_Doom", "containing all three episodes")
fact("history", "DMX SOUND LIBRARY", Q("WHY NO DOS DOOM SOURCE?", "WHAT BLOCKED THE DOS SOURCE?", "DOS SOURCE BLOCKED BY WHAT?", "DOS DOOM SOURCE PROBLEM?"),
     "wiki_Doom", "because of licensing issues with the DMX sound library")
fact("history", "PWADS", Q("WHAT ARE DOOM MOD FILES CALLED?", "CUSTOM LEVEL FILES?", "WHAT IS A PATCH WAD CALLED?", "DOOM MODS ARE CALLED?"),
     "wiki_Doom", "known as PWADs")
fact("history", "OCTOBER", Q("DOOM FILM RELEASE MONTH?", "WHAT MONTH WAS THE FILM RELEASED?", "DOOM MOVIE MONTH?", "DOOM FILM CAME OUT IN?"),
     "wiki_Doom", "a film released in October 2005")
fact("history", "FOUR", Q("HOW MANY DOOM NOVELS?", "DOOM NOVEL COUNT?", "NUMBER OF DOOM NOVELS?", "HOW MANY NOVELS DOES DOOM HAVE"),
     "wiki_Doom", "four novels")

fact("people", "CLAY MODELS", Q("WHAT DID ADRIAN CARMACK SCULPT?", "ADRIAN CARMACK BUILT WHAT?", "WHAT WERE THE FIRST MONSTER MODELS?", "DOOM MONSTER MODELS MADE OF?"),
     "wiki_Adrian_Carmack", "Adrian built clay models of the baron of Hell, the Doomguy, and the cyberdemon")
fact("people", "TWENTY", Q("SANDY PETERSEN DOOM LEVELS?", "HOW MANY DOOM LEVELS BY PETERSEN?", "PETERSEN'S DOOM LEVEL COUNT?", "LEVELS PETERSEN MADE FOR DOOM?"),
     "wiki_Sandy_Petersen", "created 20 levels for it")
fact("people", "SEVENTEEN", Q("PETERSEN DOOM TWO LEVELS?", "DOOM TWO LEVELS BY PETERSEN?", "PETERSEN'S DOOM TWO COUNT?", "HOW MANY D TWO LEVELS BY SANDY"),
     "wiki_Sandy_Petersen", "He later created 17 of the levels for Doom II")
fact("people", "SEVEN", Q("SANDY PETERSEN QUAKE LEVELS?", "HOW MANY QUAKE LEVELS BY PETERSEN?", "PETERSEN'S QUAKE LEVEL COUNT?", "LEVELS PETERSEN MADE FOR QUAKE?"),
     "wiki_Sandy_Petersen", "and seven levels for Quake")
fact("people", "SEPTEMBER", Q("WHEN DID PETERSEN JOIN ID?", "PETERSEN JOINED ID WHEN?", "PETERSEN JOIN MONTH?", "WHICH MONTH DID PETERSEN JOIN"),
     "wiki_Sandy_Petersen", "Petersen joined id Software in September 1993")
fact("people", "CALL OF CTHULHU", Q("PETERSEN'S FAMOUS RPG?", "WHAT RPG DID PETERSEN WRITE?", "PETERSEN CO WROTE WHICH RPG?", "WHICH TABLETOP GAME IS BY PETERSEN"),
     "wiki_Sandy_Petersen", "co-authored the game Call of Cthulhu")
fact("people", "ENSEMBLE STUDIOS", Q("WHERE DID PETERSEN GO AFTER ID?", "PETERSEN LEFT ID FOR?", "PETERSEN'S NEXT COMPANY?", "WHICH FIRM DID PETERSEN JOIN NEXT"),
     "wiki_Sandy_Petersen", "He left id Software for Ensemble Studios in June 1997")
fact("people", "TECH SUPPORT", Q("AMERICAN MCGEE'S FIRST ID JOB?", "MCGEE STARTED AT ID AS?", "WHAT DID MCGEE DO AT ID?", "MCGEE'S ID ROLE?"),
     "wiki_American_McGee", "he was hired at id as tech support")
fact("people", "ALICE", Q("MCGEE'S OWN GAME?", "WHAT GAME DID MCGEE MAKE AT EA?", "AMERICAN MCGEE'S GAME?", "WHICH GAME BEARS MCGEE'S NAME"),
     "wiki_American_McGee", "created his own game, American McGee's Alice")

# ----------------------------------------------------------------------------- DoomWiki: episodes
fact("episode", "PHOBOS", Q("WHERE IS EPISODE ONE SET?", "EPISODE ONE TAKES PLACE ON?", "WHICH MOON IS EPISODE ONE?", "WHAT IS THE FIRST EPISODE'S MOON?"),
     "wiki_Knee-Deep_in_the_Dead", "takes place in the UAC installations on Phobos")
fact("episode", "MARS", Q("PHOBOS IS A MOON OF WHAT?", "WHICH PLANET HAS PHOBOS?", "PHOBOS ORBITS WHICH PLANET?", "PHOBOS BELONGS TO?"),
     "wiki_Knee-Deep_in_the_Dead", "one of the moons of Mars")
fact("episode", "JOHN ROMERO", Q("WHO DESIGNED EPISODE ONE?", "EPISODE ONE DESIGNER?", "WHO MADE EPISODE ONE LEVELS?", "WHO MADE THE FIRST EPISODE?"),
     "wiki_Knee-Deep_in_the_Dead", "Designed mainly by John Romero")
fact("episode", "TECHBASE MAPS", Q("FAN E ONE STYLE MAPS?", "E ONE STYLE MAPS NAME?", "E ONE STYLE FAN MAPS?", "E ONE STYLE MAPS ARE?", "E ONE STYLE LEVELS ARE?"),
     "wiki_Knee-Deep_in_the_Dead", "are commonly referred to as techbase maps")
fact("episode", "BARONS", Q("WHO GUARDS PHOBOS ANOMALY?", "WHO GUARDS THE ANOMALY?", "EPISODE ONE'S GUARDIANS?", "WHO IS FOUGHT AT THE ANOMALY?"),
     "wiki_Knee-Deep_in_the_Dead", "defeats the powerful Barons who guard it")
fact("episode", "HELL", Q("WHERE IS EPISODE THREE SET?", "WHERE DOES INFERNO TAKE PLACE?", "INFERNO IS SET WHERE?", "EPISODE THREE SETTING?"),
     "wiki_Inferno", "This episode is set in Hell")
fact("episode", "SANDY PETERSEN", Q("WHO DESIGNED INFERNO?", "INFERNO DESIGNER?", "WHO MADE EPISODE THREE?", "WHO MADE THE INFERNO LEVELS?"),
     "wiki_Inferno", "All of the levels in this episode are credited to Sandy Petersen")
fact("episode", "DIVINE COMEDY", Q("INFERNO IS NAMED AFTER?", "WHAT INSPIRED THE NAME INFERNO?", "INFERNO NAME SOURCE?", "WHAT IS INFERNO NAMED FOR?"),
     "wiki_Inferno", "named after Inferno, part of Dante Alighieri's Divine Comedy")
fact("episode", "DIS", Q("WHAT IS THE CAPITAL OF HELL?", "CAPITAL CITY OF HELL?", "HELL'S CAPITAL?", "WHICH CITY IS HELL'S CAPITAL"),
     "wiki_Inferno", "its capital city of Dis")
for k, (nm, ans, ev_pg) in enumerate([("INFERNO", "EPISODE THREE", "wiki_Inferno"), ("KNEE DEEP", "EPISODE ONE", "wiki_Knee-Deep_in_the_Dead")]):
    fact("episode", ans, expand(["WHICH EPISODE IS {N}?", "{N} IS WHICH EPISODE?", "WHAT EPISODE IS {N}?", "{N} EPISODE?", "{N} IS EPISODE WHAT?"], N=nm), ev_pg,
         "Inferno is the third episode in Doom" if k == 0 else "Knee-Deep in the Dead is the first episode in Doom")
fact("episode", "INFERNO", Q("WHAT IS EPISODE THREE CALLED?", "EPISODE THREE NAME?", "NAME OF EPISODE THREE?", "THIRD EPISODE NAME?"),
     "wiki_Inferno", "Inferno is the third episode in Doom")

fact("episode", "EPISODE FOUR", Q("THY FLESH CONSUMED IS WHICH EPISODE?", "WHICH EPISODE IS THY FLESH?", "THY FLESH CONSUMED EPISODE?", "WHAT EPISODE IS THY FLESH?", "THY FLESH IS EPISODE WHAT?"),
     "wiki_Thy_Flesh_Consumed", "Thy Flesh Consumed is the fourth and final episode in The Ultimate Doom")
fact("episode", "THY FLESH CONSUMED", Q("WHAT IS EPISODE FOUR CALLED?", "EPISODE FOUR NAME?", "NAME OF EPISODE FOUR?", "FOURTH EPISODE NAME?", "LAST EPISODE NAME?"),
     "wiki_Thy_Flesh_Consumed", "Thy Flesh Consumed is the fourth and final episode in The Ultimate Doom")
fact("episode", "THE BIBLE", Q("WHERE ARE THE E FOUR NAMES FROM?", "E FOUR LEVEL NAMES SOURCE?", "WHERE DO E FOUR TITLES COME FROM", "E FOUR NAMES ARE FROM?"),
     "wiki_Thy_Flesh_Consumed", "are phrases taken directly from the Bible")
fact("episode", "KEVIN CLOUD", Q("WHO COMPILED THE E FOUR NAMES?", "WHO LISTED THE E FOUR NAMES?", "E FOUR NAME COMPILER?", "WHO PICKED THE E FOUR NAME LIST"),
     "wiki_Thy_Flesh_Consumed", "The list was compiled by Kevin Cloud")
fact("episode", "JOHN ROMERO", Q("WHO NAMED THE E FOUR LEVELS?", "WHO ASSIGNED E FOUR LEVEL NAMES?", "E FOUR LEVEL NAMER?", "WHO GAVE E FOUR ITS NAMES"),
     "wiki_Thy_Flesh_Consumed", "John Romero assigned names to the individual levels")
fact("episode", "TIM WILLITS", Q("WHOSE FIRST DOOM WORK WAS E FOUR?", "WHO JOINED AFTER DOOM AND DID E FOUR", "E FOUR FIRST TIME DESIGNER?", "WHO DEBUTED IN THY FLESH?"),
     "wiki_Thy_Flesh_Consumed", "This episode was also Tim Willits' first contribution to the Doom series")

# ----------------------------------------------------------------------------- DoomWiki: skills, Doom II, characters, engine
fact("skill", "SKILL FIVE", Q("WHICH SKILL IS NIGHTMARE?", "NIGHTMARE IS WHICH SKILL?", "NIGHTMARE IS SKILL WHAT?", "NIGHTMARE SKILL NUMBER?", "WHAT IS THE NIGHTMARE LEVEL?"),
     "wiki_Skill_level", "5 | Nightmare!")
fact("skill", "SKILL FOUR", Q("ULTRA VIOLENCE SKILL?", "ULTRA VIOLENCE SKILL NUMBER?", "ULTRA VIOLENCE IS SKILL?", "ULTRA VIOLENCE SKILL LEVEL?", "ULTRA VIOLENCE LEVEL NUMBER?"),
     "wiki_Skill_level", "4 | Ultra-Violence.")
fact("skill", "SKILL THREE", Q("HURT ME PLENTY SKILL?", "HURT ME PLENTY SKILL NUMBER?", "HURT ME PLENTY IS SKILL?", "HURT ME PLENTY SKILL LEVEL?", "HURT ME PLENTY LEVEL NUMBER?"),
     "wiki_Skill_level", "3 | Hurt me plenty.")
fact("skill", "SKILL TWO", Q("NOT TOO ROUGH SKILL?", "NOT TOO ROUGH SKILL NUMBER?", "NOT TOO ROUGH IS SKILL?", "NOT TOO ROUGH SKILL LEVEL?", "NOT TOO ROUGH LEVEL NUMBER?"),
     "wiki_Skill_level", "2 | Hey, not too rough.")
fact("skill", "SKILL ONE", Q("TOO YOUNG TO DIE SKILL?", "TOO YOUNG TO DIE LEVEL?", "TOO YOUNG TO DIE IS SKILL?", "TOO YOUNG TO DIE NUMBER?", "WHAT IS THE EASIEST SKILL?", "EASIEST SKILL LEVEL?"),
     "wiki_Skill_level", "1 | I'm too young to die.")
fact("skill", "DOUBLE", Q("NIGHTMARE AMMO?", "AMMO ON NIGHTMARE?", "HOW MUCH AMMO ON NIGHTMARE?", "NIGHTMARE AMMO PICKUPS?"),
     "wiki_Skill_level", "sk_nightmare | M_NMARE | The same as in sk_hard | Double")
fact("skill", "RESPAWN", Q("WHAT DO NIGHTMARE MONSTERS DO?", "NIGHTMARE MONSTERS?", "NIGHTMARE MONSTERS DO WHAT?", "DO NIGHTMARE MONSTERS RETURN?", "NIGHTMARE MONSTER BEHAVIOR?", "NIGHTMARE MONSTER TRAIT?"),
     "wiki_Skill_level", "No cheats 5 , fast 6 and respawning monsters")
fact("skill", "IDCLEV AND IDDT", Q("NIGHTMARE CHEATS THAT WORK?", "WHICH CHEATS WORK ON NIGHTMARE?", "CHEATS ALLOWED IN NIGHTMARE?", "NIGHTMARE CHEAT EXCEPTIONS?"),
     "wiki_Skill_level", "All cheat codes except idclev and iddt are disabled")
fact("skill", "HALF DAMAGE", Q("TOO YOUNG TO DIE DAMAGE?", "DAMAGE ON THE EASIEST SKILL?", "EASIEST SKILL DAMAGE TAKEN?", "SKILL ONE DAMAGE TAKEN?"),
     "wiki_Skill_level", "Players take half normal damage")
fact("history", "SEPTEMBER", Q("DOOM TWO RELEASE MONTH?", "WHAT MONTH WAS DOOM TWO OUT?", "DOOM TWO CAME OUT WHICH MONTH?", "MONTH OF DOOM TWO'S RELEASE?"),
     "wiki_Doom_II:_Hell_on_Earth", "released by id Software on September 30, 1994")
fact("history", "THIRTY TWO", Q("HOW MANY LEVELS IN DOOM TWO?", "DOOM TWO LEVEL COUNT?", "NUMBER OF DOOM TWO MAPS?", "HOW MANY DOOM TWO MAPS?"),
     "wiki_Doom_II:_Hell_on_Earth", "Featuring 32 levels in a single linear episode")
fact("history", "SUPER SHOTGUN", Q("NEW WEAPON IN DOOM TWO?", "WHAT GUN DID DOOM TWO ADD?", "DOOM TWO'S NEW GUN?", "WHICH WEAPON IS NEW IN DOOM TWO"),
     "wiki_Doom_II:_Hell_on_Earth", "an additional weapon (the super shotgun)")
fact("history", "GT INTERACTIVE", Q("WHO DISTRIBUTED DOOM TWO?", "DOOM TWO DISTRIBUTOR?", "DOOM TWO WAS SOLD BY?", "WHICH FIRM SOLD DOOM TWO?"),
     "wiki_Doom_II:_Hell_on_Earth", "with cooperation of distributor GT Interactive")
fact("character", "NEVER NAMED", Q("DOOMGUY'S REAL NAME?", "WHAT IS DOOMGUY'S NAME?", "NAME OF THE DOOM MARINE?", "WHAT IS THE MARINE'S NAME?"),
     "wiki_Doomguy", "is a futuristic marine that is never specifically referred to by name")
fact("character", "DOOM SLAYER", Q("DOOM ETERNAL HERO NAME?", "DOOM ETERNAL PROTAGONIST?", "WHO IS THE ETERNAL HERO?", "ETERNAL HERO NAME?"),
     "wiki_Doomguy", "Doom Slayer in Doom (2016) and Doom Eternal")
fact("character", "ASSAULTED OFFICER", Q("WHY WAS DOOMGUY ON MARS?", "WHY IS THE MARINE ON MARS?", "WHY WAS THE MARINE ON PHOBOS?", "DOOMGUY'S BACKSTORY?"),
     "wiki_Doomguy", "he had assaulted a superior officer who ordered him to fire upon innocent civilians")
fact("episode", "SANDY PETERSEN", Q("WHO DESIGNED EPISODE TWO?", "EPISODE TWO DESIGNER?", "WHO MADE EPISODE TWO?", "WHO MADE THE SHORES OF HELL?"),
     "wiki_The_Shores_of_Hell", "All maps in this episode are credited to Sandy Petersen")
fact("episode", "DEIMOS", Q("WHERE IS EPISODE TWO SET?", "EPISODE TWO TAKES PLACE ON?", "WHICH MOON IS EPISODE TWO?", "WHERE ARE THE SHORES OF HELL?"),
     "wiki_The_Shores_of_Hell", "The episode is set on Deimos")
fact("episode", "THE SHORES OF HELL", Q("WHAT IS EPISODE TWO CALLED?", "EPISODE TWO NAME?", "NAME OF EPISODE TWO?", "SECOND EPISODE NAME?"),
     "wiki_The_Shores_of_Hell", "The Shores of Hell is the second episode in Doom")
fact("episode", "DEIMOS", Q("WHICH MARS MOON IS SMALLER?", "SMALLER MOON OF MARS?", "OUTERMOST MARS MOON?", "WHICH MOON IS OUTERMOST?"),
     "wiki_Deimos", "Deimos is the smaller and outermost of the two moons of the planet Mars")
fact("episode", "IT VANISHES", Q("WHAT HAPPENS TO DEIMOS?", "WHAT DID DEIMOS DO BEFORE DOOM?", "DEIMOS BEFORE EPISODE ONE?", "WHERE DID DEIMOS GO?"),
     "wiki_Deimos", "Deimos mysteriously vanishes from the sky")
fact("episode", "FEAR", Q("DEIMOS MEANS?", "WHAT DOES DEIMOS MEAN?", "MEANING OF DEIMOS?", "DEIMOS TRANSLATES AS?"),
     "wiki_Deimos", 'it may be translated as "fear", "dread" or "terror"')
fact("people", "DAVE TAYLOR", Q("IDDT IS NAMED AFTER WHO?", "WHO IS IDDT NAMED FOR?", "WHO IS THE IDDT CHEAT FOR?", "IDDT NAME ORIGIN?"),
     "wiki_Dave_Taylor", 'is named for him, as is the "IDDT" cheat code')
fact("people", "DAVE TAYLOR", Q("WHO CODED THE AUTOMAP?", "WHO WROTE THE CHEAT CODES?", "WHO WROTE THE STATUS BAR?", "DOOM SPACKLE CODER?"),
     "wiki_Dave_Taylor", "writing unglamorous but essential things such as the status bar, sound library integration, the automap, level transitions, cheat codes")
fact("people", "SPACKLE CODER", Q("WHAT DOES TAYLOR CALL HIMSELF?", "DAVE TAYLOR'S ROLE?", "TAYLOR'S SELF NICKNAME?", "DAVE TAYLOR IS THE WHAT?"),
     "wiki_Dave_Taylor", 'He describes himself as the " spackle coder" on Doom')
fact("people", "BUSINESS MANAGER", Q("WHAT WAS JAY WILBUR AT ID?", "JAY WILBUR'S ROLE?", "JAY WILBUR'S JOB?", "WILBUR'S ROLE AT ID?"),
     "wiki_Jay_Wilbur", "Jay Wilbur is the former business manager and co-owner of id Software")
fact("tech", "ID TECH FOUR", Q("DOOM THREE ENGINE?", "WHICH ID TECH IS DOOM THREE?", "DOOM THREE ENGINE NAME?", "ID TECH NUMBER OF DOOM THREE?"),
     "wiki_Doom_engine", "known as id Tech 4")
fact("tech", "ID TECH FIVE", Q("RAGE ENGINE?", "WHICH ID TECH IS RAGE?", "RAGE ENGINE NAME?", "ID TECH NUMBER OF RAGE?"),
     "wiki_Doom_engine", "id Tech 5 , which powered Rage")

# ----------------------------------------------------------------------------- build
EXTRA = {'ROCKET LAUNCHER AMMO?': ['ROCKET LAUNCHER USES WHAT?', 'WHAT DOES THE LAUNCHER USE?', 'WHAT AMMO FOR ROCKET GUN?'], 'SUPER SHOTGUN AMMO?': ['SSG AMMO?', 'WHAT DOES THE SSG SHOOT?'], 'SUPER SHOTGUN PELLETS?': ['SSG PELLETS?', 'SSG PELLET COUNT?'], 'SUPER SHOTGUN SHELL USE?': ['SHELLS PER SSG SHOT?', 'SSG SHELLS PER SHOT?', 'SSG SHELL USE?', 'SHELLS USED BY THE SSG?'], 'CELLS PER BFG SHOT?': ['BFG CELLS PER SHOT?', 'BFG SHOT COST?', 'HOW MANY CELLS DOES A BFG USE'], 'CYBERDEMON IMMUNITY?': ['CYBERDEMON IGNORES WHAT?', 'CYBERDEMON RESISTS WHAT?', 'WHAT HURTS NOT THE CYBERDEMON'], 'FASTEST MONSTER IN DOOM ONE?': ['FASTEST DOOM ONE MONSTER?', 'WHO RUNS FASTEST IN DOOM ONE'], 'WHO IS THE FLAMING SKULL?': ['FLAMING SKULL?', 'WHICH ONE IS THE FLAMING SKULL', 'FLOATING FLAMING SKULL?'], 'HELL KNIGHT IS A WEAK WHAT?': ['HELL KNIGHT IS WEAK BARON?', 'HELL KNIGHT BASED ON?', 'BIGGER BROTHER OF THE KNIGHT?'], 'HELL KNIGHT DEBUT GAME?': ['HELL KNIGHT FIRST GAME?', 'HELL KNIGHT FIRST IN?', 'WHICH GAME HAS HELL KNIGHTS?'], 'WHO INVENTED THE BARON?': ['WHO CONCEIVED THE BARON?', 'BARON IDEA BY WHO?', 'WHO CAME UP WITH BARONS'], 'CHAINGUNNER OFFICIAL NAME?': ['CHAINGUNNER MANUAL NAME?', 'CHAINGUNNER ALIAS?'], 'SHOTGUN GUY LOOT?': ['SHOTGUN GUY DROP?', 'SHOTGUN GUY DROPS?', 'SERGEANT DROPS WHAT?'], 'WHERE IS THE WOLFENSTEIN SS?': ['WHERE ARE SS OFFICERS?', 'WOLFENSTEIN SS LEVELS?', 'WHERE DO NAZIS APPEAR?'], 'KEEN APPEARS IN WHICH GAME?': ['KEEN IS IN WHICH DOOM?', 'COMMANDER KEEN GAME?', 'WHICH DOOM HAS KEEN?'], 'ICON OF SIN IS ON WHICH MAP': ['ICON OF SIN MAP?', 'FINAL BOSS MAP?', 'WHERE IS THE FINAL BOSS?'], 'WHAT KILLS THE FINAL BOSS?': ['WHAT DOES THE BOSS DIE FROM?', 'WHAT ENDS DOOM TWO?', 'WHAT DO YOU KILL AT THE END?'], 'BARON OF HELL CODE NAME?': ['BARON CODE NAME?', 'BARON SOURCE NAME?', 'BARON INTERNAL NAME?'], 'CYBERDEMON CODE NAME?': ['CYBER CODE NAME?', 'CYBERDEMON SOURCE NAME?', 'CYBERDEMON IN CODE?'], 'CACODEMON CODE NAME?': ['CACO CODE NAME?', 'CACODEMON SOURCE NAME?', 'CACODEMON IN CODE?'], 'LOST SOUL CODE NAME?': ['SKULL CODE NAME?', 'LOST SOUL SOURCE NAME?', 'LOST SOUL IN CODE?'], 'HELL KNIGHT CODE NAME?': ['KNIGHT CODE NAME?', 'HELL KNIGHT SOURCE NAME?', 'HELL KNIGHT IN CODE?'], 'CHAINGUNNER CODE NAME?': ['CHAINGUNNER SOURCE NAME?', 'CHAINGUNNER IN CODE?', 'GUNNER CODE NAME?'], 'SHOTGUN GUY CODE NAME?': ['SHOTGUN GUY SOURCE NAME?', 'SHOTGUN GUY IN CODE?', 'SERGEANT CODE NAME?']}
EXTRA.update({
    'ROCKET LAUNCHER KEY?': ['ROCKET GUN KEY?', 'ROCKET LAUNCHER NUMBER?', 'WHICH KEY IS THE LAUNCHER?'],
    'PLASMA RIFLE KEY?': ['PLASMA GUN KEY?', 'PLASMA RIFLE NUMBER?', 'WHICH KEY IS THE PLASMA GUN?'],
    'SOUL SPHERE DEBUT MAP?': ['FIRST SOUL SPHERE MAP?', 'SOUL SPHERE FIRST MAP?'],
    'BLUR SPHERE DEBUT MAP?': ['FIRST BLUR SPHERE MAP?', 'BLUR SPHERE FIRST MAP?'],
    'WHAT IS THE RAD SUIT FOR?': ['RAD SUIT USE?', 'RAD SUIT PROTECTS FROM?'],
    'MEGASPHERE ARMOR TYPE?': ['MEGASPHERE ARMOR?', 'ARMOR OF A MEGASPHERE?'],
    'ID FOUNDERS CAME FROM?': ['ID FOUNDERS CAME FROM WHERE?', 'ID FOUNDERS WORKED WHERE?'],
    'DOOM ENGINE ID NAME?': ['DOOM ENGINE IS ID TECH?', 'WHICH ID TECH IS DOOM?'],
    "ROMERO'S POST ID COMPANY?": ["ROMERO'S NEXT COMPANY?", 'WHAT DID ROMERO FOUND?'],
    'DOOM RELEASE MONTH?': ['DOOM CAME OUT WHICH MONTH?', 'WHICH MONTH WAS DOOM OUT?'],
    'ADRIAN CARMACK BUILT WHAT?': ['WHAT DID ADRIAN MODEL?', 'ADRIAN CARMACK SCULPTED WHAT?', 'ADRIAN CARMACK CLAY WORK?'],
    'SANDY PETERSEN DOOM LEVELS?': ['PETERSEN DOOM LEVELS?', 'DOOM LEVELS BY PETERSEN?', 'HOW MANY DOOM LEVELS BY SANDY'],
    'SANDY PETERSEN QUAKE LEVELS?': ['PETERSEN QUAKE LEVELS?', 'QUAKE LEVELS BY PETERSEN?', "PETERSEN'S QUAKE COUNT?", 'HOW MANY QUAKE LEVELS BY SANDY'],
    'PETERSEN LEFT ID FOR?': ['PETERSEN QUIT ID FOR?', 'PETERSEN LEFT FOR WHERE?'],
    "MCGEE'S OWN GAME?": ['MCGEE OWN GAME?', "MCGEE'S ALICE GAME?", "WHICH GAME IS MCGEE'S?"],
    'SPIDER MASTERMIND PAIN ODDS?': ['MASTERMIND PAIN CHANCE?', 'MASTERMIND FLINCH CHANCE?', 'MASTERMIND PAIN ODDS?'],
    'E FOUR LEVEL NAMES SOURCE?': ['E FOUR NAME SOURCE?', 'E FOUR TITLES SOURCE?'],
    'WHO LISTED THE E FOUR NAMES?': ['WHO LISTED E FOUR NAMES?', 'E FOUR NAME LIST BY?'],
    'E FOUR FIRST TIME DESIGNER?': ['E FOUR DEBUT DESIGNER?', 'WHOSE FIRST WAS E FOUR?'],
    'CHAINSAW INSPIRATION?': ['SAW INSPIRATION?', 'CHAINSAW MOVIE ORIGIN?', 'MOVIE BEHIND THE CHAINSAW?'],
})


def norm(t: str) -> str:
    t = t.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"\s+([,.;:!?)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"\s+('s)\b", r"\1", t)
    t = re.sub(r"(\w)\s+'(\s)", r"\1'\2", t)
    return re.sub(r"\s+", " ", t).strip()


_TXT: dict[str, str] = {}
_URLS: dict[str, str] = {}


def _load(key: str):
    """Text and URL of a source, both taken from the SAME cached fetch (so the evidence check and the cited URL agree)."""
    import oracle_fetch
    if key.startswith("id_"):
        text, _ = oracle_fetch.wbfetch.fetch(oracle_fetch.ID_RAW + key[3:])
        url = oracle_fetch.ID_RAW + key[3:]
    else:
        html, url = oracle_fetch.wiki(key[5:])
        text = oracle_fetch.clean_html(html)
    _TXT[key], _URLS[key] = norm(text), url


def source_text(key: str) -> str:
    if key not in _TXT:
        _load(key)
    return _TXT[key]


def source_url(key: str) -> str:
    """URL that was opened for `key` (the archive.org capture for DoomWiki pages)."""
    if key not in _URLS:
        _load(key)
    return _URLS[key]


def check_fact(f: dict) -> list[str]:
    errs = []
    evs = f["ev"] if isinstance(f["ev"], list) else [f["ev"]]
    f["sources"] = [f["src"]]
    f["evidence"] = []
    for e in evs:
        if isinstance(e, tuple) and e[0] == "mobj":
            _, mt, field, val = e
            if int(MOBJ[mt][field]) != val:
                errs.append(f"mobj mismatch {e}")
            f["evidence"].append(f"info.c {mt} {field} = {val}")
            continue
        key, sub = e if isinstance(e, tuple) else (f["src"], e)
        if key not in f["sources"]:
            f["sources"].append(key)
        try:
            ok = norm(sub) in source_text(key)
        except FileNotFoundError as ex:
            errs.append(str(ex))
            continue
        if not ok:
            errs.append(f"evidence not found in {key}: {sub!r}")
        f["evidence"].append(f"[{key}] {sub}")
    return errs


def build(final_ids: list[int] | None = None):
    problems = []
    seen_q: dict[str, int] = {}
    keep = []
    for i, f in enumerate(FACTS):
        errs = check_fact(f)
        qs = list(dict.fromkeys(list(f["qs"]) + [x for k in f["qs"] for x in EXTRA.get(k, [])]))
        for q in qs:
            if not set(q) <= ALPHABET:
                errs.append(f"bad characters in {q!r}")
        qs = [q for q in qs if len(q) <= QMAX]       # over-long paraphrases are dropped
        if len(f["answer"]) > AMAX or not set(f["answer"]) <= ALPHABET:
            errs.append(f"bad answer {f['answer']!r}")
        if len(qs) < MIN_Q:
            errs.append(f"only {len(qs)} questions: {qs}")
        for q in qs:
            if q in seen_q:
                errs.append(f"duplicate question {q!r} (fact {seen_q[q]})")
            seen_q[q] = i
        if errs:
            problems.append((i, f["cat"], f["answer"], errs))
        else:
            f["qs"] = qs
            keep.append(f)
    return keep, problems


SPLIT_SEED = 20240611
VOCAB = "$ ABCDEFGHIJKLMNOPQRSTUVWXYZ'!?>"


# The model has ~25k integer weights.  With all 349 verified facts it cannot even memorise the training paraphrases
# (see ORACLE_REPORT notes), so oracle.json holds a category-stratified random subset of the verified pool;
# the full pool is written to oracle_pool.json.  Quotas are per category (fact counts).
QUOTA = {"health": 17, "speed": 0, "painchance": 0, "ammo": 12, "weapon": 9, "weapon-slot": 5, "item": 22, "message": 3,
         "engine": 1, "code-name": 6, "story": 5, "cheat": 14, "level": 22, "level-rev": 3, "monster": 24, "episode": 12,
         "history": 12, "people": 12, "skill": 5, "character": 3, "tech": 2}
SELECT_SEED = 4242


def select(keep: list[dict]) -> list[dict]:
    import random
    rnd = random.Random(SELECT_SEED)
    chosen = set()
    for cat, n in QUOTA.items():
        idx = [i for i, f in enumerate(keep) if f["cat"] == cat]
        chosen.update(rnd.sample(idx, min(n, len(idx))))
    missing = {f["cat"] for f in keep} - set(QUOTA)
    assert not missing, f"categories without quota: {missing}"
    return [f for i, f in enumerate(keep) if i in chosen]


def write_outputs(keep: list[dict], name: str = "oracle"):
    import random
    from collections import defaultdict
    rnd = random.Random(SPLIT_SEED)
    facts = []
    for n, f in enumerate(keep):
        held = rnd.choice(f["qs"])
        urls = []
        for key in f["sources"]:
            u = source_url(key)
            if u not in urls:
                urls.append(u)
        facts.append(dict(id=n, category=f["cat"], answer=f["answer"], questions=f["qs"], heldout=held,
                          source_urls=urls, evidence=f["evidence"]))
    T = max(len(q) + len(f["answer"]) + 2 for f in facts for q in f["questions"])
    out = dict(vocab=VOCAB, format="QUESTION>ANSWER", qmax=QMAX, amax=AMAX, T=T, split_seed=SPLIT_SEED,
               n_facts=len(facts), n_examples=sum(len(f["questions"]) for f in facts), facts=facts)
    (HERE / f"{name}.json").write_text(json.dumps(out, indent=1))
    if name != "oracle":
        return out

    by_url = defaultdict(list)
    for f in facts:
        by_url[f["source_urls"][0]].append(f)
    md = ["# Sources for `oracle.json`", "",
          f"{len(facts)} facts, {out['n_examples']} question paraphrases, produced by `make_oracle.py`.", "",
          "Every fact is tied to a source URL that `oracle_fetch.py` opened, and to an *evidence* string that `make_oracle.py` "
          "re-checks against the fetched text before the fact is emitted (build aborts otherwise).", "",
          "* **id Software's released source** (https://github.com/id-Software/DOOM, `linuxdoom-1.10/`) is read through "
          "`raw.githubusercontent.com`: `info.c` (mobjinfo health/speed fields are parsed, not retyped), `d_englsh.h`, `st_stuff.c`, "
          "`p_inter.c`, `p_pspr.c`, `d_items.c`, `doomdef.h`, `p_local.h`.",
          "* **DoomWiki** (`doomwiki.org` answers 403 to this host) is read from the Internet Archive: "
          "`https://web.archive.org/web/<year>/https://doomwiki.org/wiki/<Title>` redirects to the nearest capture of that year; the exact "
          "capture URL that was opened (2024-2025 captures) is recorded as the source URL of each fact. The page text is cached on disk "
          "(`_cache/`), so `make_oracle.py` re-checks every evidence string against exactly the capture it cites.",
          "* Numbers are spelled out, text is upper-cased, characters outside A-Z space ' ! ? are not used. Answers that would need a digit, "
          "more than 18 characters or a '.' were not turned into facts.",
          "* Questions are paraphrases (3-6 per fact, <= 28 characters); one per fact (`heldout`, seed %d) is the validation example." % SPLIT_SEED,
          "* Where a fact combines several lines of source (e.g. `clipammo` x the `P_GiveAmmo(.., 5)` call of a box), every line is cited.",
          ""]
    for url, fs in sorted(by_url.items()):
        md.append(f"## {url}")
        md.append("")
        for f in fs:
            ev = " ; ".join(e.split("] ", 1)[-1] if e.startswith("[") else e for e in f["evidence"])
            extra = [u for u in f["source_urls"][1:]]
            md.append(f"* #{f['id']} **{f['answer']}** ← {f['questions'][0]}  \n  evidence: `{ev[:300]}`" +
                      (f"  \n  also: {', '.join(extra)}" if extra else ""))
        md.append("")
    (HERE / "ORACLE_SOURCES.md").write_text("\n".join(md))
    return out


if __name__ == "__main__":
    keep, problems = build()
    print(f"{len(FACTS)} facts defined, {len(keep)} valid, {len(problems)} with problems")
    for p in problems:
        print("PROBLEM", p)
    if problems and "--force" not in sys.argv:
        sys.exit(1)
    write_outputs(keep, "oracle_pool")
    out = write_outputs(select(keep))
    from collections import Counter
    print("facts", out["n_facts"], "examples", out["n_examples"], "T", out["T"], dict(Counter(f["category"] for f in out["facts"])))
