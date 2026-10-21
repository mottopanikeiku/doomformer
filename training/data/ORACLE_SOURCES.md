# Sources for `oracle.json`

189 facts, 829 question paraphrases, produced by `make_oracle.py`.

Every fact is tied to a source URL that `oracle_fetch.py` opened, and to an *evidence* string that `make_oracle.py` re-checks against the fetched text before the fact is emitted (build aborts otherwise).

* **id Software's released source** (https://github.com/id-Software/DOOM, `linuxdoom-1.10/`) is read through `raw.githubusercontent.com`: `info.c` (mobjinfo health/speed fields are parsed, not retyped), `d_englsh.h`, `st_stuff.c`, `p_inter.c`, `p_pspr.c`, `d_items.c`, `doomdef.h`, `p_local.h`.
* **DoomWiki** (`doomwiki.org` answers 403 to this host) is read from the Internet Archive: `https://web.archive.org/web/<year>/https://doomwiki.org/wiki/<Title>` redirects to the nearest capture of that year; the exact capture URL that was opened (2024-2025 captures) is recorded as the source URL of each fact. The page text is cached on disk (`_cache/`), so `make_oracle.py` re-checks every evidence string against exactly the capture it cites.
* Numbers are spelled out, text is upper-cased, characters outside A-Z space ' ! ? are not used. Answers that would need a digit, more than 18 characters or a '.' were not turned into facts.
* Questions are paraphrases (3-6 per fact, <= 28 characters); one per fact (`heldout`, seed 20240611) is the validation example.
* Where a fact combines several lines of source (e.g. `clipammo` x the `P_GiveAmmo(.., 5)` call of a box), every line is cited.

## https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_englsh.h

* #32 **FOUR** ← SHELLS IN A SHELL PICKUP?  
  evidence: `#define GOTSHELLS "Picked up 4 shotgun shells."`
* #48 **FIND SOME MEAT!** ← CHAINSAW PICKUP MESSAGE?  
  evidence: `#define GOTCHAINSAW "A chainsaw! Find some meat!"`
* #66 **HANGAR** ← WHAT IS E ONE M ONE?  
  evidence: `#define HUSTR_E1M1	"E1M1: Hangar"`
* #67 **PHOBOS ANOMALY** ← WHAT IS E ONE M EIGHT?  
  evidence: `#define HUSTR_E1M8	"E1M8: Phobos Anomaly"`
* #68 **CONTAINMENT AREA** ← WHAT IS E TWO M TWO?  
  evidence: `#define HUSTR_E2M2	"E2M2: Containment Area"`
* #69 **COMMAND CENTER** ← WHAT IS E TWO M FIVE?  
  evidence: `#define HUSTR_E2M5	"E2M5: Command Center"`
* #70 **SPAWNING VATS** ← WHAT IS E TWO M SEVEN?  
  evidence: `#define HUSTR_E2M7	"E2M7: Spawning Vats"`
* #71 **TOWER OF BABEL** ← WHAT IS E TWO M EIGHT?  
  evidence: `#define HUSTR_E2M8	"E2M8: Tower of Babel"`
* #72 **DIS** ← WHAT IS E THREE M EIGHT?  
  evidence: `#define HUSTR_E3M8	"E3M8: Dis"`
* #73 **WARRENS** ← WHAT IS E THREE M NINE?  
  evidence: `#define HUSTR_E3M9	"E3M9: Warrens"`
* #74 **HELL BENEATH** ← WHAT IS E FOUR M ONE?  
  evidence: `#define HUSTR_E4M1	"E4M1: Hell Beneath"`
* #75 **UNRULY EVIL** ← WHAT IS E FOUR M FOUR?  
  evidence: `#define HUSTR_E4M4	"E4M4: Unruly Evil"`
* #76 **AND HELL FOLLOWED** ← WHAT IS E FOUR M SEVEN?  
  evidence: `#define HUSTR_E4M7	"E4M7: And Hell Followed"`
* #77 **FEAR** ← WHAT IS E FOUR M NINE?  
  evidence: `#define HUSTR_E4M9	"E4M9: Fear"`
* #78 **ENTRYWAY** ← WHAT IS MAP ONE?  
  evidence: `#define HUSTR_1	"level 1: entryway"`
* #79 **TRICKS AND TRAPS** ← WHAT IS MAP EIGHT?  
  evidence: `#define HUSTR_8	"level 8: tricks and traps"`
* #80 **THE PIT** ← WHAT IS MAP NINE?  
  evidence: `#define HUSTR_9	"level 9: the pit"`
* #81 **THE INMOST DENS** ← WHAT IS MAP FOURTEEN?  
  evidence: `#define HUSTR_14	"level 14: the inmost dens"`
* #82 **GOTCHA!** ← WHAT IS MAP TWENTY?  
  evidence: `#define HUSTR_20	"level 20: gotcha!"`
* #83 **NIRVANA** ← WHAT IS MAP TWENTY ONE?  
  evidence: `#define HUSTR_21	"level 21: nirvana"`
* #84 **THE CATACOMBS** ← WHAT IS MAP TWENTY TWO?  
  evidence: `#define HUSTR_22	"level 22: the catacombs"`
* #85 **MONSTER CONDO** ← WHAT IS MAP TWENTY SEVEN?  
  evidence: `#define HUSTR_27	"level 27: monster condo"`
* #86 **THE LIVING END** ← WHAT IS MAP TWENTY NINE?  
  evidence: `#define HUSTR_29	"level 29: the living end"`
* #87 **ICON OF SIN** ← WHAT IS MAP THIRTY?  
  evidence: `#define HUSTR_30	"level 30: icon of sin"`
* #88 **E ONE M ONE** ← WHICH MAP IS HANGAR?  
  evidence: `#define HUSTR_E1M1	"E1M1: Hangar"`
* #89 **MAP SEVEN** ← WHICH MAP IS DEAD SIMPLE?  
  evidence: `#define HUSTR_7	"level 7: dead simple"`
* #90 **MAP THIRTY TWO** ← WHICH MAP IS GROSSE?  
  evidence: `#define HUSTR_32	"level 32: grosse"`
* #91 **DEIMOS** ← WHAT MOON FLOATS ABOVE HELL?  
  evidence: `Deimos floats above Hell itself!`
* #92 **SPIDERDEMON** ← WHO PLANNED THE INVASION?  
  evidence: `The loathsome spiderdemon that`
* #93 **HUMANS** ← WHO BUILT THE SECRET LEVEL?  
  evidence: `LOOKS LIKE IT'S BEEN BUILT BY`
* #94 **YOUR HOME CITY** ← WHERE IS THE ALIEN BASE?  
  evidence: `YOUR OWN HOME CITY, NOT FAR FROM THE`
* #95 **GATEKEEPER** ← PLUTONIA FINAL BOSS?  
  evidence: `The Gatekeeper's evil face is splattered`

## https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_items.c

* #17 **BULLETS** ← PISTOL AMMO?  
  evidence: `// pistol am_clip,`
* #18 **SHELLS** ← SHOTGUN AMMO?  
  evidence: `// shotgun am_shell,`
* #19 **BULLETS** ← CHAINGUN AMMO?  
  evidence: `// chaingun am_clip,`
* #20 **ROCKETS** ← ROCKET LAUNCHER AMMO?  
  evidence: `// missile launcher am_misl,`
* #21 **ENERGY CELLS** ← PLASMA RIFLE AMMO?  
  evidence: `// plasma rifle am_cell,`
* #22 **ENERGY CELLS** ← BFG AMMO?  
  evidence: `// bfg 9000 am_cell,`
* #23 **NO AMMO** ← CHAINSAW AMMO?  
  evidence: `// chainsaw am_noammo,`

## https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/doomdef.h

* #29 **NINE** ← HOW MANY WEAPONS IN DOOM?  
  evidence: `wp_supershotgun, NUMWEAPONS`
* #42 **THIRTY SECONDS** ← INVULNERABILITY LASTS?  
  evidence: `INVULNTICS = (30*TICRATE),`
* #43 **SIXTY SECONDS** ← INVISIBILITY LASTS?  
  evidence: `INVISTICS = (60*TICRATE),`
* #44 **SIXTY SECONDS** ← RADIATION SUIT LASTS?  
  evidence: `IRONTICS = (60*TICRATE)`
* #45 **TWO MINUTES** ← LIGHT AMP VISOR LASTS?  
  evidence: `INFRATICS = (120*TICRATE),`
* #47 **SIX** ← HOW MANY KEY TYPES?  
  evidence: `it_blueskull, it_yellowskull, it_redskull,`
* #51 **TWO HUNDRED** ← SCREEN HEIGHT?  
  evidence: `#define SCREENHEIGHT 200`

## https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/info.c

* #0 **TWENTY** ← ZOMBIEMAN HEALTH?  
  evidence: `info.c MT_POSSESSED spawnhealth = 20`
* #1 **THIRTY** ← SHOTGUN GUY HEALTH?  
  evidence: `info.c MT_SHOTGUY spawnhealth = 30`
* #2 **SEVENTY** ← CHAINGUNNER HEALTH?  
  evidence: `info.c MT_CHAINGUY spawnhealth = 70`
* #3 **SIXTY** ← IMP HEALTH?  
  evidence: `info.c MT_TROOP spawnhealth = 60`
* #4 **ONE HUNDRED FIFTY** ← DEMON HEALTH?  
  evidence: `info.c MT_SERGEANT spawnhealth = 150`
* #5 **ONE HUNDRED FIFTY** ← SPECTRE HEALTH?  
  evidence: `info.c MT_SHADOWS spawnhealth = 150`
* #6 **ONE HUNDRED** ← LOST SOUL HEALTH?  
  evidence: `info.c MT_SKULL spawnhealth = 100`
* #7 **FOUR HUNDRED** ← CACODEMON HEALTH?  
  evidence: `info.c MT_HEAD spawnhealth = 400`
* #8 **FOUR HUNDRED** ← PAIN ELEMENTAL HEALTH?  
  evidence: `info.c MT_PAIN spawnhealth = 400`
* #9 **FIVE HUNDRED** ← HELL KNIGHT HEALTH?  
  evidence: `info.c MT_KNIGHT spawnhealth = 500`
* #10 **ONE THOUSAND** ← BARON OF HELL HEALTH?  
  evidence: `info.c MT_BRUISER spawnhealth = 1000`
* #11 **SIX HUNDRED** ← MANCUBUS HEALTH?  
  evidence: `info.c MT_FATSO spawnhealth = 600`
* #12 **FIVE HUNDRED** ← ARACHNOTRON HEALTH?  
  evidence: `info.c MT_BABY spawnhealth = 500`
* #13 **SEVEN HUNDRED** ← ARCH VILE HEALTH?  
  evidence: `info.c MT_VILE spawnhealth = 700`
* #14 **FOUR THOUSAND** ← CYBERDEMON HEALTH?  
  evidence: `info.c MT_CYBORG spawnhealth = 4000`
* #15 **THREE THOUSAND** ← SPIDER MASTERMIND HEALTH?  
  evidence: `info.c MT_SPIDER spawnhealth = 3000`
* #16 **ONE HUNDRED** ← COMMANDER KEEN HEALTH?  
  evidence: `info.c MT_KEEN spawnhealth = 100`

## https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_inter.c

* #30 **FIFTY** ← MAX SHELLS?  
  evidence: `int maxammo[NUMAMMO] = {200, 50, 300, 50}`
* #31 **THREE HUNDRED** ← MAX CELLS?  
  evidence: `int maxammo[NUMAMMO] = {200, 50, 300, 50}`
* #33 **TWENTY** ← CELLS IN A CELL PICKUP?  
  evidence: `int clipammo[NUMAMMO] = {10, 4, 20, 1}; ; case SPR_CELL:`
* #34 **ONE HUNDRED** ← CELLS IN A CELL PACK?  
  evidence: `int clipammo[NUMAMMO] = {10, 4, 20, 1}; ; case SPR_CELP:`
* #35 **TEN** ← STIMPACK HEALS HOW MUCH?  
  evidence: `case SPR_STIM: if (!P_GiveBody (player, 10))`
* #36 **TWENTY FIVE** ← MEDIKIT HEALS HOW MUCH?  
  evidence: `case SPR_MEDI: if (!P_GiveBody (player, 25))`
* #37 **TWO HUNDRED** ← MAX HEALTH FROM ITEMS?  
  evidence: `if (player->health > 200) player->health = 200;`
* #38 **DOOM TWO** ← MEGASPHERE IS IN WHICH GAME?  
  evidence: `case SPR_MEGA: if (gamemode != commercial) return;`
* #39 **ONE HUNDRED** ← GREEN ARMOR POINTS?  
  evidence: `hits = armortype*100; ; case SPR_ARM1: if (!P_GiveArmor (player, 1))`
* #40 **TWO HUNDRED** ← BLUE ARMOR POINTS?  
  evidence: `hits = armortype*100; ; case SPR_ARM2: if (!P_GiveArmor (player, 2))`
* #41 **ONE** ← ARMOR BONUS GIVES?  
  evidence: `case SPR_BON2: player->armorpoints++; // can go over 100%`
* #46 **ONE HUNDRED** ← BERSERK HEALS UP TO?  
  evidence: `if (power == pw_strength) { P_GiveBody (player, 100);`
* #49 **SUPERCHARGE!** ← SOUL SPHERE MESSAGE?  
  evidence: `player->message = GOTSUPER;`
* #50 **MEGAARMOR** ← BLUE ARMOR PICKUP MESSAGE?  
  evidence: `case SPR_ARM2: ; player->message = GOTMEGA;`

## https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/p_pspr.c

* #24 **SEVEN** ← SHOTGUN PELLETS PER SHOT?  
  evidence: `for (i=0 ; i<7 ; i++) P_GunShot (player->mo, false);`
* #25 **TWO** ← SUPER SHOTGUN SHELL USE?  
  evidence: `ammo[weaponinfo[player->readyweapon].ammo]-=2;`
* #26 **FIVE TO FIFTEEN** ← PISTOL BULLET DAMAGE?  
  evidence: `damage = 5*(P_Random ()%3+1);`
* #27 **TWO TO TWENTY** ← CHAINSAW DAMAGE?  
  evidence: `damage = 2*(P_Random ()%10+1);`
* #28 **TEN TIMES** ← BERSERK FIST DAMAGE?  
  evidence: `if (player->powers[pw_strength]) damage *= 10;`

## https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/st_stuff.c

* #52 **GOD MODE** ← WHAT DOES IDDQD DO?  
  evidence: `// iddqd ; plyr->cheats ^= CF_GODMODE;`
* #53 **WEAPONS AND KEYS** ← WHAT DOES IDKFA DO?  
  evidence: `// idkfa ; plyr->cards[i] = true; ; #define STSTR_KFAADDED "Very Happy Ammo Added"`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_englsh.h
* #54 **WEAPONS NO KEYS** ← WHAT DOES IDFA DO?  
  evidence: `// idfa ; #define STSTR_FAADDED "Ammo (no keys) Added"`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_englsh.h
* #55 **NO CLIPPING** ← WHAT DOES IDCLIP DO?  
  evidence: `// idclip ; #define STSTR_NCON "No Clipping Mode ON"`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_englsh.h
* #56 **NO CLIPPING** ← WHAT DOES IDSPISPOPD DO?  
  evidence: `// idspispopd ; #define STSTR_NCON "No Clipping Mode ON"`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_englsh.h
* #57 **POWER UP MENU** ← WHAT DOES IDBEHOLD DO?  
  evidence: `// behold ; #define STSTR_BEHOLD "inVuln, Str, Inviso, Rad, Allmap, or Lite-amp"`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_englsh.h
* #58 **INVULNERABILITY** ← WHAT DOES IDBEHOLDV DO?  
  evidence: `// beholdv ; pw_invulnerability, pw_strength, pw_invisibility, pw_ironfeet, pw_allmap, pw_infrared,`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/doomdef.h
* #59 **BERSERK** ← WHAT DOES IDBEHOLDS DO?  
  evidence: `// beholds ; pw_invulnerability, pw_strength, pw_invisibility, pw_ironfeet, pw_allmap, pw_infrared,`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/doomdef.h
* #60 **INVISIBILITY** ← WHAT DOES IDBEHOLDI DO?  
  evidence: `// beholdi ; pw_invulnerability, pw_strength, pw_invisibility, pw_ironfeet, pw_allmap, pw_infrared,`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/doomdef.h
* #61 **RADIATION SUIT** ← WHAT DOES IDBEHOLDR DO?  
  evidence: `// beholdr ; pw_invulnerability, pw_strength, pw_invisibility, pw_ironfeet, pw_allmap, pw_infrared,`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/doomdef.h
* #62 **AUTOMAP** ← WHAT DOES IDBEHOLDA DO?  
  evidence: `// beholda ; pw_invulnerability, pw_strength, pw_invisibility, pw_ironfeet, pw_allmap, pw_infrared,`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/doomdef.h
* #63 **LIGHT AMP VISOR** ← WHAT DOES IDBEHOLDL DO?  
  evidence: `// beholdl ; pw_invulnerability, pw_strength, pw_invisibility, pw_ironfeet, pw_allmap, pw_infrared,`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/doomdef.h
* #64 **COORDINATES** ← WHAT DOES IDMYPOS DO?  
  evidence: `// idmypos ; sprintf(buf, "ang=0x%x;x,y=(0x%x,0x%x)"`
* #65 **LEVEL WARP** ← WHAT DOES IDCLEV DO?  
  evidence: `// idclev ; #define STSTR_CLEV "Changing Level..."`  
  also: https://raw.githubusercontent.com/id-Software/DOOM/master/linuxdoom-1.10/d_englsh.h

## https://web.archive.org/web/20221219064003/https://doomwiki.org/wiki/Backpack

* #140 **NONE** ← BACKPACK BEHOLD CODE?  
  evidence: `The backpack is the only powerup which does not have a corresponding idbehold cheat code`
* #141 **SIX HUNDRED** ← MAX CELLS WITH BACKPACK?  
  evidence: `600 energy cell units`

## https://web.archive.org/web/20230127050541/https://doomwiki.org/wiki/Spider_mastermind

* #124 **SUPER CHAINGUN** ← SPIDER MASTERMIND GUN?  
  evidence: `a super chaingun`

## https://web.archive.org/web/20231127074929/https://doomwiki.org/wiki/Deimos

* #184 **IT VANISHES** ← WHAT HAPPENS TO DEIMOS?  
  evidence: `Deimos mysteriously vanishes from the sky`

## https://web.archive.org/web/20231216013216/https://doomwiki.org/wiki/Knee-Deep_in_the_Dead

* #163 **MARS** ← PHOBOS IS A MOON OF WHAT?  
  evidence: `one of the moons of Mars`
* #164 **JOHN ROMERO** ← WHO DESIGNED EPISODE ONE?  
  evidence: `Designed mainly by John Romero`
* #165 **TECHBASE MAPS** ← FAN E ONE STYLE MAPS?  
  evidence: `are commonly referred to as techbase maps`
* #166 **BARONS** ← WHO GUARDS PHOBOS ANOMALY?  
  evidence: `defeats the powerful Barons who guard it`

## https://web.archive.org/web/20231216060350/https://doomwiki.org/wiki/Soul_sphere

* #134 **SOUL SPHERE** ← MANUAL NAME OF SUPERCHARGE?  
  evidence: `(called soul sphere in the manuals)`
* #137 **BLUE SPHERE** ← SOUL SPHERE COLOR?  
  evidence: `floating blue sphere with a glassy appearance`
* #142 **AN EXTRA LIFE** ← SOUL SPHERE ORIGINAL PLAN?  
  evidence: `collecting it would award an extra life`

## https://web.archive.org/web/20231217021530/https://doomwiki.org/wiki/The_Shores_of_Hell

* #183 **SANDY PETERSEN** ← WHO DESIGNED EPISODE TWO?  
  evidence: `All maps in this episode are credited to Sandy Petersen`

## https://web.archive.org/web/20231220181746/https://doomwiki.org/wiki/Dave_Taylor

* #185 **SPACKLE CODER** ← DAVE TAYLOR'S ROLE?  
  evidence: `He describes himself as the " spackle coder" on Doom`

## https://web.archive.org/web/20231221075653/https://doomwiki.org/wiki/Chainsaw

* #126 **ONE** ← CHAINSAW KEY?  
  evidence: `Weapon slot | 1`
* #131 **TOM HALL** ← WHO BORROWED THE CHAINSAW?  
  evidence: `borrowed by Tom Hall from his then-girlfriend`

## https://web.archive.org/web/20240214135021/https://doomwiki.org/wiki/Computer_area_map

* #139 **UNSEEN MAP LINES** ← COMPUTER MAP SHOWS WHAT?  
  evidence: `cause unexplored lines in the current level to be shown on the automap in gray`

## https://web.archive.org/web/20241205230020/https://doomwiki.org/wiki/Doom_II:_Hell_On_Earth

* #178 **SUPER SHOTGUN** ← NEW WEAPON IN DOOM TWO?  
  evidence: `an additional weapon (the super shotgun)`
* #179 **GT INTERACTIVE** ← WHO DISTRIBUTED DOOM TWO?  
  evidence: `with cooperation of distributor GT Interactive`

## https://web.archive.org/web/20241216145354/https://doomwiki.org/wiki/Shotgun

* #128 **THREE** ← SHOTGUN KEY?  
  evidence: `Weapon slot | 3`

## https://web.archive.org/web/20241217193157/https://doomwiki.org/wiki/Rocket_launcher

* #129 **FIVE** ← ROCKET LAUNCHER KEY?  
  evidence: `Weapon slot | 5`

## https://web.archive.org/web/20241217211332/https://doomwiki.org/wiki/John_Romero

* #150 **COLORADO** ← WHERE WAS ROMERO BORN?  
  evidence: `Born in Colorado, John Romero began programming`

## https://web.archive.org/web/20241217213257/https://doomwiki.org/wiki/Invulnerability

* #136 **INVERSE COLORS** ← INVULNERABILITY SCREEN LOOK?  
  evidence: `the colors in the player's view change to inverse monochrome`

## https://web.archive.org/web/20241218124519/https://doomwiki.org/wiki/Spectre

* #98 **DEMON** ← WHAT IS A SPECTRE BASED ON?  
  evidence: `exactly the same as the demon in behavior and attributes`

## https://web.archive.org/web/20241219083240/https://doomwiki.org/wiki/Demon

* #99 **PINKY** ← DEMON NICKNAME?  
  evidence: `have often since referred to the monster as pinky`
* #113 **SERGEANT** ← DEMON CODE NAME?  
  evidence: `Enum MT_SERGEANT`

## https://web.archive.org/web/20241219083329/https://doomwiki.org/wiki/Lost_soul

* #105 **SHORES OF HELL** ← LOST SOUL FIRST EPISODE?  
  evidence: `first introduced in Doom's second episode, The Shores of Hell`
* #106 **LOST SOUL** ← WHO IS THE FLAMING SKULL?  
  evidence: `The lost soul is a flaming, floating skull with horns`
* #115 **SKULL** ← LOST SOUL CODE NAME?  
  evidence: `Enum MT_SKULL`

## https://web.archive.org/web/20241219085828/https://doomwiki.org/wiki/Berserk

* #138 **GRAY MEDIKIT** ← BERSERK LOOKS LIKE WHAT?  
  evidence: `an item that looks like a dark gray medikit`

## https://web.archive.org/web/20241225133722/https://doomwiki.org/wiki/Pistol

* #127 **TWO** ← PISTOL KEY?  
  evidence: `Weapon slot | 2`
* #133 **BERETTA** ← PISTOL IS BASED ON WHAT?  
  evidence: `created from the Beretta 92FS pistol`

## https://web.archive.org/web/20241229031750/https://doomwiki.org/wiki/Hell_knight

* #116 **KNIGHT** ← HELL KNIGHT CODE NAME?  
  evidence: `Enum MT_KNIGHT`

## https://web.archive.org/web/20241229031750/https://doomwiki.org/wiki/Plasma_rifle

* #130 **SIX** ← PLASMA RIFLE KEY?  
  evidence: `Weapon slot | 6`
* #132 **PLASMA RIFLE** ← PLASMA GUN OTHER NAME?  
  evidence: `also known as the plasma rifle in the game manuals`

## https://web.archive.org/web/20241229031750/https://doomwiki.org/wiki/Shotgun_guy

* #118 **SHOTGUY** ← SHOTGUN GUY CODE NAME?  
  evidence: `Enum MT_SHOTGUY`

## https://web.archive.org/web/20241229031750/https://doomwiki.org/wiki/Zombieman

* #125 **A CLIP** ← WHAT DOES A ZOMBIEMAN DROP?  
  evidence: `the zombie will drop another clip each time it is killed again`

## https://web.archive.org/web/20241229130042/https://doomwiki.org/wiki/Cyberdemon

* #102 **ROCKETS** ← CYBERDEMON ATTACK?  
  evidence: `It attacks by firing barrages of three rockets`
* #103 **E TWO M EIGHT** ← CYBERDEMON FIRST MAP?  
  evidence: `first appears as a boss at the end of Doom level E2M8`
* #104 **BLAST DAMAGE** ← CYBERDEMON IMMUNITY?  
  evidence: `are immune to all blast damage`

## https://web.archive.org/web/20250102213337/https://doomwiki.org/wiki/Radiation_shielding_suit

* #135 **GREEN** ← RAD SUIT SCREEN TINT?  
  evidence: `fill their vision with a green haze`

## https://web.archive.org/web/20250102221712/https://doomwiki.org/wiki/Chaingunner

* #108 **HEAVY WEAPON DUDE** ← CHAINGUNNER OFFICIAL NAME?  
  evidence: `Heavy weapon dude data`
* #117 **CHAINGUY** ← CHAINGUNNER CODE NAME?  
  evidence: `Enum MT_CHAINGUY`

## https://web.archive.org/web/20250102222043/https://doomwiki.org/wiki/Cacodemon

* #100 **BEHOLDER** ← CACODEMON LOOKS LIKE WHAT?  
  evidence: `similar to that of the Beholder`
* #101 **GREEK** ← CACODEMON NAME COMES FROM?  
  evidence: `derived from the Greek`
* #114 **HEAD** ← CACODEMON CODE NAME?  
  evidence: `Enum MT_HEAD`

## https://web.archive.org/web/20250107155406/https://doomwiki.org/wiki/Baron_of_Hell

* #107 **TOM HALL** ← WHO INVENTED THE BARON?  
  evidence: `originally conceived as the "bruiser brothers" by Tom Hall`

## https://web.archive.org/web/20250107192820/https://doomwiki.org/wiki/Skill_level

* #173 **SKILL FIVE** ← WHICH SKILL IS NIGHTMARE?  
  evidence: `5 | Nightmare!`
* #174 **SKILL THREE** ← HURT ME PLENTY SKILL?  
  evidence: `3 | Hurt me plenty.`
* #175 **SKILL TWO** ← NOT TOO ROUGH SKILL?  
  evidence: `2 | Hey, not too rough.`
* #176 **RESPAWN** ← NIGHTMARE MONSTERS?  
  evidence: `No cheats 5 , fast 6 and respawning monsters`
* #177 **HALF DAMAGE** ← TOO YOUNG TO DIE DAMAGE?  
  evidence: `Players take half normal damage`

## https://web.archive.org/web/20250108092634/https://doomwiki.org/wiki/Tom_Hall

* #151 **TOM HALL** ← WHO WROTE THE DOOM BIBLE?  
  evidence: `is the author of the Doom Bible`

## https://web.archive.org/web/20250108093111/https://doomwiki.org/wiki/Inferno

* #167 **HELL** ← WHERE IS EPISODE THREE SET?  
  evidence: `This episode is set in Hell`
* #168 **DIVINE COMEDY** ← INFERNO IS NAMED AFTER?  
  evidence: `named after Inferno, part of Dante Alighieri's Divine Comedy`
* #169 **EPISODE THREE** ← WHICH EPISODE IS INFERNO?  
  evidence: `Inferno is the third episode in Doom`
* #170 **INFERNO** ← EPISODE THREE NAME?  
  evidence: `Inferno is the third episode in Doom`

## https://web.archive.org/web/20250108151712/https://doomwiki.org/wiki/Icon_of_Sin

* #110 **MAP THIRTY** ← ICON OF SIN IS ON WHICH MAP  
  evidence: `is the final boss of Doom II, in MAP30: Icon of Sin`
* #111 **ROMERO'S HEAD** ← WHAT KILLS THE FINAL BOSS?  
  evidence: `Romero's head is the thing that must be destroyed`
* #112 **DEMONS** ← ICON OF SIN SPAWNS WHAT?  
  evidence: `allows it to spawn endless scores of demons`

## https://web.archive.org/web/20250108165910/https://doomwiki.org/wiki/Doomguy

* #180 **NEVER NAMED** ← DOOMGUY'S REAL NAME?  
  evidence: `is a futuristic marine that is never specifically referred to by name`
* #181 **DOOM SLAYER** ← DOOM ETERNAL HERO NAME?  
  evidence: `Doom Slayer in Doom (2016) and Doom Eternal`
* #182 **ASSAULTED OFFICER** ← WHY WAS DOOMGUY ON MARS?  
  evidence: `he had assaulted a superior officer who ordered him to fire upon innocent civilians`

## https://web.archive.org/web/20250108200023/https://doomwiki.org/wiki/Sandy_Petersen

* #156 **SEVENTEEN** ← PETERSEN DOOM TWO LEVELS?  
  evidence: `He later created 17 of the levels for Doom II`
* #157 **SEVEN** ← SANDY PETERSEN QUAKE LEVELS?  
  evidence: `and seven levels for Quake`
* #158 **SEPTEMBER** ← WHEN DID PETERSEN JOIN ID?  
  evidence: `Petersen joined id Software in September 1993`
* #159 **CALL OF CTHULHU** ← PETERSEN'S FAMOUS RPG?  
  evidence: `co-authored the game Call of Cthulhu`
* #160 **ENSEMBLE STUDIOS** ← PETERSEN LEFT ID FOR?  
  evidence: `He left id Software for Ensemble Studios in June 1997`

## https://web.archive.org/web/20250108215003/https://doomwiki.org/wiki/Arch-vile

* #119 **ARCH VILE** ← WHO RESURRECTS MONSTERS?  
  evidence: `The arch-vile has the power to resurrect other, lower-level monsters`
* #120 **SUMMONS FIRE** ← ARCH VILE ATTACK?  
  evidence: `summoning fire upon the targeted foe`
* #121 **NINETY** ← ARCH VILE BLAST DAMAGE?  
  evidence: `causing up to 90 points in damage`

## https://web.archive.org/web/20250109015348/https://doomwiki.org/wiki/Wolfenstein_SS

* #109 **SECRET LEVELS** ← WHERE IS THE WOLFENSTEIN SS?  
  evidence: `makes an appearance in the secret levels of Doom II`

## https://web.archive.org/web/20250111064443/https://doomwiki.org/wiki/Thy_Flesh_Consumed

* #171 **KEVIN CLOUD** ← WHO LISTED THE E FOUR NAMES?  
  evidence: `The list was compiled by Kevin Cloud`
* #172 **TIM WILLITS** ← E FOUR FIRST TIME DESIGNER?  
  evidence: `This episode was also Tim Willits' first contribution to the Doom series`

## https://web.archive.org/web/20250112043338/https://doomwiki.org/wiki/Doom_engine

* #187 **ID TECH FOUR** ← DOOM THREE ENGINE?  
  evidence: `known as id Tech 4`
* #188 **ID TECH FIVE** ← RAGE ENGINE?  
  evidence: `id Tech 5 , which powered Rage`

## https://web.archive.org/web/20250114225344/https://doomwiki.org/wiki/American_McGee

* #161 **TECH SUPPORT** ← MCGEE STARTED AT ID AS?  
  evidence: `he was hired at id as tech support`
* #162 **ALICE** ← MCGEE'S OWN GAME?  
  evidence: `created his own game, American McGee's Alice`

## https://web.archive.org/web/20250115014828/https://doomwiki.org/wiki/Id_Software

* #143 **MESQUITE TEXAS** ← WHERE WAS ID FIRST BASED?  
  evidence: `based first in Mesquite, Texas`
* #144 **FEBRUARY** ← ID FOUNDING MONTH?  
  evidence: `founded on February 1, 1991 by four members`
* #145 **FOUR** ← HOW MANY ID FOUNDERS?  
  evidence: `by four members of the computer company Softdisk`
* #146 **SOFTDISK** ← ID FOUNDERS CAME FROM?  
  evidence: `by four members of the computer company Softdisk`
* #147 **IN DEMAND** ← WHAT DOES ID STAND FOR?  
  evidence: `the name "id" came from the phrase "in demand"`
* #148 **ID TECH ONE** ← DOOM ENGINE ID NAME?  
  evidence: `with the Doom engine as id Tech 1`

## https://web.archive.org/web/20250125224050/https://doomwiki.org/wiki/Jay_Wilbur

* #186 **BUSINESS MANAGER** ← WHAT WAS JAY WILBUR AT ID?  
  evidence: `Jay Wilbur is the former business manager and co-owner of id Software`

## https://web.archive.org/web/20250127104231/https://doomwiki.org/wiki/Pain_elemental

* #96 **LOST SOULS** ← PAIN ELEMENTAL ATTACK?  
  evidence: `the pain elemental periodically spews lost souls at its opponents`
* #97 **THREE LOST SOULS** ← WHAT HAPPENS WHEN A PE DIES?  
  evidence: `When killed, the pain elemental collapses in on itself and explodes, spawning three lost souls`

## https://web.archive.org/web/20251113021412/https://doomwiki.org/wiki/John_Carmack

* #149 **OCULUS** ← WHICH VR FIRM HIRED CARMACK?  
  evidence: `Oculus VR announced John Carmack as its new Chief Technology Officer`

## https://web.archive.org/web/20251220005833/https://doomwiki.org/wiki/Arachnotron

* #122 **DEAD SIMPLE** ← ARACHNOTRON DEBUT MAP?  
  evidence: `they make their first appearance in Dead Simple`
* #123 **SPIDER MASTERMIND** ← ARACHNOTRON IS A SMALL WHAT?  
  evidence: `It is structurally the same as the spiderdemon, but weaker, and much smaller in size`

## https://web.archive.org/web/20260114202951/https://doomwiki.org/wiki/Doom

* #152 **WISCONSIN** ← DOOM FIRST UPLOAD SITE?  
  evidence: `uploaded to an FTP server at the University of Wisconsin`
* #153 **THREE** ← EPISODES IN REGISTERED DOOM?  
  evidence: `containing all three episodes`
* #154 **DMX SOUND LIBRARY** ← WHY NO DOS DOOM SOURCE?  
  evidence: `because of licensing issues with the DMX sound library`
* #155 **OCTOBER** ← DOOM FILM RELEASE MONTH?  
  evidence: `a film released in October 2005`
