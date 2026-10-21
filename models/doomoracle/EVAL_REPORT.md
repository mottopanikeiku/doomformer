# doomoracle evaluation

Model `doomoracle`: L=2, d=32, H=4, dm=96, K=7/7/7, T=47, vocab `$ ABCDEFGHIJKLMNOPQRSTUVWXYZ'!?>`. Data: 189 facts; 640 training paraphrases, 189 held-out paraphrases (one per fact).

| check | result |
|---|---|
| `intref.check_limits` on all 829 sequences | `[]` |
| torch int64 forward == `intref.forward` | 829/829 sequences |
| exact answer, greedy `intref.generate(m, q + '>')`, TRAIN paraphrases | 606/640 = 94.7 % |
| exact answer, greedy, HELD-OUT paraphrases | 67/189 = 35.4 % |

## Held-out accuracy by category

| category | exact |
|---|---|
| ammo | 6/12 |
| character | 1/3 |
| cheat | 11/14 |
| code-name | 2/6 |
| engine | 0/1 |
| episode | 1/12 |
| health | 9/17 |
| history | 2/12 |
| item | 5/22 |
| level | 11/22 |
| level-rev | 3/3 |
| message | 1/3 |
| monster | 3/24 |
| people | 3/12 |
| skill | 3/5 |
| story | 2/5 |
| tech | 1/2 |
| weapon | 1/9 |
| weapon-slot | 2/5 |

## 15 held-out samples (random, seed 0)

| question | gold | model | |
|---|---|---|---|
| SPECTRE IS A FUZZY WHAT? | DEMON | TEN | WRONG |
| BARON IDEA BY WHO? | TOM HALL | RESAN | WRONG |
| BARON OF HELL HP? | ONE THOUSAND | ONE THOUSAND | ok |
| E ONE M ONE IS CALLED? | HANGAR | PINKYWO | WRONG |
| PLASMA RIFLE NUMBER? | SIX | FTHOUSE BUND | WRONG |
| MASTERMIND SHOOTS WITH WHAT? | SUPER CHAINGUN | RESESPAWN | WRONG |
| CYBERDEMON FIRST MAP? | E TWO M EIGHT | BLAST DAMAGE | WRONG |
| WHAT IS E FOUR M NINE? | FEAR | FEAR | ok |
| ARACHNOTRON DEBUT MAP? | DEAD SIMPLE | BLLETS | WRONG |
| WHAT MOON FLOATS ABOVE HELL? | DEIMOS | TWO HUNDRED | WRONG |
| WHICH VR FIRM HIRED CARMACK? | OCULUS | GREENIY T | WRONG |
| WHAT HAPPENS WITH IDCLIP? | NO CLIPPING | NO CLIPPING | ok |
| ROCKET GUN KEY? | FIVE | FIVE | ok |
| STIMPACK HEALTH? | TEN | TEN | ok |
| E THREE M EIGHT LEVEL NAME? | DIS | DIVEENTY | WRONG |

## Questions that are not in the data

| kind | question | model says |
|---|---|---|
| UNKNOWN FACT | WHAT DOES IDDT DO? | GOD MODE |
| UNKNOWN FACT | WHO COMPOSED THE MUSIC? | INE TENVERS |
| UNKNOWN FACT | WHAT IS E ONE M FIVE? | HANGAR |
| NEW PHRASING | HOW STRONG IS AN IMP? | BUNES MAND |
| NEW PHRASING | WHICH MAP HOLDS THE SPIDER? | BUPOWER MES |
| NEW PHRASING | WHAT GOD MODE CODE? | TOM HALL |
| OUT OF SCOPE | WHAT IS THE CAPITAL OF FRANCE? | WARRENS |
