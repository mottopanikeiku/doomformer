# A counter machine inside one DOOM map

`df.Machine` compiles a program for a counter machine into one DOOM map. The registers and a one-hot program counter are
plane heights of sectors (state values, `pure.py`), voodoo dolls change them by crossing line specials, the program is the
geometry, and a halt latch ends the run. Python advances the game clock, reads the program counter and every register
after every step, and compares them with its own interpreter at tolerance 0.

```
doomformer machine multiply 7 9      # 63, 464 steps, every step checked
doomformer machine fib 20            # 6765 in four base-16 limbs, 196 steps
doomformer machine stored            # one map file, two programs loaded as initial state
doomformer machine multiply 7 9 --tamper 2     # change one line of the map; the trace diverges at the predicted step
doomformer machine list              # every ready program and its arguments
```

```python
import doomformer as df
m = df.Machine("""
top:  jzdec b done          ; b == 0 -> done, else b -= 1
      inc a
      jmp top
done: halt""", registers={"a": 3, "b": 4})
result = m.run()             # compile, play, check every step; MachineMismatch on the first differing step
result.registers             # {"a": 7, "b": 0}
```

## What is claimed, and what is not

Claimed, and checked by the runs below:

* The map holds the whole machine state (pc and registers) as plane heights, and its dolls execute every instruction:
  they read registers through gates, move the program counter and change registers through line specials. Python
  executes no instruction.
* For every program run, the pc, every register, their negated twins and every intermediate fire value of every step
  equal the Python interpreter's, at tolerance 0 (`play` raises `MachineMismatch` naming the step otherwise).
* Python sent only readout scripts and clock advances. `result.audit` counts them (`readout.LoadThenReadout`; the loads of
  the stored-program machine are listed separately).
* A data-dependent halt: the machine stops when the halt latch (the value pc_halt) is raised by a doll, and the level
  stops with it.

Not claimed:

* **The clock is not made of dolls.** The level's loop is an ACS script (OPEN script 900, `machine.controller_acs`). Per step
  it runs the rewind (script 3: teleports every doll home, restores the layer ports to their biases, closes the start
  barriers; state values stay), drops the start barriers (script 7), samples every doll's position every 8 tics until none
  moved, and holds the window while Python reads. One more thing the script reads: the halt latch's plane height, once per
  step, to leave the loop. It reads no register and decides nothing else. A doll-driven clock and rewind (teleport lines
  home, `Floor_MoveToValue` lines on every port, a self-restarting level) is not built. ZDoom's ACS is a full language,
  so this report does not say that dolls alone compute a program: dolls are the data path, program counter and
  instruction execution; ACS is the clock, the rewind and the halt check; Python is I/O.
* **It is a bounded machine.** Registers are plane heights, and a plane height only holds a bounded range (measured
  below); the program is bounded by the 32,000-unit map span (about 1,000 instructions); a finite-state machine in the
  strict sense. With unbounded counters the instruction set (`inc`, `jzdec`, `halt`) is Minsky's universal counter
  machine, so the machine is universal up to its memory bound, like any physical computer. Multi-limb registers push a
  register's range to base^limbs and never remove the bound. This is not a proof that DOOM is Turing complete.
* Self-modifying code, a stack, indirect addressing and subroutines are not built.
* It was run in ViZDoom 1.3 (ZDoom engine) only. counter.wad, conway.wad and the calculator below run in other ports
  (Boom, vanilla); this map does not.

## The construction

State values (`compile_pure(state=...)`): `pc0 ... pcN` (one-hot; `pcN` is the halted pc) and one value per register,
each with a negated twin. A value is a plane height: a floor port's floor, or a ceiling port's ceiling, with a tag. Dolls
add to it by crossing `Floor_RaiseByValue` / `Floor_LowerByValue` lines on that tag. State values are left alone by the
per-step rewind, every other value goes back to its bias.

One machine step is one pass through two layers:

| layer | dolls | what they do |
|---|---|---|
| decide | one per instruction (two for `jzdec`/`jge`/`jlt`, a corridor of base-1 gates for `addr`) | the lane gate reads pc_k >= 1, later gates read the registers the instruction tests, a crossed line raises a fire value (`f`, `z`, `n`, `t`, `u`, `b` + instruction number). Nothing here writes a state value |
| apply | one per fire, plus base-1 ADC lanes per `addr` | a fire >= 1 opens the lane; the doll crosses the lines that lower pc_k, raise pc_next and change the register |

Reads and writes are in different layers on purpose. Within a layer, the waves of dolls that share a target sector run one
after another, so in one layer the zero branch of `jzdec` could read `r` after the nonzero branch's doll had decremented it
and both branches would run. The layout never reads a value in the layer that writes it (this is a design argument; the
one-layer variant was not built to watch the race). The apply layer is a crossbar (ungated lanes): every doll crosses its
gate before any effect line is released, so the lane gates read the old state.

| instruction | effect | decide gates |
|---|---|---|
| `inc r` / `add r k` | r += 1 / k | pc_k >= 1 |
| `jzdec r L` | r == 0: goto L; else r -= 1 | pc_k >= 1, then r <= 0 or r >= 1 |
| `jge r k L`, `jlt r k L` | compare with a constant | r >= k, r <= k-1 |
| `jmp L`, `halt` (= `jmp` to pc_N) | | pc_k >= 1 |
| `addr a b` | a += b, 0 <= b <= base-1 | pc_k >= 1, then base-1 ceiling gates on b: a doll climbs as many as b allows and raises fire `b_k` once per gate; the apply layer reads that copy through base-1 lanes, each adding 1 to a |

`addr` is what makes multi-limb numbers fast. A register declared with `limbs={"a": 4}` is stored as a.0 ... a.3 (base 16
by default). The macro `bigadd A B` is one `addr` per limb, then for each limb but the top `jlt a.i base skip; add a.i
-base; add a.{i+1} 1` for the carry. `addr a a` doubles a limb: the decide layer reads the copy before the apply layer
changes it. The interpreter raises on an `addr` source outside 0..base-1, so a program that overflows its top limb is
refused instead of silently wrong. Macros `clear`, `move s d`, `copy s d t` expand to the instructions above.

Two checks the runner makes at every step, besides the values: the readout of a window is taken between two reads of the
motion script (no doll moved) and two reads of the level's status (still the same HOLD), and after the halt the level
must reach phase DONE with the last pass number, with nothing moving afterwards.

## The stored-program machine (`df.StoredMachine`, `doomformer/stored.py`)

The program is data. One map has `slots` instruction slots and `registers` registers; the compiled WAD is the same bytes
for every program. A slot's instruction is a few one-hot flag values: op (`inc` or `dec`), register, target pc. The host
writes the program (and the registers' initial values) as plane heights before the first tic, through
`readout.LoadThenReadout.load`, which sends the map's own mover script (`puke -1 tag height neg`) and is sealed before the
run; after that the host sends readout scripts and clock advances only. `result.audit["loaded"]` counts the writes.

Instructions: `inc r L` (r += 1, goto L), `jmp L`, `dec r L` (r >= 1: r -= 1, goto L; else the next slot), `halt`. Three
layers per step: fetch (the doll of the active slot copies its set flags into `cur_*` values), decide (from `cur_*` and the
registers: which register changes, which pc is next), apply (as above). Two programs on one map (`doomformer machine
stored`, `tests/test_machine.py`): `multiply` (c = a * b) and `power` (c = 2^a). The run checks the sha256 of the map
before and after each program and prints that they are equal.

What is data here, and what is not: the program and the registers are initial state that the host writes; the machine
cannot rewrite its own instruction memory (no instruction writes a flag), so self-modifying code does not exist yet.

## What Python does, what the map does

| | Python | ACS level script | dolls and lines |
|---|---|---|---|
| program | assembles it, compiles the map (ROM: program is geometry) or writes flags (stored) | none | ROM: the doll of each instruction exists only if the instruction does |
| registers, pc | reads them, compares | none | hold and change them |
| each step | advances tics, runs readout scripts | rewind, barrier drop, settle sampling, hold | decode, test, branch, update |
| halt | notices that pc_N is 1 and the level reached DONE | reads the latch, leaves the loop | raise the latch |

## Measured

All runs: ViZDoom 1.3 headless, one game each, up to 3 games at once while the machine's load average was 25 to 45, so
wall times are slow and unreliable; game tics are exact and independent of load. `doomformer machine ARGS`, logs in the
repository's work directory. Dolls include the two layers' clock dolls. A value is one number read back and compared.

| command | steps | dolls | sectors | tics per step | game tics | wall | values checked | result |
|---|---|---|---|---|---|---|---|---|
| `machine add 3 4` | 14 | 20 | 274 | 208 | 3,832 | 1 s | 360 | 7 |
| `machine multiply 7 9` | 464 | 39 | 578 | 216 | 126,340 | 17 s | 24,180 | 63 |
| `machine power 2 6` | 1,475 | 56 | 858 | 216 | 402,818 | 66 s | 115,128 | 64 |
| `machine gcd 48 18` | 647 | 73 | 1,122 | 224 | 181,950 | 37 s | 62,208 | 6 |
| `machine fib 20` (4 limbs, base 16) | 196 | 232 | 4,347 | 368 | 83,633 | 106 s | 33,884 | 6765 |
| `machine collatz 7` | 1,080 | 70 | 1,055 | 224 | 303,616 | 76 s | 97,290 | 16 Collatz steps |
| `machine pow2 40` (11 limbs, base 16) | 942 | 324 | 6,086 | 376 | 410,859 | 935 s | 231,978 | 1,099,511,627,776 |
| `machine primes 12` | 4,042 | 163 | 2,536 | 248 | 1,241,039 | 897 s | 865,202 | 5 primes |
| `machine stored`, program 1: multiply 2 * 3 | 43 | 511 | 8,984 | 560 | 27,053 | 89 s | 24,904 | 6 |
| `machine stored`, program 2: power 2 ^ 3 | 67 | 511 | 8,984 | 560 | 42,060 | 155 s | 38,488 | 8 |

Every row: 0 forbidden commands, halt latch stopped the level, the step count is the interpreter's. The first nine rows
have 0 state loads (arguments are baked into the map); the two `stored` rows are one map file (sha256 `120833f086cdfe68...`, equal before and
after both runs) with 34 state loads each, written before the first tic and then sealed.

### Negative controls

`machine multiply 7 9 --tamper 2` rewrites the apply-layer line of `inc c` (Floor_RaiseByValue amount 1 -> 2) in the compiled
WAD's TEXTMAP. The interpreter says the first step that executes that line is step 3; DOOM's first differing value is
step 3 (`r2: DOOM 2 != reference 1`). `--tamper 1 --tamper-what target` retargets the line that raises the next pc of `jzdec b
restore` to pc_halt: predicted step 38, first difference at step 38 (`pc5: 0 != 1, pc9: 1 != 0`). The untampered twin passes.
`tests/test_machine.py` runs both. In the stored machine, a map loaded with program B and checked against program A's
trace fails at step 0.

### Limits found

* **Register range.** The machine enforces |v| <= 12,000 (`tile.PORT_MAX`; the interpreter raises beyond it). Probe maps
  that lift the check (one register, `add a k` steps, every step compared): exact at 15,949 and at -15,949; at 15,950
  DOOM reads 6,614 or 6,654 instead (steps of 50 or of 10; `add a 500` from 15,500 fails at 16,000 with 6,214), and the
  negative side fails the same way (its twin reaches +15,950). Starting at +-16,000 and moving toward zero is exact.
  The cause is not investigated; the sector ceilings of the ports sit at 16,000. A register of L limbs in base b holds up
  to b^L - 1 (the `addr` source limbs must stay below b).
* **Doll and sector cost per instruction** (`layout_only` counting of K copies of one instruction, base 16 unless stated; dolls
  include the observer doll that gives each pc value a readout port):

  | instruction | dolls | sectors |
  |---|---|---|
  | `inc`, `add`, `jmp` | 3 | 43 to 46 |
  | `jzdec`, `jge`, `jlt` | 5 | 82 to 85 |
  | `addr`, base 8 | 10 | 183 |
  | `addr`, base 16 | 18 | 343 |

* **Program size.** `inc` chains of K instructions: 100 give 309 dolls and 4,696 sectors, 1,000 give 3,009 dolls and 46,106
  sectors (span 31,918 x 21,224), 1,200 still fit (55,308 sectors, 24,632 high), 1,300 do not (36,728 high, past the
  32,000-unit limit of `docs/engine_limits.md`). Only the layouts were counted; the largest program played in DOOM here
  has 44 instructions (`pow2 40`, 324 dolls) and the stored machine 12 slots (511 dolls, 8,984 sectors).
* **Tics per step.** One step is one pass of the level's loop: 208 tics for a 4-instruction program, 216 to 224 for
  programs of 9 to 19 instructions. Dolls that change the same sector run in successive waves, so every instruction that
  changes the busiest register or pc adds about 4.4 tics: a chain of K `inc a` takes 224 tics per step at K = 10, 312 at 30, 448 at 60.
  Each gate of an `addr` corridor adds about 8 tics: `pow2 6` takes 224 to 232 tics per step in base 4, 272 in base 8 and 352
  in base 16. The stored machine takes 560 tics per step (three layers, 511 dolls) and 2 to 2.3 s of wall time per step on a
  loaded machine (89 s for 43 steps, 155 s for 67), against 0.04 s per step for the 39-doll multiply.
* **Steps.** There is no bound in the map other than the level's loop count (1,000,000 passes, compiled in). The longest run
  played so far is 4,042 (`primes 12`, 1,241,039 tics) steps, exact. Wall time is steps times the tics per step divided by the engine's speed:
  4,000 to 7,400 tics per second for the small maps measured here (402,818 tics in 66 s for `power 2 6`, 303,616 in 76 s
  for `collatz 7`), 270 to 300 for the 511-doll stored machine, all on a heavily loaded 16-core machine.
* **Reading cost.** The runner reads every exposed value after every step; the stored machine exposes 566 values per step
  (10 readout pages). That readout is what makes each step checked, and part of the wall time.


## The network decides control flow

`doomformer demo --one-map --loop --until-eos --prompt DAN` (`doomformer/selfhalt.py`, details in HOW_IT_WORKS.md "Stopping at
EOS") applies the same halt-latch idea to the generation loop: the winner doll of each writing position also sets a per-position
latch (1 for the end-of-text token, 0 for every other winner), the level reads the latch of the newest position after each pass and
leaves its loop after the pass that wrote EOS, so the pass count is data. Prompts of one length run on one map file (the prompt
tokens are written as initial state through `LoadThenReadout`, then sealed). Checked: `tests/test_selfhalt.py` runs one random-weight
model (T = 4, V = 5) with two prompts on one sha256: one stops at EOS after pass 1 of 2, the other runs to its bound, both texts
equal `intref.generate`; a miswired latch (one EOS winner writes 0) fails on `halt:1` (DOOM 0, reference 1). The shipped
`doomnames_onemap` ran one pass with prompt DAN (4,000 tics, 237 s on a loaded machine, 4,482 dolls, 3,020 values exact, stopped at
EOS after pass 1 of 2). A whole real name was not run. The ACS script still reads one plane height per pass; the dolls decide its value.

## Compared with earlier DOOM computation

Pages opened for this comparison (each fact below is on the page or in a WAD file parsed by the author of the research
notes; what was not verified is marked):

| work | what is computed in the map | outside the map | program | universality claim |
|---|---|---|---|---|
| [counter.wad](https://www.doomworld.com/idgames/levels/doom2/Ports/a-c/counter), fraggle, 2006 | a 4-bit binary ripple counter from voodoo dolls and conveyors, 34 sectors, 134 linedefs (own parse of the file) | a switch line; how it is clocked is not described on any page read | fixed wiring | none |
| [conway.wad](https://github.com/pwiecz/conway.wad), Piotr Wieczorek, 2022 | Game of Life on a 13 x 13 torus: mancubi hold cells, barrels on scrolling floors count neighbours through teleporters, local handshakes replace a clock | the player sets the start cells and pulls a switch | the rule is geometry, the start state is data | none in the repo; the author calls it a "limited simulation" |
| [doom-calculator](https://github.com/nukep/doom-calculator), Danny Spencer, 2022 | adds two 2-digit decimal numbers (BCD) with binary decision diagrams of monsters, teleporters and doors, vanilla Doom 2 v1.9 | the player presses a switch per digit; one shot, no reset | geometry generated from BDDs | he writes that it "is not Turing complete" (monsters cannot be reset) |
| [half-adder](https://calabi-yau.space/blog/doom.html), Jared Candelaria, 2021 | OR, AND, NOT and REPEATER gates and one half-adder from monsters and lifts | the player, a clock of lifts | geometry | "DOOM is Turing-complete", argued from gate completeness; a NAND and a CPU are argued, not built |
| this | a counter machine: program counter and registers as plane heights, instructions executed by dolls, a program that loops and branches on data | the clock and rewind are an ACS script; Python reads and compares | geometry (ROM) or initial state (stored) | bounded machine; universal only up to its memory bound |

counter.wad and conway.wad run on Boom-compatible ports and the calculator on vanilla Doom 2; this map needs ViZDoom's ZDoom
engine and ACS. The prior works have no readback interface: their outputs are displays the player looks at. The nearest in
spirit is conway.wad, which also repeats a round inside the map; its repeat comes from a teleport graph, ours from an ACS loop.
The pages did not show a program that selects among instructions in any of them. Not verified: counter.wad's clocking and how its
bits toggle (the map was not run), the content of the counter and Doom-in-Doom videos. Doom-in-Doom (kgsws) runs machine code
through a code-execution exploit and is not map computation. A search for a programmable computer inside a DOOM map found none; that is not a proof of absence.

## Run it yourself

```
pip install doomformer
doomformer machine multiply 7 9
doomformer machine fib 20
doomformer machine stored
pytest tests/test_machine.py                  # about 25 s
```

`Machine(program, registers=..., limbs=..., base=16)`; `.reference()` is the interpreter's trace, `.compile(wad)` writes the map,
`.run(wad=None, max_steps=100_000)` runs and checks. `machine.tamper(wad, program, instruction, what="weight"|"target")` is the
negative control. `compile(opt=...)` passes the scheduling level of `pure.compile_pure` through. Measured: `multiply 3 4` and `pow2 6` run exactly at
opt 0 and opt 1 with the same tics per step (216 and 352); opt 2 and 3 need `tuning.json`, which was not present, so they
were not run.
