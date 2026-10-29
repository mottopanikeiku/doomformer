# A counter machine and a SUBLEQ machine inside one DOOM map

`df.Machine` compiles a program for a counter machine into one DOOM map. The registers and a one-hot program counter are
plane heights of sectors (state values, `pure.py`), voodoo dolls change them by crossing line specials, the program is the
geometry, and a halt latch ends the run. Python advances the game clock, reads the program counter and every register
after every step, and compares them with its own interpreter at tolerance 0. `df.SubleqMachine` (`doomformer/subleq.py`)
is a stored-program machine in the same style: 18 memory cells hold instructions and data, addresses are read through
multiplexers of equality gates, and programs rewrite their own instructions. The level's loop is an ACS script by default;
`--clock dolls` replaces it, for the counter machine and for the SUBLEQ machine, by a doll and line specials.

```
doomformer machine multiply 7 9                # 63, 464 steps, every step checked
doomformer machine multiply 7 9 --clock dolls  # the same, with no OPEN script: a loop doll and line specials are the clock
doomformer machine multiply 7 9 --clock dolls --blind   # no command during the run, one read at the end
doomformer machine fib 20                      # 6765 in four base-16 limbs, 196 steps
doomformer machine subleq all                  # one map: add, multiply 7 x 9 and a self-modifying program, one sha256
doomformer machine subleq selfmod --tamper 13  # change one decode line of the map; the trace diverges at the predicted step
doomformer machine subleq models/subleq/loop.sq     # a 103-step loop, every step checked
doomformer machine subleq models/subleq/ownwrite.sq --latch   # a step the host refuses: the MAP raises its violation latch
doomformer machine stored                      # one map file, two programs loaded as initial state (inc/dec/jmp, no indirection)
doomformer machine multiply 7 9 --tamper 2     # change one line of the map; the trace diverges at the predicted step
doomformer machine list                        # every ready program and its arguments
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

* The map holds the whole machine state (pc and registers, or pc and memory cells) as plane heights, and its dolls execute
  every instruction: they read values through gates, move the program counter and change values through line specials.
  Python executes no instruction.
* For every program run, the pc, every register or memory cell, their negated twins and every intermediate value of every
  step equal the Python interpreter's, at tolerance 0 (`play` raises `MachineMismatch` naming the step otherwise).
* Python sent only readout scripts and clock advances. `result.audit` counts them (`readout.LoadThenReadout`; the initial
  state writes of the stored-program and SUBLEQ machines are listed separately and sealed before the first tic that matters).
* A data-dependent halt: the machine stops when the halt latch (the value pc_halt) is raised by a doll, and the level
  stops with it.
* **The SUBLEQ machine is a stored-program machine with indirect addressing and self-modifying code.** Instructions and
  data share 18 memory cells; an instruction names the cell it reads and the cell it writes by address, read and written
  through multiplexers of equality gates. Three programs (add, multiply 7 x 9, a self-modifying sum) and a fourth from a source
  file run on one compiled map (one sha256 per clock); the trace shows an instruction cell change and the next
  round use the new operand. Section "The SUBLEQ machine".
* **With `--clock dolls` the map has no OPEN script.** A doll and line specials restart the level, rewind it and stop it at
  the halt latch; the ACS lump holds the readout pages, the readout guard (script 9, which fills the readback slots with
  a poison value before each page) and the stillness check (and, for SUBLEQ, the two mover scripts of the
  initial-state writes). Counter machines `add`, `multiply 7 9`, `gcd 48 18`, `fib 20` and the SUBLEQ machine run exact
  on it. Section "The doll-driven clock".

Not claimed:

* **The default clock is still an ACS script** (`df.Machine.run()` and `doomformer machine ...` without `--clock dolls`, the
  stored-program machine, the EOS demo of `selfhalt.py`): OPEN script 900, `machine.controller_acs`. Per step it
  runs the rewind (script 3: teleports every doll home, restores the layer ports to their biases, closes the start
  barriers; state values stay), drops the start barriers (script 7), samples every doll's position every 8 tics until none
  moved, and holds the window while Python reads. It also reads the halt latch's plane height once per step, to leave the
  loop. It reads no register and decides nothing else.
* **The doll clock is not "dolls alone".** Its rewind is a list of engine line specials (Floor_MoveToValue,
  Ceiling_MoveToValue, TeleportOther, Light_ChangeToValue, Thing_ChangeTID), one per action of the ACS rewind script, crossed by
  a loop doll; the engine does the actions, the doll decides when. The hold window in which Python reads is a geometric delay
  (120 tics of walking), not a handshake: Python finds it by reading a counter (`pass`) and checks that no tracked doll moved
  while it read; a reader that is too slow raises, it does not read a wrong value. The loop doll itself is not tracked by the
  stillness check. It was built and run for counter machines and for the SUBLEQ machine, not for the stored-program machine
  or the EOS demo (the same hook, not built there).
* **They are bounded machines.** Registers and memory cells are plane heights, and a plane height only holds a bounded range
  (measured below); a counter-machine program is bounded by the 32,000-unit map span (about 1,000 instructions); a SUBLEQ
  program by 18 cells of 7 bits (-64 to 63) and by one rule, that an instruction does not write its own cells (a step that does is flagged by the map's violation latch and stops the level). Both are finite-state machines in the strict sense. With unbounded counters the
  counter machine's instruction set (`inc`, `jzdec`, `halt`) is Minsky's universal counter machine, and plain SUBLEQ is a
  universal one-instruction machine with unbounded memory and words; each is universal only up to its memory bound, like any
  physical computer. **This SUBLEQ is a restricted one, and no universality argument is given for the restricted form.** The
  restriction forbids a step whose write address b is one of the three cells of the instruction being executed: the classic
  self-incrementing operand (`subleq x, self.a`, an instruction that is its own patcher), an instruction that edits its own jump
  target, and any other write of the executing instruction's a, b or c. It allows every instruction to rewrite any other
  instruction (`selfmod.sq`, `fill.sq`, `jumprewrite.sq`, `jumphi.sq`) and pointers built by patch instructions in other slots
  (`indcopy.sq`). The usual constructions that give SUBLEQ indirect access patch another instruction, so they may survive the
  restriction, but a transformation of an arbitrary program into one that obeys it was neither built nor verified here, so the
  statement "universal up to its memory bound" is made for plain SUBLEQ and the counter machine, not shown for this machine. A SUBLEQ program that leaves the cell range or the word range is refused by the interpreter, never run. One that writes the cells of the
  instruction it is executing is refused by the interpreter too, and the map flags it itself: a latch the dolls raise (section "The
  SUBLEQ machine") stops the level at that step and Python reads `viol` = 1 there.
  This is not a proof that DOOM is Turing complete.
* A stack and subroutines are not built (SUBLEQ can build them in software; none is shown).
* The machines were run in ViZDoom 1.3 first. Through the export below, the doll-clock `multiply 2 3` and `subleq add 3 4`
  ran in ZDoom 2.8.1, Zandronum 3.2.1 (with `--low-tags`), GZDoom 4.14.2 (GL and Vulkan), LZDoom 3.88b and UZDoom 5.0.3, and
  `multiply 7 9` (465 lines) in GZDoom, LZDoom and UZDoom; every other program and every ACS-clock map ran in ViZDoom only.
  counter.wad, conway.wad and the calculator below run in other ports (Boom, vanilla); the machine maps need ZDoom's ACS and line specials.

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
cannot rewrite its own instruction memory (no instruction writes a flag) and has no indirect addressing. The SUBLEQ machine
below does both.

## The SUBLEQ machine (`df.SubleqMachine`, `doomformer/subleq.py`)

```
doomformer machine subleq all                    # add, multiply 7 9, selfmod on one map
doomformer machine subleq multiply 7 9
doomformer machine subleq models/subleq/fill.sq  # a program from a source file
doomformer machine subleq selfmod --tamper 13    # negative control
doomformer machine subleq all --clock dolls      # the same with no OPEN script
```

```python
import doomformer as df
from doomformer.subleq import MULTIPLY, SELFMOD
sm = df.SubleqMachine(slots=6)       # 6 instruction slots = 18 memory cells
wad = sm.compile("subleq.wad")       # one map; its bytes do not depend on the program
sm.run(wad, MULTIPLY(7, 9))          # program and data are initial state; every step is checked
sm.run(wad, SELFMOD())               # another program, the same sha256
```

One step (the Python interpreter `subleq.interpret` is the reference; step 0 is the initial state):

```
s = pc;  a, b, c = mem[3s], mem[3s+1], mem[3s+2]     # an instruction is three memory cells
mem[b] = mem[b] - mem[a]                             # a and b are addresses: an indirect read and an indirect write
pc = c if mem[b] <= 0 else s + 1                     # c is a slot number; c < 0, c >= slots or s + 1 == slots: halt
```

Memory is 3 x slots cells (18 by default), each an integer in [-64, 63]; addresses a and b must lie in 0..17. The pc counts
instruction triples, so a jump target is a slot number and every instruction starts on a multiple of three. Any cell can be
read or written as data or run as code: `subleq m1, loop.a, next` adds one to the operand of the instruction at `loop`.
"Indirect addressing" here is what it is in any SUBLEQ: the operands a and b are addresses. A pointer held in a data cell is not
built in; a program makes one by patching an operand with a patch instruction (`models/subleq/indcopy.sq`).
The interpreter refuses a step whose address is out of range, whose result leaves [-64, 63], whose write address b is one of
the three cells of the instruction being executed (limit found by the fuzz, below), and an assembled value outside that
range; the map is never asked to run those, except the own-cell write, which the map flags itself (below). Programs are text (`subleq A B C`, `.cell name value`, `.at cell value`,
labels, `label.a/.b/.c` for an instruction's own cells; `doomformer/subleq.py` docstring).

How the map does it. A cell is two plane heights, `lo` (0..7) and `hi` (-8..7), value = 8 hi + lo, each with a negated twin.
A pass of the level's loop (one step) crosses eight layers; a layer's dolls read values that earlier layers wrote, and
no doll reads a value written in its own layer except through its lane gate (read before any release):

| layer | dolls (6 slots) | what they do |
|---|---|---|
| fetch | 42 | the doll of the active slot (lane gate pc_s >= 1) copies the digits of its three cells into the numbers `al ah` (a), `bl bh` (b), `cl cp cn` (c) by corridors of ceiling gates (a thermometer: one effect per gate passed) |
| decode | 45 | equality gates on those numbers raise the one-hot values `ad_j`, `bd_j` (the addresses) and `ct_t` (the jump target; `ct_S` is the halt latch of a target outside 0..S-1) |
| read | 414 (396 + 18 latch) | lane gate `ad_j >= 1`, equality gates on the digits of cell j: the one-hot digits `xl_v`, `xh_v` of mem[a] |
| number | 30 | one crossbar doll per one-hot digit (22) adds v to the numbers `xl` and `xh`; 8 observer dolls give state values a readout port |
| write | 54 | lane gate `bd_j >= 1`, thermometer corridors on `xl` and `xh`: cell j's two digits go down by xl and xh |
| test | 54 | lane gate `bd_j >= 1`: `le_j` when 8 hi + lo <= 0 (hi < 0, or hi = 0 and lo <= 0); `nb_j` when lo < 0 |
| settle | 36 | `le_j` -> `le`; `nb_j`: lo += 8, hi -= 1 (the borrow) |
| branch | 26 (25 + the latch's halt doll) | pc_s -= 1, pc_{s+1} += 1; if `le`: pc_{s+1} -= 1 and pc_c += 1 behind the gate `ct_c >= 1` |

The write layer changes the cell the address `b` names and no other; the read layer picks the cell the address `a` names.
The jump target c is read in the fetch layer, before the write. A step that writes one of the executing instruction's three
cells (its own a, b or c) is outside the machine: the map computes it wrongly in some cases (a fetch doll that waited at a ceiling
gate of that cell is passed by the write and then blocks the borrow; see the fuzz below), so it is not a defined step.

**The violation latch.** The map detects such a step itself. In the read layer, 18 dolls (one per cell of each slot: lane gate `pc_s`,
stage gate on the decoded store address `bd_{3s+k}`, the same one-hot values the write layer uses) raise the value `viol` when
the store address is one of the executing slot's three cells; one branch-layer doll then raises the halt latch `pc_S`, so the
level stops at that step (ACS loop or halt gate) and Python reads `viol` = 1 from a readout page. The step's other values are
not compared (they may be wrong). `doomformer machine subleq FILE --latch` and `SubleqMachine.run(..., latch=True)` run a program
the host would refuse and require the interpreter's refusal step (`interpret(p, latch=True)`, a trace that ends with the refused
step) to equal the step at which DOOM raises the latch; `interpret` without `latch` still raises, and the CLI without `--latch` refuses
such a program before any tic. Both guards agree. `models/subleq/ownwrite.sq`: the latch goes up at step 2 on the ACS clock
(`ownwrite_latch.log`) and on the doll clock (`ownwrite_latch_dolls.log`), the step before is exact. For programs that never violate the
latch is 0 at every step (every value of every run is compared, `viol` included). The latch reads the decoded store address `bd`, the one-hot
value the write layer uses; the tamper control retargets a decode line of operand a and is unchanged (it still diverges at the
predicted step: `tests/test_subleq.py`, `selfmod_tamper13.log`). What it does not do: make the step safe; the map halts instead of continuing with a wrong value. The borrow-first
update that would make it safe was not built. Why digits: a thermometer over a value of 127 levels would be a corridor of
about 130 gates, and the 18 cells' corridors write one target, so they run one after another; two digits of 8 and 16 levels
cost 22 gates per cell. (The unary version was not built; this is the reasoning, not a measurement.)

Measured (all on one compiled map per clock, 6 slots, 709 dolls and 16,576 sectors with the ACS clock, sha256 `286b198b...`; the machine was loaded,
load average 10 to 50, so wall times are slow and unreliable; game tics are exact):

| program (`doomformer machine subleq ...`) | steps | tics per step | game tics | wall | values checked | result |
|---|---|---|---|---|---|---|
| `add 3 4` | 4 | 1,624 | 6,904 | 29 s | 1,550 | c = 7 |
| `multiply 7 9` (x -= 1, p += 9 seven times; 3 instructions) | 20 | 1,624 | 33,873 | 134 s | 6,510 | p = 63 |
| `selfmod` (z = -(5 + 3 + 7); the first instruction's operand a is rewritten each round) | 10 | 1,624 | 17,011 | 70 s | 3,410 | z = -15 |
| `models/subleq/loop.sq` (103 steps, below) | 103 | 1,624 | 173,712 | 688 s | 32,240 | p = 34, q = -34 |
| `models/subleq/fill.sq` (the first instruction's store address b is rewritten; same sha256 as the others) | 10 | 1,624 | 17,011 | 72 s | 3,410 | arr = 7, 7, 7 |

The same programs with `--clock dolls` (map sha256 `eecb2053...`, 713 dolls, 17,718 sectors, 2,921 to 2,930 tics per step, the
range being the polling granularity): add 46 s, multiply 243 s for 57,801 tics, selfmod 120 s (`models/subleq/subleq_all_dolls.log`),
fill 120 s (`fill_dolls.log`), loop 1,253 s. The ACS-clock runs share the sha256 `286b198b...` (`models/subleq/subleq_all.log`);
every run prints the sha256 of the map file before and after, and they are equal. (The v0.6 maps, without the violation latch, had
the sha256s `e8866508...` and `612e72b8...`.)

The instruction cell changes in the trace (`models/subleq/subleq_all.log`, program selfmod):

```
step   2  pc -> 2   mem[0] 12 -> 13 (code: slot 0 operand a)
step   5  pc -> 2   mem[0] 13 -> 14 (code: slot 0 operand a)
step   8  pc -> 2   mem[0] 14 -> 15 (code: slot 0 operand a)
```

and the rounds in between read cells 12, 13 and 14 through that operand (`mem[15] 0 -> -5`, `-5 -> -8`, `-8 -> -15`).

Cost with memory size (layout counts for 2 to 9 slots; measured tics per step for the sizes run; the halt rule reads one
digit of the jump target, so 8 slots (24 cells) is the limit of this layout):

| slots | cells | dolls | sectors | span (x) | tics per step, measured |
|---|---|---|---|---|---|
| 2 | 6 | 261 | 5,861 | 31,966 | not run |
| 4 | 12 | 485 | 11,219 | 31,966 | not run (add does not fit 12 cells) |
| 5 | 15 | 597 | 13,913 | 31,966 | 1,512 (`multiply 2 3 --slots 5`) |
| 6 | 18 | 709 | 16,576 | 31,966 | 1,624 |
| 8 | 24 | 933 | 21,905 | 31,989 | 1,848 (`add 3 4 --slots 8`) |

Each slot adds three cells, about 112 dolls and 2,700 sectors, and about 112 tics per step (the read layer's serial waves);
the layer is already 32,000 units wide, so a larger memory needs another row of layers (not built). The default of 6 slots is
the smallest size that holds add (4 instructions and 4 data cells).

The negative control (`--tamper ADDR`): the decode line of cell ADDR, the doll that raises `ad_ADDR`, is retargeted to lower
the gate port of `ad_{ADDR+1}` in the compiled WAD. The interpreter predicts the first difference at the first step whose
operand a is ADDR. `selfmod --tamper 13`: cell 13 is first read as operand a at step 4, after the program rewrote the operand
(step 2); DOOM diverges at step 4 (`xl3: DOOM 0 != reference 1`, `xl7: DOOM 1 != reference 0`), the three steps before are exact
(`models/subleq/selfmod_tamper13.log`, with the doll clock `selfmod_tamper13_dolls.log`). `tests/test_subleq.py` runs the
step-1 version.

A longer workload, `models/subleq/loop.sq` (103 steps): two accumulators and a counter, 34 rounds of three instructions and a
halt; p counts 0 to 34 and q 0 to -34 (a decrement through a positive constant), the counter n runs -33 to 1 and ends the loop. On
the ACS clock: 103 steps, 173,712 game tics (1,624 per step), 688 s on a loaded machine (load average 10 to 15, one other game and
a fuzz), 32,240 values compared at tolerance 0, 61 state loads, sha256 `286b198b...` (the same map as every other program;
`loop.log`). Memory before: `[12, 15, 1, 13, 16, 2, 12, 14, 0, 14, 14, -1, -1, 1, -33, 0, 0, 0]` (code in cells 0 to 11, data
m1 = -1, one = 1, n = -33, p, q in cells 12 to 16); after: n = 0, p = 34, q = -34 in cells 14 to 16, the code cells unchanged.
With `--clock dolls` (sha256 `eecb2053...`, 713 dolls, 17,718 sectors, 1,082 rewind lines): 103 steps, 300,496 game tics (2,921 to 2,930
per step, the range is the polling granularity), 1,253 s on the same loaded machine, 32,656 values, exact (`loop_dolls.log`).

A bubble sort was the aim and does not fit. The machine has 18 cells (6 slots; 24 cells, 8 slots, is the limit of this layout):
at 3 cells per instruction that is 6 or 8 instructions and no data. A compare-and-swap of two adjacent cells needs, by my hand count
(not proved minimal), about seven instructions (clear a temporary, load both cells negated, subtract, branch, a three-instruction
swap), and a loop over a sorted prefix patches at least two operands (one increment instruction each) and needs a counter; four
values need about 14 instructions and 8 data cells (50 cells), five about the same code and 9 data cells. The fuzz of this
machine and these programs are the evidence, not a sort. Raising the limit means a wider address decode (the fetch numbers
`ah`, `bh` clamp at 2, the halt test reads one digit of the jump target) and about 109 dolls and 2,600 sectors per slot; 16
slots would be about 1,800 dolls and 42,000 sectors at about 2,500 tics per step (extrapolated from the table below): not built, and at the loaded rate of this
machine a 300-step sort would take hours.

A random campaign (`models/subleq/fuzz.py`, seeds printed): random 18-cell programs whose operands are valid addresses, so
they also write into their own instructions, run on the one map until they halt or until the interpreter would refuse a step
(the map is never asked to run that step). The programs are short (3 to 4 steps on average): the campaign covers many (address,
value, borrow, branch) combinations, not long runs.

* First campaign, before the rule below: with the ACS clock, 60 programs (seeds 1 to 60), 211 steps, 81,312 values compared at
  tolerance 0, all exact (`fuzz_acs_before_refusal.log`). With the doll clock, 30 programs (seeds 101 to 130): 15 exact, then
  **a mismatch at seed 122, step 2** (`fuzz_dolls_before_refusal.log`; `r11: DOOM 1 != reference 0`). The same program on the ACS
  clock fails the same way (one-program rerun, 20 s), so it is the machine, not the clock.
* Cause (read out per port on a copy of the map): the program's slot 1 writes its own operand cell (cell 5, hi digit; x has hi
  digit -1 and the low digit borrows). The fetch doll of that cell waits in front of a closed ceiling gate (hi >= 1). The write
  raises the digit by 1 and opens the gate; the doll slips into the 8-unit gate sector and stands against the freeze barrier;
  the borrow then lowers the digit by 1 and the ceiling stops at the doll's head (`ceil:r11` = 1, its twin `ceil:-r11` = 0). A
  ceiling cannot be lowered past a doll. It needs a doll that waited at a gate the same step later opens, and a later lowering
  below that gate. By the layout that can only happen to cells whose digits the fetch dolls of the executing slot read, the
  instruction's own three cells: no other cell has such a doll waiting.
* The rule: the interpreter refuses a step whose write address is one of the executing instruction's cells, and (v0.7) the map
  flags it with the latch above. The alternative, an update that moves every digit in one direction per step (decide the borrow
  first), was not built.
* Second campaign, with the rule: `python models/subleq/fuzz.py --programs 90 --steps 20 --seed 1 --workers 3`
  (`fuzz_acs.log`): 90 programs, 298 steps, 117,180 values compared at tolerance 0, 0 mismatches, map sha256 unchanged
  (10 of the 90 programs had no runnable step). With `--clock dolls` (`--programs 30 --seed 101`, `fuzz_dolls.log`): 30 programs, 129 steps, 48,984 values, 0 mismatches (3 had no runnable step).
* Latch campaign (`python models/subleq/fuzz.py --programs 400 --steps 20 --seed 1 --violating`, `fuzz_latch.log`): 28 of the 400
  random programs write their own executing instruction, among them seeds 122, 178 and 232, which DOOM computed wrongly when run
  with the guard bypassed. All 28 ran on the ACS clock with the map's latch up at the interpreter's refusal step (13 at step 1,
  9 at step 2, 5 at step 3, 1 at step 4) and every step before it equal at tolerance 0; no program was flagged that the
  interpreter admits (every other run in this section has `viol` = 0 at all its steps). With the doll clock, seeds 1 to 150: 9
  violating programs, all flagged at the interpreter's step (`fuzz_latch_dolls.log`). The latch also fires for the programs
  where DOOM happened to compute the step right: it flags the write, not the wrong value.

Four programs written independently of the machine's code (each checked against a separate 40-line
interpreter, which agreed with `interpret` on every state, and ran them in DOOM on both clocks, exact; that
is an independent run, not a result of this repository's tests) are in `models/subleq/`: `countdown.sq` (17 steps, a
data-dependent loop, ends with a == b on a nonzero value), `jumprewrite.sq` (slot 0 rewrites the jump target of slot 2, 4 -> 3;
r = -2 shows the rewritten path), `indcopy.sq` (two patch instructions write pointers into the operands of a third, which
then copies, negated, through them: dst = -21), `jumphi.sq` (a jump target rewritten to 10, hi digit 1, halts). Re-run
here: `indcopy.sq` on the ACS clock (sha256 `286b198b...`, 6,904 tics, 29 s, 53 state loads, `indcopy.log`) and on the doll clock
(sha256 `eecb2053...`, 10,957 tics, `indcopy_dolls.log`), both exact. The other three were not re-run here.

The blind run (`--clock dolls --blind`; first written as a separate script): load the initial state, seal, then send no
command of any kind (only `advance_action`) for 1.25 times the estimated length of the run, then read every value once and
compare with the interpreter's final state. `machine multiply 7 9 --clock dolls --blind`: 371,780 tics (the run needs about 235,567),
56 values exact, halt latch up, 1 read (`models/dollclock/multiply_7_9_blind.log`). `machine subleq selfmod --clock dolls --blind`:
40,762 tics (needs about 29,200), 314 values exact, z = -15 (`models/subleq/selfmod_dolls_blind.log`). The separate runs: 240,000
tics for multiply, 40,000 for selfmod, exact, and a control stopped at 12,000 tics that reads a different mid-run state. The loop
runs without Python; running on past the halt changes nothing because the halt gate keeps the final window. `--blind` needs
`--clock dolls` and does not combine with `--tamper`.
`--blind` also works with `--latch`: for a program that ends at a refused step the final read must show `viol` = 1 and the halt latch
up (only `viol` is compared there, as in the stepped run; the halt latch can read 2 there, the jump and the latch doll both raise it).
`models/subleq/ownwrite_latch_dolls_blind.log`: `ownwrite.sq`, 8,152 tics with no command, one read of 314 values, latch up.
`add 3 4 --latch` (a program that never violates) reports that the latch stayed down in every step read (`add_latch_clean.log`).

**Other engines (`--export DIR`, `--low-tags`).** `doomformer machine PROGRAM ARGS --clock dolls --export DIR` writes `DIR/machine.wad`,
the doll-clock map plus one extra ACS script, a logger (script 40, OPEN) that exists only in the export, and `DIR/expected.txt`, the
interpreter's lines, and stops without running a game. The logger does what the Python runner does (waits for the counter `pass`,
waits until nothing moves, then prints every value of the step as a `FACT machine step=k ...` line); the engine's console output is
compared with `expected.txt` line by line (`doomformer/ports_machine.py`). It works for the counter programs and `subleq`, and needs
`--clock dolls`. `--low-tags` (with `--export`) builds with every sector tag and TID below 32,768, which Zandronum 3.2.1 needs (it
does not find higher sector tags; ViZDoom prints the same lines with them). Which engines ran which machine, the times and the
speed-up shim are in [engine_limits.md](engine_limits.md#other-ports).

What is claimed here: the map decodes the addresses, reads the addressed cell, writes the addressed cell, tests the sign and
branches, with dolls and gates; Python writes the initial memory (49 to 64 plane writes: add 60, multiply 49, selfmod 59, fill 64, in the `state loads` of the logs) and then only reads. What is not: that
programs can use more than 18 cells or 7-bit words; that the wall time is short (about 8 s per step on a loaded machine;
the step is 709 dolls and 16,576 sectors, and the runner reads 385 values after each of them).

## The doll-driven clock (`--clock dolls`, `doomformer/dollclock.py`)

The question: can the clock, the rewind and the halt check come from dolls and line specials, with ACS only for readout
pages? Yes, for the counter machine and the SUBLEQ machine, with these parts:

* a **tick layer**, one extra last layer with one doll: it adds 1 to the state value `pass` and 1 to `done`, and the line it
  crosses for `done` runs Floor_LowerToLowest on the loop barrier (`compile_pure(actions=...)`);
* a **loop doll**, an extra voodoo doll in the world area on a corridor of sectors and lines: pad, a boot delay (24 tics),
  a line that lowers layer 0's start barrier (the machine starts), the loop barrier (closed; the tick doll opens it when the
  pass is done), a hold delay (120 tics; Python reads here), the **halt gate** (a floor gate whose floor is the pc_halt plane: the
  loop doll's corridor lies 24 units below the gate's floor at latch 0, a step it climbs; at latch 1 the step is 25 and the doll
  stops there for good), the **rewind lines**,
  and a Teleport_NoFog line back to the pad. The corridor folds into several rows when the chain is longer than 27,000 units;
* the **rewind** is one line special per action of ACS script 3, derived from the built map: barriers and freeze barriers go back to
  their closed heights, every doll to its home spot (TeleportOther), every non-state port to its bias, every light output to its
  level, the loop barrier closed again. The order keeps the dolls behind closed barriers: barriers, dolls, ports, lights;
* the dolls tag themselves with the line special Thing_ChangeTID instead of ACS script 4;
* the ACS lump is rebuilt after `compile_pure`: readout pages, the readout guard (script 9; it only fills the readback
  slots USER1..60 before each page) and the motion script (and scripts 1 and 2 for the
  SUBLEQ machine's initial writes). `meta.layout["machine"]["dollclock"]["acs"]` is its source,
  `models/dollclock/multiply_7_9.acs` the counter machine's as built before the guard (no script 9):

```
$ grep -n "^script\|OPEN" models/dollclock/multiply_7_9.acs | cut -c1-60
62:script 100 (void) { o1 = GetSectorLightLevel(20000); o2 = GetSectorLightLevel
66:script 8 (void) {
```

Python finds the window of step k by polling the counter `pass` (one readout page every 8 tics), waits until no tracked doll
moves, then reads every value between two motion checks and checks the counter again. In that read each readout page
is preceded by the guard script, and the read fails if the poison is not there or is not replaced (the polls of `pass`
are not guarded; the full read checks `pass` again). It sends no other command (the audit
counts 0). After the halt it checks that the counter stays and nothing moves.

Measured, counter machines (ViZDoom 1.3, one game each, three at once, load average about 35 to 50; the `ACS` columns are the
rows of the table below):

| `doomformer machine ... --clock dolls` | steps | dolls | sectors | rewind lines | tics per step | ACS clock | game tics | wall | values checked |
|---|---|---|---|---|---|---|---|---|---|
| `add 3 4` | 14 | 24 | 383 | 50 | 459 to 468 | 208 | 6,877 | 1 s | 420 |
| `multiply 7 9` | 464 | 43 | 721 | 84 | 504 to 513 | 216 | 235,567 | 60 s | 26,040 |
| `gcd 48 18` | 647 | 77 | 1,327 | 146 | 577 to 586 | 224 | 376,814 | 140 s | 64,800 |
| `fib 20` | 196 | 236 | 4,809 | 389 | 965 to 974 | 368 | 190,135 | 437 s | 34,672 |

The range is the 8-tic polling granularity. Every row: 0 other commands, 0 state loads, the halt gate stopped the level. The
step takes 2 to 2.6 times as long as with the ACS clock because the loop doll crosses one rewind line per tic (see below):
dolls + ports + lights + barriers tics, plus the hold window of 120 tics. The negative control works on this clock too:
`multiply 7 9 --clock dolls --tamper 2` diverges at step 3, predicted 3 (`models/dollclock/multiply_7_9_tamper2.log`).

Engine facts found while building it:

* **Line specials that one doll crosses in the same tic do not all act.** With rewind cells 4 units wide (the doll crosses
  about four lines per tic) 14 of the 24 TeleportOther lines of `add 3 4` did nothing (the dolls stayed where they were), and a
  Floor_MoveToValue on the loop barrier was lost in `multiply`; with cells 16 units wide (one line per tic) all 24 dolls went
  home and every restore acted. The cause was not investigated; the probe is `models/dollclock/probe_same_tic.py`, its output
  `probe_same_tic.log`. `dollclock.CELL` is 16 for every rewind line. The clock corridors of `pure.py` put several effect cells per tic and are
  exact; the failing lines are specials of a different kind (teleports, plane moves to a value), measured here only.
* TeleportOther, Floor_MoveToValue, Ceiling_MoveToValue, Light_ChangeToValue, Thing_ChangeTID and Teleport_NoFog all work as
  line specials crossed by a voodoo doll (the rewind runs 50 to 1,082 of them per step). A Floor_LowerToLowest on a barrier
  that a doll presses against works as before (the loop barrier).
* The first version let the loop doll lap on a timer when the loop barrier was not closed again: the machine then ran
  passes before the previous one ended and the counter `pass` stopped counting. Python catches it: the readout of step 2
  differed.

What it costs: one tic per rewind line plus about 200 tics that do not depend on the machine (boot, the hold window of 120,
the tick layer). A doll, a non-state port, a light and a barrier are one line each: multiply 7 9 has 84 lines and takes 508
tics per step against 216; fib 20 has 389 and takes 970 against 368; the SUBLEQ machine has 1,082 and takes 2,921 against
1,624 (1,082 + 215 = 1,297 more). The rewind could run in parallel on several loop dolls; not built, and whether the
same-tic loss applies across dolls was not tested.

## What Python does, what the map does

| | Python | ACS level script (default clock) | dolls and lines |
|---|---|---|---|
| program | assembles it, compiles the map (ROM: program is geometry) or writes flags (stored) or the initial memory (SUBLEQ) | none | ROM: the doll of each instruction exists only if the instruction does |
| registers, pc, memory | reads them, compares | none | hold and change them |
| each step | advances tics, runs readout scripts | rewind, barrier drop, settle sampling, hold | decode, test, branch, update |
| halt | notices that pc_N is 1 and the level reached DONE | reads the latch, leaves the loop | raise the latch |
| each step, `--clock dolls` | advances tics, polls the counter `pass`, reads in the window | no script except readout pages, the readout guard (it fills the readback slots before each page), the stillness check (and the movers of SUBLEQ's initial writes) | the tick doll counts, the loop doll waits, holds, rewinds (line specials) and restarts; its halt gate stops it |

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

The SUBLEQ machine's negative control (a decode line retargeted, divergence at step 4 predicted and observed, after the program
had rewritten its own operand) is in its section above; the doll clock's (`multiply 7 9 --clock dolls --tamper 2`, step 3
predicted and observed) in the clock section.

### Limits found

* **Register range.** The machine enforces |v| <= 12,000 (`tile.PORT_MAX`; the interpreter raises beyond it). Probe maps
  that lift the check (one register, `add a k` steps, every step compared): exact at 15,949 and at -15,949; at 15,950
  DOOM reads 6,614 or 6,654 instead (steps of 50 or of 10; `add a 500` from 15,500 fails at 16,000 with 6,214), and the
  negative side fails the same way (its twin reaches +15,950). Starting at +-16,000 and moving toward zero is exact.
  The cause is not investigated; the sector ceilings of the ports sit at 16,000. A register of L limbs in base b holds up
  to b^L - 1 (the `addr` source limbs must stay below b).
* **Doll clock scope.** Built and run at `opt=0` only, for `df.Machine` and `df.SubleqMachine`; the stored-program machine
  and the EOS demo keep the ACS clock.
* **SUBLEQ own-cell writes.** A step that writes one of the executing instruction's three cells is not a defined step: DOOM
  computed one such step wrong (fuzz seed 122, above) and the cause is physical (a ceiling cannot pass a doll). The host
  refuses it and the map flags it with a latch (the level stops there); the programs that never write their own cells are exact for
  everything the fuzz ran, which is not a proof for every program. The latch's coverage is checked against the interpreter on the
  random programs of `fuzz_latch.log` (every violating program flagged at the interpreter's step); it is a doll reading the decode
  lines, so a bug in those lines would corrupt it and the write together.
* **SUBLEQ value range.** Cells hold -64 to 63 (a digit of 8 levels and a digit of 16). The interpreter refuses a result
  outside it and an address outside the 18 cells; `multiply 7 9` ends at p = 63, the largest value a cell holds. Not probed
  in DOOM: what the map does when a program leaves the range (the interpreter never lets it run).
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
| this | a counter machine and a SUBLEQ machine: program counter and registers or memory as plane heights, instructions executed by dolls, programs that loop, branch on data, read and write by address and rewrite their own instructions | by default the clock and rewind are an ACS script; with `--clock dolls` a loop doll and line specials, and ACS only reads; Python reads and compares | geometry (ROM) or initial state (stored, SUBLEQ) | bounded machines; universal only up to their memory bound |

counter.wad and conway.wad run on Boom-compatible ports and the calculator on vanilla Doom 2; this map needs ViZDoom's ZDoom
engine and ACS. The prior works have no readback interface: their outputs are displays the player looks at. The nearest in
spirit is conway.wad, which also repeats a round inside the map; its repeat comes from a teleport graph, ours from an ACS loop
(default) or from a loop doll that rewinds the level with line specials and teleports itself back (`--clock dolls`).
The pages did not show a program that selects among instructions in any of them. Not verified: counter.wad's clocking and how its
bits toggle (the map was not run), the content of the counter and Doom-in-Doom videos. Doom-in-Doom (kgsws) runs machine code
through a code-execution exploit and is not map computation. A search for a programmable computer inside a DOOM map found none; that is not a proof of absence.

## Run it yourself

```
pip install doomformer
doomformer machine multiply 7 9
doomformer machine multiply 7 9 --clock dolls   # no OPEN script
doomformer machine fib 20
doomformer machine stored
doomformer machine subleq all                   # SUBLEQ: add, multiply 7 x 9, a self-modifying program; one sha256
pytest tests/test_machine.py                    # about 25 s
pytest tests/test_subleq.py                     # about 60 s; DOOMFORMER_HEAVY=1 adds the 7 x 9 and selfmod runs (5 minutes)
```

`Machine(program, registers=..., limbs=..., base=16)`; `.reference()` is the interpreter's trace, `.compile(wad)` writes the map,
`.run(wad=None, max_steps=100_000)` runs and checks. `machine.tamper(wad, program, instruction, what="weight"|"target")` is the
negative control. `Machine.compile(wad, clock="dolls")` and `.run(clock="dolls")` build and play the map without an OPEN
script. `SubleqMachine(slots=6)`: `.compile(wad, clock=...)`, `.run(wad, source, clock=...)`, `.reference(source)`;
`subleq.tamper(wad, program, address)` is its negative control. `compile(opt=...)` passes the scheduling level of `pure.compile_pure` through. Measured: `multiply 3 4` and `pow2 6` run exactly at
opt 0 and opt 1 with the same tics per step (216 and 352); opt 2 and 3 need `tuning.json`, which was not present, so they
were not run.
