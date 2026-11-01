# Related work

Doomformer combines voodoo-doll computation with an integer-network compiler and per-run readback. Its closest
connections are earlier computers built in Doom maps, neural networks on unusual substrates, and integer transformers.

## Logic in Doom maps

### 1. counter.wad (fraggle, 2006)

[counter.wad](https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a) is a hand-built 4-bit binary
ripple counter in one Boom map. Its voodoo dolls ride conveyors and trigger the logic; the readme credits Afterglow's
SMMU start map with the idea. Doomformer uses the same computing mechanism, with compiled networks, ZDoom line
specials and ACS readout.

### 2. conway.wad (pwiecz)

[conway.wad](https://github.com/pwiecz/conway.wad) generates a Boom-compatible Conway's Game of Life map with a modified
WadC. Players raise pedestals to choose the initial live cells. It generates one fixed program; Doomformer's compiler
takes integer networks as input.

### 3. Danny Spencer's adding machine

[doom-calculator](https://github.com/nukep/doom-calculator) adds two decimal digits in vanilla Doom 2 v1.9.
[Spencer's write-up](https://blog.otterstack.com/posts/202212-doom-calculator/) explains the binary decision diagrams:
rooms with two doors, teleports as edges and monsters as signals. Clojure generates the WAD. The machine runs once,
with no unbounded loops or state reset. Doomformer's dolls accumulate integer values in lights and planes, and the map
rewinds for repeated inputs. Spencer also links a [half-adder built from monsters, lifts and
teleporters](https://calabi-yau.space/blog/doom.html).

## Neural networks on unusual substrates

### 4. Minecraft redstone LeNet-5 (leamoon/StochasticNet)

[StochasticNet](https://github.com/leamoon/StochasticNet) runs LeNet-5 through Minecraft redstone and stochastic
computing. It reports up to 80% recognition on 15×15 handwritten digits, taking over 20 minutes per image in practice.
Both projects use game mechanics for network arithmetic. Doomformer uses integer arithmetic and checks each exposed
value against NumPy at tolerance 0.

### 5. GPT-2 in a spreadsheet

Ishan Anand's [Spreadsheets Are All You Need](https://spreadsheets-are-all-you-need.ai/) puts GPT-2 small into Excel to
teach transformer inference ([Ars Technica coverage](https://arstechnica.com/information-technology/2024/03/once-too-scary-to-release-gpt-2-gets-squeezed-into-an-excel-spreadsheet/)).
It runs the released GPT-2 weights. Doomformer trains small integer models for the map's arithmetic.

## Transformer arithmetic and compilation

### 6. I-BERT (arXiv 2101.01321)

[Kim et al., I-BERT](https://arxiv.org/abs/2101.01321) (ICML 2021) implement integer-only transformer inference.
GELU and the exponential use integer polynomials; LayerNorm uses an iterative integer square root. They compare
accuracy with FP32. Doomformer's exponential is a staircase and its division counts comparisons. Its models use
integer RMSNorm or no normalisation; none uses LayerNorm.

### 7. ReLU attention divided by sequence length (arXiv 2309.08586)

[Wortsman et al.](https://arxiv.org/abs/2309.08586) (2023) replace attention softmax with `L^-1 relu`, where `L` is
sequence length, in vision transformers on ImageNet. Doomformer's ReLU mode folds the causal normaliser `1/(t+1)`
into the staircase `thr_o[t]` in `intref.py`.

### 8. Attention Is Off By One (Evan Miller)

[Miller's 2023 post](https://www.evanmiller.org/attention-is-off-by-one.html) proposes `softmax_1`: add 1 to the
denominator so a head can attend to nothing. Doomformer's softmax mode uses `den = 1 + sum_u e_u`, so a query below
every score threshold attends to nothing. Its exponential is a learned staircase and its division is truncated.
Three shipped models use this mode.

### 9. Tracr and TransformerLens

[Tracr](https://github.com/google-deepmind/tracr) ([paper](https://arxiv.org/abs/2301.05062)) compiles RASP programs into
transformer weights without training. [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) exposes
and edits internal activations of GPT-style models. Doomformer compiles a trained integer network into map geometry
and supports interventions on named internal values, including `DoomTransformer(ablate=...)`.
See [interventions.md](interventions.md).

### 10. Translation validation (Pnueli, Siegel, Singerman, TACAS 1998)

[Translation validation](https://link.springer.com/chapter/10.1007/BFb0054170) checks each compiler output against its
source. Pnueli, Siegel and Singerman construct a refinement proof. Doomformer checks the compiled map by running inputs
and comparing exposed values with the integer reference. [Static extraction](verification.md#static-extraction) also
recovers an MLP's arithmetic directly from the WAD.

## Engine foundations

### 11. ViZDoom

[ViZDoom](https://github.com/Farama-Foundation/ViZDoom) ([paper](https://arxiv.org/abs/1605.02097)) is a ZDoom-based
reinforcement-learning platform with Python and C++ APIs. Doomformer uses ViZDoom 1.3.2 to run maps and read sector
lights and plane heights. Its marine policies imitate scripted experts using scene features.

### 12. Doom Wiki: Voodoo doll and Voodoo script

The Doom Wiki's [Voodoo doll](https://web.archive.org/web/20241213205025/https://doomwiki.org/wiki/Voodoo_doll) and
[Voodoo script](https://web.archive.org/web/20250317225808/https://doomwiki.org/wiki/Voodoo_script) pages describe dolls
on conveyors triggering walk lines as a limited form of scripting. They also describe silent failures and port
differences. Doomformer uses those crossings for arithmetic and reads the results back after each pass.

### 13. ZDoom Wiki: ACS, Floor_MoveToValue, Light_RaiseByValue

The ZDoom Wiki documents [ACS](https://web.archive.org/web/20241216140329/https://zdoom.org/wiki/ACS), Hexen's scripting
language extended by ZDoom, and the specials
[Floor_MoveToValue](https://web.archive.org/web/20241208113153/https://zdoom.org/wiki/Floor_MoveToValue) and
[Light_RaiseByValue](https://web.archive.org/web/20241015100259/https://zdoom.org/wiki/Light_RaiseByValue).
Doomformer uses them to write gate heights and accumulate weights. Its [engine probes](HOW_IT_WORKS.md#engine-facts)
measure details such as light saturation and doll crossings.

## Scope

- Maps are finite, with bounded memory. The [counter and SUBLEQ machines](machine.md) have explicit storage limits.
- `EXACT` means equality with the integer network on the inputs run; domain coverage is described in [verification](verification.md).
- Engine behaviour is measured primarily in ViZDoom 1.3.2. [Other ports](engine_limits.md#other-ports) have narrower coverage.
- The maps are slow. [Scaling](scaling.md) gives measured costs and the trade-off between one map and tiles.
