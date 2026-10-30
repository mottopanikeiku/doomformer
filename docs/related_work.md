# Related work

Other people have built logic in Doom maps, networks in games and integer transformers. This page lists that work. Each
entry says what it is, what Doomformer shares with it and what differs.

> [!NOTE]
> Every source was opened on 2026-10-27. Statements about a source come from the opened text. Pages that could not be
> opened are marked, and nothing is asserted from them.

| entries | theme |
|---|---|
| 1 to 3 | logic built inside Doom maps |
| 4, 5 | neural networks in a game or a spreadsheet |
| 6 to 9 | integer, ReLU-style and compiled transformers |
| 10 | checking each compiler output |
| 11 to 13 | ViZDoom, voodoo dolls and the ZDoom specials |

## Entries

### 1. counter.wad (fraggle, 2006)

- URL: https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a (reproduces the readme)
- What it is: a 4-bit binary ripple counter built from voodoo dolls and conveyor belts in one Boom map. The readme credits
  Afterglow's SMMU start map with the idea.
- Shares: voodoo dolls on conveyors as the computing mechanism.
- Differs: hand-built, with no program reading values out. Doomformer's maps are compiled and need ZDoom line specials
  and ACS.

### 2. conway.wad (pwiecz)

- URL: https://github.com/pwiecz/conway.wad
- What it is: a "Boom-compatible WAD for playing Conway's Game of Life", built with a hacked WadC. Players raise
  pedestals to make cells alive. The README names no computing mechanism.
- Shares: a map generated from source by tools.
- Differs: one fixed program with player input. Doomformer compiles arbitrary integer networks.
- Not opened (HTTP 403): https://www.doomworld.com/forum/topic/131881-conways-game-of-life-in-boom/

### 3. Danny Spencer's adding machine

- URLs: https://github.com/nukep/doom-calculator and https://blog.otterstack.com/posts/202212-doom-calculator/
- What it is: a BCD adder of two decimal digits in vanilla Doom 2 v1.9. Logic is binary decision diagrams: rooms with two
  doors, teleports as edges, monsters as the signal. Clojure generates the WAD.
- The post says "My system is not Turing complete": it has no unbounded loops and no state reset.
- Shares: generated logic inside a Doom map, with an explicit statement about Turing completeness.
- Differs: single-shot Boolean trees over monsters. Doomformer does integer multiply-adds over dolls, lights and planes,
  and rewinds one map for many inputs.
- The post also links https://calabi-yau.space/blog/doom.html, a half-adder from monsters, lifts and teleporters.

### 4. Minecraft redstone LeNet-5 (leamoon/StochasticNet)

- URL: https://github.com/leamoon/StochasticNet
- What it is: a redstone LeNet-5 in Minecraft that recognises 15x15 handwritten digits "up to 80%", using stochastic
  computing. One figure takes over 20 minutes in practice.
- Shares: a neural network run by a game's own mechanics.
- Differs: no per-value check is described. Doomformer computes integers and compares every exposed value with NumPy at
  tolerance 0.

### 5. GPT-2 in a spreadsheet

- URLs: https://arstechnica.com/information-technology/2024/03/once-too-scary-to-release-gpt-2-gets-squeezed-into-an-excel-spreadsheet/
  and https://spreadsheets-are-all-you-need.ai/
- What it is: Ishan Anand's GPT-2 small in an Excel file, built for teaching how LLMs work.
- Shares: a transformer on a substrate that is not an ML framework.
- Differs: it runs released GPT-2 weights. Doomformer trains small integer models and checks each run. The pages mention
  no correctness check.

### 6. I-BERT (arXiv 2101.01321)

- URL: https://arxiv.org/abs/2101.01321
- What it is: integer-only transformer inference (Kim et al., ICML 2021). GELU and the exponential are integer
  polynomials. LayerNorm uses an iterative integer square root. Accuracy is reported against FP32.
- Shares: integer-only inference with a softmax and an integer square root.
- Differs: Doomformer's integer model is the specification, and the check is equality with it. Its exponential is a
  staircase and its division a count of comparisons. No model has LayerNorm. I-BERT targets speed. Doomformer is slow
  ([scaling.md](scaling.md)).

### 7. ReLU attention divided by sequence length (arXiv 2309.08586)

- URL: https://arxiv.org/abs/2309.08586
- What it is: Wortsman et al. (2023) replace the attention softmax with L^-1 relu, L the sequence length, in vision
  transformers on ImageNet.
- Shares: ReLU attention with a length normaliser. Doomformer's relu mode can fold 1/(t+1) into its staircase `thr_o[t]`
  (`intref.py`).
- Differs: Doomformer's divisor is causal, t+1 keys at position t, and lives in integer thresholds, not a division.

### 8. Attention Is Off By One (Evan Miller)

- URL: https://www.evanmiller.org/attention-is-off-by-one.html
- What it is: a 2023 post proposing softmax_1, with 1 added to the denominator so a head can attend to nothing. It
  reports no experiments.
- Shares: Doomformer's softmax mode has `den = 1 + sum_u e_u`, so a query below every threshold attends to nothing.
- Differs: the exponential is a learned staircase and the division is truncated. The docs make no claim about outliers.
  Three shipped models use this mode.

### 9. Tracr and TransformerLens

- URLs: https://arxiv.org/abs/2301.05062, https://github.com/google-deepmind/tracr,
  https://github.com/TransformerLensOrg/TransformerLens
- What they are: Tracr compiles RASP programs into transformer weights with no training. TransformerLens exposes and edits
  internal activations of GPT-style models.
- Shares: a compiler whose output has known structure, and intervention on named internal values
  (`DoomTransformer(ablate=...)`, see [interventions.md](interventions.md)).
- Differs: Doomformer compiles a trained integer network into a physical substrate with exact readback. Its null slot is
  in the softmax denominator, not a BOS token.

### 10. Translation validation (Pnueli, Siegel, Singerman, TACAS 1998)

- URL: https://link.springer.com/chapter/10.1007/BFb0054170 (abstract only)
- What it is: each compiler run is followed by a check that the target code implements the source.
- Shares: the check is attached to each compilation output.
- Differs: Doomformer's check is a test on the inputs run, not a proof. There is no refinement relation.

### 11. ViZDoom

- URLs: https://github.com/Farama-Foundation/ViZDoom and https://arxiv.org/abs/1605.02097
- What it is: a ZDoom-based platform for reinforcement learning from the screen buffer, with Python and C++ APIs.
- Shares: Doomformer runs inside ViZDoom 1.3.2 and uses its Python API.
- Differs: Doomformer reads sector lights and plane heights. It has no RL and no learning from pixels.

### 12. Doom Wiki: Voodoo doll and Voodoo script

- URLs: https://web.archive.org/web/20241213205025/https://doomwiki.org/wiki/Voodoo_doll and
  https://web.archive.org/web/20250317225808/https://doomwiki.org/wiki/Voodoo_script (live pages: HTTP 403)
- What they say: dolls on conveyors trigger walk lines, "a limited form of scripting". Listed drawbacks include silent
  breakage and port differences.
- Shares: the mechanism.
- Differs: Doomformer uses it as arithmetic. Its per-run readback catches silent breakage.

### 13. ZDoom Wiki: ACS, Floor_MoveToValue, Light_RaiseByValue

- URLs (Wayback): https://web.archive.org/web/20241216140329/https://zdoom.org/wiki/ACS,
  https://web.archive.org/web/20241208113153/https://zdoom.org/wiki/Floor_MoveToValue,
  https://web.archive.org/web/20241015100259/https://zdoom.org/wiki/Light_RaiseByValue
- What they say: ACS is Hexen's scripting language, expanded by ZDoom. The two specials set a floor height and raise a
  light level.
- Shares: these are the specials and language Doomformer's maps use.
- Differs: the pages say nothing about light saturation or dolls crossing lines. Those are measured in
  [HOW_IT_WORKS.md](HOW_IT_WORKS.md#engine-facts).

## What this repo does not claim

- Turing completeness. The maps are finite and the machines have bounded memory ([machine.md](machine.md)).
- Compiler correctness. EXACT means DOOM equals the integer network on the inputs run.
- Engine generality. Engine facts are measured in ViZDoom 1.3.2. Other ports are in
  [engine_limits.md](engine_limits.md#other-ports).
- Priority. Each ingredient above exists elsewhere. The repo combines them under a per-run exact check.
- Speed. See [scaling.md](scaling.md).
