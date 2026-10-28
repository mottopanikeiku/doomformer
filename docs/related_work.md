# Related work

This page lists prior work that Doomformer builds on or resembles. Every entry is a page opened with a read or fetch tool
on 2026-10-27, and every statement about a source comes from the text that was opened. A page that could not be opened is
marked "Could not open" and nothing is asserted from it. Statements about Doomformer come from the README,
[HOW_IT_WORKS.md](HOW_IT_WORKS.md) and the docstrings here. A Wayback Machine capture is cited by its capture URL.

## Entries

### 1. counter.wad (fraggle, 2006)

- URL: https://www.wad-archive.com/wad/8d0de31d8b392fc2d7fd817e933b3cf43bca799a
- Accessed 2026-10-27. Opened; the page reproduces the readme.
- What it is: a readme dated 13 June 2006 by fraggle describes a 4-bit binary ripple counter built from voodoo dolls and
  conveyor belts with the Boom extensions, one map (MAP01), build time 1 day. The readme credits Afterglow for introducing
  the idea of scripting levels through voodoo dolls on conveyors, "through his start map back in the SMMU days".
- It requires a Boom-compatible port and says it will not work in Legacy. The archive's metadata lists engines "Boom, Legacy".
- Shares: voodoo dolls on conveyor belts as the computing mechanism.
- Differs: a hand-built counter with no program reading values out (the readme describes none). Doomformer's maps use
  ZDoom line specials and ACS, so they need ZDoom (README), and a compiler writes them.

### 2. conway.wad (pwiecz)

- URL: https://github.com/pwiecz/conway.wad
- Accessed 2026-10-27. Opened (README and file list).
- What it is: a "Boom-compatible WAD for playing Conway's Game of Life". The README says "Tested a bit in PrBoom+ and
  GzDoom" and that building it needs a hacked WadC. Players run over raised pedestals to make cells alive and pull a
  switch to start. The README names no computing mechanism, so none is asserted here.
- Shares: a map built from source files by tools (WadC, per the README), with a port caveat.
- Differs: one fixed program with player input. Doomformer compiles arbitrary integer networks, and its only measured
  engine is ViZDoom 1.3.2.
- Link check: the Otterstack post (entry 3) links a Doomworld thread, "Conway's Game of Life in Boom". Could not open
  (HTTP 403): https://www.doomworld.com/forum/topic/131881-conways-game-of-life-in-boom/. Not verified to be this repo.

### 3. Danny Spencer's adding machine

- URLs: https://github.com/nukep/doom-calculator and https://blog.otterstack.com/posts/202212-doom-calculator/
- Accessed 2026-10-27. Both opened.
- What it is: the post "Can Doom Run It? An Adding Machine in Doom" (20 Dec 2022, updated 26 Dec) says it works with the
  vanilla MS-DOS release of Doom 2 v1.9. Logic is binary decision diagrams: a node is a small room with two doors, an edge
  is a teleport, monsters carry the signal, and monsters open remote doors through doors that share a sector. It adds two
  decimal digits with a BCD adder (14 monsters for 14 trees). Clojure generates the WAD. The README says the generated map
  lacks BSP data and will not run in vanilla Doom until it is re-saved in an editor such as SLADE.
- Turing completeness: the post says "I have not proven that Doom is Turing complete" and "My system is not Turing
  complete", because it has no unbounded loops and no state reset. It calls Doom functionally complete.
- Related projects in the post, as far as they could be opened: calabi-yau.space/blog/doom.html (opened; "But can DOOM
  run it?", 22 Oct 2021, a half-adder from monsters, lifts and teleporters, which states that DOOM is Turing-complete; the
  Otterstack post says it doubts that); the Doom Wiki "Voodoo script" page (entry 12); the ripple-carry adder video
  https://www.youtube.com/watch?v=pvji3DJNnqE (page opened, no transcript, nothing asserted); the idgames counter page
  https://www.doomworld.com/idgames/levels/doom2/Ports/a-c/counter (Could not open, HTTP 403); the "Doom in Doom" video
  (not opened).
- Shares: logic inside a Doom map, written by a generator, with an explicit statement about Turing completeness.
- Differs: Boolean trees over monsters in vanilla Doom, single-shot, against integer multiply-adds over dolls and
  light and plane accumulators in ZDoom, with a rewind (HOW_IT_WORKS) so one map runs many inputs.

### 4. Minecraft redstone LeNet-5 (leamoon/StochasticNet)

- URL: https://github.com/leamoon/StochasticNet
- Accessed 2026-10-27. Opened (README and file list); the linked Bilibili video was not opened.
- What the README states: the repository is part of a redstone convolutional network in Minecraft that recognises 15x15
  handwritten digits with a LeNet-5 architecture, at accuracy "up to 80%". It uses stochastic computing instead of
  full-precision computing. Recognition takes 5 minutes per figure in theory and over 20 minutes in practice.
- Shares: a neural network executed by a game's own mechanics.
- Differs: the README describes no per-value check against a reference. Doomformer computes integers and compares every
  exposed value with a NumPy reference at tolerance 0.

### 5. GPT-2 in a spreadsheet

- URLs: https://arstechnica.com/information-technology/2024/03/once-too-scary-to-release-gpt-2-gets-squeezed-into-an-excel-spreadsheet/
  and https://spreadsheets-are-all-you-need.ai/ (linked from the article)
- Accessed 2026-10-27. Both opened; the GitHub release pages they link were not opened.
- What it is: Ars Technica (Benj Edwards, 15 Mar 2024) reports that Ishan Anand put GPT-2 small (124 million parameters)
  into an Excel XLSB file of about 1.2 GB. It runs locally, accepts 10 input tokens, and is meant to teach how LLMs work.
  The author's site says it implements GPT-2 "using vanilla JavaScript and pure Excel functions".
- Shares: a transformer on a substrate that is not an ML framework.
- Differs: it runs released GPT-2 weights for teaching. Doomformer trains small integer models (README: 1 and 2 layers) and
  checks each run. The pages do not say which number format the sheet uses and mention no correctness check.

### 6. I-BERT (arXiv 2101.01321)

- URLs: https://arxiv.org/abs/2101.01321 and https://arxiv.org/pdf/2101.01321
- Accessed 2026-10-27. Both opened.
- What it is: Kim, Gholami, Yao, Mahoney, Keutzer (ICML 2021 per the PDF). The abstract states that it quantizes the
  entire inference with integer-only arithmetic, using lightweight integer-only approximations for GELU, Softmax and
  Layer Normalization. In the PDF, GELU and the exponential are second-order polynomials evaluated in integers, the
  exponential also uses a shift, and the LayerNorm square root is an iterative integer algorithm. Matrix products are
  INT8 with INT32 accumulation. Evaluation is RoBERTa-Base/Large on GLUE against FP32, plus a 2.4 to 4.0 times INT8
  speedup on a T4 GPU (abstract).
- Shares: integer-only transformer inference including a softmax and a normalisation with an integer square root.
- Differs: I-BERT approximates float functions and reports accuracy against FP32. Doomformer's integer model is the
  specification and the check is equality with it; distance to the float model is measured separately
  (`net.quantization_report`). Its softmax mode uses a log-spaced staircase for the exponential and a count of comparisons
  for the division, with no polynomial. Its RMSNorm gets floor(sqrt(s/d)) by counting thresholds. No model has LayerNorm.
  Doomformer is about 65,000 times slower than NumPy (README); I-BERT targets speed.

### 7. ReLU attention divided by sequence length (arXiv 2309.08586)

- URLs: https://arxiv.org/abs/2309.08586 and https://arxiv.org/pdf/2309.08586
- Accessed 2026-10-27. Both opened.
- What it is: "Replacing softmax with ReLU in Vision Transformers", Wortsman, Lee, Gilmer, Kornblith (Google DeepMind), 15 Sep
  2023. It replaces the attention softmax with L^-1 relu, where L is sequence length, and reports that this mitigates the
  accuracy loss of plain ReLU. Experiments are vision transformers on ImageNet-21k and ImageNet-1k, with qk-layernorm. It
  also tries L^-a with other nonlinearities and finds a near 1 best. The text of the paper does not mention causal
  masking, integers or quantization.
- Shares: ReLU-style attention weights with a sequence-length normaliser instead of softmax. Doomformer's relu mode scores
  with a ReLU staircase (`adc_relu`, causal) and applies a per-position staircase `thr_o[t]` that "can fold a 1/(t+1)"
  (`intref.py`).
- Differs: the divisor is t+1, the number of keys position t sees (causal), and it lives in the staircase thresholds
  `thr_o[t]` ("fixed 1/(t+1) in thr_o", `training/train_brain.py --help`), not in a division. The numbers are integers
  throughout. The paper's models are vision transformers trained on ImageNet.

### 8. Attention Is Off By One (Evan Miller)

- URL: https://www.evanmiller.org/attention-is-off-by-one.html
- Accessed 2026-10-27. Opened (full text).
- What it is: an opinion post dated 24 July 2023. It proposes softmax_1(x)_i = exp(x_i) / (1 + sum_j exp(x_j)), calls
  attention that uses it QuietAttention, and argues that softmax forces every head to write something even with nothing to
  say. It reports no experiments and asks others to run them.
- Shares: the same construction. Doomformer's softmax mode has `den = 1 + sum_u e_u`, one extra key with value 0, so a
  query whose scores are all below the first threshold attends to nothing (HOW_IT_WORKS, `intref.py`).
- Differs: the exponential is quantised, `e = min(KA, floor(exp((score - c)/sigma)))` from a staircase, and the division
  is truncated integer division computed as a count of comparisons. `c` and `sigma` are learned in quantisation-aware
  training. The docs say the 1 keeps `den >= 1` and make no claim about outliers. Only three shipped models use it; the
  relu mode does not normalise weights to sum to one, so all-zero weights give zero output there too.

### 9. Tracr and TransformerLens

- URLs: https://arxiv.org/abs/2301.05062, https://github.com/google-deepmind/tracr,
  https://github.com/TransformerLensOrg/TransformerLens
- Accessed 2026-10-27. All opened (the GitHub pages through the fetch tool's repository view: README and file list).
- Tracr: Lindner et al. (Jan 2023) compile human-readable programs into decoder-only transformers with known structure,
  to serve as ground truth for interpretability methods (examples: token frequencies, sorting, parenthesis checking). The
  README describes a compiler from RASP programs to transformer weights, a Haiku model, computed directly with "no
  gradient descent required". Compiled models need a BOS token so selectors can attend to nothing; the README contrasts
  this with the softmax default of treating an all-false row as all-true.
- TransformerLens: a library for mechanistic interpretability of GPT-style models. The README says it exposes internal
  activations and lets you cache, edit, remove or replace them as the model runs. The file list names a demo notebook
  `Tracr_to_Transformer_Lens_Demo.ipynb` (name only; not opened).
- Shares with Tracr: a compiler whose output has known structure by construction, and a way to say "attend to nothing".
  Shares with TransformerLens: intervening on a named internal value (`DoomTransformer(ablate=...)` closes the doors of
  chosen units, README and HOW_IT_WORKS).
- Differs: Tracr maps a program to weights with no training; Doomformer maps a trained integer network into a physical
  substrate with exact readback. TransformerLens loads existing models to analyse them; the Doomformer README documents
  ablation by doors and a lesion demo. Its null slot sits in the softmax denominator, not in a BOS token.

### 10. Translation validation (Pnueli, Siegel, Singerman, TACAS 1998)

- URL: https://link.springer.com/chapter/10.1007/BFb0054170
- Accessed 2026-10-27. Opened (abstract page). The PDF at https://link.springer.com/content/pdf/10.1007/BFb0054170.pdf
  opened but text extraction failed, so nothing is cited from it.
- What it is: LNCS 1384, pp. 151-166. The abstract contrasts compiler verification (proving in advance that the compiler
  is correct) with translation validation, where each run of the compiler is followed by a phase that verifies the target
  code from that run implements the source. It lists a common semantic framework, a refinement relation and a
  simulation-based proof method, and illustrates them on Signal to C.
- Shares: the check is attached to each compilation output, not to the compiler.
- Differs: Doomformer's check is a test, not a proof. Python compares exposed values with an independent NumPy
  integer implementation on the inputs that were run, and the README says this says nothing about inputs that were not
  run. There is no refinement relation and the compiler is not verified. Translation validation in spirit only.

### 11. ViZDoom

- URLs: https://github.com/Farama-Foundation/ViZDoom and https://arxiv.org/abs/1605.02097
- Accessed 2026-10-27. Both opened (the repository through the fetch tool's repository view).
- What it is: the README says ViZDoom allows bots that play Doom using only the screen buffer, is primarily for research
  in machine visual learning and deep reinforcement learning, and is based on the ZDoom engine. It lists Python and C++
  APIs, off-screen rendering, and access to the actor list and map geometry, and points to the ZDoom wiki for ACS. The
  paper (Kempka, Wydmuch, Runc, Toczek, Jaskowski, May 2016) presents it as a test-bed for RL from raw visual input in a
  first-person 3D world, tested with Q-learning bots on a move-and-shoot task and a maze.
- Shares: Doomformer runs inside ViZDoom 1.3.2 and uses its Python API.
- Differs: Doomformer reads sector light levels and plane heights, and the README says it has no RL and no learning from
  pixels. The 4,096-object list limit and the engine facts in HOW_IT_WORKS were measured on 1.3.2 only.

### 12. Doom Wiki: Voodoo doll and Voodoo script

- URLs opened: https://web.archive.org/web/20241213205025/https://doomwiki.org/wiki/Voodoo_doll and
  https://web.archive.org/web/20250317225808/https://doomwiki.org/wiki/Voodoo_script
- Accessed 2026-10-27. Could not open (HTTP 403): https://doomwiki.org/wiki/Voodoo_doll,
  https://www.doomwiki.org/wiki/Voodoo_doll, https://doomwiki.org/wiki/Voodoo_script and
  https://doom.fandom.com/wiki/Voodoo_doll.
- What they say: extra player starts spawn dormant copies of the player. In Boom-compatible ports (and Heretic, Hexen,
  Strife, Doomsday) dolls on conveyors trigger walk lines, "a limited form of scripting", and counter.wad is cited as an
  example. The script page lists disadvantages: no internal state to examine, silent breakage, and sensitivity to port
  differences, including ZDoom's scrolling floor speed sometimes differing from other Boom-derived ports.
- Shares: the mechanism (dolls riding conveyors, triggering lines). Doomformer's per-run readback addresses silent
  breakage. Differs: the page frames it as scripting; Doomformer uses it as arithmetic and states results for ViZDoom 1.3.2
  only, since port differences exist.

### 13. ZDoom Wiki: ACS, Floor_MoveToValue, Light_RaiseByValue

- URLs opened (Wayback): https://web.archive.org/web/20241216140329/https://zdoom.org/wiki/ACS,
  https://web.archive.org/web/20241208113153/https://zdoom.org/wiki/Floor_MoveToValue,
  https://web.archive.org/web/20241015100259/https://zdoom.org/wiki/Light_RaiseByValue
- Accessed 2026-10-27. The live zdoom.org pages: could not open (connection reset in the read tool; a fetch with curl
  returned a bot-challenge page, no content).
- What they say: ACS is the scripting language created for Hexen and expanded by ZDoom, compiled with ACC. Special 37
  `Floor_MoveToValue(tag, speed, height, neg)` moves the floor of affected sectors to an absolute height, and tag 0 uses
  the line's back sector. Special 110 `Light_RaiseByValue(tag, value)` increases the light level in a sector by value.
- Shares: these are the specials and the scripting language Doomformer's maps use (README builds ACC on first run).
- Differs: the pages say nothing about light saturating at int16 or about dolls crossing the lines. Those are Doomformer's
  measurements (HOW_IT_WORKS, Engine facts).

## What this repo adds, and what it does not claim

Adds, in the README's words: a compiler from integer networks to ZDoom maps, a measured list of engine behaviours that
break exactness, and a tolerance-0 check against a NumPy reference.

Does not claim:

- Turing completeness or universality. The README and the docs claim none: the arithmetic runs on fixed, finite maps and
  the machines have bounded memory. A SUBLEQ machine is universal only with unbounded memory ([machine.md](machine.md)).
- Compiler correctness. EXACT is integer equality over the runs that were made, and says nothing about inputs that were not
  run. It also says nothing about whether the answer is right: it means DOOM equals the integer network.
- Engine generality. The engine facts were measured in ViZDoom 1.3.2 (ZDoom 2.8.1). The exported probe maps and the
  `hello` brain also ran in stock ZDoom 2.8.1, Zandronum 3.2.1, GZDoom 4.14.2, LZDoom and UZDoom
  ([engine_limits.md](engine_limits.md#other-ports)); nothing larger than `hello` ran in another port. The maps use UDMF
  and ACS, so the 1993 executable is out.
- Priority over the entries above. ReLU attention with a length divisor (7), a +1 in the softmax denominator (8), integer
  transformer inference (6), compiled transformers (9) and logic in Doom maps (1 to 3) exist elsewhere. The repository
  combines them under a per-run exact check.
- Speed or breadth. It is about 65,000 times slower than NumPy on a small MLP. Attention is ReLU-style in most models and
  softmax in three, no model has LayerNorm, and showcased seeds, names and lesions are picked.
