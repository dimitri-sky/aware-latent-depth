# Aware: Benchmark-First Small-LM Architecture Research

**Subtitle:** Reasoning-per-FLOP through recurrent depth, fast-weight memory, and learned halting.

## One-page research concept

Aware tests whether small language models (10M-410M parameters) can scale intelligence
through **adaptive latent depth** and **fast-weight memory** instead of only through more
parameters, more tokens, and longer chain-of-thought.

We do not claim a new paradigm. We claim a set of serious, falsifiable architecture
hypotheses, and we let matched-baseline benchmarks decide. The project is run as an
autonomous Cursor research-agent loop: hypothesis backlog -> minimum falsifier ->
smallest code diff -> tiny local run -> matched comparison -> CSV evidence ->
kill / revise / scale. Negative results are first-class deliverables.

**Main research question:** Can an autonomous Cursor research agent discover and validate
a small language-model architecture that achieves better intelligence-per-parameter and
reasoning-per-FLOP than matched modern transformers?

**Pre-registered starting claim:** "We test whether an autonomous Cursor research-agent
loop can discover small-LM architectures that improve reasoning-per-FLOP and
performance-per-parameter versus matched modern transformers." No stronger claim is made
until the 160M gate passes (variant beats matched Transformer++ on >= 3 of 5 core SAGE
families at matched training AND inference FLOPs).

## Project name

- **Aware** — the research program and the candidate architecture family.
- **SAGE** — the benchmark suite: *Symbolic and Algorithmic Generalization Evaluation*.

**GitHub repo description:** "Benchmark-first research program testing whether small LMs
can scale reasoning-per-FLOP through recurrent latent depth and fast-weight memory, run
by an autonomous Cursor research-agent loop. Negative results welcome."

## Deep literature-informed analysis

All claims below are sourced; no unsourced paper claims are permitted anywhere in this repo.

### Why transformers work (and what that implies for small models)

- Next-token prediction over large corpora is a dense, self-supervised objective whose
  loss decreases predictably with parameters, data, and compute (Kaplan et al.,
  arXiv:2001.08361; Hoffmann et al. "Chinchilla", arXiv:2203.15556).
- Attention works because it provides content-addressable retrieval over the full
  context with O(1) path length between any two tokens; induction heads implement
  in-context copying/completion and are a major driver of in-context learning
  (Olsson et al., "In-context Learning and Induction Heads", Anthropic 2022,
  arXiv:2209.11895).
- Parameter knowledge capacity is roughly constant per parameter: ~2 bits/param
  (Allen-Zhu & Li, "Physics of Language Models Part 3.3", arXiv:2404.05405; independently
  observed for looped models in Ouro, arXiv:2510.25741). **Implication: a small model
  cannot win a knowledge-storage contest. It can only plausibly win on knowledge
  manipulation, computation, and adaptation.**

### Where transformers waste compute (small-model bottlenecks)

1. **Fixed serial depth per token.** A K-layer transformer performs exactly K
   sequential computation steps per token regardless of difficulty. Many reasoning
   problems require depth that grows with problem size; parameters do not substitute for
   depth (Saunshi et al., "Reasoning with Latent Thoughts", arXiv:2502.17416: a k-layer
   model looped L times nearly matches a kL-layer model on reasoning tasks).
2. **Chain-of-thought is an expensive serial workaround.** Each CoT token costs a full
   forward pass (embedding -> all layers -> vocab softmax) plus growing attention, and
   quantizes intermediate latent state through a discrete token bottleneck. Formal
   results show looped/latent computation simulates parallelizable deterministic
   computation with only logarithmic loop counts, while CoT needs linearly many steps;
   conversely CoT wins on tasks requiring stochastic sampling/approximate counting
   (Saito et al., "To CoT or To Loop?", arXiv:2505.19245). This motivates our core
   controls in both directions.
3. **No compressed reusable state.** In-context learning is the only "fast memory" of a
   transformer: it is quadratic-cost, evanescent (lost when context ends), and
   KV memory grows linearly with session length.
4. **Uniform compute allocation.** Easy tokens and hard tokens get identical compute.
   Mixture-of-Recursions (Bae et al., arXiv:2507.10524) and Ouro's learned exit show
   per-token depth allocation is feasible.

### Evidence for the two core mechanisms

- **Recurrent latent depth.** Ouro looped LMs (Zhu et al., arXiv:2510.25741, trained to
  7.7T tokens): 1.4B looped model matches ~4B dense; controlled experiments attribute
  the gain to knowledge *manipulation*, not storage. Huginn recurrent-depth 3.5B
  (Geiping et al., arXiv:2502.05171, NeurIPS 2025 spotlight): test-time latent iteration
  improves math/code, with zero-shot per-token adaptive compute and KV-cache sharing.
  TRM (Jolicoeur-Martineau, arXiv:2510.04871): a 7M-param, 2-layer recursive net with
  deep supervision reaches 45% on ARC-AGI-1. Caveats: TRM gains rely heavily on deep
  supervision + augmentation (arXiv:2512.11847); Huginn shows "latent overthinking"
  failure modes.
- **Fast-weight / delta-rule memory.** The delta rule performs online key-value
  regression in a fast-weight matrix (Schlag et al., "Linear Transformers Are Secretly
  Fast Weight Programmers", arXiv:2102.11174). Gated DeltaNet (Yang et al., ICLR 2025,
  arXiv:2412.06464) adds a forget gate and a hardware-efficient chunked-scan training
  algorithm and beats Mamba2, DeltaNet, and Transformer++ at 340M-1.3B on language
  modeling, retrieval, and length extrapolation; hybrids with sliding-window attention
  are strongest. Kimi Linear (arXiv:2510.26692) confirms the design at large scale.
- **What we deprioritize and why.**
  - *Learned halting:* ACT (Graves, arXiv:1603.08983) is unstable and
    hyperparameter-sensitive; PonderNet (Banino et al., arXiv:2107.05407) improves
    stability but adoption at scale is absent; TRM's ablation found ACT slightly *hurt*
    (86.1 vs 87.4 on Sudoku-Extreme). Ouro's entropy-regularized early exit works as an
    *efficiency* feature. We therefore treat halting purely as an inference-FLOP
    optimizer, tested late and cheaply (hypothesis H4).
  - *Titans-style test-time neural memory* (Behrouz et al., arXiv:2501.00663):
    independent reimplementation (Di Nepi et al., arXiv:2510.09551) found chunking
    sensitivity and that memory-only test-time updates with a frozen backbone barely
    help. Delta-rule memory trained end-to-end is the better-evidenced fast-weight
    mechanism (hypothesis H5, deprioritized).
- **State-space models / RWKV.** Mamba (Gu & Dao, arXiv:2312.00752), Mamba2
  (arXiv:2405.21060), RWKV (arXiv:2305.13048) show recurrent state can replace attention
  for much of language modeling but lag on precise retrieval; Gated DeltaNet subsumes
  the mechanism we care about (decaying fast-weight state with targeted updates), so we
  test SSM-style layers through the delta-rule variant rather than separately.

## Strongest architecture hypotheses (ranked, falsifiable)

| ID | Hypothesis | Expected measurable advantage | Status |
|----|-----------|-------------------------------|--------|
| H1 | Shared-weight recurrent latent depth improves reasoning-per-FLOP | Loop-L variant beats loop-1 at equal params; approaches L-deep baseline on Algo-Exec/Rewrite | STRONG - test first |
| H2 | Gated delta-rule fast-weight memory + SWA improves in-session rule learning | Beats Transformer++ on Rule-Shift/Compress/StateGuard at matched FLOPs | STRONG - test second |
| H3 | Deep supervision at recurrent steps (+ iteration distillation) improves latent reasoning | Improves H1 variant at equal FLOPs | MEDIUM - cheap ablation on H1 |
| H4 | Learned halting improves compute allocation | Matches fixed-depth accuracy at lower mean inference FLOPs | WEAK-MIXED - efficiency only, test late |
| H5 | Titans-style test-time memory helps | Beats H2 on retention tasks | DEPRIORITIZED (replication debt) |
| H0a | Null: modern Transformer++ is already hard to beat | Variants fail to separate from B2 | live |
| H0b | Null: CoT at matched inference FLOPs beats latent computation | CoT-trained B2 wins task families | live |

**Key design consequence:** never train the full "Aware LM" (prelude + recurrent core +
delta memory + halting + coda) as a first experiment. Test single mechanisms
independently at 10-50M; compose only survivors.

## Alternative architectures considered

- Pure SSM stack (Mamba-only): weaker precise retrieval at small scale; covered via H2 hybrid.
- Modular memory/reasoning/generation separation: too many confounds for a first
  falsifier; revisit only if H1/H2 both survive.
- Non-transformer discrete program synthesis hybrid: out of scope for the LM claim.
- Bigger plain transformer: not an alternative, it is baseline B2 at 410M (the "just
  scale it" control in the ablation grid).

## What result would be publishable

- **Positive:** a 160M variant beating matched Transformer++ on >= 3/5 SAGE families at
  matched training and inference FLOPs, with the ablation grid attributing the gain to
  the mechanism (not objective/data/leakage), and a flat-or-improving 15M->160M trend.
- **Negative (equally publishable as a workshop/negative-results paper):** a rigorous
  demonstration that recurrent depth and delta-rule memory gains reported in the
  literature do not transfer to matched-FLOP small-model comparisons on procedural
  reasoning benchmarks, with the exact configurations where they vanish.
- **Meta:** evidence on whether a benchmark-first autonomous agent loop finds/kills
  architecture hypotheses faster than manual single-direction iteration (measured by
  hypotheses adjudicated per GPU-hour).

## What result would be useful on GitHub

- SAGE: a seeded, contamination-audited procedural benchmark suite anyone can regenerate.
- A fair-comparison harness: matched-param/FLOP baseline configs + analytic FLOP accounting.
- Clean minimal implementations of loop-core and gated-delta-memory variants.
- The agent workflow itself (backlog, logs, decision policy) as a reusable template.

## 30-second demo script

```
$ aware demo
[1/4] Generating a fresh private DSL (seed 90210)... done. 12 primitives, 3 hidden rules.
[2/4] Model has never seen this DSL. Feeding 8 examples...
[3/4] Query: evaluate (fold (lam x (mul x 2)) (seq 1 4))
      Aware-160M (4 latent steps): 20   [correct]  1.9 GFLOPs
      Transformer++-160M:          14   [wrong]    1.9 GFLOPs
      Transformer++-160M + CoT:    20   [correct]  11.3 GFLOPs
[4/4] Rule shift: 'mul' now means max. Re-testing... Aware adapts in 3 examples, TF++ in 9.
Same params. Same data. FLOPs on the screen. Full logs: experiments/results.csv
```

## Research-paper-style abstract (pre-registered version)

> Small language models are usually improved by adding parameters, data, or longer
> chains of thought. We investigate whether architectural mechanisms — shared-weight
> recurrent latent depth, gated delta-rule fast-weight memory, deep supervision across
> recurrent steps, and learned halting — can instead improve reasoning-per-FLOP and
> performance-per-parameter at the 10M-410M scale. We introduce SAGE, a seeded,
> procedurally generated, contamination-audited benchmark suite measuring algorithmic
> execution, symbolic rule learning, in-session adaptation, and long-context
> compression, and a fairness harness that matches parameters, data, tokenizer,
> training FLOPs, and inference FLOPs against modern Transformer++ baselines, including
> rationale-trained chain-of-thought controls. Experiments are designed, run, logged,
> and adjudicated by an autonomous Cursor research-agent loop with pre-registered
> kill/revise/scale criteria. We report positive and negative results with equal
> weight.

## Conditional success abstract (only if the 160M gate passes)

> We demonstrate that a small language model discovered through a benchmark-first
> autonomous research-agent loop outperforms matched modern transformers on
> reasoning-per-FLOP, in-session rule learning, and symbolic generalization benchmarks
> at matched training and inference FLOPs, with ablations attributing the gain to
> architecture rather than objective, data, or leakage, and with the advantage
> holding from 15M to 160M parameters.
