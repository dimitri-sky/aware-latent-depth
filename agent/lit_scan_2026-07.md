# Literature re-scan: recurrent/looped LLMs since Jan 2026

Source: owner-provided zip (papers_2026/recurrent_llm_papers/, 15 arXiv PDFs).
Digested 2026-07-04 by three parallel reviews. Actionable distillate:

## Confirmations of our results

- EXP-001B's H1 kill (dense loops lose at matched training FLOPs) replicates
  independently at scale: Sparse-Layers (2605.09165, isoFLOP 16M-305M), SpiralFormer
  (2602.11698, anchor LoopedFormer control), LoopFormer (2602.11451, iso-FLOP ~1B).
  Vanilla weight-tied dense loops are param-efficient, never FLOP-efficient.

## Ingredients that make loops pay (evidence-ranked)

1. **Per-iteration functional diversity** — break "same map R times": MoE routing
   diverges per pass (2605.09165); multi-resolution coarse-to-fine iteration
   (2602.11698, the clearest per-FLOP win: fewer FLOPs AND better accuracy).
2. **Step-aligned supervision at loop readouts** — LOTUS (2606.31779): +6.7 pts
   from supervising latent steps; unsupervised intermediate loop states are
   semantically inert (2601.10242 probing).
3. **Gated state carried across iterations, trained with per-loop supervision** —
   MELT (2605.07721): elementwise learned gate >> mean/EMA/last; chunk-wise
   training load-bearing. This is our H1+H2 fusion, validated at 1.6B (per-memory
   win; per-FLOP untested).
4. **Randomized loop counts + truncated BPTT + input injection** — RD-VLA
   (2602.07845): log-normal depth sampling, TBPTT-8; post-hoc variable depth
   fails (CART 2606.01495).
5. **Convergence-based halting** — inference-FLOP savings only (34% at parity,
   RD-VLA); never a training-FLOP win.

## Traps to control for

- **Readout blind spot** (2606.24898): per-loop CE through a normalized readout can
  yield a loop-invariant model that looks healthy on final loss. Diagnostic
  required: accuracy/CE gap between K=1 and K=trained evaluations of the SAME
  checkpoint; also track per-loop hidden-state norms.
- **Cross-iteration state is fragile as memory** (CART: gate discards 69-85%).
  Sequence-dim memory (delta layers) inside the looped block, not loop-carried
  state, is the supported combination topology.
- **Middle-cycle allocation**: loop ~30-40% of layers (prelude/coda unlooped),
  U-shaped optimum (2602.11698); full-stack looping is dominated.
- Short-budget screens can invert loop-count rankings at 10x training (CART);
  extend-before-adjudicate on any surprising ordering.

## Program implications

- Stage A: EXP-002 (H2 falsifier) unchanged — delta+SWA hybrid vs param-matched B2.
- Stage B: EXP-003 upgraded from "flip deep_supervision" to the full recipe:
  per-loop readout supervision + randomized loop count + TBPTT + K-gap diagnostic,
  with the matched-training-FLOP control arm in the same batch (lesson from
  EXP-001B).
- Stage C: EXP-005 "V3" — loop the {delta + SWA} hybrid block (MELT topology at
  15M) on memory + computation families vs param-matched AND FLOP-matched
  baselines. Niche: nobody has published this with honest FLOP accounting on
  procedural reasoning.
