# EXP-001B: H1 extension — algo_exec, extra seeds, matched-FLOP control

- Hypothesis: H1 (EXTEND round after EXP-001; does not count as a revision)
- Date: 2026-07-04 (launched overnight, pod aware-gate-exp001)
- Status: RUNNING — results on pod at /workspace/exp001b.log, DONE_B.txt marker

## Pre-registered questions and margins (before any results seen)

1. **EXP-001B-ALGO** — loop1 vs loop4 on algo_exec v3 (gate-8-validated, the
   loop-favored serial family). Margin: same as H1 — loop4 - loop1 >= +3.0 pts
   family mean, 3 seeds, diff > 2x pooled seed SD.
2. **EXP-001B-SEEDS** — loop1 vs loop4, seeds 3-5, rewrite + dsl_learn, identical
   protocol to EXP-001. Adjudication pools all 6 seeds (EXP-001 s0-2 + these):
   rewrite delta must clear 2x pooled SD over 6 seeds to count as a pass.
3. **EXP-001B-FM** — matched-training-FLOP control: loop1 at 15000 steps (~2.5x,
   matching loop4's effective-layer FLOP ratio 10/4) vs EXP-001 loop4 @ 6000.
   If loop4 still wins at matched training FLOPs, the gain is architectural.
   If loop1-fm catches up, the "equal params" win was a training-compute artifact
   and the per-FLOP claim dies (KILL-relevant evidence per decision_policy.md).

## Adjudication plan

- H1 PASS requires: margin met on >= 2 families (pooled seeds) AND loop4 advantage
  survives the FM control on at least the family mean.
- Promising-but-short again -> one final EXTEND (longer steps) before revise/park.
- Configs: experiments/configs/exp001b_{algo,seeds,flopmatch}.yaml

## Results

(to be filled after pickup)

## Verdict

(pending)
