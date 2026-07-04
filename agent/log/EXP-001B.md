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

Pod aware-exp001b (RTX 4090), all 15 jobs ok, ~1h wall. run_ids: EXP-001B-ALGO-*,
EXP-001B-SEEDS-*, EXP-001B-FM-* in experiments/results.csv. Adjudication script:
scripts/adjudicate_exp001b.py.

1. **ALGO** — algo_exec, 3 seeds: loop1 median 0.800, loop4 median 0.775,
   delta mean **-2.2 pts** (FAIL). No loop advantage on the loop-favored family.
   Caveat: loop1 already near the task's observed ceiling (~0.85).
2. **SEEDS** — pooled 6 seeds: rewrite delta mean **+9.8 pts**, 5/6 seeds positive,
   but 2xSD = 18.2 > 9.8 (FAIL, noise gate). dsl_learn mean **-0.1 pts** — the
   EXP-001 dsl_learn signal did not replicate on seeds 3-5.
3. **FM control** — loop1 @ 15000 steps (matched training FLOPs) vs loop4 @ 6000:
   rewrite 0.520 vs 0.510 (tie), dsl_learn **0.210 vs 0.155** (plain model wins).
   The equal-params loop gain is a training-compute artifact.

## Verdict: KILL (per pre-registered adjudication)

The FM condition triggered the explicit kill criterion pre-registered above and in
decision_policy.md: the gain vanishes at matched training FLOPs, and loop4 costs
~2.5x inference FLOPs, so reasoning-per-FLOP is strictly worse. Owner's
promising-but-short directive was honored: this verdict comes after a full EXTEND
round (algo_exec + 6 pooled seeds + control), not a premature kill.

Honest residual: at equal *parameters*, loop4 retains a consistent rewrite
advantage (+9.8 mean, 5/6 seeds). Loops may buy parameter efficiency, not FLOP
efficiency (consistent with Ouro's actual claim). Recorded as a lesson; a
param-efficiency-framed hypothesis may be re-proposed later if a use case (memory-
bound deployment) makes parameter count the binding constraint.

## Lesson

Equal-params comparisons flatter recurrence: extra loops are extra compute, so a
matched-training-FLOP control must run in the SAME batch as any loop experiment,
or the first result will overstate the effect. Appended to agent/lessons.md.

## Next action

EXP-002 (H2, gated delta-rule memory): 15M V2 vs param-matched B2 on rule_shift +
compress + state_guard, 3 seeds, matched tokens; dissociation check per backlog.
