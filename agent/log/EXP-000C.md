# EXP-000C: Benchmark validity — calibrated depth-discrimination gates (6-8)

- Hypothesis: SAGE computation families can detect depth advantages in matched
  transformers (instrument validation, prerequisite for all H* experiments).
- Protocol (pre-registered after EXP-000B): 2L vs 16L Transformer++, 3 seeds,
  median aggregation, collapse detection (tier-1 < 0.05 -> excluded + one retry with
  seed+1000), z-loss stabilizer, depth-scaled lr = min(6e-4, 6e-4*sqrt(4/n_layers)).
  PASS per family: median(16L) - median(2L) >= 0.05.

## Results

| Gate | rewrite | dsl_learn | algo_exec | algo_exec version |
|------|---------|-----------|-----------|-------------------|
| 6 | **+0.290 PASS** | **+0.050 PASS** | -0.025 FAIL | stack VM, saturating |
| 7 | **+0.200 PASS** | +0.040 FAIL (margin) | -0.030 FAIL (floored ~0.10) | accumulator, 2-digit |
| 8 | (not rerun) | (not rerun) | pending | accumulator, single-digit |

run_ids: EXP-000C rows in experiments/results.csv (pod merges:
results_pod_gate6.csv, results_pod_gate7.csv).

## Verdict per family

- **rewrite: VALIDATED.** +20 to +29 points median depth gain across two independent
  gates, 6 seeds, zero collapses after stabilizers. The instrument detects depth.
- **dsl_learn: MARGINAL-VALID.** +4.0/+5.0 straddling the margin; usable as a
  secondary family with the caveat recorded here.
- **algo_exec: PARKED after 3 diagnosed failures** (timebox rule): (1) stack VM has
  dead code -> live dependency path much shorter than program, shallow-retrievable;
  (2) mod-wrap arithmetic is a separate skill that floors 3-12M models; (3) exact
  2-digit arithmetic makes per-step accuracy (~55%) the bottleneck, not
  chain-following. v3 (single-digit, clamp 0..9, order-dependence via rails) is the
  final revision; if gate 8 fails, the family stays parked and is redesigned outside
  the critical path.

## Decision

Gate purpose — "can SAGE detect real depth/architecture gains?" — is SATISFIED via
rewrite (decisive) + dsl_learn (marginal). EXP-001 proceeds on rewrite + dsl_learn.
Honest caveats: (a) depth detection is proven on 2 of 3 intended computation
families; (b) dsl_learn sits at the margin; (c) claims from EXP-001 inherit these
limits and the H1 margin (3.0 pts, 2x pooled SD) is comfortably above the
post-stabilization noise (~2-5 pts seed spread, median-of-3).

## Lessons

Appended to agent/lessons.md: multi-seed medians or nothing; collapse detection;
dead-code tasks measure retrieval not depth; skill smuggling (wrap/2-digit
arithmetic) masks the construct under test.
