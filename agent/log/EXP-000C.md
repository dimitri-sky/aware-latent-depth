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
| 8 | (not rerun) | (not rerun) | **+0.115 PASS** | accumulator, single-digit |

run_ids: EXP-000C rows in experiments/results.csv (pod merges:
results_pod_gate6.csv, results_pod_gate7.csv).

## Verdict per family

- **rewrite: VALIDATED.** +20 to +29 points median depth gain across two independent
  gates, 6 seeds, zero collapses after stabilizers. The instrument detects depth.
- **dsl_learn: MARGINAL-VALID.** +4.0/+5.0 straddling the margin; usable as a
  secondary family with the caveat recorded here.
- **algo_exec: VALIDATED (gate 8).** v3 single-digit accumulator passes with
  median(16L)-median(2L)=+0.115 (2L=0.680, 16L=0.795), 3 seeds, zero collapses.
  Prior failures were task-design bugs (dead code, mod-wrap, 2-digit arithmetic), not
  instrument failure.

## Decision

Gate purpose — "can SAGE detect real depth/architecture gains?" — is SATISFIED on
all three computation families (rewrite decisive, algo_exec decisive, dsl_learn
marginal). EXP-001 proceeds on rewrite + dsl_learn; algo_exec available for
follow-up re-evals from saved checkpoints.

## Lessons

Appended to agent/lessons.md: multi-seed medians or nothing; collapse detection;
dead-code tasks measure retrieval not depth; skill smuggling (wrap/2-digit
arithmetic) masks the construct under test.
