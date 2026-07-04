# Autonomous Cursor Research-Agent Design

The agent is a Cursor agent operating on plain files and scripts in this repo. State
lives in markdown and CSV, never in the agent's memory. No MCP until training/eval
scripts work; MCP later only as a convenience layer (job launch, log pull, CSV sync,
GitHub issues).

## Agent loop (per hypothesis)

1. Write the hypothesis clearly in `agent/backlog.md`.
2. Define the expected measurable advantage (metric, family, margin).
3. Define the minimum experiment that could falsify it.
4. Implement the smallest possible code change (config-first; new module only if needed).
5. Run a tiny local experiment (CPU or RTX 3080).
6. Compare against the matched baseline (same data, tokens, params, FLOPs).
7. Save metrics to `experiments/results.csv`.
8. Update `agent/log/EXP-<id>.md`.
9. Decide: **kill / revise / scale** per the decision policy.
10. Append the transferable insight to `agent/lessons.md` and propose the next best
    experiment.

## Agent task format (backlog entry)

```markdown
### H<id>: <one-sentence hypothesis>
- Status: proposed | testing | killed | revised | scaled
- Expected advantage: <metric> on <families> by >= <margin> at matched <params/FLOPs>
- Minimum falsifier: <exact smallest experiment>
- Evidence: <paper links / EXP log links>
- Hardware-compat note: <design-time path-to-scale check>
- Est. cost: <GPU-h> | Actual: <GPU-h>
- Timebox: <max revisions / wall-clock before forced kill-or-park>
```

## Experiment-log format (`agent/log/EXP-<id>.md`)

```markdown
# EXP-<id>: <title>
- Hypothesis: H<id>
- Date / git sha / config hash
- Config diff vs baseline (exact)
- Command(s) run (copy-pasteable)
- Seeds: [...]
- Results: link to results.csv rows (run_ids)
- Verdict: kill | revise | scale (+ reasoning against pre-registered margin)
- Confounds checked / remaining
- Next action
```

## Decision policy: kill, revise, or scale

- **KILL** if the variant fails to beat the matched baseline by the pre-registered
  margin on >= 3 seeds, or the gain vanishes under the matched-FLOP control.
- **REVISE** (max 2 revisions per hypothesis) if the signal is positive but noisy, or a
  specific implementation confound is identified and named.
- **SCALE** only if the effect holds on >= 2 task families, >= 3 seeds, at matched
  params AND matched training FLOPs. Scalability considerations (2-size trend,
  hardware-compat note, overhead behavior) are weighed as evidence, not applied as an
  automatic gate. Vertical depth (why does it work?) takes priority over multi-size
  sweeps.

Default pre-registered margin: **+3.0 accuracy points on the family mean, and the
3-seed mean difference must exceed 2x the pooled seed standard deviation.**

## Guardrails (enforced)

1. No unsourced paper claims — every literature statement carries an arXiv/OpenReview link.
2. No changing baselines unfairly — baseline configs are frozen by hash
   (`experiments/configs/frozen/`); any change requires a new baseline ID.
3. No hidden data changes — data generation is seeded and logged; corpus hash recorded per run.
4. No result cherry-picking — every run appends to `results.csv` before the verdict is
   written; A/A placebo runs are executed periodically and must show no effect.
5. No large run without small-run signal recorded in an EXP log.
6. No architecture complexity without an ablation reason written in the backlog.
7. No claim without CSV or plot evidence (run_id + git sha).
8. No benchmark leakage — disjoint seed ranges at import level + contamination audit
   before every training job.
9. No vague success criteria — margins pre-registered in the backlog entry.
10. Negative results are valid outputs and get full write-ups.

## Self-checks

- **A/A placebo:** run baseline-vs-identical-baseline through the full compare pipeline;
  any "significant" difference means the harness/stats are broken.
- **Scorer correctness tests:** `pytest tests/` must pass before any experiment is logged.
- **Compute ledger:** estimated vs actual GPU-hours per backlog entry; cheapest
  informative falsifier is prioritized.
- **Timeboxes:** forced kill-or-park at the timebox; prevents brute-forcing one direction.
- **Literature re-scan:** at each phase boundary, re-search live hypotheses and log deltas.
