# Decision Policy: kill, revise, or scale

Pre-registered margin (all hypotheses unless the backlog entry says otherwise):
**+3.0 accuracy points on the family mean, and the 3-seed mean difference must exceed
2x the pooled seed standard deviation.**

- **KILL** — reserved for CLEAR negatives: variant at or below the matched baseline
  (no positive signal on any family), OR the gain fully vanishes under the
  matched-FLOP control, OR timebox exhausted with no promising signal ever recorded.
  Output: negative-result note in the EXP log + lessons.md entry. Killed is a valid,
  reportable outcome.
- **PROMISING-BUT-SHORT (owner directive, 2026-07-04): do not kill quickly.** If the
  result misses the pre-registered margin but shows a promising signal — positive
  median delta on >= 1 family, or a consistent positive trend across seeds or
  difficulty tiers — the default action is EXTEND, not kill: add seeds, extend
  training, or test the favorable regime, and only then re-adjudicate. A promising
  hypothesis gets its full timebox before any kill decision.
- **REVISE** (max 2 per hypothesis; EXTEND rounds above do not count as revisions) —
  positive but noisy signal, or a *named* implementation confound. The revision must
  state what changes and why the first result doesn't already count as evidence
  against.
- **SCALE** — effect holds on >= 2 task families, >= 3 seeds, at matched params AND
  matched training FLOPs. Scalability considerations are weighed as evidence, not
  gates: (1) 2-size trend (15M -> 50M) from runs we need anyway; a shrinking gap is a
  caution flag that triggers scrutiny, not an automatic kill; (2) hardware-compat
  note reviewed; (3) FLOP/memory overhead behavior vs size recorded. Vertical depth
  (why does it work — ablations, diagnostics) takes priority over multi-size sweeps.

Compute ledger: estimated vs actual GPU-hours per backlog entry; cheapest informative
falsifier runs first.

A/A calibration: if `scripts/aa_test.py` shows identical-model differences at or above
the margin for the current budget, raise seeds/steps before adjudicating any
hypothesis at that budget.
