# experiments/

- `results.csv` — append-only source of truth. Rows are appended by the eval harness
  BEFORE any verdict is written into an EXP log (guardrail 4).
- `configs/` — YAML experiment configs. `configs/frozen/` holds hash-frozen baseline
  configs; any modification requires a new baseline ID (guardrail 2).

## results.csv columns

| column | meaning |
|--------|---------|
| run_id | unique id `<exp>-<model>-s<seed>-<shorthash>` |
| git_sha | commit the run was made from |
| config_hash | sha256 of the resolved config |
| exp_id | EXP log id (agent/log/EXP-xxx.md) |
| model_id | B1/B2/V1-loop4/... |
| params | actual parameter count |
| family | SAGE family or `lm` (TinyStories val loss) |
| difficulty | tier 1-5 or `all` |
| metric | accuracy / val_loss / adaptation_speed / retention / flops_per_correct |
| train_tokens, train_flops | training budget actually spent |
| infer_flops_per_correct | analytic inference FLOPs / #correct |
| audit_hash | contamination-audit pass hash |
