# Phases, Gates, Cloud Plan, Failure Modes

## Phases

- **Phase 0** — research docs (this directory). No code risk.
- **Phase 1** — SAGE generators, scoring harness, contamination audit, FLOP accounting,
  trivial baselines, experiment logging. Runs locally without GPU.
  Exit: **benchmark validity gate** (headroom + 4L-vs-8L discrimination).
- **Phase 2** — autonomous research-agent loop scaffolding (backlog, logs, decision
  policy, guardrail checks, A/A placebo tooling).
- **Phase 3** — 10M-50M theory tests on RTX 3080 + cheap cloud. Fastest falsifier first:
  ~15M loop-only at loop counts {1,4} vs depth-matched Transformer++ on
  algo_exec + rewrite, 3 seeds.
- **Phase 4** — 160M Transformer++ vs strongest surviving variant, matched data/FLOPs
  (cloud). **Gate:** variant beats B2 on >= 3 of 5 core SAGE families at matched
  training AND inference FLOPs. The 15M->50M->160M trend is reported alongside and
  weighed in the 410M go/no-go, but the gate is decided on the 160M head-to-head.
- **Phase 5** — full ablation grid (see below).
- **Phase 6** — 410M only if the gate passes; 760M only if 160M->410M scaling is clean.
- **Phase 7** — demo CLI, checkpoint release, claim table, plots, CSVs, report
  (positive or negative).

## Ablation grid (Phase 5; each row answers one named confound)

| Ablation | Confound addressed |
|----------|--------------------|
| standard transformer, same params | is Transformer++ even needed? |
| Transformer++, same params | architecture vs recipe |
| matched training FLOPs | training-compute confound |
| matched inference FLOPs | inference-compute confound |
| no recurrence (loop=1) | does looping do anything? |
| fixed recurrence depth | halting vs fixed depth |
| learned halting disabled | halting contribution |
| no fast-weight memory | memory contribution |
| no delta-rule (additive fast weights) | update-rule choice |
| no deep supervision | objective confound |
| no iteration distillation | objective confound |
| plain next-token objective only | objective confound (all extras off) |
| no curriculum / shuffled curriculum | data-order confound |
| corrupted state at session boundary | is state actually used? |
| same examples in prompt, no state update | prompt memory vs neural memory |
| CoT at matched inference FLOPs | latent vs explicit reasoning |
| larger transformer baseline (410M B2 vs 160M variant) | "just scale it" |

## Cloud workflow

1. Local first: `make data`, `make train_tiny`, `make eval`, `make ablate`,
   `make report` must all work on the 3080 box.
2. Then a thin `scripts/cloud_launch.py` wrapper (RunPod/Lambda/Modal — provider chosen
   at Phase 3 exit based on price/GPU availability; prefer L40S/A100 for 50-160M,
   H100 for 410M).
3. Tracking: CSVs are the source of truth; W&B optional overlay.
4. MCP only after cloud scripts work, and only for: launching jobs, checking logs,
   pulling result CSVs, comparing runs, creating GitHub issues, syncing reports.

## RTX 3080 local workflow (10 GB)

- Repo dev, SAGE generation, CPU tests, tokenizer, contamination audit: CPU-only.
- 10-25M fp16/bf16 training runs, benchmark validity gate, agent-loop dryruns: 3080.
- 50M smoke tests: 3080 with grad accumulation (slow) or cheap cloud.
- Serious 160M+ runs: cloud only.

## Scaling plan

10M-15M (theory falsifiers, 3080) -> 50M (smoke, 3080/cloud) -> 160M (first serious
test, cloud, gated) -> 410M (only if gate passes) -> 760M (only if scaling clean).

## Failure modes tracked

| Failure | Detector / mitigation |
|---------|----------------------|
| Benchmark too easy or gameable | validity gate headroom check; trivial-solver battery |
| Recurrence gain is a depth/FLOP artifact | matched-FLOP controls B2-TF/B2-IF |
| BPTT instability at loop > 4 | truncated BPTT; stochastic loop-count sampling; gradient-norm logging |
| Delta memory helps retrieval, not reasoning | rule_shift vs algo_exec dissociation report |
| CoT wins everything at matched FLOPs | valid negative result (H0b confirmed); write it up |
| Agent self-deception / cherry-picking | A/A placebo runs, frozen baselines, append-before-verdict CSV rule |
| Scorer bug creates fake gains | known-answer tests in tests/, run before logging |
| 3080 too small for honest 50M runs | move those runs to cloud earlier |
| Seed noise swamps effect at 15M | 3 seeds minimum, pre-registered margin vs pooled SD |
