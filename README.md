# Aware

**Benchmark-first research program testing whether small LMs can scale
reasoning-per-FLOP through recurrent latent depth and fast-weight memory, run by an
autonomous Cursor research-agent loop. Negative results welcome.**

> Aware tests whether small language models can scale intelligence through adaptive
> latent depth and fast-weight memory instead of only through more parameters, more
> tokens, and longer chain-of-thought.

## Layout

```
docs/         CONCEPT, BENCHMARK (SAGE spec), AGENT, BASELINES, PLAN
sage/         benchmark generators, scoring, contamination audit, FLOP accounting
models/       GPT-2 baseline, Transformer++, loop-core variant, delta-memory variant
train/        shared training loop, data pipeline, byte tokenizer
eval/         evaluation harness, reports
agent/        hypothesis backlog, experiment logs, decision policy, lessons
experiments/  configs (frozen baselines) + results.csv (source of truth)
scripts/      make_data, train_tiny, eval, ablate, report, aa_test
tests/        scorer/generator/FLOP correctness tests (run before any experiment is logged)
```

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
make data          # generate SAGE splits (CPU)
make test          # scorer/generator/FLOP correctness
make audit         # contamination audit
make validity      # benchmark validity gate (trivial solvers + 4L-vs-8L)
make train_tiny    # first falsifier: 15M loop-1 vs loop-4 vs deep baseline
make report        # tables/plots from experiments/results.csv
```

## Rules of evidence

Every claim links a `results.csv` run_id + git sha. Baselines are frozen by config
hash. Train/eval seed ranges are disjoint at import level. Contamination audit runs
before every training job. Negative results get full write-ups in `agent/log/`.
