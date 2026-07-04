# SAGE: Symbolic and Algorithmic Generalization Evaluation

Seven procedural, seeded, parametric-difficulty task families. Every instance is
generated from `(family, difficulty, seed)`; train and eval seed ranges are disjoint by
construction. No task text is scraped from the internet, so pretraining contamination is
structurally impossible for the task *content*; the audit (below) guards the pipeline
anyway.

## Task families

| Family | Tests | Mechanism probed |
|--------|-------|------------------|
| `dsl_learn` | Learn a fresh symbolic mini-language from K examples, then evaluate expressions | abstraction formation, few-shot rule induction |
| `rewrite` | Infer hidden term-rewrite rules from before/after pairs, apply to new terms | symbolic generalization |
| `algo_exec` | Execute small programs (stack VM, list ops, arithmetic chains) | serial computation depth (loop-favored per arXiv:2505.19245) |
| `rule_shift` | Rules change mid-session; measure post-shift adaptation speed | fast-weight memory, in-session learning |
| `compress` | Answer queries requiring compression of long context into reusable state | state compression |
| `state_guard` | Retention / forgetting / interference probes across a session | memory integrity, catastrophic forgetting |
| `fresh_dsl` | Private-seed tiny DSL: execute, write, debug programs; adapt to DSL revision; transfer to sibling DSL | practical transfer (FreshDSL-CodeBench) |

Five **core families** count for the 160M gate: `dsl_learn`, `rewrite`, `algo_exec`,
`rule_shift`, `compress`. `state_guard` and `fresh_dsl` are diagnostic/practical.

## Instance format

Each generator emits three aligned forms per instance:

1. **plain**: `prompt` + `answer` (constrained output format).
2. **traced**: `prompt` + ground-truth step trace + `answer` — required to train the
   CoT baseline fairly (small models cannot produce useful CoT without rationale
   training).
3. **scoring spec**: machine-checkable. Exact match under a constrained grammar;
   program/DSL answers are scored by *execution* against the reference interpreter,
   never by string match.

Difficulty is parametric (e.g., program length, rule count, context length, shift
frequency) with tiers 1-5 per family.

## Dataset generation

- `python scripts/make_data.py --split train --families all --per-family N`
- Train seeds: `[0, 1_000_000)`. Eval seeds: `[2_000_000, 2_100_000)`. Enforced in
  `sage/generators/base.py`; the eval range is refused by the training data writer at
  import level.
- Output: JSONL per family per split under `data/sage/`.

## Contamination audit (`sage/contamination/audit.py`)

1. **Canary GUIDs**: every eval file embeds canary strings
   (`SAGE-CANARY-<guid>`). Any model output or training corpus containing a canary
   fails the audit.
2. **N-gram overlap**: 8-gram and 13-gram overlap between any training corpus
   (including TinyStories) and every eval split; overlap above the noise floor fails.
3. Audit runs before any training job (`make audit`) and its pass-hash is recorded in
   `experiments/results.csv` rows.

## Evaluation metrics

- **Accuracy** per family x difficulty tier (execution-checked where applicable).
- **FLOPs-per-correct-answer** = total inference FLOPs / #correct (primary
  reasoning-per-FLOP metric).
- **Performance-per-parameter** = accuracy at matched training FLOPs vs param count.
- **Adaptation speed** (`rule_shift`): #post-shift examples until accuracy recovers to
  pre-shift level.
- **Retention** (`state_guard`): accuracy on early-session probes late in session.
- Latency (tokens/s) and peak VRAM, reported not gated.

## Benchmark validity gate (runs before any hypothesis training)

1. **Headroom**: a trivial-solver battery (majority answer, copy-longest-literal,
   n-gram LM) must stay below 30% on every family at tier >= 2.
2. **Depth discrimination** (computation families only): a 4-layer vs 8-layer
   matched-width transformer trained identically per-family must separate by
   >= 5 points aggregate on ALL THREE of `dsl_learn`, `rewrite`, `algo_exec`.
3. **Memory-family learnability band**: `rule_shift` and `compress` are memory
   probes (H2 dissociation targets) and are NOT expected to respond to generic
   depth — requiring depth separation on them was internally inconsistent
   (pre-registered correction, 2026-07-04, before gate attempt 5; see
   agent/lessons.md). Their validity criterion: the better model's aggregate must
   land in the (0.10, 0.90) band with at least one tier in (0.15, 0.85) — i.e.,
   learnable, neither floored nor saturated.
4. If any check fails: redesign the benchmark, not the model.
