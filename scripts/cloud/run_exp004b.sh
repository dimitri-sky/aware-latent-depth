#!/usr/bin/env bash
# EXP-004/EXP-009 session B (runs after session C, the canonical EXP-004 re-run):
# remaining grok seeds 22-29 + labeled seed-2 re-run (3 CPU-bound workers), THEN
# the optional V3-CoT-long arm (2 workers; max 3 delta workers at once per the
# arc-1 memory plan, so it runs after).
# No `set -e`: the sentinel must appear even after partial failure.
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

cd /workspace
if [ ! -d /workspace/aware ]; then
  mkdir aware && cd aware
  unzip -q /workspace/aware_src.zip
  pip install -q --break-system-packages numpy pyyaml pytest tqdm matplotlib
else
  cd /workspace/aware
  unzip -qo /workspace/aware_src.zip
fi

if [ ! -f data/sage/train/algo_exec.jsonl ]; then
  python scripts/make_data.py --split train --per-family 20000 --families algo_exec,rule_shift
  python scripts/make_data.py --split eval --per-family 400 --families algo_exec,rule_shift
fi
python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval \
  || { echo "AUDIT FAILED - aborting"; echo EXP004B_ALL_DONE; exit 1; }

python -m pytest tests/ -q || echo "WARN: tests failed, continuing (results flagged)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "=== CPU lane: EXP-009 grok seeds 22-29 + seed-2 bonus (3 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp009_grok.yaml \
  --seeds 22,23,24,25,26,27,28,29 --workers 3
echo "=== EXP-009 bonus: labeled re-run of grokked seed 2 (excluded from rate) ==="
python scripts/run_exp.py --config experiments/configs/exp009_grok.yaml \
  --seeds 2 --workers 1

echo "=== optional arm (time-gated): V3-CoT-long on algo_exec s0-1 (2 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp004_algo_exec.yaml \
  --models V3-CoT-long --seeds 0,1 --workers 2 > /workspace/exp004_v3cot.log 2>&1
tail -3 /workspace/exp004_v3cot.log

echo EXP004B_ALL_DONE
