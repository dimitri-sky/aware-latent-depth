#!/usr/bin/env bash
# EXP-003 (H3 recipe loops) + EXP-002-AX (H2 dissociation) on pod B, reusing the
# EXP-002 environment. Scheduling exploits the workload split: V2-delta jobs are
# CPU-bound (~12k tok/s sequential scan, ~10GB each) and run in the BACKGROUND at
# 2 workers, while the GPU-bound loop jobs stream through the foreground at
# 1 worker (memory headroom: 2x10 + ~7 = ~27GB of 32GB; compress OOM lesson).
# No `set -e`: the sentinel must appear even after partial failure.
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

cd /workspace/aware

# Refresh source: H3 code (loop_randomize, deep supervision, K-gap diagnostic)
# and the exp003/exp002_algo_exec configs postdate the archive this pod booted on.
unzip -qo /workspace/aware_src.zip

# New families for this stage (rule_shift/compress/state_guard already on disk).
if [ ! -f data/sage/train/algo_exec.jsonl ]; then
  python scripts/make_data.py --split train --per-family 20000 --families rewrite,algo_exec
  python scripts/make_data.py --split eval --per-family 400 --families rewrite,algo_exec
  python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval
fi

python -m pytest tests/ -q || echo "WARN: tests failed, continuing (results flagged)"

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "=== BACKGROUND: EXP-002-AX V2-delta (CPU-bound, 2 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp002_algo_exec.yaml \
  --models V2-delta --workers 2 > /workspace/exp002_ax.log 2>&1 &
AX_PID=$!

echo "=== FOREGROUND: EXP-003 loop chain (GPU-bound, 1 worker) ==="
python scripts/run_exp.py --config experiments/configs/exp003_recipe.yaml --models V1R-loop4 --workers 1
python scripts/run_exp.py --config experiments/configs/exp003_recipe.yaml --models V1-loop1 --workers 1
python scripts/run_exp.py --config experiments/configs/exp003_control.yaml --workers 1

echo "=== loop chain done; waiting for EXP-002-AX V2 ==="
wait $AX_PID

echo "=== EXP-002-AX B2 (fast, 3 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp002_algo_exec.yaml --models B2-6L --workers 3

echo EXP003_ALL_DONE
