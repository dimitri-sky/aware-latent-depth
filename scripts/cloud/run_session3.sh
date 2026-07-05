#!/usr/bin/env bash
# Session 3: EXP-007 (budget robustness, 8000 steps) + EXP-005-DEN (density
# sweep). All delta jobs are CPU-bound (~10GB each); B2 jobs GPU-bound.
# Memory plan (32GB): max 2 delta workers + 1 GPU job, or 3 delta workers alone.
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
  python scripts/make_data.py --split train --per-family 20000 --families algo_exec
  python scripts/make_data.py --split eval --per-family 400 --families algo_exec
  python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval
fi

python -m pytest tests/ -q || echo "WARN: tests failed, continuing (results flagged)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo "=== PHASE A: EXP-007 V2 @8000 (2 workers, CPU-bound) + B2 @8000 (GPU) ==="
python scripts/run_exp.py --config experiments/configs/exp007_algo_exec.yaml \
  --models V2-delta --workers 2 > /workspace/exp007_v2.log 2>&1 &
V2_PID=$!
python scripts/run_exp.py --config experiments/configs/exp007_algo_exec.yaml --models B2-6L --workers 1

echo "=== PHASE B: density d1 (1 worker alongside remaining V2 load) ==="
python scripts/run_exp.py --config experiments/configs/exp005_density.yaml --models V3-d1 --workers 1
wait $V2_PID

echo "=== PHASE C: density d3 (2 workers, V2 chain done) ==="
python scripts/run_exp.py --config experiments/configs/exp005_density.yaml --models V3-d3 --workers 2

echo SESSION3_ALL_DONE
