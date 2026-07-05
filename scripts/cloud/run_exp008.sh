#!/usr/bin/env bash
# EXP-008 (50M scale check) + EXP-002-DL (dsl_learn family extension + demo
# checkpoints). Phase A interleaves CPU-bound 18M V2 jobs with the GPU-bound
# B2 jobs; phase B is the V3-50M solo chain (VRAM-bound).
# No `set -e`: the sentinel must appear even after partial failure.
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

cd /workspace/aware

if [ ! -f data/sage/train/dsl_learn.jsonl ]; then
  python scripts/make_data.py --split train --per-family 20000 --families dsl_learn
  python scripts/make_data.py --split eval --per-family 400 --families dsl_learn
  python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval
fi

echo "=== PHASE A: EXP-002-DL V2 (2 workers, CPU) + GPU chain (B2-DL, B2-50M) ==="
python scripts/run_exp.py --config experiments/configs/exp002_dsl_learn.yaml \
  --models V2-delta --workers 2 > /workspace/exp002dl_v2.log 2>&1 &
DL_PID=$!
python scripts/run_exp.py --config experiments/configs/exp002_dsl_learn.yaml --models B2-6L --workers 1
python scripts/run_exp.py --config experiments/configs/exp008_algo_exec.yaml --models B2-50M --workers 1
python scripts/run_exp.py --config experiments/configs/exp008_state_guard.yaml --models B2-50M --workers 1
wait $DL_PID

echo "=== PHASE B: V3-50M solo chain ==="
python scripts/run_exp.py --config experiments/configs/exp008_algo_exec.yaml --models V3-50M --workers 1
python scripts/run_exp.py --config experiments/configs/exp008_state_guard.yaml --models V3-50M --workers 1

echo EXP008_ALL_DONE
