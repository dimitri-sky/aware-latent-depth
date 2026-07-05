#!/usr/bin/env bash
# EXP-008 (50M scale check). V3-50M jobs run SOLO (VRAM); B2-50M jobs are
# GPU-bound and quick. Assumes the probe already passed (launcher gates on it).
# No `set -e`: the sentinel must appear even after partial failure.
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

cd /workspace/aware

echo "=== B2-50M first (fast, both families) ==="
python scripts/run_exp.py --config experiments/configs/exp008_algo_exec.yaml --models B2-50M --workers 1
python scripts/run_exp.py --config experiments/configs/exp008_state_guard.yaml --models B2-50M --workers 1

echo "=== V3-50M solo chain ==="
python scripts/run_exp.py --config experiments/configs/exp008_algo_exec.yaml --models V3-50M --workers 1
python scripts/run_exp.py --config experiments/configs/exp008_state_guard.yaml --models V3-50M --workers 1

echo EXP008_ALL_DONE
