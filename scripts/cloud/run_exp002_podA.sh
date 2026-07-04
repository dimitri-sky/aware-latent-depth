#!/usr/bin/env bash
# EXP-002 pod A (v4): V2 jobs need 7-9GB each (backward through the sequential
# scan), so max 3 concurrent per 32GB GPU. The 9-job V2 critical path is split
# across two pods: A = state_guard V2 + B2 for state_guard/rule_shift,
# B = compress + rule_shift V2 + B2 compress.
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

cd /workspace
if [ ! -f /workspace/aware/data/sage/train/rule_shift.jsonl ]; then
  rm -rf aware && mkdir aware && cd aware
  unzip -q /workspace/aware_src.zip
  pip install -q --break-system-packages numpy pyyaml pytest tqdm matplotlib
  FAMS=rule_shift,compress,state_guard
  python scripts/make_data.py --split train --per-family 20000 --families $FAMS
  python scripts/make_data.py --split eval --per-family 400 --families $FAMS
  python -m pytest tests/ -q
  python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval
else
  cd /workspace/aware
  unzip -qo /workspace/aware_src.zip scripts/run_exp.py
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== POD A PHASE 1: state_guard V2 (3 workers, one wave) ==="
python scripts/run_exp.py --config experiments/configs/exp002_state_guard.yaml --models V2-delta --workers 3

echo "=== POD A PHASE 2: B2 state_guard + rule_shift (3 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp002_state_guard.yaml --models B2-6L --workers 3
python scripts/run_exp.py --config experiments/configs/exp002_rule_shift.yaml --models B2-6L --workers 3

echo EXP002_ALL_DONE
