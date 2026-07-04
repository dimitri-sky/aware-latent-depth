#!/usr/bin/env bash
# EXP-002 pod B (v4): compress V2 then rule_shift V2, 3 workers each (one wave
# per config; 3 x 9GB fits 32GB), then B2 compress. Counterpart to pod A.
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

echo "=== POD B PHASE 1: compress V2 then rule_shift V2 (3 workers, one wave each) ==="
python scripts/run_exp.py --config experiments/configs/exp002_compress.yaml --models V2-delta --workers 3
python scripts/run_exp.py --config experiments/configs/exp002_rule_shift.yaml --models V2-delta --workers 3

echo "=== POD B PHASE 2: B2 compress (3 workers) ==="
python scripts/run_exp.py --config experiments/configs/exp002_compress.yaml --models B2-6L --workers 3

echo EXP002_ALL_DONE
