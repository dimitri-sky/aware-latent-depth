#!/usr/bin/env bash
# EXP-002 pod runner (v2, VRAM-safe): idempotent setup, then the three H2 groups
# SEQUENTIALLY with 4 workers each. v1 ran all 18 jobs concurrently and OOMed a
# 32GB 5090 (V2-delta ~4.5GB/job steady, B2 peaks ~8.5GB). 4 concurrent jobs is
# the safe envelope; expandable_segments reduces fragmentation.
# No `set -e`: the sentinel must appear even after partial failure.
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
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/run_exp.py --config experiments/configs/exp002_rule_shift.yaml --workers 4
python scripts/run_exp.py --config experiments/configs/exp002_compress.yaml --workers 4
python scripts/run_exp.py --config experiments/configs/exp002_state_guard.yaml --workers 4

echo EXP002_ALL_DONE
