#!/usr/bin/env bash
# EXP-002 pod runner (v3, critical-path scheduling): the 9 V2-delta jobs are
# ~100 min each (sequential-scan bound) while the 9 B2 jobs are ~30 min, so all
# V2 jobs run FIRST at max safe concurrency (6 x ~4.5GB = 27GB of 32GB), then the
# B2 jobs (3 at a time; ~8.5GB each). Same jobs and protocol as v2, reordered.
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
  # source may have been refreshed since first unpack (run_exp.py --models flag)
  unzip -qo /workspace/aware_src.zip scripts/run_exp.py
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== PHASE 1: all V2-delta jobs (critical path), 2 workers x 3 configs = 6 concurrent ==="
python scripts/run_exp.py --config experiments/configs/exp002_rule_shift.yaml --models V2-delta --workers 2 &
python scripts/run_exp.py --config experiments/configs/exp002_compress.yaml --models V2-delta --workers 2 &
python scripts/run_exp.py --config experiments/configs/exp002_state_guard.yaml --models V2-delta --workers 2 &
wait

echo "=== PHASE 2: all B2 jobs, 3 concurrent per config, configs sequential ==="
python scripts/run_exp.py --config experiments/configs/exp002_rule_shift.yaml --models B2-6L --workers 3
python scripts/run_exp.py --config experiments/configs/exp002_compress.yaml --models B2-6L --workers 3
python scripts/run_exp.py --config experiments/configs/exp002_state_guard.yaml --models B2-6L --workers 3

echo EXP002_ALL_DONE
