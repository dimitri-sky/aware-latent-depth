#!/usr/bin/env bash
# Mini-session: retrain the 18M algo_exec demo pair (V2-delta + B2, seed 0 —
# protocol identical to EXP-002-AX; checkpoints died with terminated pods).
# Runs both concurrently (V2 CPU-bound ~10GB, B2 GPU-bound ~8.5GB).
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

python scripts/run_exp.py --config experiments/configs/exp002_algo_exec.yaml \
  --models V2-delta --seeds 0 --workers 1 > /workspace/demo_v2.log 2>&1 &
V2_PID=$!
python scripts/run_exp.py --config experiments/configs/exp002_algo_exec.yaml \
  --models B2-6L --seeds 0 --workers 1
wait $V2_PID

echo DEMOCKPT_ALL_DONE
