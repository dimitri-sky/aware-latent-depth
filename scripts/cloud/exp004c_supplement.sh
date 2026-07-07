#!/usr/bin/env bash
# Session C supplement: re-run the 6 grok seeds (14,15,18,19,20,21) killed by the
# startup OOM — 3 delta workers + 2 GPU-lane jobs exceeded the 32GB card; the
# arc-1 memory plan (max 2 delta + 1 GPU, or 3 delta ALONE) was violated by the
# session C script. Waits for the main runner to exit so the delta workers run
# alone, per the plan. Jobs died in backward at step ~1: no eval, no results rows,
# nothing was seen — pure infra re-run, logged in EXP-009.md deviation log.
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0
cd /workspace/aware

while pgrep -f 'run_exp004c.sh' >/dev/null; do sleep 60; done
echo "main runner done $(date -u -Iseconds); starting supplemental grok seeds"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

python scripts/run_exp.py --config experiments/configs/exp009_grok.yaml \
  --seeds 14,15,18,19,20,21 --workers 3

echo EXP004CS_ALL_DONE
