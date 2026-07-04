#!/usr/bin/env bash
# Pod B supplement 2: compress V2 needs ~15.7GB/job; even 2 workers OOM (seeds 0
# and 2 both died, s1 survived alone). Wait for supplement 1, then rerun seeds 0
# and 2 SEQUENTIALLY solo, then write the final sentinel.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

while pgrep -f 'podB_supplement.sh' >/dev/null || pgrep -f 'train_single.py' >/dev/null; do
  sleep 30
done

cd /workspace/aware
python scripts/run_exp.py --config experiments/configs/exp002_compress.yaml --models V2-delta --seeds 0,2 --workers 1 >> /workspace/exp002.log 2>&1

echo EXP002_FINAL_DONE >> /workspace/exp002.log
