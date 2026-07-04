#!/usr/bin/env bash
# Pod B supplement: compress V2 OOMed at 3 workers (~11GB each). After the main
# runner finishes, rerun compress V2 at 2 workers, then write the real sentinel.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

while pgrep -f 'bash /workspace/run_exp002.sh' >/dev/null; do
  sleep 30
done

cd /workspace/aware
python scripts/run_exp.py --config experiments/configs/exp002_compress.yaml --models V2-delta --workers 2 >> /workspace/exp002.log 2>&1

echo EXP002_REALLY_ALL_DONE >> /workspace/exp002.log
