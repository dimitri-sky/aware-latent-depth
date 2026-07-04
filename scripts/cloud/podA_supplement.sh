#!/usr/bin/env bash
# Pod A supplement: state_guard V2 OOMed at 3 workers (11.5-12.3GB per job — the
# longest sequences in SAGE; only 2 fit on 32GB). Wait for the main runner to
# finish its B2 phases, rerun state_guard V2 at 2 workers, then write the REAL
# completion sentinel that the (re-pointed) watcher greps for.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AWARE_THROTTLE=0

while pgrep -f 'bash /workspace/run_exp002.sh' >/dev/null; do
  sleep 30
done

cd /workspace/aware
python scripts/run_exp.py --config experiments/configs/exp002_state_guard.yaml --models V2-delta --workers 2 >> /workspace/exp002.log 2>&1

echo EXP002_REALLY_ALL_DONE >> /workspace/exp002.log
