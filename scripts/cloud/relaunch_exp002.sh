#!/usr/bin/env bash
# Kill any previous EXP-002 processes and start the sequential VRAM-safe runner
# plus the auto-stop watcher (with RunPod env imported from PID 1).
pkill -f 'bash /workspace/run_exp002.sh'
pkill -f 'exp002_finish_and_stop'
pkill -f 'scripts/run_exp.py'
pkill -f 'scripts/train_single.py'
sleep 3

rm -f /workspace/exp002.log
rm -f /workspace/aware/experiments/results_EXP-002*.csv

nohup bash /workspace/run_exp002.sh > /workspace/exp002.log 2>&1 &
sleep 2

export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
echo "POD_ID=${RUNPOD_POD_ID:-STILL_UNSET}"
nohup bash /workspace/exp002_finish_and_stop.sh >/dev/null 2>&1 &
sleep 2

echo '=== PROCS ==='
pgrep -af 'run_exp002|finish_and_stop' | grep -v relaunch
echo '=== LOG ==='
tail -n 5 /workspace/exp002.log 2>/dev/null
