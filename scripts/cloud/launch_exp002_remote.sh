#!/usr/bin/env bash
# One-shot remote launcher: normalizes line endings, starts the EXP-002 runner and
# the auto-stop watcher, then prints a status snapshot.
cd /workspace
sed -i 's/\r$//' run_exp002.sh exp002_finish_and_stop.sh
chmod +x run_exp002.sh exp002_finish_and_stop.sh
nohup bash /workspace/run_exp002.sh > /workspace/exp002.log 2>&1 &
nohup bash /workspace/exp002_finish_and_stop.sh >/dev/null 2>&1 &
sleep 5
echo '=== PROCS ==='
pgrep -af 'run_exp002|finish_and_stop'
echo '=== RUNPODCTL ==='
which runpodctl || echo "runpodctl NOT FOUND"
echo "POD_ID=${RUNPOD_POD_ID:-unset}"
echo '=== LOG ==='
tail -n 5 /workspace/exp002.log 2>/dev/null
