#!/usr/bin/env bash
# Pod B one-shot: normalize endings, start runner + auto-stop watcher with RunPod
# env imported from PID 1, print status.
cd /workspace
sed -i 's/\r$//' run_exp002.sh exp002_finish_and_stop.sh
sed -i 's/21600/43200/' exp002_finish_and_stop.sh
chmod +x run_exp002.sh exp002_finish_and_stop.sh
nohup bash /workspace/run_exp002.sh > /workspace/exp002.log 2>&1 &
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
echo "POD_ID=${RUNPOD_POD_ID:-STILL_UNSET}"
nohup bash /workspace/exp002_finish_and_stop.sh >/dev/null 2>&1 &
sleep 3
echo '=== PROCS ==='
pgrep -af 'run_exp002|finish_and_stop' | grep -v launch_podB
echo '=== LOG ==='
tail -n 5 /workspace/exp002.log 2>/dev/null
