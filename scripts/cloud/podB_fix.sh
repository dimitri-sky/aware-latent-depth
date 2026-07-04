#!/usr/bin/env bash
# Pod B fix: re-point the watcher at the supplement sentinel and start both.
pkill -f exp002_finish_and_stop
sleep 1
sed -i "s/EXP002_ALL_DONE/EXP002_REALLY_ALL_DONE/" /workspace/exp002_finish_and_stop.sh
sed -i "s/run_exp002.sh/podB_supplement.sh/" /workspace/exp002_finish_and_stop.sh
sed -i 's/\r$//' /workspace/podB_supplement.sh /workspace/exp002_finish_and_stop.sh
chmod +x /workspace/podB_supplement.sh
nohup bash /workspace/podB_supplement.sh >/dev/null 2>&1 &
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
echo "POD_ID=${RUNPOD_POD_ID:-STILL_UNSET}"
nohup bash /workspace/exp002_finish_and_stop.sh >/dev/null 2>&1 &
sleep 2
echo '=== PROCS ==='
pgrep -af 'supplement|finish_and_stop|run_exp002' | grep -v podB_fix
