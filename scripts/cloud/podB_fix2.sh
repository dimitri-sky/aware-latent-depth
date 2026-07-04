#!/usr/bin/env bash
# Re-point pod B's watcher at the FINAL sentinel and track supplement 2.
pkill -f exp002_finish_and_stop
sleep 1
sed -i "s/EXP002_REALLY_ALL_DONE/EXP002_FINAL_DONE/" /workspace/exp002_finish_and_stop.sh
sed -i "s/podB_supplement.sh/podB_supplement2.sh/" /workspace/exp002_finish_and_stop.sh
sed -i 's/\r$//' /workspace/podB_supplement2.sh /workspace/exp002_finish_and_stop.sh
chmod +x /workspace/podB_supplement2.sh
nohup bash /workspace/podB_supplement2.sh >/dev/null 2>&1 &
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
echo "POD_ID=${RUNPOD_POD_ID:-STILL_UNSET}"
nohup bash /workspace/exp002_finish_and_stop.sh >/dev/null 2>&1 &
sleep 2
echo '=== PROCS ==='
pgrep -af 'supplement|finish_and_stop' | grep -v fix2
