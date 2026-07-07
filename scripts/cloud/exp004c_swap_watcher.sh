#!/usr/bin/env bash
# One-shot: kill the original session C watcher, start the supplement runner +
# replacement watcher (both with RUNPOD env imported from PID 1).
sed -i 's/\r$//' /workspace/exp004c_supplement.sh /workspace/exp004cs_finish_and_stop.sh
chmod +x /workspace/exp004c_supplement.sh /workspace/exp004cs_finish_and_stop.sh
pkill -f exp004c_finish_and_stop.sh
sleep 1
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/exp004c_supplement.sh > /workspace/exp004c_supp.log 2>&1 &
sleep 1
nohup bash /workspace/exp004cs_finish_and_stop.sh >/dev/null 2>&1 &
sleep 2
pgrep -af 'exp004c' | grep -v swap_watcher
