#!/usr/bin/env bash
# One-shot: extend the supplement watcher's hard cap 9h -> 13h (the supplement
# starts only after the ~5h main lanes finish; a 9h cap from watcher start could
# kill the pod mid-supplement) and restart the watcher with RUNPOD env.
sed -i 's/+ 32400/+ 46800/' /workspace/exp004cs_finish_and_stop.sh
pkill -f exp004cs_finish_and_stop.sh
sleep 1
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/exp004cs_finish_and_stop.sh >/dev/null 2>&1 &
sleep 2
pgrep -af exp004cs_finish
grep -n '46800' /workspace/exp004cs_finish_and_stop.sh
