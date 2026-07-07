#!/usr/bin/env bash
# One-shot: extend session B watcher cap 9h -> 14h (9 delta jobs + V3-CoT can
# exceed 9h at conservative throughput) and restart it with RUNPOD env.
sed -i 's/+ 32400/+ 50400/' /workspace/exp004b_finish_and_stop.sh
pkill -f exp004b_finish_and_stop.sh
sleep 1
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/exp004b_finish_and_stop.sh >/dev/null 2>&1 &
sleep 2
pgrep -af exp004b_finish
grep -n '50400' /workspace/exp004b_finish_and_stop.sh
