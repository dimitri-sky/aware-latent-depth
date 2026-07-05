#!/usr/bin/env bash
# Extend the stage12 watcher hard cap 6h -> 10h (session estimate ~5.5h; too
# tight against the original cap) and restart it with RUNPOD env.
pkill -f stage12_finish_and_stop
sleep 1
sed -i 's/+ 21600/+ 36000/; s/6h hard cap/10h hard cap/; s/6h deadline/10h deadline/' /workspace/stage12_finish_and_stop.sh
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/stage12_finish_and_stop.sh >/dev/null 2>&1 &
sleep 1
pgrep -af finish_and_stop
