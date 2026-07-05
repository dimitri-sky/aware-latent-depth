#!/usr/bin/env bash
# Detached launcher for the Stage-1/2 session on pod B.
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/run_stage12_podB.sh > /workspace/stage12.log 2>&1 &
sleep 2
nohup bash /workspace/stage12_finish_and_stop.sh >/dev/null 2>&1 &
sleep 1
pgrep -af 'stage12' | grep -v start_stage12
