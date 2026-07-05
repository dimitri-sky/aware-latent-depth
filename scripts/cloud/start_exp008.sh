#!/usr/bin/env bash
# Detached launcher for EXP-008 (run after the probe passes the gate).
export $(tr '\0' '\n' < /proc/1/environ | grep '^RUNPOD' | xargs)
nohup bash /workspace/run_exp008.sh > /workspace/exp008.log 2>&1 &
sleep 2
nohup bash /workspace/exp008_finish_and_stop.sh >/dev/null 2>&1 &
sleep 1
pgrep -af 'exp008' | grep -v start_exp008
