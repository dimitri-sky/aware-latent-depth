#!/usr/bin/env bash
nohup bash /workspace/podB_supplement2.sh > /workspace/supp2.log 2>&1 &
sleep 2
pgrep -af 'supplement|finish_and_stop' | grep -v start_supp2
