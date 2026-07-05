#!/usr/bin/env bash
# Drop the d1 arm mid-session (deviation logged in agent/log/EXP-005.md):
# throughput discovery shows d1 at 5.4k tok/s (2x d2's sequential-scan cost),
# i.e. ~5h/job — and the pre-registered flops/correct tie-break already makes
# d1 a dominated design point given d2's .965 ceiling. Killing the phase-B d1
# runner lets the main script fall through to wait (V2 @8000) + phase C (d3).
pkill -f 'models V3-d1'
pkill -f 'model-id V3-d1'
sleep 2
echo '--- remaining training procs ---'
pgrep -af 'train_single' | head -5
