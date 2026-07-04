#!/usr/bin/env bash
# EXP-002 pod runner: setup + three H2 falsifier groups (rule_shift / compress /
# state_guard), then a sentinel for the auto-stop watcher.
# No `set -e`: the sentinel must appear even after a partial failure so the watcher
# archives whatever exists and stops billing.
set -u

cd /workspace
rm -rf aware && mkdir aware && cd aware
unzip -q /workspace/aware_src.zip

pip install -q --break-system-packages numpy pyyaml pytest tqdm matplotlib

export AWARE_THROTTLE=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

FAMS=rule_shift,compress,state_guard
python scripts/make_data.py --split train --per-family 20000 --families $FAMS
python scripts/make_data.py --split eval --per-family 400 --families $FAMS
python -m pytest tests/ -q
python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval

python scripts/run_exp.py --config experiments/configs/exp002_rule_shift.yaml --workers 3
python scripts/run_exp.py --config experiments/configs/exp002_compress.yaml --workers 3
python scripts/run_exp.py --config experiments/configs/exp002_state_guard.yaml --workers 3

echo EXP002_ALL_DONE
