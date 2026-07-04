#!/usr/bin/env bash
# EXP-001B pod runner: setup + three run groups (algo_exec / seeds 3-5 /
# matched-FLOP control), then a sentinel for the auto-shutdown watcher.
# Deliberately no `set -e`: the sentinel must appear even after a partial failure
# so the watcher archives whatever exists and stops billing.
set -u

cd /workspace
rm -rf aware && mkdir aware && cd aware
unzip -q /workspace/aware_src.zip

# RunPod pytorch image ships a PEP-668 "externally managed" Python
pip install -q --break-system-packages numpy pyyaml pytest tqdm matplotlib

export AWARE_THROTTLE=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Only the three families these runs need (faster setup, same validity)
FAMS=algo_exec,rewrite,dsl_learn
python scripts/make_data.py --split train --per-family 20000 --families $FAMS
python scripts/make_data.py --split eval --per-family 400 --families $FAMS
python -m pytest tests/ -q
python -m sage.contamination.audit --train-dir data/sage/train --eval-dir data/sage/eval

python scripts/run_exp.py --config experiments/configs/exp001b_algo.yaml --workers 3
python scripts/run_exp.py --config experiments/configs/exp001b_seeds.yaml --workers 3
python scripts/run_exp.py --config experiments/configs/exp001b_flopmatch.yaml --workers 3

echo EXP001B_ALL_DONE
