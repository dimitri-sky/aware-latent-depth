#!/usr/bin/env bash
# RunPod one-shot validity gate (attempt 5+). Expects GITHUB_TOKEN for result push-back.
set -euo pipefail

REPO="${AWARE_REPO:-https://github.com/dimitri-sky/aware-research-3.git}"
WORKDIR="${WORKDIR:-/workspace/aware-research-3}"
LOG="${WORKDIR}/experiments/gate_attempt5_cloud.log"

echo "== Aware validity gate (cloud) =="
nvidia-smi || true

if [[ -d "$WORKDIR/.git" ]]; then
  cd "$WORKDIR" && git pull --ff-only || true
else
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    git clone "https://${GITHUB_TOKEN}@github.com/dimitri-sky/aware-research-3.git" "$WORKDIR"
  else
    git clone "$REPO" "$WORKDIR"
  fi
  cd "$WORKDIR"
fi

pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== generating SAGE data =="
python scripts/make_data.py --split train --per-family 20000
python scripts/make_data.py --split eval --per-family 400

echo "== running validity gate (no throttle) =="
python scripts/validity_gate.py 2>&1 | tee "$LOG"
GATE_EXIT=${PIPESTATUS[0]}

echo "== gate finished exit=$GATE_EXIT =="
cat experiments/validity_gate.json || true

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  git config user.email "runpod-gate@aware.local"
  git config user.name "RunPod Gate"
  git add experiments/results.csv experiments/validity_gate.json experiments/gate_attempt5_cloud.log
  BRANCH="gate-cloud-$(date -u +%Y%m%dT%H%M%SZ)"
  git checkout -b "$BRANCH"
  git commit -m "Gate attempt 5 cloud results (exit=$GATE_EXIT)" || true
  git push "https://${GITHUB_TOKEN}@github.com/dimitri-sky/aware-research-3.git" "HEAD:$BRANCH" || \
    echo "WARN: could not push results branch"
fi

# Stop pod to save cost (optional; requires RUNPOD_API_KEY + RUNPOD_POD_ID)
if [[ -n "${RUNPOD_API_KEY:-}" && -n "${RUNPOD_POD_ID:-}" ]]; then
  curl -sf -X POST "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" || true
fi

exit "$GATE_EXIT"
