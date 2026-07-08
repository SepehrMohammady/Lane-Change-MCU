#!/bin/bash
# Smoke-run the DMIR NAS on GPU (WSL) to validate the whole chain on real data.
# Prereqs: dmir_nas venv with tensorflow[and-cuda]; fork prepped via setup_fork.sh.
#
# Usage (from WSL):
#   VENV=~/dmir_nas FORK=~/uNAS REPO=/mnt/c/Projects/PhD/DIMIR \
#     bash "$REPO/unas/run_smoke.sh" dmir_lcr 20
set -e
CONFIG="${1:-dmir_lcr}"
ROUNDS="${2:-20}"
VENV="${VENV:-$HOME/dmir_nas}"
FORK="${FORK:-$HOME/uNAS}"
REPO="${REPO:-/mnt/c/Projects/PhD/DIMIR}"

# CUDA libs for the pip TF wheel (written by env setup); safe if absent.
[ -f "$VENV/env.sh" ] && source "$VENV/env.sh"

export DMIR_DATA_ROOT="${DMIR_DATA_ROOT:-$REPO/data}"
export DMIR_ROUNDS="$ROUNDS"
# smaller population/faster warmup for the smoke run
export DMIR_POPULATION="${DMIR_POPULATION:-20}"
export DMIR_SAMPLE="${DMIR_SAMPLE:-5}"

echo "config=$CONFIG rounds=$ROUNDS data=$DMIR_DATA_ROOT"
cd "$FORK"
"$VENV/bin/python" driver.py -c "$CONFIG" --seed 42
echo "=== artifacts ==="
ls -la "$FORK/artifacts/"*/ 2>/dev/null | tail -20
