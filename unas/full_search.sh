#!/bin/bash
# Full(er)-budget DMIR NAS search on GPU, logged to the repo.
# Usage (from WSL):
#   bash /mnt/c/Projects/PhD/DIMIR/unas/full_search.sh dmir_lcr 100 60
set -e
CONFIG="${1:-dmir_lcr}"
ROUNDS="${2:-100}"
EPOCHS="${3:-60}"
VENV="$HOME/dmir_nas"; FORK="$HOME/uNAS"; REPO="/mnt/c/Projects/PhD/DIMIR"
source "$VENV/env.sh"
cp "$REPO/unas/dmir_dataset.py" "$FORK/dataset/dmir_dataset.py"
cp "$REPO/unas/dmir_config.py"  "$FORK/configs/dmir_config.py"
export DMIR_DATA_ROOT="$REPO/data"
export DMIR_ROUNDS="$ROUNDS" DMIR_EPOCHS="$EPOCHS"
export DMIR_POPULATION="${DMIR_POPULATION:-50}" DMIR_SAMPLE="${DMIR_SAMPLE:-15}"
export TF_CPP_MIN_LOG_LEVEL=1
LOGDIR="$REPO/runs/nas"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/${CONFIG}_r${ROUNDS}_e${EPOCHS}.log"
echo "config=$CONFIG rounds=$ROUNDS epochs=$EPOCHS -> $LOG"
cd "$FORK"
"$VENV/bin/python" driver.py -c "$CONFIG" --seed 42 --save-every 10 > "$LOG" 2>&1
echo "=== done; tail ==="; tail -5 "$LOG"
echo "=== pareto models ==="
ls -la "$FORK/artifacts/$CONFIG/models/" 2>/dev/null | tail
