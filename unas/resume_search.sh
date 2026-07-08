#!/bin/bash
# Resume an interrupted DMIR search from its aging-evolution checkpoint.
# Usage (from WSL):  bash /mnt/c/Projects/PhD/DIMIR/unas/resume_search.sh dmir_lcr 150 100
set -e
CONFIG="${1:?config name}"
ROUNDS="${2:-150}"
EPOCHS="${3:-100}"
VENV="$HOME/dmir_nas"; FORK="$HOME/uNAS"; REPO="/mnt/c/Projects/PhD/DIMIR"
STATE="$FORK/artifacts/$CONFIG/${CONFIG}_agingevosearch_state.pickle"
[ -f "$STATE" ] || { echo "no checkpoint at $STATE"; exit 1; }
source "$VENV/env.sh"
cp "$REPO/unas/dmir_dataset.py" "$FORK/dataset/dmir_dataset.py"
cp "$REPO/unas/dmir_config.py"  "$FORK/configs/dmir_config.py"
python3 "$REPO/unas/patch_fork.py" "$FORK/uNAS/search_algorithms/aging_evolution.py"
export DMIR_DATA_ROOT="$REPO/data"
export DMIR_ROUNDS="$ROUNDS" DMIR_EPOCHS="$EPOCHS"
export DMIR_POPULATION="${DMIR_POPULATION:-50}" DMIR_SAMPLE="${DMIR_SAMPLE:-15}"
export TF_CPP_MIN_LOG_LEVEL=1 TF_FORCE_GPU_ALLOW_GROWTH=true
LOG="$REPO/runs/nas/${CONFIG}_resume.log"
echo "resuming $CONFIG from $STATE -> $LOG"
cd "$FORK"
"$VENV/bin/python" driver.py -c "$CONFIG" --seed 42 --save-every 10 -l "$STATE" > "$LOG" 2>&1
tail -3 "$LOG"
