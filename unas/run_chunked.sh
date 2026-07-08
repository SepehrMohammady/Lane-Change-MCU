#!/bin/bash
# Run a DMIR search to completion in fresh-process chunks, resuming from the
# aging-evolution checkpoint each time. A fresh process resets the TF/XLA
# memory that clear_session() can't fully release in a long-lived process, so
# the search survives the slow leak that OOM-kills a single long run.
#
# The search loop counts len(history); load_state restores it, so resuming with
# --rounds TARGET continues toward TARGET (it does not restart). save-every 5
# keeps checkpoints fresh so an OOM loses at most a few candidates.
#
# Usage (from WSL):  bash /mnt/c/Projects/PhD/DIMIR/unas/run_chunked.sh dmir_lcr 150 100
CONFIG="${1:?config}"; TARGET="${2:-150}"; EPOCHS="${3:-100}"
MAX_CHUNKS="${MAX_CHUNKS:-8}"
VENV="$HOME/dmir_nas"; FORK="$HOME/uNAS"; REPO="/mnt/c/Projects/PhD/DIMIR"
STATE="$FORK/artifacts/$CONFIG/${CONFIG}_agingevosearch_state.pickle"

source "$VENV/env.sh"
cp "$REPO/unas/dmir_dataset.py" "$FORK/dataset/dmir_dataset.py"
cp "$REPO/unas/dmir_config.py"  "$FORK/configs/dmir_config.py"
python3 "$REPO/unas/patch_fork.py" "$FORK/uNAS/search_algorithms/aging_evolution.py" >/dev/null
export DMIR_DATA_ROOT="$REPO/data" DMIR_ROUNDS="$TARGET" DMIR_EPOCHS="$EPOCHS"
export DMIR_POPULATION="${DMIR_POPULATION:-50}" DMIR_SAMPLE="${DMIR_SAMPLE:-15}"
export TF_CPP_MIN_LOG_LEVEL=1 TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p "$REPO/runs/nas"
cd "$FORK"

prev_hist=-1
for chunk in $(seq 1 "$MAX_CHUNKS"); do
  LOG="$REPO/runs/nas/${CONFIG}_chunk${chunk}.log"
  args=(-c "$CONFIG" --seed 42 --save-every 5)
  [ -f "$STATE" ] && args+=(-l "$STATE")
  echo ">>> $CONFIG chunk $chunk @ $(date) (target $TARGET) -> $LOG"
  "$VENV/bin/python" driver.py "${args[@]}" > "$LOG" 2>&1
  done_ok=$(grep -c "Search done" "$LOG")
  hist=$(grep -c "Training complete" "$LOG")
  echo "    chunk $chunk: +$hist candidates, search_done=$done_ok"
  [ "$done_ok" -ge 1 ] && { echo "=== $CONFIG COMPLETE ==="; break; }
  # stall guard: if a chunk made no progress at all, stop (avoid spinning)
  if [ "$hist" -eq 0 ]; then echo "!!! $CONFIG stalled (no progress); stopping"; break; fi
done
echo "$CONFIG models on disk: $(ls "$FORK/artifacts/$CONFIG/models" 2>/dev/null | wc -l)"
