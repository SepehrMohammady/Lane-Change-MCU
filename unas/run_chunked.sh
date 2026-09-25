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
# The *_v4 configs use the global-average-pooling space with the faithful resource graph
# (unas/cnn1d_gap.py) and the unique-name saver (unas/safe_saver.py); DMIR_V4_SUFFIX renames
# their artifact folder, DMIR_PARALLEL sets the number of parallel evaluations.
#
# Usage (from WSL):  bash /mnt/c/Projects/PhD/DIMIR/unas/run_chunked.sh dmir_lcr 150 100
CONFIG="${1:?config}"; TARGET="${2:-150}"; EPOCHS="${3:-100}"
MAX_CHUNKS="${MAX_CHUNKS:-8}"
RUN_TAG="$(date +%m%d%H%M)"
VENV="$HOME/dmir_nas"; FORK="$HOME/uNAS"; REPO="/mnt/c/Projects/PhD/DIMIR"
case "$CONFIG" in *_v4) NAME="$CONFIG${DMIR_V4_SUFFIX:-}" ;; *) NAME="$CONFIG" ;; esac
STATE="$FORK/artifacts/$NAME/${NAME}_agingevosearch_state.pickle"

source "$VENV/env.sh"
cp "$REPO/unas/dmir_dataset.py" "$FORK/dataset/dmir_dataset.py"
cp "$REPO/unas/dmir_config.py"  "$FORK/configs/dmir_config.py"
cp "$REPO/unas/cnn1d_gap.py"    "$FORK/configs/cnn1d_gap.py"
cp "$REPO/unas/safe_saver.py"   "$FORK/configs/safe_saver.py"
python3 "$REPO/unas/patch_fork.py" "$FORK/uNAS/search_algorithms/aging_evolution.py" >/dev/null

# idempotent registration of the v4 configs (the earlier ones were registered by setup_fork.sh)
python3 - "$FORK/driver.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
entries = {"dmir_cls_v4": "get_dmir_cls_v4_setup", "dmir_lcr_v4": "get_dmir_lcr_v4_setup",
           "dmir_lcl_v4": "get_dmir_lcl_v4_setup"}
new = [f'    "{k}": ("configs.dmir_config", "{f}"),' for k, f in entries.items() if f'"{k}"' not in s]
if new:
    s = s.replace("_CONFIGS = {", "_CONFIGS = {" + chr(10) + chr(10).join(new), 1)
    open(p, "w").write(s)
    print("registered", len(new), "dmir configs")
EOF

export DMIR_DATA_ROOT="$REPO/datasets/dmir/data" DMIR_ROUNDS="$TARGET" DMIR_EPOCHS="$EPOCHS"
export DMIR_POPULATION="${DMIR_POPULATION:-50}" DMIR_SAMPLE="${DMIR_SAMPLE:-15}"
export DMIR_PARALLEL="${DMIR_PARALLEL:-1}"
# GPU accounting for the paper: snapshot at start + 30 s utilization samples.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPULOG="$REPO/runs/nas/${NAME}_${RUN_TAG}_gpu.csv"
( nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 30 > "$GPULOG" 2>/dev/null ) &
GPUSAMPLER=$!
trap "kill $GPUSAMPLER 2>/dev/null" EXIT
export TF_CPP_MIN_LOG_LEVEL=1 TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p "$REPO/runs/nas"
cd "$FORK"

for chunk in $(seq 1 "$MAX_CHUNKS"); do
  LOG="$REPO/runs/nas/${NAME}_${RUN_TAG}_chunk${chunk}.log"
  args=(-c "$CONFIG" --seed 42 --save-every 5)
  [ -f "$STATE" ] && args+=(-l "$STATE")
  echo ">>> $NAME chunk $chunk @ $(date) (target $TARGET) -> $LOG"
  "$VENV/bin/python" driver.py "${args[@]}" > "$LOG" 2>&1
  done_ok=$(grep -c "Search done" "$LOG")
  hist=$(grep -c "Training complete" "$LOG")
  echo "    chunk $chunk: +$hist candidates, search_done=$done_ok"
  [ "$done_ok" -ge 1 ] && { echo "=== $NAME COMPLETE ==="; break; }
  # stall guard: if a chunk made no progress at all, stop (avoid spinning)
  if [ "$hist" -eq 0 ]; then echo "!!! $NAME stalled (no progress); stopping"; break; fi
done
echo "$NAME models on disk: $(ls "$FORK/artifacts/$NAME/models" 2>/dev/null | grep -c 'h5$')"
