#!/bin/bash
# Chunked highD search (same OOM-safe pattern as run_chunked.sh).
# Usage (from WSL):  bash /mnt/c/Projects/PhD/DIMIR/unas/run_chunked_highd.sh highd_cls 150 50
# The *_v2 and *_v3 configs use the global-average-pooling search space (unas/cnn1d_gap.py) and the
# fixed model saver (unas/safe_saver.py); HIGHD_V2_SUFFIX renames their artifact folder.
CONFIG="${1:?config}"; TARGET="${2:-150}"; EPOCHS="${3:-50}"
MAX_CHUNKS="${MAX_CHUNKS:-8}"
RUN_TAG="$(date +%m%d%H%M)"
VENV="$HOME/dmir_nas"; FORK="$HOME/uNAS"; REPO="/mnt/c/Projects/PhD/DIMIR"
case "$CONFIG" in *_v2|*_v3) NAME="$CONFIG${HIGHD_V2_SUFFIX:-}" ;; *) NAME="$CONFIG" ;; esac
STATE="$FORK/artifacts/$NAME/${NAME}_agingevosearch_state.pickle"

source "$VENV/env.sh"
cp "$REPO/unas/highd_dataset.py" "$FORK/dataset/highd_dataset.py"
cp "$REPO/unas/highd_config.py"  "$FORK/configs/highd_config.py"
cp "$REPO/unas/cnn1d_gap.py"      "$FORK/configs/cnn1d_gap.py"
cp "$REPO/unas/safe_saver.py"     "$FORK/configs/safe_saver.py"
python3 "$REPO/unas/patch_fork.py" "$FORK/uNAS/search_algorithms/aging_evolution.py" >/dev/null

# idempotent registration
grep -q "from .highd_dataset import HighD_Dataset" "$FORK/dataset/__init__.py" \
  || printf '\nfrom .highd_dataset import HighD_Dataset\n' >> "$FORK/dataset/__init__.py"
python3 - "$FORK/driver.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
entries = {"highd_cls": "get_highd_cls_setup", "highd_cls_tight": "get_highd_cls_tight_setup",
           "highd_ttlc": "get_highd_ttlc_setup", "highd_cls_v2": "get_highd_cls_v2_setup",
           "highd_ttlc_v2": "get_highd_ttlc_v2_setup", "highd_cls_v3": "get_highd_cls_v3_setup"}
new = [f'    "{k}": ("configs.highd_config", "{f}"),' for k, f in entries.items() if f'"{k}"' not in s]
if new:
    s = s.replace("_CONFIGS = {", "_CONFIGS = {" + chr(10) + chr(10).join(new), 1)
    open(p, "w").write(s)
    print("registered", len(new), "highd configs")
EOF

export HIGHD_DATA_ROOT="$REPO/datasets/highd/data/prepared"
export HIGHD_ROUNDS="$TARGET" HIGHD_EPOCHS="$EPOCHS"
export HIGHD_POPULATION="${HIGHD_POPULATION:-50}" HIGHD_SAMPLE="${HIGHD_SAMPLE:-15}"
export HIGHD_PARALLEL="${HIGHD_PARALLEL:-1}"
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
  if [ "$hist" -eq 0 ]; then echo "!!! $NAME stalled (no progress); stopping"; break; fi
done
echo "$NAME models on disk: $(ls "$FORK/artifacts/$NAME/models" 2>/dev/null | grep -c 'h5$')"
