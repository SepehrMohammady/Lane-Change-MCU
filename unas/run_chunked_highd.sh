#!/bin/bash
# Chunked highD search (same OOM-safe pattern as run_chunked.sh).
# Usage (from WSL):  bash /mnt/c/Projects/PhD/DIMIR/unas/run_chunked_highd.sh highd_cls 150 50
CONFIG="${1:?config}"; TARGET="${2:-150}"; EPOCHS="${3:-50}"
MAX_CHUNKS="${MAX_CHUNKS:-8}"
RUN_TAG="$(date +%m%d%H%M)"
VENV="$HOME/dmir_nas"; FORK="$HOME/uNAS"; REPO="/mnt/c/Projects/PhD/DIMIR"
STATE="$FORK/artifacts/$CONFIG/${CONFIG}_agingevosearch_state.pickle"

source "$VENV/env.sh"
cp "$REPO/unas/highd_dataset.py" "$FORK/dataset/highd_dataset.py"
cp "$REPO/unas/highd_config.py"  "$FORK/configs/highd_config.py"
python3 "$REPO/unas/patch_fork.py" "$FORK/uNAS/search_algorithms/aging_evolution.py" >/dev/null

# idempotent registration
grep -q "from .highd_dataset import HighD_Dataset" "$FORK/dataset/__init__.py" \
  || printf '\nfrom .highd_dataset import HighD_Dataset\n' >> "$FORK/dataset/__init__.py"
grep -q '"highd_cls"' "$FORK/driver.py" || python3 - "$FORK/driver.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace('_CONFIGS = {',
              '_CONFIGS = {\n'
              '    "highd_cls": ("configs.highd_config", "get_highd_cls_setup"),\n'
              '    "highd_ttlc": ("configs.highd_config", "get_highd_ttlc_setup"),')
open(p, "w").write(s)
print("registered highd configs")
EOF

export HIGHD_DATA_ROOT="$REPO/datasets/highd/data/prepared"
export HIGHD_ROUNDS="$TARGET" HIGHD_EPOCHS="$EPOCHS"
export HIGHD_POPULATION="${HIGHD_POPULATION:-50}" HIGHD_SAMPLE="${HIGHD_SAMPLE:-15}"
export HIGHD_PARALLEL="${HIGHD_PARALLEL:-1}"
# GPU accounting for the paper: snapshot at start + 30 s utilization samples.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPULOG="$REPO/runs/nas/${CONFIG}_${RUN_TAG}_gpu.csv"
( nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 30 > "$GPULOG" 2>/dev/null ) &
GPUSAMPLER=$!
trap "kill $GPUSAMPLER 2>/dev/null" EXIT
export TF_CPP_MIN_LOG_LEVEL=1 TF_FORCE_GPU_ALLOW_GROWTH=true
mkdir -p "$REPO/runs/nas"
cd "$FORK"

prev_hist=-1
for chunk in $(seq 1 "$MAX_CHUNKS"); do
  LOG="$REPO/runs/nas/${CONFIG}_${RUN_TAG}_chunk${chunk}.log"
  args=(-c "$CONFIG" --seed 42 --save-every 5)
  [ -f "$STATE" ] && args+=(-l "$STATE")
  echo ">>> $CONFIG chunk $chunk @ $(date) (target $TARGET) -> $LOG"
  "$VENV/bin/python" driver.py "${args[@]}" > "$LOG" 2>&1
  done_ok=$(grep -c "Search done" "$LOG")
  hist=$(grep -c "Training complete" "$LOG")
  echo "    chunk $chunk: +$hist candidates, search_done=$done_ok"
  [ "$done_ok" -ge 1 ] && { echo "=== $CONFIG COMPLETE ==="; break; }
  if [ "$hist" -eq 0 ]; then echo "!!! $CONFIG stalled (no progress); stopping"; break; fi
done
echo "$CONFIG models on disk: $(ls "$FORK/artifacts/$CONFIG/models" 2>/dev/null | wc -l)"
