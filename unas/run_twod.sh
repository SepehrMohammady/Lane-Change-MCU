#!/bin/bash
# Two-domain search pilot (2026-11-04): the final highD classifier search (the fork's aging evolution,
# 150 training runs, faithful resource graph, budgets 8,211 B and 26,240 MACs, error bound 0.06, 50 epochs,
# two workers) on highD and exiD pooled, with the arm's validation objective; then the arm's own choice
# (unas/select_by_seeds.py: the 12 best single runs x 5 seeds) and five fresh seeds for that choice trained
# on the pooled data (unas/final_eval.py, the same recipe for every arm).
#
#   bash /mnt/c/Projects/PhD/DIMIR/unas/run_twod.sh pooled|worst SEED
#
#   pooled  the fork's objective on the pooled validation windows (74% of them are exiD's)
#   worst   the lower of the two domains' validation accuracies, for early stopping, as the search's
#           validation error and for the ranking in select_by_seeds.py (unas/two_domain.py)
#
# Runs are named twod_cls_v4_k3_<arm>_s<seed> (series after the trainer fix of 2026-11-03); results go
# under datasets/exid/results/. SMOKE=1 checks the mechanics with a tiny budget, one-epoch retraining and
# separate files (_smoke in every name).
ARM="${1:?pooled|worst}"; SEED="${2:?seed}"
case "$ARM" in
  pooled) VALKEY=val_accuracy ;;
  worst)  VALKEY=val_worst_accuracy ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
MAX_CHUNKS="${MAX_CHUNKS:-16}"
RUN_TAG="$(date +%m%d%H%M)"
VENV="$HOME/dmir_nas"; FORK="$HOME/uNAS"; REPO="/mnt/c/Projects/PhD/DIMIR"

TAG=_k3; BUDGET=150; POP=50; SAMPLE=15; K=12; EPOCHS=50
if [ "${SMOKE:-0}" = 1 ]; then
  TAG=_k3_smoke; BUDGET=8; POP=4; SAMPLE=2; K=2; EPOCHS=2
  export NAS_SMOKE_BOUNDS=1 EPOCH_CAP=1 SEEDS=0,1 FINAL_SEEDS=100,101 FINAL_TAG=_smoke
fi
NAME="twod_cls_v4${TAG}_${ARM}_s${SEED}"
STATE="$FORK/artifacts/$NAME/${NAME}_agingevosearch_state.pickle"
export HIGHD_DATA_ROOT="$REPO/datasets/highd/data/prepared" EXID_DATA_ROOT="$REPO/datasets/exid/data/prepared"
export HIGHD_V2_SUFFIX="${TAG}_${ARM}_s${SEED}" HIGHD_ROUNDS=$BUDGET HIGHD_EPOCHS=$EPOCHS
export HIGHD_POPULATION=$POP HIGHD_SAMPLE=$SAMPLE HIGHD_PARALLEL=2
# read by the config and, in the Ray workers started from this environment, by the fork's trainer
export TWOD_OBJECTIVE="$ARM" NAS_VAL_KEY="$VALKEY"
mkdir -p "$REPO/runs/nas"

source "$VENV/env.sh"
cp "$REPO/unas/highd_dataset.py" "$FORK/dataset/highd_dataset.py"
cp "$REPO/unas/highd_config.py"  "$FORK/configs/highd_config.py"
cp "$REPO/unas/cnn1d_gap.py"     "$FORK/configs/cnn1d_gap.py"
cp "$REPO/unas/safe_saver.py"    "$FORK/configs/safe_saver.py"
cp "$REPO/unas/two_domain.py"    "$FORK/configs/two_domain.py"
python3 "$REPO/unas/patch_fork.py" "$FORK/uNAS/search_algorithms/aging_evolution.py" >/dev/null

# idempotent registration
python3 - "$FORK/driver.py" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
if '"twod_cls_v4"' not in s:
    s = s.replace("_CONFIGS = {", '_CONFIGS = {\n    "twod_cls_v4": ("configs.highd_config", "get_twod_cls_v4_setup"),', 1)
    open(p, "w").write(s)
    print("registered twod_cls_v4")
PY
grep -q "from .highd_dataset import HighD_Dataset" "$FORK/dataset/__init__.py" \
  || printf '\nfrom .highd_dataset import HighD_Dataset\n' >> "$FORK/dataset/__init__.py"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
GPULOG="$REPO/runs/nas/${NAME}_${RUN_TAG}_gpu.csv"
( nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader -l 30 > "$GPULOG" 2>/dev/null ) &
GPUSAMPLER=$!
trap "kill $GPUSAMPLER 2>/dev/null" EXIT
export TF_CPP_MIN_LOG_LEVEL=1 TF_FORCE_GPU_ALLOW_GROWTH=true
cd "$FORK"

for chunk in $(seq 1 "$MAX_CHUNKS"); do
  LOG="$REPO/runs/nas/${NAME}_${RUN_TAG}_chunk${chunk}.log"
  args=(-c twod_cls_v4 --seed "$SEED" --save-every 5)
  [ -f "$STATE" ] && args+=(-l "$STATE")
  echo ">>> $NAME chunk $chunk @ $(date) ($ARM, $BUDGET runs) -> $LOG"
  "$VENV/bin/python" driver.py "${args[@]}" > "$LOG" 2>&1
  done_ok=$(grep -c "Search done" "$LOG")
  runs=$(grep -c "Training complete" "$LOG")
  echo "    chunk $chunk: +$runs runs, search_done=$done_ok"
  [ "$done_ok" -ge 1 ] && { echo "=== $NAME search complete ==="; break; }
  if [ "$runs" -eq 0 ]; then echo "!!! $NAME stalled (no progress); stopping"; exit 1; fi
done
[ "$done_ok" -ge 1 ] || { echo "!!! $NAME did not finish in $MAX_CHUNKS chunks"; exit 1; }

"$VENV/bin/python" "$REPO/unas/select_by_seeds.py" "$NAME" "$K" 2>&1 \
  | grep -E --line-buffered "seed [0-9]:|best|smallest|saved candidates|no sidecars|Traceback|Error"
"$VENV/bin/python" "$REPO/unas/final_eval.py" "$NAME" 2>&1 | grep -E --line-buffered "seed|chosen|mean|Traceback|Error"
echo "=== $NAME done @ $(date) ==="
