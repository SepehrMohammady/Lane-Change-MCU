#!/bin/bash
# Part A (2026-11-02): one run of one search variant, in fresh-process chunks like run_chunked*.sh,
# then the variant's own final choice and the five fresh-seed evaluation of that choice.
#
#   bash /mnt/c/Projects/PhD/DIMIR/unas/run_noise_aware.sh highd_cls|dmir_cls MODE SEED
#   NAME_TAG=_k3 bash ...   names the runs highd_cls_v4_k3_<mode>_s<seed> (series after the trainer fix of
#                           2026-11-03; the runs of 2026-11-02 used the accumulated metric, unas/patch_fork.py)
#
#   single    the fork's aging evolution with the final-search setup: 150 runs, then
#             unas/select_by_seeds.py (12 best single runs x 5 seeds = 60 runs)
#   random    random search: 150 runs, then the same selection (60 runs)
#   fixed     70 candidates x 3 runs (population 23, sample 7: the fork's ratios at 70 candidates)
#   adaptive  210 runs with re-runs as in unas/noise_aware_search.py
#   costfirst (2026-11-10): the fork's aging evolution with the noise-aware cost-first parent
#             selection of unas/cost_first_search.py, 150 runs, then select_by_seeds.py with SEL_RULE=costfirst
#             (12 cheapest plateau candidates x 5 seeds, cheapest within CF_MARGIN of the best mean).
#             CF_TAU and CF_MARGIN per task were fixed beforehand from the uNAS searches' five-seed runs.
#
# Every variant uses the same space, resource model, budgets, recipe, epochs and two workers;
# 210 training runs each. --seed fixes the search's own random choices, so the same seed gives every
# variant the same first random architectures. unas/final_eval.py then retrains the choice with
# five fresh seeds (not part of the budget).
TASK="${1:?highd_cls|dmir_cls}"; MODE="${2:?single|random|fixed|adaptive}"; SEED="${3:?seed}"
MAX_CHUNKS="${MAX_CHUNKS:-16}"
RUN_TAG="$(date +%m%d%H%M)"
VENV="$HOME/dmir_nas"; FORK="$HOME/uNAS"; REPO="/mnt/c/Projects/PhD/DIMIR"
case "$MODE" in single|random|fixed|adaptive|costfirst) ;; *) echo "unknown mode $MODE"; exit 2 ;; esac

POP=50; SAMPLE=15; K=12; TAG="${NAME_TAG:-}"; HE=50; DE=80     # NAME_TAG: suffix for a new series of runs
[ "$MODE" = fixed ] && { POP=23; SAMPLE=7; }
case "$MODE" in single|random|costfirst) BUDGET=150 ;; *) BUDGET=210 ;; esac
if [ "${SMOKE:-0}" = 1 ]; then      # mechanics only: tiny budget, short training, separate files
  TAG="${NAME_TAG:-}_smoke"; POP=4; SAMPLE=2; K=2; HE=2; DE=2
  [ "$MODE" = fixed ] && POP=2
  case "$MODE" in single|random|costfirst) BUDGET=8 ;; *) BUDGET=14 ;; esac
  export NAS_SMOKE_BOUNDS=1 NAS_NMAX=3 NAS_FINAL_SHARE=0.3 EPOCH_CAP=1 SEEDS=0,1 FINAL_SEEDS=100,101 FINAL_TAG=_smoke
fi

case "$TASK" in
  highd_cls)
    DS=highd; BASE=highd_cls_v4; CF_TAU=0.00339; CF_MARGIN=0.00214
    export HIGHD_DATA_ROOT="$REPO/datasets/highd/data/prepared" HIGHD_V2_SUFFIX="${TAG}_${MODE}_s${SEED}"
    export HIGHD_ROUNDS="$BUDGET" HIGHD_EPOCHS=$HE HIGHD_POPULATION=$POP HIGHD_SAMPLE=$SAMPLE HIGHD_PARALLEL=2 ;;
  dmir_cls)
    DS=dmir; BASE=dmir_cls_v4; CF_TAU=0.00584; CF_MARGIN=0.00369
    export DMIR_DATA_ROOT="$REPO/datasets/dmir/data" DMIR_V4_SUFFIX="${TAG}_${MODE}_s${SEED}" DMIR_REG_METRIC=mae
    export DMIR_ROUNDS="$BUDGET" DMIR_EPOCHS=$DE DMIR_POPULATION=$POP DMIR_SAMPLE=$SAMPLE DMIR_PARALLEL=2 ;;
  *) echo "unknown task $TASK"; exit 2 ;;
esac
NAME="${BASE}${TAG}_${MODE}_s${SEED}"
case "$MODE" in single) CONFIG="$BASE" ;; costfirst) CONFIG="${BASE}cf" ;; *) CONFIG="${BASE}n" ;; esac
export CF_TAU CF_MARGIN
STATE="$FORK/artifacts/$NAME/${NAME}_agingevosearch_state.pickle"      # single and costfirst; the others keep their own
OUT="$REPO/datasets/$DS/results/nas-noise"
mkdir -p "$OUT" "$REPO/runs/nas"
export NAS_MODE="$MODE" NAS_BUDGET="$BUDGET" NAS_LOG="$OUT/${NAME}_runs.jsonl"

source "$VENV/env.sh"
cp "$REPO/unas/highd_dataset.py"      "$FORK/dataset/highd_dataset.py"
cp "$REPO/unas/highd_config.py"       "$FORK/configs/highd_config.py"
cp "$REPO/unas/dmir_dataset.py"       "$FORK/dataset/dmir_dataset.py"
cp "$REPO/unas/dmir_config.py"        "$FORK/configs/dmir_config.py"
cp "$REPO/unas/cnn1d_gap.py"          "$FORK/configs/cnn1d_gap.py"
cp "$REPO/unas/safe_saver.py"         "$FORK/configs/safe_saver.py"
cp "$REPO/unas/noise_aware_search.py" "$FORK/configs/noise_aware_search.py"
cp "$REPO/unas/cost_first_search.py"  "$FORK/configs/cost_first_search.py"
python3 "$REPO/unas/patch_fork.py" "$FORK/uNAS/search_algorithms/aging_evolution.py" >/dev/null

# idempotent: register the configs, and let the fork accept a subclass of AgingEvoSearch
python3 - "$FORK" <<'EOF'
import sys
fork = sys.argv[1]
p = f"{fork}/driver.py"
s = open(p).read()
entries = {"highd_cls_v4": "configs.highd_config", "highd_cls_v4n": "configs.highd_config",
           "highd_cls_v4cf": "configs.highd_config", "dmir_cls_v4": "configs.dmir_config",
           "dmir_cls_v4n": "configs.dmir_config", "dmir_cls_v4cf": "configs.dmir_config"}
new = [f'    "{k}": ("{m}", "get_{k}_setup"),' for k, m in entries.items() if f'"{k}"' not in s]
if new:
    s = s.replace("_CONFIGS = {", "_CONFIGS = {" + chr(10) + chr(10).join(new), 1)
    open(p, "w").write(s)
    print("registered", len(new), "configs")
p = f"{fork}/uNAS/uNAS.py"
s = open(p).read()
old = 'if setup["config"]["search_algorithm"] not in [AgingEvoSearch, BayesOpt]:'
new_check = 'if not issubclass(setup["config"]["search_algorithm"], (AgingEvoSearch, BayesOpt)):'
if old in s:
    open(p, "w").write(s.replace(old, new_check))
    print("uNAS.py: subclasses of AgingEvoSearch accepted")
EOF
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
  args=(-c "$CONFIG" --seed "$SEED" --save-every 5)
  case "$MODE" in single|costfirst) [ -f "$STATE" ] && args+=(-l "$STATE") ;; esac
  echo ">>> $NAME chunk $chunk @ $(date) ($MODE, $BUDGET runs) -> $LOG"
  "$VENV/bin/python" driver.py "${args[@]}" > "$LOG" 2>&1
  done_ok=$(grep -c "Search done" "$LOG")
  runs=$(grep -c "Training complete" "$LOG")
  echo "    chunk $chunk: +$runs runs, search_done=$done_ok"
  [ "$done_ok" -ge 1 ] && { echo "=== $NAME search complete ==="; break; }
  if [ "$runs" -eq 0 ]; then echo "!!! $NAME stalled (no progress); stopping"; exit 1; fi
done
[ "$done_ok" -ge 1 ] || { echo "!!! $NAME did not finish in $MAX_CHUNKS chunks"; exit 1; }

# the variant's own final choice
case "$MODE" in
  single|random)
    "$VENV/bin/python" "$REPO/unas/select_by_seeds.py" "$NAME" "$K" 2>&1 \
      | grep -E --line-buffered "seed [0-9]:|best|smallest|saved candidates|no sidecars|Traceback|Error" ;;
  costfirst)
    SEL_RULE=costfirst "$VENV/bin/python" "$REPO/unas/select_by_seeds.py" "$NAME" "$K" 2>&1 \
      | grep -E --line-buffered "seed [0-9]:|best|smallest|chosen|saved candidates|no sidecars|Traceback|Error" ;;
  *)
    mkdir -p "$OUT/$NAME"
    cp "$FORK/artifacts/$NAME/${NAME}_final.json" "$FORK/artifacts/$NAME/${NAME}_final.h5" "$OUT/$NAME/" ;;
esac
# five fresh seeds for the chosen network, the same for every variant
"$VENV/bin/python" "$REPO/unas/final_eval.py" "$NAME" 2>&1 | grep -E --line-buffered "seed|chosen|mean|Traceback|Error"
echo "=== $NAME done @ $(date) ==="
