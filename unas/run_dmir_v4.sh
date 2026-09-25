#!/bin/bash
# LCIR v4 queue (2026-09-25):
#  1. dmir_cls_v4, dmir_lcr_v4, dmir_lcl_v4 searches: faithful resource graph, budgets at the
#     hand-designed DSCNN's cost, 150 candidates each, two workers;
#  2. after each search, the five-seed validation choice (unas/select_by_seeds.py);
#  3. DSCNN-recipe controls of the three choices, and the cost-count check of the searches.
# Usage (from Windows, detached): wsl -e bash /mnt/c/Projects/PhD/DIMIR/unas/run_dmir_v4.sh
R=/mnt/c/Projects/PhD/DIMIR
LOG=$R/runs/nas/dmir_v4_pipeline.log
source ~/dmir_nas/env.sh
export DMIR_DATA_ROOT=$R/datasets/dmir/data TF_CPP_MIN_LOG_LEVEL=2

echo "$(date +%H:%M) queue start" >> $LOG
for spec in "dmir_cls_v4 80 mae" "dmir_lcr_v4 100 rmse" "dmir_lcl_v4 100 rmse"; do
  set -- $spec
  echo "$(date +%H:%M) $1 search start" >> $LOG
  DMIR_REG_METRIC=$3 DMIR_PARALLEL=2 bash $R/unas/run_chunked.sh $1 150 $2 >> $LOG 2>&1
  echo "$(date +%H:%M) $1 selection start" >> $LOG
  cd ~/uNAS
  ~/dmir_nas/bin/python $R/unas/select_by_seeds.py $1 12 2>&1 \
    | grep -E --line-buffered "seed [0-9]:|best|smallest|saved candidates|Traceback|Error" >> $LOG
done

python3 - > /tmp/sv_extra_dmir_v4.json <<EOF
import json
R = "$R"
out = {}
for search, task, loss in (("dmir_cls_v4", "classification", "logits"),
                           ("dmir_lcr_v4", "regression_lcr", "mse"),
                           ("dmir_lcl_v4", "regression_lcl", "mse")):
    try:
        best = json.load(open(f"{R}/datasets/dmir/results/nas-v2/{search}/selection.json"))["best"]
    except FileNotFoundError:
        continue
    out[f"{search}_best"] = ["dmir", f"datasets/dmir/results/nas-v2/{search}/{best}.h5", task, {}, loss, {}]
print(json.dumps(out))
EOF
echo "$(date +%H:%M) controls: $(cat /tmp/sv_extra_dmir_v4.json | head -c 400)" >> $LOG
cd ~/uNAS
SV_EXTRA_JSON=/tmp/sv_extra_dmir_v4.json RECIPE=dscnn ~/dmir_nas/bin/python $R/unas/seed_variance.py \
  dmir_cls_v4_best dmir_lcr_v4_best dmir_lcl_v4_best 2>&1 | grep -E --line-buffered "seed [0-9]:|Traceback|Error" >> $LOG
CUDA_VISIBLE_DEVICES= ~/dmir_nas/bin/python $R/unas/resource_bias.py dmir_cls_v4 dmir_lcr_v4 dmir_lcl_v4 2>&1 \
  | grep -E "^dmir|Traceback|Error" >> $LOG
echo "$(date +%H:%M) DMIR-V4-QUEUE-DONE" >> $LOG
