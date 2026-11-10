#!/bin/bash
# LCIR searches on min-max scaled inputs (2026-11-10). The corrected seed-42 searches of the three LCIR heads
# again (same configs, budgets, epochs, metrics, 150 candidates, two workers), with every input feature
# scaled to 0-1 by the training range (runs named *_mm*, unas/dmir_dataset.py), each followed by its
# five-seed selection (rule "best"). Then unas/input_scaling_check.py trains the new choices (scaled
# inputs) and the raw-input choices (raw inputs) with fresh seeds 100-104 and scores float and full int8
# (int8 I/O) on the test set: datasets/dmir/results/seeds/input_scaling_mm.jsonl.
# Usage (from Windows, detached): wsl -e bash /mnt/c/Projects/PhD/DIMIR/unas/run_mm_searches.sh
R=/mnt/c/Projects/PhD/DIMIR
LOG=$R/runs/nas/mm_searches.log
source ~/dmir_nas/env.sh
export DMIR_DATA_ROOT=$R/datasets/dmir/data TF_CPP_MIN_LOG_LEVEL=2
mkdir -p $R/runs/nas

echo "$(date '+%m-%d %H:%M') queue start" >> $LOG
echo "$(date '+%m-%d %H:%M') dmir_cls_v4_k3_mm_single_s42 search, selection, fresh seeds" >> $LOG
NAME_TAG=_k3_mm bash $R/unas/run_noise_aware.sh dmir_cls single 42 >> $LOG 2>&1

for spec in "dmir_lcr_v4 100 rmse" "dmir_lcl_v4 100 rmse"; do
  set -- $spec
  echo "$(date '+%m-%d %H:%M') ${1}_k3_mm search start" >> $LOG
  DMIR_V4_SUFFIX=_k3_mm DMIR_REG_METRIC=$3 DMIR_PARALLEL=2 bash $R/unas/run_chunked.sh $1 150 $2 >> $LOG 2>&1
  echo "$(date '+%m-%d %H:%M') ${1}_k3_mm selection start" >> $LOG
  cd ~/uNAS
  ~/dmir_nas/bin/python $R/unas/select_by_seeds.py "${1}_k3_mm" 12 2>&1 \
    | grep -E --line-buffered "seed [0-9]:|best|smallest|saved candidates|scaled|Traceback|Error" >> $LOG
done

echo "$(date '+%m-%d %H:%M') fresh seeds 100-104, float and int8, new and raw-input choices" >> $LOG
cd ~/uNAS
SEEDS=100,101,102,103,104 OUT_NAME=input_scaling_mm.jsonl \
  CHOICES_FROM=dmir_cls_v4_k3_mm_single_s42,dmir_lcr_v4_k3_mm,dmir_lcl_v4_k3_mm,dmir_cls_v4_k3_single_s42,dmir_lcr_v4_k3,dmir_lcl_v4_k3 \
  ~/dmir_nas/bin/python $R/unas/input_scaling_check.py 2>&1 | grep -E --line-buffered "seed|Traceback|Error" >> $LOG
echo "$(date '+%m-%d %H:%M') MM-SEARCHES-DONE" >> $LOG
