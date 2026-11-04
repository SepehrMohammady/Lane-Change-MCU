#!/bin/bash
# Corrected final regression searches (2026-11-04): the paper's v4 regression searches again after the
# trainer fix of 2026-11-03 (unas/patch_fork.py patch 3), with the original configs, 150 candidates,
# epochs, metrics, seed 42 and two workers; the names carry _k3. Each search is followed by the
# five-seed validation choice (unas/select_by_seeds.py). The classification searches of this series
# ran through run_noise_aware.sh (single mode: highd_cls_v4_k3_single_s42, dmir_cls_v4_k3_single_s42).
# Usage (from Windows, detached): wsl -e bash /mnt/c/Projects/PhD/DIMIR/unas/run_v4_k3.sh
R=/mnt/c/Projects/PhD/DIMIR
LOG=$R/runs/nas/v4_k3_pipeline.log
source ~/dmir_nas/env.sh
export DMIR_DATA_ROOT=$R/datasets/dmir/data HIGHD_DATA_ROOT=$R/datasets/highd/data/prepared TF_CPP_MIN_LOG_LEVEL=2

echo "$(date '+%m-%d %H:%M') queue start" >> $LOG
for spec in "dmir_lcr_v4 100 rmse" "dmir_lcl_v4 100 rmse" "highd_ttlc_v4 50 mae"; do
  set -- $spec
  echo "$(date '+%m-%d %H:%M') ${1}_k3 search start" >> $LOG
  case "$1" in
    dmir_*)  DMIR_V4_SUFFIX=_k3 DMIR_REG_METRIC=$3 DMIR_PARALLEL=2 bash $R/unas/run_chunked.sh $1 150 $2 >> $LOG 2>&1 ;;
    highd_*) HIGHD_V2_SUFFIX=_k3 HIGHD_PARALLEL=2 bash $R/unas/run_chunked_highd.sh $1 150 $2 >> $LOG 2>&1 ;;
  esac
  echo "$(date '+%m-%d %H:%M') ${1}_k3 selection start" >> $LOG
  cd ~/uNAS
  ~/dmir_nas/bin/python $R/unas/select_by_seeds.py "${1}_k3" 12 2>&1 \
    | grep -E --line-buffered "seed [0-9]:|best|smallest|saved candidates|Traceback|Error" >> $LOG
done
echo "$(date '+%m-%d %H:%M') V4-K3-QUEUE-DONE" >> $LOG
