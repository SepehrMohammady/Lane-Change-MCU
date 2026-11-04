#!/bin/bash
# Two-domain pilot queue (2026-11-04), rules fixed before the first run: the highD-only choice trained on
# highD + exiD pooled with the fresh seeds, then the pooled and worst arms with search seed 42
# (unas/run_twod.sh). One job at a time: a search takes most of the WSL memory.
R=/mnt/c/Projects/PhD/DIMIR; LOG=$R/runs/nas/twod_pilot.log
mkdir -p $R/runs/nas
source ~/dmir_nas/env.sh
cd ~/uNAS
export HIGHD_DATA_ROOT=$R/datasets/highd/data/prepared EXID_DATA_ROOT=$R/datasets/exid/data/prepared
echo "$(date '+%m-%d %H:%M') highd_cls_v4_k3_single_s42 choice on the pooled data" >> $LOG
FINAL_DATA=pooled ~/dmir_nas/bin/python $R/unas/final_eval.py highd_cls_v4_k3_single_s42 2>&1 \
  | grep -E --line-buffered "seed|chosen|mean|Traceback|Error" >> $LOG
for arm in pooled worst; do
  echo "$(date '+%m-%d %H:%M') twod $arm start" >> $LOG
  bash $R/unas/run_twod.sh $arm 42 >> $LOG 2>&1
done
echo "$(date '+%m-%d %H:%M') TWOD-PILOT-DONE" >> $LOG
