#!/bin/bash
# Cost-first search pilot (2026-11-10), rules fixed before the first run: fresh seeds for the uNAS baseline
# with the cost-first final choice (selection_costfirst.json), then the cost-first search on the highD
# classifier and LCIR intention with search seeds 42 and 43 (unas/run_noise_aware.sh, mode costfirst).
# One job at a time: a search takes most of the WSL memory.
R=/mnt/c/Projects/PhD/DIMIR; LOG=$R/runs/nas/costfirst_pilot.log
mkdir -p $R/runs/nas
source ~/dmir_nas/env.sh
cd ~/uNAS
export HIGHD_DATA_ROOT=$R/datasets/highd/data/prepared DMIR_DATA_ROOT=$R/datasets/dmir/data
for s in highd_cls_v4_k3_single_s42 highd_cls_v4_k3_single_s43 dmir_cls_v4_k3_single_s42 dmir_cls_v4_k3_single_s43; do
  echo "$(date '+%m-%d %H:%M') $s, cost-first choice, fresh seeds" >> $LOG
  FINAL_SEL_TAG=_costfirst FINAL_TAG=_costfirst ~/dmir_nas/bin/python $R/unas/final_eval.py $s 2>&1 \
    | grep -E --line-buffered "seed|chosen|mean|Traceback|Error" >> $LOG
done
for task in highd_cls dmir_cls; do
  for seed in 42 43; do
    echo "$(date '+%m-%d %H:%M') $task costfirst s$seed start" >> $LOG
    NAME_TAG=_k3 bash $R/unas/run_noise_aware.sh $task costfirst $seed >> $LOG 2>&1
  done
done
echo "$(date '+%m-%d %H:%M') COSTFIRST-PILOT-DONE" >> $LOG
