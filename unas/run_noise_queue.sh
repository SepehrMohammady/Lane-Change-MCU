#!/bin/bash
# Run several unas/run_noise_aware.sh jobs one after the other (one GPU, 15 GB WSL: never two at once).
#   bash /mnt/c/Projects/PhD/DIMIR/unas/run_noise_queue.sh "highd_cls random 42" "highd_cls fixed 42" ...
# Progress: runs/nas/noise_queue.log; each job also writes its own chunk logs.
REPO=/mnt/c/Projects/PhD/DIMIR
LOG=$REPO/runs/nas/noise_queue.log
mkdir -p "$REPO/runs/nas"
for job in "$@"; do
  set -- $job
  echo "$(date '+%m-%d %H:%M') start $job" >> "$LOG"
  bash "$REPO/unas/run_noise_aware.sh" "$1" "$2" "$3" >> "$LOG" 2>&1
  echo "$(date '+%m-%d %H:%M') end $job (exit $?)" >> "$LOG"
done
echo "$(date '+%m-%d %H:%M') queue done" >> "$LOG"
