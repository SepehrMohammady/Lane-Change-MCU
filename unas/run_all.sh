#!/bin/bash
# Queue the core DMIR NAS searches sequentially on one GPU (overnight run).
# Regression (primary, un-leaky) first, then classification + no-indicator
# ablation. Each logs to runs/nas/. Usage (from WSL):
#   bash /mnt/c/Projects/PhD/DIMIR/unas/run_all.sh
# No `set -e`: a failure in one search must not abort the remaining queue.
REPO="/mnt/c/Projects/PhD/DIMIR"
RUN="$REPO/unas/full_search.sh"

run() { echo ">>> $* @ $(date)"; bash "$RUN" "$@" || echo "!!! $1 FAILED (continuing)"; }

#    config           rounds epochs
run  dmir_lcr         150    100
run  dmir_lcl         150    100
run  dmir_cls         150     80
run  dmir_cls_noind   150     80

echo "=== ALL SEARCHES DONE ==="
for c in dmir_lcr dmir_lcl dmir_cls dmir_cls_noind; do
  echo "--- $c pareto models ---"
  ls "$HOME/uNAS/artifacts/$c/models/" 2>/dev/null | wc -l
done
