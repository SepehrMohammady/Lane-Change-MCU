#!/bin/bash
# Queue the core DMIR NAS searches sequentially on one GPU (overnight run).
# Regression (primary, un-leaky) first, then classification + no-indicator
# ablation. Each logs to runs/nas/. Usage (from WSL):
#   bash /mnt/c/Projects/PhD/DIMIR/unas/run_all.sh
set -e
REPO="/mnt/c/Projects/PhD/DIMIR"
RUN="$REPO/unas/full_search.sh"

#            config           rounds epochs
bash "$RUN"  dmir_lcr         150    100
bash "$RUN"  dmir_lcl         150    100
bash "$RUN"  dmir_cls         150     80
bash "$RUN"  dmir_cls_noind   150     80

echo "=== ALL SEARCHES DONE ==="
for c in dmir_lcr dmir_lcl dmir_cls dmir_cls_noind; do
  echo "--- $c pareto models ---"
  ls "$HOME/uNAS/artifacts/$c/models/" 2>/dev/null | wc -l
done
