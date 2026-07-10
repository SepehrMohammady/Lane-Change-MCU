#!/bin/bash
# RMSE-aware regression searches (fresh, save all models) to close the gap vs
# the internal-reference Transformers on RMSE. Distinct config names ->
# separate artifacts dirs (artifacts/dmir_lcr_rmse, dmir_lcl_rmse).
# Prereq: run setup_fork.sh once (registers the *_rmse configs + patches the
# trainer to be RMSE-aware). Usage (from WSL):
#   bash /mnt/c/Projects/PhD/DIMIR/unas/run_rmse.sh
export DMIR_REG_METRIC=rmse
export DMIR_SAVE_CRITERIA=all
REPO="/mnt/c/Projects/PhD/DIMIR"
bash "$REPO/unas/run_chunked.sh" dmir_lcr_rmse 150 100
bash "$REPO/unas/run_chunked.sh" dmir_lcl_rmse 150 100
echo "=== RMSE REGRESSION SEARCHES DONE ==="
for c in dmir_lcr_rmse dmir_lcl_rmse; do
  echo "$c: $(ls "$HOME/uNAS/artifacts/$c/models" 2>/dev/null | wc -l) models"
done
