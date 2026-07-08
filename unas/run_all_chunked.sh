#!/bin/bash
# Complete all four DMIR searches to their targets via chunked resume.
# Resumes each from its existing checkpoint, so no work is repeated.
REPO="/mnt/c/Projects/PhD/DIMIR"
C="$REPO/unas/run_chunked.sh"
bash "$C" dmir_lcr       150 100
bash "$C" dmir_lcl       150 100
bash "$C" dmir_cls       150  80
bash "$C" dmir_cls_noind 150  80
echo "=== ALL SEARCHES COMPLETE (or capped) ==="
for c in dmir_lcr dmir_lcl dmir_cls dmir_cls_noind; do
  echo "$c: $(ls "$HOME/uNAS/artifacts/$c/models" 2>/dev/null | wc -l) models"
done
