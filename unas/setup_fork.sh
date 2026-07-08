#!/bin/bash
# Register the DMIR adapters into a local checkout of the ELIOS uNAS fork.
# Usage (from WSL):
#   FORK=~/uNAS REPO=/mnt/c/Projects/PhD/DIMIR bash "$REPO/unas/setup_fork.sh"
set -e
FORK="${FORK:-$HOME/uNAS}"
REPO="${REPO:-/mnt/c/Projects/PhD/DIMIR}"

if [ ! -d "$FORK/uNAS" ]; then
  echo "cloning ELIOS uNAS fork into $FORK"
  git clone https://github.com/Elios-Lab/uNAS.git "$FORK"
fi

echo "copying adapters into the fork"
cp "$REPO/unas/dmir_dataset.py" "$FORK/dataset/dmir_dataset.py"
cp "$REPO/unas/dmir_config.py"  "$FORK/configs/dmir_config.py"

# register dataset export (idempotent; leading newline in case the file
# does not end in one, else the import fuses onto the last line)
grep -q "from .dmir_dataset import DMIR_Dataset" "$FORK/dataset/__init__.py" \
  || printf '\nfrom .dmir_dataset import DMIR_Dataset\n' >> "$FORK/dataset/__init__.py"

# register configs in driver.py's _CONFIGS (idempotent, via python patch)
python3 - "$FORK/driver.py" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
entries = {
    "dmir_lcr":       ("configs.dmir_config", "get_dmir_lcr_setup"),
    "dmir_lcl":       ("configs.dmir_config", "get_dmir_lcl_setup"),
    "dmir_cls":       ("configs.dmir_config", "get_dmir_cls_setup"),
    "dmir_cls_noind": ("configs.dmir_config", "get_dmir_cls_noind_setup"),
}
lines = "".join(
    f'    "{k}": ("{m}", "{f}"),\n' for k, (m, f) in entries.items()
    if f'"{k}"' not in s
)
if lines:
    s = re.sub(r"_CONFIGS\s*=\s*\{", "_CONFIGS = {\n" + lines, s, count=1)
    open(p, "w", encoding="utf-8").write(s)
    print("registered:", ", ".join(entries))
else:
    print("configs already registered")
PYEOF

echo "done. Fork at: $FORK"
