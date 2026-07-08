#!/bin/bash
# One-shot WSL2 GPU environment for the ELIOS uNAS fork on the RTX 5070.
# Verified 2026-07-08: TF 2.21 + CUDA 12.9 sees the Blackwell GPU (sm_120).
#
# Usage (from WSL Ubuntu):  bash /mnt/c/Projects/PhD/DIMIR/unas/setup_wsl_env.sh
set -e
VENV="$HOME/dmir_nas"
python3 -m venv "$VENV"
PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip wheel

# TensorFlow with bundled CUDA (latest; brings CUDA 12.9 -> Blackwell sm_120).
"$PIP" install "tensorflow[and-cuda]"

# The pip TF wheel needs its CUDA libs on LD_LIBRARY_PATH and ptxas on PATH.
SP="$VENV/lib/python3.10/site-packages"
NVLIBS=$(find "$SP/nvidia" -name "*.so*" -printf "%h\n" 2>/dev/null | sort -u | tr '\n' ':')
cat > "$VENV/env.sh" <<EOF
# source before using the dmir_nas venv
export LD_LIBRARY_PATH="${NVLIBS}${SP}/tensorflow:\$LD_LIBRARY_PATH"
export PATH="$SP/nvidia/cuda_nvcc/bin:\$PATH"
EOF

# Fork dependencies. dragonfly-opt's legacy build imports numpy -> no isolation.
"$PIP" install scikit-learn scipy tqdm matplotlib ray
"$PIP" install Cython
"$PIP" install --no-build-isolation dragonfly-opt
# tfmot is imported at the top of model_trainer.py (QAT is off during search);
# tfmot 0.8.1 imports fine and pulls tf_keras.
"$PIP" install tensorflow-model-optimization tf_keras

echo "=== GPU check ==="
source "$VENV/env.sh"
"$VENV/bin/python" - <<'PYEOF'
import tensorflow as tf
print("TF", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PYEOF
echo "done. Next: bash unas/setup_fork.sh ; bash unas/run_smoke.sh dmir_lcr 20"
