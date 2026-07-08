"""DMIR dataset adapter for the ELIOS uNAS fork.

Drop this file into the fork's `dataset/` directory and register it in
`dataset/__init__.py`. It serves the real prepared DMIR windows (50 timesteps
x 31 channels) for the three tasks, using the same sanitation as our PyTorch
pipeline (val/test clipped to the per-channel train range; see docs/DATA.md).

uNAS convention: return UNBATCHED tf.data.Dataset of (window, label) with
window shape (timesteps, channels) — our pickles are already (N, 50, 31),
channels-last, so no transpose is needed. num_classes = 1 selects regression.

The prepared pickles already follow the official driver-wise split
(verified: scripts/analysis/verify_split.py), so we load train/val/test as-is.

Optional `drop_indicators=True` removes the turn-signal channels for the
label-leak ablation (indicator channels: classification 28/29, regression 3/4;
see docs/research/feature-map.md).
"""
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from uNAS.dataset import Dataset

# DMIR repo's data/ directory. Override with the DMIR_DATA_ROOT env var — the
# fork runs under WSL/Linux where the Windows repo is at /mnt/c/... e.g.
#   export DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/data
DATA_ROOT = Path(os.environ.get("DMIR_DATA_ROOT", r"C:\Projects\PhD\DIMIR\data"))

_LAYOUT = {
    "classification": ("data-classification", "multi", 3),
    "regression_lcl": ("data-regression-lcl", "lcl", 1),
    "regression_lcr": ("data-prepared-lcr", "lcr", 1),
}
_INDICATOR_IDX = {"classification": [28, 29],
                  "regression_lcl": [3, 4], "regression_lcr": [3, 4]}


def _load(folder, sfx, split):
    base = DATA_ROOT / folder
    with open(base / f"x_{split}_{sfx}.pkl", "rb") as f:
        x = pickle.load(f).astype(np.float32)
    with open(base / f"y_{split}_{sfx}.pkl", "rb") as f:
        y = pickle.load(f)
    return x, y


class DMIR_Dataset(Dataset):
    def __init__(self, task="regression_lcr", clip_to_train_range=True,
                 drop_indicators=False):
        if task not in _LAYOUT:
            raise ValueError(f"task must be one of {list(_LAYOUT)}")
        folder, sfx, n_out = _LAYOUT[task]
        self._task = task
        self._num_classes = n_out

        xtr, ytr = _load(folder, sfx, "train")
        xva, yva = _load(folder, sfx, "val")
        xte, yte = _load(folder, sfx, "test")

        if clip_to_train_range:
            lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
            np.clip(xva, lo, hi, out=xva)
            np.clip(xte, lo, hi, out=xte)

        if drop_indicators:
            keep = [c for c in range(xtr.shape[2])
                    if c not in _INDICATOR_IDX[task]]
            xtr, xva, xte = xtr[:, :, keep], xva[:, :, keep], xte[:, :, keep]

        cast = np.int64 if n_out > 1 else np.float32
        self._data = {
            "train": (xtr, ytr.astype(cast)),
            "val": (xva, yva.astype(cast)),
            "test": (xte, yte.astype(cast)),
        }
        self._input_shape = (xtr.shape[1], xtr.shape[2])  # (50, 31) or (50, 29)

    def _ds(self, split):
        x, y = self._data[split]
        return tf.data.Dataset.from_tensor_slices((x, y))

    def train_dataset(self) -> tf.data.Dataset:
        return self._ds("train")

    def validation_dataset(self) -> tf.data.Dataset:
        return self._ds("val")

    def test_dataset(self) -> tf.data.Dataset:
        return self._ds("test")

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self._input_shape
