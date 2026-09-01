"""highD dataset adapter for the ELIOS uNAS fork.

Serves the prepared highD lane-change windows (10 timesteps x 18 features,
EarlyLCPred protocol — see datasets/highd/prepare_highd.py). Windows are the
26 slide positions over each 35-frame scenario; features are min-max
normalized with train statistics (their Dataset.py rule).

Tasks:
  highd_cls  — 3-class {LK, RLC, LLC} over all windows
  highd_ttlc — TTLC regression (seconds, their (26-s)/5 convention) on LC
               windows only

Env: HIGHD_DATA_ROOT points at datasets/highd/data/prepared (WSL path).
"""
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from uNAS.dataset import Dataset

DATA_ROOT = Path(os.environ.get(
    "HIGHD_DATA_ROOT", r"C:\Projects\PhD\DIMIR\datasets\highd\data\prepared"))

IN_LEN, SEQ_LEN, FPS = 10, 35, 5
N_SLIDES = SEQ_LEN - IN_LEN + 1


def _windows(split):
    z = np.load(DATA_ROOT / f"{split}.npz")
    feats, label, cross = z["feats"], z["label"], z["cross_idx"]
    S = len(feats)
    sw = np.lib.stride_tricks.sliding_window_view(feats, IN_LEN, axis=1)
    x = np.ascontiguousarray(sw.transpose(0, 1, 3, 2)).reshape(
        S * N_SLIDES, IN_LEN, feats.shape[2]).astype(np.float32)
    y = np.repeat(label, N_SLIDES)
    s_idx = np.tile(np.arange(N_SLIDES), S)
    cx = np.repeat(cross.astype(np.float32), N_SLIDES)
    ttlc = (cx - (s_idx + IN_LEN) + 1) / FPS
    ttlc[cx < 0] = np.nan
    return x, y, ttlc


class HighD_Dataset(Dataset):
    def __init__(self, task="highd_cls"):
        assert task in ("highd_cls", "highd_ttlc")
        self._task = task
        cls = task == "highd_cls"
        self._num_classes = 3 if cls else 1

        splits = {s: _windows(s) for s in ("train", "val", "test")}
        lo = splits["train"][0].reshape(-1, 18).min(0)
        hi = splits["train"][0].reshape(-1, 18).max(0)

        self._data = {}
        for s, (x, y, ttlc) in splits.items():
            xn = ((x - lo) / (hi - lo + 1e-12)).astype(np.float32)
            if cls:
                self._data[s] = (xn, y.astype(np.int64))
            else:
                m = y != 0                       # LC windows only, ttlc in [0.2, 5.2]
                self._data[s] = (xn[m], ttlc[m].astype(np.float32))
        self._input_shape = (IN_LEN, 18)

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
