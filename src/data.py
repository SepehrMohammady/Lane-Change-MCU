"""Data loading and sanitation for the prepared DIMIR splits.

The dataset arrives as pickled numpy arrays, already windowed and
normalised by the data provider: windows of 50 timesteps x 31 features.

  classification : y in {0: none, 1: LCR intention, 2: LCL intention}
  regression_*   : y = time-to-lane-change in [0.0, 4.0] s, step 0.1

Known data facts (verified 2026-07-07, see docs/DATA.md):
  * train/val/test are balanced for classification (1/3 each class);
  * test splits contain rare extreme spikes (|x| up to ~5e6) on feature
    pairs (12, 13) for classification/LCL and (14, 15) for LCR, absent
    from train — ``clip_to_train_range`` handles them;
  * feature 7 is constant zero in the classification training split.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config, TASK_LAYOUT


# ---------------------------------------------------------------------------
# raw loading
# ---------------------------------------------------------------------------

def load_split(cfg: Config, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split ('train' | 'val' | 'test') as float32 arrays."""
    folder, suffix = TASK_LAYOUT[cfg.task]
    base = cfg.data_dir / folder
    with open(base / f"x_{split}_{suffix}.pkl", "rb") as f:
        x = pickle.load(f).astype(np.float32)
    with open(base / f"y_{split}_{suffix}.pkl", "rb") as f:
        y = pickle.load(f)
    y = y.astype(np.int64) if cfg.is_classification else y.astype(np.float32)
    return x, y


@dataclass
class SplitBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_min: np.ndarray | None = None   # per-feature clip bounds (from train)
    train_max: np.ndarray | None = None
    n_clipped_val: int = 0
    n_clipped_test: int = 0

    def summary(self) -> str:
        lines = []
        for name in ("train", "val", "test"):
            x = getattr(self, f"x_{name}")
            y = getattr(self, f"y_{name}")
            lines.append(
                f"{name:5s}  x{tuple(x.shape)}  y{tuple(y.shape)}  "
                f"x-range [{x.min():.2f}, {x.max():.2f}]"
            )
        if self.train_min is not None:
            lines.append(
                f"clipped cells -> val: {self.n_clipped_val}, test: {self.n_clipped_test}"
            )
        return "\n".join(lines)


def load_task(cfg: Config) -> SplitBundle:
    """Load all three splits, optionally clip val/test to train range,
    optionally take a stratified subset (for smoke tests)."""
    x_tr, y_tr = load_split(cfg, "train")
    x_va, y_va = load_split(cfg, "val")
    x_te, y_te = load_split(cfg, "test")

    bundle = SplitBundle(x_tr, y_tr, x_va, y_va, x_te, y_te)

    if cfg.clip_to_train_range:
        bundle.train_min = x_tr.min(axis=(0, 1))
        bundle.train_max = x_tr.max(axis=(0, 1))
        for name in ("val", "test"):
            x = getattr(bundle, f"x_{name}")
            below = x < bundle.train_min
            above = x > bundle.train_max
            n = int(below.sum() + above.sum())
            np.clip(x, bundle.train_min, bundle.train_max, out=x)
            setattr(bundle, f"n_clipped_{name}" if name != "val" else "n_clipped_val", n)

    if cfg.subset_fraction < 1.0:
        rng = np.random.default_rng(cfg.seed)
        for name in ("train", "val", "test"):
            x = getattr(bundle, f"x_{name}")
            y = getattr(bundle, f"y_{name}")
            n = max(int(len(x) * cfg.subset_fraction), 64)
            idx = _stratified_indices(y, n, rng, cfg.is_classification)
            setattr(bundle, f"x_{name}", x[idx])
            setattr(bundle, f"y_{name}", y[idx])

    return bundle


def _stratified_indices(y, n, rng, is_classification) -> np.ndarray:
    if not is_classification:
        return rng.choice(len(y), size=min(n, len(y)), replace=False)
    idx = []
    classes = np.unique(y)
    per = max(n // len(classes), 8)
    for c in classes:
        pool = np.where(y == c)[0]
        idx.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
    return np.concatenate(idx)


# ---------------------------------------------------------------------------
# torch loaders
# ---------------------------------------------------------------------------

def make_loaders(cfg: Config, bundle: SplitBundle) -> dict[str, DataLoader]:
    """TensorDataset loaders; x transposed to (N, C=31, T=50) for Conv1d."""
    loaders = {}
    for name in ("train", "val", "test"):
        x = torch.from_numpy(getattr(bundle, f"x_{name}")).transpose(1, 2).contiguous()
        y = torch.from_numpy(getattr(bundle, f"y_{name}"))
        loaders[name] = DataLoader(
            TensorDataset(x, y),
            batch_size=cfg.batch_size,
            shuffle=(name == "train"),
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
