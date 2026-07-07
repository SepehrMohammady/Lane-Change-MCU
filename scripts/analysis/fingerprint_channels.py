"""Fingerprint the 31 pickle channels to infer identity + normalization.

Signatures:
  - constant channel (std=0)            -> egoLaneWidth (constant 3.75 raw)
  - few unique values                   -> indicators / classification flags
  - monotonic within windows            -> fileTime / UTCTime
  - adjacent-channel exact-equal frac   -> right/left lane pairs
  - mean~0, std~1                       -> StandardScaler confirmation
"""
import pickle
from pathlib import Path

import numpy as np

DATA = Path(r"C:\Projects\PhD\DIMIR\data")

for task, folder, sfx in [("cls", "data-classification", "multi"),
                          ("lcr", "data-prepared-lcr", "lcr")]:
    with open(DATA / folder / f"x_train_{sfx}.pkl", "rb") as f:
        x = pickle.load(f)
    n = min(20000, len(x))
    xs = x[:n]
    print(f"\n=== {task} train (n={n}) ===")
    print(f"{'ch':>3} {'mean':>8} {'std':>8} {'uniq':>7} {'mono%':>6} {'eq_next%':>8} {'corr_next':>9}")
    flat = xs.reshape(-1, xs.shape[2])
    for c in range(xs.shape[2]):
        col = flat[:, c]
        mean, std = col.mean(), col.std()
        uniq = len(np.unique(col[:200000]))
        diffs = np.diff(xs[:, :, c], axis=1)
        mono = float((diffs >= 0).all(axis=1).mean()) * 100
        if c + 1 < xs.shape[2]:
            nxt = flat[:, c + 1]
            eq = float((col == nxt).mean()) * 100
            sd = col.std() * nxt.std()
            corr = float(np.corrcoef(col, nxt)[0, 1]) if sd > 0 else float("nan")
        else:
            eq, corr = float("nan"), float("nan")
        print(f"{c:>3} {mean:>8.3f} {std:>8.3f} {uniq:>7} {mono:>6.1f} {eq:>8.1f} {corr:>9.3f}")
print("\nDONE")
