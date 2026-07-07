"""Quantify the turn-indicator label leak in the 3-class classification task.

The feature set includes the driver's turn-signal state (ternary
DirectionIndicator, split into two binary channels). If drivers signal before
a lane change, those channels nearly announce the label, so headline accuracy
would mostly measure "read the blinker" rather than anticipation.

Verified channel positions (see docs/research/feature-map.md):
  classification: indicators at channels 28 (left) and 29 (right).

Findings (2026-07-07): indicator-only test accuracy 0.815 (full model 0.915,
published-internal reference 0.92). Blinker-active-in-window rate: class 0
(no-intent) 6.5%, class 1 (LCR) 91.7%, class 2 (LCL) 70.7%. Implication:
classification headline accuracy is largely explained by the turn signal;
the meaningful contribution is the time-to-lane-change regression and the
deployment efficiency, plus classification on the no-signal subset.

Usage:  .venv\\Scripts\\python scripts/analysis/indicator_leak.py
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

DATA = Path(__file__).resolve().parents[2] / "data" / "data-classification"
IND = [28, 29]  # left/right indicator binaries (verified)


def load(split):
    with open(DATA / f"x_{split}_multi.pkl", "rb") as f:
        x = pickle.load(f).astype(np.float32)
    with open(DATA / f"y_{split}_multi.pkl", "rb") as f:
        y = pickle.load(f).astype(np.int64)
    return x, y


def feats(x):
    ind = x[:, :, IND]
    return np.concatenate([ind.mean(1), ind.max(1), ind.min(1),
                           ind[:, -1, :], ind[:, -10:, :].mean(1)], axis=1)


def main():
    xtr, ytr = load("train")
    clf = LogisticRegression(max_iter=2000).fit(feats(xtr), ytr)
    for name in ("val", "test"):
        x, y = load(name)
        print(f"indicator-ONLY accuracy [{name}]: {(clf.predict(feats(x)) == y).mean():.4f}")

    on = np.abs(xtr[:, :, IND]).max(axis=(1, 2)) > 1e-6
    print("\nblinker-active-in-window rate per class (train):")
    for c, nm in [(0, "no-intent"), (1, "LCR"), (2, "LCL")]:
        print(f"  class {c} ({nm}): {on[ytr == c].mean():.3f}")

    xte, yte = load("test")
    hard = ~(np.abs(xte[:, :, IND]).max(axis=(1, 2)) > 1e-6)
    print(f"\ntest windows with NO blinker: {hard.mean():.3f} ({hard.sum()}/{len(hard)})")
    print("no-blinker class counts:",
          {int(c): int((yte[hard] == c).sum()) for c in (0, 1, 2)})


if __name__ == "__main__":
    main()
