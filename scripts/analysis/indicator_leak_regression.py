"""Does the turn signal trivially predict time-to-lane-change?

Regression layout has indicators at channels 3 (left) and 4 (right). We fit a
gradient-boosted regressor on indicator-only window summaries and compare test
RMSE against our DSCNN (LCR 0.439, LCL 0.459) and the published Transformer
(0.5102 single-TTLC; internal reference 0.42/0.44).

Usage:  .venv\\Scripts\\python scripts/analysis/indicator_leak_regression.py <lcl|lcr>
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = {"lcl": ("data-regression-lcl", "lcl"),
          "lcr": ("data-prepared-lcr", "lcr")}
IND = [3, 4]  # left/right indicator channels (regression layout)


def load(task, split):
    folder, sfx = LAYOUT[task]
    base = ROOT / "data" / folder
    with open(base / f"x_{split}_{sfx}.pkl", "rb") as f:
        x = pickle.load(f).astype(np.float32)
    with open(base / f"y_{split}_{sfx}.pkl", "rb") as f:
        y = pickle.load(f).astype(np.float32)
    return x, y


def feats(x):
    ind = x[:, :, IND]
    return np.concatenate([ind.mean(1), ind.max(1), ind[:, -1, :],
                           ind[:, -10:, :].mean(1),
                           np.abs(np.diff(ind, axis=1)).sum(1)], axis=1)


def main(task):
    # clip val/test like the pipeline does, so indicators are comparable
    xtr, ytr = load(task, "train")
    xte, yte = load(task, "test")
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    np.clip(xte, lo, hi, out=xte)

    reg = HistGradientBoostingRegressor(max_iter=300).fit(feats(xtr), ytr)
    pred = reg.predict(feats(xte))
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    mae = float(np.mean(np.abs(pred - yte)))
    baseline_rmse = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    print(f"[{task}] indicator-only test RMSE {rmse:.4f}  MAE {mae:.4f}  "
          f"(predict-mean RMSE {baseline_rmse:.4f})")
    print(f"        our DSCNN: {'0.459' if task == 'lcl' else '0.439'}  "
          f"internal ref: {'0.44' if task == 'lcl' else '0.42'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lcr")
