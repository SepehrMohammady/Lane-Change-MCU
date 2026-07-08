"""Independently evaluate saved NAS models on the DMIR test set.

Sanity-checks the fork's reported errors with our own numpy/sklearn metrics,
and compares against trivial baselines (predict-mean / predict-median). Run in
the WSL dmir_nas venv:

  source ~/dmir_nas/env.sh
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/data \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/verify_models.py \
    regression_lcr ~/uNAS/artifacts/dmir_lcr/models
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

LAYOUT = {"regression_lcr": ("data-prepared-lcr", "lcr"),
          "regression_lcl": ("data-regression-lcl", "lcl"),
          "classification": ("data-classification", "multi")}


def load_split(task, split):
    folder, sfx = LAYOUT[task]
    base = Path(os.environ["DMIR_DATA_ROOT"]) / folder
    with open(base / f"x_{split}_{sfx}.pkl", "rb") as f:
        x = pickle.load(f).astype(np.float32)
    with open(base / f"y_{split}_{sfx}.pkl", "rb") as f:
        y = pickle.load(f)
    return x, y


def main(task, models_dir):
    xtr, _ = load_split(task, "train")
    xte, yte = load_split(task, "test")
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    np.clip(xte, lo, hi, out=xte)
    is_cls = task == "classification"

    if is_cls:
        base = (yte == np.bincount(yte).argmax()).mean()
        print(f"majority-class test acc: {base:.4f}")
    else:
        yte = yte.astype(np.float32)
        mean_mae = np.mean(np.abs(yte - yte.mean()))
        med_mae = np.mean(np.abs(yte - np.median(yte)))
        print(f"baselines: predict-mean MAE {mean_mae:.4f}, predict-median MAE {med_mae:.4f}")

    rows = []
    for mp in sorted(Path(models_dir).glob("*.h5")):
        try:
            m = tf.keras.models.load_model(mp, compile=False)
            p = m.predict(xte, batch_size=512, verbose=0)
            n = m.count_params()
            if is_cls:
                acc = (p.argmax(1) == yte).mean()
                rows.append((mp.name, n, acc, None))
            else:
                p = p.squeeze(-1)
                mae = float(np.mean(np.abs(p - yte)))
                rmse = float(np.sqrt(np.mean((p - yte) ** 2)))
                rows.append((mp.name, n, mae, rmse))
        except Exception as e:  # noqa: BLE001
            print(f"  {mp.name}: load/eval FAIL {e!r}"[:120])

    key = (lambda r: -r[2]) if is_cls else (lambda r: r[2])
    rows.sort(key=key)
    print(f"\n{'model':<18}{'params':>10}  " + ("acc" if is_cls else "MAE      RMSE"))
    for name, n, a, b in rows[:12]:
        extra = f"{a:.4f}" if is_cls else f"{a:.4f}   {b:.4f}"
        print(f"{name:<18}{n:>10}  {extra}")
    print(f"\n{len(rows)} models evaluated; best shown first.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
