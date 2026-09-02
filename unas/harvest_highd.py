"""Harvest highD search fronts: independently re-evaluate every saved .h5 on
the test windows with our own metrics (never trust console numbers).

Run in the WSL dmir_nas venv, CPU-only so a live search keeps the GPU:
  source ~/dmir_nas/env.sh
  CUDA_VISIBLE_DEVICES= HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/harvest_highd.py \
    ~/uNAS/artifacts <out_dir> [highd_cls highd_ttlc]

Writes <out_dir>/<task>.csv: model, params, int8_KB, and test metrics
(accuracy/macro-F1 for cls; MAE/RMSE seconds for ttlc, their gt convention).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import keras  # noqa: E402

IN_LEN, SEQ_LEN, FPS = 10, 35, 5
N_SLIDES = SEQ_LEN - IN_LEN + 1
DATA_ROOT = Path(os.environ["HIGHD_DATA_ROOT"])


def windows(split):
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


def main(artifacts, out_dir, tasks):
    xtr, _, _ = windows("train")
    lo = xtr.reshape(-1, 18).min(0)
    hi = xtr.reshape(-1, 18).max(0)
    xte, yte, tte = windows("test")
    xn = ((xte - lo) / (hi - lo + 1e-12)).astype(np.float32)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        cls = task.startswith("highd_cls")
        mdir = Path(artifacts).expanduser() / task / "models"
        if not mdir.is_dir():
            print(f"{task}: no models dir, skipping"); continue
        if cls:
            x, y = xn, yte.astype(np.int64)
        else:
            m = yte != 0
            x, y = xn[m], tte[m].astype(np.float32)

        rows = []
        for h5 in sorted(mdir.glob("*.h5")):
            try:
                model = keras.models.load_model(h5, compile=False)
            except Exception as e:
                print(f"  {h5.name}: LOAD FAILED {e}"); continue
            params = int(model.count_params())
            p = model.predict(x, batch_size=4096, verbose=0)
            if cls:
                pred = p.argmax(1)
                acc = float((pred == y).mean())
                f1 = []
                for c in range(3):
                    tp = ((pred == c) & (y == c)).sum()
                    fp = ((pred == c) & (y != c)).sum()
                    fn = ((pred != c) & (y == c)).sum()
                    f1.append(2 * tp / max(2 * tp + fp + fn, 1))
                rows.append({"model": h5.stem, "params": params,
                             "int8_KB": round(params / 1024, 2),
                             "test_acc": round(acc, 5),
                             "test_macro_f1": round(float(np.mean(f1)), 5)})
                print(f"  {h5.stem}: {params:>7,} params  acc {acc:.4f}")
            else:
                e = p.squeeze(-1) - y
                mae, rmse = float(np.abs(e).mean()), float(np.sqrt((e**2).mean()))
                rows.append({"model": h5.stem, "params": params,
                             "int8_KB": round(params / 1024, 2),
                             "test_mae": round(mae, 5), "test_rmse": round(rmse, 5)})
                print(f"  {h5.stem}: {params:>7,} params  MAE {mae:.4f}  RMSE {rmse:.4f}")
            keras.backend.clear_session()

        if rows:
            path = out / f"{task}.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
            print(f"{task}: {len(rows)} models -> {path}")


if __name__ == "__main__":
    artifacts = sys.argv[1] if len(sys.argv) > 1 else "~/uNAS/artifacts"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    tasks = sys.argv[3:] or ["highd_cls", "highd_ttlc"]
    main(artifacts, out_dir, tasks)
