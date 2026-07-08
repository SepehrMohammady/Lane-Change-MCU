"""Harvest the Pareto fronts from a set of finished/crashed searches.

For each task, independently evaluate every saved .h5 on the test set with our
own metrics (never trust the fork's console numbers), record params (== int8
weight bytes, the fork's `ms`), and write a per-task CSV plus a printed summary
with the accuracy/footprint front. Exact peak-RAM/MACs are deferred to ST Edge
AI at deployment.

Run in the WSL dmir_nas venv:
  source ~/dmir_nas/env.sh
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/data \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/harvest_fronts.py \
    ~/uNAS/artifacts /mnt/c/Projects/PhD/DIMIR/runs/nas/fronts
"""
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

TASKS = {
    "dmir_lcr": ("regression_lcr", "data-prepared-lcr", "lcr"),
    "dmir_lcl": ("regression_lcl", "data-regression-lcl", "lcl"),
    "dmir_cls": ("classification", "data-classification", "multi"),
    "dmir_cls_noind": ("classification", "data-classification", "multi"),
}
# SOTA / baseline reference points for context.
REF = {
    "dmir_lcr": "SOTA MAE 0.298 / RMSE 0.510; our DSCNN MAE 0.318 / RMSE 0.439",
    "dmir_lcl": "SOTA MAE 0.298 / RMSE 0.510; our DSCNN MAE 0.333 / RMSE 0.459",
    "dmir_cls": "internal ref acc 92%; our DSCNN 91.5%; blinker-only 81.5%",
    "dmir_cls_noind": "no-indicator ablation (blinker channels removed)",
}


def load_test(task_kind, folder, sfx, drop_indicators):
    root = Path(os.environ["DMIR_DATA_ROOT"]) / folder
    def rd(split, xy):
        with open(root / f"{xy}_{split}_{sfx}.pkl", "rb") as f:
            return pickle.load(f)
    xtr = rd("train", "x").astype(np.float32)
    xte = rd("test", "x").astype(np.float32)
    yte = rd("test", "y")
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    np.clip(xte, lo, hi, out=xte)
    if drop_indicators:
        idx = [3, 4] if task_kind != "classification" else [28, 29]
        keep = [c for c in range(xte.shape[2]) if c not in idx]
        xte = xte[:, :, keep]
    if task_kind == "classification":
        return xte, yte.astype(np.int64)
    return xte, yte.astype(np.float32)


def harvest(task, artifacts_dir, out_dir):
    kind, folder, sfx = TASKS[task]
    drop_ind = task.endswith("_noind")
    models_dir = Path(artifacts_dir) / task / "models"
    files = sorted(models_dir.glob("*.h5"))
    if not files:
        print(f"[{task}] no models"); return None
    xte, yte = load_test(kind, folder, sfx, drop_ind)
    is_cls = kind == "classification"

    rows = []
    for mp in files:
        try:
            m = tf.keras.models.load_model(mp, compile=False)
            p = m.predict(xte, batch_size=1024, verbose=0)
            n = int(m.count_params())
            if is_cls:
                acc = float((p.argmax(1) == yte).mean())
                rows.append({"model": mp.name, "params": n, "int8_KB": round(n / 1024, 1),
                             "test_acc": round(acc, 4)})
            else:
                p = p.squeeze(-1)
                mae = float(np.mean(np.abs(p - yte)))
                rmse = float(np.sqrt(np.mean((p - yte) ** 2)))
                rows.append({"model": mp.name, "params": n, "int8_KB": round(n / 1024, 1),
                             "test_mae": round(mae, 4), "test_rmse": round(rmse, 4)})
        except Exception as e:  # noqa: BLE001
            print(f"  [{task}] {mp.name} FAIL {str(e)[:80]}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(out_dir) / f"{task}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    key = "test_acc" if is_cls else "test_mae"
    rows.sort(key=lambda r: -r[key] if is_cls else r[key])
    print(f"\n=== {task} ({len(rows)} models) — {REF[task]} ===")
    print(f"  ref: {csv_path}")
    best = rows[0]
    # smallest model within 0.01 of the best metric (deployment sweet spot)
    thr = best[key] - 0.01 if is_cls else best[key] + 0.01
    ok = [r for r in rows if (r[key] >= thr if is_cls else r[key] <= thr)]
    smallest = min(ok, key=lambda r: r["params"])
    hdr = "acc" if is_cls else "MAE     RMSE"
    print(f"  {'model':<16}{'params':>9}{'KB':>7}   {hdr}")
    for tag, r in [("BEST", best), ("SMALLEST~best", smallest)]:
        metric = f"{r['test_acc']:.4f}" if is_cls else f"{r['test_mae']:.4f}  {r['test_rmse']:.4f}"
        print(f"  {tag:<14} {r['model']:<12}{r['params']:>9}{r['int8_KB']:>7}   {metric}")
    return {"task": task, "n": len(rows), "best": best, "smallest_near_best": smallest}


def main(artifacts_dir, out_dir):
    summary = []
    for task in TASKS:
        r = harvest(task, artifacts_dir, out_dir)
        if r:
            summary.append(r)
    print("\n================ SUMMARY ================")
    for s in summary:
        b = s["best"]
        m = f"acc {b['test_acc']}" if "test_acc" in b else f"MAE {b['test_mae']} / RMSE {b['test_rmse']}"
        print(f"{s['task']:<16} {s['n']:>2} models  best: {m} @ {b['int8_KB']} KB int8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
