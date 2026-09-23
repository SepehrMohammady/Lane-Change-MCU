"""Zero-shot transfer of the deployed highD searched models to exiD.

Each Keras model keeps the weights it was deployed with and sees exiD test
windows normalized with the highD training statistics, exactly as a model
flashed for highD would see exiD traffic. The highD test set is evaluated too,
as a check that the model and normalization are the deployed ones. The exiD
breakdown uses the lane-change kind from datasets/exid/prepare_exid.py.

Run in the WSL dmir_nas venv:
  source ~/dmir_nas/env.sh; cd ~/uNAS
  HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/transfer_eval.py
Output: datasets/exid/results/transfer_searched.jsonl
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
import keras  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "unas"))
sys.path.insert(0, os.path.expanduser("~/uNAS"))
from highd_dataset import _windows, IN_LEN, N_SLIDES  # noqa: E402

EXID = REPO / "datasets" / "exid" / "data" / "prepared"
HIGHD = REPO / "datasets" / "highd" / "data" / "prepared"
OUT = REPO / "datasets" / "exid" / "results" / "transfer_searched.jsonl"
MODELS = {"highd_cls_aaaaap": ("datasets/highd/results/models/highd_cls_tight_model_aaaaap.h5", "cls"),
          "highd_cls_aaaaam": ("datasets/highd/results/models/highd_cls_model_aaaaam.h5", "cls"),
          "highd_ttlc_aaaaaw": ("datasets/highd/results/models/highd_ttlc_model_aaaaaw.h5", "ttlc")}


def main():
    kinds = json.loads((EXID / "meta.json").read_text())["kinds"]
    xtr, _, _ = _windows("train", HIGHD)
    lo, hi = xtr.reshape(-1, 18).min(0), xtr.reshape(-1, 18).max(0)
    tests = {}
    for name, root in (("highD", HIGHD), ("exiD", EXID)):
        x, y, ttlc = _windows("test", root)
        kind = np.load(root / "test.npz")["kind"] if name == "exiD" else None
        tests[name] = (((x - lo) / (hi - lo + 1e-12)).astype(np.float32), y, ttlc, kind)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for key, (h5, task) in MODELS.items():
        model = keras.models.load_model(REPO / h5, compile=False)
        for name, (x, y, ttlc, kind) in tests.items():
            p = model.predict(x, batch_size=4096, verbose=0)
            by = {}
            if task == "cls":
                pred = p.argmax(1)
                hit = pred == y
                f1 = []
                for c in range(3):
                    tp = int(((pred == c) & (y == c)).sum())
                    fp = int(((pred == c) & (y != c)).sum())
                    fn = int(((pred != c) & (y == c)).sum())
                    f1.append(2 * tp / max(2 * tp + fp + fn, 1))
                m = {"acc": float(hit.mean()), "macro_f1": float(np.mean(f1))}
                if kind is not None:
                    kw = np.repeat(kind, N_SLIDES)
                    by = {kinds[k]: {"acc": float(hit[kw == k].mean()), "scenarios": int((kind == k).sum())}
                          for k in np.unique(kind)}
            else:
                lc = y != 0
                e = p.reshape(-1)[lc] - ttlc[lc]
                m = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
                if kind is not None:
                    kw = np.repeat(kind, N_SLIDES)[lc]
                    by = {kinds[k]: {"rmse": float(np.sqrt((e[kw == k] ** 2).mean())),
                                     "mae": float(np.abs(e[kw == k]).mean()),
                                     "scenarios": int((kind == k).sum())}
                          for k in np.unique(kind) if k != 0}
            rec = {"method": "zero-shot", "model": key, "source_h5": h5, "task": task,
                   "params": int(model.count_params()), "trained_on": "highD", "eval_on": name,
                   "normalization": "minmax-train of highD", "metrics": {**m, "by_group": by} if by else m,
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{key} on {name}] {m}", flush=True)
        keras.backend.clear_session()


if __name__ == "__main__":
    main()
