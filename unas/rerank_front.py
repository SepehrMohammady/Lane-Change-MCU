"""Re-rank a saved search front by seed-averaged accuracy.

The search scores every candidate with one training run, so the model it keeps as
the winner is partly the one whose single run happened to go well (on highD the
5,347-parameter winner reported 91.15% but averages 88.9% over five seeds). This
script takes every front model above an accuracy floor, retrains each from scratch
with several seeds under the search's own recipe (reusing seed_variance.py), and
writes one record per (model, seed). The front is then re-ranked by the mean.

Run in the WSL dmir_nas venv:
  source ~/dmir_nas/env.sh; cd ~/uNAS
  HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/rerank_front.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_variance as sv                                  # noqa: E402  (sets up TF, keras, adapters)

tf, keras = sv.tf, sv.keras
REPO = sv.REPO
FLOOR = float(os.environ.get("RERANK_FLOOR", "0.895"))       # the 17-model re-rank used 0.885
CEIL = float(os.environ.get("RERANK_CEIL", "inf"))          # upper limit, to split a re-rank over two runs
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
FRONTS = {   # front csv -> directory of that search's saved models (fork artifacts)
    "highd_cls": ("datasets/highd/results/nas-fronts/highd_cls.csv", "~/uNAS/artifacts/highd_cls/models"),
    "highd_cls_tight": ("datasets/highd/results/nas-fronts/highd_cls_tight.csv", "~/uNAS/artifacts/highd_cls_tight/models"),
}
OUT = REPO / os.environ.get("RERANK_OUT", "datasets/highd/results/seeds/rerank_cls.jsonl")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(r["search"], r["model"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())}
    ds = sv.HighD_Dataset(task="highd_cls")
    xte, yte = ds._data["test"]
    todo = []
    for search, (csv_path, model_dir) in FRONTS.items():
        for r in csv.DictReader(open(REPO / csv_path)):
            if FLOOR <= float(r["test_acc"]) < CEIL:
                todo.append((search, r["model"].replace(".h5", ""), int(r["params"]), float(r["test_acc"]),
                             Path(os.path.expanduser(model_dir)) / (r["model"].replace(".h5", "") + ".h5")))
    todo.sort(key=lambda t: t[2])
    print(f"{len(todo)} front models at or above {FLOOR:.3f}; {len(SEEDS)} seeds each", flush=True)
    for search, model, params, reported, h5 in todo:
        for seed in SEEDS:
            if (search, model, seed) in done:
                continue
            keras.utils.set_random_seed(seed)
            net = keras.models.clone_model(keras.models.load_model(h5, compile=False))
            loss, metrics = sv.compile_args("logits")
            net.compile(optimizer="adam", loss=loss, metrics=metrics)
            train = ds.train_dataset().shuffle(sv.BATCH * 8, seed=seed).batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
            val = ds.validation_dataset().batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
            t0 = time.perf_counter()
            hist = net.fit(train, validation_data=val, epochs=50, verbose=0, callbacks=sv.callbacks("highd", "logits"))
            acc = float((net.predict(xte, batch_size=2048, verbose=0).argmax(1) == yte).mean())
            rec = {"search": search, "model": model, "params": params, "seed": seed, "reported_acc": reported,
                   "test_acc": acc, "epochs_run": len(hist.history["loss"]), "wall_s": round(time.perf_counter() - t0, 1),
                   "recipe": "search recipe", "train_order": getattr(ds, "train_order", "as stored"),
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{search}/{model} {params}] seed {seed}: {acc:.4f} (reported {reported:.4f})", flush=True)
            keras.backend.clear_session()


if __name__ == "__main__":
    main()
