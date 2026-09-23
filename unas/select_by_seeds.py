"""Choose the final model of a v2 search by its five-seed mean on validation data.

The search scores each candidate with a single training run, and the fork's own saver
kept candidates by their test error, so the earlier deployed models were chosen on
single runs and partly on test data. For the v2 searches (configs highd_*_v2, saver
unas/safe_saver.py) this script:

  1. reads every saved candidate of the search (the JSON sidecar next to each .h5);
  2. shortlists the K candidates with the best single-run validation metric;
  3. retrains each from scratch with five seeds under the search recipe (optimizer,
     callbacks and epochs of seed_variance.py with RECIPE=search), recording validation
     and test metrics for every seed;
  4. applies two rules fixed before any result was seen, both on validation data only:
       best     highest mean validation accuracy (TTLC: lowest mean validation RMSE);
       smallest fewest parameters among the candidates whose mean is within one
                standard error of the best mean.
     Test metrics are reported for every candidate but never used to choose.

Run in the WSL dmir_nas venv, from the fork:
  source ~/dmir_nas/env.sh; cd ~/uNAS
  HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/select_by_seeds.py highd_cls_v2 [K]

Output: datasets/highd/results/seeds/select_<search>.jsonl (one line per candidate and
seed; resumable) and datasets/highd/results/nas-v2/<search>/ (the shortlisted .h5 files
with their sidecars, and selection.json).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_variance as sv                                  # noqa: E402  (sets up TF, keras, adapters)

tf, keras = sv.tf, sv.keras
REPO = sv.REPO
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]


def load_candidates(search: str) -> list[dict]:
    models = Path(os.path.expanduser(f"~/uNAS/artifacts/{search}/models"))
    rows = []
    for side in sorted(models.glob("model_*.json")):
        r = json.loads(side.read_text())
        r.pop("layers_config", None)
        r["h5"] = str(side.with_suffix(".h5"))
        rows.append(r)
    return rows


def train_one(h5: str, ds, task: str, seed: int) -> dict:
    keras.utils.set_random_seed(seed)
    net = keras.models.clone_model(keras.models.load_model(h5, compile=False))
    loss_kind = "logits" if task == "cls" else "mae"
    loss, metrics = sv.compile_args(loss_kind)
    net.compile(optimizer="adam", loss=loss, metrics=metrics)
    train = ds.train_dataset().shuffle(sv.BATCH * 8, seed=seed).batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    val = ds.validation_dataset().batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    t0 = time.perf_counter()
    hist = net.fit(train, validation_data=val, epochs=50, verbose=0, callbacks=sv.callbacks("highd", loss_kind))
    out = {"epochs_run": len(hist.history["loss"]), "wall_s": round(time.perf_counter() - t0, 1)}
    for split in ("val", "test"):
        x, y = ds._data[split]
        p = net.predict(x, batch_size=4096, verbose=0)
        if task == "cls":
            pred = p.argmax(1)
            f1 = []
            for c in range(3):
                tp, fp = int(((pred == c) & (y == c)).sum()), int(((pred == c) & (y != c)).sum())
                fn = int(((pred != c) & (y == c)).sum())
                f1.append(2 * tp / max(2 * tp + fp + fn, 1))
            out[split] = {"acc": float((pred == y).mean()), "macro_f1": float(np.mean(f1))}
        else:
            e = p.reshape(-1) - y
            out[split] = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean())}
    keras.backend.clear_session()
    return out


def summarize(rows: list[dict], key: str, split: str) -> dict:
    v = [r[split][key] for r in rows]
    return {"mean": st.fmean(v), "std": st.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}


def main(search: str, k: int = 12) -> None:
    task = "cls" if "_cls" in search else "ttlc"
    key, sign = ("acc", -1) if task == "cls" else ("rmse", 1)     # sort ascending on sign * metric
    ds = sv.HighD_Dataset(task="highd_cls" if task == "cls" else "highd_ttlc")
    cands = load_candidates(search)
    if not cands:
        sys.exit(f"no sidecars under ~/uNAS/artifacts/{search}/models")
    # single-run validation metric as the search saw it: val_error = 1 - acc, or val MAE
    cands.sort(key=lambda r: (r["val_error"], r["params"]))
    short = cands[:k]
    dest = REPO / "datasets/highd/results/nas-v2" / search
    dest.mkdir(parents=True, exist_ok=True)
    for r in short:
        for suffix in (".h5", ".json"):
            src = Path(r["h5"]).with_suffix(suffix)
            shutil.copy2(src, dest / src.name)
    out = REPO / "datasets/highd/results/seeds" / f"select_{search}.jsonl"
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[(r["model"], r["seed"])] = r
    print(f"{search}: {len(cands)} saved candidates, shortlist {len(short)}, seeds {SEEDS}", flush=True)
    for r in short:
        name = Path(r["h5"]).stem
        for seed in SEEDS:
            if (name, seed) in done:
                continue
            res = train_one(r["h5"], ds, task, seed)
            rec = {"search": search, "model": name, "params": r["params"], "seed": seed,
                   "search_val_error": r["val_error"], "pmu": r["pmu"], "ms": r["ms"], "macs": r["macs"],
                   "recipe": "search recipe", **res, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            done[(name, seed)] = rec
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{name} {r['params']}] seed {seed}: val {res['val']} test {res['test']}", flush=True)

    table = []
    for r in short:
        name = Path(r["h5"]).stem
        runs = [done[(name, s)] for s in SEEDS if (name, s) in done]
        table.append({"model": name, "params": r["params"], "pmu": r["pmu"], "ms": r["ms"], "macs": r["macs"],
                      "search_val_error": r["val_error"], "val": summarize(runs, key, "val"),
                      "test": summarize(runs, key, "test")})
    table.sort(key=lambda t: sign * t["val"]["mean"])
    best = table[0]
    se = best["val"]["std"] / math.sqrt(best["val"]["n"])
    near = [t for t in table if sign * (t["val"]["mean"] - best["val"]["mean"]) <= se]
    smallest = min(near, key=lambda t: t["params"])
    sel = {"search": search, "task": task, "metric": key, "seeds": SEEDS, "shortlist": len(short),
           "saved_candidates": len(cands),
           "rules": {"best": "best mean validation metric",
                     "smallest": "fewest parameters within one standard error of the best mean validation metric"},
           "best": best["model"], "smallest": smallest["model"], "table": table,
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (dest / "selection.json").write_text(json.dumps(sel, indent=1))
    for t in table:
        tag = ("best " if t is best else "") + ("smallest" if t is smallest else "")
        print(f"{t['model']:>34} {t['params']:>7,}  val {t['val']['mean']:.4f}±{t['val']['std']:.4f}  "
              f"test {t['test']['mean']:.4f}±{t['test']['std']:.4f}  {tag}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12)
