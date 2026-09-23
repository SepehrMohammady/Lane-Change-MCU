"""Does the fork's shuffle buffer hurt training on scenario-ordered data?

The fork's trainer and our Keras retraining scripts shuffle with a buffer of
batch_size * 8 = 2,048 windows, while the prepared highD and exiD training sets are
stored scenario by scenario in recording order (exiD also by location), so a buffer
holds about 79 neighbouring scenarios and a batch can be all lane keeping or all one
junction. The PyTorch trainer permutes the whole set. This script trains one Keras
architecture with the search recipe twice per seed: on the data as stored, and on the
same data permuted once before training (the buffer then mixes the whole set), and
writes the test accuracy of both.

Run in the WSL dmir_nas venv, from the fork:
  source ~/dmir_nas/env.sh; cd ~/uNAS
  HIGHD_DATA_ROOT=... ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/shuffle_check.py <key> [seeds]
  key: a RUNS entry of seed_variance.py (dataset, h5 and task are taken from it)
Output: datasets/<dataset>/results/seeds/shuffle_check.jsonl
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_variance as sv                                  # noqa: E402  (sets up TF, keras, adapters)

tf, keras = sv.tf, sv.keras


def run(key, seeds):
    ds_name, h5, task, kw, loss_kind, _ = sv.RUNS[key]
    ds = sv.HighD_Dataset(task=task, root=sv.EXID_ROOT if ds_name == "exid" else None)
    x0, y0 = ds._data["train"]
    xte, yte = ds._data["test"]
    out = sv.REPO / f"datasets/{'exid' if ds_name == 'exid' else 'highd'}/results/seeds/shuffle_check.jsonl"
    for seed in seeds:
        for order in ("as stored", "permuted once"):
            if order == "permuted once":
                perm = np.random.default_rng(seed).permutation(len(y0))
                ds._data["train"] = (x0[perm], y0[perm])
            else:
                ds._data["train"] = (x0, y0)
            keras.utils.set_random_seed(seed)
            model = keras.models.clone_model(keras.models.load_model(sv.REPO / h5, compile=False))
            loss, metrics = sv.compile_args(loss_kind)
            model.compile(optimizer="adam", loss=loss, metrics=metrics)
            train = ds.train_dataset().shuffle(sv.BATCH * 8, seed=seed).batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
            val = ds.validation_dataset().batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
            t0 = time.perf_counter()
            hist = model.fit(train, validation_data=val, epochs=50, verbose=0, callbacks=sv.callbacks("highd", loss_kind))
            acc = float((model.predict(xte, batch_size=4096, verbose=0).argmax(1) == yte).mean())
            rec = {"key": key, "seed": seed, "order": order, "test_acc": acc,
                   "val_acc_best": float(max(hist.history["val_accuracy"])), "epochs_run": len(hist.history["loss"]),
                   "wall_s": round(time.perf_counter() - t0, 1), "recipe": "search recipe",
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{key}] seed {seed} {order}: test {acc:.4f} ({rec['epochs_run']} ep)", flush=True)
            keras.backend.clear_session()


if __name__ == "__main__":
    run(sys.argv[1], [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")])
