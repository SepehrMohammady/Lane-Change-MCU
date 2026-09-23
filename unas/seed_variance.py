"""Seed variance of the final models: is a result a property of the architecture,
or of one lucky initialisation?

Every headline number in the project comes from a single training run. Here each
final architecture is re-initialised and retrained from scratch with several
seeds, using exactly the recipe the search trained it with (the fork's
ModelTrainer, reproduced below: same dataset adapter and clipping/normalisation,
same shuffle buffer, batch size, optimizer, loss, callbacks and epoch cap), and
scored on the test split with metrics computed here in numpy.

REF_cnn_multi is the colleague's reference CNN. Its original training recipe is
unknown, so its row is a *controlled* retrain under our recipe: it removes the
recipe as a confound from the searched-vs-reference comparison, and is labelled
as such everywhere it is reported.

RECIPE=final retrains instead with the recipe of the hand-built highD baseline
(datasets/highd/train_highd.py: AdamW lr 3e-3, weight decay 1e-4, batch 256, at
most 60 epochs, early stopping on the validation metric with patience 8 and the
best weights restored). Searched and hand-built models then differ only in the
architecture, which is the comparison the paper needs. Output goes to
seed_variance_final.jsonl.

RECIPE=dscnn does the same for DMIR against the hand-built DSCNN
(src/train.py via scripts/run_baseline.py): AdamW lr 3e-3, weight decay 1e-4,
per-epoch cosine decay over 60 epochs, batch 256, early stopping with patience 10
on validation accuracy (classifiers) or validation RMSE (regressors) with the best
weights restored, and MSE loss for every regressor. Output goes to
seed_variance_dscnn.jsonl.

The exid_* runs train the architectures found by the highD searches on exiD
(same scenario format, datasets/exid/prepare_exid.py), with the highD search
callbacks and epoch cap; they have no single run of their own to compare with.

Resume-safe: (model, seed) pairs already in the output file are skipped.

Run in the WSL dmir_nas venv:
  source ~/dmir_nas/env.sh; cd ~/uNAS
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data \
  HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/seed_variance.py [only-key ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices("GPU"):          # leave VRAM for the torch baselines
    tf.config.experimental.set_memory_growth(g, True)

import keras  # noqa: E402  (Keras 3, the one the fork's models are built with)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "unas"))                   # our dataset adapters
sys.path.insert(0, os.path.expanduser("~/uNAS"))         # the fork's Dataset base class
from dmir_dataset import DMIR_Dataset                    # noqa: E402
from highd_dataset import HighD_Dataset                  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
EPOCH_CAP = int(os.environ.get("EPOCH_CAP", "0"))      # smoke tests only
RECIPE = os.environ.get("RECIPE", "search")              # "search", "final" or "dscnn"
OUT_NAME = os.environ.get("OUT_NAME", {"search": "seed_variance.jsonl", "final": "seed_variance_final.jsonl",
                                      "dscnn": "seed_variance_dscnn.jsonl"}[RECIPE])
RECIPE_LABEL = {"final": "baseline recipe (AdamW 3e-3, wd 1e-4, <=60 ep, patience 8)",
                "dscnn": "DSCNN recipe (AdamW 3e-3, wd 1e-4, cosine, <=60 ep, patience 10, MSE)"}
BATCH = 256

# key: (dataset, h5 path, task, adapter kwargs, loss kind, reference metric from the original run)
RUNS = {
    "highd_cls_aaaaap":  ("highd", "datasets/highd/results/models/highd_cls_tight_model_aaaaap.h5",
                          "highd_cls", {}, "logits", {"acc": 0.91153}),
    "highd_ttlc_aaaaaw": ("highd", "datasets/highd/results/models/highd_ttlc_model_aaaaaw.h5",
                          "highd_ttlc", {}, "mae", {"mae": 0.16561, "rmse": 0.26076}),
    "highd_cls_aaaaam":  ("highd", "datasets/highd/results/models/highd_cls_model_aaaaam.h5",
                          "highd_cls", {}, "logits", {"acc": 0.90526}),
    "dmir_cls_best":     ("dmir", "datasets/dmir/results/deploy/cls_best.h5",
                          "classification", {}, "logits", {"acc": 0.9208}),
    "dmir_cls_tiny":     ("dmir", "datasets/dmir/results/deploy/cls_tiny.h5",
                          "classification", {}, "logits", {"acc": 0.9130}),
    "dmir_lcr_best":     ("dmir", "datasets/dmir/results/deploy/lcr_best.h5",
                          "regression_lcr", {}, "mae", {"mae": 0.2865, "rmse": 0.4466}),
    "dmir_lcl_best":     ("dmir", "datasets/dmir/results/deploy/lcl_best.h5",
                          "regression_lcl", {}, "mse", {"mae": 0.3165, "rmse": 0.4660}),
    "dmir_cls_noind":    ("dmir", "datasets/dmir/results/deploy/cls_noind.h5",
                          "classification", {"drop_indicators": True}, "logits", {}),
    "dmir_ref_cnn":      ("dmir", "datasets/dmir/results/deploy/REF_cnn_multi.h5",
                          "classification", {}, "softmax", {"acc": 0.9169}),
    # architectures from the highD searches, trained on exiD
    "exid_cls_aaaaap":   ("exid", "datasets/highd/results/models/highd_cls_tight_model_aaaaap.h5",
                          "highd_cls", {}, "logits", {}),
    "exid_cls_aaaaam":   ("exid", "datasets/highd/results/models/highd_cls_model_aaaaam.h5",
                          "highd_cls", {}, "logits", {}),
    "exid_ttlc_aaaaaw":  ("exid", "datasets/highd/results/models/highd_ttlc_model_aaaaaw.h5",
                          "highd_ttlc", {}, "mae", {}),
    # v2 search (global-average-pooling space, five-seed validation choice, unas/select_by_seeds.py);
    # its search-recipe seeds are in datasets/highd/results/seeds/select_highd_cls_v2.jsonl
    "highd_cls_v2_best": ("highd", "datasets/highd/results/nas-v2/highd_cls_v2/model_1790187214617755015_765300.h5",
                          "highd_cls", {}, "logits", {"acc": 0.9046}),
    "exid_cls_v2_best":  ("exid", "datasets/highd/results/nas-v2/highd_cls_v2/model_1790187214617755015_765300.h5",
                          "highd_cls", {}, "logits", {}),
    # the hand-designed CNN's layer sequence built in the v2 space (unas/build_hand_like.py):
    # same architecture as the hand-designed CNN, trained by the search's recipe
    "highd_cls_hand_in_v2": ("highd", "datasets/highd/results/nas-v2/hand_like_in_v2_space.h5",
                             "highd_cls", {}, "logits", {}),
    "exid_cls_hand_in_v2":  ("exid", "datasets/highd/results/nas-v2/hand_like_in_v2_space.h5",
                             "highd_cls", {}, "logits", {}),
}
# Runs chosen at run time (for example a search's chosen model): SV_EXTRA_JSON names a JSON
# file {key: [dataset, h5, task, adapter kwargs, loss kind, reference metric]}.
if os.environ.get("SV_EXTRA_JSON"):
    RUNS.update({k: tuple(v) for k, v in json.loads(Path(os.environ["SV_EXTRA_JSON"]).read_text()).items()})
EXID_ROOT = REPO / "datasets" / "exid" / "data" / "prepared"


def callbacks(ds_name, loss_kind):
    """The search configs' callbacks (unas/dmir_config.py, unas/highd_config.py)."""
    cls = loss_kind in ("logits", "softmax")
    if ds_name == "dmir":
        if cls:
            return [keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=12,
                                                  restore_best_weights=True),
                    keras.callbacks.TerminateOnNaN()]
        mon = "val_rmse" if loss_kind == "mse" else "val_mae"
        return [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5,
                                                  patience=15, min_lr=1e-6),
                keras.callbacks.EarlyStopping(monitor=mon, mode="min", patience=20, min_delta=0.005,
                                              restore_best_weights=True),
                keras.callbacks.TerminateOnNaN()]
    if cls:
        return [keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8,
                                              restore_best_weights=True),
                keras.callbacks.TerminateOnNaN()]
    return [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5,
                                              patience=6, min_lr=1e-6),
            keras.callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=10, min_delta=0.002,
                                          restore_best_weights=True),
            keras.callbacks.TerminateOnNaN()]


def final_callbacks(loss_kind):
    """Baseline recipe (train_highd.py): early stopping on the validation metric, patience 8, best restored."""
    cls = loss_kind in ("logits", "softmax")
    mon, mode = ("val_accuracy", "max") if cls else ("val_mae", "min")
    return [keras.callbacks.EarlyStopping(monitor=mon, mode=mode, patience=8, restore_best_weights=True),
            keras.callbacks.TerminateOnNaN()]


def dscnn_callbacks(loss_kind):
    """The DMIR DSCNN recipe (src/train.py): cosine per epoch, patience 10 on val accuracy / val RMSE."""
    cls = loss_kind in ("logits", "softmax")
    mon, mode = ("val_accuracy", "max") if cls else ("val_rmse", "min")
    cosine = keras.callbacks.LearningRateScheduler(lambda epoch: 3e-3 * 0.5 * (1 + np.cos(np.pi * epoch / 60)))
    return [cosine, keras.callbacks.EarlyStopping(monitor=mon, mode=mode, patience=10, restore_best_weights=True),
            keras.callbacks.TerminateOnNaN()]


def compile_args(loss_kind):
    """The fork's ModelTrainer loss/metric choice (uNAS/model_trainer.py)."""
    if loss_kind == "logits":
        return (keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                [keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    if loss_kind == "softmax":
        return (keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                [keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    if loss_kind == "mse":
        return keras.losses.MeanSquaredError(), [keras.metrics.RootMeanSquaredError(name="rmse")]
    return keras.losses.MeanAbsoluteError(), [keras.metrics.MeanAbsoluteError(name="mae")]


def main(only):
    out = {"dmir": REPO / "datasets/dmir/results/seeds" / OUT_NAME,
           "highd": REPO / "datasets/highd/results/seeds" / OUT_NAME,
           "exid": REPO / "datasets/exid/results/seeds" / OUT_NAME}
    done = set()
    for f in out.values():
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            done |= {(r["key"], r["seed"]) for r in map(json.loads, f.read_text().splitlines())}

    cache = {}
    for key, (ds_name, h5, task, kw, loss_kind, orig) in RUNS.items():
        if only and key not in only:
            continue
        todo = [s for s in SEEDS if (key, s) not in done]
        if not todo:
            print(f"[{key}] all seeds done", flush=True)
            continue
        ck = (ds_name, task, tuple(sorted(kw.items())))
        if ck not in cache:
            cache[ck] = (DMIR_Dataset(task=task, **kw) if ds_name == "dmir"
                         else HighD_Dataset(task=task, root=EXID_ROOT if ds_name == "exid" else None))
        ds = cache[ck]
        xte, yte = ds._data["test"]
        is_cls = loss_kind in ("logits", "softmax")

        for seed in todo:
            keras.utils.set_random_seed(seed)
            template = keras.models.load_model(REPO / h5, compile=False)
            model = keras.models.clone_model(template)          # same graph, fresh initialisation
            del template
            if RECIPE == "dscnn" and loss_kind in ("mae", "mse"):
                loss_kind = "mse"                            # the DSCNN trains every regressor on MSE
            loss, metrics = compile_args(loss_kind)
            if RECIPE == "dscnn":
                opt = keras.optimizers.AdamW(learning_rate=3e-3, weight_decay=1e-4)
            elif RECIPE == "final":
                if loss_kind == "mse":                       # final recipe monitors MAE for regression
                    metrics = [keras.metrics.MeanAbsoluteError(name="mae")]
                opt = keras.optimizers.AdamW(learning_rate=3e-3, weight_decay=1e-4)
            else:
                opt = "adam"
            model.compile(optimizer=opt, loss=loss, metrics=metrics)
            train = ds.train_dataset().shuffle(BATCH * 8, seed=seed).batch(BATCH).prefetch(tf.data.AUTOTUNE)
            val = ds.validation_dataset().batch(BATCH).prefetch(tf.data.AUTOTUNE)
            if RECIPE == "dscnn":
                epochs, cbs = EPOCH_CAP or 60, dscnn_callbacks(loss_kind)
            elif RECIPE == "final":
                epochs, cbs = EPOCH_CAP or 60, final_callbacks(loss_kind)
            else:
                epochs, cbs = EPOCH_CAP or (100 if ds_name == "dmir" else 50), callbacks(ds_name, loss_kind)   # DMIR searches ran with DMIR_EPOCHS=100 (unas/run_chunked.sh)
            t0 = time.perf_counter()
            hist = model.fit(train, validation_data=val, epochs=epochs, verbose=0, callbacks=cbs)
            wall = time.perf_counter() - t0
            p = model.predict(xte, batch_size=2048, verbose=0)
            if is_cls:
                pred = p.argmax(1)
                metrics_out = {"acc": float((pred == yte).mean())}
                f1 = []
                for c in range(3):
                    tp = int(((pred == c) & (yte == c)).sum())
                    fp = int(((pred == c) & (yte != c)).sum())
                    fn = int(((pred != c) & (yte == c)).sum())
                    f1.append(2 * tp / max(2 * tp + fp + fn, 1))
                metrics_out["macro_f1"] = float(np.mean(f1))
            else:
                e = p.reshape(-1) - yte
                metrics_out = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
            rec = {"key": key, "dataset": ds_name, "task": task, "source_h5": h5,
                   "params": int(model.count_params()), "seed": seed,
                   "recipe": (("search recipe (controlled retrain)" if key == "dmir_ref_cnn" else "search recipe")
                              if RECIPE == "search" else RECIPE_LABEL[RECIPE]),
                   "epochs_run": len(hist.history["loss"]), "wall_s": round(wall, 1),
                   "test": metrics_out, "original_run": orig,
                   "train_order": getattr(ds, "train_order", "as stored"),
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(out[ds_name], "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{key}] seed {seed}: {metrics_out}  ({rec['epochs_run']} ep, {wall:.0f} s)", flush=True)
            keras.backend.clear_session()


if __name__ == "__main__":
    main(set(sys.argv[1:]))
