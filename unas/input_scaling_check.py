"""Does per-feature input scaling remove the int8 loss on LCIR? (2026-11-05)

The LCIR inputs are raw units (train ranges 0 to 449 per feature) and an int8 input tensor has one scale,
so up to half the features span fewer than 8 int8 steps; quantizing the input alone costs the corrected
intention choice 4.8 points and the LCR regressor 0.22 s of RMSE. This script trains the three corrected
LCIR choices (seed-42 searches, rule "best") with seeds 0-4 on raw inputs, as searched, and on per-feature
min-max scaled inputs (training statistics; validation and test are clipped to the training range first,
as in every LCIR evaluation), with the LCIR search recipe of select_by_seeds.py, then converts every
trained network to full int8 with int8 I/O (500 calibration windows, rng 42, as prepare_deploy.py and
export_int8_io.py) and scores float and int8 on validation and test. The rule was fixed beforehand
(local notes, 2026-11-05).

Run in the WSL dmir_nas venv, from the fork (as select_by_seeds.py):
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/input_scaling_check.py [task ...]

Output: datasets/dmir/results/seeds/input_scaling.jsonl (one line per task, input variant and seed;
resumable). SEEDS and EPOCH_CAP as in seed_variance.py (smoke tests: EPOCH_CAP=1 SEEDS=0
OUT_NAME=input_scaling_smoke.jsonl).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_by_seeds as sel                               # noqa: E402  (sets up TF, keras, adapters)

tf, keras, sv = sel.tf, sel.keras, sel.sv
REPO = sel.REPO
NV = REPO / "datasets" / "dmir" / "results" / "nas-v2"
CHOICES = {  # task -> (search, chosen network, rule "best")
    "classification": ("dmir_cls_v4_k3_single_s42", "model_1793742305537376579_3303354"),
    "regression_lcr": ("dmir_lcr_v4_k3", "model_1793791756469877447_696071"),
    "regression_lcl": ("dmir_lcl_v4_k3", "model_1793796654473538939_1632373"),
}
OUT = REPO / "datasets" / "dmir" / "results" / "seeds" / os.environ.get("OUT_NAME", "input_scaling.jsonl")


def scaled(ds):
    """The same splits with every feature min-max scaled by the training windows' range."""
    xtr = ds._data["train"][0]
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    span = (hi - lo).astype(np.float32)
    span[span == 0] = 1.0                                    # constant features (intention: feature 7) stay 0
    for s, (x, y) in list(ds._data.items()):
        ds._data[s] = (((x - lo) / span).astype(np.float32), y)
    return ds


def convert_int8_io(model, xtr):
    """Full-integer int8 PTQ with int8 I/O, as export_int8_io.convert_int8(int8_io=True)."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    rng = np.random.default_rng(42)
    s = xtr[rng.choice(len(xtr), size=min(500, len(xtr)), replace=False)]
    conv.representative_dataset = lambda: ([s[i:i + 1]] for i in range(len(s)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def predict_int8(tfl, x):
    """Quantize the input and dequantize the output through the file's own interface."""
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    (si, zi), (so, zo) = inp["quantization"], out["quantization"]
    xq = np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)
    preds = np.empty((len(x),) + tuple(out["shape"][1:]), np.float32)
    for i in range(len(x)):
        it.set_tensor(inp["index"], xq[i:i + 1])
        it.invoke()
        preds[i] = (it.get_tensor(out["index"])[0].astype(np.float32) - zo) * so
    return preds


def metric(p, y, cls):
    if cls:
        return {"acc": float((p.argmax(1) == y).mean())}
    e = p.reshape(-1) - y.reshape(-1)
    return {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean())}


def train(h5, ds, cls, seed):
    """select_by_seeds.train_one's LCIR recipe, keeping the trained network."""
    keras.utils.set_random_seed(seed)
    net = keras.models.clone_model(keras.models.load_model(h5, compile=False))
    loss_kind = "logits" if cls else "mse"
    loss, metrics = sv.compile_args(loss_kind)
    net.compile(optimizer="adam", loss=loss, metrics=metrics)
    tr = ds.train_dataset().shuffle(sv.BATCH * 8, seed=seed).batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    va = ds.validation_dataset().batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    hist = net.fit(tr, validation_data=va, epochs=sv.EPOCH_CAP or (80 if cls else 100), verbose=0,
                   callbacks=sv.callbacks("dmir", loss_kind))
    return net, len(hist.history["loss"])


def main(tasks):
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["inputs"], r["seed"]))
    for task in tasks:
        search, model = CHOICES[task]
        h5 = str(NV / search / f"{model}.h5")
        cls = task == "classification"
        for variant in ("raw", "scaled"):
            todo = [s for s in sv.SEEDS if (task, variant, s) not in done]
            if not todo:
                continue
            ds = sv.DMIR_Dataset(task=task)
            if variant == "scaled":
                ds = scaled(ds)
            for seed in todo:
                t0 = time.perf_counter()
                net, epochs = train(h5, ds, cls, seed)
                tfl = convert_int8_io(net, ds._data["train"][0])
                rec = {"task": task, "inputs": variant, "seed": seed, "search": search, "model": model,
                       "params": int(net.count_params()), "recipe": "search recipe", "epochs_run": epochs,
                       "int8_io_bytes": len(tfl)}
                for split in ("val", "test"):
                    x, y = ds._data[split]
                    rec[f"float_{split}"] = metric(net.predict(x, batch_size=4096, verbose=0), y, cls)
                    rec[f"int8_{split}"] = metric(predict_int8(tfl, x), y, cls)
                rec["wall_s"] = round(time.perf_counter() - t0, 1)
                rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{task} {variant}] seed {seed}: float test {rec['float_test']} int8 test {rec['int8_test']} "
                      f"({epochs} ep, {rec['wall_s']} s)", flush=True)
                keras.backend.clear_session()
            del ds


if __name__ == "__main__":
    main(sys.argv[1:] or list(CHOICES))
