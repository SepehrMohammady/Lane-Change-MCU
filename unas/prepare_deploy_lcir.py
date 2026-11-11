"""Deployment files for an LCIR model chosen by a v4 search (or any LCIR .h5).

Emits, next to the hand-designed DSCNN builds and in their format:
  <stem>_float32.tflite, <stem>_int8.tflite (int8 weights and activations, float32 I/O),
  <stem>_int8_io.tflite (int8 I/O) and <stem>_deploy.json with the test metrics of the
  Keras model and of each file. Inputs are clipped to the training range as in every LCIR
  evaluation (prepare_deploy.prep); int8 files are evaluated through their tensor
  interface (quantized input, dequantized output), as export_int8_io.py does.

A model from a search on scaled inputs (folder name with "_mm", DMIR_SCALE=minmax) gets the
same per-feature min-max scaling as in its search (training windows' range after clipping,
constant features 0) for calibration and evaluation; the deploy record then carries the
scaling's offsets and spans, which the board applies before quantizing the input. A
DMIR_SCALE value that contradicts the folder name is refused.

CALIB_CLIP=<B> (2026-11-11, unas/input_clip_check.py) clips the calibration windows of the int8 builds to
[-B, B]: the LCIR features are standardized with rare extremes of several hundred, and TFLite takes the
int8 input range from the calibration windows' minimum and maximum. The network and the float32 build do
not change; the int8 files are scored on the test windows as they are, through the input quantization,
which saturates (as the board's quantizer does). The deploy record carries the bound.

Run in the WSL dmir_nas venv, CPU:
  source ~/dmir_nas/env.sh
  CUDA_VISIBLE_DEVICES= DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/prepare_deploy_lcir.py \
    <classification|regression_lcr|regression_lcl> <model.h5> <out_dir> <stem>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_int8_io import convert_int8   # noqa: E402
from prepare_deploy import convert, prep   # noqa: E402


def predict(tflite_bytes, x):
    """Predictions of a TFLite file, quantizing the input and dequantizing the output
    when the interface is integer."""
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    int_in = inp["dtype"] in (np.int8, np.uint8)
    int_out = out["dtype"] in (np.int8, np.uint8)
    preds = []
    for i in range(len(x)):
        xi = x[i:i + 1]
        if int_in:
            info = np.iinfo(inp["dtype"])
            xi = np.clip(np.round(xi / in_scale + in_zp), info.min, info.max).astype(inp["dtype"])
        it.set_tensor(inp["index"], xi)
        it.invoke()
        y = it.get_tensor(out["index"])[0].astype(np.float32).copy()
        preds.append((y - out_zp) * out_scale if int_out else y)
    return np.array(preds)


def metrics(p, y, is_cls):
    if is_cls:
        return {"acc": round(float((p.argmax(1) == y).mean()), 5)}
    e = p.reshape(-1) - y.reshape(-1)
    return {"rmse": round(float(np.sqrt((e ** 2).mean())), 5), "mae": round(float(np.abs(e).mean()), 5)}


def input_scaling(h5):
    """'minmax' for a model from a search on scaled inputs (*_mm* folder), else '', as
    select_by_seeds.set_scaling decides for the searches."""
    want = "minmax" if "_mm" in Path(h5).resolve().parent.name else ""
    have = os.environ.get("DMIR_SCALE", "")
    if have and have != want:
        sys.exit(f"DMIR_SCALE={have!r} contradicts the folder of {h5}")
    return want


def scale_minmax(xtr, xte):
    """Per-feature min-max scaling by the training windows' range, as DMIR_SCALE=minmax in
    dmir_dataset.py (test already clipped to that range; constant features become 0)."""
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    span = (hi - lo).astype(np.float32)
    span[span == 0] = 1.0
    return ((xtr - lo) / span).astype(np.float32), ((xte - lo) / span).astype(np.float32), lo, span


def main(task, h5, out_dir, stem):
    xtr, xte, yte, is_cls = prep(task)
    scaling = input_scaling(h5)
    if scaling == "minmax":
        xtr, xte, lo, span = scale_minmax(xtr, xte)
        print(f"{task}: inputs min-max scaled per feature (training range)", flush=True)
    bound = float(os.environ["CALIB_CLIP"]) if os.environ.get("CALIB_CLIP") else None
    if bound is not None and scaling:
        sys.exit("CALIB_CLIP is for raw-input networks (standardized features), not for scaled inputs")
    xcal = xtr if bound is None else np.clip(xtr, -bound, bound)
    model = tf.keras.models.load_model(h5, compile=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"task": task, "source_h5": Path(h5).name, "params": int(model.count_params()),
           "input_shape": list(xte.shape[1:]), "input_scaling": scaling or "raw",
           **({"calibration_clip": bound} if bound is not None else {}),
           **({"scale_lo": [float(v) for v in lo], "scale_span": [float(v) for v in span]}
              if scaling == "minmax" else {}),
           "keras_float": metrics(model.predict(xte, batch_size=4096, verbose=0), yte, is_cls),
           "variants": {}}
    builds = (("float32", convert(model, xtr, "float32")),
              ("int8", convert_int8(model, xcal, int8_io=False)),
              ("int8_io", convert_int8(model, xcal, int8_io=True)))
    for name, tfl in builds:
        (out / f"{stem}_{name}.tflite").write_bytes(tfl)
        rec["variants"][name] = {"bytes": len(tfl), **metrics(predict(tfl, xte), yte, is_cls)}
        print(f"{stem}_{name}: {len(tfl):,} B  {rec['variants'][name]}", flush=True)
    (out / f"{stem}_deploy.json").write_text(json.dumps(rec, indent=1))
    print("->", out / f"{stem}_deploy.json")


if __name__ == "__main__":
    main(*sys.argv[1:5])
