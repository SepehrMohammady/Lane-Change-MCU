"""Deployment files for an LCIR model chosen by a v4 search (or any LCIR .h5).

Emits, next to the hand-designed DSCNN builds and in their format:
  <stem>_float32.tflite, <stem>_int8.tflite (int8 weights and activations, float32 I/O),
  <stem>_int8_io.tflite (int8 I/O) and <stem>_deploy.json with the test metrics of the
  Keras model and of each file. Inputs are clipped to the training range as in every LCIR
  evaluation (prepare_deploy.prep); int8 files are evaluated through their tensor
  interface (quantized input, dequantized output), as export_int8_io.py does.

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


def main(task, h5, out_dir, stem):
    xtr, xte, yte, is_cls = prep(task)
    model = tf.keras.models.load_model(h5, compile=False)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"task": task, "source_h5": Path(h5).name, "params": int(model.count_params()),
           "input_shape": list(xte.shape[1:]),
           "keras_float": metrics(model.predict(xte, batch_size=4096, verbose=0), yte, is_cls),
           "variants": {}}
    builds = (("float32", convert(model, xtr, "float32")),
              ("int8", convert_int8(model, xtr, int8_io=False)),
              ("int8_io", convert_int8(model, xtr, int8_io=True)))
    for name, tfl in builds:
        (out / f"{stem}_{name}.tflite").write_bytes(tfl)
        rec["variants"][name] = {"bytes": len(tfl), **metrics(predict(tfl, xte), yte, is_cls)}
        print(f"{stem}_{name}: {len(tfl):,} B  {rec['variants'][name]}", flush=True)
    (out / f"{stem}_deploy.json").write_text(json.dumps(rec, indent=1))
    print("->", out / f"{stem}_deploy.json")


if __name__ == "__main__":
    main(*sys.argv[1:5])
