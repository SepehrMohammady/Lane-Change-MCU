"""INT8 post-training quantization + honest accuracy check for chosen models.

Converts a saved Keras .h5 to an INT8 TFLite model (weights + activations),
calibrated on real DMIR train windows, then evaluates BOTH the float model and
the INT8 TFLite model on the test set so we can report the quantization drop and
the real .tflite byte size. A second uint8-I/O TFLite is also written for the ST
Edge AI toolchain (which prefers integer tensor interfaces).

Run in the WSL dmir_nas venv:
  source ~/dmir_nas/env.sh
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/data \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/quantize_eval.py \
    <task> <h5_path> <out_dir>
task in {regression_lcr, regression_lcl, classification, classification_noind}
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

LAYOUT = {
    "regression_lcr": ("data-prepared-lcr", "lcr", False, False),
    "regression_lcl": ("data-regression-lcl", "lcl", False, False),
    "classification": ("data-classification", "multi", True, False),
    "classification_noind": ("data-classification", "multi", True, True),
}


def load(task, split, xy):
    folder, sfx, _, _ = LAYOUT[task]
    with open(Path(os.environ["DMIR_DATA_ROOT"]) / folder / f"{xy}_{split}_{sfx}.pkl", "rb") as f:
        return pickle.load(f)


def prep(task):
    folder, sfx, is_cls, drop_ind = LAYOUT[task]
    xtr = load(task, "train", "x").astype(np.float32)
    xte = load(task, "test", "x").astype(np.float32)
    yte = load(task, "test", "y")
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    np.clip(xte, lo, hi, out=xte)
    if drop_ind:
        idx = [28, 29]  # classification indicator channels
        keep = [c for c in range(xtr.shape[2]) if c not in idx]
        xtr, xte = xtr[:, :, keep], xte[:, :, keep]
    return xtr, xte, (yte.astype(np.int64) if is_cls else yte.astype(np.float32)), is_cls


def to_tflite(model, xtr, mode="int16x8"):
    """mode: 'int16x8' (int8 weights + int16 activations, our deployment
    default — preserves accuracy on the wide-dynamic-range DMIR inputs) or
    'int8' (full int8, kept for the honesty comparison; degrades badly here)."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    rng = np.random.default_rng(42)
    sample = xtr[rng.choice(len(xtr), size=min(500, len(xtr)), replace=False)]
    conv.representative_dataset = lambda: ([sample[i:i + 1]] for i in range(len(sample)))
    if mode == "int8":
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    else:
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    return conv.convert()


def eval_tflite(tflite_bytes, xte, yte, is_cls):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = []
    for i in range(len(xte)):
        x = xte[i:i + 1].astype(inp["dtype"]) if inp["dtype"] != np.float32 else xte[i:i + 1]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        preds.append(interp.get_tensor(out["index"])[0].copy())
    p = np.array(preds)
    if is_cls:
        return float((p.argmax(1) == yte).mean())
    return float(np.mean(np.abs(p.squeeze(-1) - yte)))


def main(task, h5_path, out_dir):
    xtr, xte, yte, is_cls = prep(task)
    model = tf.keras.models.load_model(h5_path, compile=False)
    fp = model.predict(xte, batch_size=1024, verbose=0)
    float_metric = float((fp.argmax(1) == yte).mean()) if is_cls \
        else float(np.mean(np.abs(fp.squeeze(-1) - yte)))

    tfl = to_tflite(model, xtr, mode="int16x8")
    q_metric = eval_tflite(tfl, xte, yte, is_cls)

    name = Path(h5_path).stem
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{task}_{name}_int16x8.tflite").write_bytes(tfl)

    mname = "acc" if is_cls else "MAE"
    drop = (float_metric - q_metric) if is_cls else (q_metric - float_metric)
    print(f"{task:22} {name:16} params={model.count_params():>7} "
          f"tflite={len(tfl)/1024:6.1f}KB  "
          f"float_{mname}={float_metric:.4f}  int16x8_{mname}={q_metric:.4f}  "
          f"drop={drop:+.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
