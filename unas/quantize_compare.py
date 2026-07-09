"""Compare quantization strategies for one model: float vs full-int8 vs int16x8.

The DMIR inputs have very different per-channel scales, which hurts full-int8
(per-tensor int8 activations). int16x8 keeps 8-bit weights but 16-bit
activations, usually recovering most of the accuracy at ~the same flash size.

  ~/dmir_nas/bin/python quantize_compare.py <task> <h5_path>
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


def prep(task):
    folder, sfx, is_cls, drop_ind = LAYOUT[task]
    root = Path(os.environ["DMIR_DATA_ROOT"]) / folder
    def rd(s, xy):
        with open(root / f"{xy}_{s}_{sfx}.pkl", "rb") as f:
            return pickle.load(f)
    xtr = rd("train", "x").astype(np.float32); xte = rd("test", "x").astype(np.float32)
    yte = rd("test", "y")
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1)); np.clip(xte, lo, hi, out=xte)
    if drop_ind:
        keep = [c for c in range(xtr.shape[2]) if c not in (28, 29)]
        xtr, xte = xtr[:, :, keep], xte[:, :, keep]
    return xtr, xte, (yte.astype(np.int64) if is_cls else yte.astype(np.float32)), is_cls


def convert(model, xtr, mode):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    rng = np.random.default_rng(42)
    s = xtr[rng.choice(len(xtr), size=min(500, len(xtr)), replace=False)]
    conv.representative_dataset = lambda: ([s[i:i + 1]] for i in range(len(s)))
    if mode == "int8":
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    elif mode == "int16x8":
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    return conv.convert()


def evaluate(tflite, xte, yte, is_cls):
    it = tf.lite.Interpreter(model_content=tflite); it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds = []
    for i in range(len(xte)):
        it.set_tensor(inp["index"], xte[i:i + 1].astype(inp["dtype"]))
        it.invoke(); preds.append(it.get_tensor(out["index"])[0].copy())
    p = np.array(preds)
    return float((p.argmax(1) == yte).mean()) if is_cls else float(np.mean(np.abs(p.squeeze(-1) - yte)))


def main(task, h5):
    xtr, xte, yte, is_cls = prep(task)
    m = tf.keras.models.load_model(h5, compile=False)
    fp = m.predict(xte, batch_size=1024, verbose=0)
    fm = float((fp.argmax(1) == yte).mean()) if is_cls else float(np.mean(np.abs(fp.squeeze(-1) - yte)))
    name = "acc" if is_cls else "MAE"
    print(f"{task} {Path(h5).stem}: float {name}={fm:.4f}")
    for mode in ("int8", "int16x8"):
        try:
            b = convert(m, xtr, mode)
            met = evaluate(b, xte, yte, is_cls)
            print(f"   {mode:8} {name}={met:.4f}  size={len(b)/1024:.1f}KB")
        except Exception as e:  # noqa: BLE001
            print(f"   {mode:8} FAILED {str(e)[:80]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
