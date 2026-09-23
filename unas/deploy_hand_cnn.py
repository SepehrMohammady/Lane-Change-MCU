"""TFLite builds of the hand-designed DSCNN, made the same way as the searched models'.

Input: an .npz written by scripts/export_hand_cnn.py (PyTorch weights in Keras
layout plus the preprocessed windows). The network is rebuilt in Keras layer by
layer (explicit zero padding, so the strided convolutions see the same samples
as PyTorch's symmetric padding), the weights are loaded, and the Keras outputs
are compared with the PyTorch outputs stored in the bundle. The model is then
converted exactly as in unas/prepare_deploy_highd.py: float32, full-int8 PTQ
with float32 I/O, and full-int8 PTQ with int8 I/O, calibrated on the 500
windows of the bundle. Every file is scored on the full test set with an
interface-honouring evaluator (quantize in, dequantize out for int8 I/O).

Run in the WSL dmir_nas venv, CPU-safe:
  source ~/dmir_nas/env.sh
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/deploy_hand_cnn.py <bundle.npz> <out_dir> [f32-suffix]

Emits <stem>_<f32|float32>.tflite, <stem>_int8.tflite, <stem>_int8_io.tflite and
<stem>_deploy.json (parity, test metrics of the Keras model and of each file).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf   # noqa: E402
import keras              # noqa: E402

L = keras.layers


def build(z) -> keras.Model:
    T, C = z["x_test"].shape[1:]
    widths = [int(v) for v in z["widths"]]
    k, n_out = int(z["kernel"]), int(z["n_outputs"])
    inp = keras.Input((T, C), name="input")
    x = L.ZeroPadding1D(k // 2)(inp)
    x = L.Conv1D(widths[0], k, strides=2, use_bias=False, name="stem_conv")(x)
    x = L.BatchNormalization(epsilon=1e-5, name="stem_bn")(x)
    x = L.ReLU()(x)
    for i, c_out in enumerate(widths[1:]):
        x = L.ZeroPadding1D(k // 2)(x)
        x = L.DepthwiseConv1D(k, strides=2, use_bias=False, name=f"b{i}_dw")(x)
        x = L.Conv1D(c_out, 1, use_bias=False, name=f"b{i}_pw")(x)
        x = L.BatchNormalization(epsilon=1e-5, name=f"b{i}_bn")(x)
        x = L.ReLU()(x)
    x = L.GlobalAveragePooling1D()(x)
    out = L.Dense(n_out, name="dense")(x)
    m = keras.Model(inp, out, name="hand_dscnn")

    def bn(p):
        return [z[f"{p}_gamma"], z[f"{p}_beta"], z[f"{p}_mean"], z[f"{p}_var"]]
    m.get_layer("stem_conv").set_weights([z["stem_conv"]])
    m.get_layer("stem_bn").set_weights(bn("stem_bn"))
    for i in range(len(widths) - 1):
        m.get_layer(f"b{i}_dw").set_weights([z[f"b{i}_dw"]])
        m.get_layer(f"b{i}_pw").set_weights([z[f"b{i}_pw"]])
        m.get_layer(f"b{i}_bn").set_weights(bn(f"b{i}_bn"))
    m.get_layer("dense").set_weights([z["dense_kernel"], z["dense_bias"]])
    return m


def convert(model, calib, mode, int8_io=False):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: ([calib[i:i + 1]] for i in range(len(calib)))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if int8_io:
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(tfl, x):
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    preds = []
    for i in range(len(x)):
        xi = x[i:i + 1]
        if inp["dtype"] == np.int8:
            xi = np.clip(np.round(xi / in_scale + in_zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], xi)
        it.invoke()
        p = it.get_tensor(out["index"])[0].astype(np.float32).copy()
        if out["dtype"] == np.int8:
            p = (p - out_zp) * out_scale
        preds.append(p)
    return np.array(preds)


def metrics(z, p):
    task = str(z["task"])
    if task == "cls":
        y = z["y_test"].astype(np.int64)
        return {"acc": round(float((p.argmax(1) == y).mean()), 5)}
    if task == "ttlc":                       # highD: windows of lane-change scenarios only,
        keep = z["y_test"] != 0              # as prepare_deploy_highd.py (some lane-keeping
        t = z["ttlc_test"]                   # scenarios carry a crossing index, so ttlc is not NaN)
        e = p.reshape(-1)[keep] - t[keep]
    else:                                    # LCIR regression: every window
        e = p.reshape(-1) - z["y_test"].reshape(-1)
    return {"rmse": round(float(np.sqrt((e ** 2).mean())), 5), "mae": round(float(np.abs(e).mean()), 5)}


def main(bundle, out_dir, f32_suffix="f32"):
    z = np.load(bundle)
    stem = Path(bundle).stem
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = build(z)
    parity = float(np.abs(model.predict(z["x_ref"], verbose=0) - z["y_ref"]).max())
    print(f"{stem}: Keras vs PyTorch max |diff| = {parity:.2e}")
    if parity > 1e-3:
        sys.exit(f"parity check failed for {stem}")
    rec = {"task": str(z["task"]), "source": str(z["source"]), "params": int(z["params"]),
           "input_shape": list(z["x_test"].shape[1:]), "parity_max_abs_diff": parity,
           "keras_float": metrics(z, model.predict(z["x_test"], batch_size=4096, verbose=0)),
           "variants": {}}
    calib = z["x_calib"].astype(np.float32)
    x = z["x_test"].astype(np.float32)
    for name, mode, io in ((f32_suffix, "f32", False), ("int8", "int8", False), ("int8_io", "int8", True)):
        tfl = convert(model, calib, mode, io)
        (out / f"{stem}_{name}.tflite").write_bytes(tfl)
        rec["variants"][name] = {"bytes": len(tfl), **metrics(z, run_tflite(tfl, x))}
        print(f"  {name}: {len(tfl)} B, {rec['variants'][name]}")
    (out / f"{stem}_deploy.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
