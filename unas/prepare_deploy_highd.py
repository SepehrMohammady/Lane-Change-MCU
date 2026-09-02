"""Deployment artifacts for highD search winners: full-int8 PTQ TFLite in both
tensor interfaces (float32 I/O and int8 I/O — the DMIR campaign showed the
interface change is accuracy-neutral and saves RAM + on-device casts).

Every artifact is evaluated on the real test windows with an
interface-honouring evaluator (quantize-in / dequantize-out for int8 I/O;
a plain cast would destroy the input — the eval_tflite trap documented in
export_int8_io.py).

Run in the WSL dmir_nas venv, CPU-safe:
  source ~/dmir_nas/env.sh
  CUDA_VISIBLE_DEVICES= HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/prepare_deploy_highd.py \
    <task: highd_cls|highd_ttlc> <model.h5> <out_dir>

Emits <stem>_f32.tflite, <stem>_int8.tflite, <stem>_int8_io.tflite +
<stem>_deploy.json with test metrics for all three.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import tensorflow as tf   # noqa: E402
import keras              # noqa: E402

IN_LEN, SEQ_LEN, FPS = 10, 35, 5
N_SLIDES = SEQ_LEN - IN_LEN + 1
DATA_ROOT = Path(os.environ["HIGHD_DATA_ROOT"])


def windows(split):
    z = np.load(DATA_ROOT / f"{split}.npz")
    feats, label, cross = z["feats"], z["label"], z["cross_idx"]
    S = len(feats)
    sw = np.lib.stride_tricks.sliding_window_view(feats, IN_LEN, axis=1)
    x = np.ascontiguousarray(sw.transpose(0, 1, 3, 2)).reshape(
        S * N_SLIDES, IN_LEN, feats.shape[2]).astype(np.float32)
    y = np.repeat(label, N_SLIDES)
    s_idx = np.tile(np.arange(N_SLIDES), S)
    cx = np.repeat(cross.astype(np.float32), N_SLIDES)
    ttlc = (cx - (s_idx + IN_LEN) + 1) / FPS
    ttlc[cx < 0] = np.nan
    return x, y, ttlc


def convert(model, xtr, mode, int8_io=False):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        rng = np.random.default_rng(42)
        s = xtr[rng.choice(len(xtr), size=min(500, len(xtr)), replace=False)]
        conv.representative_dataset = lambda: ([s[i:i + 1]] for i in range(len(s)))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if int8_io:
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
    return conv.convert()


def eval_tflite(tfl, x, y, is_cls):
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    int_in = inp["dtype"] == np.int8
    int_out = out["dtype"] == np.int8
    preds = []
    for i in range(len(x)):
        xi = x[i:i + 1]
        if int_in:
            xi = np.clip(np.round(xi / in_scale + in_zp), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], xi)
        it.invoke()
        p = it.get_tensor(out["index"])[0].astype(np.float32).copy()
        if int_out:
            p = (p - out_zp) * out_scale
        preds.append(p)
    p = np.array(preds)
    if is_cls:
        return {"acc": round(float((p.argmax(1) == y).mean()), 5)}
    e = p.squeeze(-1) - y
    return {"mae": round(float(np.abs(e).mean()), 5),
            "rmse": round(float(np.sqrt((e ** 2).mean())), 5)}


def main(task, h5_path, out_dir):
    is_cls = task == "highd_cls"
    xtr, _, _ = windows("train")
    lo = xtr.reshape(-1, 18).min(0); hi = xtr.reshape(-1, 18).max(0)
    xtr_n = ((xtr - lo) / (hi - lo + 1e-12)).astype(np.float32)
    xte, yte, tte = windows("test")
    xte_n = ((xte - lo) / (hi - lo + 1e-12)).astype(np.float32)
    if is_cls:
        ye = yte.astype(np.int64)
    else:
        m = yte != 0
        xte_n, ye = xte_n[m], tte[m].astype(np.float32)

    model = keras.models.load_model(h5_path, compile=False)
    stem = f"{task}_{Path(h5_path).stem}"
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    res = {"task": task, "source_h5": Path(h5_path).name,
           "params": int(model.count_params()), "variants": {}}

    for tag, mode, io8 in (("f32", "f32", False),
                           ("int8", "int8", False),
                           ("int8_io", "int8", True)):
        tfl = convert(model, xtr_n, mode, io8)
        path = out / f"{stem}_{tag}.tflite"
        path.write_bytes(tfl)
        met = eval_tflite(tfl, xte_n, ye, is_cls)
        res["variants"][tag] = {"bytes": len(tfl), **met}
        print(f"{stem}_{tag}: {len(tfl):>8,} B  {met}", flush=True)

    (out / f"{stem}_deploy.json").write_text(json.dumps(res, indent=1))
    print(f"-> {out / (stem + '_deploy.json')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
