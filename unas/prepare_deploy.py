"""Prepare ST-Edge-AI-compatible deployment artifacts + honest accuracy table.

ST Edge AI Core / Developer Cloud accept only float32, int8, uint8 tensors —
int16x8 is NOT supported (and on Cortex-M is silently dequantized to float32,
which would yield misleading benchmark numbers). See docs/research/deployment.md.

So for each chosen model we emit:
  * <name>_float32.tflite  — accuracy-exact, matches the baseline's FP32
    deployment methodology (the fair like-for-like comparison);
  * <name>_int8.tflite     — full-integer int8 PTQ (ST's recommended scheme),
    with its honest accuracy cost measured here.
The .h5 is also copied so it can be uploaded directly as a Keras float model.

Run in the WSL dmir_nas venv.
"""
import os
import pickle
import shutil
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

    xtr = rd("train", "x").astype(np.float32)
    xte = rd("test", "x").astype(np.float32)
    yte = rd("test", "y")
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    np.clip(xte, lo, hi, out=xte)
    if drop_ind:
        keep = [c for c in range(xtr.shape[2]) if c not in (28, 29)]
        xtr, xte = xtr[:, :, keep], xte[:, :, keep]
    return xtr, xte, (yte.astype(np.int64) if is_cls else yte.astype(np.float32)), is_cls


def convert(model, xtr, mode):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        rng = np.random.default_rng(42)
        s = xtr[rng.choice(len(xtr), size=min(500, len(xtr)), replace=False)]
        conv.representative_dataset = lambda: ([s[i:i + 1]] for i in range(len(s)))
        # ST's privileged scheme for Cortex-M: full-integer int8, per-channel.
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return conv.convert()


def eval_tflite(b, xte, yte, is_cls):
    it = tf.lite.Interpreter(model_content=b)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds = []
    for i in range(len(xte)):
        it.set_tensor(inp["index"], xte[i:i + 1].astype(inp["dtype"]))
        it.invoke()
        preds.append(it.get_tensor(out["index"])[0].copy())
    p = np.array(preds)
    if is_cls:
        return float((p.argmax(1) == yte).mean())
    return float(np.mean(np.abs(p.squeeze(-1) - yte)))


def main(out_dir):
    jobs = [
        ("classification", "dmir_cls/models/model_aaaabl.h5", "cls_best"),
        ("classification", "dmir_cls/models/model_aaaaat.h5", "cls_tiny"),
        ("classification_noind", "dmir_cls_noind/models/model_aaaaah.h5", "cls_noind"),
        ("regression_lcr", "dmir_lcr/models/model_aaaaan.h5", "lcr_best"),
        ("regression_lcl", "dmir_lcl_rmse/models/model_aaaabu.h5", "lcl_best"),
    ]
    art = Path(os.environ.get("ARTIFACTS", str(Path.home() / "uNAS" / "artifacts")))
    ref = Path(os.environ.get("REF_MODELS_DIR", str(Path.home() / "ref_models")))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    todo = [(t, art / rel, tag) for t, rel, tag in jobs]
    todo.append(("classification", ref / "cnn_multi.h5", "REF_cnn_multi"))

    for task, h5, tag in todo:
        if not Path(h5).exists():
            print(f"{tag}: MISSING {h5}"); continue
        xtr, xte, yte, is_cls = prep(task)
        m = tf.keras.models.load_model(h5, compile=False)
        fp = m.predict(xte, batch_size=1024, verbose=0)
        fm = float((fp.argmax(1) == yte).mean()) if is_cls \
            else float(np.mean(np.abs(fp.squeeze(-1) - yte)))
        shutil.copy(h5, out / f"{tag}.h5")
        b32 = convert(m, xtr, "float32")
        (out / f"{tag}_float32.tflite").write_bytes(b32)
        b8 = convert(m, xtr, "int8")
        (out / f"{tag}_int8.tflite").write_bytes(b8)
        q8 = eval_tflite(b8, xte, yte, is_cls)
        rows.append((tag, m.count_params(), len(b32) / 1024, len(b8) / 1024, fm, q8, is_cls))
        mn = "acc" if is_cls else "MAE"
        print(f"{tag:14} params={m.count_params():>7}  fp32={len(b32)/1024:6.1f}KB  "
              f"int8={len(b8)/1024:6.1f}KB  float_{mn}={fm:.4f}  int8_{mn}={q8:.4f}")

    print("\n=== files in", out, "===")
    for f in sorted(out.iterdir()):
        print(f"  {f.stat().st_size/1024:8.1f} KB  {f.name}")


if __name__ == "__main__":
    main(sys.argv[1])
