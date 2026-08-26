"""Re-export a model to full-int8 TFLite with an **int8 tensor interface**.

Why: `prepare_deploy.py` leaves `inference_input_type`/`inference_output_type` at
their float32 default, so even the int8 models take FLOAT32 in and out. ST Edge
AI must then allocate a 50x31 float32 input buffer — 6,200 B — which dominates
the activation arena and is why cls_best_int8 measured 8,096 B of RAM instead of
the predicted ~3 KB (datasets/dmir/docs/deployment.md, "Input-bound"). Quantizing the
*weights* does not quantize the *interface*; this script fixes the interface.

Setting an int8 interface changes how the model must be fed: the caller has to
quantize with the input tensor's own (scale, zero_point) rather than hand it
floats. `eval_tflite` in prepare_deploy.py/quantize_eval.py only *casts*
(`.astype(dtype)`), which is a no-op for float32 I/O — correct for every number
measured so far — but would silently destroy an int8-I/O input. So this script
carries its own quantizing evaluator and uses it to PROVE the re-export is
equivalent: the int8-I/O model must reproduce the float32-I/O int8 accuracy
(cls_best: 0.8686). If it does, the only thing that changed is the interface.

Run in the WSL dmir_nas venv:
  source ~/dmir_nas/env.sh
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data \
    ~/dmir_nas/bin/python unas/export_int8_io.py <task> <model.h5> <out_dir>
  task in {classification, classification_noind, regression_lcr, regression_lcl}

Emits <out_dir>/<stem>_int8_io.tflite + a comparison against the float32-I/O
build, including the predicted RAM saving to check against the ST report.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_deploy import LAYOUT, prep

# Measured float32-I/O int8 references (datasets/dmir/docs/deployment.md).
REFERENCE = {
    "classification": {"model": "cls_best", "int8_metric": 0.8686,
                       "measured_ram_B": 8096, "measured_flash_B": 106738},
}


def convert_int8(model, xtr, int8_io):
    """Full-integer int8 PTQ. int8_io=True also makes the *interface* int8."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    rng = np.random.default_rng(42)          # same seed/size as prepare_deploy
    s = xtr[rng.choice(len(xtr), size=min(500, len(xtr)), replace=False)]
    conv.representative_dataset = lambda: ([s[i:i + 1]] for i in range(len(s)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if int8_io:
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def eval_quantized(tflite_bytes, xte, yte, is_cls):
    """Evaluate honouring the tensor interface: quantize the input and
    dequantize the output when the interface is integer."""
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    int_in = inp["dtype"] in (np.int8, np.uint8)
    int_out = out["dtype"] in (np.int8, np.uint8)
    info = np.iinfo(inp["dtype"]) if int_in else None

    preds = []
    for i in range(len(xte)):
        x = xte[i:i + 1]
        if int_in:
            if in_scale == 0:
                raise SystemExit("int8 input but scale is 0 — no quantization params")
            q = np.round(x / in_scale + in_zp)
            x = np.clip(q, info.min, info.max).astype(inp["dtype"])
        it.set_tensor(inp["index"], x)
        it.invoke()
        y = it.get_tensor(out["index"])[0].astype(np.float32).copy()
        if int_out:
            y = (y - out_zp) * out_scale
        preds.append(y)

    p = np.array(preds)
    if is_cls:
        return float((p.argmax(1) == yte).mean())
    return float(np.mean(np.abs(p.squeeze(-1) - yte)))


def describe(tflite_bytes):
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    return {"bytes": len(tflite_bytes),
            "in_dtype": inp["dtype"].__name__, "in_shape": list(map(int, inp["shape"])),
            "out_dtype": out["dtype"].__name__,
            "in_scale": float(inp["quantization"][0]),
            "in_zero_point": int(inp["quantization"][1])}


def main(task, h5_path, out_dir):
    xtr, xte, yte, is_cls = prep(task)
    model = tf.keras.models.load_model(h5_path, compile=False)
    stem = Path(h5_path).stem
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # control: the float32-I/O int8 build, exactly as prepare_deploy.py makes it
    f32io = convert_int8(model, xtr, int8_io=False)
    m_f32io = eval_quantized(f32io, xte, yte, is_cls)

    # the re-export under test
    i8io = convert_int8(model, xtr, int8_io=True)
    m_i8io = eval_quantized(i8io, xte, yte, is_cls)
    path = out / f"{stem}_int8_io.tflite"
    path.write_bytes(i8io)

    d32, d8 = describe(f32io), describe(i8io)
    elems = int(np.prod(d8["in_shape"]))
    saving = elems * 3          # float32 (4 B) -> int8 (1 B) per element
    mname = "acc" if is_cls else "MAE"
    ref = REFERENCE.get(task)

    print(f"\n==== {stem} — int8 weights, two interfaces ====")
    print(f"float32 I/O : {d32['in_dtype']:>7} in / {d32['out_dtype']:<7} out  "
          f"{d32['bytes']:>7,} B  test {mname} {m_f32io:.4f}")
    print(f"int8    I/O : {d8['in_dtype']:>7} in / {d8['out_dtype']:<7} out  "
          f"{d8['bytes']:>7,} B  test {mname} {m_i8io:.4f}")
    print(f"input quantization: scale={d8['in_scale']:.6g} zero_point={d8['in_zero_point']}")
    delta = (m_i8io - m_f32io) if is_cls else (m_f32io - m_i8io)
    print(f"interface change costs: {delta:+.4f} {mname} "
          f"({'equivalent' if abs(m_i8io - m_f32io) < 5e-3 else 'NOT equivalent — investigate'})")
    if ref:
        print(f"anchor vs measured float32-I/O int8 ({ref['int8_metric']}): "
              f"{'match' if abs(m_f32io - ref['int8_metric']) < 5e-3 else 'MISMATCH'}")
        print(f"\nPrediction to check on ST Edge AI:")
        print(f"  input buffer {elems * 4:,} B (float32) -> {elems:,} B (int8), "
              f"saving {saving:,} B")
        print(f"  measured RAM was {ref['measured_ram_B']:,} B -> expect "
              f"~{ref['measured_ram_B'] - saving:,} B")
        print(f"  the conversion_0 float->int8 cast op should disappear")
    print(f"\nupload: {path}")

    (out / f"{stem}_int8_io_result.json").write_text(json.dumps({
        "task": task, "model": stem, "metric": mname,
        "float32_io": {**d32, "test_metric": m_f32io},
        "int8_io": {**d8, "test_metric": m_i8io},
        "interface_equivalent": bool(abs(m_i8io - m_f32io) < 5e-3),
        "input_elems": elems,
        "predicted_ram_saving_B": saving,
        "predicted_ram_B": (ref["measured_ram_B"] - saving) if ref else None,
    }, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
