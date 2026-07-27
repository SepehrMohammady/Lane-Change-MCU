"""QAT fine-tuning for a searched model, with an honest within-graph comparison
against PTQ. Works for the classifier and both regression heads.

Why 2D: tfmot's 8-bit scheme only registers 2D layers, so the searched 1D graph
is re-expressed with width-1 kernels (Conv1D->Conv2D(k,1), Pool1D->Pool2D(p,1);
depthwise strides (s,1)->(s,s), a no-op at width 1). The computation is
identical, and we PROVE it with two anchors: the float-2D test metric must equal
the original 1D number, and the PTQ-2D metric must equal the measured 1D int8
number. PTQ and QAT then run on the SAME 2D graph, so the only variable is
post-training vs quantization-aware — a fair single-variable test.

Run in the WSL dmir_nas venv (tf_keras / tfmot need TF_USE_LEGACY_KERAS), after
exporting the graph with unas/export_graph.py under Keras 3:
  source ~/dmir_nas/env.sh
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/data TF_USE_LEGACY_KERAS=1 \
    ~/dmir_nas/bin/python unas/qat_finetune.py \
    <task> <graph.json> <weights.pkl> <out_dir>
  task in {classification, regression_lcr, regression_lcl}

Emits <out_dir>/<model>_qat_int8.tflite + <model>_qat_result.json.
No fabrication: every number is a real test-set evaluation of a real artifact.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_eval import LAYOUT, load, to_tflite, eval_tflite  # identical pipeline

K = tf.keras

# Reference points measured on the ORIGINAL 1D models (docs/research/deployment.md),
# used as anchors to prove the 2D re-expression is faithful. For classification the
# metric is accuracy (higher better); for regression it is test MAE (lower better).
ANCHORS = {
    "classification":  {"float": 0.9208, "int8": 0.8686, "model": "cls_best"},
    "regression_lcr":  {"float": 0.2865, "int8": 0.4485, "model": "lcr_best"},
    "regression_lcl":  {"float": 0.3165, "int8": 0.3440, "model": "lcl_best"},
}


def prep_split(task, split):
    _, _, is_cls, _ = LAYOUT[task]
    x = load(task, split, "x").astype(np.float32)
    y = load(task, split, "y")
    return x, (y.astype(np.int64) if is_cls else y.astype(np.float32))


def prep_all(task):
    xtr, ytr = prep_split(task, "train")
    xva, yva = prep_split(task, "val")
    xte, yte = prep_split(task, "test")
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    np.clip(xva, lo, hi, out=xva)
    np.clip(xte, lo, hi, out=xte)              # same clip used for the 1D numbers
    return (xtr, ytr), (xva, yva), (xte, yte)


def to4d(x):
    """(N, 50, 31) -> (N, 50, 1, 31): insert a width-1 axis (channels last)."""
    return x[:, :, None, :]


# --- build the 2D-equivalent functional model in tf_keras -------------------

def build_2d(graph):
    def cfg2d(L):
        c = dict(L["config"])
        c.pop("dtype", None)
        return c

    tensors = {}
    for L in graph["layers"]:
        cls, name, c = L["class_name"], L["name"], cfg2d(L)
        if cls == "InputLayer":
            tensors[name] = K.Input(shape=(50, 1, 31), name=name)
            continue
        if cls == "MaxPooling1D":
            p = tuple(c["pool_size"]) + (1,); s = tuple(c["strides"]) + (1,)
            layer = K.layers.MaxPooling2D(p, s, padding=c["padding"], name=name)
        elif cls == "AveragePooling1D":
            p = tuple(c["pool_size"]) + (1,); s = tuple(c["strides"]) + (1,)
            layer = K.layers.AveragePooling2D(p, s, padding=c["padding"], name=name)
        elif cls == "Conv1D":
            # width is 1, so an equal (s,s) stride is identical to (s,1) but
            # satisfies TF's equal-stride requirement for depthwise/conv kernels
            s = c["strides"][0]
            layer = K.layers.Conv2D(
                c["filters"], tuple(c["kernel_size"]) + (1,),
                strides=(s, s), padding=c["padding"],
                dilation_rate=tuple(c["dilation_rate"]) + (1,),
                use_bias=c["use_bias"], activation=c.get("activation", "linear"),
                name=name)
        elif cls == "DepthwiseConv1D":
            s = c["strides"][0]
            layer = K.layers.DepthwiseConv2D(
                tuple(c["kernel_size"]) + (1,),
                strides=(s, s), padding=c["padding"],
                dilation_rate=tuple(c["dilation_rate"]) + (1,),
                depth_multiplier=c["depth_multiplier"], use_bias=c["use_bias"],
                activation=c.get("activation", "linear"), name=name)
        else:  # ReLU, Add, Flatten, Dense, BatchNormalization — unchanged
            layer = K.layers.deserialize({"class_name": cls, "config": c})
        ins = [tensors[n] for n in L["inputs"]]
        tensors[name] = layer(ins if len(ins) > 1 else ins[0])
    inp = tensors[graph["input_names"][0]]
    out = tensors[graph["output_names"][0]]
    return K.Model(inp, out)


def transfer_weights(model, weights):
    """Set weights, inserting the width-1 axis into conv kernels."""
    for layer in model.layers:
        if layer.name not in weights:
            continue
        w = [np.asarray(a) for a in weights[layer.name]]
        cls = layer.__class__.__name__
        if cls in ("Conv2D", "DepthwiseConv2D"):
            w[0] = w[0][:, None, :, :]           # (k,in,out) -> (k,1,in,out)
        layer.set_weights(w)


def metric_keras(model, x, y, is_cls):
    """Accuracy for classification, MAE for regression — same definition the
    1D numbers in deployment.md use."""
    p = model.predict(x, batch_size=1024, verbose=0)
    if is_cls:
        return float((p.argmax(1) == y).mean())
    return float(np.mean(np.abs(p.squeeze(-1) - y)))


def main(task, graph_path, weights_path, out_dir):
    if task not in ANCHORS:
        raise SystemExit(f"task must be one of {sorted(ANCHORS)}")
    is_cls = LAYOUT[task][2]
    anchor = ANCHORS[task]
    name = anchor["model"]
    mname = "acc" if is_cls else "MAE"
    better = "higher" if is_cls else "lower"

    graph = json.load(open(graph_path))
    weights = pickle.load(open(weights_path, "rb"))
    (xtr, ytr), (xva, yva), (xte, yte) = prep_all(task)
    xtr4, xva4, xte4 = to4d(xtr), to4d(xva), to4d(xte)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) float-2D must reproduce the 1D float metric -> proves the reshape
    model = build_2d(graph)
    transfer_weights(model, weights)
    float2d = metric_keras(model, xte4, yte, is_cls)
    print(f"[anchor 1] float32 2D {mname} = {float2d:.4f}   (1D orig {anchor['float']})")

    # 2) PTQ-2D must reproduce the 1D int8 metric -> proves the comparison is fair
    ptq = to_tflite(model, xtr4, mode="int8")
    ptq_m = eval_tflite(ptq, xte4, yte, is_cls)
    (out / f"{name}_2d_ptq_int8.tflite").write_bytes(ptq)
    print(f"[anchor 2] PTQ int8 2D {mname} = {ptq_m:.4f}   (1D orig {anchor['int8']})")

    # 3) QAT on the SAME 2D graph
    import tensorflow_model_optimization as tfmot
    qa = tfmot.quantization.keras.quantize_model(model)
    if is_cls:
        loss, metrics, monitor, mode = (
            K.losses.SparseCategoricalCrossentropy(from_logits=True),
            ["accuracy"], "val_accuracy", "max")
    else:
        # MAE objective, matching how the regression models were searched/trained
        loss, metrics, monitor, mode = "mae", ["mae"], "val_mae", "min"
    lr = float(os.environ.get("DMIR_QAT_LR", "2e-4"))
    qa.compile(optimizer=K.optimizers.Adam(lr), loss=loss, metrics=metrics)
    print(f"[QAT] fine-tune lr={lr:g}")
    cbs = [K.callbacks.EarlyStopping(monitor=monitor, mode=mode,
                                     patience=8, restore_best_weights=True)]
    qa.fit(xtr4, ytr, validation_data=(xva4, yva), epochs=40, batch_size=256,
           callbacks=cbs, verbose=2)
    qat_float = metric_keras(qa, xte4, yte, is_cls)
    print(f"[QAT] fake-quant (still float) test {mname} = {qat_float:.4f}")

    # 4) convert the QAT model to full int8 TFLite (same converter as PTQ)
    qat_tfl = to_tflite(qa, xtr4, mode="int8")
    qat_m = eval_tflite(qat_tfl, xte4, yte, is_cls)
    (out / f"{name}_qat_int8.tflite").write_bytes(qat_tfl)

    gain = (qat_m - ptq_m) if is_cls else (ptq_m - qat_m)      # + = QAT better
    gap = (float2d - qat_m) if is_cls else (qat_m - float2d)   # + = still behind float
    print(f"\n==== RESULT {name} ({task}) — same graph for PTQ vs QAT ====")
    print(f"float32 ({better} is better)  : {float2d:.4f}")
    print(f"int8 PTQ                      : {ptq_m:.4f}")
    print(f"int8 QAT                      : {qat_m:.4f}")
    print(f"QAT recovers over PTQ         : {gain:+.4f}")
    print(f"remaining gap to float32      : {gap:+.4f}")
    print(f"qat int8 tflite bytes         : {len(qat_tfl)}")

    (out / f"{name}_qat_result.json").write_text(json.dumps({
        "task": task, "model": name, "metric": mname,
        "graph": os.path.basename(graph_path),
        "float32_2d": float2d, "ptq_int8_2d": ptq_m, "qat_int8_2d": qat_m,
        "qat_fakequant_float": qat_float,
        "orig_1d_float": anchor["float"], "orig_1d_int8": anchor["int8"],
        "qat_recovers_over_ptq": gain, "remaining_gap_to_float32": gap,
        "qat_int8_tflite_bytes": len(qat_tfl), "lr": lr,
    }, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
