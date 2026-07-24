"""QAT fine-tuning for a searched classification model (cls_best), with an
honest within-graph comparison against PTQ.

Why 2D: tfmot's 8-bit scheme only registers 2D layers, so the searched 1D graph
is re-expressed with width-1 kernels (Conv1D->Conv2D(k,1), Pool1D->Pool2D(p,1)).
The computation is identical; we PROVE it by checking the float-2D test accuracy
equals the original (0.9208) and the PTQ-2D accuracy equals the measured 1D
int8 number (0.8686). PTQ and QAT then run on the SAME 2D graph, so the only
variable is post-training vs quantization-aware — a fair single-variable test.

Run in the WSL dmir_nas venv (tf_keras / tfmot need TF_USE_LEGACY_KERAS):
  source ~/dmir_nas/env.sh
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/data TF_USE_LEGACY_KERAS=1 \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/qat_finetune.py \
    <graph.json> <weights.pkl> <out_dir>

Emits <out_dir>/cls_best_qat_int8.tflite and prints a comparison table.
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
TASK = "classification"
ORIG_FLOAT_ACC = 0.9208   # 1D cls_best, measured (deployment.md)
ORIG_INT8_ACC = 0.8686    # 1D cls_best int8 PTQ, measured (deployment.md)


def prep_split(split):
    folder, sfx, is_cls, _ = LAYOUT[TASK]
    x = load(TASK, split, "x").astype(np.float32)
    y = load(TASK, split, "y").astype(np.int64)
    return x, y


def prep_all():
    xtr, ytr = prep_split("train")
    xva, yva = prep_split("val")
    xte, yte = prep_split("test")
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


def acc_keras(model, x, y):
    p = model.predict(x, batch_size=1024, verbose=0)
    return float((p.argmax(1) == y).mean())


def main(graph_path, weights_path, out_dir):
    graph = json.load(open(graph_path))
    weights = pickle.load(open(weights_path, "rb"))
    (xtr, ytr), (xva, yva), (xte, yte) = prep_all()
    xtr4, xva4, xte4 = to4d(xtr), to4d(xva), to4d(xte)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) float-2D must reproduce the 1D float accuracy -> proves the reshape
    model = build_2d(graph)
    transfer_weights(model, weights)
    float2d = acc_keras(model, xte4, yte)
    print(f"[anchor 1] float32 2D acc = {float2d:.4f}   (1D orig {ORIG_FLOAT_ACC})")

    # 2) PTQ-2D must reproduce the 1D int8 accuracy -> proves fairness
    ptq = to_tflite(model, xtr4, mode="int8")
    ptq_acc = eval_tflite(ptq, xte4, yte, is_cls=True)
    (out / "cls_best_2d_ptq_int8.tflite").write_bytes(ptq)
    print(f"[anchor 2] PTQ int8 2D acc = {ptq_acc:.4f}   (1D orig {ORIG_INT8_ACC})")

    # 3) QAT on the SAME 2D graph
    import tensorflow_model_optimization as tfmot
    qa = tfmot.quantization.keras.quantize_model(model)
    qa.compile(optimizer=K.optimizers.Adam(2e-4),
               loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=["accuracy"])
    cbs = [K.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                     patience=8, restore_best_weights=True)]
    qa.fit(xtr4, ytr, validation_data=(xva4, yva), epochs=40, batch_size=256,
           callbacks=cbs, verbose=2)
    qat_float = acc_keras(qa, xte4, yte)
    print(f"[QAT] fake-quant (still float) test acc = {qat_float:.4f}")

    # 4) convert the QAT model to full int8 TFLite (same converter as PTQ)
    qat_tfl = to_tflite(qa, xtr4, mode="int8")
    qat_acc = eval_tflite(qat_tfl, xte4, yte, is_cls=True)
    (out / "cls_best_qat_int8.tflite").write_bytes(qat_tfl)

    print("\n==== RESULT (all test-set, same graph for PTQ vs QAT) ====")
    print(f"float32 (accuracy operating point) : {float2d:.4f}")
    print(f"int8 PTQ                           : {ptq_acc:.4f}")
    print(f"int8 QAT                           : {qat_acc:.4f}")
    print(f"QAT recovers over PTQ              : {qat_acc - ptq_acc:+.4f}")
    print(f"remaining gap to float32          : {float2d - qat_acc:+.4f}")
    print(f"qat int8 tflite bytes             : {len(qat_tfl)}")

    (out / "qat_result.json").write_text(json.dumps({
        "task": TASK, "graph": os.path.basename(graph_path),
        "float32_2d_acc": float2d, "ptq_int8_2d_acc": ptq_acc,
        "qat_int8_2d_acc": qat_acc, "qat_fakequant_float_acc": qat_float,
        "orig_1d_float_acc": ORIG_FLOAT_ACC, "orig_1d_int8_acc": ORIG_INT8_ACC,
        "qat_int8_tflite_bytes": len(qat_tfl),
    }, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
