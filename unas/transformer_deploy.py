"""Deploy the colleague's reference Transformers to TFLite and the ST board farm.

Why this exists: the internal reference models for DMIR regression are
Transformers, and the published SOTA (Forneris et al., IEEE SPL 2026) is also a
Transformer. Partners have asked whether Transformers are the right choice for
the on-vehicle model. The project had never put one on a board, so the answer
was an opinion. This script turns it into a measurement.

The `.keras` files are HDF5 with three custom classes that are not in any public
repository (`Custom>PositionalEncoding`, `Custom>TransformerEncoder`,
`Custom>LastTimestep`), so `load_model` fails. Their *configuration* is stored in
the file, however, so the architecture is rebuilt exactly from that config and
the ORIGINAL trained weights are loaded into it by name. Nothing is retrained
and nothing is invented; the rebuild is verified by reproducing the reported
test RMSE before anything is exported.

IMPORTANT, and stated in the emitted record: the rebuild reproduces the
architecture exactly (parameter counts match the originals to the unit:
333,505 for LCR and 49,089 for LCL) but does NOT reproduce the reported test
RMSE, so the residual mapping error is unresolved. Accuracy is therefore never
claimed from these artifacts. They are used only to measure DEPLOYMENT COST --
latency, flash and RAM -- which is a function of the operator graph and tensor
shapes, not of the values in the weights.

Encoder block, read off the stored weight names and shapes:
    MHA(d_model, heads, key_dim = d_model // heads)  ->  add  ->  LayerNorm
    Dense(ff_dim, relu, no bias) -> Dense(d_model, no bias)  ->  add  ->  LayerNorm

Run in the WSL dmir_nas venv:
  source ~/dmir_nas/env.sh
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data \
    ~/dmir_nas/bin/python unas/transformer_deploy.py <lcr|lcl> <out_dir>

Emits <out_dir>/transformer_<task>_fp32.tflite plus a JSON record with the
rebuild check, the conversion outcome and the artifact size.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_eval import LAYOUT, load  # same data path as every other number

K = tf.keras


class AddPositional(K.layers.Layer):
    """Learned positional embedding added to the sequence (the original
    `Custom>PositionalEncoding`: an Embedding(timesteps, embed_dim) whose output
    is added to the projected input)."""

    def __init__(self, timesteps, embed_dim, **kw):
        super().__init__(**kw)
        self.timesteps, self.embed_dim = timesteps, embed_dim

    def build(self, _):
        self.pos = self.add_weight(shape=(self.timesteps, self.embed_dim),
                                   initializer="zeros", trainable=True, name="embeddings")

    def call(self, x):
        return x + self.pos

    def get_config(self):
        return {**super().get_config(), "timesteps": self.timesteps,
                "embed_dim": self.embed_dim}

SRC = {"lcr": "Materials/Models/transformer_lcr.keras",
       "lcl": "Materials/Models/transformer_lcl.keras"}
TASK = {"lcr": "regression_lcr", "lcl": "regression_lcl"}
# Reported test RMSE for these exact files (paper/NOTES.md, colleague's numbers).
REPORTED_RMSE = {"lcr": 0.42, "lcl": 0.44}
REPO = Path(__file__).resolve().parent.parent


def encoder_block(x, d_model, num_heads, ff_dim, tag):
    """One TransformerEncoder, matching the stored weight names and shapes."""
    key_dim = d_model // num_heads
    attn = K.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"mha{tag}")(x, x, x)
    x = K.layers.LayerNormalization(epsilon=1e-6, name=f"ln{tag}a")(x + attn)
    ff = K.layers.Dense(ff_dim, activation="relu", use_bias=False, name=f"ff{tag}a")(x)
    ff = K.layers.Dense(d_model, use_bias=False, name=f"ff{tag}b")(ff)
    return K.layers.LayerNormalization(epsilon=1e-6, name=f"ln{tag}b")(x + ff)


def build(cfg_layers, timesteps, n_feat):
    """Rebuild the graph from the stored config."""
    inp = K.Input(shape=(timesteps, n_feat), name="input_1")
    x = inp
    blocks = []
    proj_units = head_units = None
    for L in cfg_layers:
        cn, c = L["class_name"], L.get("config", {})
        if cn == "Dense" and proj_units is None:
            proj_units = c["units"]
            x = K.layers.Dense(proj_units, name="proj")(x)
        elif cn == "Custom>PositionalEncoding":
            x = AddPositional(timesteps, c["embed_dim"], name="posemb")(x)
        elif cn == "Custom>TransformerEncoder":
            tag = len(blocks)
            x = encoder_block(x, c["d_model"], c["num_heads"], c["ff_dim"], tag)
            blocks.append(tag)
        elif cn == "Custom>LastTimestep":
            x = K.layers.Lambda(lambda t: t[:, -1, :], name="last")(x)
        elif cn == "Dense" and c.get("activation") == "relu":
            head_units = c["units"]
            x = K.layers.Dense(head_units, activation="relu", name="head")(x)
        elif cn == "Dense" and c["units"] == 1:
            x = K.layers.Dense(1, name="out")(x)
    return K.Model(inp, x), blocks


def load_weights(model, h5path, blocks):
    """Copy the original trained weights in by matching stored dataset paths."""
    h = h5py.File(h5path, "r")
    W = {}

    def collect(g, prefix=""):
        for k in g:
            item = g[k]
            if isinstance(item, h5py.Dataset):
                W[prefix + k] = np.array(item)
            else:
                collect(item, prefix + k + "/")
    collect(h["model_weights"])

    def find(*needles):
        hits = [v for p, v in W.items() if all(n in p for n in needles)]
        if len(hits) != 1:
            raise KeyError(f"{needles} matched {len(hits)} datasets")
        return hits[0]

    def enc(i, *needles):
        pre = "transformer_encoder/" if i == 0 else f"transformer_encoder_{i}/"
        hits = [v for p, v in W.items() if p.startswith(pre) and all(n in p for n in needles)]
        if len(hits) != 1:
            raise KeyError(f"block {i} {needles} matched {len(hits)}")
        return hits[0]

    model.get_layer("proj").set_weights([find("dense/dense/kernel"),
                                         np.zeros(find("dense/dense/kernel").shape[1], np.float32)])
    model.get_layer("posemb").set_weights([find("positional_encoding", "embeddings")])
    for i in blocks:
        model.get_layer(f"mha{i}").set_weights([
            enc(i, "query/kernel"), enc(i, "query/bias"),
            enc(i, "key/kernel"), enc(i, "key/bias"),
            enc(i, "value/kernel"), enc(i, "value/bias"),
            enc(i, "attention_output/kernel"), enc(i, "attention_output/bias")])
        # LayerNorm indices continue across blocks (block 1 holds _2 and _3),
        # so take the two inside this block's prefix in stored order.
        pre = "transformer_encoder/" if i == 0 else f"transformer_encoder_{i}/"
        lns = sorted({p.split("layer_normalization")[1].split("/")[0]
                      for p in W if p.startswith(pre) and "layer_normalization" in p})
        for slot, suffix in zip(("a", "b"), lns):
            g = [v for p, v in W.items() if p.startswith(pre)
                 and f"layer_normalization{suffix}/gamma" in p][0]
            b = [v for p, v in W.items() if p.startswith(pre)
                 and f"layer_normalization{suffix}/beta" in p][0]
            model.get_layer(f"ln{i}{slot}").set_weights([g, b])
        ffk = sorted([(p, v) for p, v in W.items()
                      if p.startswith("transformer_encoder/" if i == 0 else f"transformer_encoder_{i}/")
                      and "dense" in p and "kernel" in p], key=lambda kv: kv[0])
        model.get_layer(f"ff{i}a").set_weights([ffk[0][1]])
        model.get_layer(f"ff{i}b").set_weights([ffk[1][1]])
    # head: the two Dense layers that are not the projection
    heads = sorted([(p, v) for p, v in W.items()
                    if p.startswith("dense_") and "kernel" in p], key=lambda kv: kv[0])
    hb = sorted([(p, v) for p, v in W.items()
                 if p.startswith("dense_") and "bias" in p], key=lambda kv: kv[0])
    model.get_layer("head").set_weights([heads[0][1], hb[0][1]])
    model.get_layer("out").set_weights([heads[1][1], hb[1][1]])
    return model


def prep(task):
    folder, sfx, is_cls, _ = LAYOUT[task]
    xtr = load(task, "train", "x").astype(np.float32)
    xte = load(task, "test", "x").astype(np.float32)
    yte = load(task, "test", "y").astype(np.float32)
    lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
    np.clip(xte, lo, hi, out=xte)
    return xtr, xte, yte


def main(which, out_dir):
    src = REPO / SRC[which]
    task = TASK[which]
    h = h5py.File(src, "r")
    cfg = json.loads(h.attrs["model_config"])["config"]["layers"]

    xtr, xte, yte = prep(task)
    model, blocks = build(cfg, xte.shape[1], xte.shape[2])
    load_weights(model, src, blocks)
    params = int(model.count_params())

    pred = model.predict(xte, batch_size=512, verbose=0).squeeze(-1)
    rmse = float(np.sqrt(((pred - yte) ** 2).mean()))
    mae = float(np.abs(pred - yte).mean())
    rebuilt_ok = abs(rmse - REPORTED_RMSE[which]) < 0.03
    print(f"[rebuild] {which}: {params:,} params | test RMSE {rmse:.4f} "
          f"(reported {REPORTED_RMSE[which]}) MAE {mae:.4f} -> "
          f"{'MATCH' if rebuilt_ok else 'MISMATCH, do not deploy'}", flush=True)

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rec = {"model": f"transformer_{which}", "source": SRC[which], "params": params,
           "rebuilt_test_rmse": round(rmse, 4), "rebuilt_test_mae": round(mae, 4),
           "reported_rmse": REPORTED_RMSE[which], "rebuild_matches_reported": rebuilt_ok,
           "usable_for": "deployment cost only (latency / flash / RAM depend on the "
                         "operator graph and tensor shapes, not on weight values)",
           "not_usable_for": "any accuracy claim -- the rebuilt RMSE does not match the "
                             "reported figure, so the weight mapping is unverified"}

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    try:
        tfl = conv.convert()
        rec["tflite_builtin_only"] = True
    except Exception as e:                       # needs ops outside the MCU runtime
        rec["tflite_builtin_only"] = False
        rec["builtin_error"] = str(e)[:400]
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                          tf.lite.OpsSet.SELECT_TF_OPS]
        tfl = conv.convert()
    path = out / f"transformer_{which}_fp32.tflite"
    path.write_bytes(tfl)
    rec["tflite_bytes"] = len(tfl)

    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    ops = sorted({d["op_name"] for d in it._get_ops_details()}) if hasattr(it, "_get_ops_details") else []
    rec["ops"] = ops
    rec["flex_ops"] = [o for o in ops if o.lower().startswith("flex")]
    print(f"[tflite] {len(tfl):,} B | builtin-only={rec['tflite_builtin_only']} | "
          f"{len(ops)} distinct ops | flex: {rec['flex_ops']}", flush=True)
    print("  ops:", ", ".join(ops), flush=True)

    (out / f"transformer_{which}_deploy.json").write_text(json.dumps(rec, indent=1))
    print("->", path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
