"""Independently evaluate the colleague's reference models on the DMIR test set.

Confirms the internal-reference numbers (92% / 0.42 / 0.44) and gives honest,
like-for-like test metrics + parameter counts for the paper's comparison row.
Evaluates both raw and train-range-clipped test inputs (their pipeline may or
may not clip). Run in the WSL dmir_nas venv.
"""
import os
import pickle
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


@keras.saving.register_keras_serializable(package="Custom")
class PositionalEncoding(keras.layers.Layer):
    """Reconstructed from the ELIOS LCIR framework (learned position embedding
    added to the input), so the reference Transformers deserialize."""

    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.position_embedding = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        return x + self.position_embedding(positions)

    def get_config(self):
        c = super().get_config()
        c.update(sequence_length=self.sequence_length, embed_dim=self.embed_dim)
        return c


CUSTOM = {"PositionalEncoding": PositionalEncoding,
          "Custom>PositionalEncoding": PositionalEncoding}

MODELS = Path(os.environ.get("REF_MODELS_DIR",
                             "/mnt/c/Projects/PhD/DIMIR/Materials/Models"))
# The .keras files are actually legacy HDF5 (magic \x89HDF); Keras 3 loads them
# via the .h5 legacy path, so we load the renamed .h5 copies.
JOBS = [
    ("cnn_multi.h5",        "classification",   "data-classification", "multi"),
    ("transformer_lcr.h5",  "regression_lcr",   "data-prepared-lcr",   "lcr"),
    ("transformer_lcl.h5",  "regression_lcl",   "data-regression-lcl", "lcl"),
]


def load_test(folder, sfx):
    root = Path(os.environ["DMIR_DATA_ROOT"]) / folder
    def rd(s, xy):
        with open(root / f"{xy}_{s}_{sfx}.pkl", "rb") as f:
            return pickle.load(f)
    xtr = rd("train", "x").astype(np.float32)
    xte = rd("test", "x").astype(np.float32)
    yte = rd("test", "y")
    return xtr, xte, yte


def metrics(model, x, y, is_cls):
    p = model.predict(x, batch_size=512, verbose=0)
    if is_cls:
        return {"acc": float((p.argmax(1) == y.astype(np.int64)).mean())}
    p = p.squeeze(-1)
    y = y.astype(np.float32)
    return {"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae": float(np.mean(np.abs(p - y)))}


def main():
    for fname, task, folder, sfx in JOBS:
        path = MODELS / fname
        if not path.exists():
            print(f"{fname}: MISSING"); continue
        try:
            m = tf.keras.models.load_model(path, compile=False, custom_objects=CUSTOM)
        except Exception as e:  # noqa: BLE001
            print(f"{fname}: load FAILED {str(e)[:140]}"); continue
        is_cls = task == "classification"
        xtr, xte, yte = load_test(folder, sfx)
        # adapt to the model's expected input shape if it differs from (50,31)
        want = m.input_shape[1:]
        xte_use = xte
        note = ""
        if tuple(want) != xte.shape[1:]:
            if tuple(want) == xte.shape[1:][::-1]:
                xte_use = np.transpose(xte, (0, 2, 1)); note = " (transposed)"
            else:
                note = f" [WARN model wants {want}, data {xte.shape[1:]}]"
        lo, hi = xtr.min(axis=(0, 1)), xtr.max(axis=(0, 1))
        xte_clip = np.clip(xte_use if not note.startswith(" (transp") else xte, lo, hi)
        if note.startswith(" (transp"):
            xte_clip = np.transpose(xte_clip, (0, 2, 1))
        raw = metrics(m, xte_use, yte, is_cls)
        clip = metrics(m, xte_clip, yte, is_cls)
        print(f"{fname:24} params={m.count_params():>8}  in={m.input_shape[1:]}{note}")
        print(f"    raw  : {raw}")
        print(f"    clip : {clip}")


if __name__ == "__main__":
    main()
