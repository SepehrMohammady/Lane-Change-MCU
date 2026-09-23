"""Export the trained hand-designed DSCNNs for the TFLite and board pipeline.

The DSCNN is a PyTorch model; the searched models are Keras models converted to
TFLite by unas/prepare_deploy*.py. To measure both the same way, this script
writes, per model, an .npz that unas/deploy_hand_cnn.py (WSL, TensorFlow) turns
into the same three TFLite builds (float32, int8 PTQ with float I/O, int8 PTQ
with int8 I/O):

  * the weights in Keras layout (conv kernels (k, in, out), depthwise kernels
    (k, channels, 1), dense kernels (in, out), BatchNorm statistics),
  * 256 test windows and the PyTorch outputs on them, for a parity check,
  * the 500 calibration windows that unas/prepare_deploy*.py would draw from
    the same training windows (np.random.default_rng(42)),
  * the preprocessed test windows and targets, so every TFLite file can be
    scored on exactly the inputs the PyTorch evaluation uses.

All windows are channel-last (N, T, C), the layout of the searched models.

    python scripts/export_hand_cnn.py

Output: datasets/<dmir|highd>/data/hand_export/<name>.npz (gitignored: holds data).
LCIR weights come from scripts/run_baseline.py <task> --save; highD weights are
the committed checkpoints scored in datasets/highd/results/their_protocol_eval.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config                      # noqa: E402
from src.data import load_task                     # noqa: E402
from src.lc_windows import load_split, make_windows, norm_stats, apply_norm  # noqa: E402
from src.models import BaselineDSCNN               # noqa: E402

N_REF, N_CALIB = 256, 500


def keras_weights(model: BaselineDSCNN) -> dict[str, np.ndarray]:
    """PyTorch parameters in Keras layout, keyed by the layer names of deploy_hand_cnn.py."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    w = {"stem_conv": sd["stem.0.weight"].transpose(2, 1, 0)}
    for k, dst in (("weight", "gamma"), ("bias", "beta"), ("running_mean", "mean"), ("running_var", "var")):
        w[f"stem_bn_{dst}"] = sd[f"stem.1.{k}"]
    for i in range(len(model.blocks)):
        p = f"blocks.{i}.net"
        w[f"b{i}_dw"] = sd[f"{p}.0.weight"].transpose(2, 0, 1)      # (c, 1, k) -> (k, c, 1)
        w[f"b{i}_pw"] = sd[f"{p}.1.weight"].transpose(2, 1, 0)      # (out, in, 1) -> (1, in, out)
        for k, dst in (("weight", "gamma"), ("bias", "beta"), ("running_mean", "mean"), ("running_var", "var")):
            w[f"b{i}_bn_{dst}"] = sd[f"{p}.2.{k}"]
    w["dense_kernel"] = sd["head.3.weight"].T
    w["dense_bias"] = sd["head.3.bias"]
    return w


def reference(model: BaselineDSCNN, x_nlc: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(x_nlc.transpose(0, 2, 1)))).numpy()


def calib_sample(x_train: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    return x_train[rng.choice(len(x_train), size=min(N_CALIB, len(x_train)), replace=False)]


def save(out: Path, model: BaselineDSCNN, task: str, source: str, x_train, x_test, extras: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    x_ref = x_test[:N_REF]
    np.savez_compressed(out, task=task, source=source,
                        widths=np.array([32, 48, 64]), kernel=5,
                        n_features=x_test.shape[2], n_outputs=model.head[3].out_features,
                        params=sum(p.numel() for p in model.parameters()),
                        x_ref=x_ref, y_ref=reference(model, x_ref),
                        x_calib=calib_sample(x_train), x_test=x_test,
                        **extras, **keras_weights(model))
    print(f"{out.relative_to(ROOT)}: {x_test.shape[0]} test windows, "
          f"{sum(p.numel() for p in model.parameters())} parameters")


def export_lcir(task: str) -> None:
    ckpt = ROOT / "datasets/dmir/results/hand" / f"baseline-dscnn-deploy_{task}.pt"
    cfg = Config(task=task)
    bundle = load_task(cfg)
    model = BaselineDSCNN(n_outputs=cfg.n_outputs)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    save(ROOT / "datasets/dmir/data/hand_export" / f"lcir_dscnn_{task}.npz", model,
         "cls" if cfg.is_classification else "reg", str(ckpt.relative_to(ROOT)),
         bundle.x_train.astype(np.float32), bundle.x_test.astype(np.float32),
         {"y_test": bundle.y_test})


def export_highd(task: str) -> None:
    prep = ROOT / "datasets/highd/data/prepared"
    ckpt = ROOT / "datasets/highd/results" / f"highd_baseline_{task}_v2.pt"
    xtr, _, _ = make_windows(*load_split(prep, "train"))
    lo, hi = norm_stats(xtr)
    xte, yte, tte = make_windows(*load_split(prep, "test"))
    model = BaselineDSCNN(n_features=18, n_outputs=3 if task == "cls" else 1)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    save(ROOT / "datasets/highd/data/hand_export" / f"highd_dscnn_{task}.npz", model,
         "cls" if task == "cls" else "ttlc", str(ckpt.relative_to(ROOT)),
         apply_norm(xtr, lo, hi).astype(np.float32), apply_norm(xte, lo, hi).astype(np.float32),
         {"y_test": yte, "ttlc_test": tte.astype(np.float32)})


if __name__ == "__main__":
    for t in ("classification", "regression_lcr", "regression_lcl"):
        export_lcir(t)
    for t in ("cls", "ttlc"):
        export_highd(t)
