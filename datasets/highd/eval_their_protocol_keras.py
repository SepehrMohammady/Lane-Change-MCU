"""Score a searched (Keras) highD classifier with EarlyLCPred's metric definitions.

datasets/highd/eval_their_protocol.py scores the hand-designed PyTorch networks; this
script does the same for a Keras model from the searches, so the published model,
the hand-designed CNN and the searched models sit in one table. Two steps, because
Keras lives in the WSL environment and the transcribed metrics (src/lc_windows.py)
import torch, which lives in the Windows venv:

  1. WSL   eval_their_protocol_keras.py probs <cls.h5> [<ttlc.h5>]
           softmax outputs on the 26 windows of every test scenario (and TTLC
           predictions on the lane-change scenarios), inputs min-max scaled with the
           training windows exactly as unas/prepare_deploy_highd.py does;
           written to data/eval/<stem>_probs.npz (local, derived from highD data)
  2. venv  eval_their_protocol_keras.py score <cls.h5 stem>
           early_metrics() of src/lc_windows.py on those outputs; written to
           results/their_protocol_eval_<stem>.json

WSL:  source ~/dmir_nas/env.sh; CUDA_VISIBLE_DEVICES= HIGHD_DATA_ROOT=.../datasets/highd/data/prepared \\
        ~/dmir_nas/bin/python datasets/highd/eval_their_protocol_keras.py probs <cls.h5> [<ttlc.h5>]
venv: .venv\\Scripts\\python datasets/highd/eval_their_protocol_keras.py score <stem>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREP = HERE / "data" / "prepared"
EVAL = HERE / "data" / "eval"
IN_LEN, SEQ_LEN, FPS = 10, 35, 5
N_SLIDES = SEQ_LEN - IN_LEN + 1


def windows(split):
    """Same slicing as unas/prepare_deploy_highd.py and src/lc_windows.make_windows."""
    z = np.load(PREP / f"{split}.npz")
    feats, label, cross = z["feats"], z["label"], z["cross_idx"]
    S = len(feats)
    sw = np.lib.stride_tricks.sliding_window_view(feats, IN_LEN, axis=1)
    x = np.ascontiguousarray(sw.transpose(0, 1, 3, 2)).reshape(S * N_SLIDES, IN_LEN, feats.shape[2])
    return x.astype(np.float32), label, cross


def main_probs(cls_h5: str, ttlc_h5: str | None = None) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import keras
    xtr, _, _ = windows("train")
    lo, hi = xtr.reshape(-1, 18).min(0), xtr.reshape(-1, 18).max(0)
    x, label, cross = windows("test")
    xn = ((x - lo) / (hi - lo + 1e-12)).astype(np.float32)
    S = len(label)
    m = keras.models.load_model(cls_h5, compile=False)
    logits = m.predict(xn, batch_size=4096, verbose=0).reshape(S, N_SLIDES, 3)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    out = {"probs": (e / e.sum(-1, keepdims=True)).astype(np.float32), "label": label}
    if ttlc_h5:
        r = keras.models.load_model(ttlc_h5, compile=False)
        is_lc = label != 0
        out["ttlc_pred"] = r.predict(xn, batch_size=4096, verbose=0).reshape(S, N_SLIDES)[is_lc]
    EVAL.mkdir(parents=True, exist_ok=True)
    stem = Path(cls_h5).stem
    np.savez(EVAL / f"{stem}_probs.npz", **out, cls_h5=Path(cls_h5).name,
             ttlc_h5=Path(ttlc_h5).name if ttlc_h5 else "")
    print(f"{stem}: {S} test scenarios -> {EVAL / (stem + '_probs.npz')}")


def main_score(stem: str) -> None:
    sys.path.insert(0, str(ROOT))
    from src.lc_windows import early_metrics
    z = np.load(EVAL / f"{stem}_probs.npz")
    ttlc = z["ttlc_pred"] if "ttlc_pred" in z.files else None
    m = early_metrics(z["probs"], z["label"], ttlc)
    out = {"model": stem, "source_h5": str(z["cls_h5"]), "protocol": "EarlyLCPred utils.py, exact formulas",
           "test_scenarios": int(len(z["label"])),
           "accuracy": round(m["accuracy"], 4), "recall": round(m["recall"], 4),
           "precision": round(m["precision"], 4), "f1": round(m["f1"], 4), "auc": round(m["auc"], 4),
           "tau_f_s": round(m["tau_f_s"], 2), "tau_c_s": round(m["tau_c_s"], 2)}
    if ttlc is not None:
        out.update(ttlc_source_h5=str(z["ttlc_h5"]), ttlc_rmse_s=round(m["ttlc_rmse_s"], 4))
    (HERE / "results" / f"their_protocol_eval_{stem}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "probs":
        main_probs(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif sys.argv[1] == "score":
        main_score(sys.argv[2])
    else:
        sys.exit("usage: eval_their_protocol_keras.py probs <cls.h5> [<ttlc.h5>] | score <stem>")
