"""Evaluate our trained highD models with EarlyLCPred's EXACT metric definitions
(Materials/EarlyLCPred/utils.py), so the comparison against their Table III is
apples-to-apples:

- accuracy  = mean over the 26 slide positions of per-slide accuracy (equal
  slide counts make this identical to overall window accuracy)
- recall    = mean over slides of TP_t/(TP_t+FN_t); a hit needs the exact
  3-class label (RLC misread as LLC counts as FN)
- precision = mean over slides of TP_t/(TP_t+FP_t), where FP also counts a
  wrong-type LC (their double-count convention, Table II)
- F1        = harmonic mean of the two averages above
- AUC       = their 101-point threshold sweep on the LC-class softmax scores
  (a class counts as predicted when its prob >= thr; LK is the complement)
- tau_f     = avg time of the EARLIEST correct prediction before crossing
- tau_c     = avg ROBUST prediction time (ACCEPTED_GAP=0): predictions must
  stay correct from that point to the crossing
- RMSE      = sqrt(mean over slides of per-slide MSE) on LC scenarios, ground
  truth (26-i)/5 in [0.2, 5.2] s (their calc_regression_metrics)

Usage: .venv\\Scripts\\python datasets/highd/eval_their_protocol.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src.models.baseline import BaselineDSCNN                      # noqa: E402
from train_highd import (load_split, make_windows, norm_stats,      # noqa: E402
                         apply_norm, IN_LEN, SEQ_LEN, FPS, N_SLIDES)
from src.lc_windows import early_metrics                            # noqa: E402

FPS_F = float(FPS)


def predict(model, x, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 4096):
            xb = torch.from_numpy(x[i:i + 4096]).to(device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, *_ = (v for v in [*load_split("train")])
    tr_x, _, _ = make_windows(*load_split("train"))
    lo, hi = norm_stats(tr_x)

    feats, label, cross = load_split("test")
    S = len(label)
    x, y, _ = make_windows(feats, label, cross)
    xn = np.ascontiguousarray(
        apply_norm(x, lo, hi).astype(np.float32).transpose(0, 2, 1))

    cls = BaselineDSCNN(n_features=18, n_outputs=3).to(device)
    cls.load_state_dict(torch.load(HERE / "results" / "highd_baseline_cls_v2.pt",
                                   weights_only=True))
    logits = predict(cls, xn, device).reshape(S, N_SLIDES, 3)
    probs = torch.softmax(torch.from_numpy(logits), -1).numpy()
    preds = probs.argmax(-1)                      # (S, 26)
    labels = label.astype(np.int64)               # (S,)

    ttlc_model = BaselineDSCNN(n_features=18, n_outputs=1).to(device)
    ttlc_model.load_state_dict(torch.load(
        HERE / "results" / "highd_baseline_ttlc_v2.pt", weights_only=True))
    is_lc = labels != 0
    ttlc_pred = predict(ttlc_model, xn, device).reshape(S, N_SLIDES)[is_lc]
    m = early_metrics(probs, labels, ttlc_pred)
    accuracy, recall, precision, f1, auc = m["accuracy"], m["recall"], m["precision"], m["f1"], m["auc"]
    tau_f, tau_c, rmse, mean_mse = m["tau_f_s"], m["tau_c_s"], m["ttlc_rmse_s"], m["ttlc_mean_mse"]

    out = {"protocol": "EarlyLCPred utils.py, exact formulas",
           "test_scenarios": int(S), "lc_scenarios": int(is_lc.sum()),
           "accuracy": round(accuracy, 4), "recall": round(recall, 4),
           "precision": round(precision, 4), "f1": round(f1, 4),
           "auc": round(auc, 4),
           "tau_f_s": round(tau_f, 2), "tau_c_s": round(tau_c, 2),
           "ttlc_rmse_s": round(rmse, 4), "ttlc_mean_mse": round(mean_mse, 4),
           "paper_table3_proposed": {"accuracy": 0.83, "recall": 0.85,
                                     "precision": 0.85, "f1": 0.85, "auc": 0.88,
                                     "tau_f_s": 4.75, "tau_c_s": 3.96,
                                     "rmse_s": 0.629},
           "caveat": "our test split has 693 scenarios vs their 698 (0.7%); "
                     "train/val match their counts exactly (7487/932)"}
    print(json.dumps(out, indent=1))
    (HERE / "results" / "their_protocol_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
