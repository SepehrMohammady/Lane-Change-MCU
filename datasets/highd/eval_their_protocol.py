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

    T = N_SLIDES
    hits = preds == labels[:, None]
    is_lc = labels != 0

    # --- their calc_classification_metrics -----------------------------------
    acc_t = hits.mean(0)
    tp_t = (hits & is_lc[:, None]).sum(0)
    fn_t = (~hits & is_lc[:, None]).sum(0)
    fp_t = ((~hits) & (~is_lc)[:, None]).sum(0) + \
           ((~hits) & is_lc[:, None] & (preds != 0)).sum(0)
    recall_t = tp_t / np.maximum(tp_t + fn_t, 1)
    prec_t = tp_t / np.maximum(tp_t + fp_t, 1)
    accuracy = float(acc_t.mean())
    recall = float(recall_t.mean())
    precision = float(prec_t.mean())
    f1 = 2 * precision * recall / (precision + recall)

    # --- their AUC: threshold sweep over LC-class probabilities --------------
    thr_range = np.arange(0, 101) / 100
    tpr_v, fpr_v = np.zeros(101), np.zeros(101)
    n = S * T
    for i, thr in enumerate(thr_range):
        m = probs >= thr                          # (S,T,3)
        lk = ~(m[:, :, 1] | m[:, :, 2])
        sc = np.stack([np.where(lk, probs[:, :, 0], -1.0),
                       np.where(m[:, :, 1], probs[:, :, 1], 0.0),
                       np.where(m[:, :, 2], probs[:, :, 2], 0.0)], -1)
        pr = sc.argmax(-1)
        h = pr == labels[:, None]
        tp = (h & is_lc[:, None]).sum() / n
        fn = (~h & is_lc[:, None]).sum() / n
        fp = (((~h) & (~is_lc)[:, None]).sum() +
              ((~h) & is_lc[:, None] & (pr != 0)).sum()) / n
        tn = (h & (~is_lc)[:, None]).sum() / n
        tpr_v[i] = tp / max(tp + fn, 1e-12)
        fpr_v[i] = fp / max(fp + tn, 1e-12)
    order = np.argsort(fpr_v)
    auc = float(np.trapezoid(tpr_v[order], fpr_v[order]))

    # --- their calc_avg_pred_time (ACCEPTED_GAP=0) ---------------------------
    tps_lc = (hits & is_lc[:, None])[is_lc]       # (n_lc, T)
    r = np.flip(tps_lc, 1)                        # index 0 = closest to crossing
    n_lc = len(r)
    first_false = np.full(n_lc, T, dtype=float)
    last_true = np.zeros(n_lc)
    for i in range(n_lc):
        nz = np.nonzero(r[i])[0]
        if len(nz):
            last_true[i] = nz[-1] + 1
        ff = np.nonzero(~r[i])[0]
        if len(ff):
            first_false[i] = ff[0]
    tau_c = float(first_false.mean() / FPS_F)
    tau_f = float(last_true.mean() / FPS_F)

    # --- their calc_regression_metrics ---------------------------------------
    ttlc_model = BaselineDSCNN(n_features=18, n_outputs=1).to(device)
    ttlc_model.load_state_dict(torch.load(
        HERE / "results" / "highd_baseline_ttlc_v2.pt", weights_only=True))
    ttlc_pred = predict(ttlc_model, xn, device).reshape(S, N_SLIDES)[is_lc]
    gt = (T - np.arange(T)) / FPS_F               # 5.2 .. 0.2
    mse_t = ((ttlc_pred - gt[None, :]) ** 2).mean(0)
    rmse = float(np.sqrt(mse_t.mean()))
    mean_mse = float(mse_t.mean())

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
