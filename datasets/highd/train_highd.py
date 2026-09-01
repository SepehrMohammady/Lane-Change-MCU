"""Baseline DSCNN on highD lane-change prediction (EarlyLCPred protocol).

Windows: input 10 frames (2 s @ 5 Hz) slid over each 35-frame scenario at all
26 positions. Classification: 3-class {LK, RLC, LLC} on every window — for LC
scenarios early windows are labeled with the upcoming manoeuvre, so this IS
the early-prediction task (TTLC of a window ranges 0..5 s). Regression:
time-to-lane-change in seconds at window end, trained on LC windows only.

Normalization: per-feature min-max fitted on train (their Dataset.py rule).

Usage: .venv\\Scripts\\python datasets/highd/train_highd.py <cls|ttlc> [run_name]
Logs to datasets/highd/logs/experiments.jsonl (same record shape as
src.log_utils.ExperimentLogger).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.baseline import BaselineDSCNN          # noqa: E402
from src.env_utils import env_report, seed_everything  # noqa: E402

PREP = HERE / "data" / "prepared"
LOGS = HERE / "logs"
RESULTS = HERE / "results"

IN_LEN, SEQ_LEN, FPS = 10, 35, 5
N_SLIDES = SEQ_LEN - IN_LEN + 1        # 26
SEED = 42


def load_split(split: str):
    z = np.load(PREP / f"{split}.npz")
    return z["feats"], z["label"], z["cross_idx"]


def make_windows(feats, label, cross):
    """(S,35,18) -> windows x (S*26,10,18), y class, ttlc seconds (NaN if n/a)."""
    S = len(feats)
    sw = np.lib.stride_tricks.sliding_window_view(feats, IN_LEN, axis=1)
    # sliding_window_view gives (S, 26, 18, 10) -> (S, 26, 10, 18)
    x = np.ascontiguousarray(sw.transpose(0, 1, 3, 2)).reshape(S * N_SLIDES, IN_LEN, feats.shape[2])
    y = np.repeat(label, N_SLIDES)
    s_idx = np.tile(np.arange(N_SLIDES), S)
    cx = np.repeat(cross.astype(np.float32), N_SLIDES)
    # their exact label convention (utils.py train_model): TTLC measured from
    # the LAST INPUT FRAME, i.e. (SEQ_LEN - s - IN_SEQ_LEN + 1)/FPS in [0.2, 5.2]
    ttlc = (cx - (s_idx + IN_LEN) + 1) / FPS
    ttlc[cx < 0] = np.nan
    return x, y, ttlc


def norm_stats(x):
    flat = x.reshape(-1, x.shape[2])
    return flat.min(0), flat.max(0)


def apply_norm(x, lo, hi):
    return (x - lo) / (hi - lo + 1e-12)


def batches(*arrays, bs, shuffle, device):
    n = len(arrays[0])
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield [torch.from_numpy(a[j]).to(device) for a in arrays]


def main(task: str, run_name: str):
    assert task in ("cls", "ttlc")
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    record = {"run_name": run_name,
              "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": {"dataset": "highD", "protocol": "EarlyLCPred T-IV 2022",
                         "task": task, "model": "BaselineDSCNN",
                         "in_len": IN_LEN, "n_features": 18, "seed": SEED,
                         "widths": [32, 48, 64], "lr": 3e-3, "batch_size": 256,
                         "epochs": 60, "early_stop_patience": 8,
                         "normalization": "minmax-train"},
              "env": env_report(), "metrics": {}}

    data = {}
    for split in ("train", "val", "test"):
        x, y, ttlc = make_windows(*load_split(split))
        data[split] = (x, y, ttlc)
    lo, hi = norm_stats(data["train"][0])

    if task == "cls":
        sel = {s: np.ones(len(data[s][0]), bool) for s in data}
        n_out = 3
    else:  # ttlc regression: LC windows only (label != 0 -> ttlc in [0,5])
        sel = {s: (data[s][1] != 0) for s in data}
        n_out = 1

    tensors = {}
    for s in data:
        x, y, ttlc = data[s]
        m = sel[s]
        xs = apply_norm(x[m], lo, hi).astype(np.float32).transpose(0, 2, 1)  # (N,C,T)
        ys = y[m].astype(np.int64) if task == "cls" else ttlc[m].astype(np.float32)
        tensors[s] = (np.ascontiguousarray(xs), ys)
        print(f"{s}: {len(ys):,} windows", flush=True)

    model = BaselineDSCNN(n_features=18, n_outputs=n_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    record["config"]["n_params"] = n_params
    print(f"params: {n_params:,}  device: {device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss() if task == "cls" else nn.L1Loss()

    def evaluate(split):
        model.eval()
        xs, ys = tensors[split]
        preds = []
        with torch.no_grad():
            for (xb,) in batches(xs, bs=2048, shuffle=False, device=device):
                preds.append(model(xb).cpu().numpy())
        p = np.concatenate(preds)
        if task == "cls":
            pred = p.argmax(1)
            acc = float((pred == ys).mean())
            f1 = []
            for c in range(3):
                tp = ((pred == c) & (ys == c)).sum()
                fp = ((pred == c) & (ys != c)).sum()
                fn = ((pred != c) & (ys == c)).sum()
                f1.append(2 * tp / max(2 * tp + fp + fn, 1))
            return {"acc": acc, "macro_f1": float(np.mean(f1))}, pred
        else:
            err = p.squeeze(1) - ys
            return {"mae": float(np.abs(err).mean()),
                    "rmse": float(np.sqrt((err ** 2).mean()))}, p.squeeze(1)

    key, mode = ("acc", 1) if task == "cls" else ("mae", -1)
    best, best_state, patience = -np.inf, None, 0
    for epoch in range(60):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in batches(*tensors["train"], bs=256, shuffle=True, device=device):
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out if task == "cls" else out.squeeze(1), yb)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        vm, _ = evaluate("val")
        score = mode * vm[key]
        print(f"epoch {epoch:2d}  train_loss {tot/nb:.4f}  val {vm}", flush=True)
        if score > best:
            best, patience = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break
    model.load_state_dict(best_state)

    vm, _ = evaluate("val")
    tm, test_pred = evaluate("test")
    record["metrics"] = {f"val_{k}": v for k, v in vm.items()}
    record["metrics"].update({f"test_{k}": v for k, v in tm.items()})

    # early-prediction breakdown on test: metric vs TTLC bucket (LC windows)
    x, y, ttlc = data["test"]
    m = sel["test"]
    ttlc_sel, y_sel = ttlc[m], y[m]
    lc = ~np.isnan(ttlc_sel) & (y_sel != 0) if task == "cls" else np.ones(len(ttlc_sel), bool)
    per_ttlc = {}
    for lo_t in range(0, 5):
        b = lc & (ttlc_sel >= lo_t) & (ttlc_sel < lo_t + 1)
        if b.sum() == 0:
            continue
        if task == "cls":
            per_ttlc[f"{lo_t}-{lo_t+1}s"] = {"acc": float((test_pred[b] == y_sel[b]).mean()),
                                             "n": int(b.sum())}
        else:
            e = test_pred[b] - ttlc_sel[b]
            per_ttlc[f"{lo_t}-{lo_t+1}s"] = {"rmse": float(np.sqrt((e**2).mean())),
                                             "n": int(b.sum())}
    record["metrics"]["test_by_ttlc"] = per_ttlc

    record["duration_s"] = round(time.perf_counter() - t0, 1)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RESULTS / f"{run_name}.pt")
    print(json.dumps(record["metrics"], indent=1), flush=True)


if __name__ == "__main__":
    task = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"highd_baseline_{task}"
    main(task, run_name)
