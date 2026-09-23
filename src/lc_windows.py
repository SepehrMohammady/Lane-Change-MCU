"""Lane-change prediction on scenario windows, shared by the highD and exiD pipelines.

Both datasets are prepared in the scenario format of Mozaffari et al. (T-IV 2022,
EarlyLCPred): 35 frames at 5 Hz per scenario, 18 features per frame. Windows of
10 frames are taken at all 26 positions. A window carries the scenario class
(0 lane keep, 1 right change, 2 left change) and, for lane changes, the time to
lane change measured from its last frame, (cross_idx - (s + 10) + 1) / 5 s,
which lies in [0.2, 5.2] s.

This module holds the window slicing, the min-max normalization fitted on the
training split, the training loop of the hand-designed DSCNN, and the
early-prediction metrics transcribed from their utils.py. The per-dataset
scripts (datasets/highd/train_highd.py, datasets/exid/train_exid.py) only set
paths and labels.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.env_utils import env_report, seed_everything
from src.models.baseline import BaselineDSCNN

IN_LEN, SEQ_LEN, FPS = 10, 35, 5
N_SLIDES = SEQ_LEN - IN_LEN + 1        # 26


# ------------------------------------------------------------------ data
def load_split(prep: Path, split: str, extras: tuple[str, ...] = ()):
    """feats, label, cross_idx of one split, plus the requested extra arrays."""
    z = np.load(Path(prep) / f"{split}.npz")
    base = (z["feats"], z["label"], z["cross_idx"])
    if not extras:
        return base
    return base + ({k: z[k] for k in extras if k in z.files},)


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


def predict(model, x, device, bs: int = 4096) -> np.ndarray:
    """Raw model outputs for windows already normalized and laid out as (N, C, T)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            outs.append(model(torch.from_numpy(x[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(outs)


def to_model_input(x, lo, hi):
    return np.ascontiguousarray(apply_norm(x, lo, hi).astype(np.float32).transpose(0, 2, 1))


# ------------------------------------------------------------------ metrics of Mozaffari et al.
def early_metrics(probs: np.ndarray, labels: np.ndarray, ttlc_pred: np.ndarray | None = None) -> dict:
    """Their classification metrics, AUC sweep, prediction times and TTLC RMSE.

    probs: (S, 26, 3) softmax outputs per scenario and slide; labels: (S,) scenario
    classes; ttlc_pred: (n_lc, 26) predicted TTLC for the lane-change scenarios in
    scenario order, or None.
    """
    S, T = probs.shape[:2]
    preds = probs.argmax(-1)
    labels = labels.astype(np.int64)
    hits = preds == labels[:, None]
    is_lc = labels != 0

    # calc_classification_metrics
    acc_t = hits.mean(0)
    tp_t = (hits & is_lc[:, None]).sum(0)
    fn_t = (~hits & is_lc[:, None]).sum(0)
    fp_t = ((~hits) & (~is_lc)[:, None]).sum(0) + ((~hits) & is_lc[:, None] & (preds != 0)).sum(0)
    recall = float((tp_t / np.maximum(tp_t + fn_t, 1)).mean())
    precision = float((tp_t / np.maximum(tp_t + fp_t, 1)).mean())
    out = {"accuracy": float(acc_t.mean()), "recall": recall, "precision": precision,
           "f1": 2 * precision * recall / (precision + recall)}

    # AUC: threshold sweep over the lane-change class probabilities
    tpr_v, fpr_v = np.zeros(101), np.zeros(101)
    n = S * T
    for i, thr in enumerate(np.arange(0, 101) / 100):
        m = probs >= thr
        lk = ~(m[:, :, 1] | m[:, :, 2])
        sc = np.stack([np.where(lk, probs[:, :, 0], -1.0),
                       np.where(m[:, :, 1], probs[:, :, 1], 0.0),
                       np.where(m[:, :, 2], probs[:, :, 2], 0.0)], -1)
        pr = sc.argmax(-1)
        h = pr == labels[:, None]
        tp = (h & is_lc[:, None]).sum() / n
        fn = (~h & is_lc[:, None]).sum() / n
        fp = (((~h) & (~is_lc)[:, None]).sum() + ((~h) & is_lc[:, None] & (pr != 0)).sum()) / n
        tn = (h & (~is_lc)[:, None]).sum() / n
        tpr_v[i] = tp / max(tp + fn, 1e-12)
        fpr_v[i] = fp / max(fp + tn, 1e-12)
    order = np.argsort(fpr_v)
    out["auc"] = float(np.trapezoid(tpr_v[order], fpr_v[order]))

    # calc_avg_pred_time (ACCEPTED_GAP = 0)
    r = np.flip((hits & is_lc[:, None])[is_lc], 1)      # index 0 = closest to the crossing
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
    out["tau_c_s"] = float(first_false.mean() / FPS)
    out["tau_f_s"] = float(last_true.mean() / FPS)

    # calc_regression_metrics
    if ttlc_pred is not None:
        gt = (T - np.arange(T)) / float(FPS)            # 5.2 .. 0.2
        mse_t = ((ttlc_pred - gt[None, :]) ** 2).mean(0)
        out["ttlc_rmse_s"] = float(np.sqrt(mse_t.mean()))
        out["ttlc_mean_mse"] = float(mse_t.mean())
    return out


def evaluate_scenarios(model, feats, label, cross, lo, hi, task: str, device,
                       groups: np.ndarray | None = None, group_names: list[str] | None = None) -> dict:
    """Window metrics, the early-prediction metrics of Mozaffari et al. and a per-group breakdown.

    feats (S,35,18) are raw features, normalized here with (lo, hi). groups is an
    optional per-scenario code (e.g. exiD lane-change kind) for the breakdown.
    """
    x, y, ttlc = make_windows(feats, label, cross)
    S = len(label)
    out = predict(model, to_model_input(x, lo, hi), device)
    res: dict = {}
    if task == "cls":
        pred = out.argmax(1)
        res["acc"] = float((pred == y).mean())
        f1 = []
        for c in range(3):
            tp = ((pred == c) & (y == c)).sum()
            fp = ((pred == c) & (y != c)).sum()
            fn = ((pred != c) & (y == c)).sum()
            f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        res["macro_f1"] = float(np.mean(f1))
        probs = torch.softmax(torch.from_numpy(out.reshape(S, N_SLIDES, 3)), -1).numpy()
        res["early"] = early_metrics(probs, label)
        hit = (pred == y).reshape(S, N_SLIDES)
    else:
        lc = label != 0
        pred = out.reshape(S, N_SLIDES)[lc]
        gt = ttlc.reshape(S, N_SLIDES)[lc]
        err = pred - gt
        res["mae"] = float(np.abs(err).mean())
        res["rmse"] = float(np.sqrt((err ** 2).mean()))
    if groups is not None:
        by = {}
        for g in np.unique(groups):
            name = group_names[g] if group_names else str(int(g))
            sel = groups == g
            if task == "cls":
                by[name] = {"acc": float(hit[sel].mean()), "scenarios": int(sel.sum())}
            else:
                sel_lc = sel[label != 0]
                if sel_lc.sum():
                    e = err[sel_lc]
                    by[name] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                                "scenarios": int(sel_lc.sum())}
        res["by_group"] = by
    return res


# ------------------------------------------------------------------ training
def train_dscnn(task: str, run_name: str, *, prep: Path, logs: Path, results: Path | None,
                dataset: str, protocol: str, seed: int, save_weights: bool = False, log: bool = True,
                init_state: dict | None = None, norm: tuple[np.ndarray, np.ndarray] | None = None,
                lr: float = 3e-3, train_fraction: float = 1.0, extra_config: dict | None = None,
                verbose: bool = True) -> dict:
    """Train the hand-designed DSCNN on one task and evaluate it on val and test.

    init_state starts from given weights (fine-tuning); norm replaces the
    min-max statistics of this dataset's training split (e.g. the source
    dataset's when fine-tuning); train_fraction keeps a random share of the
    training scenarios. Returns the log record, the model, the normalization
    and the test predictions.
    """
    assert task in ("cls", "ttlc")
    seed_everything(seed)
    # deterministic cuDNN kernels: a repeated seed repeats the run (runs logged
    # before 2026-09-23 used the default kernels and do not repeat exactly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    config = {"dataset": dataset, "protocol": protocol,
              "task": task, "model": "BaselineDSCNN",
              "in_len": IN_LEN, "n_features": 18, "seed": seed,
              "widths": [32, 48, 64], "lr": lr, "batch_size": 256,
              "epochs": 60, "early_stop_patience": 8,
              "normalization": "minmax-train", "cudnn_deterministic": True}
    config.update(extra_config or {})
    record = {"run_name": run_name,
              "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": config, "env": env_report(), "metrics": {}}

    data = {}
    for split in ("train", "val", "test"):
        feats, label, cross = load_split(prep, split)
        if split == "train" and train_fraction < 1.0:
            keep = np.random.RandomState(seed).rand(len(label)) < train_fraction
            feats, label, cross = feats[keep], label[keep], cross[keep]
            config["train_fraction"] = train_fraction
            config["train_scenarios"] = int(keep.sum())
        data[split] = make_windows(feats, label, cross)
    lo, hi = norm if norm is not None else norm_stats(data["train"][0])

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
        if verbose:
            print(f"{s}: {len(ys):,} windows", flush=True)

    model = BaselineDSCNN(n_features=18, n_outputs=n_out).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    n_params = sum(p.numel() for p in model.parameters())
    config["n_params"] = n_params
    if verbose:
        print(f"params: {n_params:,}  device: {device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
            return {"acc": acc, "macro_f1": float(np.mean(f1))}, p
        err = p.squeeze(1) - ys
        return {"mae": float(np.abs(err).mean()),
                "rmse": float(np.sqrt((err ** 2).mean()))}, p.squeeze(1)

    key, mode = ("acc", 1) if task == "cls" else ("mae", -1)
    best, best_state, patience, epochs_run = -np.inf, None, 0, 0
    for epoch in range(60):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in batches(*tensors["train"], bs=256, shuffle=True, device=device):
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out if task == "cls" else out.squeeze(1), yb)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        epochs_run = epoch + 1
        vm, _ = evaluate("val")
        score = mode * vm[key]
        if verbose:
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
    tm, test_out = evaluate("test")
    record["metrics"] = {f"val_{k}": v for k, v in vm.items()}
    record["metrics"].update({f"test_{k}": v for k, v in tm.items()})

    # early-prediction breakdown on test: metric vs TTLC bucket (LC windows)
    x, y, ttlc = data["test"]
    m = sel["test"]
    ttlc_sel, y_sel = ttlc[m], y[m]
    test_pred = test_out.argmax(1) if task == "cls" else test_out
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
    record["metrics"]["epochs_run"] = epochs_run

    record["duration_s"] = round(time.perf_counter() - t0, 1)
    if log:
        Path(logs).mkdir(parents=True, exist_ok=True)
        with open(Path(logs) / "experiments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if save_weights and results is not None:
        Path(results).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(results) / f"{run_name}.pt")
    if verbose:
        print(json.dumps(record["metrics"], indent=1), flush=True)
    return {"record": record, "model": model, "norm": (lo, hi), "test_out": test_out,
            "test_mask": m, "device": device}
