"""Training and evaluation loops for classification and regression."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config


def _forward_loss(model, x, y, criterion, is_classification):
    out = model(x)
    if is_classification:
        return criterion(out, y), out
    return criterion(out.squeeze(-1), y), out


def train_model(cfg: Config, model: nn.Module, loaders: dict[str, DataLoader],
                device: torch.device, verbose: bool = True) -> dict:
    """Train with early stopping on validation metric; restore best weights.

    Returns a history dict with per-epoch losses/metrics and timing.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss() if cfg.is_classification else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "val_metric": [], "lr": []}
    best_val = -math.inf if cfg.is_classification else math.inf
    best_state, best_epoch = None, -1
    t0 = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss, _ = _forward_loss(model, x, y, criterion, cfg.is_classification)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(x)
            n += len(x)
        sched.step()

        val = evaluate(cfg, model, loaders["val"], device)
        train_loss = running / n
        metric = val["accuracy"] if cfg.is_classification else val["rmse"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_metric"].append(metric)
        history["lr"].append(sched.get_last_lr()[0])

        improved = metric > best_val if cfg.is_classification else metric < best_val
        if improved:
            best_val, best_epoch = metric, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            name = "acc" if cfg.is_classification else "rmse"
            star = " *" if improved else ""
            print(f"epoch {epoch + 1:3d}/{cfg.epochs}  train_loss {train_loss:.4f}  "
                  f"val_loss {val['loss']:.4f}  val_{name} {metric:.4f}{star}")
        if epoch - best_epoch >= cfg.early_stop_patience:
            if verbose:
                print(f"early stop at epoch {epoch + 1} (best epoch {best_epoch + 1})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch + 1
    history["best_val_metric"] = float(best_val)
    history["train_time_s"] = round(time.perf_counter() - t0, 1)
    return history


@torch.no_grad()
def evaluate(cfg: Config, model: nn.Module, loader: DataLoader,
             device: torch.device) -> dict:
    """Loss + task metrics on one loader.

    classification -> accuracy, macro-F1, per-class recall, confusion matrix
    regression     -> RMSE, MAE
    """
    model.eval()
    criterion = nn.CrossEntropyLoss() if cfg.is_classification else nn.MSELoss()
    losses, n = 0.0, 0
    preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss, out = _forward_loss(model, x, y, criterion, cfg.is_classification)
        losses += loss.item() * len(x)
        n += len(x)
        preds.append(out.float().cpu().numpy())
        targets.append(y.cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(targets)
    res = {"loss": losses / n}

    if cfg.is_classification:
        yhat = p.argmax(axis=1)
        res["accuracy"] = float((yhat == t).mean())
        n_cls = p.shape[1]
        cm = np.zeros((n_cls, n_cls), dtype=int)
        for a, b in zip(t, yhat):
            cm[a, b] += 1
        res["confusion_matrix"] = cm.tolist()
        f1s, recalls = [], []
        for c in range(n_cls):
            tp = cm[c, c]
            prec = tp / max(cm[:, c].sum(), 1)
            rec = tp / max(cm[c, :].sum(), 1)
            f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
            recalls.append(rec)
        res["macro_f1"] = float(np.mean(f1s))
        res["per_class_recall"] = [round(float(r), 4) for r in recalls]
    else:
        yhat = p.squeeze(-1)
        res["rmse"] = float(np.sqrt(np.mean((yhat - t) ** 2)))
        res["mae"] = float(np.mean(np.abs(yhat - t)))
    return res
