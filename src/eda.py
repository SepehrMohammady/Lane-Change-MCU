"""Exploratory data analysis plots — clean, publication-ready defaults."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .data import SplitBundle

CLASS_NAMES = {0: "no intention", 1: "LCR intention", 2: "LCL intention"}

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def plot_label_distribution(bundle: SplitBundle, is_classification: bool):
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharey=is_classification)
    for ax, split in zip(axes, ("train", "val", "test")):
        y = getattr(bundle, f"y_{split}")
        if is_classification:
            vals, counts = np.unique(y, return_counts=True)
            ax.bar([CLASS_NAMES.get(int(v), str(v)) for v in vals], counts,
                   color="#4878A8")
            ax.tick_params(axis="x", rotation=20)
        else:
            ax.hist(y, bins=41, color="#4878A8")
            ax.set_xlabel("time to lane change [s]")
        ax.set_title(f"{split} (n={len(y):,})")
    fig.suptitle("Label distribution per split", y=1.04)
    fig.tight_layout()
    return fig


def plot_feature_ranges(bundle: SplitBundle):
    """Per-feature min/max/std on the training split."""
    x = bundle.x_train
    mins = x.min(axis=(0, 1))
    maxs = x.max(axis=(0, 1))
    stds = x.std(axis=(0, 1))
    idx = np.arange(x.shape[2])
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].fill_between(idx, mins, maxs, alpha=0.35, color="#4878A8",
                         label="train min-max")
    axes[0].set_ylabel("value range")
    axes[0].legend()
    axes[1].bar(idx, stds, color="#4878A8")
    axes[1].set_ylabel("std")
    axes[1].set_xlabel("feature index")
    axes[1].set_xticks(idx)
    fig.suptitle("Per-feature range and spread (train)")
    fig.tight_layout()
    return fig


def plot_example_windows(bundle: SplitBundle, is_classification: bool,
                         features: tuple[int, ...] = (0, 1, 2, 8), seed: int = 0):
    """One random training window per class (or per target tercile)."""
    rng = np.random.default_rng(seed)
    x, y = bundle.x_train, bundle.y_train
    if is_classification:
        groups = [(c, np.where(y == c)[0]) for c in np.unique(y)]
        titles = [CLASS_NAMES.get(int(c), str(c)) for c, _ in groups]
    else:
        edges = np.quantile(y, [0, 1 / 3, 2 / 3, 1])
        groups = [(i, np.where((y >= edges[i]) & (y <= edges[i + 1]))[0])
                  for i in range(3)]
        titles = [f"t in [{edges[i]:.1f}, {edges[i + 1]:.1f}] s" for i in range(3)]

    fig, axes = plt.subplots(1, len(groups), figsize=(11, 3), sharey=True)
    for ax, (_, pool), title in zip(axes, groups, titles):
        i = rng.choice(pool)
        for f in features:
            ax.plot(x[i, :, f], label=f"feat {f}", lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("timestep")
    axes[0].legend(fontsize=8)
    fig.suptitle("Example windows (train)", y=1.04)
    fig.tight_layout()
    return fig


def plot_history(history: dict, is_classification: bool):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
    name = "accuracy" if is_classification else "RMSE [s]"
    axes[1].plot(epochs, history["val_metric"], color="#B8544F")
    axes[1].axvline(history["best_epoch"], ls="--", c="gray", lw=1,
                    label=f"best (ep {history['best_epoch']})")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel(f"val {name}"); axes[1].legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names=None):
    cm = np.asarray(cm)
    names = class_names or [CLASS_NAMES[i] for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    return fig
