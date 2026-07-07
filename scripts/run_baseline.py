"""Train the baseline DSCNN on one task with full data and log the run.

Usage:  python scripts/run_baseline.py <task>
        task in {classification, regression_lcl, regression_lcr}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data import load_task, make_loaders
from src.env_utils import pick_device, seed_everything
from src.log_utils import ExperimentLogger
from src.models import BaselineDSCNN
from src.train import evaluate, train_model


def main(task: str) -> None:
    cfg = Config(task=task, run_name="baseline-dscnn",
                 notes="depthwise-separable CNN baseline")
    seed_everything(cfg.seed)
    device = pick_device()
    bundle = load_task(cfg)
    print(bundle.summary())
    loaders = make_loaders(cfg, bundle)
    model = BaselineDSCNN(n_outputs=cfg.n_outputs)
    logger = ExperimentLogger(cfg)
    history = train_model(cfg, model, loaders, device)
    test = evaluate(cfg, model, loaders["test"], device)
    logger.log_metrics(best_epoch=history["best_epoch"],
                       best_val_metric=history["best_val_metric"],
                       train_time_s=history["train_time_s"],
                       **{f"test_{k}": v for k, v in test.items()
                          if k != "confusion_matrix"})
    print("test:", {k: round(v, 4) for k, v in test.items()
                    if isinstance(v, float)})
    print("logged ->", logger.finish())


if __name__ == "__main__":
    main(sys.argv[1])
