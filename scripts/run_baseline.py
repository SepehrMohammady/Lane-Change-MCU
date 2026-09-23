"""Train the baseline DSCNN on one task with full data and log the run.

Usage:  python scripts/run_baseline.py <task> [seed] [--save]
        task in {classification, regression_lcl, regression_lcr}
        seed    optional; runs other than the default seed are logged as
                baseline-dscnn-seed<n> (seed study, see nas-results.md)
        --save  keep the trained weights for deployment, as
                datasets/dmir/results/hand/<run_name>_<task>.pt; the run is
                logged as baseline-dscnn-deploy (default seed)
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

import torch  # noqa: E402


def main(task: str, seed: int | None = None, save: bool = False) -> None:
    cfg = Config(task=task, run_name="baseline-dscnn",
                 notes="depthwise-separable CNN baseline")
    if save:
        cfg.run_name = "baseline-dscnn-deploy"
        cfg.notes = "depthwise-separable CNN baseline, weights kept for the board runs"
    if seed is not None:
        cfg.seed = seed
        cfg.run_name = f"baseline-dscnn-seed{seed}"
        cfg.notes = "depthwise-separable CNN baseline, seed study"
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
    if save:
        out = Path(__file__).resolve().parent.parent / "datasets" / "dmir" / "results" / "hand"
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / f"{cfg.run_name}_{task}.pt")
        print("weights ->", out / f"{cfg.run_name}_{task}.pt")
    print("logged ->", logger.finish())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0], int(args[1]) if len(args) > 1 else None, save="--save" in sys.argv)
