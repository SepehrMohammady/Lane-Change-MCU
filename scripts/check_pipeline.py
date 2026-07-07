"""Pipeline smoke test — run after every code change.

Exercises the full chain on a small stratified subset of the REAL data
for all three tasks: load -> sanitise -> loaders -> model -> train
(2 epochs) -> evaluate -> log. Fails loudly on any breakage.

Usage:  python scripts/check_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import Config, TASKS
from src.data import load_task, make_loaders
from src.env_utils import env_report, pick_device, seed_everything
from src.log_utils import ExperimentLogger
from src.models import BaselineDSCNN
from src.train import evaluate, train_model


def main() -> None:
    print("env:", env_report())
    device = pick_device()
    failures = []

    for task in TASKS:
        print(f"\n--- smoke test: {task} ---")
        cfg = Config(
            task=task,
            subset_fraction=0.02,
            epochs=2,
            early_stop_patience=5,
            batch_size=128,
            run_name=f"smoke-{task}",
            notes="automated pipeline check",
        )
        try:
            seed_everything(cfg.seed)
            bundle = load_task(cfg)
            print(bundle.summary())
            loaders = make_loaders(cfg, bundle)
            model = BaselineDSCNN(n_outputs=cfg.n_outputs)
            logger = ExperimentLogger(cfg)
            history = train_model(cfg, model, loaders, device, verbose=False)
            test = evaluate(cfg, model, loaders["test"], device)
            key = "accuracy" if cfg.is_classification else "rmse"
            print(f"OK  best_val={history['best_val_metric']:.4f}  "
                  f"test_{key}={test[key]:.4f}  time={history['train_time_s']}s")
            logger.log_metrics(smoke=True, **{f"test_{key}": test[key]})
            logger.finish()
        except Exception as e:  # noqa: BLE001 — report every task's failure
            failures.append((task, repr(e)))
            print(f"FAIL  {e!r}")

    print("\n" + "=" * 50)
    if failures:
        print("PIPELINE CHECK FAILED:")
        for task, err in failures:
            print(f"  {task}: {err}")
        sys.exit(1)
    print("PIPELINE CHECK PASSED (all tasks)")


if __name__ == "__main__":
    main()
