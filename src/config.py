"""Experiment configuration.

Every knob the pipeline exposes lives here. The notebook builds one
``Config`` instance in its configuration cell; all downstream modules
receive it explicitly — no hidden globals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TASKS = ("classification", "regression_lcl", "regression_lcr")

# Folder / file-suffix layout of the prepared dataset on disk.
TASK_LAYOUT = {
    "classification": ("data-classification", "multi"),
    "regression_lcl": ("data-regression-lcl", "lcl"),
    "regression_lcr": ("data-prepared-lcr", "lcr"),
}


@dataclass
class Config:
    # --- task -----------------------------------------------------------
    task: str = "classification"          # one of TASKS
    seed: int = 42

    # --- data -----------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    # Clip val/test features to the per-feature range observed on train.
    # Motivation: test splits contain a handful of extreme spikes
    # (|x| up to ~5e6 on feature pairs 12/13 and 14/15) never seen in
    # training; clipping keeps them in-distribution without touching
    # the training data. Documented in docs/DATA.md.
    clip_to_train_range: bool = True
    subset_fraction: float = 1.0          # <1.0 = stratified subset (smoke tests)

    # --- training -------------------------------------------------------
    batch_size: int = 256
    epochs: int = 60
    lr: float = 3e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    num_workers: int = 0                  # Windows: keep 0 unless benchmarked
    use_amp: bool = True                  # mixed precision on GPU

    # --- bookkeeping ----------------------------------------------------
    run_name: str = "baseline"
    log_dir: Path = PROJECT_ROOT / "logs"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        self.data_dir = Path(self.data_dir)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def n_outputs(self) -> int:
        return 3 if self.is_classification else 1

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def __repr__(self) -> str:  # compact, notebook-friendly
        return "Config(\n" + "\n".join(
            f"  {k} = {v!r}" for k, v in self.to_dict().items()
        ) + "\n)"
