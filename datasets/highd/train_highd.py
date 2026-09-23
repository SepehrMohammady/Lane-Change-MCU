"""Baseline DSCNN on highD lane-change prediction (EarlyLCPred protocol).

Windows: input 10 frames (2 s @ 5 Hz) slid over each 35-frame scenario at all
26 positions. Classification: 3-class {LK, RLC, LLC} on every window; for LC
scenarios early windows are labeled with the upcoming manoeuvre, so this is the
early-prediction task (TTLC of a window ranges 0..5 s). Regression:
time-to-lane-change in seconds at window end, trained on LC windows only.

Normalization: per-feature min-max fitted on train (their Dataset.py rule).
The training loop is shared with exiD in src/lc_windows.py.

Usage: .venv\\Scripts\\python datasets/highd/train_highd.py <cls|ttlc> [run_name] [seed]
Logs to datasets/highd/logs/experiments.jsonl (same record shape as
src.log_utils.ExperimentLogger).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src import lc_windows as W                                     # noqa: E402
from src.lc_windows import (IN_LEN, SEQ_LEN, FPS, N_SLIDES,          # noqa: E402,F401
                            make_windows, norm_stats, apply_norm)

PREP = HERE / "data" / "prepared"
LOGS = HERE / "logs"
RESULTS = HERE / "results"
SEED = 42
DATASET, PROTOCOL = "highD", "EarlyLCPred T-IV 2022"


def load_split(split: str):
    return W.load_split(PREP, split)


def main(task: str, run_name: str, seed: int = SEED, save_weights: bool = True, log: bool = True):
    return W.train_dscnn(task, run_name, prep=PREP, logs=LOGS, results=RESULTS, dataset=DATASET,
                         protocol=PROTOCOL, seed=seed, save_weights=save_weights, log=log)


if __name__ == "__main__":
    # python train_highd.py <cls|ttlc> [run_name] [seed]
    # A seed other than the default is a variance run: metrics are logged, weights are not kept.
    task = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"highd_baseline_{task}"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else SEED
    main(task, run_name, seed=seed, save_weights=(seed == SEED))
