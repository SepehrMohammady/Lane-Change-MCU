"""Hand-designed DSCNN on exiD lane-change prediction (highD scenario format).

Same windows, model and recipe as datasets/highd/train_highd.py (shared code in
src/lc_windows.py); only the data differ. See prepare_exid.py for the protocol.

Usage: .venv\\Scripts\\python datasets/exid/train_exid.py <cls|ttlc> [run_name] [seed]
Logs to datasets/exid/logs/experiments.jsonl; weights to datasets/exid/results/weights/.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src import lc_windows as W                                     # noqa: E402

PREP = HERE / "data" / "prepared"
LOGS = HERE / "logs"
WEIGHTS = HERE / "results" / "weights"
SEED = 42
DATASET, PROTOCOL = "exiD", "EarlyLCPred T-IV 2022, adapted to exiD (datasets/exid/prepare_exid.py)"


def main(task: str, run_name: str, seed: int = SEED, save_weights: bool = True, log: bool = True, **kw):
    return W.train_dscnn(task, run_name, prep=PREP, logs=LOGS, results=WEIGHTS, dataset=DATASET,
                         protocol=PROTOCOL, seed=seed, save_weights=save_weights, log=log, **kw)


if __name__ == "__main__":
    task = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) > 2 else f"exid_baseline_{task}"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else SEED
    main(task, run_name, seed=seed)
