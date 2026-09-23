"""Control for the highD -> exiD zero-shot result: same weights, exiD normalization.

run_experiments.py applies the highD models to exiD with the highD min-max
statistics, as a deployed highD model would. exiD values fall partly outside
highD's ranges (wider lanes, slower traffic), so part of the drop could come from
input scaling rather than from different driving. Here the same five highD
models see exiD test windows scaled with the exiD training statistics, without
any retraining. Appends method "zero-shot, exiD scaling" to results/transfer.jsonl.

Run: .venv\\Scripts\\python datasets/exid/zero_shot_renorm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

import run_experiments as R                                         # noqa: E402

METHOD = "zero-shot, exiD scaling"


def main():
    skip = R.done()
    tests = {"exiD": R.test_split(R.EXID)}
    lo, hi = R.norm_of(R.EXID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for task in ("cls", "ttlc"):
        for seed in R.SEEDS:
            src = R.HIGHD["weights"] / f"highd_source_{task}_seed{seed}.pt"
            R.evaluate_all(R.load_model(src, task, device), device, task, lo, hi,
                           {"method": METHOD, "trained_on": "highD", "seed": seed,
                            "normalization": "minmax-train of exiD"}, tests, skip)


if __name__ == "__main__":
    main()
