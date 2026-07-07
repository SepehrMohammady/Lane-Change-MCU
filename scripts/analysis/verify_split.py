"""Evidence: the prepared pickles follow the official driver-wise split.

Method: 50-step sliding windows from a user's raw H5 session (Hi-Drive CDF,
from Materials/Overtaking2.zip), z-scored per window, matched by Pearson
correlation against every pickle window. Correlation is invariant to the
(unknown) per-channel affine normalization, so r > 0.9999 on a high-entropy
channel (YawRate, LatAcceleration) identifies the same underlying window.

Result (2026-07-07, see LOGBOOK): official test users (13, 2) match ONLY the
pickle test split; official val users (10, 5) ONLY val; train user (22) ONLY
train. SteeringAngle shows spurious cross-hits (quantized, low-entropy
windows) — do not use it as evidence.

Usage: extract user H5 files to Materials/h5-samples/, then
  .venv\\Scripts\\python scripts/analysis/verify_split.py
"""
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
H5DIR = ROOT / "Materials" / "h5-samples"
DATA = ROOT / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = 50
RAW_FIELDS = ["LatAcceleration", "SteeringAngle", "YawRate"]
PICKLE_CH = [0, 1, 2, 3, 4, 5, 6]  # ego block; layout-agnostic scan

# Official protocol (published repo): val users {5,8,10,12,16,19,27},
# test users {2,7,13,18,25,31,36}, train = remaining 36 of 50.
USERS = {"user13LC": "TEST", "user2LC": "TEST", "user10LC": "VAL",
         "user5LC": "VAL", "user22LC": "TRAIN"}


def zscore_windows(a: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(a).float().to(DEV)
    m = t.mean(dim=1, keepdim=True)
    s = t.std(dim=1, keepdim=True)
    return torch.where(s > 1e-8, (t - m) / s, torch.zeros_like(t))


def raw_windows(user: str, field: str) -> np.ndarray:
    with h5py.File(H5DIR / f"{user}.h5", "r") as f:
        series = f["egoVehicle"][field].astype(np.float64)
    n = len(series) - W + 1
    idx = np.arange(n)[:, None] + np.arange(W)[None, :]
    return series[idx]


def load_x(split: str) -> np.ndarray:
    with open(DATA / "data-classification" / f"x_{split}_multi.pkl", "rb") as f:
        return pickle.load(f).astype(np.float32)


def count_matches(zraw: torch.Tensor, x: np.ndarray, ch: int) -> tuple[int, float]:
    zp = zscore_windows(x[:, :, ch].astype(np.float64))
    best, hits = 0.0, 0
    for i in range(0, len(zp), 4096):
        corr = zp[i:i + 4096] @ zraw.T / (W - 1)
        best = max(best, corr.max().item())
        hits += int((corr > 0.9999).any(dim=1).sum().item())
    return hits, best


def main() -> None:
    missing = [u for u in USERS if not (H5DIR / f"{u}.h5").exists()]
    if missing:
        sys.exit(f"missing H5 files in {H5DIR}: {missing} — extract from Materials/Overtaking2.zip")
    xs = {s: load_x(s) for s in ["train", "val", "test"]}
    for user, role in USERS.items():
        print(f"\n=== {user} (official role: {role}) ===")
        for field in RAW_FIELDS:
            zr = zscore_windows(raw_windows(user, field))
            line = [f"{field:>16}"]
            for split in ["train", "val", "test"]:
                hits, best = 0, 0.0
                for ch in PICKLE_CH:
                    h, b = count_matches(zr, xs[split], ch)
                    hits = max(hits, h)
                    best = max(best, b)
                line.append(f"{split}: hits={hits:5d} best_r={best:.4f}")
            print("  ".join(line))


if __name__ == "__main__":
    main()
