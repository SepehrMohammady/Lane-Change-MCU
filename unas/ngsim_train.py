"""NGSIM's own networks: the reference for the gap at the NGSIM sites (2026-11-26).

The online studies measure how much of the gap between a network trained elsewhere and one trained at the site the
device closes. For NGSIM the site-trained reference is arm K of unas/track_train_check.py with exiD's recipe, unchanged:
the same architecture and training function, windows from complete tracks whose label is known, balanced by class,
trained on NGSIM's training recordings (1 and 4) with early stopping on its validation recordings (2 and 5), seeds
100-104. Inputs are scaled with the range of NGSIM's training frames (NGSIM has no protocol windows). Nothing is scored
on NGSIM's test recordings here: the test is scored once, in the online study, after its rule is fixed.

  WSL: TRACK_DATASET=exid HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/exid/data/prepared \\
       ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/ngsim_train.py
Output: datasets/ngsim/data/track_train/K_<model>_seed<s>.h5 and scaling.npz (not versioned),
datasets/ngsim/results/seeds/ngsim_train.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("TRACK_DATASET", "exid")             # lets track_train_check import; its file paths are replaced
sys.path.insert(0, str(Path(__file__).resolve().parent))
import track_train_check as ttc                              # noqa: E402

REPO = ttc.REPO
PROBE = REPO / "firmware" / "build" / "probe"


def tracks_file(split: str = "test") -> Path:
    """datasets/ngsim/prepare_ngsim.py's files (not imported: it needs pandas, which the WSL venv lacks)."""
    return PROBE / f"ngsim_tracks_{split}.npz"


ttc.tracks_file = tracks_file
KEEP = REPO / "datasets" / "ngsim" / "data" / "track_train"
OUT = REPO / "datasets" / "ngsim" / "results" / "seeds" / os.environ.get("OUT_NAME", "ngsim_train.jsonl")
SEEDS = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]


def scaling() -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(tracks_file("train"))["feats"].astype(np.float32)
    lo = raw.min(0)
    return lo, raw.max(0) - lo + 1e-12


def main() -> None:
    import input_scaling_check as isc

    sv = isc.sv
    KEEP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lo, span = scaling()
    np.savez(KEEP / "scaling.npz", lo=lo, span=span)
    x, y, counts = ttc.track_windows("train", lo, span, np.random.default_rng(0), True)
    val = ttc.track_windows("val", lo, span, np.random.default_rng(0), True)[:2]
    done = {r["seed"] for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    smoke = "_smoke" if sv.EPOCH_CAP else ""
    for seed in SEEDS:
        if seed in done and not smoke:
            continue
        t0 = time.perf_counter()
        net, epochs = ttc.train(sv, (x, y), val, seed)
        net.save(str(KEEP / f"K_{ttc.MODEL}_seed{seed}{smoke}.h5"))
        logits = net.predict(val[0], batch_size=4096, verbose=0)
        rec = {"arm": "K", "seed": seed, "training_windows": int(len(x)), "track_windows_per_class_before_balancing": counts,
               "early_stopping_windows": int(len(val[0])), "epochs_run": epochs, "params": int(net.count_params()),
               "validation_accuracy_balanced_windows": round(float((logits.argmax(1) == val[1]).mean()), 5),
               "wall_s": round(time.perf_counter() - t0, 1), "smoke": bool(smoke),
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if not smoke:
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"seed {seed}: {len(x)} windows, {epochs} epochs, {rec['wall_s']} s", flush=True)
        sv.keras.backend.clear_session()


if __name__ == "__main__":
    main()
