"""Check of the live features (datasets/dmir/live_features.py) against the lab's converter on the same drive
(2026-11-17).

A drive with the client and live_loop.install() writes the features computed during the drive (a JSON-lines
log) and the simulator's recording (data_collection/logs/log_NNN.txt). The lab's converter
(data_preparation/generate_CDF_Carla_log_to_H5.py of the framework) turns the recording into a session file
afterwards, from which scenario_race.raw_matrix takes the 31 raw signals. Fixed beforehand (local notes, 2026-11-17 14:55): the live features equal the converter's
within 1e-3 relative (max(1, |value|)) in at least 99% of the values, at the expected alignment (the k-th
kept frame of the hook = the k-th row of the session file). Alignments of -2 to +2 frames are reported too.

  .venv\\Scripts\\python datasets/dmir/live_check.py <live jsonl> <session h5>
The summary is written next to the live log (<name>_check.json), not versioned.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import scenario_race as sr                                   # noqa: E402
from src.features import FEATURES_BY_TASK                   # noqa: E402

NAMES = FEATURES_BY_TASK["classification"]
TOL, SHARE = 1e-3, 0.99


def load(live: Path, h5: Path):
    import h5py

    rows = [json.loads(line) for line in live.read_text().splitlines()[1:]]
    L = np.array([r["row"] for r in rows], float)
    with h5py.File(h5, "r") as f:
        ego, ll, ob = f["egoVehicle"][()], f["laneLines"][()], f["objects"][()]
    C = sr.raw_matrix(ego, ll, ob)
    return L, C, rows


def compare(L: np.ndarray, C: np.ndarray, offset: int):
    """Live row k against session row k + offset."""
    a0, b0 = max(0, -offset), max(0, offset)
    n = min(len(L) - a0, len(C) - b0)
    if n <= 0:
        return None
    a, b = L[a0:a0 + n], C[b0:b0 + n]
    ok = np.abs(a - b) <= TOL * np.maximum(1.0, np.abs(b))
    return {"frames": int(n), "share_all": round(float(ok.mean()), 5),
            "share_per_feature": {nm: round(float(s), 4) for nm, s in zip(NAMES, ok.mean(0))},
            "max_abs_diff_per_feature": {nm: float(f"{d:.3g}") for nm, d in zip(NAMES, np.abs(a - b).max(0))}}


def main(live: str, h5: str):
    L, C, rows = load(Path(live), Path(h5))
    res = {"live_frames": len(L), "session_frames": len(C), "tolerance": TOL, "rule_share": SHARE,
           "alignments": {str(o): (lambda c: c and {k: c[k] for k in ("frames", "share_all")})(compare(L, C, o)) for o in range(-2, 3)}}
    res["expected_alignment"] = compare(L, C, 0)
    res["met"] = bool(res["expected_alignment"] and res["expected_alignment"]["share_all"] >= SHARE)
    out = Path(live).with_name(Path(live).stem + "_check.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "expected_alignment"}, indent=1))
    e = res["expected_alignment"]
    if e:
        bad = {k: v for k, v in e["share_per_feature"].items() if v < 1.0}
        print("features below 100% at the expected alignment:", json.dumps(bad))
        print("largest differences:", json.dumps({k: e["max_abs_diff_per_feature"][k] for k in bad}))


if __name__ == "__main__":
    main(*sys.argv[1:3])
