"""Does the highD classifier recognize the lane changes of simulated Traffic Manager vehicles? (2026-11-11)

Rule fixed before any network output on simulated traffic existed (local notes, 2026-11-11): GO for the
highD network in a CARLA loop if (a) at least 80% of the simulated lane-change scenarios carry the correct
class in the window that ends 2 s before the crossing and (b) at most 10% of the lane-keeping windows are
classified as a lane change, with at least 100 lane-change scenarios. The network is the deployed int8
file with highD's input statistics; its outputs come from the reference of unas/hil_prepare.py
(firmware/build/stream/carla_<stem>/reference.npz, TFLite interpreter), which the board reproduces.

  .venv\\Scripts\\python datasets/carla/transfer_check.py
Output: datasets/carla/results/transfer_check.json (aggregates; highD's test split next to it).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src.lc_windows import early_metrics                    # noqa: E402

STEM = "highd_cls_model_1793711684616882545_453917_int8_io"
IN_LEN, SEQ_LEN, FPS = 10, 35, 5
MIN_LC, MIN_EARLY, MAX_FALSE = 100, 0.80, 0.10


def summary(folder: str) -> dict:
    z = np.load(ROOT / "firmware" / "build" / "stream" / folder / "reference.npz")
    out, label = z["ref"].astype(np.int32), z["label"].astype(np.int64)
    pred = out.argmax(-1)                                   # (S, 26); int8 logits share one scale
    lc = label != 0
    before = (SEQ_LEN - (np.arange(pred.shape[1]) + IN_LEN) + 1) / FPS      # s from window end to the crossing
    right = pred[lc] == label[lc][:, None]
    deq = (out.astype(np.float32) - int(z["out_zp"])) * float(z["out_scale"])
    e = np.exp(deq - deq.max(-1, keepdims=True))
    m = early_metrics(e / e.sum(-1, keepdims=True), label)
    w2 = int(np.flatnonzero(np.isclose(before, 2.0))[0])
    return {"scenarios": {"right_lc": int((label == 1).sum()), "left_lc": int((label == 2).sum()),
                          "lane_keeping": int((label == 0).sum())},
            "correct_class_2s_before_crossing": round(float(right[:, w2].mean()), 4),
            "lane_keeping_windows_called_lane_change": round(float((pred[~lc] != 0).mean()), 4),
            "correct_class_by_seconds_before_crossing": {f"{b:.1f}": round(float(r), 3)
                                                         for b, r in zip(before[::-1], right.mean(0)[::-1])},
            "protocol_metrics": {k: round(float(v), 4) for k, v in m.items() if not isinstance(v, (list, dict))}}


def main():
    sim = summary("carla_" + STEM)
    n_lc = sim["scenarios"]["right_lc"] + sim["scenarios"]["left_lc"]
    a = sim["correct_class_2s_before_crossing"] >= MIN_EARLY
    b = sim["lane_keeping_windows_called_lane_change"] <= MAX_FALSE
    res = {"network": STEM + ".tflite", "input_statistics": "highD training split",
           "rule": {"min_lane_change_scenarios": MIN_LC, "a_min_correct_2s_before": MIN_EARLY,
                    "b_max_lane_keeping_windows_called_lane_change": MAX_FALSE},
           "enough_scenarios": n_lc >= MIN_LC, "a_met": bool(a), "b_met": bool(b),
           "verdict": "GO" if (a and b and n_lc >= MIN_LC) else "NO GO",
           "simulated_traffic_manager": sim, "highd_test": summary(STEM),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "transfer_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
