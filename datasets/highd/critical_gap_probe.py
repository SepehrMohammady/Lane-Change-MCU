"""Does the highD intention network still announce a lane change when a vehicle sits at a critical gap
behind in the target lane? (2026-11-17, a probe before any closed loop on scripted cut-ins.)

Real traffic holds no conflict: in the recorded lane changes of highD the follower in the target lane is
never near (datasets/highd/cutin_headroom.py, 2026-11-11). A closed loop that is to show what an early
prediction is worth therefore needs a scripted critical gap around a recorded lane change (the cut-in
matrix of UN R157), and such gaps are outside the training data. This probe changes one input only.

Scenarios: the test split as prepared (prepare_highd.py). In every lane-change scenario the distance to
the follower in the target lane (dist_rfv for a right, dist_lfv for a left lane change) is replaced over
all 35 frames by  g + LENGTH + dv * t  (t = time to the crossing, g = gap at the crossing, dv = closing
speed, capped at the protocol's 400 m); in every lane-keeping scenario the same is done for the left lane
where it exists, else the right. Everything else is as recorded.

Fixed beforehand (local notes, 2026-11-17 11:10): the prediction "survives" if, for the paper's network,
in every cell with g <= 30 m the share of lane-change scenarios with the correct class in the window that
ends 2 s before the crossing is at least 0.8 of its value on the unchanged scenarios, and the share of
lane-keeping windows called a lane change rises by less than 10 points.

Two steps, because TensorFlow lives in the WSL NAS venv and the protocol's metrics in the Windows one:
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/datasets/highd/critical_gap_probe.py     # the networks' outputs
  .venv\\Scripts\\python datasets/highd/critical_gap_probe.py --report
Output: datasets/highd/results/hil/critical_gap_probe.json (shares only, no trajectories); the outputs of
the first step stay in firmware/build/probe/ (not versioned).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PREP = ROOT / "datasets" / "highd" / "data" / "prepared"
FEATURE_NAMES = json.loads((PREP / "meta.json").read_text())["features"]     # as written by prepare_highd.py
DEPLOY = ROOT / "datasets" / "highd" / "results" / "deploy"
OUT = ROOT / "datasets" / "highd" / "results" / "hil" / "critical_gap_probe.json"
PROBS = ROOT / "firmware" / "build" / "probe" / "critical_gap_probs.npz"
NETS = {"searched (the paper's choice)": "highd_cls_model_1793711684616882545_453917_int8_io.tflite",
        "hand-designed CNN": "highd_dscnn_cls_int8_io.tflite"}
IN_LEN, SEQ_LEN, FPS = 10, 35, 5.0
GAPS, CLOSING = (5.0, 10.0, 20.0, 30.0, 50.0), (0.0, 5.0, 10.0)
LENGTH, D_DEFAULT = 4.6, 400.0                               # a car's length; the protocol's "no vehicle"
RFV, LFV = FEATURE_NAMES.index("dist_rfv"), FEATURE_NAMES.index("dist_lfv")
LEFT_EXISTS = FEATURE_NAMES.index("left_lane_exists")
AT_2S = int(SEQ_LEN - 2.0 * FPS) - IN_LEN                    # the window whose last frame is 2 s before the crossing
KEEP, RISE, CRITICAL = 0.8, 0.10, 30.0                       # the rule


def scripted(feats: np.ndarray, label: np.ndarray, gap: float, dv: float) -> np.ndarray:
    x = feats.copy()
    t = (SEQ_LEN - np.arange(SEQ_LEN)) / FPS                 # time to the crossing of every frame
    d = np.minimum(gap + LENGTH + dv * t, D_DEFAULT).astype(np.float32)
    for s in range(len(x)):
        if label[s] == 1:
            col = RFV
        elif label[s] == 2:
            col = LFV
        else:
            col = LFV if x[s, :, LEFT_EXISTS].min() > 0.5 else RFV
        x[s, :, col] = d
    return x


def outputs(tfl: Path, feats: np.ndarray, lo, span) -> np.ndarray:
    """Softmax of the int8 outputs for the 26 windows of every scenario, with the board's arithmetic."""
    import tensorflow as tf

    it = tf.lite.Interpreter(model_path=str(tfl))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    (si, zi), (so, zo) = inp["quantization"], out["quantization"]
    q = np.clip(np.rint(((feats - lo) / span) / np.float32(si) + np.float32(zi)), -128, 127).astype(np.int8)
    n = SEQ_LEN - IN_LEN + 1
    res = np.empty((len(q), n, 3), np.float32)
    for s in range(len(q)):
        for k in range(n):
            it.set_tensor(inp["index"], q[s, k:k + IN_LEN][None])
            it.invoke()
            res[s, k] = (it.get_tensor(out["index"]).reshape(-1).astype(np.float32) - zo) * so
    z = res - res.max(-1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(-1, keepdims=True)


def measures(probs: np.ndarray, label: np.ndarray) -> dict:
    from src.lc_windows import early_metrics

    pred = probs.argmax(-1)
    lc = label != 0
    m = early_metrics(probs, label)
    return {"lane_changes_correct_2_s_before": round(float((pred[lc, AT_2S] == label[lc]).mean()), 4),
            "right_correct_2_s_before": round(float((pred[label == 1, AT_2S] == 1).mean()), 4),
            "left_correct_2_s_before": round(float((pred[label == 2, AT_2S] == 2).mean()), 4),
            "lane_keeping_windows_called_a_lane_change": round(float((pred[~lc] != 0).mean()), 4),
            "accuracy": round(m["accuracy"], 4), "tau_c_s": round(m["tau_c_s"], 3)}


def cells():
    return [(gap, dv) for gap in GAPS for dv in CLOSING]


def run():
    """WSL: the softmax outputs of both networks on the unchanged and on every scripted set."""
    train = np.load(PREP / "train.npz")["feats"].astype(np.float32)
    lo = train.reshape(-1, train.shape[-1]).min(0)
    span = train.reshape(-1, train.shape[-1]).max(0) - lo + 1e-12
    z = np.load(PREP / "test.npz")
    feats, label = z["feats"].astype(np.float32), z["label"].astype(np.int64)
    out = {}
    for n, file in enumerate(NETS.values()):
        out[f"n{n}_unchanged"] = outputs(DEPLOY / file, feats, lo, span)
        for c, (gap, dv) in enumerate(cells()):
            out[f"n{n}_c{c}"] = outputs(DEPLOY / file, scripted(feats, label, gap, dv), lo, span)
            print(f"{file}: gap {gap:g} m, closing {dv:g} m/s done", flush=True)
    PROBS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROBS, **out)
    print("OUTPUTS-DONE")


def report():
    z = np.load(PREP / "test.npz")
    feats, label = z["feats"].astype(np.float32), z["label"].astype(np.int64)
    probs = np.load(PROBS)
    real = {name: [round(float(v), 1) for v in np.percentile(d[d < D_DEFAULT - 1], (5, 50, 95))] + [round(float((d >= D_DEFAULT - 1).mean()), 3)]
            for name, d in (("right lane changes, dist_rfv at the last frame", feats[label == 1][:, -1, RFV]),
                            ("left lane changes, dist_lfv at the last frame", feats[label == 2][:, -1, LFV]))}
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "scenarios": {"lane_keeping": int((label == 0).sum()), "right": int((label == 1).sum()), "left": int((label == 2).sum())},
           "recorded_follower_distance_m_5_50_95_percentile_and_share_without_one": real,
           "scripted": "dist to the target-lane follower = gap + 4.6 m + closing speed * time to the crossing, capped at 400 m",
           "rule": f"survives if, for the paper's network, in every cell with gap <= {CRITICAL:g} m the share correct 2 s before "
                   f"the crossing is at least {KEEP} of the unchanged value and the lane-keeping false-alarm share rises by less than "
                   f"{RISE * 100:g} points", "networks": {}}
    for n, (name, file) in enumerate(NETS.items()):
        base = measures(probs[f"n{n}_unchanged"], label)
        rows = [{"gap_m": gap, "closing_m_s": dv, **measures(probs[f"n{n}_c{c}"], label)} for c, (gap, dv) in enumerate(cells())]
        crit = [c for c in rows if c["gap_m"] <= CRITICAL]
        worst = min(c["lane_changes_correct_2_s_before"] for c in crit)
        rise = max(c["lane_keeping_windows_called_a_lane_change"] for c in crit) - base["lane_keeping_windows_called_a_lane_change"]
        res["networks"][name] = {"file": file, "unchanged": base, "cells": rows,
                                 "lowest_share_correct_2_s_before_in_critical_cells": worst,
                                 "largest_rise_of_lane_keeping_false_alarms": round(rise, 4),
                                 "survives": bool(worst >= KEEP * base["lane_changes_correct_2_s_before"] - 1e-9 and rise < RISE)}
    res["survives"] = res["networks"]["searched (the paper's choice)"]["survives"]
    OUT.write_text(json.dumps(res, indent=1))
    for name, r in res["networks"].items():
        print(name, "unchanged:", r["unchanged"])
        for c in r["cells"]:
            print(f"  gap {c['gap_m']:4.0f} m, closing {c['closing_m_s']:4.0f} m/s: correct 2 s before {c['lane_changes_correct_2_s_before']:.3f} "
                  f"(right {c['right_correct_2_s_before']:.3f}, left {c['left_correct_2_s_before']:.3f}), lane keeping called a lane change "
                  f"{c['lane_keeping_windows_called_a_lane_change']:.3f}, accuracy {c['accuracy']:.3f}, tau_c {c['tau_c_s']:.2f} s")
        print("  survives:", r["survives"])


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
