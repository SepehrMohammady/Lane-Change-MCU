"""Simulated traffic in the format of the highD protocol (2026-11-11).

Input: the recordings of unas/carla_record.py (CARLA Traffic Manager vehicles on a straight three-lane
carriageway, state at 5 Hz). The traffic is simulated and its lane changes are scripted by the Traffic
Manager; this is not data about real drivers and is used only to test the link from a simulator to the
board and what a network trained on highD does on such traffic.

Features: the 18 features of the protocol from the simulator's state with unas/sim_features.py
(neighbours found geometrically; that module reproduces highD's own features on highD's test
recordings in 99.87% of vehicle frames or more). Accelerations are the change of the simulator's velocity over the last 0.2 s
(one frame), a causal estimate; distances are measured between rear ends.

Scenarios, as datasets/highd/prepare_highd.py cuts them: a lane change = the 35 frames before the first
frame with the vehicle's centre in the new lane, all in one lane (label 1 = to the right, 2 = to the
left); lane keeping = 35-frame segments on a 35-frame grid without a crossing in them or in the 25 frames
after, sampled to (right + left changes) // 2 with a fixed generator.

  .venv\\Scripts\\python datasets/carla/prepare_carla.py [recording.npz ...]     (default: data/tm_seed*.npz)
Output: datasets/carla/data/prepared/test.npz (feats (S,35,18) f32 | label | cross_idx | rec | tv), not
versioned, and datasets/carla/results/kinematics.json (aggregates, next to highD's test lane changes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "unas"))
from sim_features import frame_features                     # noqa: E402

SEQ_LEN, PRED_LEN, FPS = 35, 25, 5


def features(z) -> tuple[np.ndarray, np.ndarray]:
    """(T, N, 18) features (NaN rows for vehicles the simulator removed) and (T, N) lane numbers."""
    meta = json.loads(str(z["meta"]))
    w = float(meta["lane_width"])
    x, y, vx, vy = (z[k].astype(np.float64) for k in ("x", "y", "vx", "vy"))
    length = 2.0 * z["half_len"].astype(np.float64)
    T, N = x.shape
    lane = np.clip(np.floor(np.nan_to_num(y) / w), 0, 2).astype(int)        # 0 = leftmost; y grows to the right
    ax = np.zeros_like(vx)
    ay = np.zeros_like(vy)
    ax[1:], ay[1:] = np.diff(vx, axis=0) * FPS, np.diff(vy, axis=0) * FPS    # change over the last 0.2 s
    out = np.full((T, N, 18), np.nan, np.float32)
    for t in range(T):
        ok = ~np.isnan(x[t])
        out[t, ok] = frame_features(x[t, ok] - length[ok] / 2.0, length[ok], vx[t, ok], ax[t, ok],
                                    -vy[t, ok], -ay[t, ok], lane[t, ok], y[t, ok] - lane[t, ok] * w,
                                    np.full(int(ok.sum()), w), 3, ref="rear")
    return out, lane


def scenarios(feats: np.ndarray, lane: np.ndarray):
    lc, lk = [], []
    T, N = lane.shape
    for i in range(N):
        cross = np.flatnonzero(np.diff(lane[:, i]) != 0) + 1
        for c in cross:
            first = c - SEQ_LEN
            f = feats[first:c, i] if first >= 0 else None
            if f is None or len(set(lane[first:c, i])) > 1 or np.isnan(f).any():
                continue
            lc.append((f, 1 if lane[c, i] > lane[c - 1, i] else 2, SEQ_LEN, i, int(c)))
        for last in range(SEQ_LEN, T - PRED_LEN + 1, SEQ_LEN):
            first = last - SEQ_LEN
            if len(set(lane[first:last + PRED_LEN, i])) > 1 or np.isnan(feats[first:last, i]).any():
                continue
            lk.append((feats[first:last, i], 0, -1, i, int(last)))
    return lc, lk


def lateral(feats: np.ndarray, label: np.ndarray) -> dict:
    """Lateral kinematics of lane-change scenarios from the features (index 0: lateral velocity,
    positive to the left; index 4: distance to the left marking)."""
    lcs = label != 0
    v = np.abs(feats[lcs, :, 0])
    sign = np.where(label[lcs] == 2, 1.0, -1.0)[:, None]                    # toward the new lane
    moved = sign * (feats[lcs, :1, 4] - feats[lcs, :, 4])
    q = lambda a: [round(float(np.percentile(a, p)), 2) for p in (10, 50, 90)]   # noqa: E731
    onset = np.array([(SEQ_LEN - np.flatnonzero(r > 0.2)[0]) / FPS if (r > 0.2).any() else 0.0 for r in v])
    return {"lane_change_scenarios": int(lcs.sum()),
            "peak_lateral_speed_m_s_p10_p50_p90": q(v.max(1)),
            "lateral_speed_2s_before_p10_p50_p90": q(v[:, SEQ_LEN - 10]),
            "lateral_speed_1s_before_p10_p50_p90": q(v[:, SEQ_LEN - 5]),
            "moved_toward_new_lane_m_2s_before_p50": round(float(np.median(moved[:, SEQ_LEN - 10])), 2),
            "moved_toward_new_lane_m_1s_before_p50": round(float(np.median(moved[:, SEQ_LEN - 5])), 2),
            "moved_toward_new_lane_m_last_frame_p50": round(float(np.median(moved[:, -1])), 2),
            "seconds_before_crossing_lateral_speed_first_above_0.2_p10_p50_p90": q(onset)}


def main(paths):
    paths = [Path(p) for p in paths] or sorted((HERE / "data").glob("tm_seed*.npz"))
    lc, lk, recs = [], [], []
    for r, p in enumerate(paths):
        z = np.load(p)
        meta = json.loads(str(z["meta"]))
        f, lane = features(z)
        a, b = scenarios(f, lane)
        lc += [(*s, r) for s in a]
        lk += [(*s, r) for s in b]
        recs.append({"file": p.name, **{k: meta[k] for k in ("seed", "spawned", "lane_change_percentage", "sim_s",
                                                             "real_time_factor")},
                     "vehicles_removed_by_the_simulator": int(np.isnan(z["x"][-1]).sum()),
                     "lane_changes": len(a), "lane_keeping_candidates": len(b)})
    rng = np.random.default_rng(0)
    n_lk = min(len(lk), len(lc) // 2)
    lk = [lk[i] for i in sorted(rng.choice(len(lk), size=n_lk, replace=False))]
    rows = lc + lk
    feats = np.stack([s[0] for s in rows]).astype(np.float32)
    label = np.array([s[1] for s in rows], np.int8)
    out = HERE / "data" / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "test.npz", feats=feats, label=label,
                        cross_idx=np.array([s[2] for s in rows], np.int16),
                        tv=np.array([s[3] for s in rows], np.int32), rec=np.array([s[5] for s in rows], np.int16))
    hd = np.load(ROOT / "datasets" / "highd" / "data" / "prepared" / "test.npz")
    res = {"recordings": recs, "scenarios": {"right_lc": int((label == 1).sum()), "left_lc": int((label == 2).sum()),
                                             "lane_keeping": int((label == 0).sum())},
           "simulated_traffic_manager": lateral(feats, label),
           "highd_test": lateral(hd["feats"], hd["label"])}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "kinematics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
