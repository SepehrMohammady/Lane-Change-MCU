"""highD -> lane-change scenarios, following EarlyLCPred's protocol exactly.

Reimplements the scenario extraction of Mozaffari et al. (T-IV 2022,
github.com/SajjadMzf/EarlyLCPred, data_prep/) so our numbers are comparable at
the protocol level:

- 25 Hz tracks downsampled to 5 Hz (keep frame % 5 == 0)
- LC scenario  = 35 frames (7 s) ending at the lane-crossing frame; valid only
  if the target vehicle (TV) stays in one lane for the whole 35 frames
- labels: 0 = lane keep, 1 = right LC, 2 = left LC (direction-normalized via
  drivingDirection, exactly their get_last_idxs logic)
- LK scenarios are undersampled to (RLC+LLC)/2 per recording, preferring
  tracks whose crossing happens 25 frames after the scenario (known TTLC),
  exactly their get_lk_scenarios selection
- features: their 18-dim "state_ours" vector per frame (lat/long velocity &
  acceleration, lateral position to left marking, PV/FV relative velocity,
  8 surrounding-vehicle longitudinal distances with their alongside-fallback
  rule, lane existence flags, lane width). Deviation, documented: we compute
  lateral features in metres where their code mixes image-scaled units; the
  difference is a per-feature affine map, which min-max normalization (their
  own normalization) removes exactly.
- split by recording id, theirs: train 1-50, val 51-55, test 56-60

Output: datasets/highd/data/prepared/{train,val,test}.npz
  feats (S,35,18) f32 | label (S,) i8 | cross_idx (S,) i16 | rec (S,) i16 | tv (S,) i32
  cross_idx = scenario-local index of the lane-crossing frame: 35 for LC,
  50 for LK with a later crossing, -1 for LK with none (TTLC undefined).
Windows for training are sliced later (input 10 frames; slide s in [0,25];
TTLC at window end = (cross_idx - (s+10)) / 5 seconds).

Run: .venv\\Scripts\\python datasets/highd/prepare_highd.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "data"
OUT = HERE / "data" / "prepared"

FPS_DIV = 5                 # 25 Hz -> 5 Hz
SEQ_LEN = 35
PRED_LEN = 25
SPLITS = {"train": range(1, 51), "val": range(51, 56), "test": range(56, 61)}

FEATURE_NAMES = [
    "lat_velocity", "long_velocity", "lat_acceleration", "long_acceleration",
    "lat_pos_left_marking", "rel_velo_pv", "dist_pv", "rel_velo_fv", "dist_fv",
    "dist_rpv", "dist_rv", "dist_rfv", "dist_lpv", "dist_lv", "dist_lfv",
    "left_lane_exists", "right_lane_exists", "lane_width",
]
D_DEFAULT = 400.0           # their "no vehicle" longitudinal distance


class Recording:
    """One recording, downsampled, with O(1) per-(id, frame) lookups."""

    def __init__(self, rec_id: int):
        self.rec_id = rec_id
        p = RAW / f"{rec_id:02d}"
        tracks = pd.read_csv(f"{p}_tracks.csv")
        tracks = tracks[tracks.frame % FPS_DIV == 0]
        meta = pd.read_csv(f"{p}_recordingMeta.csv").iloc[0]
        static = pd.read_csv(f"{p}_tracksMeta.csv").set_index("id")

        self.upper = np.array([float(v) for v in meta.upperLaneMarkings.split(";")])
        self.lower = np.array([float(v) for v in meta.lowerLaneMarkings.split(";")])
        self.direction = static.drivingDirection.to_dict()

        cols = ["frame", "x", "y", "height", "xVelocity", "yVelocity",
                "xAcceleration", "yAcceleration", "laneId",
                "precedingId", "followingId",
                "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
                "rightPrecedingId", "rightAlongsideId", "rightFollowingId"]
        self.T: dict[int, dict[str, np.ndarray]] = {}
        for tid, g in tracks[["id"] + cols].groupby("id", sort=False):
            g = g.sort_values("frame")
            self.T[tid] = {c: g[c].to_numpy() for c in cols}
            self.T[tid]["first"] = int(g.frame.iloc[0])

    def row(self, tid: int, frame: int) -> int | None:
        """Row index of vehicle tid at absolute frame, or None."""
        t = self.T.get(tid)
        if t is None:
            return None
        i = (frame - t["first"]) // FPS_DIV
        if i < 0 or i >= len(t["frame"]) or t["frame"][i] != frame:
            return None
        return int(i)

    # --- surrounding vehicles, their get_svs alongside-fallback rule --------
    def _side(self, tv: dict, i: int, frame: int, along: str, prec: str, foll: str):
        """Return (v_id, pv_id, fv_id) for one side at track row i."""
        a = int(tv[along][i])
        if a != 0:
            return a, int(tv[prec][i]), int(tv[foll][i])
        c1, c2 = int(tv[prec][i]), int(tv[foll][i])
        if c1 == 0 and c2 == 0:
            return 0, 0, 0
        if c1 == 0:
            v = c2
        elif c2 == 0:
            v = c1
        else:
            x_tv = tv["x"][i]
            r1, r2 = self.row(c1, frame), self.row(c2, frame)
            x1 = self.T[c1]["x"][r1] if r1 is not None else np.inf
            x2 = self.T[c2]["x"][r2] if r2 is not None else np.inf
            v = c1 if abs(x1 - x_tv) < abs(x2 - x_tv) else c2
        rv_row = self.row(v, frame)
        if rv_row is None:
            return 0, 0, 0
        vd = self.T[v]
        if v == c1:      # nearest is the preceding candidate -> its own PV
            return v, int(vd["precedingId"][rv_row]), 0
        else:            # nearest is the following candidate -> its own FV
            return v, 0, int(vd["followingId"][rv_row])

    def features(self, tid: int, first: int, last: int) -> np.ndarray | None:
        """(SEQ_LEN, 18) state_ours features for track rows [first, last)."""
        tv = self.T[tid]
        d = self.direction[tid]
        sgn = 1.0 if d == 1 else -1.0
        markings = self.upper if d == 1 else self.lower
        out = np.zeros((last - first, 18), dtype=np.float32)

        for k, i in enumerate(range(first, last)):
            frame = int(tv["frame"][i])
            y_c = tv["y"][i] + tv["height"][i] / 2.0
            lane_ind = 0
            for ind in range(len(markings) - 1, -1, -1):
                if y_c > markings[ind]:
                    lane_ind = ind
                    break
            if lane_ind + 1 >= len(markings):
                return None                      # their `valid = False`
            lane_width = markings[lane_ind + 1] - markings[lane_ind]
            left_ind = lane_ind + 1 if d == 1 else lane_ind
            n_mark = len(markings)
            left_exists = 0.0 if ((lane_ind + 2 == n_mark and d == 1)
                                  or (lane_ind == 0 and d == 2)) else 1.0
            right_exists = 0.0 if ((lane_ind + 2 == n_mark and d == 2)
                                   or (lane_ind == 0 and d == 1)) else 1.0

            pv, fv = int(tv["precedingId"][i]), int(tv["followingId"][i])
            rv, rpv, rfv = self._side(tv, i, frame, "rightAlongsideId",
                                      "rightPrecedingId", "rightFollowingId")
            lv, lpv, lfv = self._side(tv, i, frame, "leftAlongsideId",
                                      "leftPrecedingId", "leftFollowingId")

            def dist(vid):
                r = self.row(vid, frame) if vid else None
                return abs(self.T[vid]["x"][r] - tv["x"][i]) if r is not None else D_DEFAULT

            def rvx(vid):
                r = self.row(vid, frame) if vid else None
                return sgn * (self.T[vid]["xVelocity"][r] - tv["xVelocity"][i]) if r is not None else 0.0

            out[k] = (
                sgn * tv["yVelocity"][i], sgn * tv["xVelocity"][i],
                sgn * tv["yAcceleration"][i], sgn * tv["xAcceleration"][i],
                abs(y_c - markings[left_ind]),
                rvx(pv), dist(pv), rvx(fv), dist(fv),
                dist(rpv), dist(rv), dist(rfv),
                dist(lpv), dist(lv), dist(lfv),
                left_exists, right_exists, lane_width,
            )
        return out


def crossings(lane: np.ndarray, direction: int):
    """All lane-crossing row indices + labels, their get_last_idxs('lc')."""
    idxs, labels = [], []
    pos = 0
    while True:
        start_lane = lane[pos]
        rest = np.nonzero(lane[pos:] != start_lane)[0]
        if len(rest) == 0:
            break
        pos = pos + int(rest[0])
        cur = lane[pos]
        if direction == 1:
            lab = 1 if cur < start_lane else 2
        else:
            lab = 1 if cur > start_lane else 2
        idxs.append(pos)
        labels.append(lab)
    return idxs, labels


def extract_recording(rec_id: int):
    rec = Recording(rec_id)
    lc, lk_known, lk_unknown = [], [], []
    for tid, tv in rec.T.items():
        d = rec.direction[tid]
        lane = tv["laneId"]
        cr, labs = crossings(lane, d)

        for c, lab in zip(cr, labs):
            first = c - SEQ_LEN
            if first < 0 or len(set(lane[first:c])) > 1:
                continue
            f = rec.features(tid, first, c)
            if f is not None:
                lc.append((f, lab, SEQ_LEN, tid))

        # one LK scenario per track (their get_lk_scenarios)
        if cr:
            last = cr[0] - PRED_LEN
            known = True
        else:
            last = len(lane) - PRED_LEN
            known = False
        first = last - SEQ_LEN
        if first < 0 or len(set(lane[first:last])) > 1:
            continue
        f = rec.features(tid, first, last)
        if f is None:
            continue
        (lk_known if known else lk_unknown).append(
            (f, 0, SEQ_LEN + PRED_LEN if known else -1, tid))

    rlc = sum(1 for s in lc if s[1] == 1)
    llc = sum(1 for s in lc if s[1] == 2)
    lk_count = (rlc + llc) // 2
    lk = lk_known[:lk_count]
    if len(lk) < lk_count:
        lk += lk_unknown[:lk_count - len(lk)]
    return lc + lk, rlc, llc, len(lk)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, recs in SPLITS.items():
        feats, labels, cross, rec_ids, tvs = [], [], [], [], []
        t0 = time.time()
        for r in recs:
            scen, rlc, llc, lk = extract_recording(r)
            for f, lab, cx, tid in scen:
                feats.append(f); labels.append(lab); cross.append(cx)
                rec_ids.append(r); tvs.append(tid)
            print(f"  rec {r:02d}: RLC {rlc:4d}  LLC {llc:4d}  LK {lk:4d}", flush=True)
        x = np.stack(feats).astype(np.float32)
        y = np.array(labels, dtype=np.int8)
        np.savez_compressed(
            OUT / f"{split}.npz", feats=x, label=y,
            cross_idx=np.array(cross, dtype=np.int16),
            rec=np.array(rec_ids, dtype=np.int16), tv=np.array(tvs, dtype=np.int32))
        stats[split] = {"scenarios": len(y),
                        "LK": int((y == 0).sum()), "RLC": int((y == 1).sum()),
                        "LLC": int((y == 2).sum()),
                        "seconds": round(time.time() - t0, 1)}
        print(f"{split}: {stats[split]}", flush=True)

    (OUT / "meta.json").write_text(json.dumps({
        "protocol": "EarlyLCPred (Mozaffari et al., T-IV 2022)",
        "fps": 25 // FPS_DIV, "seq_len": SEQ_LEN, "in_seq_len": 10,
        "pred_len": PRED_LEN, "labels": {"0": "LK", "1": "RLC", "2": "LLC"},
        "features": FEATURE_NAMES, "splits": {k: [min(v), max(v)] for k, v in SPLITS.items()},
        "stats": stats}, indent=1))
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
