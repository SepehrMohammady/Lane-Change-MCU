"""The 18 features of the highD protocol from a traffic state (2026-11-11).

datasets/highd/prepare_highd.py reads the surrounding vehicles from highD's own id columns. A simulator
(or a perception stack) gives positions only, so here the neighbours are found geometrically, with highD's
definitions: preceding and following = the nearest vehicle ahead of and behind the target vehicle (TV) in
its lane; in a side lane, alongside = a vehicle whose length overlaps TV's, preceding and following = the
nearest ones entirely ahead and entirely behind. The alongside-fallback rule and the feature order are
those of prepare_highd.py, and so are its signs: lateral quantities are positive toward the driver's
left; longitudinal velocity, acceleration and relative velocities carry a minus sign (the protocol's
convention: -speed along the driving direction).

frame_features() takes one frame of N vehicles that drive toward +s on one carriageway:
  rear, length      along-road position of the rear end and the vehicle length (m)
  v, a              speed and acceleration along the road (m/s, m/s2)
  v_left, a_left    lateral velocity and acceleration, positive toward the driver's left
  lane              lane number, 0 = leftmost
  d_left            distance from the vehicle's centre to the left marking of its lane (m)
  lane_width        width of that lane (m)
  n_lanes           lanes of the carriageway (scalar or per vehicle)
  ref               "rear" or "front": the point distances are measured between. highD's x is the bounding
                    box's smaller x, i.e. the rear end of vehicles driving toward +x and the front end of
                    the others; a simulator uses "rear".

Self-check on highD (Windows venv): the geometric neighbours against the dataset's id columns,
  .venv\\Scripts\\python unas/sim_features.py <recording id> [...]
writes, per recording and feature, the share of vehicle frames where this module equals
prepare_highd.features() to datasets/highd/results/hil/sim_features_check.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

D_DEFAULT = 400.0           # the protocol's "no vehicle" distance


def neighbours(rear: np.ndarray, length: np.ndarray, lane: np.ndarray):
    """Per vehicle: preceding and following in its lane, and (alongside, preceding, following) in the lane
    to the left and to the right; -1 = none. Vehicles drive toward +s."""
    n = len(rear)
    front = rear + length
    centre = rear + length / 2.0
    pv = np.full(n, -1)
    fv = np.full(n, -1)
    side = {+1: np.full((n, 3), -1), -1: np.full((n, 3), -1)}       # +1 = right (lane + 1), -1 = left
    for i in range(n):
        same = np.flatnonzero(lane == lane[i])
        ahead = same[centre[same] > centre[i]]
        behind = same[centre[same] < centre[i]]
        if len(ahead):
            pv[i] = ahead[np.argmin(centre[ahead])]
        if len(behind):
            fv[i] = behind[np.argmax(centre[behind])]
        for step in (+1, -1):
            other = np.flatnonzero(lane == lane[i] + step)
            if not len(other):
                continue
            over = other[(rear[other] < front[i]) & (front[other] > rear[i])]
            before = other[rear[other] >= front[i]]
            after = other[front[other] <= rear[i]]
            if len(over):
                side[step][i, 0] = over[np.argmin(np.abs(centre[over] - centre[i]))]
            if len(before):
                side[step][i, 1] = before[np.argmin(rear[before])]
            if len(after):
                side[step][i, 2] = after[np.argmax(front[after])]
    return pv, fv, side[-1], side[+1]


def frame_features(rear, length, v, a, v_left, a_left, lane, d_left, lane_width, n_lanes, ref: str = "rear"):
    """(N, 18) float32 features of one frame, in the order of prepare_highd.FEATURE_NAMES."""
    rear, length = np.asarray(rear, np.float64), np.asarray(length, np.float64)
    v = np.asarray(v, np.float64)
    lane = np.asarray(lane)
    n = len(rear)
    point = rear if ref == "rear" else rear + length
    pv, fv, left, right = neighbours(rear, length, lane)

    def fallback(tri, i):
        """prepare_highd.Recording._side: (vehicle, its preceding, its following) for one side."""
        along, c1, c2 = tri[i]
        if along >= 0:
            return along, c1, c2
        if c1 < 0 and c2 < 0:
            return -1, -1, -1
        if c1 < 0:
            vv = c2
        elif c2 < 0:
            vv = c1
        else:
            vv = c1 if abs(point[c1] - point[i]) < abs(point[c2] - point[i]) else c2
        return (vv, pv[vv], -1) if vv == c1 else (vv, -1, fv[vv])

    def dist(j, i):
        return abs(point[j] - point[i]) if j >= 0 else D_DEFAULT

    def rvx(j, i):
        return -(v[j] - v[i]) if j >= 0 else 0.0

    n_lanes = np.broadcast_to(np.asarray(n_lanes), (n,))
    out = np.zeros((n, 18), np.float32)
    for i in range(n):
        rv, rpv, rfv = fallback(right, i)
        lv, lpv, lfv = fallback(left, i)
        out[i] = (v_left[i], -v[i], a_left[i], -a[i], d_left[i],
                  rvx(pv[i], i), dist(pv[i], i), rvx(fv[i], i), dist(fv[i], i),
                  dist(rpv, i), dist(rv, i), dist(rfv, i),
                  dist(lpv, i), dist(lv, i), dist(lfv, i),
                  0.0 if lane[i] == 0 else 1.0, 0.0 if lane[i] == n_lanes[i] - 1 else 1.0, lane_width[i])
    return out


def check_highd(rec_id: int) -> dict:
    """This module against prepare_highd.features() on one highD recording (both directions)."""
    import pandas as pd

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "datasets" / "highd"))
    import prepare_highd as ph

    rec = ph.Recording(rec_id)
    cols = ["frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
    tr = pd.read_csv(ph.RAW / f"{rec_id:02d}_tracks.csv", usecols=cols)
    tr = tr[tr.frame % ph.FPS_DIV == 0]
    tr["d"] = tr.id.map(rec.direction)
    want = {}                                               # (id, frame) -> features of prepare_highd
    for tid, t in rec.T.items():
        f = rec.features(tid, 0, len(t["frame"]))
        if f is not None:
            for k, frame in enumerate(t["frame"]):
                want[(tid, int(frame))] = f[k]
    equal = np.zeros(18)
    close = np.zeros(18)
    total = 0
    for (frame, d), g in tr.groupby(["frame", "d"], sort=False):
        marks = rec.upper if d == 1 else rec.lower
        yc = (g.y + g.height / 2.0).to_numpy()
        ind = np.array([max([k for k in range(len(marks)) if y > marks[k]], default=0) for y in yc])
        ok = ind + 1 < len(marks)                           # prepare_highd: outside the markings = invalid
        if not ok.all():
            g, yc, ind = g[ok], yc[ok], ind[ok]
        if not len(g):
            continue
        width = marks[ind + 1] - marks[ind]
        nl = len(marks) - 1
        if d == 2:                                          # toward +x: rear = x, left = smaller y
            feats = frame_features(g.x.to_numpy(), g.width.to_numpy(), g.xVelocity.to_numpy(),
                                   g.xAcceleration.to_numpy(), -g.yVelocity.to_numpy(), -g.yAcceleration.to_numpy(),
                                   ind, np.abs(yc - marks[ind]), width, nl, ref="rear")
        else:                                               # toward -x: mirrored, highD's x is the front end
            feats = frame_features(-(g.x + g.width).to_numpy(), g.width.to_numpy(), -g.xVelocity.to_numpy(),
                                   -g.xAcceleration.to_numpy(), g.yVelocity.to_numpy(), g.yAcceleration.to_numpy(),
                                   nl - 1 - ind, np.abs(yc - marks[ind + 1]), width, nl, ref="front")
        for row, tid in zip(feats, g.id.to_numpy()):
            w = want.get((int(tid), int(frame)))
            if w is None:
                continue
            equal += np.abs(row - w) <= 1e-4
            close += np.abs(row - w) <= 0.05 * np.maximum(1.0, np.abs(w))
            total += 1
    res = {"recording": rec_id, "vehicle_frames": total,
           "equal_share": {n: round(float(e / total), 4) for n, e in zip(ph.FEATURE_NAMES, equal)},
           "within_5_percent_share": {n: round(float(c / total), 4) for n, c in zip(ph.FEATURE_NAMES, close)}}
    return res


if __name__ == "__main__":
    import json

    results = []
    for arg in sys.argv[1:]:
        r = check_highd(int(arg))
        results.append(r)
        print(json.dumps({"recording": r["recording"], "vehicle_frames": r["vehicle_frames"],
                          "lowest_equal_share": min(r["equal_share"].values())}), flush=True)
    out = Path(__file__).resolve().parent.parent / "datasets" / "highd" / "results" / "hil" / "sim_features_check.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out.name}")
