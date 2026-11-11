"""Headroom of early lane-change knowledge for a following car, on recorded highD cut-ins (2026-11-11).

Before any closed loop with a network, a board or a simulator: what can a follower gain at all when it
learns of a lane change k seconds before the lane crossing? Recorded data only.

Scenarios: the lane changes of the test split (recordings 56-60, the protocol's scenarios of
prepare_highd.py) whose target vehicle (TV) has a follower F in the target lane at the crossing frame; F
must stay in that lane and be tracked from 7 s before the crossing to at least 3 s after it (the window
ends 5 s after it or when F's or TV's track ends). F is replaced by an ego car (a point mass that starts
at F's recorded position and speed); every other vehicle replays its recording at 25 Hz.

Ego controller: the ACC car-following model of Milanes and Shladover (Transp. Res. C, 2014),
  a = K1 (gap - S0 - T v) + K2 (v_leader - v),  K1 = 0.23 1/s^2, K2 = 0.07 1/s, S0 = 2 m,
with T = the recorded driver's own time gap at the start (the ego starts in equilibrium; 1.5 s without a
leader, at least 0.2 s), cruise a = 0.4 (v_set - v) with v_set = F's highest recorded speed in the window,
the lower of both when a leader is within 150 m, limits -8 to +2 m/s^2, no actuator lag. The leader is the
nearest replayed vehicle ahead in the ego's lane. From its recognition time on, a TV that is still in its
old lane is a second, anticipated leader whose braking demand is bounded at 1.5 m/s^2; once it is in the
ego's lane it is an ordinary leader.

Arms: reactive (TV is recognized at the crossing frame, k = 0) and oracle k = 1..5 s before it; the
recorded driver's own acceleration is reported next to them. Outcome per scenario: peak deceleration,
minimum time gap, minimum time to collision, contact with the leader.

Rule fixed before the first run (paper notes, 2026-11-11): GO if, for k = 2 s, at least 30 scenarios
exceed 2.0 m/s^2 peak deceleration in the reactive arm and the oracle at least halves that number.

  .venv\\Scripts\\python datasets/highd/cutin_headroom.py [highd|exid]
Output: datasets/<ds>/results/hil/cutin_headroom.json (aggregates only); per-scenario values stay in
datasets/<ds>/data/replay/ (not versioned).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_highd as ph                                   # noqa: E402

FPS = 25
DT = 1.0 / FPS
PRE, POST, POST_MIN = 7 * FPS, 5 * FPS, 3 * FPS              # frames before / after the crossing
K1, K2, S0, KV = 0.23, 0.07, 2.0, 0.4
RANGE, A_MIN, A_MAX, A_ANT = 150.0, -8.0, 2.0, -1.5
T_FREE, T_FLOOR = 1.5, 0.2
ARMS = (0, 1, 2, 3, 4, 5)                                    # recognition, seconds before the crossing
RULE_K, RULE_DECEL, RULE_MIN_EVENTS = 2, 2.0, 30


class Rec25:
    """One recording at 25 Hz in along-road coordinates (s grows in the driving direction)."""

    def __init__(self, raw: Path, rec_id: int):
        p = raw / f"{rec_id:02d}"
        tr = pd.read_csv(f"{p}_tracks.csv", usecols=["frame", "id", "x", "width", "xVelocity", "xAcceleration", "laneId"])
        direction = pd.read_csv(f"{p}_tracksMeta.csv").set_index("id").drivingDirection
        sgn = np.where(tr.id.map(direction).to_numpy() == 2, 1.0, -1.0)
        x, w = tr.x.to_numpy(), tr.width.to_numpy()
        tr["s_rear"] = np.where(sgn > 0, x, -(x + w))
        tr["s_front"] = np.where(sgn > 0, x + w, -x)
        tr["v"] = sgn * tr.xVelocity.to_numpy()
        tr["a"] = sgn * tr.xAcceleration.to_numpy()
        self.T: dict[int, dict] = {}
        for tid, g in tr.groupby("id", sort=False):
            g = g.sort_values("frame")
            fr = g.frame.to_numpy()
            assert (np.diff(fr) == 1).all(), (rec_id, tid)
            self.T[int(tid)] = {c: g[c].to_numpy() for c in ("s_rear", "s_front", "v", "a", "laneId")}
            self.T[int(tid)]["first"], self.T[int(tid)]["last"] = int(fr[0]), int(fr[-1])
        self.by_frame = {int(f): (g.id.to_numpy(), g.laneId.to_numpy(), g.s_rear.to_numpy(), g.s_front.to_numpy(),
                                  g.v.to_numpy()) for f, g in tr.groupby("frame", sort=False)}

    def leader(self, frame: int, lane: int, s_front: float, skip: int):
        """Nearest vehicle in the lane whose front is ahead of s_front: (gap to its rear, its speed) or None."""
        ids, lanes, sr, sf, v = self.by_frame[frame]
        m = (lanes == lane) & (ids != skip) & (sf > s_front)
        if not m.any():
            return None
        j = np.flatnonzero(m)[np.argmin(sr[m])]
        return float(sr[j] - s_front), float(v[j])


def lc_scenarios(rec: ph.Recording):
    """The protocol's lane-change scenarios of one recording, in prepare_highd's order:
    (TV id, crossing frame, label, follower id in the new lane at the crossing frame)."""
    out = []
    for tid, tv in rec.T.items():
        cr, labs = ph.crossings(tv["laneId"], rec.direction[tid])
        for c, lab in zip(cr, labs):
            first = c - ph.SEQ_LEN
            if first < 0 or len(set(tv["laneId"][first:c])) > 1 or rec.features(tid, first, c) is None:
                continue
            out.append((int(tid), int(tv["frame"][c]), int(lab), int(tv["followingId"][c])))
    return out


def window(R: Rec25, tv: int, F: int, f_c: int):
    """(f0, f_end) if the scenario is eligible, else a reason."""
    if F == 0 or F not in R.T:
        return "no follower"
    tF, tT = R.T[F], R.T[tv]
    f0, f_end = f_c - PRE, min(f_c + POST, tF["last"], tT["last"])
    if f0 < tF["first"] or f0 < tT["first"]:
        return "follower or TV not tracked 7 s before"
    if f_end < f_c + POST_MIN:
        return "less than 3 s after the crossing"
    lanes = tF["laneId"][f0 - tF["first"]:f_end - tF["first"] + 1]
    if len(set(lanes.tolist())) > 1 or lanes[0] != tT["laneId"][f_c - tT["first"]]:
        return "follower changes lane"
    return f0, f_end


def drive(R: Rec25, tv: int, F: int, f_c: int, f0: int, f_end: int, k: float | None):
    """One run. k = None follows the recorded driver; otherwise the ACC with TV recognized k s before f_c."""
    tF, tT = R.T[F], R.T[tv]
    i0 = f0 - tF["first"]
    lane = int(tF["laneId"][i0])
    s, v = float(tF["s_front"][i0]), float(tF["v"][i0])
    v_set = float(tF["v"][i0:f_end - tF["first"] + 1].max())
    lead = R.leader(f0, lane, s, F)
    T = max(T_FLOOR, (lead[0] - S0) / max(v, 1.0)) if lead and lead[0] <= RANGE else T_FREE
    f_rec = f_c - int(round(k * FPS)) if k is not None else None
    peak, min_gap, min_thw, min_ttc = 0.0, np.inf, np.inf, np.inf
    for f in range(f0, f_end + 1):
        if k is None:                                        # the recorded driver
            i = f - tF["first"]
            s, v, a = float(tF["s_front"][i]), float(tF["v"][i]), float(tF["a"][i])
        lead = R.leader(f, lane, s, F)
        if k is not None:
            a = KV * (v_set - v)
            if lead and lead[0] <= RANGE:
                a = min(a, K1 * (lead[0] - S0 - T * v) + K2 * (lead[1] - v))
            j = f - tT["first"]
            if f >= f_rec and tT["laneId"][j] != lane:       # anticipated leader, bounded braking
                gap_t = float(tT["s_rear"][j]) - s
                a = min(a, max(A_ANT, K1 * (gap_t - S0 - T * v) + K2 * (float(tT["v"][j]) - v)))
            a = float(np.clip(a, A_MIN, A_MAX))
        peak = max(peak, -a)
        if lead:
            gap, vl = lead
            min_gap = min(min_gap, gap)
            if v > 1.0:
                min_thw = min(min_thw, gap / v)
            if v - vl > 0.1:
                min_ttc = min(min_ttc, gap / (v - vl))
        if k is not None:
            v = max(0.0, v + a * DT)
            s += v * DT
    return peak, min_gap, min_thw, min_ttc


def pct(v, qs):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {f"p{q}": round(float(np.percentile(v, q)), 3) for q in qs} if len(v) else {}


def summarize(peak, gap, thw, ttc):
    if len(peak) == 0:
        return {"scenarios": 0}
    return {"scenarios": len(peak), "over_2.0": int((peak > 2.0).sum()), "over_3.5": int((peak > 3.5).sum()),
            "contact": int((gap <= 0).sum()),
            "peak_decel": {"mean": round(float(peak.mean()), 3), **pct(peak, (50, 75, 90, 95, 99, 100))},
            "min_time_gap_s": pct(thw, (1, 5, 10, 50)), "min_ttc_s": pct(ttc, (1, 5, 10, 50)),
            "ttc_under_2s": int((ttc < 2.0).sum())}


def main(ds: str = "highd"):
    root = HERE.parent / ds
    raw, prepared = root / "data" / "data", root / "data" / "prepared"
    ph.RAW = raw                                             # prepare_highd reads the same layout for exiD
    test = np.load(prepared / "test.npz")
    recs = sorted(set(test["rec"].tolist()))
    rows, skipped = [], {}
    n_lc = 0
    for r in recs:
        scen = lc_scenarios(ph.Recording(r))
        want = test["tv"][(test["rec"] == r) & (test["label"] != 0)]
        assert [s[0] for s in scen] == want.tolist(), f"recording {r}: scenarios differ from test.npz"
        n_lc += len(scen)
        R = Rec25(raw, r)
        for tv, f_c, lab, F in scen:
            w = window(R, tv, F, f_c)
            if isinstance(w, str):
                skipped[w] = skipped.get(w, 0) + 1
                continue
            res = {k: drive(R, tv, F, f_c, *w, k) for k in (None,) + ARMS}
            rows.append((r, tv, F, f_c, lab, (w[1] - f_c) / FPS, res))
        print(f"recording {r}: {len(scen)} lane changes, {len(rows)} eligible so far", flush=True)

    lab = np.array([x[4] for x in rows])
    arr = {k: np.array([x[6][k] for x in rows], float) for k in (None,) + ARMS}      # (N, 4)
    out = {"dataset": ds, "recordings": recs, "lane_change_scenarios": n_lc, "eligible": len(rows),
           "skipped": skipped, "right_lc": int((lab == 1).sum()), "left_lc": int((lab == 2).sum()),
           "controller": {"K1": K1, "K2": K2, "S0": S0, "KV": KV, "range_m": RANGE, "a_min": A_MIN, "a_max": A_MAX,
                          "a_anticipation": A_ANT, "T": "recorded time gap at the start", "dt": DT},
           "recorded_driver": summarize(*arr[None].T), "arms": {}}
    for k in ARMS:
        out["arms"][f"k{k}"] = {"all": summarize(*arr[k].T),
                                "right_lc": summarize(*arr[k][lab == 1].T), "left_lc": summarize(*arr[k][lab == 2].T),
                                "peak_decel_cut_vs_reactive": {
                                    "mean": round(float((arr[0][:, 0] - arr[k][:, 0]).mean()), 3),
                                    **pct(arr[0][:, 0] - arr[k][:, 0], (50, 90, 99))}}
    n_r = out["arms"]["k0"]["all"]["over_2.0"]
    n_o = out["arms"][f"k{RULE_K}"]["all"]["over_2.0"]
    out["rule"] = {"k_s": RULE_K, "threshold_mps2": RULE_DECEL, "reactive_events": n_r, "oracle_events": n_o,
                   "enough_events": n_r >= RULE_MIN_EVENTS, "halved": n_o <= 0.5 * n_r,
                   "go": bool(n_r >= RULE_MIN_EVENTS and n_o <= 0.5 * n_r)}
    out["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res_dir = root / "results" / "hil"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "cutin_headroom.json").write_text(json.dumps(out, indent=1))
    keep = root / "data" / "replay"
    keep.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(keep / "cutin_headroom_scenarios.npz",
                        rec=[x[0] for x in rows], tv=[x[1] for x in rows], follower=[x[2] for x in rows],
                        cross_frame=[x[3] for x in rows], label=lab, seconds_after=[x[5] for x in rows],
                        recorded=arr[None], **{f"k{k}": arr[k] for k in ARMS})
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
