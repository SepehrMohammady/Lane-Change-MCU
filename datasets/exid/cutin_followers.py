"""How do recorded followers react to a lane change in front of them? exiD test split (2026-11-11).

Descriptive, recorded drivers only (no simulated controller, no network): for every lane change of the
test split, the vehicle directly behind the target vehicle (TV) at the crossing frame (TV's rearId) is
followed from 7 s before the crossing to 5 s after it (at least 3 s after it; the window of
datasets/highd/cutin_headroom.py). Reported: the follower's peak deceleration, TV's speed minus the
follower's at the crossing, the time gap at the crossing and the minimum time to collision while TV is
the follower's leader. It answers whether these lane changes hold a conflict that an earlier prediction
could resolve for the follower.

  .venv\\Scripts\\python datasets/exid/cutin_followers.py
Output: datasets/exid/results/hil/cutin_followers.json (aggregates only).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FPS = 25
PRE, POST, POST_MIN = 7 * FPS, 5 * FPS, 3 * FPS
COLS = ["trackId", "frame", "lonVelocity", "lonAcceleration", "leadId", "rearId", "leadDHW", "leadTHW", "leadTTC"]


def collect():
    idx = pd.read_csv(ROOT / "data" / "prepared" / "index.csv")
    idx = idx[(idx.split == "test") & (idx.label != 0)]
    rows, skipped = [], {}
    for rec, g in idx.groupby("rec"):
        tr = pd.read_csv(ROOT / "data" / "raw" / "data" / f"{rec:02d}_tracks.csv", usecols=COLS)
        T = {int(t): d.sort_values("frame").reset_index(drop=True) for t, d in tr.groupby("trackId", sort=False)}
        for s in g.itertuples():
            tv, f_c = int(s.tv), int(s.cross_frame)
            d = T[tv]
            r = d[d.frame == f_c]
            F = int(r.rearId.iloc[0]) if len(r) and pd.notna(r.rearId.iloc[0]) else -1
            if F < 0 or F not in T:
                why = "no follower"
            else:
                dF = T[F]
                f_end = min(f_c + POST, int(dF.frame.iloc[-1]), int(d.frame.iloc[-1]))
                why = ("follower not tracked 7 s before" if int(dF.frame.iloc[0]) > f_c - PRE
                       else "less than 3 s after the crossing" if f_end < f_c + POST_MIN else "")
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            w = dF[(dF.frame >= f_c - PRE) & (dF.frame <= f_end)]
            at = dF[dF.frame == f_c].iloc[0]
            after = w[(w.frame >= f_c) & (w.leadId == tv)]
            ttc = after.leadTTC[after.leadTTC > 0].min() if len(after) else np.nan
            rows.append((int(s.label), str(s.kind), float(-w.lonAcceleration.min()),
                         float(r.lonVelocity.iloc[0] - at.lonVelocity),
                         float(at.leadDHW / max(at.lonVelocity, 1.0)) if at.leadId == tv else np.nan, float(ttc)))
        print(f"recording {rec}: {len(g)} lane changes", flush=True)
    return len(idx), rows, skipped


def summarize(a: np.ndarray) -> dict:
    q = lambda v, p: round(float(np.nanpercentile(v, p)), 2)                     # noqa: E731
    return {"scenarios": len(a),
            "peak_decel": {"p50": q(a[:, 0], 50), "p90": q(a[:, 0], 90), "p99": q(a[:, 0], 99),
                           "max": round(float(np.nanmax(a[:, 0])), 2)},
            "over_1.0": int((a[:, 0] > 1.0).sum()), "over_2.0": int((a[:, 0] > 2.0).sum()),
            "over_3.5": int((a[:, 0] > 3.5).sum()),
            "tv_minus_follower_speed": {"p10": q(a[:, 1], 10), "p50": q(a[:, 1], 50), "p90": q(a[:, 1], 90)},
            "tv_slower": int((a[:, 1] < 0).sum()),
            "time_gap_at_crossing_s": {"p10": q(a[:, 2], 10), "p50": q(a[:, 2], 50)},
            "min_ttc_under_3s": int((a[:, 3] < 3).sum()), "min_ttc_under_2s": int((a[:, 3] < 2).sum())}


def main():
    n, rows, skipped = collect()
    lab = np.array([r[0] for r in rows])
    kind = np.array([r[1] for r in rows])
    a = np.array([r[2:] for r in rows], float)
    out = {"dataset": "exid", "split": "test", "lane_change_scenarios": n, "eligible": len(rows), "skipped": skipped,
           "all": summarize(a), "right_lc": summarize(a[lab == 1]), "left_lc": summarize(a[lab == 2]),
           "by_kind": {k: summarize(a[kind == k]) for k in sorted(set(kind.tolist()))},
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    res = ROOT / "results" / "hil"
    res.mkdir(parents=True, exist_ok=True)
    (res / "cutin_followers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("lane_change_scenarios", "eligible", "skipped", "all")}, indent=1))


if __name__ == "__main__":
    main()
