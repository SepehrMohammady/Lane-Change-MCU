"""Personalisation with the check on LCIR: each driver's device keeps its update only while it helps (2026-11-24).

Post hoc (local notes, 2026-11-24 15:18, written before this script): the LCIR pilot's devices (unas/online_pilot.py,
one held-out driver's drive per device, the cross-validation networks, the selected update G eta 0.003 lambda 0.01 with
the same generators) plus the check of unas/online_switch.py: the flashed and the updated layer are scored on the last
N released labels and the updated layer's class is given only while it was right more often (ties and the first N
labels: the flashed layer). N in {256, 1024, 4096} is chosen on the 14 drives looked at before (arm D) and scored once
on the 34 other drives, which the pilot's update has already been scored on.

Criterion, all of: (a) mean gain in frames correct over the flashed networks at least 0.5 point; (b) no drive more than
1 point below its flashed value; (c) timely share at most 5 points below flashed; (d) alarm runs of 0.5 s or more per
hour not above flashed. Listed: per-drive gains, the eight drivers with the lowest flashed accuracy, arm L.

  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data ~/dmir_nas/bin/python \\
    /mnt/c/Projects/PhD/DIMIR/unas/online_personal.py tune | final
  .venv\\Scripts\\python unas/online_personal.py --report
Output: datasets/dmir/results/seeds/online_personal.jsonl and datasets/dmir/results/hil/online_personal.json.
"""
from __future__ import annotations

import json
import os
from collections import deque
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_pilot as op                                    # noqa: E402  (numpy only at import)

cv = op.cv
OUT = op.REPO / "datasets" / "dmir" / "results" / "seeds" / os.environ.get("OUT_NAME", "online_personal.jsonl")
SUMMARY = op.REPO / "datasets" / "dmir" / "results" / "hil" / "online_personal.json"
SETTING = next(s for s in op.SETTINGS if s["id"] == "G_eta0.003_lam0.01")
GRID = (256, 1024, 4096)
GAIN, DROP, TIMELY = 0.005, 0.01, 0.05                       # the criterion
KEYS = ("frames_correct", "timely_share", op.A5)


def switched(stream: list, fixed: dict, adapted: dict, n_last: int) -> dict:
    """Per session the class of every window with the layer chosen on the last n_last released labels; labels are
    released as in online_pilot.adapt (frame t's label from frame t + 41 on, if usable), carried across sessions."""
    hist, sf, sa, out = deque(), 0, 0, {}
    for s in stream:
        label, usable, name = s["label"], s["usable"], s["name"]
        fx, ad = fixed[name], adapted[name]
        cls = np.empty(len(label), np.int64)
        for j in range(len(label)):
            i = j - op.DELAY
            if i >= 0 and usable[i]:
                cf, ca = int(fx[i] == label[i]), int(ad[i] == label[i])
                hist.append((cf, ca))
                sf, sa = sf + cf, sa + ca
                if len(hist) > n_last:
                    of, oa = hist.popleft()
                    sf, sa = sf - of, sa - oa
            cls[j] = ad[j] if (len(hist) >= n_last and sa > sf) else fx[j]
        out[name] = cls
    return out


def records() -> list:
    return [json.loads(line) for line in OUT.read_text().splitlines()] if OUT.exists() else []


def run(phase: str) -> None:
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters

    keras = isc.keras
    os.environ["DMIR_SCALE"] = ""
    table = cv.sessions_table()
    if phase == "tune":
        grid, arms, drives = GRID, ("D",), op.TUNE
    else:
        n = selection(records())["selected"]
        if n is None:
            raise SystemExit("tuning incomplete")
        grid, arms, drives = (n,), ("D", "L"), op.FRESH
    done = {(r["phase"], r["fold"], r["arm"], r["seed"]) for r in records()}
    for k, fold in enumerate(cv.FOLDS):
        held = [u for u in fold if u in drives]
        if not held:
            continue
        for arm in arms:
            for seed in (100, 101, 102, 103, 104):
                if (phase, k, arm, seed) in done:
                    continue
                t0 = time.perf_counter()
                feat, w0, b0 = op.network(keras, k, arm, seed)
                per = {}
                for u in held:
                    stream = op.stream_of(table, u, feat)
                    fixed = {s["name"]: np.argmax(s["f"] @ w0 + b0, -1) for s in stream}
                    rng = np.random.default_rng([k, seed, u, op.SETTINGS.index(SETTING)])
                    adapted, info = op.adapt(stream, w0, b0, SETTING, rng)
                    names = [s["name"] for s in stream]
                    rec = {"fixed": op.score(table, names, fixed)[f"user{u}LC"],
                           "adapted": op.score(table, names, adapted)[f"user{u}LC"]}
                    for n in grid:
                        rec[f"switch_{n}"] = op.score(table, names, switched(stream, fixed, adapted, n))[f"user{u}LC"]
                    per[f"user{u}"] = rec
                rec = {"phase": phase, "fold": k, "arm": arm, "seed": seed, "setting": SETTING["id"], "grid": list(grid),
                       "drives": per, "float32": True, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                keras.backend.clear_session()
                print(f"{phase} fold {k} arm {arm} seed {seed}: {len(held)} drives, {time.perf_counter() - t0:.0f} s",
                      flush=True)                            # no result is printed while running


def per_drive(rows: list, phase: str, arm: str, key: str) -> dict:
    """Per drive: mean over its five networks of the three measures, for the flashed layer and for `key`."""
    by = {}
    for r in rows:
        if (r["phase"], r["arm"]) != (phase, arm):
            continue
        for u, d in r["drives"].items():
            if key in d:
                by.setdefault(u, []).append(d)
    out = {}
    for u, ds in by.items():
        row = {"networks": len(ds)}
        for m in KEYS:
            f = [op.measures_of(d["fixed"])[m] for d in ds]
            a = [op.measures_of(d[key])[m] for d in ds]
            row[m] = None if any(v is None for v in f + a) else [round(st.mean(f), 5), round(st.mean(a), 5)]
        out[u] = row
    return out


def summary(tab: dict) -> dict:
    g = [v["frames_correct"][1] - v["frames_correct"][0] for v in tab.values()]
    out = {"drives": len(tab), "networks_per_drive": sorted({v["networks"] for v in tab.values()})}
    for m in KEYS:
        pairs = [v[m] for v in tab.values() if v[m] is not None]
        out[m] = [round(st.mean(x[0] for x in pairs), 5), round(st.mean(x[1] for x in pairs), 5)] if pairs else None
    out.update({"mean_gain": round(st.mean(g), 5), "drives_with_a_gain": int(sum(x > 0 for x in g)),
                "worst_drive_gain": round(min(g), 5), "best_drive_gain": round(max(g), 5)})
    return out


def qualifies(s: dict) -> bool:
    return (s["timely_share"][1] >= s["timely_share"][0] - TIMELY - 1e-12
            and s[op.A5][1] <= s[op.A5][0] + 1e-9)


def selection(rows: list) -> dict:
    if not any(r["phase"] == "tune" for r in rows):
        return {"selected": None, "complete": False}
    cands = []
    for n in GRID:
        s = summary(per_drive(rows, "tune", "D", f"switch_{n}"))
        cands.append({"N": n, "qualifies": qualifies(s), **s})
    if any(c["drives"] != 14 or c["networks_per_drive"] != [5] for c in cands):
        return {"selected": None, "complete": False}
    qual = [c for c in cands if c["qualifies"]]
    best = max(qual or cands, key=lambda c: c["frames_correct"][1])
    upd = summary(per_drive(rows, "tune", "D", "adapted"))
    return {"selected": best["N"], "qualified": bool(qual), "candidates": cands, "update_alone": upd, "complete": True}


def report() -> None:
    rows = records()
    sel = selection(rows)
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes 2026-11-24 15:18",
           "selection": sel}
    n = sel.get("selected")
    if n is not None and any(r["phase"] == "final" for r in rows):
        key = f"switch_{n}"
        tab = per_drive(rows, "final", "D", key)
        s = summary(tab)
        upd = summary(per_drive(rows, "final", "D", "adapted"))
        low = sorted(tab.items(), key=lambda kv: kv[1]["frames_correct"][0])[:8]
        crit = {"a_mean_gain": s["mean_gain"], "a_met": s["mean_gain"] >= GAIN - 1e-12,
                "b_worst_drive_gain": s["worst_drive_gain"], "b_met": s["worst_drive_gain"] >= -DROP - 1e-12,
                "c_timely_share": s["timely_share"], "d_alarm_runs": s[op.A5]}
        crit["c_met"] = s["timely_share"][1] >= s["timely_share"][0] - TIMELY - 1e-12
        crit["d_met"] = s[op.A5][1] <= s[op.A5][0] + 1e-9
        crit["complete"] = s["drives"] == 34
        crit["met"] = bool(crit["complete"] and crit["a_met"] and crit["b_met"] and crit["c_met"] and crit["d_met"])
        out.update({"final_D_check": s, "final_D_update_alone": upd, "criterion": crit,
                    "per_drive_frames_correct": {u: v["frames_correct"] for u, v in sorted(tab.items())},
                    "lowest_eight_flashed": {u: v["frames_correct"] for u, v in low},
                    "lowest_eight_mean_gain": round(st.mean(v["frames_correct"][1] - v["frames_correct"][0] for _, v in low), 5)})
        tl = per_drive(rows, "final", "L", key)
        if tl:
            out["final_L_check"] = summary(tl)
            out["final_L_update_alone"] = summary(per_drive(rows, "final", "L", "adapted"))
    SUMMARY.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in out if k not in ("per_drive_frames_correct",)}, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] in ("tune", "final"):
        run(sys.argv[1])
    else:
        raise SystemExit("usage: online_personal.py tune | final | --report")
