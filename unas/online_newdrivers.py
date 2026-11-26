"""Personalisation on new drivers: the confirmation of the 2026-11-24 result on drivers recorded after it (2026-11-26).

The post hoc study (unas/online_personal.py, local notes 2026-11-24 15:18) gave one device per LCIR driver, the
cross-validation networks, the update G (eta 0.003, lambda 0.01) and the check with N = 1024, and found on 34 drivers
+0.92 points of frames correct, 23 of 34 better, worst -0.87. This script applies exactly that, unchanged, to drivers
who drive the lab's simulator after it (Materials/lcir-framework/NEW_DRIVERS.md); the rule is in the local notes of
2026-11-26 and was fixed before the first new driver.

  drives     Materials/lcir-framework/new_drivers/drives.csv (local; codes only): code (N01, N02, ...), drive (LC or N),
             log (log_NNN, converted by prepare_on_this_laptop.py into data_preparation/h5/), order (1 or 2).
             A drive is left out, before any scoring, if its pace (log_NNN.txt.session.json) lies outside LCIR's range
             0.83-0.99 of real time or it covers less than 90% of LCIR's median distance measured the same way from the H5 (29.58 km lane
             changes, 14.82 km baseline; the dashboard shows 32 and 15 km).
             A driver counts if the lane-change drive counts.
  devices    as in the post hoc study: the stream is the lane-change session, then the baseline session; scored on
             the lane-change session; networks arm D of cross-validation fold (i - 1) mod 6 for driver Ni, seeds
             100-104; arm L listed.

  .venv\\Scripts\\python unas/online_newdrivers.py build      # drives.csv -> intention_sessions_new.npz (local)
  WSL: DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data ~/dmir_nas/bin/python unas/online_newdrivers.py run
  .venv\\Scripts\\python unas/online_newdrivers.py --report
Output (local, never versioned: the drives are personal data): Materials/lcir-framework/new_drivers/results/.
"""
from __future__ import annotations

import csv
import json
import os
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_pilot as op                                    # noqa: E402  (numpy only at import)
import online_personal as opp                                # noqa: E402

cv = op.cv
REPO = op.REPO
FRAMEWORK = REPO / "Materials" / "lcir-framework"
NEW = Path(os.environ.get("NEW_DRIVERS_DIR", FRAMEWORK / "new_drivers"))
LOGS = Path(os.environ.get("NEW_LOGS_DIR", FRAMEWORK / "data_collection" / "logs"))
H5 = Path(os.environ.get("NEW_H5_DIR", FRAMEWORK / "data_preparation" / "h5"))
IGNORE_EXCLUSIONS = os.environ.get("IGNORE_EXCLUSIONS") == "1"   # pipeline checks on test drives only
SESSIONS = NEW / "intention_sessions_new.npz"
OUT = NEW / "results" / os.environ.get("OUT_NAME", "online_newdrivers.jsonl")
SUMMARY = NEW / "results" / "online_newdrivers.json"
SETTING = opp.SETTING
N_CHECK = 1024
PACE = (0.83, 0.99)
DISTANCE_KM = {"LC": 29.58, "N": 14.82}                     # LCIR's medians, measured the same way (speed x time)
MIN_DRIVERS = 15
GAIN, DROP, TIMELY, ALPHA = 0.005, 0.01, 0.05, 0.05          # the criterion


REPRO_USER = int(os.environ.get("REPRO_AS_USER", "0"))     # check: rerun an LCIR driver as the 2026-11-24 study did


def fold_of(code: str) -> int:
    if REPRO_USER:
        return next(k for k, f in enumerate(cv.FOLDS) if REPRO_USER in f)
    return (int(code[1:]) - 1) % len(cv.FOLDS)


# ------------------------------------------------------------------------------------------------- the drives
def drive_checks(log: str) -> dict:
    """Pace reached (from the helper's session file) and distance covered (from the converted H5)."""
    import h5py

    js = next((p for p in (LOGS / f"{log}.txt.session.json", LOGS / f"{log}.session.json") if p.exists()), None)
    pace = json.loads(js.read_text())["pace_reached"]["pace"] if js else None
    with h5py.File(H5 / f"{log}.h5", "r") as f:
        e = f["egoVehicle"][()]
    dt = float(np.median(np.diff(e["FileTime"].astype(np.float64))))
    km = float(np.nansum(e["VehicleSpeed"]) * dt / 1000.0)
    return {"pace": pace, "km": round(km, 2), "frames": int(len(e))}


def build() -> None:
    """drives.csv -> the sessions of the drives that count, in the layout of datasets/dmir/session_stream.py."""
    sys.path.insert(0, str(REPO / "datasets" / "dmir"))
    import session_stream as ss

    ss.sr.H5DIR = H5
    rows = list(csv.DictReader(open(NEW / "drives.csv", newline="")))
    kept, report = [], []
    for r in rows:
        chk = drive_checks(r["log"])
        ok_pace = chk["pace"] is not None and PACE[0] <= chk["pace"] <= PACE[1]
        ok_km = chk["km"] >= 0.9 * DISTANCE_KM[r["drive"]]
        report.append({**r, **chk, "counts": bool(ok_pace and ok_km)})
        if (ok_pace and ok_km) or IGNORE_EXCLUSIONS:
            kept.append(r)
    lc = {r["code"] for r in kept if r["drive"] == "LC"}
    kept = [r for r in kept if r["code"] in lc]
    out = []
    for r in kept:
        ego, ll, ob, _ = ss.sr.load_session(r["log"])
        ttlc, direction, events = ss.frame_labels(ll)
        out.append((f"{r['code']}{r['drive']}", ss.sr.raw_matrix(ego, ll, ob), ttlc, direction, events))
    task = cv.TASK
    np.savez_compressed(
        SESSIONS, feats=np.concatenate([ss.to_layout(m, task).astype(np.float32) for _, m, _, _, _ in out]),
        seq_lens=np.array([len(m) for _, m, _, _, _ in out]), names=np.array([n for n, _, _, _, _ in out]),
        ttlc=np.concatenate([t for _, _, t, _, _ in out]).astype(np.float32),
        direction=np.concatenate([d for _, _, _, d, _ in out]),
        events=json.dumps({n: [[int(i), int(dd)] for i, dd in ev] for n, _, _, _, ev in out}))
    (NEW / "results").mkdir(parents=True, exist_ok=True)
    (NEW / "results" / "drives_checked.json").write_text(json.dumps(report, indent=1))
    print(f"{len(lc)} drivers count; {len(out)} sessions -> {SESSIONS}")
    for r in report:
        print(f"{r['code']} {r['drive']} {r['log']}: pace {r['pace']}, {r['km']} km, counts {r['counts']}")


def table_new() -> dict:
    """The new sessions in the form of stream_cv_check.sessions_table (same scaling and clipping)."""
    n = json.loads((cv.STREAM / f"{cv.TASK}_norm.json").read_text())
    a, b, lo, hi = (np.array(n[k]) for k in ("a", "b", "train_min", "train_max"))
    z = np.load(SESSIONS)
    lens = z["seq_lens"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    events = json.loads(str(z["events"]))
    table = {}
    for s, name in enumerate(str(v) for v in z["names"]):
        i, k = int(starts[s]), int(lens[s])
        raw = z["feats"][i:i + k]
        table[name] = {"feats": raw, "x": np.clip(raw.astype(np.float64) * a + b, lo, hi).astype(np.float32),
                       "ttlc": z["ttlc"][i:i + k], "direction": z["direction"][i:i + k], "events": events[name]}
    return table


def stream_of(table: dict, code: str, feat) -> list:
    """As online_pilot.stream_of, for a new driver: the lane-change session, then the baseline session."""
    out = []
    for name in (f"{code}LC", f"{code}N"):
        if name not in table:
            continue
        s = table[name]
        x = np.clip(s["x"], -op.BOUND, op.BOUND)
        win = np.lib.stride_tricks.sliding_window_view(x, op.W, axis=0).transpose(0, 2, 1)
        f = feat.predict(np.ascontiguousarray(win), batch_size=4096, verbose=0).astype(np.float64)
        t = s["ttlc"][op.W - 1:]
        label = np.where(t <= 4.0, s["direction"][op.W - 1:], 0).astype(np.int64)
        out.append({"name": name, "f": f, "label": label, "usable": ~np.isnan(t)})
    return out


# ------------------------------------------------------------------------------------------------- the runs
def records() -> list:
    return [json.loads(line) for line in OUT.read_text().splitlines()] if OUT.exists() else []


def run() -> None:
    import input_scaling_check as isc

    keras = isc.keras
    os.environ["DMIR_SCALE"] = ""
    table = table_new()
    codes = sorted({n[:-2] if n.endswith("LC") else n[:-1] for n in table})
    codes = [c for c in codes if f"{c}LC" in table]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["code"], r["arm"], r["seed"]) for r in records()}
    for code in codes:
        k = fold_of(code)
        for arm in ("D", "L"):
            for seed in (100, 101, 102, 103, 104):
                if (code, arm, seed) in done:
                    continue
                t0 = time.perf_counter()
                feat, w0, b0 = op.network(keras, k, arm, seed)
                stream = stream_of(table, code, feat)
                fixed = {s["name"]: np.argmax(s["f"] @ w0 + b0, -1) for s in stream}
                rng = np.random.default_rng([k, seed, REPRO_USER or 1000 + int(code[1:]), op.SETTINGS.index(SETTING)])
                adapted, info = op.adapt(stream, w0, b0, SETTING, rng)
                names = [s["name"] for s in stream]
                rec = {"fixed": op.score(table, names, fixed)[f"{code}LC"],
                       "adapted": op.score(table, names, adapted)[f"{code}LC"],
                       f"switch_{N_CHECK}": op.score(table, names, opp.switched(stream, fixed, adapted, N_CHECK))[f"{code}LC"]}
                out = {"code": code, "fold": k, "arm": arm, "seed": seed, "setting": SETTING["id"], "check_n": N_CHECK,
                       "drive": rec, "info": info, "float32": True,
                       "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(out) + "\n")
                keras.backend.clear_session()
                print(f"{code} fold {k} arm {arm} seed {seed}: {time.perf_counter() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------- the report
def per_driver(rows: list, arm: str, key: str) -> dict:
    by = {}
    for r in rows:
        if r["arm"] == arm:
            by.setdefault(r["code"], []).append(r["drive"])
    out = {}
    for code, ds in sorted(by.items()):
        row = {"networks": len(ds)}
        for m in opp.KEYS:
            f = [op.measures_of(d["fixed"])[m] for d in ds]
            a = [op.measures_of(d[key])[m] for d in ds]
            row[m] = None if any(v is None for v in f + a) else [round(st.mean(f), 5), round(st.mean(a), 5)]
        out[code] = row
    return out


def report() -> None:
    from scipy import stats

    rows = records()
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes 2026-11-26"}
    for arm in ("D", "L"):
        for key in (f"switch_{N_CHECK}", "adapted"):
            tab = per_driver(rows, arm, key)
            res[f"{arm}_{key}"] = {"per_driver": tab, "summary": opp.summary(tab) if tab else None}
    tab = res[f"D_switch_{N_CHECK}"]["per_driver"]
    complete = bool(tab) and all(v["networks"] == 5 for v in tab.values())
    if complete:
        g = np.array([v["frames_correct"][1] - v["frames_correct"][0] for v in tab.values()])
        s = res[f"D_switch_{N_CHECK}"]["summary"]
        p = float(stats.wilcoxon(g, alternative="greater").pvalue) if np.any(g != 0) else 1.0
        crit = {"drivers": len(g), "enough_drivers": len(g) >= MIN_DRIVERS, "a_mean_gain": round(float(g.mean()), 5),
                "a_wilcoxon_p_one_sided": round(p, 5), "a_met": bool(g.mean() >= GAIN - 1e-12 and p < ALPHA),
                "b_worst_drive_gain": round(float(g.min()), 5), "b_met": bool(g.min() >= -DROP - 1e-12),
                "c_timely_share": s["timely_share"], "c_met": s["timely_share"][1] >= s["timely_share"][0] - TIMELY - 1e-12,
                "d_alarm_runs": s[op.A5], "d_met": s[op.A5][1] <= s[op.A5][0] + 1e-9}
        crit["met"] = bool(crit["enough_drivers"] and crit["a_met"] and crit["b_met"] and crit["c_met"] and crit["d_met"])
        res["criterion"] = crit
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("criterion", {"complete": False}), indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] in ("build", "run"):
        {"build": build, "run": run}[sys.argv[1]]()
    else:
        raise SystemExit("usage: online_newdrivers.py build | run | --report")
