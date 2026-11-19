"""Online learning at highD sites: exiD networks that keep learning on the device (2026-11-19).

The reverse direction of unas/online_site_pilot.py, as a confirmation without tuning: exiD's arm K networks (seeds
100-104, inputs with exiD's training range, as deployed at exiD) run at highD's locations, one device per location
going through its training, validation and test recordings in recording order (location 6 has test recordings only:
its device starts cold); each window's label is used 26 frames later; the update is the setting the forward pilot
selected (eta 0.003, lambda 0), unchanged. Reference: highD's own arm K, fixed. Fixed before the script existed
(local notes, 2026-11-19 18:19).

Criterion on highD's 5 test recordings: (a) gain in windows correct at least half the gap to highD's own arm K and at
least 3 points; (b) a gain on at least 4 of the 5 recordings; (c) correct 2 s before at most 5 points below the fixed
layer's; (d) runs of false calls of 0.6 s or more per vehicle hour not above the fixed layer's.

  WSL: SEEDS=100 OUT_NAME=online_reverse_s100.jsonl ~/dmir_nas/bin/python unas/online_site_reverse.py run
  .venv\\Scripts\\python unas/online_site_reverse.py --report
Output: datasets/highd/results/seeds/online_reverse*.jsonl, datasets/highd/results/hil/online_reverse.json.
"""
from __future__ import annotations

import collections
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
import online_site_pilot as site                             # noqa: E402
import track_stream as ts                                    # noqa: E402  (highD track measures and files)

REPO = site.REPO
META = REPO / "datasets" / "highd" / "data" / "data"
RES = REPO / "datasets" / "highd" / "results"
OUT = RES / "seeds" / os.environ.get("OUT_NAME", "online_reverse.jsonl")
SUMMARY = RES / "hil" / "online_reverse.json"
SETTING = {"id": "G_eta0.003_lam0.0", "family": "G", "eta": 0.003, "lam": 0.0}
HALF, MIN_GAIN, WINS, EARLY = 0.5, 0.03, 4, 0.05


def windows_by_location() -> dict:
    """As online_site_pilot.windows_by_location, for highD's complete tracks (not cut, so seg_start = 0)."""
    first5, loc_of = {}, {}
    rows = collections.defaultdict(list)
    for rank, split in enumerate(site.SPLITS):
        z = np.load(ts.tracks_file(split))
        lens = z["lens"].astype(np.int64)
        w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
        cross = json.loads(str(z["cross"]))
        for r in sorted(set(z["rec"].tolist())):
            with open(META / f"{r:02d}_recordingMeta.csv", newline="") as fh:
                loc_of[r] = int(next(csv.DictReader(fh))["locationId"])
            with open(META / f"{r:02d}_tracksMeta.csv", newline="") as fh:
                for row in csv.DictReader(fh):
                    first5[(r, int(row["id"]))] = -(-int(float(row["initialFrame"])) // 5) * 5
        for t in range(len(lens)):
            n, r = int(lens[t]), int(z["rec"][t])
            k, _, _, to_next, next_dir, _, _ = ts.phases(n, cross[t])
            known = (to_next <= ts.HORIZON) | ((n - 1) - k >= ts.HORIZON)
            label = np.where(to_next <= ts.HORIZON, next_dir, 0)
            t_abs = first5[(r, int(z["tv"][t]))] + 5 * k
            m = len(k)
            rows[loc_of[r]].append(np.stack([np.full(m, rank), w_start[t] + np.arange(m), np.full(m, t), label,
                                             known.astype(np.int64), t_abs, np.full(m, r)], axis=1))
    out = {}
    for loc, parts in rows.items():
        a = np.concatenate(parts)
        out[loc] = a[np.lexsort((a[:, 2], a[:, 5], a[:, 6], a[:, 0]))]
    return out


def scaled_views(prep: Path) -> dict:
    raw = np.load(prep / "train.npz")["feats"].astype(np.float32)
    lo = raw.reshape(-1, raw.shape[-1]).min(0)
    span = raw.reshape(-1, raw.shape[-1]).max(0) - lo + 1e-12
    views = {}
    for split in site.SPLITS:
        z = np.load(ts.tracks_file(split))
        feats = ((z["feats"] - lo) / span).astype(np.float32)
        lens = z["lens"].astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        first = np.concatenate([a + np.arange(n - ts.IN_LEN + 1) for a, n in zip(starts, lens)])
        views[split] = (np.lib.stride_tricks.sliding_window_view(feats, ts.IN_LEN, axis=0), first)
    return views


def split_measures(split: str, pred_split: np.ndarray, loc_of_rec: dict) -> dict:
    z = np.load(ts.tracks_file(split))
    lens = z["lens"].astype(np.int64)
    cross = json.loads(str(z["cross"]))
    w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
    res = {"all": site.compact(ts.measures_of(pred_split, lens, cross)), "per_recording": {}, "per_location": {}}
    groups = [("per_recording", r, z["rec"] == r) for r in sorted(set(z["rec"].tolist()))]
    groups += [("per_location", loc, np.isin(z["rec"], [r for r, lc in loc_of_rec.items() if lc == loc]))
               for loc in sorted(set(loc_of_rec[r] for r in set(z["rec"].tolist())))]
    for key, name, mask in groups:
        tr = np.flatnonzero(mask)
        p = np.concatenate([pred_split[w_start[t]:w_start[t] + lens[t] - ts.IN_LEN + 1] for t in tr])
        res[key][str(name)] = site.compact(ts.measures_of(p, lens[tr], [cross[t] for t in tr]))
    return res


def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_reverse*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def run() -> None:
    import input_scaling_check as isc

    keras = isc.keras
    seeds = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]
    done = {(r["net"], r["seed"]) for r in records()}
    by_loc = windows_by_location()
    loc_of_rec = {int(w[0, 6]): loc for loc, w in by_loc.items()}
    for loc, w in by_loc.items():
        for r in set(w[:, 6].tolist()):
            loc_of_rec[int(r)] = loc
    rank = site.SPLITS.index("test")
    n_test = int((np.load(ts.tracks_file("test"))["lens"] - ts.IN_LEN + 1).sum())
    for name in ("exiD_K", "highD_K"):
        keep, arm, prep = site.NETS[name]
        views = scaled_views(prep)
        for seed in seeds:
            if (name, seed) in done:
                continue
            t1 = time.perf_counter()
            feat, w0, b0 = site.network(keras, keep, arm, seed)
            fixed, adapted = np.full(n_test, -1, np.int8), np.full(n_test, -1, np.int8)
            info = {}
            for loc, w in by_loc.items():
                f = site.features(feat, views, w)
                sc = w[:, 0] == rank
                if sc.any():
                    fixed[w[sc, 1]] = np.argmax(f[sc] @ w0 + b0, axis=1)
                pred, inf = site.adapt_site(f, w, w0, b0, SETTING, np.random.default_rng([seed, loc, 11]))
                if sc.any():
                    adapted[w[sc, 1]] = pred[sc]
                info[str(loc)] = inf
                del f
            keras.backend.clear_session()
            assert (fixed >= 0).all() and (adapted >= 0).all()
            rec = {"net": name, "seed": seed, "setting": SETTING["id"], "split": "test",
                   "fixed": split_measures("test", fixed, loc_of_rec), "adapted": split_measures("test", adapted, loc_of_rec),
                   "info": info, "float32": True, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{name} seed {seed}: {time.perf_counter() - t1:.0f} s", flush=True)


def report() -> None:
    rows = records()
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes 2026-11-19 18:19"}
    nets = {}
    for name in ("exiD_K", "highD_K"):
        got = [r for r in rows if r["net"] == name]
        if not got:
            continue
        e = {"networks": len(got)}
        for key in site.KEYS:
            e[key] = {k: round(st.mean(r[k]["all"][key] for r in got), 4) for k in ("fixed", "adapted")}
        e["per_recording_windows_correct"] = {
            rr: [round(st.mean(r[k]["per_recording"][rr]["windows_correct"] for r in got), 4) for k in ("fixed", "adapted")]
            for rr in sorted(got[0]["fixed"]["per_recording"], key=int)}
        e["per_location_windows_correct"] = {
            lc: [round(st.mean(r[k]["per_location"][lc]["windows_correct"] for r in got), 4) for k in ("fixed", "adapted")]
            for lc in sorted(got[0]["fixed"]["per_location"], key=int)}
        nets[name] = e
        print(f"{name} ({len(got)} networks): " + "; ".join(f"{k} {e[k]['fixed']} -> {e[k]['adapted']}" for k in site.KEYS))
        print("   per recording:", e["per_recording_windows_correct"], "per location:", e["per_location_windows_correct"])
    out["nets"] = nets
    if "exiD_K" in nets and "highD_K" in nets:
        x, h = nets["exiD_K"], nets["highD_K"]
        f_, a_, u_ = x["windows_correct"]["fixed"], x["windows_correct"]["adapted"], h["windows_correct"]["fixed"]
        gains = [a - f for f, a in x["per_recording_windows_correct"].values()]
        crit = {"fixed_exiD_net": f_, "adapted_exiD_net": a_, "highD_own": u_, "gap": round(u_ - f_, 4),
                "gain": round(a_ - f_, 4), "share_of_gap": round((a_ - f_) / (u_ - f_), 4) if u_ != f_ else None,
                "a_met": (a_ - f_) >= max(HALF * (u_ - f_), MIN_GAIN) - 1e-12,
                "b_recordings_with_a_gain": int(sum(g > 0 for g in gains)), "b_met": sum(g > 0 for g in gains) >= WINS,
                "c": [x["correct_2_s_before"]["fixed"], x["correct_2_s_before"]["adapted"]],
                "c_met": x["correct_2_s_before"]["adapted"] >= x["correct_2_s_before"]["fixed"] - EARLY - 1e-12,
                "d": [x["runs_0.6_s_per_vehicle_hour"]["fixed"], x["runs_0.6_s_per_vehicle_hour"]["adapted"]],
                "d_met": x["runs_0.6_s_per_vehicle_hour"]["adapted"] <= x["runs_0.6_s_per_vehicle_hour"]["fixed"] + 1e-12,
                "complete": x["networks"] == 5 and h["networks"] == 5}
        crit["met"] = bool(crit["complete"] and crit["a_met"] and crit["b_met"] and crit["c_met"] and crit["d_met"])
        out["criterion"] = crit
        print(json.dumps(crit, indent=1))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
    else:
        raise SystemExit("usage: online_site_reverse.py run | --report")
