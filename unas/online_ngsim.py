"""On-device learning at the NGSIM sites (US-101, I-80): the fresh-site test of the frozen method (2026-11-26).

The devices, update and check are those of unas/online_site_pilot.py and unas/online_switch.py, unchanged; only the
data differ: NGSIM's complete tracks (datasets/ngsim/prepare_ngsim.py), one device per location (I-80 = 0, US-101 = 1)
running through its training, validation and test recordings in time order, every window in the order of the
absolute time of its last frame (Global_Time in 40-ms units, so 5 Hz frames are 5 units apart and labels are used
130 units = 26 frames later, as at exiD). Networks: highD arm K (seeds 100-104) with highD's training range, as is;
NGSIM's own arm K (unas/ngsim_train.py) with NGSIM's training range, fixed, as the reference. The rule (local notes)
is fixed before any NGSIM test recording is scored; until then only SMOKE runs (validation only) are allowed.

  WSL: SEEDS=100 OUT_NAME=online_ngsim_s100.jsonl ~/dmir_nas/bin/python unas/online_ngsim.py run
  WSL: SMOKE=1 OUT_NAME=online_ngsim_smoke.jsonl ~/dmir_nas/bin/python unas/online_ngsim.py run   # validation only
Output: datasets/ngsim/results/seeds/online_ngsim*.jsonl.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_site_pilot as site                             # noqa: E402

ts = site.ts
REPO = site.REPO
PROBE = REPO / "firmware" / "build" / "probe"
NG_KEEP = REPO / "datasets" / "ngsim" / "data" / "track_train"
OUT = REPO / "datasets" / "ngsim" / "results" / "seeds" / os.environ.get("OUT_NAME", "online_ngsim.jsonl")
SEEDS = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]
SMOKE = os.environ.get("SMOKE") == "1"
CHECK_N = 4096
GUARDS = [int(v) for v in os.environ.get("GUARDS", "1").split(",")]   # 1 = the device learns from the labels as given
SETTING = next(s for s in site.SETTINGS if s["id"] == "G_eta0.003_lam0.0")
SIDX = site.SETTINGS.index(SETTING)


def tracks_file(split: str = "test") -> Path:
    return PROBE / f"ngsim_tracks_{split}.npz"


site.tsx.tracks_file = tracks_file                          # site.scaled_views and site.split_measures read NGSIM


def windows_by_location() -> dict:
    """As site.windows_by_location, for NGSIM: split, split-local window, track, class, label known, absolute time
    (40-ms units), recording."""
    rows = collections.defaultdict(list)
    for rank, split in enumerate(site.SPLITS):
        z = np.load(tracks_file(split))
        lens = z["lens"].astype(np.int64)
        w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
        cross = json.loads(str(z["cross"]))
        for t in range(len(lens)):
            n, r, loc = int(lens[t]), int(z["rec"][t]), int(z["loc"][t])
            k, _, _, to_next, next_dir, _, _ = ts.phases(n, cross[t])
            known = (to_next <= ts.HORIZON) | ((n - 1) - k >= ts.HORIZON)
            label = np.where(to_next <= ts.HORIZON, next_dir, 0)
            t_abs = (int(z["t_first_ms"][t]) + 200 * k) // 40
            m = len(k)
            rows[loc].append(np.stack([np.full(m, rank), w_start[t] + np.arange(m), np.full(m, t), label,
                                       known.astype(np.int64), t_abs, np.full(m, r)], axis=1))
    out = {}
    for loc, parts in rows.items():
        a = np.concatenate(parts)
        order = np.lexsort((a[:, 2], a[:, 5], a[:, 6], a[:, 0]))
        out[loc] = a[order]
    return out


def scaled_views_from(lo: np.ndarray, span: np.ndarray) -> dict:
    """As site.scaled_views, with an explicit range (highD's for highD networks, NGSIM's for its own)."""
    views = {}
    for split in site.SPLITS:
        z = np.load(tracks_file(split))
        feats = ((z["feats"] - lo) / span).astype(np.float32)
        lens = z["lens"].astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        first = np.concatenate([a + np.arange(n - ts.IN_LEN + 1) for a, n in zip(starts, lens)])
        views[split] = (np.lib.stride_tricks.sliding_window_view(feats, ts.IN_LEN, axis=0), first)
    return views


def highd_range(prep: Path):
    raw = np.load(prep / "train.npz")["feats"].astype(np.float32)
    lo = raw.reshape(-1, raw.shape[-1]).min(0)
    return lo, raw.reshape(-1, raw.shape[-1]).max(0) - lo + 1e-12


def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_ngsim*.jsonl")):
        if "smoke" not in path.name:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def run() -> None:
    import input_scaling_check as isc
    import online_guard as og                                # its relabelling: hysteresis g on the lane sequence
    import online_switch as osw

    keras = isc.keras
    scored = ("val",) if SMOKE else ("val", "test")
    by_loc = windows_by_location()
    n_split = {sp: int((np.load(tracks_file(sp))["lens"] - ts.IN_LEN + 1).sum()) for sp in scored}
    ranks = {sp: site.SPLITS.index(sp) for sp in scored}
    keep, arm, prep = site.NETS["highD_K"]
    views_h = scaled_views_from(*highd_range(prep))
    sc = np.load(NG_KEEP / "scaling.npz")
    views_n = scaled_views_from(sc["lo"], sc["span"])
    labs = {}
    for g in GUARDS:                                         # the device's labels: NGSIM's lane sequence with guard g
        if g > 1:
            labs[g] = {}
            for rank, split in enumerate(site.SPLITS):
                base = og.split_base(split)
                lb, kn, _ = og.relabel(split, 0.0, g, np.zeros(int(base["lens"].sum())), base)
                labs[g][rank] = (lb.astype(np.int8), kn.astype(bool))
    done = {(r["seed"],) for r in records()} if not SMOKE else set()
    for seed in (SEEDS[:1] if SMOKE else SEEDS):
        if (seed,) in done:
            continue
        t0 = time.perf_counter()
        feat, w0, b0 = site.network(keras, keep, arm, seed)
        featn, wn0, bn0 = site.network(keras, NG_KEEP, "K", seed)
        keys = ["fixed", "own"] + [f"{k}_g{g}" for g in GUARDS for k in ("update", "check")]
        cls = {key: {sp: np.full(n, -1, np.int8) for sp, n in n_split.items()} for key in keys}
        info = {}
        for loc, w in by_loc.items():
            f = site.features(feat, views_h, w)
            fx = np.argmax(f @ w0 + b0, axis=1).astype(np.int8)
            per_g = {}
            for g in GUARDS:
                wd = w if g == 1 else og.with_labels(w, labs[g])
                site.OFFSET = (ts.HORIZON + g - 1) * 5
                adapted, ad_info = site.adapt_site(f, wd, w0, b0, SETTING, np.random.default_rng([seed, loc, SIDX]))
                check, used = osw.switched(wd, fx, adapted, CHECK_N)
                site.OFFSET = ts.HORIZON * 5
                per_g[g] = (adapted, check)
                info[f"{loc}_g{g}"] = {**ad_info, "share_of_frames_with_the_update": round(used, 4),
                                       "labels_known": int(wd[:, 4].sum()), "windows": int(len(w))}
            del f
            fn = site.features(featn, views_n, w)
            own = np.argmax(fn @ wn0 + bn0, axis=1).astype(np.int8)
            del fn
            for sp, rk in ranks.items():
                s = w[:, 0] == rk
                cls["fixed"][sp][w[s, 1]], cls["own"][sp][w[s, 1]] = fx[s], own[s]
                for g, (adapted, check) in per_g.items():
                    cls[f"update_g{g}"][sp][w[s, 1]], cls[f"check_g{g}"][sp][w[s, 1]] = adapted[s], check[s]
        keras.backend.clear_session()
        rec = {"seed": seed, "setting": SETTING["id"], "check_n": CHECK_N, "guards": GUARDS, "info": info, "float32": True,
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        for sp in scored:
            assert all((cls[k][sp] >= 0).all() for k in cls)
            rec[sp] = {k: site.split_measures(sp, cls[k][sp]) for k in cls}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"seed {seed}: {time.perf_counter() - t0:.0f} s", flush=True)    # no result is printed while running


# ------------------------------------------------------------------------------------------------- the report
GAP_MIN, HALF, KEEP_CHECK, EARLY = 0.10, 0.5, 0.01, 0.05     # the criterion (local notes)
SUMMARY = REPO / "datasets" / "ngsim" / "results" / "hil" / "online_ngsim.json"


def report() -> None:
    import statistics as st

    rows = [r for r in records() if "test" in r]
    if len(rows) < 5:
        raise SystemExit(f"{len(rows)} of 5 seeds")
    g_frozen = int(os.environ.get("FROZEN_GUARD", "1"))
    keys = [k for k in rows[0]["test"]]
    mean = lambda sp, k, m, where="all": round(st.mean(                                       # noqa: E731
        (r[sp][k]["all"][m] if where == "all" else r[sp][k]["per_recording"][where][m]) for r in rows), 5)
    res = {sp: {k: {m: mean(sp, k, m) for m in ("windows_correct", "correct_2_s_before", "runs_0.6_s_per_vehicle_hour")}
                for k in keys} for sp in ("val", "test")}
    recs = sorted(rows[0]["test"]["fixed"]["per_recording"])
    res["test_per_recording"] = {rr: {k: mean("test", k, "windows_correct", rr) for k in keys} for rr in recs}
    t_ = res["test"]
    fx, own = t_["fixed"]["windows_correct"], t_["own"]["windows_correct"]
    ck, up = t_[f"check_g{g_frozen}"], t_[f"update_g{g_frozen}"]
    gap = own - fx
    crit = {"frozen_guard": g_frozen, "gap": round(gap, 5), "N1_gap_exists": gap >= GAP_MIN - 1e-12,
            "N2_gain": round(ck["windows_correct"] - fx, 5),
            "N2_share_of_gap": round((ck["windows_correct"] - fx) / gap, 4) if gap > 0 else None,
            "N2_every_location": all(res["test_per_recording"][rr][f"check_g{g_frozen}"] > res["test_per_recording"][rr]["fixed"]
                                     for rr in recs),
            "N3_correct_2_s_before": [t_["fixed"]["correct_2_s_before"], ck["correct_2_s_before"]],
            "N3_runs": [t_["fixed"]["runs_0.6_s_per_vehicle_hour"], ck["runs_0.6_s_per_vehicle_hour"]],
            "N4_check_minus_update": round(ck["windows_correct"] - up["windows_correct"], 5)}
    crit["N2_met"] = bool(gap > 0 and ck["windows_correct"] - fx >= HALF * gap - 1e-12 and crit["N2_every_location"])
    crit["N3_met"] = bool(ck["correct_2_s_before"] >= t_["fixed"]["correct_2_s_before"] - EARLY - 1e-12
                          and ck["runs_0.6_s_per_vehicle_hour"] <= t_["fixed"]["runs_0.6_s_per_vehicle_hour"] + 1e-9)
    crit["N4_met"] = bool(ck["windows_correct"] >= up["windows_correct"] - KEEP_CHECK - 1e-12)
    crit["applies"] = bool(crit["N1_gap_exists"])
    crit["met"] = bool(crit["applies"] and crit["N2_met"] and crit["N3_met"] and crit["N4_met"])
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes (NGSIM rule)",
           "results": res, "criterion": crit}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(out, indent=1))
    print(json.dumps(crit, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        run()
    elif "--report" in sys.argv:
        report()
    else:
        raise SystemExit("usage: online_ngsim.py run | --report")
