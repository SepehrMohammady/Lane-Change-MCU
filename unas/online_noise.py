"""Robustness of the self-labels: on-device learning at exiD sites when the device's own labels are wrong (2026-11-25).

Fixed before this script existed (local notes, 2026-11-25 15:02). The devices, networks (highD arm K, seeds 100-104),
update (eta 0.003, lambda 0) and check (N = 4096) are those of unas/online_site_pilot.py and unas/online_switch.py.
Simulated label errors corrupt only the labels the device learns from and compares on; scoring always uses the
dataset's labels. One noise draw per seed and location (generator [seed, location, 13], so the corrupted sets are
nested across levels):
  flip     each label-known window's label replaced, with probability p, by one of the other two classes at random
           (p = 0.05 / 0.10 / 0.20; tracking jitter)
  missed   for each track holding a lane change, with probability p all its lane-change labels become lane keeping
           (p = 0.10 / 0.25; the tracker misses the crossing)
p = 0 must reproduce the site study. Criterion on the 14 test recordings, windows correct, the update alone: (R1) flip
p = 0.10 keeps at least 75% of the clean gain over the fixed network; (R2) missed p = 0.25 keeps at least 50% of the
clean gain, and correct 2 s before the crossing stays at most 5 points below the fixed network's.

  WSL: SEEDS=100 OUT_NAME=online_noise_s100.jsonl ~/dmir_nas/bin/python unas/online_noise.py run
  .venv\\Scripts\\python unas/online_noise.py --report
Output: datasets/exid/results/seeds/online_noise*.jsonl, datasets/exid/results/hil/online_noise.json.
"""
from __future__ import annotations

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
import online_switch as osw                                  # noqa: E402

RES = site.RESULTS
OUT = RES / "seeds" / os.environ.get("OUT_NAME", "online_noise.jsonl")
SUMMARY = RES / "hil" / "online_noise.json"
SETTING = osw.SETTING
SIDX = site.SETTINGS.index(SETTING)
CHECK_N = 4096
CONFIGS = [("clean", 0.0), ("flip", 0.05), ("flip", 0.10), ("flip", 0.20), ("missed", 0.10), ("missed", 0.25)]
R1_KEEP, R2_KEEP, EARLY = 0.75, 0.50, 0.05                  # the criterion
SEEDS = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]


def corrupt(w: np.ndarray, model: str, p: float, rng) -> np.ndarray:
    """A copy of one location's stream with the label column (3) corrupted as the device would receive it."""
    wn = w.copy()
    if model == "clean" or p == 0:
        return wn
    known, lab = wn[:, 4].astype(bool), wn[:, 3]
    if model == "flip":
        u, shift = rng.random(len(wn)), rng.integers(1, 3, size=len(wn))
        hit = known & (u < p)
        wn[hit, 3] = (lab[hit] + shift[hit]) % 3
    elif model == "missed":
        key = wn[:, 0] * 10_000_000 + wn[:, 2]               # split and track: one vehicle
        lc_keys = np.unique(key[known & (lab > 0)])
        u = rng.random(len(lc_keys))
        missed = lc_keys[u < p]
        hit = np.isin(key, missed) & (lab > 0)
        wn[hit, 3] = 0
    else:
        raise ValueError(model)
    return wn


def cfg_id(model: str, p: float) -> str:
    return "clean" if model == "clean" else f"{model}_{p:g}"


def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_noise*.jsonl")):
        if "smoke" not in path.name:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def run() -> None:
    import input_scaling_check as isc

    keras = isc.keras
    done = {(r["config"], r["seed"]) for r in records()}
    t0 = time.perf_counter()
    by_loc = site.windows_by_location()
    print(f"streams built in {time.perf_counter() - t0:.0f} s", flush=True)
    rank = site.SPLITS.index("test")
    n_test = int((np.load(site.tsx.tracks_file("test"))["lens"] - site.ts.IN_LEN + 1).sum())
    keep, arm, prep = site.NETS["highD_K"]
    views = site.scaled_views(prep)
    for seed in SEEDS:
        todo = [(m, p) for m, p in CONFIGS if (cfg_id(m, p), seed) not in done]
        if not todo:
            continue
        t1 = time.perf_counter()
        feat, w0, b0 = site.network(keras, keep, arm, seed)
        fixed = np.full(n_test, -1, np.int8)
        outs = {cfg_id(m, p): {"adapted": np.full(n_test, -1, np.int8), "check": np.full(n_test, -1, np.int8)} for m, p in todo}
        info = {cfg_id(m, p): {} for m, p in todo}
        for loc, w in by_loc.items():
            f = site.features(feat, views, w)
            s = w[:, 0] == rank
            fx = np.argmax(f @ w0 + b0, axis=1).astype(np.int8)
            fixed[w[s, 1]] = fx[s]
            for m, p in todo:
                key = cfg_id(m, p)
                wn = corrupt(w, m, p, np.random.default_rng([seed, loc, 13]))
                adapted, _ = site.adapt_site(f, wn, w0, b0, SETTING, np.random.default_rng([seed, loc, SIDX]))
                check, used = osw.switched(wn, fx, adapted, CHECK_N)
                outs[key]["adapted"][w[s, 1]], outs[key]["check"][w[s, 1]] = adapted[s], check[s]
                kn = w[:, 4].astype(bool)
                info[key][str(loc)] = {"labels_changed": int((wn[kn, 3] != w[kn, 3]).sum()), "labels": int(kn.sum()),
                                       "share_of_frames_with_the_update": round(used, 4)}
            del f
        keras.backend.clear_session()
        assert (fixed >= 0).all()
        fm = site.split_measures("test", fixed)
        for m, p in todo:
            key = cfg_id(m, p)
            assert all((a >= 0).all() for a in outs[key].values())
            rec = {"config": key, "model": m, "p": p, "seed": seed, "setting": SETTING["id"], "check_n": CHECK_N,
                   "fixed": fm, "adapted": site.split_measures("test", outs[key]["adapted"]),
                   "check": site.split_measures("test", outs[key]["check"]), "info": info[key], "float32": True,
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"seed {seed}: {len(todo)} configurations, {time.perf_counter() - t1:.0f} s", flush=True)


def report() -> None:
    rows = records()
    res = {}
    for m, p in CONFIGS:
        key = cfg_id(m, p)
        got = [r for r in rows if r["config"] == key]
        if len(got) < 5:
            continue
        mean = lambda which, k: round(st.mean(r[which]["all"][k] for r in got), 5)            # noqa: E731
        res[key] = {k: {which: mean(which, k) for which in ("fixed", "adapted", "check")}
                    for k in ("windows_correct", "correct_2_s_before", "runs_0.6_s_per_vehicle_hour")}
        res[key]["labels_changed_share"] = round(sum(v["labels_changed"] for r in got for v in r["info"].values())
                                                 / sum(v["labels"] for r in got for v in r["info"].values()), 5)
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes 2026-11-25 15:02",
           "results": res}
    if all(k in res for k in ("clean", "flip_0.1", "missed_0.25")):
        wc = lambda k, which: res[k]["windows_correct"][which]                                   # noqa: E731
        clean_gain = wc("clean", "adapted") - wc("clean", "fixed")
        g1 = wc("flip_0.1", "adapted") - wc("flip_0.1", "fixed")
        g2 = wc("missed_0.25", "adapted") - wc("missed_0.25", "fixed")
        c2 = res["missed_0.25"]["correct_2_s_before"]
        crit = {"clean_gain": round(clean_gain, 5), "R1_flip_0.10_gain": round(g1, 5), "R1_share": round(g1 / clean_gain, 4),
                "R1_met": g1 >= R1_KEEP * clean_gain - 1e-12, "R2_missed_0.25_gain": round(g2, 5),
                "R2_share": round(g2 / clean_gain, 4), "R2_correct_2_s_before": [c2["fixed"], c2["adapted"]],
                "R2_met": g2 >= R2_KEEP * clean_gain - 1e-12 and c2["adapted"] >= c2["fixed"] - EARLY - 1e-12}
        crit["met"] = bool(crit["R1_met"] and crit["R2_met"])
        out["criterion"] = crit
    SUMMARY.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
    else:
        raise SystemExit("usage: online_noise.py run | --report")
