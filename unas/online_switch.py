"""The device checks its own update on the delayed labels (2026-11-19, designed after the leave-one-location-out result).

Fixed before this script existed (local notes, 2026-11-19 23:01). The device keeps the flashed last layer next to the
updated one (the update of unas/online_site_pilot.py, unchanged). For every released label it notes whether each
layer's class for that window, given at the window's time, was right; for the current frame it gives the updated
layer's classes only if, over the last N released labels, the updated layer was right more often than the flashed one
(ties and the first N labels: the flashed layer). N in {256, 1024, 4096}.

  tune    N on validation recordings: highD arm K at exiD's sites, and the left-out networks of unas/online_lolo.py
          at their own locations; largest mean windows correct among the N that keep correct 2 s before within 5
          points of the flashed layer and runs of 0.6 s or more per vehicle hour not above it in both
  final   the selected N once on the test recordings: (T1) highD arm K at exiD's sites, (T3) exiD's own arm K, (T2)
          the left-out networks
  fresh   the selected N on highD's validation recordings 51-55 (no online study has scored them): exiD arm K at
          highD's sites and highD's own arm K
Criterion (all): T1 windows correct >= the update's - 1 point; T2 >= left-out fixed + 3 points and every location >=
its fixed - 1 point; T3 >= fixed - 0.5 point; F >= max(fixed, update) - 0.5 point for both networks; everywhere correct
2 s before >= flashed - 5 points and runs of 0.6 s or more per vehicle hour <= flashed. T1-T3 use test recordings
already scored by other methods; only F is untouched data.

WSL (dmir_nas venv), one process per phase, study and seed (STUDY, SEEDS, OUT_NAME):
  STUDY=site SEEDS=100 OUT_NAME=online_switch_tune_site_s100.jsonl ~/dmir_nas/bin/python unas/online_switch.py tune
  .venv\\Scripts\\python unas/online_switch.py --report
Records: datasets/exid/results/seeds/online_switch*.jsonl; report: datasets/exid/results/hil/online_switch.json.
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("TRACK_DATASET", "exid")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_lolo as lolo                                   # noqa: E402  (numpy only at import)
import online_site_pilot as site                             # noqa: E402
import online_site_reverse as rev                            # noqa: E402
import track_stream as ts                                    # noqa: E402
import track_stream_exid as tsx                              # noqa: E402

GRID = (256, 1024, 4096)
SETTING = next(s for s in site.SETTINGS if s["id"] == "G_eta0.003_lam0.0")
SIDX = site.SETTINGS.index(SETTING)
RES = site.RESULTS
OUT = RES / "seeds" / os.environ.get("OUT_NAME", "online_switch.jsonl")
SUMMARY = RES / "hil" / "online_switch.json"
EARLY = 0.05
SEEDS = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]


def switched(w: np.ndarray, fixed: np.ndarray, adapted: np.ndarray, n_last: int):
    """Classes of one location's stream (device order) with the layer chosen frame by frame on the last n_last
    released labels; labels are released as in online_site_pilot.adapt_site."""
    label, known, t_abs, rec = w[:, 3], w[:, 4].astype(bool), w[:, 5], w[:, 6]
    out = np.empty(len(w), np.int8)
    pend, hist = collections.deque(), collections.deque()
    sf = sa = used = 0
    cut = np.flatnonzero((np.diff(t_abs) != 0) | (np.diff(rec) != 0)) + 1
    for g0, g1 in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(w)]])):
        r, t = rec[g0], t_abs[g0]
        while pend and rec[pend[0]] != r:
            pend.popleft()
        while pend and t_abs[pend[0]] + site.OFFSET <= t:
            i = pend.popleft()
            cf, ca = int(fixed[i] == label[i]), int(adapted[i] == label[i])
            hist.append((cf, ca))
            sf, sa = sf + cf, sa + ca
            if len(hist) > n_last:
                of, oa = hist.popleft()
                sf, sa = sf - of, sa - oa
        use = len(hist) >= n_last and sa > sf
        out[g0:g1] = adapted[g0:g1] if use else fixed[g0:g1]
        used += (g1 - g0) * use
        pend.extend(int(i) for i in range(g0, g1) if known[i])
    return out, used / len(w)


def per_location_exid(split: str, pred: np.ndarray) -> dict:
    z = np.load(tsx.tracks_file(split))
    lens, cross = z["lens"].astype(np.int64), json.loads(str(z["cross"]))
    w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
    out = {}
    for loc in sorted(set(z["loc"].tolist())):
        tr = np.flatnonzero(z["loc"] == loc)
        p = np.concatenate([pred[w_start[t]:w_start[t] + lens[t] - ts.IN_LEN + 1] for t in tr])
        out[str(loc)] = site.compact(ts.measures_of(p, lens[tr], [cross[t] for t in tr]))
    return out


def plan(phase: str, study: str) -> list:
    """(network name, scored split, splits streamed) for a phase and study."""
    table = {("tune", "site"): [("highD_K", "val", 2)], ("tune", "lolo"): [("left_out", "val", 2)],
             ("final", "site"): [("highD_K", "test", 3), ("exiD_K", "test", 3)], ("final", "lolo"): [("left_out", "test", 3)],
             ("fresh", "highd"): [("exiD_K", "val", 2), ("highD_K", "val", 2)]}
    return table[(phase, study)]


def run(phase: str, study: str) -> None:
    import input_scaling_check as isc

    keras = isc.keras
    grid = GRID if phase == "tune" else (selection(records())["selected"],)
    if grid[0] is None:
        raise SystemExit("no N selected yet")
    done = {(r["phase"], r["study"], r["net"], r["seed"]) for r in records()}
    highd = study == "highd"
    t0 = time.perf_counter()
    by_loc_all = rev.windows_by_location() if highd else site.windows_by_location()
    loc_of_rec = {int(r): loc for loc, w in by_loc_all.items() for r in set(w[:, 6].tolist())} if highd else None
    print(f"streams built in {time.perf_counter() - t0:.0f} s", flush=True)
    for name, scored, n_splits in plan(phase, study):
        todo = [s for s in SEEDS if (phase, study, name, s) not in done]
        if not todo:
            continue
        by_loc = {loc: w[w[:, 0] < n_splits] for loc, w in by_loc_all.items()}
        rank = site.SPLITS.index(scored)
        tracks_file = ts.tracks_file if highd else tsx.tracks_file
        n_scored = int((np.load(tracks_file(scored))["lens"] - ts.IN_LEN + 1).sum())
        views = None
        if name != "left_out":
            keep, arm, prep = site.NETS[name]
            views = rev.scaled_views(prep) if highd else site.scaled_views(prep)
        for seed in todo:
            t1 = time.perf_counter()
            arrs = {"fixed": np.full(n_scored, -1, np.int8), "adapted": np.full(n_scored, -1, np.int8)}
            arrs.update({f"switch_{n}": np.full(n_scored, -1, np.int8) for n in grid})
            share = {f"switch_{n}": {} for n in grid}
            if name != "left_out":
                feat, w0, b0 = site.network(keras, keep, arm, seed)
            for loc, w in by_loc.items():
                if name == "left_out":
                    sc_ = np.load(lolo.KEEP / f"scaling_without_loc{loc}.npz")
                    views = lolo.views_with(sc_["lo"], sc_["span"])
                    feat, w0, b0 = lolo.network(keras, loc, seed)
                f = site.features(feat, views, w)
                fixed = np.argmax(f @ w0 + b0, axis=1).astype(np.int8)
                gen = [seed, loc, 11] if highd else [seed, loc, SIDX]
                adapted, _ = site.adapt_site(f, w, w0, b0, SETTING, np.random.default_rng(gen))
                s = w[:, 0] == rank
                if s.any():
                    arrs["fixed"][w[s, 1]], arrs["adapted"][w[s, 1]] = fixed[s], adapted[s]
                for n in grid:
                    out, used = switched(w, fixed, adapted, n)
                    if s.any():
                        arrs[f"switch_{n}"][w[s, 1]] = out[s]
                    share[f"switch_{n}"][str(loc)] = round(used, 4)
                del f
                if name == "left_out":
                    keras.backend.clear_session()
            keras.backend.clear_session()
            assert all((a >= 0).all() for a in arrs.values()), "a scored window was not scored"
            rec = {"phase": phase, "study": study, "net": name, "seed": seed, "split": scored, "setting": SETTING["id"],
                   "measures": {}, "share_of_frames_with_the_update": share, "float32": True}
            for key, a in arrs.items():
                m = rev.split_measures(scored, a, loc_of_rec) if highd else site.split_measures(scored, a)
                if not highd:
                    m["per_location"] = per_location_exid(scored, a)
                rec["measures"][key] = m
            rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{phase} {study} {name} seed {seed}: {time.perf_counter() - t1:.0f} s", flush=True)


def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_switch*.jsonl")):
        if "smoke" not in path.name:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def mean(rows: list, key: str, measure: str, where: str = "all", loc: str | None = None):
    vals = [r["measures"][key][where][loc][measure] if loc else r["measures"][key][where][measure] for r in rows]
    return round(st.mean(vals), 5)


def selection(rows: list) -> dict:
    groups = {k: [r for r in rows if (r["phase"], r["study"]) == ("tune", k)] for k in ("site", "lolo")}
    if any(len(g) < 5 for g in groups.values()):
        return {"selected": None, "complete": False}
    cands = []
    for n in GRID:
        key = f"switch_{n}"
        wc = [mean(g, key, "windows_correct") for g in groups.values()]
        ok = all(mean(g, key, "correct_2_s_before") >= mean(g, "fixed", "correct_2_s_before") - EARLY - 1e-12
                 and mean(g, key, "runs_0.6_s_per_vehicle_hour") <= mean(g, "fixed", "runs_0.6_s_per_vehicle_hour") + 1e-9
                 for g in groups.values())
        cands.append({"N": n, "mean_windows_correct": round(st.mean(wc), 5), "site": wc[0], "lolo": wc[1], "qualifies": ok})
    qual = [c for c in cands if c["qualifies"]]
    best = max(qual or cands, key=lambda c: c["mean_windows_correct"])
    ref = {k: {"fixed": mean(g, "fixed", "windows_correct"), "adapted": mean(g, "adapted", "windows_correct")}
           for k, g in groups.items()}
    return {"selected": best["N"], "qualified": bool(qual), "candidates": cands, "validation_reference": ref, "complete": True}


def report() -> None:
    rows = records()
    sel = selection(rows)
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes 2026-11-19 23:01",
           "selection": sel}
    print("selection:", json.dumps(sel))
    n = sel.get("selected")
    if n is not None:
        key = f"switch_{n}"
        res, checks = {}, {}
        for (phase, study, net) in (("final", "site", "highD_K"), ("final", "site", "exiD_K"), ("final", "lolo", "left_out"),
                                    ("fresh", "highd", "exiD_K"), ("fresh", "highd", "highD_K")):
            g = [r for r in rows if (r["phase"], r["study"], r["net"]) == (phase, study, net)]
            if len(g) < 5:
                continue
            e = {k: {m: mean(g, k, m) for m in ("windows_correct", "correct_2_s_before", "runs_0.6_s_per_vehicle_hour")}
                 for k in ("fixed", "adapted", key)}
            e["per_location_windows_correct"] = {loc: [mean(g, k, "windows_correct", "per_location", loc) for k in ("fixed", "adapted", key)]
                                                 for loc in g[0]["measures"]["fixed"]["per_location"]}
            e["share_of_frames_with_the_update"] = {loc: round(st.mean(r["share_of_frames_with_the_update"][key][loc] for r in g), 4)
                                                    for loc in g[0]["share_of_frames_with_the_update"][key]}
            e["guard"] = (e[key]["correct_2_s_before"] >= e["fixed"]["correct_2_s_before"] - EARLY - 1e-12
                          and e[key]["runs_0.6_s_per_vehicle_hour"] <= e["fixed"]["runs_0.6_s_per_vehicle_hour"] + 1e-9)
            res[f"{phase} {study} {net}"] = e
        wc = lambda k, which: res[k][which]["windows_correct"]                     # noqa: E731
        if "final site highD_K" in res:
            k = "final site highD_K"
            checks["T1"] = wc(k, key) >= wc(k, "adapted") - 0.01 - 1e-12 and res[k]["guard"]
        if "final lolo left_out" in res:
            k = "final lolo left_out"
            locs_ok = all(v[2] >= v[0] - 0.01 - 1e-12 for v in res[k]["per_location_windows_correct"].values())
            checks["T2"] = wc(k, key) >= wc(k, "fixed") + 0.03 - 1e-12 and locs_ok and res[k]["guard"]
            checks["T2_every_location_within_1_point"] = locs_ok
        if "final site exiD_K" in res:
            k = "final site exiD_K"
            checks["T3"] = wc(k, key) >= wc(k, "fixed") - 0.005 - 1e-12 and res[k]["guard"]
        fresh = [k for k in ("fresh highd exiD_K", "fresh highd highD_K") if k in res]
        if len(fresh) == 2:
            checks["F"] = all(wc(k, key) >= max(wc(k, "fixed"), wc(k, "adapted")) - 0.005 - 1e-12 and res[k]["guard"] for k in fresh)
        checks["complete"] = all(c in checks for c in ("T1", "T2", "T3", "F"))
        checks["met"] = bool(checks["complete"] and checks["T1"] and checks["T2"] and checks["T3"] and checks["F"])
        out["results"], out["criterion"] = res, checks
        for k, e in res.items():
            print(k, {x: e[x] for x in ("fixed", "adapted", key)}, "guard", e["guard"])
        print(json.dumps(checks, indent=1))
    SUMMARY.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] in ("tune", "final", "fresh"):
        run(sys.argv[1], os.environ["STUDY"])
    else:
        raise SystemExit("usage: STUDY=site|lolo|highd online_switch.py tune | final | fresh | --report")
