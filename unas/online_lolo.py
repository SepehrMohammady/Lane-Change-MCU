"""A second shift for on-device learning: new exiD locations for exiD networks, leave one location out (2026-11-19).

Fixed before this script existed (local notes, 2026-11-19 21:56). For each exiD location L, networks of the searched
architecture are trained with the arm K recipe of track_train_check.py (seeds 100-104) on the label-known
complete-track windows of the training recordings of the other six locations (early stopping on their validation
recordings; input range from their training tracks). A device at L runs through L's training, validation and test
recordings with L's left-out network and the update the site pilot selected (eta 0.003, lambda 0, no new tuning),
labels 26 frames later, exactly as unas/online_site_pilot.py. Every test track is classified by its location's
networks and the measures are taken over the 14 test recordings at once and per recording.

Criterion, all of: (0) exiD's own arm K networks (all seven locations, fixed; the site pilot's records) at least 3
points of windows correct above the left-out networks, fixed (else no shift to adapt: not met); (a) gain of the update
at least half that gap and at least 3 points; (b) a gain on at least 11 of the 14 recordings; (c) correct 2 s before
the crossing at most 5 points below the left-out networks fixed; (d) runs of false calls of 0.6 s or more per vehicle
hour not above theirs. Listed: per location, the share of the gap closed.

WSL (dmir_nas venv, the fork's TensorFlow), several processes may share the work with LOCS / SEEDS and their own
record files (TRAIN_OUT, OUT_NAME):
  TRACK_DATASET=exid HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/exid/data/prepared LOCS=0,1,2 \\
    TRAIN_OUT=online_lolo_train_a.jsonl ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/online_lolo.py train
  ... unas/online_lolo.py run          # the devices, after every network is trained (SEEDS=100 etc., OUT_NAME)
  .venv\\Scripts\\python unas/online_lolo.py --report
SMOKE=1 trains one location and one seed for two epochs into *_smoke files, to check the code.
Networks: datasets/exid/data/track_train_lolo/ (not versioned). Records: datasets/exid/results/seeds/online_lolo*.jsonl;
report: datasets/exid/results/hil/online_lolo.json.
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

os.environ.setdefault("TRACK_DATASET", "exid")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "datasets" / "highd"))
sys.path.insert(0, str(REPO / "datasets" / "exid"))
import online_site_pilot as site                             # noqa: E402  (numpy only at import)
import track_stream as ts                                    # noqa: E402
import track_stream_exid as tsx                              # noqa: E402

MODEL = site.MODEL
KEEP = REPO / "datasets" / "exid" / "data" / "track_train_lolo"
RES = REPO / "datasets" / "exid" / "results"
SMOKE = os.environ.get("SMOKE") == "1"
SUFFIX = "_smoke" if SMOKE else ""
TRAIN_OUT = RES / "seeds" / os.environ.get("TRAIN_OUT", f"online_lolo_train{SUFFIX}.jsonl")
OUT = RES / "seeds" / os.environ.get("OUT_NAME", f"online_lolo{SUFFIX}.jsonl")
SUMMARY = RES / "hil" / "online_lolo.json"
SETTING = next(s for s in site.SETTINGS if s["id"] == "G_eta0.003_lam0.0")
GAP_MIN, HALF, MIN_GAIN, WINS, EARLY = 0.03, 0.5, 0.03, 11, 0.05     # the criterion
LOCS = [int(v) for v in os.environ.get("LOCS", "0,1,2,3,4,5,6").split(",")]
SEEDS = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]


def scaling_without(loc: int):
    """Input range from the training tracks of the other locations."""
    z = np.load(tsx.tracks_file("train"))
    lens = z["lens"].astype(np.int64)
    rows = np.repeat(z["loc"] != loc, lens)
    f = z["feats"][rows].astype(np.float32)
    lo = f.min(0)
    return lo, f.max(0) - lo + 1e-12


def track_windows_without(split: str, lo, span, rng, loc: int):
    """track_train_check.track_windows(known_only=True) on the tracks of the other locations."""
    z = np.load(tsx.tracks_file(split))
    feats = ((z["feats"] - lo) / span).astype(np.float32)
    lens = z["lens"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    first, lab = [], []
    for a, n, cr, where in zip(starts, lens, json.loads(str(z["cross"])), z["loc"]):
        if int(where) == loc:
            continue
        k, _, _, to_next, next_dir, _, _ = ts.phases(int(n), cr)
        keep = (to_next <= ts.HORIZON) | ((int(n) - 1) - k >= ts.HORIZON)
        first.append((a + k - (ts.IN_LEN - 1))[keep])
        lab.append(np.where(to_next <= ts.HORIZON, next_dir, 0)[keep])
    first, lab = np.concatenate(first), np.concatenate(lab)
    counts = np.bincount(lab, minlength=3)
    pick = np.concatenate([rng.choice(np.flatnonzero(lab == c), size=int(counts.min()), replace=False) for c in range(3)])
    pick = pick[rng.permutation(len(pick))]
    win = np.lib.stride_tricks.sliding_window_view(feats, ts.IN_LEN, axis=0)
    return np.ascontiguousarray(win[first[pick]].transpose(0, 2, 1)), lab[pick].astype(np.int64), [int(c) for c in counts]


def net_path(loc: int, seed: int) -> Path:
    return KEEP / f"K_{MODEL}_without_loc{loc}_seed{seed}{SUFFIX}.h5"


def train_phase() -> None:
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters
    import track_train_check as ttc                          # the arm K recipe (TRACK_DATASET=exid)

    keras, sv = isc.keras, isc.sv
    assert ttc.DATASET == "exid" and "exid" in os.environ.get("HIGHD_DATA_ROOT", "")
    if SMOKE:
        sv.EPOCH_CAP = 2
    KEEP.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for path in TRAIN_OUT.parent.glob("online_lolo_train*.jsonl"):
        if ("smoke" in path.name) == SMOKE:                  # the smoke record must not count as trained
            done |= {(r["location"], r["seed"]) for r in map(json.loads, path.read_text().splitlines())}
    for loc in (LOCS[:1] if SMOKE else LOCS):
        seeds = [s for s in (SEEDS[:1] if SMOKE else SEEDS) if (loc, s) not in done]
        if not seeds:
            continue
        lo, span = scaling_without(loc)
        np.savez(KEEP / f"scaling_without_loc{loc}{SUFFIX}.npz", lo=lo, span=span)
        x, y, counts = track_windows_without("train", lo, span, np.random.default_rng(0), loc)
        val = track_windows_without("val", lo, span, np.random.default_rng(0), loc)[:2]
        for seed in seeds:
            t0 = time.perf_counter()
            net, epochs = ttc.train(sv, (x, y), val, seed)
            net.save(str(net_path(loc, seed)))
            rec = {"location": loc, "seed": seed, "training_windows": int(len(x)),
                   "windows_per_class_before_balancing": counts, "early_stopping_windows": int(len(val[0])),
                   "epochs_run": epochs, "params": int(net.count_params()), "wall_s": round(time.perf_counter() - t0, 1),
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(TRAIN_OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"location {loc} seed {seed}: {len(x)} windows, {epochs} epochs, {rec['wall_s']} s", flush=True)
            keras.backend.clear_session()


def views_with(lo, span) -> dict:
    """online_site_pilot.scaled_views with a given input range."""
    views = {}
    for split in site.SPLITS:
        z = np.load(tsx.tracks_file(split))
        feats = ((z["feats"] - lo) / span).astype(np.float32)
        lens = z["lens"].astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        first = np.concatenate([a + np.arange(n - ts.IN_LEN + 1) for a, n in zip(starts, lens)])
        views[split] = (np.lib.stride_tricks.sliding_window_view(feats, ts.IN_LEN, axis=0), first)
    return views


def network(keras, loc: int, seed: int):
    net = keras.models.load_model(str(net_path(loc, seed)), compile=False)
    flat = [lay for lay in net.layers if type(lay).__name__ == "Flatten"]
    dense = [lay for lay in net.layers if type(lay).__name__ == "Dense"]
    assert len(flat) == 1 and len(dense) == 1 and net.layers[-1] is dense[0]
    w, b = dense[0].get_weights()
    return keras.Model(net.inputs, flat[0].output), w.astype(np.float64), b.astype(np.float64)


def per_location(pred: np.ndarray) -> dict:
    z = np.load(tsx.tracks_file("test"))
    lens, cross = z["lens"].astype(np.int64), json.loads(str(z["cross"]))
    w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
    out = {}
    for loc in sorted(set(z["loc"].tolist())):
        tr = np.flatnonzero(z["loc"] == loc)
        p = np.concatenate([pred[w_start[t]:w_start[t] + lens[t] - ts.IN_LEN + 1] for t in tr])
        out[str(loc)] = site.compact(ts.measures_of(p, lens[tr], [cross[t] for t in tr]))
    return out


def own_fixed_test(keras, seed: int) -> np.ndarray:
    """exiD's own arm K network (all seven locations; exiD's protocol range), fixed, on every test window."""
    keep, arm, prep = site.NETS["exiD_K"]
    raw = np.load(prep / "train.npz")["feats"].astype(np.float32)
    lo = raw.reshape(-1, raw.shape[-1]).min(0)
    span = raw.reshape(-1, raw.shape[-1]).max(0) - lo + 1e-12
    z = np.load(tsx.tracks_file("test"))
    feats = ((z["feats"] - lo) / span).astype(np.float32)
    lens = z["lens"].astype(np.int64)
    first = np.concatenate([a + np.arange(n - ts.IN_LEN + 1) for a, n in zip(np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)])
    x = np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(feats, ts.IN_LEN, axis=0)[first].transpose(0, 2, 1))
    feat, w0, b0 = site.network(keras, keep, arm, seed)
    return np.argmax(feat.predict(x, batch_size=8192, verbose=0) @ w0 + b0, axis=1).astype(np.int8)


def run_phase() -> None:
    import input_scaling_check as isc

    keras = isc.keras
    locs = LOCS[:1] if SMOKE else list(range(7))
    done = {r["seed"] for r in records()}
    seeds = [s for s in (SEEDS[:1] if SMOKE else SEEDS) if s not in done]
    if not seeds:
        return
    missing = [(loc, s) for loc in locs for s in seeds if not net_path(loc, s).exists()]
    if missing:
        raise SystemExit(f"networks missing: {missing}")
    t0 = time.perf_counter()
    by_loc = site.windows_by_location()
    print(f"streams built in {time.perf_counter() - t0:.0f} s", flush=True)
    rank = site.SPLITS.index("test")
    n_test = int((np.load(tsx.tracks_file("test"))["lens"] - ts.IN_LEN + 1).sum())
    for seed in seeds:
        t1 = time.perf_counter()
        fixed, adapted, info = np.full(n_test, -1, np.int8), np.full(n_test, -1, np.int8), {}
        for loc in locs:
            sc = np.load(KEEP / f"scaling_without_loc{loc}{SUFFIX}.npz")
            views = views_with(sc["lo"], sc["span"])
            w = by_loc[loc]
            feat, w0, b0 = network(keras, loc, seed)
            f = site.features(feat, views, w)
            s = w[:, 0] == rank
            fixed[w[s, 1]] = np.argmax(f[s] @ w0 + b0, axis=1)
            pred, inf = site.adapt_site(f, w, w0, b0, SETTING, np.random.default_rng([seed, loc, site.SETTINGS.index(SETTING)]))
            adapted[w[s, 1]] = pred[s]
            info[str(loc)] = inf
            del views, f
            keras.backend.clear_session()
        rec = {"seed": seed, "setting": SETTING["id"], "split": "test", "info": info, "float32": True}
        if SMOKE:
            rec["smoke_windows_scored"] = int((adapted >= 0).sum())
        else:
            assert (fixed >= 0).all() and (adapted >= 0).all()
            own = own_fixed_test(keras, seed)
            keras.backend.clear_session()
            rec["fixed"], rec["adapted"] = site.split_measures("test", fixed), site.split_measures("test", adapted)
            rec["fixed"]["per_location"], rec["adapted"]["per_location"] = per_location(fixed), per_location(adapted)
            rec["exiD_own_K_fixed"] = {"all": site.compact(ts.measures_of(own, *test_tracks_info())),
                                       "per_location": per_location(own)}
        rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"seed {seed}: {time.perf_counter() - t1:.0f} s", flush=True)  # no result is printed while running


def test_tracks_info():
    z = np.load(tsx.tracks_file("test"))
    return z["lens"].astype(np.int64), json.loads(str(z["cross"]))


def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_lolo*.jsonl")):
        if "train" not in path.name and ("smoke" in path.name) == SMOKE:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def report() -> None:
    rows = records()
    own = [r for r in site.records() if r["phase"] == "final" and r["net"] == "exiD_K" and r["setting"] == SETTING["id"]]
    assert sorted(r["seed"] for r in own) == [100, 101, 102, 103, 104]
    u = round(st.mean(r["fixed"]["all"]["windows_correct"] for r in own), 4)          # the reference of the rule
    u_check = round(st.mean(r["exiD_own_K_fixed"]["all"]["windows_correct"] for r in rows), 4)
    u_loc = {loc: round(st.mean(r["exiD_own_K_fixed"]["per_location"][loc]["windows_correct"] for r in rows), 4)
             for loc in rows[0]["exiD_own_K_fixed"]["per_location"]}
    m = lambda which, key: round(st.mean(r[which]["all"][key] for r in rows), 4)      # noqa: E731
    f, a = m("fixed", "windows_correct"), m("adapted", "windows_correct")
    recs = sorted(rows[0]["fixed"]["per_recording"], key=int)
    gains = [st.mean(r["adapted"]["per_recording"][k]["windows_correct"] - r["fixed"]["per_recording"][k]["windows_correct"]
                     for r in rows) for k in recs]
    locs = sorted(rows[0]["fixed"]["per_location"], key=int)
    per_loc = {k: {"fixed": round(st.mean(r["fixed"]["per_location"][k]["windows_correct"] for r in rows), 4),
                   "adapted": round(st.mean(r["adapted"]["per_location"][k]["windows_correct"] for r in rows), 4)}
               for k in locs}
    gap = u - f
    crit = {"rule": "local notes 2026-11-19 21:56", "networks": len(rows), "seeds": sorted(r["seed"] for r in rows),
            "exiD_own_K_fixed": u, "left_out_fixed": f, "left_out_adapted": a, "gap": round(gap, 4),
            "gain": round(a - f, 4), "share_of_gap": round((a - f) / gap, 4) if gap > 0 else None,
            "0_gap_exists": gap >= GAP_MIN - 1e-12,
            "a_met": (a - f) >= max(HALF * gap, MIN_GAIN) - 1e-12,
            "b_recordings_with_a_gain": int(sum(g > 0 for g in gains)), "b_of": len(gains), "b_met": sum(g > 0 for g in gains) >= WINS,
            "c_correct_2_s_before": [m("fixed", "correct_2_s_before"), m("adapted", "correct_2_s_before")],
            "d_runs": [m("fixed", "runs_0.6_s_per_vehicle_hour"), m("adapted", "runs_0.6_s_per_vehicle_hour")]}
    crit["c_met"] = crit["c_correct_2_s_before"][1] >= crit["c_correct_2_s_before"][0] - EARLY - 1e-12
    crit["d_met"] = crit["d_runs"][1] <= crit["d_runs"][0] + 1e-12
    crit["complete"] = len(rows) == 5 and len(gains) == 14
    crit["met"] = bool(crit["complete"] and crit["0_gap_exists"] and crit["a_met"] and crit["b_met"] and crit["c_met"] and crit["d_met"])
    for k, v in per_loc.items():
        g = u_loc[k] - v["fixed"]
        v["exiD_own_K"] = u_loc[k]
        v["share_of_gap"] = round((v["adapted"] - v["fixed"]) / g, 4) if g > 0 else None
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "criterion": crit,
           "exiD_own_K_recomputed_here": u_check,
           "per_recording_gain": {k: round(g, 4) for k, g in zip(recs, gains)}, "per_location": per_loc,
           "other_measures": {k: [m("fixed", k), m("adapted", k)] for k in site.KEYS}}
    SUMMARY.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        train_phase()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        run_phase()
    else:
        raise SystemExit("usage: online_lolo.py train | run | --report")
