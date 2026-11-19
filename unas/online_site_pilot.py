"""Online learning at exiD sites: a highD network that keeps learning on the device (2026-11-19).

A network trained on highD and deployed at an exiD site (a roadside unit or a vehicle there) meets road layouts it
has not seen. The device knows, 26 frames (5.2 s) after every window, which class that window had. This script asks
whether the device can use these labels to update the network's last layer and close part of the gap to a network
trained offline on exiD with labels. Everything below was fixed before the script existed (local notes, 2026-11-19
17:36); the update is the gradient family of unas/online_pilot.py, unchanged.

  data       exiD's complete tracks (datasets/exid/track_stream_exid.py) and its own split. One device per location,
             running through the training, then validation, then test recordings of that location (each split in
             recording-number order); within a recording every window of every vehicle in the order of the absolute
             time of its last frame (tracksMeta initialFrame on the 5 Hz grid, plus 5 x (seg_start + k)), ties in
             track order. State carried across a location's recordings, reset between locations.
  networks   float32 on the PC: highD arm K (datasets/highd/data/track_train/K_*, seeds 100-104), inputs with highD's
             training range (as is); listed: highD arm P; reference: exiD's own arm K with exiD's range, fixed (and,
             listed, with the same update). Fixed layer: the Dense layer on the Flatten output.
  labels     class = direction of a crossing 1 to 26 frames after the window's last frame, else lane keeping; known if
             such a crossing follows or the track goes on for 26 more frames; used 26 frames after the window's last
             frame, within the same recording.
  settings   eta 0.003 / 0.01 / 0.03 x lambda 0 / 0.01 (memory 128 per class, 4 per class per step, start at 20).
  selection  highD arm K, validation recordings: largest gain in windows correct (label known), among the settings
             whose correct-2-s-before is at most 5 points below the fixed layer's and whose runs of false calls of
             0.6 s or more per vehicle hour are not above it.
  criterion  selected setting, highD arm K, the 14 test recordings: (a) gain in windows correct at least half the gap
             to exiD's own arm K and at least 3 points; (b) a gain on at least 11 recordings; (c) correct 2 s before
             the crossing at most 5 points below the fixed layer's; (d) runs of 0.6 s or more per vehicle hour not
             above the fixed layer's.

Run in the WSL dmir_nas venv, from the fork; SEEDS=100 (etc.) and OUT_NAME=online_site_s100.jsonl let several
processes share the work, each with its own record file:
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/online_site_pilot.py tune     # validation recordings
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/online_site_pilot.py final    # test recordings
  .venv\\Scripts\\python unas/online_site_pilot.py --report
SMOKE=1 (with OUT_NAME) runs one location, one seed and two settings, to check the code.
Output: datasets/exid/results/seeds/online_site*.jsonl and datasets/exid/results/hil/online_site.json.
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "datasets" / "highd"))
sys.path.insert(0, str(REPO / "datasets" / "exid"))
import online_pilot as op                                    # noqa: E402  (numpy only: the update's constants)
import track_stream as ts                                    # noqa: E402  (highD track measures)
import track_stream_exid as tsx                              # noqa: E402  (exiD complete tracks)

MODEL = "model_1793711684616882545_453917"
NETS = {"highD_K": (REPO / "datasets" / "highd" / "data" / "track_train", "K", ts.PREP),
        "highD_P": (REPO / "datasets" / "highd" / "data" / "track_train", "P", ts.PREP),
        "exiD_K": (REPO / "datasets" / "exid" / "data" / "track_train", "K", tsx.PREP)}
RESULTS = REPO / "datasets" / "exid" / "results"
OUT = RESULTS / "seeds" / os.environ.get("OUT_NAME", "online_site.jsonl")
SUMMARY = RESULTS / "hil" / "online_site.json"
RAW_META = REPO / "datasets" / "exid" / "data" / "raw" / "data"
SPLITS = ("train", "val", "test")
OFFSET = ts.HORIZON * 5                                      # 26 frames at 5 Hz, in 25 Hz frame numbers
SETTINGS = [s for s in op.SETTINGS if s["family"] == "G"]
HALF, MIN_GAIN, WINS, EARLY = 0.5, 0.03, 11, 0.05           # the criterion
assert len(SETTINGS) == 6 and OFFSET == 130


# --------------------------------------------------------------------------------------------- the device's stream
def windows_by_location() -> dict:
    """Per location, every window of its recordings in device order: split, split-local window index, track, class,
    label known, absolute time (25 Hz frame number), recording."""
    import csv

    first5 = {}
    rows = collections.defaultdict(list)
    for rank, split in enumerate(SPLITS):
        z = np.load(tsx.tracks_file(split))
        lens = z["lens"].astype(np.int64)
        w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
        cross = json.loads(str(z["cross"]))
        for r in sorted(set(z["rec"].tolist())):
            with open(RAW_META / f"{r:02d}_tracksMeta.csv", newline="") as fh:
                for row in csv.DictReader(fh):
                    first5[(int(r), int(row["trackId"]))] = -(-int(float(row["initialFrame"])) // 5) * 5
        for t in range(len(lens)):
            n, r, loc = int(lens[t]), int(z["rec"][t]), int(z["loc"][t])
            k, _, _, to_next, next_dir, _, _ = ts.phases(n, cross[t])
            known = (to_next <= ts.HORIZON) | ((n - 1) - k >= ts.HORIZON)
            label = np.where(to_next <= ts.HORIZON, next_dir, 0)
            t_abs = first5[(r, int(z["tv"][t]))] + 5 * (int(z["seg_start"][t]) + k)
            m = len(k)
            rows[loc].append(np.stack([np.full(m, rank), w_start[t] + np.arange(m), np.full(m, t), label,
                                       known.astype(np.int64), t_abs, np.full(m, r)], axis=1))
    out = {}
    for loc, parts in rows.items():
        a = np.concatenate(parts)
        order = np.lexsort((a[:, 2], a[:, 5], a[:, 6], a[:, 0]))   # split, recording, time, track
        out[loc] = a[order]
    return out


def scaled_views(prep: Path) -> dict:
    """Per split: the sliding windows (first frame, feature, step) of its tracks, scaled with the given range."""
    raw = np.load(prep / "train.npz")["feats"].astype(np.float32)
    lo = raw.reshape(-1, raw.shape[-1]).min(0)
    span = raw.reshape(-1, raw.shape[-1]).max(0) - lo + 1e-12
    views = {}
    for split in SPLITS:
        z = np.load(tsx.tracks_file(split))
        feats = ((z["feats"] - lo) / span).astype(np.float32)
        lens = z["lens"].astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        first = np.concatenate([a + np.arange(n - ts.IN_LEN + 1) for a, n in zip(starts, lens)])
        views[split] = (np.lib.stride_tricks.sliding_window_view(feats, ts.IN_LEN, axis=0), first)
    return views


def features(feat, views: dict, w: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Flatten outputs of the windows of one location (rows of windows_by_location), in device order."""
    out = np.empty((len(w), feat.output_shape[-1]), np.float32)
    for rank, split in enumerate(SPLITS):
        sel = np.flatnonzero(w[:, 0] == rank)
        view, first = views[split]
        for a in range(0, len(sel), chunk):
            s = sel[a:a + chunk]
            x = np.ascontiguousarray(view[first[w[s, 1]]].transpose(0, 2, 1))
            out[s] = feat.predict(x, batch_size=8192, verbose=0)
    return out


# ------------------------------------------------------------------------------------------------- the update
def adapt_site(f: np.ndarray, w: np.ndarray, w0: np.ndarray, b0: np.ndarray, setting: dict, rng):
    """The class of every window with the layer as it is at that window's time; labels used OFFSET later."""
    label, known, t_abs, rec = w[:, 3], w[:, 4].astype(bool), w[:, 5], w[:, 6]
    d = f.shape[1]
    wt, bt = w0.astype(np.float64).copy(), b0.astype(np.float64).copy()
    mem = np.zeros((3, op.MEM, d))
    count, pos = np.zeros(3, np.int64), np.zeros(3, np.int64)
    eye, rep = np.eye(3), np.repeat(np.arange(3), op.PER_CLASS)
    pred = np.empty(len(f), np.int8)
    pend = collections.deque()
    updates, warm_at = 0, None
    cut = np.flatnonzero((np.diff(t_abs) != 0) | (np.diff(rec) != 0)) + 1
    for g0, g1 in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(f)]])):
        r, t = rec[g0], t_abs[g0]
        while pend and rec[pend[0]] != r:                   # labels do not survive the end of a recording
            pend.popleft()
        while pend and t_abs[pend[0]] + OFFSET <= t:
            i = pend.popleft()
            c = int(label[i])
            mem[c, pos[c]] = f[i]
            pos[c] = (pos[c] + 1) % op.MEM
            count[c] = min(count[c] + 1, op.MEM)
            if (count >= op.MIN_CLASS).all():
                if warm_at is None:
                    warm_at = int(i)
                x = np.concatenate([mem[k, rng.integers(0, count[k], op.PER_CLASS)] for k in range(3)])
                gr = (op.softmax(x @ wt + bt) - eye[rep]) / len(x)
                wt = wt - setting["eta"] * (x.T @ gr + 2 * setting["lam"] * (wt - w0))
                bt = bt - setting["eta"] * (gr.sum(0) + 2 * setting["lam"] * (bt - b0))
                updates += 1
        pred[g0:g1] = np.argmax(f[g0:g1] @ wt + bt, axis=1)
        pend.extend(int(i) for i in range(g0, g1) if known[i])
    return pred, {"updates": updates, "warm_from_window": warm_at}


# ------------------------------------------------------------------------------------------------- the measures
def compact(m: dict) -> dict:
    lk = m["label_known"]
    return {"windows_correct": lk["windows_correct"],
            "correct_2_s_before": m["correct_class_in_the_window_ending_before_the_crossing"]["2_s"]["share"],
            "runs_0.6_s_per_vehicle_hour": m["runs_of_free_windows_called_a_lane_change_per_vehicle_hour"]["0.6_s_or_more"],
            "opposite_without_a_return": lk["windows_holding_a_completed_lane_change_opposite_called_without_a_return"]["share"],
            "lane_keeping_from_6_s_called": lk["lane_keeping_from_6_s_windows_called_a_lane_change"]["share"]}


def split_measures(split: str, pred_split: np.ndarray) -> dict:
    """Over the split and per recording, from the classes of the split's windows in track order."""
    z = np.load(tsx.tracks_file(split))
    lens = z["lens"].astype(np.int64)
    cross = json.loads(str(z["cross"]))
    w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
    res = {"all": compact(ts.measures_of(pred_split, lens, cross)), "per_recording": {}}
    for r in sorted(set(z["rec"].tolist())):
        tr = np.flatnonzero(z["rec"] == r)
        p = np.concatenate([pred_split[w_start[t]:w_start[t] + lens[t] - ts.IN_LEN + 1] for t in tr])
        res["per_recording"][str(r)] = compact(ts.measures_of(p, lens[tr], [cross[t] for t in tr]))
    return res


# ------------------------------------------------------------------------------------------------- the runs
def network(keras, keep: Path, arm: str, seed: int):
    net = keras.models.load_model(str(keep / f"{arm}_{MODEL}_seed{seed}.h5"), compile=False)
    flat = [lay for lay in net.layers if type(lay).__name__ == "Flatten"]
    dense = [lay for lay in net.layers if type(lay).__name__ == "Dense"]
    assert len(flat) == 1 and len(dense) == 1 and net.layers[-1] is dense[0]
    w, b = dense[0].get_weights()
    return keras.Model(net.inputs, flat[0].output), w.astype(np.float64), b.astype(np.float64)


def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_site*.jsonl")):
        if "smoke" not in path.name:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def run(phase: str) -> None:
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters

    keras = isc.keras
    smoke = os.environ.get("SMOKE") == "1"
    if smoke and "OUT_NAME" not in os.environ:
        raise SystemExit("SMOKE=1 needs its own OUT_NAME")
    if phase == "tune":
        plan, scored, stream_splits = [("highD_K", SETTINGS)], "val", 2
    else:
        chosen = selection(summaries(records()))["selected"]
        if chosen is None:
            print("NO GO: no setting qualified on the validation recordings; the test recordings are not scored")
            return
        sel = [s for s in SETTINGS if s["id"] == chosen]
        plan, scored, stream_splits = [("highD_K", sel), ("highD_P", sel), ("exiD_K", sel)], "test", 3
    seeds = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]
    if smoke:
        plan, seeds = [("highD_K", [SETTINGS[0], SETTINGS[5]])], seeds[:1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["phase"], r["setting"], r["net"], r["seed"]) for r in records()}
    t0 = time.perf_counter()
    by_loc = windows_by_location()
    if smoke:
        by_loc = {min(by_loc, key=lambda k: len(by_loc[k])): by_loc[min(by_loc, key=lambda k: len(by_loc[k]))]}
    by_loc = {loc: w[w[:, 0] < stream_splits] for loc, w in by_loc.items()}
    print(f"streams built in {time.perf_counter() - t0:.0f} s: {sum(len(w) for w in by_loc.values())} windows", flush=True)
    rank = SPLITS.index(scored)
    n_scored = int((np.load(tsx.tracks_file(scored))["lens"] - ts.IN_LEN + 1).sum())
    for name, settings in plan:
        keep, arm, prep = NETS[name]
        views = scaled_views(prep)
        for seed in seeds:
            todo = [s for s in settings if (phase, s["id"], name, seed) not in done]
            if not todo:
                continue
            t1 = time.perf_counter()
            feat, w0, b0 = network(keras, keep, arm, seed)
            fixed = np.full(n_scored, -1, np.int8)
            adapted = {s["id"]: np.full(n_scored, -1, np.int8) for s in todo}
            info = {s["id"]: {} for s in todo}
            for loc, w in by_loc.items():
                f = features(feat, views, w)
                sc = w[:, 0] == rank
                fixed[w[sc, 1]] = np.argmax(f[sc] @ w0 + b0, axis=1)
                for s in todo:
                    rng = np.random.default_rng([seed, loc, SETTINGS.index(s)])
                    pred, inf = adapt_site(f, w, w0, b0, s, rng)
                    adapted[s["id"]][w[sc, 1]] = pred[sc]
                    info[s["id"]][str(loc)] = inf
            keras.backend.clear_session()
            if not smoke:
                assert (fixed >= 0).all(), "a window of the scored split was not scored"
            fixed_m = split_measures(scored, fixed) if not smoke else None
            for s in todo:
                rec = {"phase": phase, "setting": s["id"], "net": name, "seed": seed, "split": scored,
                       "fixed": fixed_m, "adapted": split_measures(scored, adapted[s["id"]]) if not smoke else None,
                       "info": info[s["id"]], "float32": True,
                       "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                if smoke:
                    sc_all = adapted[s["id"]] >= 0
                    rec["smoke_windows_scored"] = int(sc_all.sum())
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            print(f"{phase} {name} seed {seed}: {len(todo)} settings, {time.perf_counter() - t1:.0f} s",
                  flush=True)                               # no result is printed while running


# ------------------------------------------------------------------------------------------------- the report
KEYS = ("windows_correct", "correct_2_s_before", "runs_0.6_s_per_vehicle_hour", "opposite_without_a_return",
        "lane_keeping_from_6_s_called")


def mean_of(rows: list, which: str, key: str):
    vals = [r[which]["all"][key] for r in rows]
    return None if any(v is None for v in vals) else round(st.mean(vals), 4)


def summaries(rows: list) -> dict:
    res = {"tune": {}, "final": {}}
    for phase in ("tune", "final"):
        for name in NETS:
            for s in SETTINGS:
                got = [r for r in rows if (r["phase"], r["net"], r["setting"]) == (phase, name, s["id"])]
                if not got:
                    continue
                entry = {"networks": len(got), "seeds": sorted(r["seed"] for r in got)}
                for key in KEYS:
                    entry[key] = {"fixed": mean_of(got, "fixed", key), "adapted": mean_of(got, "adapted", key)}
                recs = sorted(got[0]["fixed"]["per_recording"], key=int)
                entry["per_recording_windows_correct"] = {
                    r: [round(st.mean(g["fixed"]["per_recording"][r]["windows_correct"] for g in got), 4),
                        round(st.mean(g["adapted"]["per_recording"][r]["windows_correct"] for g in got), 4)] for r in recs}
                res[phase][f"{name} {s['id']}"] = entry
    return res


def qualifies(e: dict) -> bool:
    return (e["correct_2_s_before"]["adapted"] >= e["correct_2_s_before"]["fixed"] - EARLY - 1e-12
            and e["runs_0.6_s_per_vehicle_hour"]["adapted"] <= e["runs_0.6_s_per_vehicle_hour"]["fixed"] + 1e-12)


def selection(res: dict) -> dict:
    tune = {k.split(" ", 1)[1]: v for k, v in res["tune"].items() if k.startswith("highD_K ")}
    complete = {k: v for k, v in tune.items() if v["networks"] == 5}
    if len(complete) < len(SETTINGS):
        return {"selected": None, "complete": False, "settings_complete": sorted(complete)}
    ok = [(v["windows_correct"]["adapted"] - v["windows_correct"]["fixed"], k) for k, v in complete.items() if qualifies(v)]
    return {"selected": max(ok)[1] if ok else None, "complete": True, "qualified": sorted(k for _, k in ok)}


def report() -> None:
    rows = records()
    res = summaries(rows)
    sel = selection(res)
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "records": len(rows),
           "rule": "local notes 2026-11-19 17:36", "selection": sel, "tune": res["tune"], "final": res["final"]}
    for key, e in res["tune"].items():
        wc, c2, r6 = e["windows_correct"], e["correct_2_s_before"], e["runs_0.6_s_per_vehicle_hour"]
        print(f"tune {key}: windows correct {wc['fixed']:.4f} -> {wc['adapted']:.4f} ({wc['adapted'] - wc['fixed']:+.4f}); "
              f"2 s before {c2['fixed']:.4f} -> {c2['adapted']:.4f}; runs >=0.6 s/vh {r6['fixed']:.1f} -> {r6['adapted']:.1f}; "
              f"{'qualifies' if qualifies(e) else 'does not qualify'}")
    print("selection:", json.dumps(sel))
    chosen = sel.get("selected")
    if chosen and f"highD_K {chosen}" in res["final"] and f"exiD_K {chosen}" in res["final"]:
        hk, ek = res["final"][f"highD_K {chosen}"], res["final"][f"exiD_K {chosen}"]
        f_, a_, u_ = hk["windows_correct"]["fixed"], hk["windows_correct"]["adapted"], ek["windows_correct"]["fixed"]
        gains = [a - f for f, a in hk["per_recording_windows_correct"].values()]
        crit = {"rule": "selected setting, highD arm K, the 14 test recordings: (a) gain in windows correct at least half "
                        "the gap to exiD's own arm K and at least 3 points; (b) a gain on at least 11 recordings; (c) "
                        "correct 2 s before at most 5 points below the fixed layer's; (d) runs of 0.6 s or more per "
                        "vehicle hour not above the fixed layer's",
                "networks": [hk["networks"], ek["networks"]],
                "fixed_highD": f_, "adapted_highD": a_, "exiD_own": u_, "gap": round(u_ - f_, 4),
                "gain": round(a_ - f_, 4), "share_of_gap": round((a_ - f_) / (u_ - f_), 4) if u_ != f_ else None,
                "a_met": (a_ - f_) >= max(HALF * (u_ - f_), MIN_GAIN) - 1e-12,
                "b_recordings_with_a_gain": int(sum(g > 0 for g in gains)), "b_of": len(gains),
                "b_met": sum(g > 0 for g in gains) >= WINS,
                "c_correct_2_s_before": [hk["correct_2_s_before"]["fixed"], hk["correct_2_s_before"]["adapted"]],
                "c_met": hk["correct_2_s_before"]["adapted"] >= hk["correct_2_s_before"]["fixed"] - EARLY - 1e-12,
                "d_runs": [hk["runs_0.6_s_per_vehicle_hour"]["fixed"], hk["runs_0.6_s_per_vehicle_hour"]["adapted"]],
                "d_met": hk["runs_0.6_s_per_vehicle_hour"]["adapted"] <= hk["runs_0.6_s_per_vehicle_hour"]["fixed"] + 1e-12}
        crit["complete"] = hk["networks"] == 5 and ek["networks"] == 5 and len(gains) == 14
        crit["met"] = bool(crit["complete"] and crit["a_met"] and crit["b_met"] and crit["c_met"] and crit["d_met"])
        out["criterion"] = crit
        for key, e in res["final"].items():
            print(f"final {key}: " + "; ".join(f"{k} {e[k]['fixed']} -> {e[k]['adapted']}" for k in KEYS))
        print(json.dumps(crit, indent=1))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] in ("tune", "final"):
        run(sys.argv[1])
    else:
        raise SystemExit("usage: online_site_pilot.py tune | final | --report")
