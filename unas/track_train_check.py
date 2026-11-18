"""highD: the intention network trained on windows from complete tracks (2026-11-17).

datasets/highd/track_stream.py showed that the deployed highD networks, which score 91-93% on the protocol's
scenarios, call a lane change in more than a fifth of the windows of a vehicle that keeps its lane, and
announce the opposite lane change after a lane change: the protocol's scenarios hold neither ordinary lane
keeping in its real proportion nor a completed lane change. This script repeats on highD the LCIR study of
unas/stream_train_check.py. Everything below was fixed before the script existed (local notes, 2026-11-17
11:18).

  split      as in the protocol: recordings 1-50 train, 51-55 stop the training early, 56-60 test
  arms       same architecture (the paper's searched highD classifier, 2,593 parameters), same recipe (the
             search's, select_by_seeds.train_one), seeds 100-104, inputs scaled with the range of the
             protocol's training windows in both arms:
             P  the protocol's training windows, early stopping on the protocol's validation windows;
             T  every 10-frame window of every track of the training recordings, class = the direction of
                the next crossing if it comes 1 to 26 frames after the window's last frame, else lane
                keeping, whatever the window holds; classes balanced by subsampling to the smallest class
                (rng 0); early stopping on the same construction for recordings 51-55.
  int8       as deployed: 500 calibration windows of the arm's own training set (rng 42); scored as int8
             on the PC.
  measures   on the complete tracks of recordings 56-60 (datasets/highd/track_stream.measures_of) and on
             the 18,018 windows of the protocol's test scenarios; mean over the five seeds.
  criterion  all four: (a) free windows called a lane change: T at most half of P; (b) the opposite
             direction called in windows holding a completed lane change: T at most half of P; (c) lane
             changes with the correct class 2 s before the crossing: T at most 5 points below P;
             (d) accuracy on the protocol's test windows: T at most 3 points below P.

Correction of the same day (local notes, 11:47 and 11:48): a highD track lasts a median 13 s, so the label of a
window is known only if a crossing follows within 26 frames or the track goes on for 26 more frames. The first
"free windows" measure counted windows whose label is unknown, and arm T was trained with such windows labelled
lane keeping. Arm K is T restricted to windows whose label is known (training and early stopping); --rescore
scores the kept P and T networks again with track_stream.measures_of, which now also gives the measures over
windows with a known label. Criterion for K against P, all four: (a') lane-keeping windows from 6 s after the
track's start called a lane change, K at most half of P; (b') the opposite direction called in windows that hold
a completed lane change not followed by a crossing back, K at most half of P; (c) and (d) as above.

Needs datasets/highd/track_stream.py build train / val / test (Windows). Run in the WSL NAS venv, from the fork:
  HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared SEEDS=100,101,102,103,104 \\
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/track_train_check.py
  ... unas/track_train_check.py K           # the arm with labels known only
  ... unas/track_train_check.py --rescore   # the kept P and T networks, measures of the corrected track_stream
  .venv\\Scripts\\python unas/track_train_check.py --report
Output: datasets/highd/results/seeds/track_windows.jsonl (one line per arm and seed; resumable) and
datasets/highd/results/hil/track_windows.json (the report). The networks are kept in
datasets/highd/data/track_train/ (not versioned).

exiD (2026-11-18, local notes 16:30): TRACK_DATASET=exid runs arms P and K on exiD's own split, with the complete
tracks of datasets/exid/track_stream_exid.py and HIGHD_DATA_ROOT pointing at datasets/exid/data/prepared; same
architecture, recipe, seeds and int8 conversion. Criterion, all four: (1) arm P calls the opposite direction in at
least half of the windows that hold a completed lane change not followed by a crossing back; (2) K at most half of
P there; (3) K correct 2 s before the crossing at most 5 points below P; (4) K accuracy on the protocol's test
windows at most 3 points below P. Files under datasets/exid/ instead of datasets/highd/; the class of every test
window is kept next to the networks for the listing by kind of lane change.
  TRACK_DATASET=exid HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/exid/data/prepared SEEDS=100,101,102,103,104 \\
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/track_train_check.py
  TRACK_DATASET=exid .venv\\Scripts\\python unas/track_train_check.py --report
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "datasets" / "highd"))
import track_stream as ts                                    # noqa: E402  (numpy only)

DATASET = os.environ.get("TRACK_DATASET", "highd")
ARMS = {"P": "the protocol's windows", "T": "windows from complete tracks",
        "K": "windows from complete tracks whose label is known"}
if DATASET == "exid":
    sys.path.insert(0, str(REPO / "datasets" / "exid"))
    import track_stream_exid as tsx                          # noqa: E402  (numpy only)
    tracks_file, PREP = tsx.tracks_file, tsx.PREP
    ARMS = {a: ARMS[a] for a in ("P", "K")}
elif DATASET == "highd":
    tracks_file, PREP = ts.tracks_file, ts.PREP
else:
    sys.exit(f"TRACK_DATASET must be highd or exid, not {DATASET!r}")
BASE = REPO / "datasets" / DATASET
KEEP = BASE / "data" / "track_train"
OUT = BASE / "results" / "seeds" / os.environ.get("OUT_NAME", "track_windows.jsonl")
RESCORED = OUT.with_name(OUT.stem + "_rescored.jsonl")
SUMMARY = BASE / "results" / "hil" / "track_windows.json"
SEARCH, MODEL = "highd_cls_v4_k3_single_s42", "model_1793711684616882545_453917"
H5 = REPO / "datasets" / "highd" / "results" / "nas-v2" / SEARCH / f"{MODEL}.h5"
HALF, EARLY, PROTOCOL = 0.5, 0.05, 0.03                      # the criterion
EXISTS = 0.5                                                 # exiD part (1): the shortcut exists in arm P


def track_windows(split: str, lo, span, rng, known_only: bool = False):
    """The balanced windows of a split's complete tracks, scaled: x, class, counts per class before balancing.
    known_only: only windows whose label is known (a crossing within 26 frames, or 26 more frames of the track)."""
    z = np.load(tracks_file(split))
    feats = ((z["feats"] - lo) / span).astype(np.float32)
    lens = z["lens"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    first, lab = [], []
    for a, n, cr in zip(starts, lens, json.loads(str(z["cross"]))):
        k, _, _, to_next, next_dir, _, _ = ts.phases(int(n), cr)
        keep = ((to_next <= ts.HORIZON) | ((int(n) - 1) - k >= ts.HORIZON)) if known_only else np.ones(len(k), bool)
        first.append((a + k - (ts.IN_LEN - 1))[keep])
        lab.append(np.where(to_next <= ts.HORIZON, next_dir, 0)[keep])
    first, lab = np.concatenate(first), np.concatenate(lab)
    counts = np.bincount(lab, minlength=3)
    pick = np.concatenate([rng.choice(np.flatnonzero(lab == c), size=int(counts.min()), replace=False) for c in range(3)])
    pick = pick[rng.permutation(len(pick))]
    win = np.lib.stride_tricks.sliding_window_view(feats, ts.IN_LEN, axis=0)
    return np.ascontiguousarray(win[first[pick]].transpose(0, 2, 1)), lab[pick].astype(np.int64), [int(c) for c in counts]


def train(sv, train_xy, val_xy, seed: int):
    """select_by_seeds.train_one's highD recipe, keeping the trained network."""
    tf, keras = sv.tf, sv.keras
    keras.utils.set_random_seed(seed)
    net = keras.models.clone_model(keras.models.load_model(str(H5), compile=False))
    loss, metrics = sv.compile_args("logits")
    net.compile(optimizer="adam", loss=loss, metrics=metrics)
    tr = tf.data.Dataset.from_tensor_slices(train_xy).shuffle(sv.BATCH * 8, seed=seed).batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    va = tf.data.Dataset.from_tensor_slices(val_xy).batch(sv.BATCH).prefetch(tf.data.AUTOTUNE)
    hist = net.fit(tr, validation_data=va, epochs=sv.EPOCH_CAP or 50, verbose=0, callbacks=sv.callbacks("highd", "logits"))
    return net, len(hist.history["loss"])


def classes_int8(tf, tfl: bytes, x: np.ndarray) -> np.ndarray:
    """The class of every window (scaled float inputs), through the file's own int8 interface."""
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = inp["quantization"]
    q = np.clip(np.rint(x / np.float32(si) + np.float32(zi)), -128, 127).astype(np.int8)
    cls = np.empty(len(q), np.int64)
    for i in range(len(q)):
        it.set_tensor(inp["index"], q[i:i + 1])
        it.invoke()
        cls[i] = int(it.get_tensor(out["index"]).reshape(-1).astype(np.int16).argmax())
    return cls


def scaling():
    raw = np.load(PREP / "train.npz")["feats"].astype(np.float32)
    lo = raw.reshape(-1, raw.shape[-1]).min(0)
    return lo, raw.reshape(-1, raw.shape[-1]).max(0) - lo + 1e-12


def test_tracks(lo, span):
    """Every window of every test track, scaled, in track order."""
    z = np.load(tracks_file("test"))
    lens, cross = z["lens"].astype(np.int64), json.loads(str(z["cross"]))
    tracks = ((z["feats"] - lo) / span).astype(np.float32)
    w_first = np.concatenate([a + np.arange(n - ts.IN_LEN + 1) for a, n in zip(np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)])
    return lens, cross, np.ascontiguousarray(np.lib.stride_tricks.sliding_window_view(tracks, ts.IN_LEN, axis=0)[w_first].transpose(0, 2, 1))


def rescore():
    """WSL: the kept P and T networks on the test tracks with the measures of the corrected track_stream."""
    import tensorflow as tf

    done = {(r["arm"], r["seed"]) for r in map(json.loads, RESCORED.read_text().splitlines())} if RESCORED.exists() else set()
    lens, cross, track_x = test_tracks(*scaling())
    for r in map(json.loads, OUT.read_text().splitlines()):
        if r["arm"] not in ("P", "T") or (r["arm"], r["seed"]) in done:
            continue
        tfl = (KEEP / f"{r['arm']}_{MODEL}_seed{r['seed']}_int8_io.tflite").read_bytes()
        rec = {"arm": r["arm"], "seed": r["seed"], "complete_tracks_test": ts.measures_of(classes_int8(tf, tfl, track_x), lens, cross),
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with open(RESCORED, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"arm {r['arm']} seed {r['seed']} rescored", flush=True)


def run(arms):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters

    tf, keras, sv = isc.tf, isc.keras, isc.sv
    KEEP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["arm"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    smoke = "_smoke" if sv.EPOCH_CAP else ""

    root = os.environ.get("HIGHD_DATA_ROOT", "")
    if ("exid" in root) != (DATASET == "exid"):
        sys.exit(f"HIGHD_DATA_ROOT ({root!r}) does not match TRACK_DATASET={DATASET}")
    ds = sv.HighD_Dataset(task="highd_cls")                  # the protocol's windows, scaled, training set permuted once
    lo, span = scaling()
    xte, yte = ds._data["test"]
    lens, cross, track_x = test_tracks(lo, span)
    for arm in arms:
        if all((arm, s) in done for s in sv.SEEDS):
            continue
        if arm == "P":
            (x, y), val, counts = ds._data["train"], ds._data["val"], None
        else:
            x, y, counts = track_windows("train", lo, span, np.random.default_rng(0), arm == "K")
            val = track_windows("val", lo, span, np.random.default_rng(0), arm == "K")[:2]
        for seed in sv.SEEDS:
            if (arm, seed) in done:
                continue
            t0 = time.perf_counter()
            kept = KEEP / f"{arm}_{MODEL}_seed{seed}{smoke}.h5"
            if kept.exists():
                net, epochs = keras.models.load_model(str(kept), compile=False), None
            else:
                net, epochs = train(sv, (x, y), val, seed)
                net.save(str(kept))
            tfl = isc.convert_int8_io(net, x)
            (KEEP / f"{arm}_{MODEL}_seed{seed}{smoke}_int8_io.tflite").write_bytes(tfl)
            cls_tracks = classes_int8(tf, tfl, track_x)
            if DATASET == "exid":                            # kept for the listing by kind of lane change
                np.savez_compressed(KEEP / f"{arm}_{MODEL}_seed{seed}{smoke}_test_tracks.npz", cls=cls_tracks.astype(np.int8))
            rec = {"arm": arm, "trained_on": ARMS[arm], "seed": seed, "training_windows": int(len(x)),
                   "track_windows_per_class_before_balancing": counts, "early_stopping_windows": int(len(val[0])),
                   "epochs_run": epochs, "params": int(net.count_params()), "int8_io_bytes": len(tfl),
                   "protocol_test_windows": int(len(xte)),
                   "protocol_test_accuracy_int8": round(float((classes_int8(tf, tfl, xte) == yte).mean()), 5),
                   "complete_tracks_test": ts.measures_of(cls_tracks, lens, cross),
                   "wall_s": round(time.perf_counter() - t0, 1),
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"arm {arm} seed {seed}: {len(x)} training windows, {epochs} epochs, {rec['wall_s']} s", flush=True)
            keras.backend.clear_session()                    # no result is printed while the run is going


MEASURES = {
    "free_windows_called_a_lane_change": lambda m: m["free_windows_called_a_lane_change"]["share"],
    "free_windows_called_a_lane_change_tracks_without_a_lane_change":
        lambda m: m["free_windows_called_a_lane_change_tracks_without_a_lane_change"]["share"],
    "opposite_called_in_windows_holding_a_completed_lane_change":
        lambda m: m["windows_holding_a_completed_lane_change"]["called_the_opposite_direction"],
    "correct_2_s_before_the_crossing": lambda m: m["correct_class_in_the_window_ending_before_the_crossing"]["2_s"]["share"],
    "correct_1_s_before": lambda m: m["correct_class_in_the_window_ending_before_the_crossing"]["1_s"]["share"],
    "correct_3_s_before": lambda m: m["correct_class_in_the_window_ending_before_the_crossing"]["3_s"]["share"],
    "correct_4_s_before": lambda m: m["correct_class_in_the_window_ending_before_the_crossing"]["4_s"]["share"],
    "correct_5_s_before": lambda m: m["correct_class_in_the_window_ending_before_the_crossing"]["5_s"]["share"],
    "windows_before_a_crossing_correct": lambda m: m["windows_before_a_crossing_correct"]["share"],
    "early_windows_called_a_lane_change": lambda m: m["early_windows_called_a_lane_change"]["share"],
    "free_windows_1.8_to_5_s_after_a_crossing_opposite": lambda m: m["free_windows_1.8_to_5_s_after_a_crossing"]["called_the_opposite_direction"],
    "free_windows_1.8_to_5_s_after_a_crossing_completed": lambda m: m["free_windows_1.8_to_5_s_after_a_crossing"]["called_the_completed_direction"],
    "runs_per_vehicle_hour": lambda m: m["runs_of_free_windows_called_a_lane_change_per_vehicle_hour"]["any_length"],
    "runs_of_0.6_s_or_more_per_vehicle_hour": lambda m: m["runs_of_free_windows_called_a_lane_change_per_vehicle_hour"]["0.6_s_or_more"],
    "windows_correct_on_complete_tracks": lambda m: m["windows_correct_with_the_labels_above"],
}


KNOWN = {
    "lane_keeping_from_6_s_called": lambda m: m["label_known"]["lane_keeping_from_6_s_windows_called_a_lane_change"]["share"],
    "lane_keeping_first_6_s_called": lambda m: m["label_known"]["lane_keeping_first_6_s_windows_called_a_lane_change"]["share"],
    "lane_keeping_called": lambda m: m["label_known"]["lane_keeping_windows_called_a_lane_change"]["share"],
    "lane_keeping_tracks_without_a_lane_change_from_6_s_called":
        lambda m: m["label_known"]["lane_keeping_tracks_without_a_lane_change_from_6_s_windows_called_a_lane_change"]["share"],
    "opposite_without_a_return": lambda m: m["label_known"]["windows_holding_a_completed_lane_change_opposite_called_without_a_return"]["share"],
    "windows_correct_label_known": lambda m: m["label_known"]["windows_correct"],
    "correct_2_s_before_the_crossing": lambda m: m["correct_class_in_the_window_ending_before_the_crossing"]["2_s"]["share"],
    "correct_4_s_before": lambda m: m["correct_class_in_the_window_ending_before_the_crossing"]["4_s"]["share"],
}


def report():
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]

    def mean_std(vals):
        return {"mean": round(st.mean(vals), 5), "std": round(st.stdev(vals), 5) if len(vals) > 1 else None,
                "values": [round(v, 5) for v in vals]}
    arms = {}
    for arm in ARMS:
        rs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"])
        if not rs:
            continue
        arms[arm] = {"trained_on": ARMS[arm], "seeds": [r["seed"] for r in rs], "training_windows": rs[0]["training_windows"],
                     "track_windows_per_class_before_balancing": rs[0]["track_windows_per_class_before_balancing"],
                     "epochs_run": [r["epochs_run"] for r in rs],
                     "protocol_test_accuracy_int8": mean_std([r["protocol_test_accuracy_int8"] for r in rs]),
                     **{name: mean_std([f(r["complete_tracks_test"]) for r in rs]) for name, f in MEASURES.items()}}
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "records": len(rows), "arms": arms}
    if "P" in arms and "T" in arms:
        p, t = arms["P"], arms["T"]
        v = lambda a, k: a[k]["mean"]                        # noqa: E731
        free, opp = "free_windows_called_a_lane_change", "opposite_called_in_windows_holding_a_completed_lane_change"
        early, prot = "correct_2_s_before_the_crossing", "protocol_test_accuracy_int8"
        a_ok = v(t, free) <= HALF * v(p, free) + 1e-12
        b_ok = v(t, opp) <= HALF * v(p, opp) + 1e-12
        c_ok = v(t, early) >= v(p, early) - EARLY - 1e-12
        d_ok = v(t, prot) >= v(p, prot) - PROTOCOL - 1e-12
        res["criterion"] = {
            "rule": "means over the seeds: (a) free windows called a lane change, T at most half of P; (b) the opposite direction "
                    "called in windows holding a completed lane change, T at most half of P; (c) correct 2 s before the crossing, "
                    "T at most 5 points below P; (d) accuracy on the protocol's test windows, T at most 3 points below P",
            "a_free_windows_called_a_lane_change_P_T": [v(p, free), v(t, free)], "a_met": bool(a_ok),
            "b_opposite_after_a_lane_change_P_T": [v(p, opp), v(t, opp)], "b_met": bool(b_ok),
            "c_correct_2_s_before_P_T": [v(p, early), v(t, early)], "c_met": bool(c_ok),
            "d_protocol_test_accuracy_P_T": [v(p, prot), v(t, prot)], "d_met": bool(d_ok),
            "met": bool(a_ok and b_ok and c_ok and d_ok)}
    # the corrected measures (11:48): K's own records, P and T rescored
    known_rows = [r for r in rows if r["arm"] == "K"]
    if RESCORED.exists():
        known_rows += [json.loads(line) for line in RESCORED.read_text().splitlines()]
    acc = {r["arm"]: {} for r in rows}
    for r in rows:
        acc[r["arm"]][r["seed"]] = r["protocol_test_accuracy_int8"]
    lk = {}
    for arm in ARMS:
        rs = sorted((r for r in known_rows if r["arm"] == arm), key=lambda r: r["seed"])
        if rs:
            lk[arm] = {"seeds": [r["seed"] for r in rs],
                       "protocol_test_accuracy_int8": mean_std([acc[arm][r["seed"]] for r in rs]),
                       **{name: mean_std([f(r["complete_tracks_test"]) for r in rs]) for name, f in KNOWN.items()}}
    res["label_known"] = {"note": "windows whose label is known only (correction of 11:47); K was trained after the correction, "
                                  "P and T are the kept networks scored again", "arms": lk}
    if "K" in lk and "P" in lk:
        p, k = lk["P"], lk["K"]
        v = lambda a, key: a[key]["mean"]                    # noqa: E731
        a_ok = v(k, "lane_keeping_from_6_s_called") <= HALF * v(p, "lane_keeping_from_6_s_called") + 1e-12
        b_ok = v(k, "opposite_without_a_return") <= HALF * v(p, "opposite_without_a_return") + 1e-12
        c_ok = v(k, "correct_2_s_before_the_crossing") >= v(p, "correct_2_s_before_the_crossing") - EARLY - 1e-12
        d_ok = v(k, "protocol_test_accuracy_int8") >= v(p, "protocol_test_accuracy_int8") - PROTOCOL - 1e-12
        res["label_known"]["criterion_K_against_P"] = {
            "a_lane_keeping_from_6_s_called_P_K": [v(p, "lane_keeping_from_6_s_called"), v(k, "lane_keeping_from_6_s_called")], "a_met": bool(a_ok),
            "b_opposite_without_a_return_P_K": [v(p, "opposite_without_a_return"), v(k, "opposite_without_a_return")], "b_met": bool(b_ok),
            "c_correct_2_s_before_P_K": [v(p, "correct_2_s_before_the_crossing"), v(k, "correct_2_s_before_the_crossing")], "c_met": bool(c_ok),
            "d_protocol_test_accuracy_P_K": [v(p, "protocol_test_accuracy_int8"), v(k, "protocol_test_accuracy_int8")], "d_met": bool(d_ok),
            "met": bool(a_ok and b_ok and c_ok and d_ok)}
    SUMMARY.write_text(json.dumps(res, indent=1))
    for arm, a in lk.items():
        print(f"label known, arm {arm}, seeds {a['seeds']}")
        for name in ("protocol_test_accuracy_int8", *KNOWN):
            print(f"   {name:66s} {a[name]['mean']:10.4f} +- {a[name]['std']}")
    print(json.dumps(res["label_known"].get("criterion_K_against_P"), indent=1))
    for arm, a in arms.items():
        print(f"arm {arm} ({a['trained_on']}), seeds {a['seeds']}, {a['training_windows']} training windows, epochs {a['epochs_run']}")
        for name in ("protocol_test_accuracy_int8", *MEASURES):
            print(f"   {name:66s} {a[name]['mean']:10.4f} +- {a[name]['std']}")
    print(json.dumps(res.get("criterion"), indent=1))


def opposite_by_kind(pred_all: np.ndarray, lens: np.ndarray, cross: list, kinds: list) -> dict:
    """exiD: the opposite direction called in windows that hold a completed lane change not followed by a crossing
    back (the "label_known" measure of track_stream.measures_of), split by the kind of the completed lane change."""
    w_starts = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
    acc = np.zeros((len(tsx.KIND_NAMES), 2), np.int64)
    for t, (n, cr, kd) in enumerate(zip(lens, cross, kinds)):
        if not cr:
            continue
        k, phase, _, to_next, next_dir, since, prev_dir = ts.phases(int(n), cr)
        pred = pred_all[w_starts[t]:w_starts[t] + len(k)]
        known = (to_next <= ts.HORIZON) | ((int(n) - 1) - k >= ts.HORIZON)
        back = (to_next <= ts.HORIZON) & (next_dir == 3 - prev_dir)
        m = (phase == 3) & known & ~back
        if not m.any():
            continue
        prev = np.searchsorted(np.array([c for c, _ in cr]), k, side="right") - 1
        kind = np.array(kd)[np.clip(prev, 0, None)]
        opposite = (pred != 0) & (pred != prev_dir)
        for kk in range(len(tsx.KIND_NAMES)):
            mm = m & (kind == kk)
            acc[kk] += (int((opposite & mm).sum()), int(mm.sum()))
    return {tsx.KIND_NAMES[kk]: {"share": ts.share(*acc[kk]), "windows": int(acc[kk][1])} for kk in range(len(tsx.KIND_NAMES))}


def report_exid():
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    z = np.load(tracks_file("test"))
    lens = z["lens"].astype(np.int64)
    cross, kinds = json.loads(str(z["cross"])), json.loads(str(z["kinds"]))

    def mean_std(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {"mean": round(st.mean(vals), 5), "std": round(st.stdev(vals), 5) if len(vals) > 1 else None,
                "values": [round(v, 5) for v in vals]}
    listed = {"runs_per_vehicle_hour": MEASURES["runs_per_vehicle_hour"],
              "runs_of_0.6_s_or_more_per_vehicle_hour": MEASURES["runs_of_0.6_s_or_more_per_vehicle_hour"],
              "correct_1_s_before": MEASURES["correct_1_s_before"], "correct_3_s_before": MEASURES["correct_3_s_before"],
              "correct_5_s_before": MEASURES["correct_5_s_before"]}
    arms = {}
    for arm in ARMS:
        rs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"])
        if not rs:
            continue
        kinds_by_seed = [opposite_by_kind(np.load(KEEP / f"{arm}_{MODEL}_seed{r['seed']}_test_tracks.npz")["cls"].astype(np.int64),
                                          lens, cross, kinds) for r in rs]
        arms[arm] = {"trained_on": ARMS[arm], "seeds": [r["seed"] for r in rs], "training_windows": rs[0]["training_windows"],
                     "track_windows_per_class_before_balancing": rs[0]["track_windows_per_class_before_balancing"],
                     "epochs_run": [r["epochs_run"] for r in rs],
                     "protocol_test_accuracy_int8": mean_std([r["protocol_test_accuracy_int8"] for r in rs]),
                     **{name: mean_std([f(r["complete_tracks_test"]) for r in rs]) for name, f in KNOWN.items()},
                     **{name: mean_std([f(r["complete_tracks_test"]) for r in rs]) for name, f in listed.items()},
                     "opposite_without_a_return_by_kind_of_the_completed_lane_change": {
                         name: {"share": mean_std([b[name]["share"] for b in kinds_by_seed]), "windows": kinds_by_seed[0][name]["windows"]}
                         for name in tsx.KIND_NAMES[1:]}}
    d = np.array([dd for c in cross for _, dd in c])
    k = np.array([kk for ks in kinds for kk in ks])
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dataset": "exiD", "records": len(rows),
           "test_tracks": {"segments": int(len(lens)), "frames": int(lens.sum()), "vehicle_hours": round(float(lens.sum()) / ts.FPS / 3600, 2),
                           "recordings": sorted({int(r) for r in z["rec"]}), "crossings": int(len(d)),
                           "right": int((d == 1).sum()), "left": int((d == 2).sum()),
                           "by_kind": {tsx.KIND_NAMES[kk]: int((k == kk).sum()) for kk in range(1, len(tsx.KIND_NAMES))}},
           "arms": arms}
    if "P" in arms and "K" in arms:
        p, kk_ = arms["P"], arms["K"]
        v = lambda a, key: a[key]["mean"]                    # noqa: E731
        one = v(p, "opposite_without_a_return") >= EXISTS - 1e-12
        two = v(kk_, "opposite_without_a_return") <= HALF * v(p, "opposite_without_a_return") + 1e-12
        three = v(kk_, "correct_2_s_before_the_crossing") >= v(p, "correct_2_s_before_the_crossing") - EARLY - 1e-12
        four = v(kk_, "protocol_test_accuracy_int8") >= v(p, "protocol_test_accuracy_int8") - PROTOCOL - 1e-12
        res["criterion"] = {
            "rule": "means over the seeds, fixed 2026-11-18 16:30: (1) P calls the opposite direction in at least half of the "
                    "windows that hold a completed lane change not followed by a crossing back; (2) K at most half of P there; "
                    "(3) K correct 2 s before the crossing at most 5 points below P; (4) K accuracy on the protocol's test windows "
                    "at most 3 points below P",
            "1_opposite_without_a_return_P": v(p, "opposite_without_a_return"), "1_met": bool(one),
            "2_opposite_without_a_return_P_K": [v(p, "opposite_without_a_return"), v(kk_, "opposite_without_a_return")], "2_met": bool(two),
            "3_correct_2_s_before_P_K": [v(p, "correct_2_s_before_the_crossing"), v(kk_, "correct_2_s_before_the_crossing")], "3_met": bool(three),
            "4_protocol_test_accuracy_P_K": [v(p, "protocol_test_accuracy_int8"), v(kk_, "protocol_test_accuracy_int8")], "4_met": bool(four),
            "met": bool(one and two and three and four)}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["test_tracks"]))
    for arm, a in arms.items():
        print(f"arm {arm} ({a['trained_on']}), seeds {a['seeds']}, {a['training_windows']} training windows, epochs {a['epochs_run']}")
        for name in ("protocol_test_accuracy_int8", *KNOWN, *listed):
            m = a[name]
            print(f"   {name:66s} {m['mean']:10.4f} +- {m['std']}" if m else f"   {name:66s} -")
        for name, b in a["opposite_without_a_return_by_kind_of_the_completed_lane_change"].items():
            print(f"   opposite without a return, after {name:28s} {b['share']['mean'] if b['share'] else float('nan'):10.4f} ({b['windows']} windows)")
    print(json.dumps(res.get("criterion"), indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report_exid() if DATASET == "exid" else report()
    elif "--rescore" in sys.argv:
        rescore()
    else:
        run([a for a in sys.argv[1:] if a in ARMS] or list(ARMS))
