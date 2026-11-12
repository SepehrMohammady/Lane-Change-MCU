"""LCIR intention network trained on windows taken from whole drives (2026-11-12).

The prepared LCIR windows hold a shortcut: no free-driving window contains a completed lane change, while
half of the windows before a right lane change do (datasets/dmir/lane_change_pairs.py). A network trained
on them announces the opposite lane change after every lane change, and no rule on its output removes
that on whole drives (datasets/dmir/decision_rule.py). This script asks whether the training windows are
the cause. The architecture (the intention network of the raw-input search), the recipe
(input_scaling_check.train) and the seeds are the same in every arm; only the training windows differ:

  A    the lab's training pickle (36 drivers): the five networks kept by input_clip_check.py, not retrained;
  A22  the lab's windows of the 22 training drivers that are in the local archive (the windows of the
       training pickle located in their sessions), early stopping on the lab's validation pickle;
  B    every frame of the same 22 drivers' sessions with lane changes, from the 50th frame on and not
       ignored, gives a window with the class of its last frame, whatever else the window holds; classes
       balanced by random subsampling (rng 0); early stopping on the validation drivers' windows built
       the same way;
  C    B plus the windows of the baseline sessions, before balancing;
  B36  as B with all 36 training drivers: the 22 of the archive and users 37-50, whose sessions with lane
       changes are in the dataset's public record (session_stream.py build more).

Each network is converted to int8 with int8 I/O as deployed (500 calibration windows of its own training
set, clipped to +-4) and scored on the whole drives of the validation and the test drivers with the
measures and the rule selection of decision_rule.py, on the lab's test pickle, and on what it says after
a lane change. Arms, measures and the criterion were fixed beforehand (local notes, 2026-11-12 11:31; A22
added at 11:34, before any training, when the training pickle turned out to hold 14 drivers that the
archive does not; B36 added at 11:39, before any result was read, when those drivers' sessions were found
in the public record): B is adopted if, on the test drivers and as means over the five seeds of the raw
output, its false alarms per hour are at most half of A's and its timely share is at most 5 points lower.
B36 against A is the same comparison with equal drivers.

Needs datasets/dmir/session_stream.py build, build val, build train and (for B36) build more. Run in the WSL dmir_nas venv, from
the fork (as input_clip_check.py):
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data SEEDS=100,101,102,103,104 \\
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/stream_train_check.py [arm ...]
  .venv\\Scripts\\python unas/stream_train_check.py --report     # means, the criterion, the summary file

Output: datasets/dmir/results/seeds/stream_windows.jsonl (one line per arm and seed; resumable) and
datasets/dmir/results/hil/stream_windows.json (the report). The trained networks and their int8 files are
kept in datasets/dmir/data/stream_train/ (not versioned). EPOCH_CAP=1 SEEDS=0 OUT_NAME=..._smoke.jsonl
for a smoke test.
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
sys.path.insert(0, str(REPO / "datasets" / "dmir"))
import decision_rule as dr                                   # noqa: E402  (numpy only)

STREAM = REPO / "datasets" / "dmir" / "data" / "stream"
KEEP = REPO / "datasets" / "dmir" / "data" / "stream_train"
OUT = REPO / "datasets" / "dmir" / "results" / "seeds" / os.environ.get("OUT_NAME", "stream_windows.jsonl")
SUMMARY = REPO / "datasets" / "dmir" / "results" / "hil" / "stream_windows.json"
TASK, W, BOUND, FPS = "classification", 50, 4.0, 10
ARMS = ("A", "A22", "B", "C", "B36")


def frames(split: str):
    """The split's sessions: raw frames, the frames as the networks get them, and the session table."""
    z = np.load(STREAM / f"{TASK}_sessions{'' if split == 'test' else '_' + split}.npz")
    n = json.loads((STREAM / f"{TASK}_norm.json").read_text())
    x = np.clip(z["feats"].astype(np.float64) * np.array(n["a"]) + np.array(n["b"]),
                np.array(n["train_min"]), np.array(n["train_max"])).astype(np.float32)
    return z, x


def stream_windows(z, x, phases, rng):
    """One window per scored frame of the sessions of the given phases, classes balanced by subsampling."""
    lens = z["seq_lens"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    last, label = [], []
    for s, name in enumerate(str(v) for v in z["names"]):
        if not name.endswith(phases):
            continue
        a, n = int(starts[s]), int(lens[s])
        t = z["ttlc"][a:a + n]
        cls = np.where(t <= 4.0, z["direction"][a:a + n], 0)
        f = np.flatnonzero(~np.isnan(t) & (np.arange(n) >= W - 1))
        last.append(a + f)
        label.append(cls[f])
    last, label = np.concatenate(last), np.concatenate(label).astype(np.int64)
    counts = np.bincount(label, minlength=3)
    keep = np.concatenate([rng.choice(np.flatnonzero(label == c), size=int(counts.min()), replace=False)
                           for c in range(3)])
    keep = keep[rng.permutation(len(keep))]
    view = np.lib.stride_tricks.sliding_window_view(x, W, axis=0)       # [first frame, feature, step]
    return np.ascontiguousarray(view[last[keep] - (W - 1)].transpose(0, 2, 1)), label[keep], counts.tolist()


def merged(*parts):
    """Several session sets as one, for taking windows."""
    z = {k: np.concatenate([p[0][k] for p in parts]) for k in ("seq_lens", "names", "ttlc", "direction")}
    return z, np.concatenate([p[1] for p in parts])


class Windows:
    """What input_scaling_check.train needs of a dataset."""

    def __init__(self, tf, train, val):
        self._tf, self._data = tf, {"train": train, "val": val}

    def train_dataset(self):
        return self._tf.data.Dataset.from_tensor_slices(self._data["train"])

    def validation_dataset(self):
        return self._tf.data.Dataset.from_tensor_slices(self._data["val"])


def drive_outputs(tf, icc, tfl: bytes, z, x):
    """int8 outputs of every window of every session, in session order, and the output scale."""
    lens = z["seq_lens"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    outs, last, scale = [], [], None
    for a, n in zip(starts, lens):
        win = np.lib.stride_tricks.sliding_window_view(x[a:a + n], W, axis=0).transpose(0, 2, 1)
        o, (scale, _) = icc.predict_int8(tf, tfl, np.ascontiguousarray(win), as_int8=True)
        outs.append(o)
        last.append(a + np.arange(W - 1, n))
    return np.concatenate(outs), np.concatenate(last), scale


def drive_measures(z, outs, last, scale):
    ref = {"seq_lens": z["seq_lens"], "names": z["names"], "window_last_frame": last, "events": z["events"],
           "feats": z["feats"], "ttlc": z["ttlc"], "direction": z["direction"], "out_scale": scale}
    sessions = dr.sessions_of(ref, outs)
    acc = {}
    for key, pick in (("all", lambda n: True), ("lane_change_sessions", lambda n: n.endswith("LC")),
                      ("baseline_sessions", lambda n: n.endswith("N"))):
        ok = np.concatenate([(s["cls"] == s["label"])[s["scored"]] for s in sessions if pick(s["name"])])
        acc[key] = round(float(ok.mean()), 4)
    after = np.zeros(3, int)                                 # free driving / the completed direction / the opposite
    for s in sessions:
        if not s["name"].endswith("LC"):
            continue
        free = s["scored"] & (s["label"] == 0)
        for i in np.flatnonzero(s["jump"]):
            d = int(s["label"][i - 1]) if i and s["label"][i - 1] else 0
            if not d:
                continue
            f = np.arange(i + 15, min(i + 64, len(free)))
            f = f[free[f]]
            p = s["cls"][f]
            after += np.array([(p == 0).sum(), (p == d).sum(), ((p != 0) & (p != d)).sum()])
    return sessions, acc, {k: round(float(v / max(after.sum(), 1)), 4)
                           for k, v in zip(("free driving", "the completed direction", "the opposite direction"), after)}


def run(arms):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import input_clip_check as icc
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters

    tf, keras, sv = isc.tf, isc.keras, isc.sv
    os.environ["DMIR_SCALE"] = ""
    KEEP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["arm"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    search, model = isc.CHOICES[TASK]
    h5 = str(isc.NV / search / f"{model}.h5")
    smoke = "_smoke" if sv.EPOCH_CAP else ""

    lab = sv.DMIR_Dataset(task=TASK)                         # the lab's pickles, val and test clipped to the training range
    xtr, ytr = lab._data["train"]
    drives = {split: frames(split) for split in ("train", "val", "test")}
    ztr, ftr = drives["train"]
    val_b = stream_windows(*drives["val"], "LC", np.random.default_rng(0))
    sets = {}

    def training_set(arm):
        if arm not in sets:
            if arm == "A":
                sets[arm] = (xtr, ytr, None, "the lab's training pickle")
            elif arm == "A22":
                mask = np.load(STREAM / f"{TASK}_train_pickle_located.npy")
                sets[arm] = (xtr[mask], ytr[mask], lab._data["val"], "the lab's windows of the 22 local drivers")
            else:
                src = merged(drives["train"], frames("more")) if arm == "B36" else (ztr, ftr)
                x, y, counts = stream_windows(*src, ("LC", "N") if arm == "C" else "LC", np.random.default_rng(0))
                sets[arm] = (x, y, val_b[:2], f"stream windows, scored frames per class before balancing {counts}")
        return sets[arm]

    for arm in arms:
        x, y, val, what = training_set(arm)
        rng = np.random.default_rng(42)                      # the 500 calibration windows, as convert_int8_io
        calib = np.clip(x[rng.choice(len(x), size=min(500, len(x)), replace=False)], -BOUND, BOUND).astype(np.float32)
        for seed in sv.SEEDS:
            if (arm, seed) in done:
                continue
            t0 = time.perf_counter()
            if arm == "A":
                kept, epochs = REPO / "datasets" / "dmir" / "data" / "clip_check" / f"{TASK}_{model}_seed{seed}.h5", None
                net = keras.models.load_model(str(kept), compile=False)
            else:
                kept = KEEP / f"{arm}_{model}_seed{seed}{smoke}.h5"
                if kept.exists():
                    net, epochs = keras.models.load_model(str(kept), compile=False), None
                else:
                    net, epochs = isc.train(h5, Windows(tf, (x, y), val), True, seed)
                    net.save(str(kept))
            conv = tf.lite.TFLiteConverter.from_keras_model(net)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = lambda s=calib: ([s[i:i + 1]] for i in range(len(s)))
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
            tfl = conv.convert()
            (KEEP / f"{arm}_{model}_seed{seed}{smoke}_int8_io.tflite").write_bytes(tfl)
            rec = {"arm": arm, "seed": seed, "training_windows": int(len(x)), "training_set": what,
                   "epochs_run": epochs, "params": int(net.count_params()), "int8_io_bytes": len(tfl),
                   "calibration_clip": BOUND}
            xte, yte = lab._data["test"]
            rec["lab_test_pickle_accuracy"] = {
                "float32": round(isc.metric(net.predict(xte, batch_size=4096, verbose=0), yte, True)["acc"], 5),
                "int8": round(isc.metric(icc.predict_int8(tf, tfl, xte)[0], yte, True)["acc"], 5)}
            scored = {}
            for split in ("val", "test"):
                z, fx = drives[split]
                outs, last, scale = drive_outputs(tf, icc, tfl, z, fx)
                scored[split] = drive_measures(z, outs, last, scale)
            grid_val = dr.grid(scored["val"][0])
            floor = grid_val[dr.RAW]["timely_share"] - dr.MARGIN
            sel = min((r for r, v in grid_val.items() if v["timely_share"] >= floor - 1e-12),
                      key=lambda r: (grid_val[r]["false_alarms"], r[1], r[2], r[0]))
            rec["selected_rule"] = dr.named(sel)
            for split, key in (("val", "validation_drivers"), ("test", "test_drivers")):
                sessions, acc, after = scored[split]
                rec[key] = {"raw_output": grid_val[dr.RAW] if split == "val" else dr.score(sessions, dr.RAW),
                            "selected": grid_val[sel] if split == "val" else dr.score(sessions, sel),
                            "frame_accuracy": acc, "output_after_a_lane_change": after,
                            "baseline_sessions_raw_output": dr.score(sessions, dr.RAW, lambda n: n.endswith("N"))}
            rec["wall_s"] = round(time.perf_counter() - t0, 1)
            rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t = rec["test_drivers"]["raw_output"]
            print(f"arm {arm} seed {seed}: test raw timely {t['timely_share']:.3f}, {t['false_alarms_per_hour']}/h, "
                  f"lab test pickle int8 {rec['lab_test_pickle_accuracy']['int8']:.4f} ({epochs} ep, {rec['wall_s']} s)",
                  flush=True)


def report():
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    seeds = sorted({r["seed"] for r in rows})

    def mean_std(vals):
        return {"mean": round(st.mean(vals), 4), "std": round(st.stdev(vals), 4) if len(vals) > 1 else None,
                "values": [round(v, 4) for v in vals]}

    def arm_summary(arm):
        rs = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["seed"])
        if not rs:
            return None
        out = {"seeds": [r["seed"] for r in rs], "training_windows": rs[0]["training_windows"],
               "training_set": rs[0]["training_set"],
               "lab_test_pickle_accuracy_int8": mean_std([r["lab_test_pickle_accuracy"]["int8"] for r in rs]),
               "lab_test_pickle_accuracy_float32": mean_std([r["lab_test_pickle_accuracy"]["float32"] for r in rs]),
               "selected_rules": [r["selected_rule"] for r in rs]}
        for key in ("validation_drivers", "test_drivers"):
            d = {}
            for which in ("raw_output", "selected"):
                d[which] = {m: mean_std([r[key][which][m] for r in rs])
                            for m in ("timely_share", "false_alarms_per_hour", "median_lead_s", "free_frames_in_alarm",
                                      "missed_share")}
            d["frame_accuracy"] = {k: mean_std([r[key]["frame_accuracy"][k] for r in rs])
                                   for k in ("all", "lane_change_sessions", "baseline_sessions")}
            d["output_after_a_lane_change"] = {k: mean_std([r[key]["output_after_a_lane_change"][k] for r in rs])
                                               for k in ("free driving", "the completed direction", "the opposite direction")}
            d["baseline_sessions_false_alarms_per_hour_raw_output"] = mean_std(
                [r[key]["baseline_sessions_raw_output"]["false_alarms_per_hour"] for r in rs])
            out[key] = d
        return out

    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seeds": seeds,
           "arms": {a: s for a in ARMS if (s := arm_summary(a))}}

    def against(arm, base):
        a, b = res["arms"].get(arm), res["arms"].get(base)
        if not a or not b:
            return None
        ta, tb = a["test_drivers"]["raw_output"], b["test_drivers"]["raw_output"]
        halves = ta["false_alarms_per_hour"]["mean"] <= dr.HALVE * tb["false_alarms_per_hour"]["mean"]
        keeps = ta["timely_share"]["mean"] >= tb["timely_share"]["mean"] - dr.MARGIN - 1e-12
        return {"false_alarms_per_hour": [ta["false_alarms_per_hour"]["mean"], tb["false_alarms_per_hour"]["mean"]],
                "timely_share": [ta["timely_share"]["mean"], tb["timely_share"]["mean"]],
                "false_alarms_at_least_halved": bool(halves), "timely_share_within_5_points": bool(keeps),
                "met": bool(halves and keeps)}
    res["criterion"] = {"rule": "test drivers, means over the seeds, raw output: false alarms per hour at most half "
                                "and timely share at most 5 points lower",
                        "B_against_A": against("B", "A"), "B36_against_A_equal_drivers": against("B36", "A"),
                        "B_against_A22_for_information": against("B", "A22"),
                        "C_against_A_for_information": against("C", "A")}
    SUMMARY.write_text(json.dumps(res, indent=1))
    for a, s in res["arms"].items():
        for key in ("validation_drivers", "test_drivers"):
            r, c = s[key]["raw_output"], s[key]["selected"]
            print(f"{a:4s} {key[:4]}: raw timely {r['timely_share']['mean']:.3f} +- {r['timely_share']['std']}, "
                  f"{r['false_alarms_per_hour']['mean']:.1f}/h +- {r['false_alarms_per_hour']['std']}; with rule "
                  f"{c['timely_share']['mean']:.3f}, {c['false_alarms_per_hour']['mean']:.1f}/h; frame accuracy "
                  f"{s[key]['frame_accuracy']['all']['mean']:.3f}; after a lane change opposite "
                  f"{s[key]['output_after_a_lane_change']['the opposite direction']['mean']:.3f}")
        print(f"     lab test pickle int8 {s['lab_test_pickle_accuracy_int8']['mean']:.4f} +- "
              f"{s['lab_test_pickle_accuracy_int8']['std']}; {s['training_windows']} training windows")
    print(json.dumps(res["criterion"], indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        run([a for a in sys.argv[1:] if a in ARMS] or list(ARMS))
