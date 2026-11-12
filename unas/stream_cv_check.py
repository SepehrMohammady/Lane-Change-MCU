"""Windows from whole drives on drivers nobody has looked at: a cross-validation over drivers (2026-11-12).

unas/stream_train_check.py showed, on the official test drivers, that an intention network trained on
windows taken from whole drives is right on about 90% of the frames of whole drives, against 80% for the
same network trained on the lab's prepared windows. That was a listed measure, not that study's criterion
(which failed), the test drivers had been looked at several times, and one of the seven is a copy of a
training drive (datasets/dmir/duplicate_sessions.py). This script repeats the comparison with the measure
fixed first and on drivers that no result had used. Everything below was fixed before the script existed
(local notes, 2026-11-12 15:20).

  drives     the 48 different drives with lane changes (users 1-50 without 24 and 25, whose files are
             copies of 23 and 9). "Not looked at": the 34 that were never a test or validation drive
             (all but 2, 7, 13, 18, 31, 36; 5, 8, 10, 12, 16, 19, 27; and 9, in the test sets as user 25).
  folds      the user numbers in ascending order, permuted with numpy default_rng(20261112); six folds of
             eight. Fold k: its drives are held out, those of fold k+1 (mod 6) stop the training early,
             the other 32 train.
  arms       same architecture (the deployed intention network, 2,336 parameters), same recipe
             (input_scaling_check.train), seeds 100-104, inputs as deployed:
             L  the lab's windows of the training drives (the rows of the three prepared intention sets
                cut from their sessions, datasets/dmir/session_stream.py rows; an identical row once),
                early stopping on the lab's windows of the early-stopping drives;
             D  windows from whole drives: every frame from the 50th on that is not ignored, class of its
                last frame, classes balanced by subsampling (rng 0), sessions with lane changes; early
                stopping on the same construction for the early-stopping drives.
  int8       as deployed: 500 calibration windows of the arm's own training set (rng 42), limited to
             +-4; scored as int8 on the PC.
  measures   per held-out drive, session with lane changes. PRIMARY: share of the scored frames with the
             correct class. Also: accuracy on the lab's windows of that drive; share of the free-labelled
             frames 1.5 to 6.3 s after a lane change on which the opposite lane change is announced; the
             raw output's event measures of datasets/dmir/decision_rule.py; the baseline session where
             there is one. A drive's value is the mean over the seeds of the fold that holds it out.
  criterion  on the 34 drives not looked at, all three: (a) mean of D minus L in frames correct at least
             8 points; (b) D above L for at least 30 of the 34 drives; (c) D's mean accuracy on the lab's
             windows at most 2 points below L's.

Run in the WSL dmir_nas venv, from the fork (as stream_train_check.py); FOLDS_TO_RUN=0,1,2 limits a
process to some folds, OUT_NAME gives it its own record file:
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data SEEDS=100,101,102,103,104 \\
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/stream_cv_check.py
  .venv\\Scripts\\python unas/stream_cv_check.py --report

Needs datasets/dmir/session_stream.py build, build val, build train, build more, rows (and build
more_baseline for the baseline sessions of users 37-46). Output: datasets/dmir/results/seeds/stream_cv.jsonl
(one line per fold, arm and seed; resumable) and datasets/dmir/results/hil/stream_cv.json (the report).
The networks are kept in datasets/dmir/data/stream_cv/ (not versioned).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_train_check as stc                             # noqa: E402  (numpy only)

dr = stc.dr
STREAM, TASK = stc.STREAM, stc.TASK
KEEP = REPO / "datasets" / "dmir" / "data" / "stream_cv"
OUT = REPO / "datasets" / "dmir" / "results" / "seeds" / os.environ.get("OUT_NAME", "stream_cv.jsonl")
SUMMARY = REPO / "datasets" / "dmir" / "results" / "hil" / f"{OUT.stem}.json"
COPIES = (24, 25)
DRIVES = [u for u in range(1, 51) if u not in COPIES]
LOOKED_AT = (2, 7, 13, 18, 31, 36, 5, 8, 10, 12, 16, 19, 27, 9)
_ORDER = np.random.default_rng(20261112).permutation(len(DRIVES))
FOLDS = [sorted(DRIVES[i] for i in _ORDER[8 * k:8 * k + 8]) for k in range(6)]
ARMS = ("L", "D")
GAIN, WINS, COST = 0.08, 30, 0.02                            # the criterion


def sessions_table() -> dict:
    """Every session by name: raw frames, the frames as the networks get them, labels, lane changes."""
    n = json.loads((STREAM / f"{TASK}_norm.json").read_text())
    a, b, lo, hi = (np.array(n[k]) for k in ("a", "b", "train_min", "train_max"))
    table = {}
    for split in ("test", "val", "train", "more", "more_baseline"):
        path = STREAM / f"{TASK}_sessions{'' if split == 'test' else '_' + split}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        lens = z["seq_lens"].astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        events = json.loads(str(z["events"]))
        for s, name in enumerate(str(v) for v in z["names"]):
            i, k = int(starts[s]), int(lens[s])
            raw = z["feats"][i:i + k]
            table[name] = {"feats": raw, "x": np.clip(raw.astype(np.float64) * a + b, lo, hi).astype(np.float32),
                           "ttlc": z["ttlc"][i:i + k], "direction": z["direction"][i:i + k],
                           "events": events[name]}
    return table


def pack(table: dict, names: list):
    """Some sessions as one set, in the form stream_train_check and decision_rule take."""
    z = {"seq_lens": np.array([len(table[n]["x"]) for n in names]), "names": np.array(names),
         "feats": np.concatenate([table[n]["feats"] for n in names]),
         "ttlc": np.concatenate([table[n]["ttlc"] for n in names]),
         "direction": np.concatenate([table[n]["direction"] for n in names]),
         "events": json.dumps({n: table[n]["events"] for n in names})}
    return z, np.concatenate([table[n]["x"] for n in names])


def lab_rows(lab, rows, drives):
    """The lab's windows that were cut from the given drives, an identical row once."""
    xs, ys = [], []
    for part in ("train", "val", "test"):
        x, y = lab._data[part]
        m = np.isin(rows[f"{part}_drive"], drives) & rows[f"{part}_first"]
        xs.append(x[m])
        ys.append(np.asarray(y)[m])
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


def per_session(s: dict) -> dict:
    ok = (s["cls"] == s["label"])[s["scored"]]
    free = s["scored"] & (s["label"] == 0)
    after = np.zeros(3, int)                                 # free driving / the completed direction / the opposite
    for i in np.flatnonzero(s["jump"]):
        d = int(s["label"][i - 1]) if i and s["label"][i - 1] else 0
        if not d:
            continue
        f = np.arange(i + 15, min(i + 64, len(free)))
        f = f[free[f]]
        p = s["cls"][f]
        after += np.array([(p == 0).sum(), (p == d).sum(), ((p != 0) & (p != d)).sum()])
    return {"frames": int(ok.size), "frames_correct": round(float(ok.mean()), 5),
            "after_a_lane_change": {"frames": int(after.sum()), "completed_direction": int(after[1]),
                                    "opposite_direction": int(after[2])},
            "raw_output": dr.score([s], dr.RAW)}


def score_drives(tf, icc, tfl: bytes, table: dict, names: list) -> dict:
    z, x = pack(table, names)
    outs, last, scale = stc.drive_outputs(tf, icc, tfl, z, x)
    return {s["name"]: per_session(s) for s in dr.sessions_of({**z, "window_last_frame": last, "out_scale": scale}, outs)}


def run(folds, arms):
    import input_clip_check as icc
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters

    tf, keras, sv = isc.tf, isc.keras, isc.sv
    os.environ["DMIR_SCALE"] = ""
    KEEP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = ({(r["fold"], r["arm"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())}
            if OUT.exists() else set())
    search, model = isc.CHOICES[TASK]
    h5 = str(isc.NV / search / f"{model}.h5")
    smoke = "_smoke" if sv.EPOCH_CAP else ""

    lab = sv.DMIR_Dataset(task=TASK)                         # the lab's sets, val and test limited to the training range
    rows = np.load(STREAM / f"{TASK}_rows_by_drive.npz")
    for part in ("train", "val", "test"):
        assert len(lab._data[part][0]) == len(rows[f"{part}_drive"]), part
    table = sessions_table()
    for k in folds:
        held, stop = FOLDS[k], FOLDS[(k + 1) % 6]
        train = [u for u in DRIVES if u not in held and u not in stop]
        names_held = [f"user{u}LC" for u in held] + [f"user{u}N" for u in held if f"user{u}N" in table]
        lab_held = {u: lab_rows(lab, rows, [u]) for u in held}
        for arm in arms:
            if all((k, arm, s) in done for s in sv.SEEDS):
                continue
            if arm == "L":
                x, y = lab_rows(lab, rows, train)
                val, what = lab_rows(lab, rows, stop), "the lab's windows of the training drives"
            else:
                x, y, counts = stc.stream_windows(*pack(table, [f"user{u}LC" for u in train]), "LC",
                                                  np.random.default_rng(0))
                val = stc.stream_windows(*pack(table, [f"user{u}LC" for u in stop]), "LC", np.random.default_rng(0))[:2]
                what = f"windows from whole drives, scored frames per class before balancing {counts}"
            rng = np.random.default_rng(42)                  # the 500 calibration windows, as convert_int8_io
            calib = np.clip(x[rng.choice(len(x), size=min(500, len(x)), replace=False)],
                            -stc.BOUND, stc.BOUND).astype(np.float32)
            for seed in sv.SEEDS:
                if (k, arm, seed) in done:
                    continue
                t0 = time.perf_counter()
                kept = KEEP / f"fold{k}_{arm}_seed{seed}{smoke}.h5"
                if kept.exists():
                    net, epochs = keras.models.load_model(str(kept), compile=False), None
                else:
                    net, epochs = isc.train(h5, stc.Windows(tf, (x, y), val), True, seed)
                    net.save(str(kept))
                conv = tf.lite.TFLiteConverter.from_keras_model(net)
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                conv.representative_dataset = lambda s=calib: ([s[i:i + 1]] for i in range(len(s)))
                conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                conv.inference_input_type = tf.int8
                conv.inference_output_type = tf.int8
                tfl = conv.convert()
                (KEEP / f"fold{k}_{arm}_seed{seed}{smoke}_int8_io.tflite").write_bytes(tfl)
                scored = score_drives(tf, icc, tfl, table, names_held)
                per = {}
                for u in held:
                    lx, ly = lab_held[u]
                    d = dict(scored[f"user{u}LC"])
                    d["lab_windows"] = int(len(lx))
                    d["lab_window_accuracy"] = round(float((icc.predict_int8(tf, tfl, lx)[0].argmax(1) == ly).mean()), 5)
                    if f"user{u}N" in scored:
                        d["baseline_session"] = scored[f"user{u}N"]
                    per[f"user{u}"] = d
                rec = {"fold": k, "arm": arm, "seed": seed, "held_out": held, "early_stopping_drives": stop,
                       "training_drives": train, "training_windows": int(len(x)), "training_set": what,
                       "early_stopping_windows": int(len(val[0])), "epochs_run": epochs,
                       "params": int(net.count_params()), "int8_io_bytes": len(tfl), "calibration_clip": stc.BOUND,
                       "drives": per, "wall_s": round(time.perf_counter() - t0, 1),
                       "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                with open(OUT, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"fold {k} arm {arm} seed {seed}: {len(x)} training windows, {epochs} epochs, "
                      f"{rec['wall_s']} s", flush=True)            # no result is printed while the run is going
                keras.backend.clear_session()


def report():
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    per = {arm: {} for arm in ARMS}                          # arm -> user -> the records of its seeds
    fold_of = {}
    for r in rows:
        for u, d in r["drives"].items():
            per[r["arm"]].setdefault(u, []).append(d)
            fold_of[u] = r["fold"]
    users = sorted(set(per["L"]) & set(per["D"]), key=lambda u: int(u[4:]))
    seeds = {arm: sorted({r["seed"] for r in rows if r["arm"] == arm}) for arm in ARMS}

    def rate(d, key):                                        # an event rate of the raw output, None without events
        return d["raw_output"].get(key)

    measures = {
        "frames_correct": lambda d: d["frames_correct"],
        "accuracy_on_the_labs_windows": lambda d: d["lab_window_accuracy"],
        "opposite_announced_after_a_lane_change": lambda d: (d["after_a_lane_change"]["opposite_direction"]
                                                             / max(d["after_a_lane_change"]["frames"], 1)),
        "timely_share": lambda d: rate(d, "timely_share"),
        "alarm_runs_per_hour": lambda d: rate(d, "false_alarms_per_hour"),
        "alarm_runs_of_0.5_s_or_more_per_hour": lambda d: rate(d, "false_alarms_of_0.5_s_or_more_per_hour"),
        "free_frames_in_alarm": lambda d: rate(d, "free_frames_in_alarm"),
        "baseline_session_frames_correct": lambda d: d.get("baseline_session", {}).get("frames_correct"),
        "baseline_session_alarm_runs_per_hour": lambda d: (d.get("baseline_session", {}).get("raw_output", {})
                                                           .get("false_alarms_per_hour")),
    }

    def value(arm, u, f):
        vals = [f(d) for d in per[arm][u]]
        return None if any(v is None for v in vals) else st.mean(vals)

    drive_rows = []
    for u in users:
        row = {"drive": u, "fold": fold_of[u], "looked_at_before": int(u[4:]) in LOOKED_AT,
               "seeds": [len(per[a][u]) for a in ARMS]}
        for name, f in measures.items():
            row[name] = {a: (None if (v := value(a, u, f)) is None else round(v, 4)) for a in ARMS}
        drive_rows.append(row)

    def summary(pick):
        sel = [r for r in drive_rows if pick(r)]
        out = {"drives": len(sel)}
        for name in measures:
            pairs = [(r[name]["L"], r[name]["D"]) for r in sel if r[name]["L"] is not None and r[name]["D"] is not None]
            if not pairs:
                continue
            l, d = np.array(pairs).T
            out[name] = {"drives_with_a_value": len(pairs), "L": round(float(l.mean()), 4), "D": round(float(d.mean()), 4),
                         "D_minus_L": round(float((d - l).mean()), 4), "D_above_L_in": int((d > l).sum())}
            if name == "frames_correct":
                rng = np.random.default_rng(0)
                boot = (d - l)[rng.integers(0, len(l), size=(10000, len(l)))].mean(axis=1)
                out[name]["D_minus_L_95_interval_over_drives"] = [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]
                out[name]["smallest_and_largest_gain"] = [round(float((d - l).min()), 4), round(float((d - l).max()), 4)]
        return out

    fresh = summary(lambda r: not r["looked_at_before"])
    fc, lab = fresh.get("frames_correct", {}), fresh.get("accuracy_on_the_labs_windows", {})
    a_ok = fc.get("D_minus_L", 0) >= GAIN - 1e-12
    b_ok = fc.get("D_above_L_in", 0) >= WINS
    c_ok = lab.get("D_minus_L", -1) >= -COST - 1e-12
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "records": len(rows), "records_expected": len(FOLDS) * len(ARMS) * 5, "seeds": seeds,
           "folds": [{"fold": k, "held_out": FOLDS[k], "early_stopping": FOLDS[(k + 1) % 6]} for k in range(6)],
           "training_windows": {a: sorted({r["training_windows"] for r in rows if r["arm"] == a}) for a in ARMS},
           "epochs_run_mean": {a: round(st.mean(r["epochs_run"] for r in rows if r["arm"] == a and r["epochs_run"]), 1)
                               for a in ARMS if any(r["arm"] == a and r["epochs_run"] for r in rows)},
           "criterion": {"rule": "on the 34 drives not looked at before: (a) mean of D minus L in frames correct "
                                 "at least 8 points, (b) D above L for at least 30 drives, (c) D's mean accuracy "
                                 "on the lab's windows at most 2 points below L's",
                         "a_gain": fc.get("D_minus_L"), "a_met": bool(a_ok),
                         "b_drives_with_D_above_L": fc.get("D_above_L_in"), "b_met": bool(b_ok),
                         "c_cost_on_the_labs_windows": lab.get("D_minus_L"), "c_met": bool(c_ok),
                         "met": bool(a_ok and b_ok and c_ok)},
           "drives_not_looked_at_before": fresh,
           "drives_looked_at_before": summary(lambda r: r["looked_at_before"]),
           "all_drives": summary(lambda r: True),
           "per_drive": drive_rows}
    SUMMARY.write_text(json.dumps(res, indent=1))
    print(f"{len(rows)} of {res['records_expected']} records; seeds {seeds}")
    for key in ("drives_not_looked_at_before", "drives_looked_at_before", "all_drives"):
        s = res[key]
        print(f"{key} ({s['drives']} drives):")
        for name in measures:
            if name in s:
                m = s[name]
                extra = (f", 95% interval {m['D_minus_L_95_interval_over_drives']}, range {m['smallest_and_largest_gain']}"
                         if name == "frames_correct" else "")
                print(f"   {name}: L {m['L']:.4f}, D {m['D']:.4f}, D-L {m['D_minus_L']:+.4f}, D above L in "
                      f"{m['D_above_L_in']} of {m['drives_with_a_value']}{extra}")
    print(json.dumps(res["criterion"], indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        todo = [int(v) for v in os.environ.get("FOLDS_TO_RUN", "0,1,2,3,4,5").split(",")]
        run(todo, [a for a in sys.argv[1:] if a in ARMS] or list(ARMS))
