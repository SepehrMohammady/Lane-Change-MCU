"""Self-labelled learning of the last layer on whole LCIR drives: the online-learning pilot (2026-11-19).

The deployed intention network runs on a microcontroller after it is flashed. The car knows, 41 frames (4.1 s)
after every frame, which class that frame had: the lab's frame rule needs only the lane-line signal
(datasets/dmir/session_stream.py, frame_labels). This script asks whether the device can use these labels to
update the network's last layer while it drives, and whether that improves on the fixed networks of the
cross-validation over drivers (unas/stream_cv_check.py). Everything below was fixed before the script existed
(local notes, 2026-11-19 16:35).

  networks   the 60 of the cross-validation (datasets/dmir/data/stream_cv/, arms L and D, seeds 100-104),
             float32 on the PC; inputs as in the CV, clipped to +-4 like their int8 calibration; a drive is
             scored with the five networks of the fold that held it out. Fixed layer: the trained Dense layer
             (119 x 3 + 3) on the Flatten output; the feature layers never change.
  stream     per drive the session with lane changes, then its baseline session where there is one, the
             state carried over; one output per frame from the 50th on.
  labels     the label of frame t is used from frame t+41 on, if frame t has an output and is not ignored.
  memory     per class the last 128 labelled feature vectors; updates start once every class holds 20.
  settings   G: one gradient step of softmax cross-entropy per new label on 4 vectors per class from the
             memories, plus lambda times the squared distance to the fixed layer; eta 0.003 / 0.01 / 0.03 x
             lambda 0 / 0.01. R: every 10 frames, the fixed layer plus the class-balanced ridge solution that
             maps [f; 1] to s (onehot(y) - softmax(fixed logits)), alpha = a x trace of the class-balanced Gram
             matrix; s 2 / 8 x a 0.01 / 0.1.
  selection  arm D, the 14 drives looked at before: the largest mean gain in frames correct, among the
             settings whose timely share is at most 5 points below the fixed layer's and whose alarm runs of
             0.5 s or more per hour are not above it. None qualifies: NO GO, the 34 drives are not scored.
  criterion  selected setting, arm D, the 34 drives not looked at, all four: (a) mean gain in frames correct of
             at least 2.0 points, (b) a gain on at least 24 drives, (c) timely share at most 5 points below
             the fixed layer's, (d) alarm runs of 0.5 s or more per hour not above the fixed layer's.

Measures as in the CV, prequential (every frame is scored with the layer as it is at that frame); per drive the
mean over its five networks.

Run in the WSL dmir_nas venv, from the fork, as stream_cv_check.py:
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data ~/dmir_nas/bin/python \\
    /mnt/c/Projects/PhD/DIMIR/unas/online_pilot.py tune     # the 14 drives, ten settings, arm D
  ... online_pilot.py final                                 # the 34 drives, the selected setting, arms D and L
  .venv\\Scripts\\python unas/online_pilot.py --report
SMOKE=1 (with OUT_NAME set) runs one fold, one seed, one drive and two settings, to check the code.
Output: datasets/dmir/results/seeds/online_pilot.jsonl (one line per phase, setting, fold, arm and seed; resumable)
and datasets/dmir/results/hil/online_pilot.json (the report).
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
import stream_cv_check as cv                                 # noqa: E402  (numpy only)

dr, W, BOUND = cv.dr, cv.stc.W, cv.stc.BOUND
OUT = REPO / "datasets" / "dmir" / "results" / "seeds" / os.environ.get("OUT_NAME", "online_pilot.jsonl")
SUMMARY = REPO / "datasets" / "dmir" / "results" / "hil" / f"{OUT.stem}.json"
CV_REPORT = REPO / "datasets" / "dmir" / "results" / "hil" / "stream_cv.json"
DELAY, MEM, MIN_CLASS, PER_CLASS, REFIT = 41, 128, 20, 4, 10
SETTINGS = ([{"id": f"G_eta{e}_lam{lam}", "family": "G", "eta": e, "lam": lam}
             for e in (0.003, 0.01, 0.03) for lam in (0.0, 0.01)]
            + [{"id": f"R_s{s}_a{a}", "family": "R", "s": s, "a": a} for s in (2, 8) for a in (0.01, 0.1)])
TUNE = sorted(cv.LOOKED_AT)
FRESH = [u for u in cv.DRIVES if u not in cv.LOOKED_AT]
GAIN, WINS, TIMELY = 0.02, 24, 0.05                          # the criterion
A5 = "alarm_runs_of_0.5_s_or_more_per_hour"
MEASURES = ("frames_correct", "second_half_frames_correct", "opposite_announced_after_a_lane_change",
            "timely_share", "alarm_runs_per_hour", A5, "baseline_session_frames_correct",
            "baseline_session_alarm_runs_per_hour")
assert len(TUNE) == 14 and len(FRESH) == 34


# ------------------------------------------------------------------------------------------------ the networks
def network(keras, k: int, arm: str, seed: int):
    """Feature model (up to Flatten) and the fixed last layer of one network of the cross-validation."""
    net = keras.models.load_model(str(cv.KEEP / f"fold{k}_{arm}_seed{seed}.h5"), compile=False)
    flat = [lay for lay in net.layers if type(lay).__name__ == "Flatten"]
    dense = [lay for lay in net.layers if type(lay).__name__ == "Dense"]
    assert len(flat) == 1 and len(dense) == 1 and net.layers[-1] is dense[0]
    w, b = dense[0].get_weights()
    return keras.Model(net.inputs, flat[0].output), w.astype(np.float64), b.astype(np.float64)


def stream_of(table: dict, u: int, feat) -> list:
    """The drive's sessions in order: features of every window, the frame labels, usable for learning."""
    out = []
    for name in (f"user{u}LC", f"user{u}N"):
        if name not in table:
            continue
        s = table[name]
        x = np.clip(s["x"], -BOUND, BOUND)
        win = np.lib.stride_tricks.sliding_window_view(x, W, axis=0).transpose(0, 2, 1)
        f = feat.predict(np.ascontiguousarray(win), batch_size=4096, verbose=0).astype(np.float64)
        t = s["ttlc"][W - 1:]
        label = np.where(t <= 4.0, s["direction"][W - 1:], 0).astype(np.int64)
        out.append({"name": name, "f": f, "label": label, "usable": ~np.isnan(t)})
    return out


# ---------------------------------------------------------------------------------------------- the update
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def adapt(stream: list, w0: np.ndarray, b0: np.ndarray, setting: dict, rng) -> tuple[dict, dict]:
    """Predicted class of every window, the layer updated from the labels as they become known."""
    d = w0.shape[0]
    w, b = w0.copy(), b0.copy()
    feats = np.zeros((3, MEM, d))
    resid = np.zeros((3, MEM, 3))
    count, pos = np.zeros(3, np.int64), np.zeros(3, np.int64)
    ridge = setting["family"] == "R"
    gram = np.zeros((3, d + 1, d + 1))
    cross = np.zeros((3, d + 1, 3))
    eye, warm_at, updates, before = np.eye(3), None, 0, 0
    out = {}
    for s in stream:
        f, label, usable = s["f"], s["label"], s["usable"]
        r_all = setting["s"] * (eye[label] - softmax(f @ w0 + b0)) if ridge else None
        cls = np.empty(len(f), np.int64)
        for j in range(len(f)):
            i = j - DELAY
            if i >= 0 and usable[i]:
                c = int(label[i])
                slot = int(pos[c])
                if ridge:
                    if count[c] == MEM:                      # the oldest vector leaves the memory
                        fo = np.append(feats[c, slot], 1.0)
                        gram[c] -= np.outer(fo, fo)
                        cross[c] -= np.outer(fo, resid[c, slot])
                    fn = np.append(f[i], 1.0)
                    gram[c] += np.outer(fn, fn)
                    cross[c] += np.outer(fn, r_all[i])
                    resid[c, slot] = r_all[i]
                feats[c, slot] = f[i]
                pos[c] = (slot + 1) % MEM
                count[c] = min(count[c] + 1, MEM)
                if (count >= MIN_CLASS).all():
                    if warm_at is None:
                        warm_at = before + j
                    if not ridge:
                        x = np.concatenate([feats[k, rng.integers(0, count[k], PER_CLASS)] for k in range(3)])
                        g = (softmax(x @ w + b) - eye[np.repeat(np.arange(3), PER_CLASS)]) / len(x)
                        w = w - setting["eta"] * (x.T @ g + 2 * setting["lam"] * (w - w0))
                        b = b - setting["eta"] * (g.sum(0) + 2 * setting["lam"] * (b - b0))
                        updates += 1
            if ridge and warm_at is not None and j % REFIT == 0:
                a = sum(gram[k] / count[k] for k in range(3))
                q = sum(cross[k] / count[k] for k in range(3))
                theta = np.linalg.solve(a + setting["a"] * np.trace(a) * np.eye(d + 1), q)
                w, b = w0 + theta[:d], b0 + theta[d]
                updates += 1
            cls[j] = int(np.argmax(f[j] @ w + b))
        out[s["name"]] = cls
        before += len(f)
    return out, {"warm_from_window": warm_at, "updates": updates}


# ---------------------------------------------------------------------------------------------- the measures
def score(table: dict, names: list, cls_by_name: dict) -> dict:
    """The CV's per-session measures (stream_cv_check.per_session) of a class sequence per session."""
    z, _ = cv.pack(table, names)
    lens = z["seq_lens"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    last = np.concatenate([a + np.arange(W - 1, n) for a, n in zip(starts, lens)])
    cls = np.concatenate([cls_by_name[n] for n in names])
    outs = np.zeros((len(cls), 3), np.int8)
    outs[np.arange(len(cls)), cls] = 127                    # the argmax is the class, as sessions_of reads it
    ref = {"seq_lens": z["seq_lens"], "names": z["names"], "window_last_frame": last, "events": z["events"],
           "feats": z["feats"], "ttlc": z["ttlc"], "direction": z["direction"], "out_scale": 1.0}
    res = {}
    for s in dr.sessions_of(ref, outs):
        d = cv.per_session(s)
        ok = s["scored"] & (np.arange(len(s["cls"])) >= len(s["cls"]) // 2)
        d["second_half_frames_correct"] = round(float((s["cls"] == s["label"])[ok].mean()), 5)
        res[s["name"]] = d
    return res


def drive_record(table: dict, u: int, fixed: dict, adapted: dict, info: dict) -> dict:
    names = [n for n in (f"user{u}LC", f"user{u}N") if n in table]
    rec = {}
    for key, cls in (("fixed", fixed), ("adapted", adapted)):
        sc = score(table, names, cls)
        d = dict(sc[f"user{u}LC"])
        if f"user{u}N" in sc:
            d["baseline_session"] = sc[f"user{u}N"]
        rec[key] = d
    return {**rec, **info}


# ---------------------------------------------------------------------------------------------- the runs
def records() -> list:
    return [json.loads(line) for line in OUT.read_text().splitlines()] if OUT.exists() else []


def run(phase: str) -> None:
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters

    keras = isc.keras
    os.environ["DMIR_SCALE"] = ""
    smoke = os.environ.get("SMOKE") == "1"
    if smoke and "OUT_NAME" not in os.environ:
        raise SystemExit("SMOKE=1 needs its own OUT_NAME")
    table = cv.sessions_table()
    if phase == "tune":
        settings, arms, drives = SETTINGS, ("D",), TUNE
    else:
        chosen = selection(summaries())["selected"]
        if chosen is None:
            print("NO GO: no setting qualified on the 14 drives looked at before; the 34 drives are not scored")
            return
        settings, arms, drives = [s for s in SETTINGS if s["id"] == chosen], ("D", "L"), FRESH
    seeds = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]
    if smoke:
        settings, seeds = [SETTINGS[0], SETTINGS[6]], seeds[:1]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["phase"], r["setting"], r["fold"], r["arm"], r["seed"]) for r in records()}
    for k, fold in enumerate(cv.FOLDS):
        held = [u for u in fold if u in drives]
        if smoke:
            held = held[:1]
        if not held:
            continue
        for arm in arms:
            for seed in seeds:
                todo = [s for s in settings if (phase, s["id"], k, arm, seed) not in done]
                if not todo:
                    continue
                t0 = time.perf_counter()
                feat, w0, b0 = network(keras, k, arm, seed)
                streams = {u: stream_of(table, u, feat) for u in held}
                fixed = {u: {s["name"]: np.argmax(s["f"] @ w0 + b0, -1) for s in streams[u]} for u in held}
                for setting in todo:
                    per = {}
                    for u in held:
                        rng = np.random.default_rng([k, seed, u, SETTINGS.index(setting)])
                        cls, info = adapt(streams[u], w0, b0, setting, rng)
                        per[f"user{u}"] = drive_record(table, u, fixed[u], cls, info)
                    rec = {"phase": phase, "setting": setting["id"], "fold": k, "arm": arm, "seed": seed,
                           "drives": per, "float32": True,
                           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                    with open(OUT, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                keras.backend.clear_session()
                print(f"{phase} fold {k} arm {arm} seed {seed}: {len(held)} drives, {len(todo)} settings, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)     # no result is printed while running
        if smoke:
            break


# ---------------------------------------------------------------------------------------------- the report
def measures_of(d: dict) -> dict:
    ro, base = d["raw_output"], d.get("baseline_session", {})
    after = d["after_a_lane_change"]
    return {"frames_correct": d["frames_correct"], "second_half_frames_correct": d["second_half_frames_correct"],
            "opposite_announced_after_a_lane_change": after["opposite_direction"] / max(after["frames"], 1),
            "timely_share": ro.get("timely_share"), "alarm_runs_per_hour": ro.get("false_alarms_per_hour"),
            A5: ro.get("false_alarms_of_0.5_s_or_more_per_hour"),
            "baseline_session_frames_correct": base.get("frames_correct"),
            "baseline_session_alarm_runs_per_hour": base.get("raw_output", {}).get("false_alarms_per_hour")}


def per_drive(rows: list, phase: str, setting: str, arm: str) -> dict:
    """drive -> measure -> (fixed, adapted), each the mean over the drive's networks."""
    got = {}
    for r in rows:
        if (r["phase"], r["setting"], r["arm"]) == (phase, setting, arm):
            for u, d in r["drives"].items():
                got.setdefault(u, []).append(d)
    out = {}
    for u, ds in got.items():
        fx, ad = [measures_of(d["fixed"]) for d in ds], [measures_of(d["adapted"]) for d in ds]
        row = {"networks": len(ds)}
        for name in MEASURES:
            vals = [m[name] for m in fx] + [m[name] for m in ad]
            row[name] = (None if any(v is None for v in vals)
                         else (st.mean(m[name] for m in fx), st.mean(m[name] for m in ad)))
        out[u] = row
    return out


def summary_of(tab: dict) -> dict:
    out = {"drives": len(tab), "networks_per_drive": sorted({r["networks"] for r in tab.values()})}
    for name in MEASURES:
        pairs = [tab[u][name] for u in tab if tab[u][name] is not None]
        if not pairs:
            continue
        f, a = np.array(pairs).T
        out[name] = {"drives_with_a_value": len(pairs), "fixed": round(float(f.mean()), 4),
                     "adapted": round(float(a.mean()), 4), "gain": round(float((a - f).mean()), 4),
                     "adapted_above_fixed_in": int((a > f).sum())}
        if name == "frames_correct":
            rng = np.random.default_rng(0)
            boot = (a - f)[rng.integers(0, len(f), size=(10000, len(f)))].mean(axis=1)
            out[name]["gain_95_interval_over_drives"] = [round(float(np.percentile(boot, q)), 4) for q in (2.5, 97.5)]
            out[name]["smallest_and_largest_gain"] = [round(float((a - f).min()), 4), round(float((a - f).max()), 4)]
    return out


def summaries() -> dict:
    rows = records()
    res = {"tune": {}, "final": {}}
    for s in SETTINGS:
        tab = per_drive(rows, "tune", s["id"], "D")
        if tab:
            res["tune"][s["id"]] = summary_of(tab)
    for arm in ("D", "L"):
        for s in SETTINGS:
            tab = per_drive(rows, "final", s["id"], arm)
            if tab:
                res["final"][f"{s['id']} {arm}"] = {"summary": summary_of(tab), "per_drive": tab}
    return res


def qualifies(s: dict) -> bool:
    return (s["timely_share"]["adapted"] >= s["timely_share"]["fixed"] - TIMELY - 1e-12
            and s[A5]["adapted"] <= s[A5]["fixed"] + 1e-12)


def selection(res: dict) -> dict:
    tune = res["tune"]
    complete = {k: v for k, v in tune.items() if v["drives"] == len(TUNE) and v["networks_per_drive"] == [5]}
    if len(complete) < len(SETTINGS):
        return {"selected": None, "complete": False, "settings_complete": sorted(complete)}
    ok = [(v["frames_correct"]["gain"], k) for k, v in complete.items() if qualifies(v)]
    return {"selected": max(ok)[1] if ok else None, "complete": True,
            "qualified": sorted(k for _, k in ok)}


def report() -> None:
    res = summaries()
    sel = selection(res)
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "records": len(records()),
           "rule": "local notes 2026-11-19 16:35", "selection": sel,
           "tune": res["tune"]}
    print(f"{out['records']} records")
    for k, v in res["tune"].items():
        fc, ts, a5 = v["frames_correct"], v["timely_share"], v[A5]
        print(f"tune {k:18s} {v['drives']} drives: frames correct {fc['fixed']:.4f} -> {fc['adapted']:.4f} "
              f"({fc['gain']:+.4f}, {fc['adapted_above_fixed_in']} better); timely {ts['fixed']:.4f} -> "
              f"{ts['adapted']:.4f}; runs >=0.5 s/h {a5['fixed']:.1f} -> {a5['adapted']:.1f}; "
              f"{'qualifies' if qualifies(v) else 'does not qualify'}")
    print("selection:", json.dumps(sel))
    chosen = sel["selected"]
    if chosen and f"{chosen} D" in res["final"]:
        fin = res["final"]
        out["final"] = {key: val["summary"] for key, val in fin.items()}
        out["final_per_drive"] = {key: val["per_drive"] for key, val in fin.items()}
        d = fin[f"{chosen} D"]["summary"]
        fc, ts, a5 = d["frames_correct"], d["timely_share"], d[A5]
        crit = {"rule": "selected setting, arm D, the 34 drives not looked at: (a) mean gain in frames correct at "
                        "least 2.0 points, (b) a gain on at least 24 drives, (c) timely share at most 5 points below "
                        "the fixed layer's, (d) alarm runs of 0.5 s or more per hour not above the fixed layer's",
                "drives": d["drives"], "networks_per_drive": d["networks_per_drive"],
                "a_gain": fc["gain"], "a_met": fc["gain"] >= GAIN - 1e-12,
                "b_drives_with_a_gain": fc["adapted_above_fixed_in"], "b_met": fc["adapted_above_fixed_in"] >= WINS,
                "c_timely": [ts["fixed"], ts["adapted"]], "c_met": ts["adapted"] >= ts["fixed"] - TIMELY - 1e-12,
                "d_runs": [a5["fixed"], a5["adapted"]], "d_met": a5["adapted"] <= a5["fixed"] + 1e-12}
        crit["complete"] = d["drives"] == len(FRESH) and d["networks_per_drive"] == [5]
        crit["met"] = bool(crit["complete"] and crit["a_met"] and crit["b_met"] and crit["c_met"] and crit["d_met"])
        out["criterion"] = crit
        if CV_REPORT.exists():                               # the float fixed layer against the CV's int8 numbers
            cvd = {r["drive"]: r["frames_correct"] for r in json.loads(CV_REPORT.read_text())["per_drive"]}
            for arm in ("D", "L"):
                key = f"{chosen} {arm}"
                if key in fin:
                    tab = fin[key]["per_drive"]
                    diffs = [tab[u]["frames_correct"][0] - cvd[u][arm] for u in tab if u in cvd]
                    out.setdefault("float_fixed_minus_cv_int8_frames_correct", {})[arm] = {
                        "mean": round(float(np.mean(diffs)), 4), "largest": round(float(np.max(np.abs(diffs))), 4)}
        for key, val in out["final"].items():
            print(f"final {key}: " + "; ".join(f"{m} {val[m]['fixed']:.4f} -> {val[m]['adapted']:.4f}"
                                               for m in MEASURES if m in val))
        print(json.dumps(crit, indent=1))
    SUMMARY.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] in ("tune", "final"):
        run(sys.argv[1])
    else:
        raise SystemExit("usage: online_pilot.py tune | final | --report")
