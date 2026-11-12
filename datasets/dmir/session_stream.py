"""LCIR on whole drives: the raw sessions of the official test drivers as frame streams (2026-11-11).

The prepared pickles hold shuffled, class-balanced 50-step windows. A deployed network sees something
else: every frame of a drive, in order. This script rebuilds that stream for the seven official test
drivers (2, 7, 13, 18, 25, 31, 36), both phases of each (LC = lane changes allowed, N = baseline without
overtaking), from the raw 10 Hz sessions of the lab's data-preparation archive (Hi-Drive CDF, H5).

The provider's preprocessing is recovered, not guessed (as datasets/dmir/scenario_race.py did for the
classification layout): every pickle channel is an affine map of one raw signal, and the map is fitted
on windows that are located in the task's test pickle by correlation and must then reproduce them. The
LCR and LCL pickles have their own scalers, so each task gets its own map.

Frame labels follow the lab's rule (LC-Intention-Framework, h5_to_CSV_and_labeling.py): a lane change at
frame i when the distance to the right lane line jumps by more than half a lane width, to the left if
that distance is then under half a lane width. In the prepared pickles the time to the lane change is
0.0 s on the last frame before the jump, so frames i-41..i-1 carry 4.0..0.0 s here, back to the previous
lane change but not past it; the 15 frames from i on are ignored unless the next lane change relabels
them; every other frame is free driving. With this rule the frame labels equal the pickles on every
located window: all 15,170 targets of the LCR and LCL test pickles and all 19,722 classes of the
classification test pickle. The pickles hold fewer lane changes than this rule finds in the sessions (382 of 527 for
the test drivers); why the others were left out is not known.

  .venv\\Scripts\\python datasets/dmir/session_stream.py build
Output (not versioned, datasets/dmir/data/stream/): <task>_sessions.npz (raw frames of all sessions in
the task's channel order, session lengths and names, frame labels) and <task>_norm.json (the map, its
fit error, the training range); a summary goes to datasets/dmir/results/hil/session_stream.json.
The frames are replayed on the board by unas/hil_prepare.py and unas/hil_stream.py (a copy of the
network's file under datasets/dmir/data/sessions/ selects this mode).

The seven official validation drivers (5, 8, 10, 12, 16, 19, 27), for choices that must not be made on
the test drivers (2026-11-12):
  .venv\\Scripts\\python datasets/dmir/session_stream.py build val
The map is not fitted again: the one recovered on the test drivers is applied unchanged and must
reproduce the windows of the validation pickles, which is a check of the map on other drivers. Output:
<task>_sessions_val.npz and datasets/dmir/results/hil/session_stream_val.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import scenario_race as sr                                  # noqa: E402  (load_session, raw_matrix, windows)
from src.features import FEATURES_BY_TASK                   # noqa: E402

DATA = HERE / "data"
OUT = DATA / "stream"
ZIP = ROOT / "Materials" / "Overtaking2.zip"
USERS = {"test": (2, 7, 13, 18, 25, 31, 36), "val": (5, 8, 10, 12, 16, 19, 27)}   # the official split
PHASES = ("LC", "N")
PICKLES = {"classification": ("data-classification", "multi"), "regression_lcr": ("data-prepared-lcr", "lcr"),
           "regression_lcl": ("data-regression-lcl", "lcl")}
W = 50


def ensure_h5(split: str = "test") -> list[str]:
    """The sessions of the split's drivers, unpacked next to the other raw samples."""
    names = [f"user{u}{ph}" for u in USERS[split] for ph in PHASES]
    missing = [n for n in names if not (sr.H5DIR / f"{n}.h5").exists()]
    if missing:
        with zipfile.ZipFile(ZIP) as z:
            members = {i.filename.split("/")[-1]: i for i in z.infolist()}
            for n in missing:
                (sr.H5DIR / f"{n}.h5").write_bytes(z.read(members[f"{n}.h5"]))
    return names


def to_layout(m_cls: np.ndarray, task: str) -> np.ndarray:
    """Raw matrix of scenario_race (classification channel order) in the task's channel order."""
    src = FEATURES_BY_TASK["classification"]
    return m_cls[:, [src.index(n) for n in FEATURES_BY_TASK[task]]]


def frame_labels(ll) -> tuple[np.ndarray, np.ndarray, list]:
    """Per frame: time to the lane change (4.1 = free driving, NaN = ignored) and its direction
    (0 none, 1 right, 2 left); and the lane changes as (frame, direction). The lab's rule."""
    right = ll["sLaneLine"]["Dy"][:, 0].astype(float)
    width = ll["EgoLaneWidth"].astype(float)
    n = len(right)
    events, prev = [], right[0]
    for i in range(n):
        if abs(right[i] - prev) > width[i] * 2 / 4:
            events.append((i, 2 if right[i] < width[i] / 2 else 1))
        prev = right[i]
    ttlc = np.full(n, 4.1)
    direction = np.zeros(n, np.int8)
    previous = -1
    for i, d in events:                                     # in time order
        for off in range(41):
            k = i - 1 - off                                 # 0.0 s on the last frame before the jump
            if k < 0 or k < previous:                       # back to the previous lane change, not past it
                break
            ttlc[k], direction[k] = round(0.1 * off, 1), d
        for off in range(15):
            k = i + off
            if k >= n:
                break
            ttlc[k], direction[k] = np.nan, 0
        previous = i
    return ttlc, direction, events


def locate(m: np.ndarray, x: np.ndarray, ch: tuple[int, int]) -> np.ndarray:
    """(pickle window, raw start) pairs of identical windows, by correlation on two channels."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def z(a):
        t = torch.from_numpy(np.asarray(a, np.float64)).float().to(dev)
        mu, s = t.mean(1, keepdim=True), t.std(1, keepdim=True)
        return torch.where(s > 1e-8, (t - mu) / s, torch.zeros_like(t))
    r0, r1 = z(sr.windows(m[:, ch[0]])), z(sr.windows(m[:, ch[1]]))
    p0, p1 = z(x[:, :, ch[0]]), z(x[:, :, ch[1]])
    pairs = []
    for i in range(0, len(x), 4096):
        c = torch.minimum(p0[i:i + 4096] @ r0.T, p1[i:i + 4096] @ r1.T) / (W - 1)
        best, arg = c.max(dim=1)
        for k in torch.where(best > 0.9995)[0]:
            pairs.append((i + int(k), int(arg[k])))
    return np.array(pairs).reshape(-1, 2)


def fit(samples: list[tuple[np.ndarray, np.ndarray]], n_feat: int):
    """pickle = a * raw + b per channel from (raw windows, pickle windows) of all drivers."""
    a, b, err = np.zeros(n_feat), np.zeros(n_feat), np.zeros(n_feat)
    for c in range(n_feat):
        xr = np.concatenate([r[:, :, c].reshape(-1) for r, _ in samples])
        yp = np.concatenate([p[:, :, c].reshape(-1) for _, p in samples])
        if xr.std() < 1e-12:
            a[c], b[c] = 0.0, float(np.median(yp))
        else:
            (a[c], b[c]), *_ = np.linalg.lstsq(np.stack([xr, np.ones_like(xr)], 1), yp, rcond=None)
        err[c] = float(np.abs(a[c] * xr + b[c] - yp).max())
    return a, b, err


def build(split: str = "test"):
    """Sessions of the split's drivers. test: the map is fitted and written; val: the map of the test
    drivers is read and checked on the windows of the validation pickles."""
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if split == "test" else f"_{split}"
    names = ensure_h5(split)
    sessions = {}
    for n in names:
        ego, ll, ob, _ = sr.load_session(n)
        ttlc, direction, events = frame_labels(ll)
        sessions[n] = (sr.raw_matrix(ego, ll, ob), ttlc, direction, events)
    in_pickle_by_task = {}
    summary = {"sessions": {n: {"frames": int(len(s[0])), "minutes": round(len(s[0]) / 600, 1),
                                "lane_changes_right": sum(d == 1 for _, d in s[3]),
                                "lane_changes_left": sum(d == 2 for _, d in s[3])} for n, s in sessions.items()},
               "tasks": {}}
    for task, (folder, sfx) in PICKLES.items():
        feats = FEATURES_BY_TASK[task]
        ch = (feats.index("yawRate"), feats.index("latAcceleration"))
        x_test = pickle.load(open(DATA / folder / f"x_{split}_{sfx}.pkl", "rb")).astype(np.float64)
        samples, found, stray, label_gap, in_pickle = {}, set(), {}, 0.0, {}
        y_test = np.asarray(pickle.load(open(DATA / folder / f"y_{split}_{sfx}.pkl", "rb"))).reshape(-1)
        for n in names:
            m = to_layout(sessions[n][0], task)
            pairs = locate(m, x_test, ch)
            if not len(pairs):
                continue
            if not n.endswith("LC"):                        # the pickles were cut from the LC sessions; a few
                stray[n] = int(len(pairs))                  # look-alike windows of quiet driving turn up elsewhere
                continue
            samples[n] = (sr.windows(m)[pairs[:, 1]], x_test[pairs[:, 0]])
            found.update(int(p) for p in pairs[:, 0])
            last = pairs[:, 1] + W - 1                      # the frame each located window ends on
            in_pickle[n] = set(int(v) for v in last)
            if task != "classification":                    # the pickle's target against the frame label
                label_gap = max(label_gap, float(np.nanmax(np.abs(y_test[pairs[:, 0]] - sessions[n][1][last]))))
            else:
                cls = np.where(np.isnan(sessions[n][1][last]), -1,
                               np.where(sessions[n][1][last] <= 4.0, sessions[n][2][last], 0))
                label_gap = max(label_gap, float((cls != y_test[pairs[:, 0]]).mean()))
        if split == "test":
            a, b, err = fit(list(samples.values()), len(feats))
            x_train = pickle.load(open(DATA / folder / f"x_train_{sfx}.pkl", "rb"))
            lo_std, hi_std = (x_train.min(axis=(0, 1)).astype(np.float64),
                              x_train.max(axis=(0, 1)).astype(np.float64))
            del x_train
            (OUT / f"{task}_norm.json").write_text(json.dumps(
                {"task": task, "features": feats, "a": a.tolist(), "b": b.tolist(), "max_abs_error": err.tolist(),
                 "train_min": lo_std.tolist(), "train_max": hi_std.tolist(),
                 "note": "pickle = a * raw + b per channel; training range in pickle units"}, indent=1))
        else:                                               # the map recovered on the test drivers, unchanged
            norm = json.loads((OUT / f"{task}_norm.json").read_text())
            a, b = np.array(norm["a"]), np.array(norm["b"])
            err = np.array([max(float(np.abs(a[c] * r[:, :, c] + b[c] - p[:, :, c]).max())
                                for r, p in samples.values()) for c in range(len(feats))])
        per_driver = {n: float(max(np.abs(a[c] * r[:, :, c] + b[c] - p[:, :, c]).max() for c in range(len(feats))))
                      for n, (r, p) in samples.items()}     # the one map on each driver's windows
        ok = bool(err.max() <= 1e-5)
        lens, cat, ttlc, direction, events, held = [], [], [], [], {}, []
        for n in names:
            m, t, d, ev = sessions[n]
            lens.append(len(m))
            cat.append(to_layout(m, task).astype(np.float32))
            ttlc.append(t)
            direction.append(d)
            events[n] = [[int(i), int(dd)] for i, dd in ev]
            flag = np.zeros(len(m), bool)                   # frames a window of the task's test pickle ends on
            flag[sorted(in_pickle.get(n, ()))] = True
            held.append(flag)
        np.savez_compressed(OUT / f"{task}_sessions{tag}.npz", feats=np.concatenate(cat), seq_lens=np.array(lens),
                            names=np.array(names), ttlc=np.concatenate(ttlc).astype(np.float32),
                            direction=np.concatenate(direction), events=json.dumps(events),
                            **{f"in_{split}_pickle": np.concatenate(held)})
        summary["tasks"][task] = {
            f"{split}_pickle_windows": int(len(x_test)), "located_in_the_sessions": len(found),
            "map": "fitted on these windows" if split == "test" else "recovered on the test drivers, applied unchanged",
            "sessions_the_pickle_was_cut_from": sorted(samples), "look_alike_windows_elsewhere": stray,
            "max_abs_error_of_the_map": float(err.max()), "worst_channel": feats[int(err.argmax())],
            "max_abs_error_per_driver": per_driver, "recovered": ok,
            ("share_of_located_windows_with_another_class" if task == "classification"
             else "max_abs_difference_to_the_pickle_target_s"): label_gap,
            "unscaled_channels": [feats[c] for c in range(len(feats)) if abs(a[c] - 1) < 1e-9 and abs(b[c]) < 1e-9],
            "constant_channels": [feats[c] for c in range(len(feats)) if a[c] == 0]}
        in_pickle_by_task[task] = in_pickle
        print(json.dumps({task: {k: v for k, v in summary["tasks"][task].items() if k != "max_abs_error_per_driver"}}),
              flush=True)
    # lane changes of the sessions that the regression test pickles hold (their 41 frames before the jump)
    held = {}
    for n in names:
        if not n.endswith("LC"):
            continue
        frames_in = set()
        for task in ("regression_lcr", "regression_lcl"):
            frames_in |= in_pickle_by_task.get(task, {}).get(n, set())
        ev = [i for i, _ in sessions[n][3] if i >= W + 41]
        held[n] = {"lane_changes": len(ev),
                   "in_the_test_pickles": int(sum(sum(k in frames_in for k in range(i - 41, i)) >= 20 for i in ev))}
    summary[f"lane_changes_held_by_the_{split}_pickles"] = held
    res = HERE / "results" / "hil"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"session_stream{tag}.json").write_text(json.dumps(summary, indent=1))
    tot = sum(s["frames"] for s in summary["sessions"].values())
    print(f"{len(names)} sessions, {tot} frames = {tot / 600:.0f} minutes; "
          f"{sum(s['lane_changes_right'] + s['lane_changes_left'] for s in summary['sessions'].values())} lane changes")


if __name__ == "__main__":
    {"build": build}[sys.argv[1] if len(sys.argv) > 1 else "build"](sys.argv[2] if len(sys.argv) > 2 else "test")
