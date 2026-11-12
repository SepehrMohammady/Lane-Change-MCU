"""LCIR: the session files of users 24 and 25 hold the drives of users 23 and 9 (found 2026-11-12).

A scan of the raw sessions of all 50 drivers showed two pairs of files with the same signals:

  user25LC.h5 = user9LC.h5 and user25N.h5 = user9N.h5   (first frame 2024-09-25, the day user 9 drove)
  user24LC.h5 = user23LC.h5                             (first frame 2024-10-10 13:47, between user 23's
                                                         baseline session and user 24's)

The files are not byte-identical (they were written on different days) but every signal of every frame
is. The sessions with lane changes are those of the dataset's public record (Zenodo, DOI
10.5281/zenodo.16686054: the local files have the record's checksums, and the record lists the same
checksum for user9LC.csv and user25LC.csv, and for user23LC.csv and user24LC.csv). So the dataset holds
48 different drives with lane changes, not 50.

Why it matters: user 25 is one of the seven official test drivers and user 9 is a training driver. The
windows of the test sets that were cut from "user 25" are windows of a drive the networks were trained
on. Users 23 and 24 are both training drivers; their drive only counts twice in training.

This script measures what that does to the numbers of this repository:

  sessions   pairs of raw session files with identical signals (all files in Materials/h5-samples);
  pickles    per task: the rows of the test set that come from the copied drive, how many of them are
             bit-identical to a row of the training set, and the same count over all test rows;
  board      the deployed networks' outputs on the test sets, as measured on the board (unas/hil_stream.py):
             all rows, the rows of the six other test drivers, and the rows of the copied drive;
  drives     the whole drives of the test drivers (datasets/dmir/decision_rule.py): raw output and the
             rule chosen on the validation drivers, with and without the sessions of user 25.

Nothing here was fixed beforehand: the copies were found, not looked for. The numbers without user 25 are
the ones that describe drivers the networks have not seen; the pre-set tests of decision_rule.py and
unas/stream_train_check.py were made on all seven test drivers and stay as they were reported.

  .venv\\Scripts\\python datasets/dmir/duplicate_sessions.py
Needs session_stream.py build (the map and the sessions of the test drivers) and, for `board` and
`drives`, the replays in firmware/build/stream/. Output: datasets/dmir/results/hil/duplicate_sessions.json.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import decision_rule as dr                                  # noqa: E402
import scenario_race as sr                                  # noqa: E402
import session_stream as ss                                 # noqa: E402
from src.features import FEATURES_BY_TASK                   # noqa: E402

COPIES = {"user25": "user9", "user24": "user23"}            # the file's user: the user whose drive it holds
GROUPS = ("egoVehicle", "laneLines", "objects", "positioning")
NETS = {"searched on raw inputs": "lcir_v4k3sat", "searched on scaled inputs": "lcir_v4k3mm"}
OUT = HERE / "results" / "hil" / "duplicate_sessions.json"


def identical_sessions() -> list:
    """Groups of raw session files whose signals are the same in every frame."""
    import h5py
    by_digest = {}
    for p in sorted(sr.H5DIR.glob("user*.h5"), key=lambda q: (int("".join(c for c in q.stem if c.isdigit())), q.stem)):
        with h5py.File(p, "r") as f:
            h = hashlib.md5()
            for g in GROUPS:
                h.update(f[g][()].tobytes())
            first = float(f["egoVehicle"]["UTCTime"][0])
            entry = {"file": p.name, "frames": int(len(f["egoVehicle"])),
                     "written": str(f.attrs.get("creation_date", "")),
                     "file_md5": hashlib.md5(p.read_bytes()).hexdigest()}
        by_digest.setdefault(h.hexdigest(), {"first_frame_clock": first, "files": []})["files"].append(entry)
    groups = []
    for g in by_digest.values():
        if len(g["files"]) > 1:
            t = g["first_frame_clock"]
            t = t / 1e9 if t > 1e17 else t / 1e6 if t > 1e14 else t / 1e3 if t > 1e11 else t
            groups.append({"first_frame": datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                           "frames": g["files"][0]["frames"],
                           "files": [{k: e[k] for k in ("file", "written")} for e in g["files"]],
                           "files_byte_identical": len({e["file_md5"] for e in g["files"]}) == 1})
    return groups


def load_pickle(task: str, split: str):
    folder, sfx = ss.PICKLES[task]
    x = pickle.load(open(ss.DATA / folder / f"x_{split}_{sfx}.pkl", "rb"))
    y = np.asarray(pickle.load(open(ss.DATA / folder / f"y_{split}_{sfx}.pkl", "rb"))).reshape(-1)
    return x, y


def digests(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x)
    return np.array([hashlib.blake2b(x[i].tobytes(), digest_size=12).digest() for i in range(len(x))])


def rows_by_session(task: str, x_test: np.ndarray) -> dict:
    """Rows of the task's test set by the session with lane changes they were cut from."""
    norm = json.loads((ss.OUT / f"{task}_norm.json").read_text())
    a, b = np.array(norm["a"]), np.array(norm["b"])
    feats = FEATURES_BY_TASK[task]
    ch = (feats.index("yawRate"), feats.index("latAcceleration"))
    x64 = x_test.astype(np.float64)
    rows = {}
    for u in ss.USERS["test"]:
        n = f"user{u}LC"
        m = ss.to_layout(sr.raw_matrix(*sr.load_session(n)[:3]), task)
        pairs = ss.locate(m, x64, ch)
        gap = np.abs(sr.windows(m)[pairs[:, 1]] * a + b - x64[pairs[:, 0]]).max(axis=(1, 2))
        rows[n] = np.unique(pairs[gap <= 1e-5][:, 0])
    return rows


def pickles(task: str, rows: dict) -> dict:
    x_test, _ = load_pickle(task, "test")
    x_train, _ = load_pickle(task, "train")
    train = set(digests(x_train).tolist())
    del x_train
    in_train = np.array([d in train for d in digests(x_test).tolist()])
    copied = np.zeros(len(x_test), bool)
    copied[rows["user25LC"]] = True
    assigned = np.zeros(len(x_test), int)
    for r in rows.values():
        assigned[r] += 1
    return {"test_rows": int(len(x_test)), "rows_located_in_exactly_one_session": int((assigned == 1).sum()),
            "rows_per_test_driver": {n[:-2]: int(len(r)) for n, r in rows.items()},
            "rows_from_the_copied_drive": int(copied.sum()),
            "share_of_the_test_set": round(float(copied.mean()), 4),
            "of_them_bit_identical_to_a_training_row": int((copied & in_train).sum()),
            "test_rows_bit_identical_to_a_training_row": int(in_train.sum()),
            "of_them_from_the_six_other_drivers": int((in_train & ~copied).sum())}


def board(task: str, rows: dict) -> dict:
    """The deployed networks on the test set, from the board's outputs."""
    out = {}
    for name, stem in NETS.items():
        folder = dr.STREAM / f"{stem}_{task}_int8_io"
        if not (folder / "board_out_rate0.npz").exists():
            continue
        ref = np.load(folder / "reference.npz")
        o = np.load(folder / "board_out_rate0.npz")["outs"]
        y = ref["label"]
        copied = np.zeros(len(y), bool)
        copied[rows["user25LC"]] = True
        if task == "classification":
            ok = o.reshape(len(o), -1).argmax(1) == y

            def metric(m):
                return {"windows": int(m.sum()), "accuracy": round(float(ok[m].mean()), 4)}
        else:
            err = (o.reshape(-1).astype(np.float64) - float(ref["out_zp"])) * float(ref["out_scale"]) - y

            def metric(m):
                return {"windows": int(m.sum()), "rmse_s": round(float(np.sqrt((err[m] ** 2).mean())), 4)}
        per = {}
        for n, r in rows.items():
            m = np.zeros(len(y), bool)
            m[r] = True
            per[n[:-2]] = metric(m)
        out[name] = {"all_seven_test_drivers": metric(np.ones(len(y), bool)),
                     "six_drivers_not_seen_in_training": metric(~copied),
                     "the_copied_drive_only": metric(copied), "per_driver": per}
    return out


def drives() -> dict:
    """Whole drives of the test drivers, board outputs: with and without the sessions of user 25."""
    rule_file = HERE / "results" / "hil" / "decision_rule.json"
    chosen = json.loads(rule_file.read_text())["networks"] if rule_file.exists() else {}
    out = {}
    for name, stem in dr.NETS.items():
        folder = dr.STREAM / f"sessions_{stem}_int8_io"
        if not (folder / "board_out_rate0.npz").exists():
            continue
        sessions, source, _ = dr.load(stem, "test")
        sel = chosen.get(name, {}).get("validation_drivers", {}).get("selected_rule")
        rule = (sel["theta"], sel["hold_off_frames"], sel["persistence_frames"]) if sel else None

        def describe(pick):
            ok = np.concatenate([(s["cls"] == s["label"])[s["scored"]] for s in sessions if pick(s["name"])])
            lc = np.concatenate([(s["cls"] == s["label"])[s["scored"]] for s in sessions
                                 if pick(s["name"]) and s["name"].endswith("LC")])
            d = {"sessions": sum(pick(s["name"]) for s in sessions),
                 "frames_correct": round(float(ok.mean()), 4),
                 "frames_correct_in_sessions_with_lane_changes": round(float(lc.mean()), 4),
                 "raw_output": dr.score(sessions, dr.RAW, pick)}
            if rule:
                d["rule_chosen_on_the_validation_drivers"] = {**dr.named(rule), **dr.score(sessions, rule, pick)}
                r, c = d["raw_output"], d["rule_chosen_on_the_validation_drivers"]
                d["false_alarms_at_least_halved"] = bool(c["false_alarms_per_hour"] <= dr.HALVE * r["false_alarms_per_hour"])
                d["timely_share_within_5_points"] = bool(c["timely_share"] >= r["timely_share"] - dr.MARGIN - 1e-12)
            return d
        out[name] = {"outputs_from": source,
                     "all_seven_test_drivers": describe(lambda n: True),
                     "six_drivers_not_seen_in_training": describe(lambda n: not n.startswith("user25")),
                     "the_copied_drive_only": describe(lambda n: n.startswith("user25"))}
    return out


def main():
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "copies": {k: f"holds the drive of {v}" for k, v in COPIES.items()},
           "note": "found on 2026-11-12, not looked for; nothing here was fixed beforehand",
           "sessions_with_identical_signals": identical_sessions(), "tasks": {}}
    for g in res["sessions_with_identical_signals"]:
        print("identical signals:", [f["file"] for f in g["files"]], "first frame", g["first_frame"], flush=True)
    for task in ss.PICKLES:
        x_test, _ = load_pickle(task, "test")
        rows = rows_by_session(task, x_test)
        del x_test
        res["tasks"][task] = {"test_set": pickles(task, rows), "deployed_networks_on_the_board": board(task, rows)}
        t = res["tasks"][task]["test_set"]
        print(f"{task}: {t['rows_from_the_copied_drive']} of {t['test_rows']} test rows from the copied drive "
              f"({t['share_of_the_test_set']:.1%}), {t['of_them_bit_identical_to_a_training_row']} of them bit-identical "
              f"to a training row; {t['of_them_from_the_six_other_drivers']} such rows from the other drivers", flush=True)
        for name, b in res["tasks"][task]["deployed_networks_on_the_board"].items():
            print(f"   {name}: all {b['all_seven_test_drivers']}, six drivers {b['six_drivers_not_seen_in_training']}, "
                  f"copied drive {b['the_copied_drive_only']}")
    res["whole_drives_of_the_test_drivers"] = drives()
    for name, d in res["whole_drives_of_the_test_drivers"].items():
        for key in ("all_seven_test_drivers", "six_drivers_not_seen_in_training", "the_copied_drive_only"):
            v = d[key]
            r, c = v["raw_output"], v.get("rule_chosen_on_the_validation_drivers", {})
            print(f"{name}, {key}: frames correct {v['frames_correct']:.4f} "
                  f"({v['frames_correct_in_sessions_with_lane_changes']:.4f} in sessions with lane changes); raw "
                  f"{r['timely_share']:.3f} timely, {r['false_alarms_per_hour']}/h; rule "
                  f"{c.get('timely_share')}, {c.get('false_alarms_per_hour')}/h")
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
