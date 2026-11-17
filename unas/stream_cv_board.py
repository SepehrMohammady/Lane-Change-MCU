"""The cross-validation's networks, measured on the board (2026-11-17).

unas/stream_cv_check.py compared, as int8 on the PC and over the 48 different LCIR drives, the intention
network trained on the lab's prepared windows (arm L) and on windows taken from whole drives (arm D): six
folds, five seeds, 60 networks. Here those 60 int8 files go through the STM32H7B3I-DK: for a network of
fold k the whole sessions of the fold's eight held-out drives, frame by frame (unas/hil_prepare.py,
unas/hil_stream.py; the board limits, maps and quantizes the raw features itself and decides on every
frame from the 50th on), and the study's report over drives is computed again from the board's outputs.
The files are kept as replayed in datasets/dmir/results/hil/stream_cv_board/.

This repeats on the hardware numbers that are known from the PC; it is not a new test of the finding, and
a replay is not a closed loop. Fixed beforehand (local notes, 2026-11-17 10:56):
  (1) agreement  for each network the board's int8 outputs equal those of the TFLite interpreter, given the
                 int8 inputs of the board's arithmetic, in at least 99.9% of its windows, and no output
                 differs by more than 2 LSB;
  (2) numbers    (i) for every held-out drive and arm the share of correct frames in the session with lane
                 changes (mean over the five seeds) from the board's outputs lies within 0.1 point of the
                 same share from the TFLite outputs; (ii) the study's criterion, evaluated on the board's
                 outputs, is met on the 34 drives that had never been a test or validation drive (whole-drive
                 windows minus lab's windows at least 8 points, above in at least 30 drives, at most 2
                 points below on the lab's windows); (iii) the two arm means of correct frames over those
                 34 drives lie within 0.2 points of the study's record;
  (3) cost       the 60 firmware images have the same flash and static RAM size to within 1%, and the mean
                 network time per frame of each lies within 3% of the mean of the 60 (3%: a network whose
                 padding value before the second convolution is 0 was 2.1% faster on 2026-11-16).
The study's accuracy on the lab's windows was computed on the stored windows, which are not replayed. In
its place stands the accuracy on the frames that the lab's windows of a drive end on (the in_*_pickle
marks of datasets/dmir/session_stream.py), from the same replay.

  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/stream_cv_board.py prepare     # WSL (TensorFlow)
  .venv\\Scripts\\python unas/stream_cv_board.py replay                                # Windows, board connected
  .venv\\Scripts\\python unas/stream_cv_board.py --report

Output: datasets/dmir/results/hil/stream_cv_board.jsonl (one record of hil_stream.py per replay; resumable)
and datasets/dmir/results/hil/stream_cv_board.json (the report).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_cv_check as cv                                 # noqa: E402  (numpy only at import)

stc, dr = cv.stc, cv.dr
HIL = REPO / "datasets" / "dmir" / "results" / "hil"
FILES = HIL / "stream_cv_board"                              # the 60 int8 files, as replayed
RECORDS = HIL / "stream_cv_board.jsonl"
SUMMARY = HIL / "stream_cv_board.json"
SESSIONS = REPO / "datasets" / "dmir" / "data" / "sessions"
BUILD = REPO / "firmware" / "build" / "stream"
GEN = REPO / "firmware" / "build" / "gen"
SEEDS = (100, 101, 102, 103, 104)
IDENTICAL, LSB, SHARE, STUDY, SIZE, TIME = 0.999, 2, 0.001, 0.002, 0.01, 0.03
SPLITS = ("test", "val", "train", "more", "more_baseline")


def nets():
    return [(k, arm, seed, FILES / f"fold{k}_{arm}_seed{seed}_int8_io.tflite")
            for k in range(len(cv.FOLDS)) for arm in cv.ARMS for seed in SEEDS]


def copy_of(tfl: Path) -> Path:
    return SESSIONS / f"sessions_{tfl.name}"


def held_sessions(known, k: int) -> list:
    """The sessions of fold k's held-out drives, as the study scored them."""
    held = cv.FOLDS[k]
    return [f"user{u}LC" for u in held] + [f"user{u}N" for u in held if f"user{u}N" in known]


def prepare():
    """WSL: the 60 files into the results folder (from the study's kept networks, once), a copy that selects
    the session streams, and the header and host reference of each for its fold's held-out sessions."""
    import hil_prepare as hp

    os.environ["LCIR_TASK"] = stc.TASK
    os.environ.pop("LCIR_SPLIT", None)
    FILES.mkdir(parents=True, exist_ok=True)
    table = cv.sessions_table()
    for k in range(len(cv.FOLDS)):
        z, _ = cv.pack(table, held_sessions(table, k))
        for fold, _, _, tfl in nets():
            if fold != k or ((BUILD / copy_of(tfl).stem / "reference.npz").exists() and copy_of(tfl).exists()):
                continue
            if not tfl.exists():
                shutil.copyfile(cv.KEEP / tfl.name, tfl)
            shutil.copyfile(tfl, copy_of(tfl))
            hp.lcir_sessions(copy_of(tfl), sessions=z)


def replayed() -> list:
    return [json.loads(line) for line in RECORDS.read_text().splitlines()] if RECORDS.exists() else []


def record_of(tfl: Path, records: list):
    """The complete top-speed record of a network, if there is one."""
    with np.load(BUILD / copy_of(tfl).stem / "reference.npz") as z:
        n = len(z["seq_lens"])
    return next((r for r in records if r["model"] == copy_of(tfl).name and r["rate_hz"] == 0
                 and len(r["sessions"]) == n), None)


def hil(tfl: Path, *args: str, out: Path = RECORDS) -> bool:
    cmd = [sys.executable, str(REPO / "unas" / "hil_stream.py"), copy_of(tfl).relative_to(REPO).as_posix(), *args]
    r = subprocess.run(cmd, env={**os.environ, "HIL_OUT": str(out)}, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"{tfl.name}: the replay failed ({r.returncode})\n{r.stdout[-800:]}\n{r.stderr[-1500:]}", flush=True)
    return r.returncode == 0


def smoke(out: str):
    """One network, its first session only, to a record file of its own."""
    tfl = nets()[0][3]
    if hil(tfl, "--rate", "0", "--limit", "1", out=Path(out)):
        r = json.loads(Path(out).read_text().splitlines()[-1])
        print(f"{tfl.name}, {r['sessions']}: {r['windows']} windows, {r['agreement']['differing_windows']} differ")


def replay():
    """Windows: every network that has no complete top-speed record yet."""
    for _, _, _, tfl in nets():
        if record_of(tfl, replayed()):
            continue
        if hil(tfl, "--rate", "0", "--save"):
            r = replayed()[-1]
            print(f"{datetime.now():%H:%M} {tfl.name}: {r['windows']} windows, {r['agreement']['differing_windows']} "
                  f"differ from the reference (at most {r['agreement']['max_abs_lsb']} LSB), network "
                  f"{r['board_ms']['network']['mean']} ms per frame", flush=True)


def marks() -> dict:
    """Per session: the frames that a window of the lab's prepared sets ends on."""
    out = {}
    for split in SPLITS:
        path = stc.STREAM / f"{stc.TASK}_sessions{'' if split == 'test' else '_' + split}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        ends = np.cumsum(z["seq_lens"].astype(np.int64))
        for name, m in zip((str(n) for n in z["names"]), np.split(z[f"in_{split}_pickle"].astype(bool), ends[:-1])):
            out[name] = m
    return out


def padding_value(tfl: Path):
    """The value the generated C code pads the second convolution's input with (the pooled tensor's zero point)."""
    src = GEN / copy_of(tfl).stem / "network.c"
    m = re.search(r"conv2d_7_pad_before_value_data\[\]\s*=\s*\{\s*(-?\d+)\s*\}", src.read_text(errors="replace")) if src.exists() else None
    return int(m.group(1)) if m else None


def report():
    study_rows = {(r["fold"], r["arm"], r["seed"]): r for r in map(json.loads, cv.OUT.read_text().splitlines())}
    study = json.loads(cv.SUMMARY.read_text())
    done, lab = replayed(), marks()
    rows = {"board": [], "tflite": []}
    per_net, unscored = [], 0
    for k, arm, seed, tfl in nets():
        rec = record_of(tfl, done)
        if rec is None:
            sys.exit(f"{tfl.name}: no top-speed record yet")
        folder = BUILD / copy_of(tfl).stem
        with np.load(folder / "reference.npz") as z:
            ref = {key: z[key] for key in z.files}
        want, got = ref["ref"], np.load(folder / "board_out_rate0.npz")["outs"]
        if got.shape != want.shape:
            sys.exit(f"{tfl.name}: the board replay holds {len(got)} of {len(want)} windows")
        same = lambda a, b: hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()  # noqa: E731
        if str(ref["tflite"]) != copy_of(tfl).name or not same(copy_of(tfl), tfl) or not same(tfl, cv.KEEP / tfl.name):
            sys.exit(f"{tfl.name}: the reference was made from another file than the study's")
        for source, outs in (("board", got), ("tflite", want)):
            sessions = {s["name"]: s for s in dr.sessions_of(ref, outs)}
            per = {}
            for u in cv.FOLDS[k]:
                s = sessions[f"user{u}LC"]
                d = cv.per_session(s)
                m = lab[s["name"]]
                d["lab_windows"] = int(m.sum())
                d["lab_window_accuracy"] = round(float((s["cls"][m] == s["label"][m]).mean()), 5)
                if source == "board":
                    unscored += int((m & ~s["scored"]).sum())
                if f"user{u}N" in sessions:
                    d["baseline_session"] = cv.per_session(sessions[f"user{u}N"])
                per[f"user{u}"] = d
            rows[source].append({"fold": k, "arm": arm, "seed": seed, "drives": per, "epochs_run": None,
                                 "training_windows": study_rows[(k, arm, seed)]["training_windows"]})
        per_net.append({"fold": k, "arm": arm, "seed": seed, "file": tfl.name, "bytes": tfl.stat().st_size,
                        "sha256": hashlib.sha256(tfl.read_bytes()).hexdigest(),
                        "sessions": len(ref["seq_lens"]), "frames": int(ref["seq_lens"].sum()), "windows": int(len(got)),
                        "agreement": {"identical_windows": round(float((got == want).all(-1).mean()), 6),
                                      "differing_windows": int((got != want).any(-1).sum()),
                                      "max_abs_lsb": int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max()),
                                      "decisions_differing": int((got.argmax(-1) != want.argmax(-1)).sum())},
                        "firmware": rec["firmware"], "network_ms_per_frame": rec["board_ms"]["network"]["mean"],
                        "network_cycles_per_frame": rec["board_cycles"]["network"]["mean"],
                        "preprocessing_cycles_per_frame": rec["board_cycles"]["preprocess"]["mean"],
                        "padding_value_before_the_second_convolution": padding_value(tfl),
                        "replayed_utc": rec["utc"]})

    board, tflite = cv.summarize(rows["board"]), cv.summarize(rows["tflite"])

    def drive_means(source):                                 # (drive, arm) -> mean over the seeds, not rounded
        vals = {}
        for r in rows[source]:
            for u, d in r["drives"].items():
                vals.setdefault((u, r["arm"]), []).append(d["frames_correct"])
        return {key: st.mean(v) for key, v in vals.items()}
    mb, mt = drive_means("board"), drive_means("tflite")
    worst_drive = max(abs(mb[key] - mt[key]) for key in mb)
    fresh, key = "drives_not_looked_at_before", "frames_correct"
    against_study = {arm: round(abs(board[fresh][key][arm] - study[fresh][key][arm]), 5) for arm in cv.ARMS}
    flash = [e["firmware"]["flash_B"] for e in per_net]
    ram = [e["firmware"]["ram_static_B"] for e in per_net]
    ms = [e["network_ms_per_frame"] for e in per_net]
    dev = [abs(v - st.mean(ms)) / st.mean(ms) for v in ms]
    spread = {"flash": (max(flash) - min(flash)) / min(flash), "ram": (max(ram) - min(ram)) / min(ram), "time": max(dev)}
    rule = {
        "agreement": {"least_identical_share": min(e["agreement"]["identical_windows"] for e in per_net),
                      "differing_windows_of_all": [sum(e["agreement"]["differing_windows"] for e in per_net),
                                                   sum(e["windows"] for e in per_net)],
                      "networks_with_a_difference": sum(e["agreement"]["differing_windows"] > 0 for e in per_net),
                      "decisions_differing": sum(e["agreement"]["decisions_differing"] for e in per_net),
                      "max_abs_lsb": max(e["agreement"]["max_abs_lsb"] for e in per_net),
                      "met": bool(all(e["agreement"]["identical_windows"] >= IDENTICAL and e["agreement"]["max_abs_lsb"] <= LSB
                                      for e in per_net))},
        "numbers": {"largest_difference_of_a_drive_board_against_tflite": round(worst_drive, 6),
                    "criterion_of_the_study_on_the_board_outputs": board["criterion"],
                    "arm_means_against_the_study_record": against_study,
                    "met": bool(worst_drive <= SHARE + 1e-9 and board["criterion"]["met"]
                                and max(against_study.values()) <= STUDY + 1e-9)},
        "cost": {"flash_B": sorted(set(flash)), "ram_static_B": sorted(set(ram)),
                 "network_ms_per_frame": [min(ms), round(st.mean(ms), 4), max(ms)],
                 "largest_relative_spread": {k: round(v, 5) for k, v in spread.items()},
                 "networks_more_than_2_percent_from_the_mean": [
                     {"file": e["file"], "ms": e["network_ms_per_frame"],
                      "padding_value": e["padding_value_before_the_second_convolution"]}
                     for e, d in zip(per_net, dev) if d > 0.02],
                 "padding_values": sorted({e["padding_value_before_the_second_convolution"] for e in per_net},
                                          key=lambda v: (v is None, v)),
                 "met": bool(spread["flash"] <= SIZE and spread["ram"] <= SIZE and spread["time"] <= TIME)}}
    rule["met"] = bool(all(rule[k]["met"] for k in ("agreement", "numbers", "cost")))
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "board": done[0]["board"],
           "networks": len(per_net), "decisions": sum(e["windows"] for e in per_net),
           "frames": sum(e["frames"] for e in per_net), "rule": rule,
           "note": "accuracy_on_the_labs_windows is here the accuracy on the frames that the lab's windows of a drive "
                   "end on, from the same replay; the study computed it on the stored windows",
           "frames_of_the_labs_windows_that_are_not_scored_frames": unscored,
           "from_the_board_outputs": board, "from_the_tflite_outputs": tflite,
           "study_record_pc": {k: study[k] for k in ("criterion", fresh, "drives_looked_at_before", "all_drives")},
           "per_network": per_net}
    SUMMARY.write_text(json.dumps(res, indent=1))
    print("from the board's outputs")
    cv.show(board)
    print(json.dumps(rule, indent=1))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "smoke":
        smoke(sys.argv[2])
    else:
        {"prepare": prepare, "replay": replay, "--report": report}.get(
            step, lambda: sys.exit("prepare | smoke <record file> | replay | --report"))()
