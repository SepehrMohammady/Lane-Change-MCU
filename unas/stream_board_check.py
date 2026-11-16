"""The two kinds of LCIR training windows, measured on the board (2026-11-16).

unas/stream_train_check.py compared, as int8 on the PC, the LCIR intention network trained on the lab's
prepared windows (its arm A) and on windows taken from the whole drives of the same 36 training files (its
arm B36), five seeds each. Here those ten int8 files go through the STM32H7B3I-DK: the whole sessions of
the seven official test drivers frame by frame (unas/hil_prepare.py, unas/hil_stream.py; the board limits,
maps and quantizes the raw features itself and decides on every frame from the 50th on), and every measure
of the study is computed again from the board's outputs. The files are kept as replayed in
datasets/dmir/results/hil/stream_board/ (lab_windows_seed*, drive_windows_seed*).

This repeats on the hardware numbers that are known from the PC; it is not a new test of the finding.
Fixed beforehand (local notes, 2026-11-16 17:20) is how closely the board has to return them:
  (1) agreement  for each network the board's int8 outputs equal those of the TFLite interpreter, given the
                 int8 inputs of the board's arithmetic, in at least 99.9% of the windows, and no output
                 differs by more than 2 LSB;
  (2) numbers    for each network every share of the study computed from the board's outputs lies within
                 0.1 point of the same share computed from the TFLite outputs, the alarm runs per hour
                 within 2%; and the two arm means of the share of correct frames in the sessions with lane
                 changes lie within 0.2 points of the study's records (the study quantized the inputs from
                 float64, the board works in float32), for the seven test drivers and for the six that are
                 not in the training set;
  (3) cost       the ten firmware images have the same flash and static RAM size to within 1%, and the mean
                 network time per frame of each lies within 2% of the mean of the ten.
Then, for information, one drive in real time (10 Hz) with one network trained on windows from whole
drives, chosen without the test drivers: the seed with the highest share of correct frames in the
validation drivers' sessions with lane changes in the study's records.

  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/stream_board_check.py prepare     # WSL (TensorFlow)
  .venv\\Scripts\\python unas/stream_board_check.py replay                                # Windows, board connected
  .venv\\Scripts\\python unas/stream_board_check.py realtime
  .venv\\Scripts\\python unas/stream_board_check.py --report

Output: datasets/dmir/results/hil/stream_board.jsonl (one record of hil_stream.py per replay; resumable) and
datasets/dmir/results/hil/stream_board.json (the report).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics as st
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stream_train_check as stc                             # noqa: E402  (numpy only at import)

dr = stc.dr
HIL = REPO / "datasets" / "dmir" / "results" / "hil"
FILES = HIL / "stream_board"                                 # the ten int8 files, as replayed
RECORDS = HIL / "stream_board.jsonl"
SUMMARY = HIL / "stream_board.json"
SESSIONS = REPO / "datasets" / "dmir" / "data" / "sessions"
BUILD = REPO / "firmware" / "build" / "stream"
ARMS = {"A": "lab_windows", "B36": "drive_windows"}          # the study's arm -> the name used here
SEEDS = (100, 101, 102, 103, 104)
REAL_TIME = ("user13LC", 10.0)                               # the drive and rate of the real-time run of 2026-11-11
IDENTICAL, LSB, SHARE, RUNS, STUDY, SIZE, TIME = 0.999, 2, 0.001, 0.02, 0.002, 0.01, 0.02
SHARES = ("frames_correct", "frames_correct_sessions_with_lane_changes", "frames_correct_baseline_sessions",
          "opposite_announced_after_a_lane_change", "announced_2_s_early", "free_frames_in_alarm",
          "with_the_rule_announced_2_s_early", "accuracy_on_the_frames_of_the_lab_test_windows")
RATES = ("alarm_runs_per_hour", "alarm_runs_of_0.5_s_or_more_per_hour", "with_the_rule_alarm_runs_per_hour")


def nets():
    return [(arm, seed, FILES / f"{name}_seed{seed}_int8_io.tflite") for arm, name in ARMS.items() for seed in SEEDS]


def copy_of(tfl: Path) -> Path:
    return SESSIONS / f"sessions_{tfl.name}"


def study_records(path: Path) -> dict:
    return {(r["arm"], r["seed"]): r for r in map(json.loads, path.read_text().splitlines())}


def prepare():
    """WSL: the ten files into the results folder (from the study's kept networks, once), a copy that selects
    the session streams, and the header and host reference of each."""
    import hil_prepare as hp

    os.environ["LCIR_TASK"] = stc.TASK
    os.environ.pop("LCIR_SPLIT", None)
    FILES.mkdir(parents=True, exist_ok=True)
    for arm, seed, tfl in nets():
        if not tfl.exists():
            kept = next(stc.KEEP.glob(f"{arm}_model_*_seed{seed}_int8_io.tflite"))
            shutil.copyfile(kept, tfl)
        shutil.copyfile(tfl, copy_of(tfl))
        hp.lcir_sessions(copy_of(tfl))


def replayed() -> list:
    return [json.loads(line) for line in RECORDS.read_text().splitlines()] if RECORDS.exists() else []


def hil(tfl: Path, *args: str) -> bool:
    cmd = [sys.executable, str(REPO / "unas" / "hil_stream.py"), copy_of(tfl).relative_to(REPO).as_posix(), *args, "--save"]
    r = subprocess.run(cmd, env={**os.environ, "HIL_OUT": str(RECORDS)}, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"{tfl.name}: the replay failed ({r.returncode})\n{r.stdout[-800:]}\n{r.stderr[-1500:]}", flush=True)
    return r.returncode == 0


def replay():
    """Windows: every network that has no complete top-speed record yet."""
    for arm, seed, tfl in nets():
        if any(r["model"] == copy_of(tfl).name and r["rate_hz"] == 0 and len(r["sessions"]) == 14 for r in replayed()):
            continue
        if hil(tfl, "--rate", "0"):
            r = replayed()[-1]
            print(f"{datetime.now():%H:%M} {tfl.name}: {r['windows']} windows, {r['agreement']['differing_windows']} "
                  f"differ from the reference (at most {r['agreement']['max_abs_lsb']} LSB), network "
                  f"{r['board_ms']['network']['mean']} ms per frame", flush=True)


def chosen_seed() -> int:
    """The network for the real-time run: best on the validation drivers in the study's records."""
    rec = study_records(stc.OUT)
    return max(SEEDS, key=lambda s: (rec[("B36", s)]["validation_drivers"]["frame_accuracy"]["lane_change_sessions"], -s))


def realtime():
    seed = chosen_seed()
    tfl = FILES / f"{ARMS['B36']}_seed{seed}_int8_io.tflite"
    name, rate = REAL_TIME
    print(f"real time: {tfl.name} (chosen on the validation drivers), {name} at {rate:g} Hz", flush=True)
    if hil(tfl, "--rate", f"{rate:g}", "--seq", name):
        r = replayed()[-1]
        print(f"{datetime.now():%H:%M} {r['windows']} decisions, {r['agreement']['differing_windows']} differ; "
              f"share of the period {r['compute_share_every_frame']}", flush=True)


def measures(ref, outs: np.ndarray, rule: tuple, keep) -> dict:
    sessions, acc, after = stc.drive_measures(ref, outs, ref["window_last_frame"], float(ref["out_scale"]), keep)
    raw, sel = dr.score(sessions, dr.RAW), dr.score(sessions, rule)
    return {"frames_correct": acc["all"], "frames_correct_sessions_with_lane_changes": acc["lane_change_sessions"],
            "frames_correct_baseline_sessions": acc["baseline_sessions"],
            "opposite_announced_after_a_lane_change": after["the opposite direction"],
            "announced_2_s_early": raw["timely_share"], "lane_changes": raw["lane_changes"],
            "alarm_runs_per_hour": raw["false_alarms_per_hour"],
            "alarm_runs_of_0.5_s_or_more_per_hour": raw["false_alarms_of_0.5_s_or_more_per_hour"],
            "free_frames_in_alarm": raw["free_frames_in_alarm"], "free_driving_hours": raw["free_driving_hours"],
            "with_the_rule_announced_2_s_early": sel["timely_share"],
            "with_the_rule_alarm_runs_per_hour": sel["false_alarms_per_hour"]}


def report():
    first, six = study_records(stc.OUT), study_records(stc.SIX)
    done = replayed()
    held_all = np.load(stc.STREAM / f"{stc.TASK}_sessions.npz")["in_test_pickle"]
    groups = {"seven_test_drivers": lambda name: True,
              "six_test_drivers_not_in_the_training_set": lambda name: not name.startswith(stc.COPIED)}
    per_net = []
    for arm, seed, tfl in nets():
        rec = next((r for r in done if r["model"] == copy_of(tfl).name and r["rate_hz"] == 0 and len(r["sessions"]) == 14), None)
        if rec is None:
            sys.exit(f"{tfl.name}: no top-speed record yet")
        folder = BUILD / copy_of(tfl).stem
        with np.load(folder / "reference.npz") as z:
            ref = {k: z[k] for k in z.files}
        want, got = ref["ref"], np.load(folder / "board_out_rate0.npz")["outs"]
        if got.shape != want.shape:
            sys.exit(f"{tfl.name}: the board replay holds {len(got)} of {len(want)} windows")
        if str(ref["tflite"]) != copy_of(tfl).name or hashlib.sha256(copy_of(tfl).read_bytes()).digest() != hashlib.sha256(tfl.read_bytes()).digest():
            sys.exit(f"{tfl.name}: the reference was made from another file")
        sel = first[(arm, seed)]["selected_rule"]
        rule = (sel["theta"], sel["hold_off_frames"], sel["persistence_frames"])
        last = ref["window_last_frame"].astype(np.int64)
        lens = ref["seq_lens"].astype(np.int64)
        sess = np.searchsorted(np.cumsum(lens), last, side="right")
        names = [str(n) for n in ref["names"]]
        label = np.where(ref["ttlc"][last] <= 4.0, ref["direction"][last], 0)
        held = held_all[last]
        entry = {"arm": arm, "trained_on": ARMS[arm], "seed": seed, "file": tfl.name, "bytes": tfl.stat().st_size,
                 "sha256": hashlib.sha256(tfl.read_bytes()).hexdigest(), "rule_selected_in_the_study": sel,
                 "windows": int(len(got)),
                 "agreement": {"identical_windows": round(float((got == want).all(-1).mean()), 6),
                               "differing_windows": int((got != want).any(-1).sum()),
                               "max_abs_lsb": int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max()),
                               "decisions_differing": int((got.argmax(-1) != want.argmax(-1)).sum())},
                 "firmware": rec["firmware"], "network_ms_per_frame": rec["board_ms"]["network"]["mean"],
                 "network_cycles_per_frame": rec["board_cycles"]["network"]["mean"],
                 "preprocessing_cycles_per_frame": rec["board_cycles"]["preprocess"]["mean"],
                 "host_round_trip_ms": rec["host_round_trip_ms"]["mean"], "replayed_utc": rec["utc"]}
        for key, keep in groups.items():
            rows = held & np.array([keep(n) for n in names])[sess]
            entry[key] = {}
            for source, outs in (("board", got), ("tflite", want)):
                m = measures(ref, outs, rule, keep)
                m["accuracy_on_the_frames_of_the_lab_test_windows"] = round(float((outs.argmax(-1)[rows] == label[rows]).mean()), 5)
                m["frames_of_the_lab_test_windows"] = int(rows.sum())
                entry[key][source] = m
        t7, t6 = first[(arm, seed)]["test_drivers"], six[(arm, seed)]["test_drivers"]
        entry["study_record_pc"] = {
            "seven_test_drivers": {"frames_correct_sessions_with_lane_changes": t7["frame_accuracy"]["lane_change_sessions"],
                                   "frames_correct": t7["frame_accuracy"]["all"],
                                   "lab_test_set_accuracy": first[(arm, seed)]["lab_test_pickle_accuracy"]["int8"]},
            "six_test_drivers_not_in_the_training_set": {
                "frames_correct_sessions_with_lane_changes": t6["frame_accuracy"]["lane_change_sessions"],
                "frames_correct": t6["frame_accuracy"]["all"],
                "lab_test_set_accuracy": six[(arm, seed)]["lab_test_pickle_accuracy_int8"]}}
        per_net.append(entry)

    def mean_std(vals):
        return {"mean": round(st.mean(vals), 5), "std": round(st.stdev(vals), 5), "values": vals}

    keys = SHARES + RATES
    arms = {}
    for arm, name in ARMS.items():
        rs = [e for e in per_net if e["arm"] == arm]
        arms[name] = {"study_arm": arm, "seeds": [e["seed"] for e in rs],
                      "network_ms_per_frame": mean_std([e["network_ms_per_frame"] for e in rs]),
                      "firmware_flash_B": sorted({e["firmware"]["flash_B"] for e in rs}),
                      "firmware_ram_static_B": sorted({e["firmware"]["ram_static_B"] for e in rs})}
        for g in groups:
            arms[name][g] = {"board": {k: mean_std([e[g]["board"][k] for e in rs]) for k in keys},
                             "study_record_pc": {k: mean_std([e["study_record_pc"][g][k] for e in rs])
                                                 for k in ("frames_correct_sessions_with_lane_changes", "frames_correct",
                                                           "lab_test_set_accuracy")}}
    lab, drv = (arms[ARMS[a]] for a in ("A", "B36"))
    gain = {g: {k: round(drv[g]["board"][k]["mean"] - lab[g]["board"][k]["mean"], 5) for k in keys} for g in groups}

    # the rule
    worst_share = max(abs(e[g]["board"][k] - e[g]["tflite"][k]) for e in per_net for g in groups for k in SHARES)
    worst_rate = max(abs(e[g]["board"][k] - e[g]["tflite"][k]) / max(e[g]["tflite"][k], 1e-9) for e in per_net for g in groups for k in RATES)
    key = "frames_correct_sessions_with_lane_changes"
    worst_study = max(abs(a[g]["board"][key]["mean"] - a[g]["study_record_pc"][key]["mean"]) for a in arms.values() for g in groups)
    flash = [e["firmware"]["flash_B"] for e in per_net]
    ram = [e["firmware"]["ram_static_B"] for e in per_net]
    ms = [e["network_ms_per_frame"] for e in per_net]
    spread = {"flash": (max(flash) - min(flash)) / min(flash), "ram": (max(ram) - min(ram)) / min(ram),
              "time": max(abs(v - st.mean(ms)) for v in ms) / st.mean(ms)}
    rule = {
        "agreement": {"least_identical_share": min(e["agreement"]["identical_windows"] for e in per_net),
                      "differing_windows_of_all": [sum(e["agreement"]["differing_windows"] for e in per_net),
                                                   sum(e["windows"] for e in per_net)],
                      "decisions_differing": sum(e["agreement"]["decisions_differing"] for e in per_net),
                      "max_abs_lsb": max(e["agreement"]["max_abs_lsb"] for e in per_net),
                      "met": bool(all(e["agreement"]["identical_windows"] >= IDENTICAL and e["agreement"]["max_abs_lsb"] <= LSB
                                      for e in per_net))},
        "numbers": {"largest_difference_of_a_share_board_against_tflite": round(worst_share, 5),
                    "largest_relative_difference_of_alarm_runs_per_hour": round(worst_rate, 5),
                    "largest_difference_of_an_arm_mean_against_the_study_record": round(worst_study, 5),
                    "met": bool(worst_share <= SHARE + 1e-9 and worst_rate <= RUNS + 1e-9 and worst_study <= STUDY + 1e-9)},
        "cost": {"flash_B": sorted(set(flash)), "ram_static_B": sorted(set(ram)),
                 "network_ms_per_frame": [min(ms), round(st.mean(ms), 4), max(ms)],
                 "largest_relative_spread": {k: round(v, 5) for k, v in spread.items()},
                 "met": bool(spread["flash"] <= SIZE and spread["ram"] <= SIZE and spread["time"] <= TIME)}}
    rule["met"] = bool(all(rule[k]["met"] for k in ("agreement", "numbers", "cost")))
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "board": done[0]["board"],
           "sessions": len(names), "frames": int(lens.sum()), "decisions_per_network": int(len(last)),
           "rule": rule, "trained_on": arms, "drive_windows_minus_lab_windows_board": gain, "networks": per_net}
    rt = [r for r in done if r["rate_hz"] > 0]
    if rt:
        r = rt[-1]
        res["real_time"] = {"file": r["model"][len("sessions_"):], "chosen_on": "validation drivers, study records",
                            "session": r["sessions"], "rate_hz": r["rate_hz"], "frames": r["frames"],
                            "decisions": r["windows"], "agreement": r["agreement"],
                            "share_of_the_frame_period_computing": r["compute_share_every_frame"],
                            "frame_lateness_ms": r["frame_lateness_ms"], "network_ms_per_frame": r["board_ms"]["network"]["mean"]}
    SUMMARY.write_text(json.dumps(res, indent=1))
    for g in groups:
        print(g)
        for k in keys:
            a, b = lab[g]["board"][k], drv[g]["board"][k]
            print(f"  {k:50s} lab {a['mean']:9.4f} +- {a['std']:.4f}   drives {b['mean']:9.4f} +- {b['std']:.4f}")
        for name, a in arms.items():
            s = a[g]["study_record_pc"]
            print(f"  study records, {name}: frames correct, sessions with lane changes {s[key]['mean']:.4f}; "
                  f"all {s['frames_correct']['mean']:.4f}; lab test set {s['lab_test_set_accuracy']['mean']:.4f}")
    print(json.dumps(rule, indent=1))
    if rt:
        print(json.dumps(res["real_time"]))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"prepare": prepare, "replay": replay, "realtime": realtime, "--report": report}.get(
        step, lambda: sys.exit("prepare | replay | realtime | --report"))()
