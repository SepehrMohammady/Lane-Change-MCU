"""LCIR on whole drives: what the deployed networks do on every frame of the test drivers' sessions.

Scores the outputs of unas/hil_stream.py on the session streams of datasets/dmir/session_stream.py (the
board's outputs, or the host reference when a network has not been replayed yet). The measures were
fixed before any network output on whole sessions existed (local notes, 2026-11-11):

  intention   frame accuracy and recall per class over the scored frames of the LC sessions;
              false alarms: maximal runs of consecutive scored frames predicted as a lane change that hold
              no frame labelled as before a lane change, for runs of at least 1 and at least 5 frames, per
              hour of free driving, LC and N sessions apart;
              detection: for each lane change with at least 90 frames before it, the length of the
              uninterrupted run of correct-direction predictions that ends on the last frame before the
              jump (0 = missed);
  time to LC  RMSE over the frames labelled 0.0-4.0 s of the head's direction, and over those where the
              raw-input intention network predicts that direction.
Added after the labels were checked against the pickles (the lab's test sets hold 382 of the 527 lane
changes of these sessions): every measure also for the frames or lane changes inside and outside the
test pickles.
Added after the first outputs were seen (post hoc, marked as such in the output): the free-driving
frames split by whether the 50-frame window still holds a completed lane change. The lab's free-driving
windows never do (a window with an ignored frame is dropped, so they start at least 1.5 s after a lane
change and end at least 6.4 s after it), while a stream delivers such windows after every lane change.

  .venv\\Scripts\\python datasets/dmir/session_metrics.py
Output: datasets/dmir/results/hil/session_metrics.json (aggregates only).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STREAM = ROOT / "firmware" / "build" / "stream"
NETS = {  # name -> (deploy stem, task)
    "intention, searched on raw inputs": ("lcir_v4k3sat_classification", "classification"),
    "intention, searched on scaled inputs": ("lcir_v4k3mm_classification", "classification"),
    "LCR, searched on raw inputs": ("lcir_v4k3sat_regression_lcr", "regression_lcr"),
    "LCR, searched on scaled inputs": ("lcir_v4k3mm_regression_lcr", "regression_lcr"),
    "LCL, searched on raw inputs": ("lcir_v4k3sat_regression_lcl", "regression_lcl"),
    "LCL, searched on scaled inputs": ("lcir_v4k3mm_regression_lcl", "regression_lcl"),
}
GATE = "lcir_v4k3sat_classification"
FPS, W = 10, 50


def load(stem: str):
    folder = STREAM / f"sessions_{stem}_int8_io"
    ref = np.load(folder / "reference.npz")
    board = folder / "board_out_rate0.npz"
    if board.exists():
        outs, source = np.load(board)["outs"], "board"
        if len(outs) != len(ref["ref"]):
            raise RuntimeError(f"{stem}: the board replay holds {len(outs)} of {len(ref['ref'])} windows")
    else:
        outs, source = ref["ref"], "host reference (not replayed on the board yet)"
    deq = (outs.astype(np.float32) - int(ref["out_zp"])) * float(ref["out_scale"])
    return ref, deq, source, int((outs != ref["ref"]).any(-1).sum())


def frames_of(ref, task):
    """Per window: session index, frame in the session, label (time to LC, direction), in the test pickle."""
    lens = ref["seq_lens"]
    last = ref["window_last_frame"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    sess = np.searchsorted(np.cumsum(lens), last, side="right")
    held = np.load(HERE / "data" / "stream" / f"{task}_sessions.npz")["in_test_pickle"]
    return sess, last - starts[sess], ref["ttlc"][last], ref["direction"][last], held[last]


def runs(mask: np.ndarray):
    """(start, end) of every maximal run of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def intention(ref, deq):
    names = [str(n) for n in ref["names"]]
    sess, frame, ttlc, direction, held = frames_of(ref, "classification")
    pred = deq.argmax(-1)
    scored = ~np.isnan(ttlc)
    label = np.where(ttlc <= 4.0, direction, 0)
    lc_sess = np.array([n.endswith("LC") for n in names])[sess]
    out = {}
    m = scored & lc_sess
    out["lc_sessions"] = {
        "scored_frames": int(m.sum()), "frame_accuracy": round(float((pred[m] == label[m]).mean()), 4),
        "recall": {c: round(float((pred[m & (label == k)] == k).mean()), 4)
                   for k, c in ((0, "free driving"), (1, "before a right lane change"), (2, "before a left lane change"))},
        "frames_per_class": {c: int((m & (label == k)).sum()) for k, c in ((0, "free"), (1, "right"), (2, "left"))},
        "accuracy_on_the_frames_of_the_test_pickle": round(float((pred[held] == label[held]).mean()), 5),
        "frames_of_the_test_pickle": int(held.sum())}
    # false alarms, per session so that runs do not join across sessions
    for key, pick in (("lc_sessions", lambda n: n.endswith("LC")), ("n_sessions", lambda n: n.endswith("N"))):
        alarms1 = alarms5 = free = 0
        longest = 0
        for s, n in enumerate(names):
            if not pick(n):
                continue
            w = sess == s
            alarm = scored[w] & (pred[w] != 0)
            before_lc = scored[w] & (label[w] != 0)
            free += int((scored[w] & (label[w] == 0)).sum())
            for a, b in runs(alarm):
                if before_lc[a:b].any():
                    continue
                alarms1 += 1
                alarms5 += int(b - a >= 5)
                longest = max(longest, int(b - a))
        hours = free / FPS / 3600
        out.setdefault(key, {})["false_alarms"] = {
            "free_driving_minutes": round(free / FPS / 60, 1), "runs_of_1_frame_or_more": alarms1,
            "runs_of_5_frames_or_more": alarms5, "per_hour_1_frame_or_more": round(alarms1 / hours, 1),
            "per_hour_5_frames_or_more": round(alarms5 / hours, 1), "longest_run_s": round(longest / FPS, 1),
            "free_frames_called_lane_change": round(float((pred[scored & (label == 0) & np.array([pick(names[k]) for k in sess])] != 0).mean()), 4)}
    # detection lead per lane change
    events = json.loads(str(ref["events"]))
    held_reg = (np.load(HERE / "data" / "stream" / "regression_lcr_sessions.npz")["in_test_pickle"] |
                np.load(HERE / "data" / "stream" / "regression_lcl_sessions.npz")["in_test_pickle"])[ref["window_last_frame"]]
    lead, in_pickle = [], []
    clear = np.ones(len(pred), bool)                        # the window holds no completed lane change
    for s, n in enumerate(names):
        w = np.flatnonzero(sess == s)
        by_frame = {int(f): k for f, k in zip(frame[w], w)}
        for i, _ in events[n]:
            clear[w[(frame[w] >= i) & (frame[w] < i + 64)]] = False
        for i, d in events[n]:
            if i < 90:
                continue
            k, run = i - 1, 0
            while k in by_frame and pred[by_frame[k]] == d:
                run += 1
                k -= 1
            lead.append(run / FPS)
            in_pickle.append(bool(held_reg[[by_frame[f] for f in range(i - 41, i) if f in by_frame]].mean() >= 0.5))
    lead, in_pickle = np.array(lead), np.array(in_pickle)

    def lead_summary(v):
        return {"lane_changes": int(len(v)), "median_s": round(float(np.median(v)), 2),
                "p10_s": round(float(np.percentile(v, 10)), 2), "mean_s": round(float(v.mean()), 2),
                "at_least_1s": round(float((v >= 1).mean()), 3), "at_least_2s": round(float((v >= 2).mean()), 3),
                "at_least_3s": round(float((v >= 3).mean()), 3), "missed": round(float((v == 0).mean()), 3)}
    out["detection_lead"] = {"all": lead_summary(lead), "lane_changes_in_the_test_pickles": lead_summary(lead[in_pickle]),
                             "lane_changes_left_out_of_the_pickles": lead_summary(lead[~in_pickle])}
    # post hoc: free driving in the LC sessions, by what the window holds
    free = scored & (label == 0) & lc_sess
    post = {}
    for key, m in (("window_without_a_completed_lane_change", free & clear),
                   ("window_still_holding_a_completed_lane_change", free & ~clear)):
        alarms5 = 0
        for s, n in enumerate(names):
            if not n.endswith("LC"):
                continue
            w = sess == s
            before_lc = scored[w] & (label[w] != 0)
            for a, b in runs(m[w] & (pred[w] != 0)):
                alarms5 += int(b - a >= 5 and not before_lc[a:b].any())
        post[key] = {"free_frames": int(m.sum()), "minutes": round(float(m.sum()) / FPS / 60, 1),
                     "called_lane_change": round(float((pred[m] != 0).mean()), 4),
                     "false_alarm_runs_of_5_frames_or_more_per_hour": round(alarms5 / (m.sum() / FPS / 3600), 1)}
    keep = scored & lc_sess & (clear | (label != 0))
    post["frame_accuracy_without_such_windows"] = round(float((pred[keep] == label[keep]).mean()), 4)
    post["frames_without_such_windows"] = int(keep.sum())
    out["post_hoc_by_window_content_lc_sessions"] = post
    return out, pred


def ttlc_head(ref, deq, task, gate_pred):
    sess, frame, ttlc, direction, held = frames_of(ref, task)
    want = 1 if task == "regression_lcr" else 2
    sel = ~np.isnan(ttlc) & (ttlc <= 4.0) & (direction == want)
    err = deq[:, 0] - ttlc

    def rmse(m):
        return {"frames": int(m.sum()), "rmse_s": round(float(np.sqrt((err[m] ** 2).mean())), 4),
                "mae_s": round(float(np.abs(err[m]).mean()), 4)}
    gated = sel & (gate_pred == want)
    return {"all_frames_before_its_lane_changes": rmse(sel), "frames_of_the_test_pickle": rmse(sel & held),
            "frames_left_out_of_the_pickle": rmse(sel & ~held),
            "where_the_intention_network_predicts_this_direction": {**rmse(gated), "share_of_frames": round(float(gated.sum() / sel.sum()), 3)}}


def main():
    res = {"networks": {}, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    gate_ref, gate_deq, _, _ = load(GATE)
    gate_pred = gate_deq.argmax(-1)
    for name, (stem, task) in NETS.items():
        ref, deq, source, differing = load(stem)
        entry = {"file": f"{stem}_int8_io.tflite", "outputs_from": source, "windows": int(len(deq)),
                 "windows_differing_from_the_host_reference": differing}
        if task == "classification":
            entry.update(intention(ref, deq)[0])
        else:
            entry.update(ttlc_head(ref, deq, task, gate_pred))
        res["networks"][name] = entry
    lens = gate_ref["seq_lens"]
    names = [str(n) for n in gate_ref["names"]]
    res["sessions"] = {"count": len(lens), "minutes": round(float(lens.sum()) / 600, 1),
                       "lc_minutes": round(float(sum(n for n, s in zip(lens, names) if s.endswith("LC"))) / 600, 1),
                       "n_minutes": round(float(sum(n for n, s in zip(lens, names) if s.endswith("N"))) / 600, 1)}
    out = HERE / "results" / "hil" / "session_metrics.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
