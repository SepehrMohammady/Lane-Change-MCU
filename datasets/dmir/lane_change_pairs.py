"""LCIR: lane changes come in pairs, and what that does to the prepared windows and to a deployed network.

Looked at after the decision-rule study (datasets/dmir/decision_rule.py, 2026-11-12), to explain why a
hold-off after a lane change removes true detections. Not planned beforehand: everything here is a
description, nothing was selected with it.

  dataset (test and validation drivers, sessions with lane changes; datasets/dmir/session_stream.py):
    - the gap between a lane change and the one before it, and the directions: on the two-lane road
      every lane change is the opposite of the previous one, and the return after an overtake follows
      within seconds;
    - the windows of the classification pickles that hold a completed lane change (a lane-line jump
      inside their 50 frames), by class: none of the free-driving windows do, so in the prepared data
      such a window always means that a lane change of the opposite direction is coming;
    - the same situation on whole drives: free-driving frames whose window holds a completed lane change.
  network (validation drivers only, PC reference of the intention network searched on raw inputs):
    - its output on the frames labelled free driving 1.5 to 6.3 s after a lane change (the window still
      holds that lane change): free driving, the direction of the completed lane change, or the opposite
      one; and how often the opposite lane change then follows within 10 s of the completed one.

  .venv\\Scripts\\python datasets/dmir/lane_change_pairs.py
Output: datasets/dmir/results/hil/lane_change_pairs.json (aggregates only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STREAM = HERE / "data" / "stream"
REF = ROOT / "firmware" / "build" / "stream" / "sessions_lcir_v4k3sat_classification_int8_io_val" / "reference.npz"
W, FAR = 50, 10 ** 6


def dataset(split: str) -> dict:
    tag = "" if split == "test" else f"_{split}"
    z = np.load(STREAM / f"classification_sessions{tag}.npz")
    names = [str(n) for n in z["names"]]
    lens = z["seq_lens"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    events = json.loads(str(z["events"]))
    held, ttlc, direction = z[f"in_{split}_pickle"], z["ttlc"], z["direction"]
    gaps, dirs, same = [], [], 0
    n_cls, with_jump, opposite = np.zeros(3, int), np.zeros(3, int), np.zeros(3, int)
    free = free_jump = 0
    for s, n in enumerate(names):
        if not n.endswith("LC"):
            continue
        a, ln = int(starts[s]), int(lens[s])
        ev = events[n]
        for j, (i, d) in enumerate(ev):
            if j:
                gaps.append(i - ev[j - 1][0])
                dirs.append(d)
                same += int(d == ev[j - 1][1])
        last_jump, last_dir = np.full(ln, -FAR), np.zeros(ln, int)
        for i, d in ev:
            last_jump[i:], last_dir[i:] = i, d
        t = ttlc[a:a + ln]
        cls = np.where(np.isnan(t), -1, np.where(t <= 4.0, direction[a:a + ln], 0))
        frame = np.arange(ln)
        has = frame - last_jump <= W - 1                    # the window ending on this frame holds a jump
        h = held[a:a + ln]
        for c in range(3):
            m = h & (cls == c)
            n_cls[c] += int(m.sum())
            with_jump[c] += int((m & has).sum())
            opposite[c] += int((m & has & (last_dir == 3 - c)).sum()) if c else 0
        m = (frame >= W - 1) & (cls == 0)
        free += int(m.sum())
        free_jump += int((m & has).sum())
    gaps, dirs = np.array(gaps), np.array(dirs)
    quick = gaps < 84
    klass = ("free driving", "before a right lane change", "before a left lane change")
    return {
        "lane_changes_with_a_previous_one": int(len(gaps)),
        "same_direction_as_the_previous_one": same,
        "gap_to_the_previous_one_s": {"p10": round(float(np.percentile(gaps, 10)) / 10, 1),
                                      "p25": round(float(np.percentile(gaps, 25)) / 10, 1),
                                      "median": round(float(np.median(gaps)) / 10, 1),
                                      "p75": round(float(np.percentile(gaps, 75)) / 10, 1),
                                      "p90": round(float(np.percentile(gaps, 90)) / 10, 1)},
        "share_within": {f"{lim / 10:.1f}_s": round(float((gaps < lim).mean()), 3) for lim in (35, 50, 70, 84, 150)},
        "to_the_right_among_those_within_8.4_s": round(float((dirs[quick] == 1).mean()), 3),
        "to_the_right_among_the_others": round(float((dirs[~quick] == 1).mean()), 3),
        f"{split}_pickle_windows_by_class": {
            c: {"windows": int(n_cls[k]), "holding_a_completed_lane_change": int(with_jump[k]),
                "share": round(float(with_jump[k] / n_cls[k]), 3),
                **({"of_the_opposite_direction": int(opposite[k])} if k else {})} for k, c in enumerate(klass)},
        "whole_drives_free_driving_frames": {"frames": free, "window_holds_a_completed_lane_change": free_jump,
                                             "share": round(free_jump / free, 3)},
    }


def after_a_lane_change() -> dict:
    ref = np.load(REF)
    lens = ref["seq_lens"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    last, pred_w = ref["window_last_frame"], ref["ref"].argmax(-1)
    events = json.loads(str(ref["events"]))
    count = {"free driving": 0, "the completed direction": 0, "the opposite direction": 0}
    by_offset = np.zeros((64, 3), int)
    announced = followed = lane_changes = 0
    for s, n in enumerate(str(v) for v in ref["names"]):
        if not n.endswith("LC"):
            continue
        a, ln = int(starts[s]), int(lens[s])
        pred = np.zeros(ln, np.int8)
        m = (last >= a) & (last < a + ln)
        pred[last[m] - a] = pred_w[m]
        t = ref["ttlc"][a:a + ln]
        ev = events[n]
        for j, (i, d) in enumerate(ev):
            lane_changes += 1
            nxt = ev[j + 1] if j + 1 < len(ev) else (FAR, 0)
            said = False
            for f in range(i + 15, min(i + 64, ln)):
                if np.isnan(t[f]) or t[f] <= 4.0:           # frames labelled free driving only
                    continue
                k = 0 if pred[f] == 0 else (1 if pred[f] == d else 2)
                count[list(count)[k]] += 1
                by_offset[f - i, k] += 1
                said |= k == 2
            announced += int(said)
            followed += int(said and nxt[1] == 3 - d and nxt[0] - i <= 100)
    total = sum(count.values())
    return {
        "note": "post hoc; validation drivers; frames labelled free driving 1.5 to 6.3 s after a lane change",
        "network": "intention, searched on raw inputs (PC reference)", "lane_changes": lane_changes,
        "frames": total, "output_share": {k: round(v / total, 3) for k, v in count.items()},
        "output_share_by_time_after_the_lane_change": {
            f"{o / 10:.1f}_s": {k: round(float(by_offset[o, c] / max(by_offset[o].sum(), 1)), 2)
                                for c, k in enumerate(count)} for o in range(15, 64, 8)},
        "lane_changes_after_which_the_opposite_one_is_announced": announced,
        "of_those_the_opposite_one_follows_within_10_s": followed,
        "share": round(followed / announced, 3)}


def main():
    res = {"test_drivers": dataset("test"), "validation_drivers": dataset("val"),
           "after_a_lane_change": after_a_lane_change()}
    out = HERE / "results" / "hil" / "lane_change_pairs.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
