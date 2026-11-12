"""LCIR on whole drives: a decision rule that turns the intention network's frame outputs into an alarm.

On whole drives the frame-by-frame output of the intention network raises many false alarms in lane
keeping and stays on for seconds after a lane change (datasets/dmir/session_metrics.py). A deployed
system would not show the raw output; it would apply a rule. The rule is chosen here on the sessions of
the seven official validation drivers and then scored once on the seven test drivers. Family, selection
and the criterion on the test drivers were fixed before any validation session was built (local notes,
2026-11-12 11:00):

  threshold    a frame is a candidate if the predicted class is a lane change and its softmax probability
               is at least theta, theta in {0 (argmax), 0.6, 0.8, 0.9};
  hold-off     no alarm for H frames from the ego's own lane-line jump on (the distance to the right lane
               line changes by more than half a lane width between two frames), H in {0, 15, 30, 50, 64};
  persistence  the alarm is on at a frame if the last K frames are candidates of the same direction,
               K in {1, 3, 5, 10}.
  80 rules; (0, 0, 1) is the raw output. The hold-off switches the alarm off and does not restart the
  count of candidates.

  measures     timely share: lane changes (with at least 90 frames before them) whose alarm of the
               correct direction is on over the whole last 2 s before the jump (frames i-20..i-1);
               false alarms per hour of free driving: maximal runs of alarm frames, cut at the ignored
               frames, that hold no frame labelled as before a lane change, of any length, sessions with
               and without lane changes pooled; also the median lead and the share of free-driving frames
               in alarm.
  selection    on the validation drivers: among the rules whose timely share is at most 5 points below
               the raw output's, the one with the fewest false alarms; ties: smaller H, smaller K,
               smaller theta.
  criterion    on the test drivers the selected rule is adopted if it at least halves the raw output's
               false alarms per hour and its timely share stays within 5 points of the raw output's.

Everything the rule needs is available on the board: the int8 outputs of the network and two of the 31
input features (distance to the right lane line, lane width). The probability test is written on the int8
outputs so that a microcontroller evaluates it identically: with d1, d2 the differences between the largest
output and the two others, p >= theta is exp(-s*d1) + exp(-s*d2) <= 1/theta - 1, the exponentials taken
from a float32 table of 256 entries (s = the output scale).

Outputs scored: the PC reference (TFLite) for the validation drivers, the board's own outputs for the test
drivers (unas/hil_stream.py; both intention networks are bit-identical to the PC on all 104,288 test
windows). Inputs: firmware/build/stream/sessions_<stem>_int8_io[_val]/ of unas/hil_prepare.py (LCIR_SPLIT=val
for the validation drivers).

  .venv\\Scripts\\python datasets/dmir/decision_rule.py
Output: datasets/dmir/results/hil/decision_rule.json (aggregates only).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src.features import FEATURES_BY_TASK                   # noqa: E402

STREAM = ROOT / "firmware" / "build" / "stream"
NETS = {"intention, searched on raw inputs": "lcir_v4k3sat_classification",
        "intention, searched on scaled inputs": "lcir_v4k3mm_classification"}
THETAS, HOLDS, KS = (0.0, 0.6, 0.8, 0.9), (0, 15, 30, 50, 64), (1, 3, 5, 10)
RAW = (0.0, 0, 1)
FPS, MARGIN, HALVE = 10, 0.05, 0.5
RIGHT = FEATURES_BY_TASK["classification"].index("rightLaneLine")
WIDTH = FEATURES_BY_TASK["classification"].index("egoLaneWidth")


def load(stem: str, split: str):
    """Per session: predicted class, the probability test's left side, labels, jumps, lane changes."""
    folder = STREAM / f"sessions_{stem}_int8_io{'' if split == 'test' else '_' + split}"
    ref = np.load(folder / "reference.npz")
    if split == "test":
        outs, source = np.load(folder / "board_out_rate0.npz")["outs"], "board"
        if outs.shape != ref["ref"].shape:
            raise RuntimeError(f"{stem}: the board replay holds {len(outs)} of {len(ref['ref'])} windows")
    else:
        outs, source = ref["ref"], "PC reference (TFLite)"
    expo = np.exp(-np.float32(ref["out_scale"]) * np.arange(256, dtype=np.float32)).astype(np.float32)
    lens = ref["seq_lens"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    last = ref["window_last_frame"].astype(np.int64)
    events = json.loads(str(ref["events"]))
    feats, ttlc_all, dir_all = ref["feats"], ref["ttlc"], ref["direction"]
    sessions = []
    for s, name in enumerate(str(n) for n in ref["names"]):
        a, n = int(starts[s]), int(lens[s])
        m = (last >= a) & (last < a + n)
        f = last[m] - a                                     # the frames that have an output
        q = outs[m].astype(np.int16)
        k = q.argmax(-1)                                    # first of equal outputs, as on the board
        others = np.array([[1, 2], [0, 2], [0, 1]])[k]
        d = q[np.arange(len(q)), k][:, None] - np.take_along_axis(q, others, 1)
        cls, e, has_out = np.zeros(n, np.int8), np.full(n, np.inf, np.float32), np.zeros(n, bool)
        cls[f], e[f], has_out[f] = k, expo[d[:, 0]] + expo[d[:, 1]], True
        right, width = feats[a:a + n, RIGHT], feats[a:a + n, WIDTH]
        jump = np.zeros(n, bool)
        jump[1:] = np.abs(right[1:] - right[:-1]) > width[1:] * np.float32(0.5)
        if [int(i) for i in np.flatnonzero(jump)] != [i for i, _ in events[name]]:
            raise RuntimeError(f"{name}: the lane-line jumps found in the frames differ from the labelled ones")
        ttlc = ttlc_all[a:a + n]
        scored = has_out & ~np.isnan(ttlc)
        label = np.where(ttlc <= 4.0, dir_all[a:a + n], 0)   # NaN compares False: ignored frames are 0 here
        sessions.append({"name": name, "cls": cls, "e": e, "has_out": has_out, "jump": jump, "scored": scored,
                         "label": label, "events": [(i, d_) for i, d_ in events[name] if i >= 90]})
    return sessions, source, int((outs != ref["ref"]).any(-1).sum())


def alarm_of(s: dict, theta: float, hold: int, k: int) -> np.ndarray:
    """Per frame: 0 = no alarm, 1 = right lane change announced, 2 = left."""
    ok = s["has_out"] & (s["cls"] != 0)
    if theta > 0:
        ok &= s["e"] <= np.float32(1.0 / theta - 1.0)
    cand = np.where(ok, s["cls"], 0)
    n = len(cand)
    change = np.concatenate([[True], cand[1:] != cand[:-1]])
    run = np.arange(n) - np.maximum.accumulate(np.where(change, np.arange(n), 0)) + 1
    alarm = np.where((cand != 0) & (run >= k), cand, 0)
    for i in np.flatnonzero(s["jump"]):
        alarm[i:i + hold] = 0
    return alarm


def runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def score(sessions: list, rule: tuple, pick=lambda name: True) -> dict:
    theta, hold, k = rule
    leads, fa, fa5, free, free_alarm = [], 0, 0, 0, 0
    for s in sessions:
        if not pick(s["name"]):
            continue
        alarm = alarm_of(s, theta, hold, k)
        for i, d in s["events"]:
            j = i - 1
            while j >= 0 and alarm[j] == d:
                j -= 1
            leads.append(i - 1 - j)
        on = s["scored"] & (alarm != 0)
        before_lc = s["scored"] & (s["label"] != 0)
        is_free = s["scored"] & (s["label"] == 0)
        free += int(is_free.sum())
        free_alarm += int((is_free & (alarm != 0)).sum())
        for a, b in runs(on):
            if not before_lc[a:b].any():
                fa += 1
                fa5 += int(b - a >= 5)
    leads = np.array(leads)
    hours = free / FPS / 3600
    out = {"lane_changes": int(len(leads)), "false_alarms": fa, "free_driving_hours": round(hours, 3),
           "false_alarms_per_hour": round(fa / hours, 1) if hours else None,
           "false_alarms_of_0.5_s_or_more_per_hour": round(fa5 / hours, 1) if hours else None,
           "free_frames_in_alarm": round(free_alarm / free, 4) if free else None}
    if len(leads):
        out.update(timely=int((leads >= 20).sum()), timely_share=round(float((leads >= 20).mean()), 4),
                   median_lead_s=round(float(np.median(leads)) / FPS, 2),
                   missed_share=round(float((leads == 0).mean()), 4))
    return out


def named(rule: tuple) -> dict:
    return {"theta": rule[0], "hold_off_frames": rule[1], "persistence_frames": rule[2]}


def grid(sessions: list) -> dict:
    return {(t, h, k): score(sessions, (t, h, k)) for t in THETAS for h in HOLDS for k in KS}


def table(g: dict) -> list:
    keep = ("timely_share", "median_lead_s", "missed_share", "false_alarms", "false_alarms_per_hour",
            "free_frames_in_alarm")
    return [{**named(r), **{k: v[k] for k in keep}} for r, v in g.items()]


def study(stem: str) -> dict:
    val, val_src, _ = load(stem, "val")
    test, test_src, differing = load(stem, "test")
    gv, gt = grid(val), grid(test)
    floor = gv[RAW]["timely_share"] - MARGIN
    eligible = [r for r, v in gv.items() if v["timely_share"] >= floor - 1e-12]
    sel = min(eligible, key=lambda r: (gv[r]["false_alarms"], r[1], r[2], r[0]))
    raw_t, sel_t = gt[RAW], gt[sel]
    halves = sel_t["false_alarms_per_hour"] <= HALVE * raw_t["false_alarms_per_hour"]
    keeps = sel_t["timely_share"] >= raw_t["timely_share"] - MARGIN - 1e-12
    drivers = sorted({s["name"].rstrip("LCN") for s in test}, key=lambda u: int(u[4:]))

    def describe(sessions, rule):
        return {"lane_change_sessions": score(sessions, rule, lambda n: n.endswith("LC")),
                "baseline_sessions": score(sessions, rule, lambda n: n.endswith("N"))}

    def minutes(sessions):
        return round(sum(len(s["cls"]) for s in sessions) / FPS / 60, 1)
    return {
        "file": f"{stem}_int8_io.tflite",
        "validation_drivers": {
            "outputs_from": val_src, "sessions": len(val), "minutes": minutes(val),
            "raw_output": gv[RAW], "rules_within_the_margin": len(eligible),
            "selected_rule": named(sel), "selected": gv[sel], "all_rules": table(gv)},
        "test_drivers": {
            "outputs_from": test_src, "windows_differing_from_the_pc_reference": differing,
            "sessions": len(test), "minutes": minutes(test),
            "raw_output": raw_t, "selected": sel_t,
            "false_alarms_at_least_halved": bool(halves), "timely_share_within_5_points": bool(keeps),
            "adopted": bool(halves and keeps),
            "by_session_type_for_information": {"raw_output": describe(test, RAW), "selected": describe(test, sel)},
            "per_driver_for_information": {
                u: {"raw_output": score(test, RAW, lambda n, u=u: n.rstrip("LCN") == u),
                    "selected": score(test, sel, lambda n, u=u: n.rstrip("LCN") == u)} for u in drivers},
            "all_rules_for_information": table(gt),
            "note": "the rule was selected on the validation drivers only; the other rules on the test drivers "
                    "are listed for information and were not used"},
    }


def main():
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "family": {"theta": THETAS, "hold_off_frames": HOLDS, "persistence_frames": KS, "rules": 80,
                      "raw_output": named(RAW)},
           "selection": "validation drivers: fewest false alarms among the rules whose timely share is at most "
                        "5 points below the raw output's; ties: smaller hold-off, persistence, theta",
           "criterion": "test drivers: false alarms per hour at most half of the raw output's and timely share "
                        "within 5 points of the raw output's",
           "networks": {name: study(stem) for name, stem in NETS.items()}}
    out = HERE / "results" / "hil" / "decision_rule.json"
    out.write_text(json.dumps(res, indent=1))
    for name, n in res["networks"].items():
        v, t = n["validation_drivers"], n["test_drivers"]
        print(name)
        for part, d in (("validation", v), ("test", t)):
            for key in ("raw_output", "selected"):
                m = d[key]
                print(f"  {part:10s} {key:10s} timely {m['timely_share']:.3f} of {m['lane_changes']}, "
                      f"median lead {m['median_lead_s']} s, false alarms {m['false_alarms']} = "
                      f"{m['false_alarms_per_hour']}/h over {m['free_driving_hours']} h, "
                      f"free frames in alarm {m['free_frames_in_alarm']:.3f}")
        print(f"  selected rule {v['selected_rule']} ({v['rules_within_the_margin']} of 80 rules within the margin); "
              f"adopted on the test drivers: {t['adopted']}")


if __name__ == "__main__":
    sys.exit(main())
