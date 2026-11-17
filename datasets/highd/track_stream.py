"""highD on complete vehicle tracks: what the protocol's scenarios do not score (2026-11-17).

The protocol scores 35-frame scenarios: lane-change scenarios that stay in one lane until the crossing, and
lane-keeping scenarios that end 5 s before a crossing or before the end of the track. No window of it holds
a completed lane change, and nothing in it says how often a tracked vehicle is called a lane change on a
road. Here the deployed int8 networks see every 10-frame window of every track of the test recordings
(56-60), the question asked on LCIR's whole drives (datasets/dmir/session_stream.py).

A window ending at frame k (5 Hz), with c a row where the lane changes:
  holds    it holds a completed lane change: k-9 < c <= k
  before   else, the next crossing comes 1 to 26 frames after k (the protocol's 0.2 to 5.2 s); label = its direction
  early    else, the next crossing comes 27 to 50 frames after k (where the protocol's lane-keeping scenarios
           of tracks with a crossing lie)
  free     otherwise
The measures were fixed before the outputs existed (local notes, 2026-11-17 11:14); there is no criterion,
this is a first measurement.

Correction after the first results (same day, on our side): a highD track lasts a median 13 s, and a vehicle
may change lanes just after it leaves the recorded section, so the label of a window is known only if a
crossing follows within 26 frames or the track goes on for at least 26 more frames. The "free" measures
above count windows whose label is unknown (the last 5.2 s of every track). The measures under "label_known"
count only windows whose label is known, split by the window's position in the track (its last frame before
or from frame 30, i.e. 6 s after the vehicle entered the section, where the deployed networks call far more
lane changes); "opposite_without_a_return" counts an opposite call after a lane change only if no crossing
back follows within 26 frames.

  .venv\\Scripts\\python datasets/highd/track_stream.py build [split]    # Windows: features of every track (test)
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/datasets/highd/track_stream.py run     # WSL: the outputs
  .venv\\Scripts\\python datasets/highd/track_stream.py --report
Output: datasets/highd/results/hil/track_stream.json (shares and counts only). Tracks and outputs stay in
firmware/build/probe/ (not versioned: licensed data).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREP = HERE / "data" / "prepared"
DEPLOY = HERE / "results" / "deploy"
PROBE = ROOT / "firmware" / "build" / "probe"
OUTS = PROBE / "highd_tracks_outs.npz"
OUT = HERE / "results" / "hil" / "track_stream.json"
NETS = {"searched (the paper's choice)": "highd_cls_model_1793711684616882545_453917_int8_io.tflite",
        "hand-designed CNN": "highd_dscnn_cls_int8_io.tflite"}
IN_LEN, FPS = 10, 5.0
HORIZON, EARLY = 26, 50                                      # frames: the protocol's 5.2 s; the early zone ends at 10 s
LEADS = (5, 10, 15, 20, 25)                                  # frames before the crossing: 1 to 5 s


def tracks_file(split: str = "test") -> Path:
    return PROBE / f"highd_tracks_{split}.npz"


def build(split: str = "test"):
    sys.path.insert(0, str(HERE))
    import prepare_highd as ph

    feats, lens, recs, tvs, cross = [], [], [], [], []
    short = invalid = 0
    for r in ph.SPLITS[split]:
        rec = ph.Recording(r)
        for tid, tv in rec.T.items():
            n = len(tv["frame"])
            if n < IN_LEN:
                short += 1
                continue
            f = rec.features(tid, 0, n)
            if f is None:
                invalid += 1
                continue
            cr, labs = ph.crossings(tv["laneId"], rec.direction[tid])
            feats.append(f)
            lens.append(n)
            recs.append(r)
            tvs.append(tid)
            cross.append([[int(c), int(d)] for c, d in zip(cr, labs)])
        print(f"recording {r}: {len(lens)} tracks so far", flush=True)
    PROBE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tracks_file(split), feats=np.concatenate(feats).astype(np.float32), lens=np.array(lens), rec=np.array(recs),
                        tv=np.array(tvs), cross=json.dumps(cross), short=short, invalid=invalid)
    print(f"{len(lens)} tracks, {int(sum(lens))} frames; left out: {short} shorter than a window, {invalid} with an invalid frame")


def run():
    import tensorflow as tf

    train = np.load(PREP / "train.npz")["feats"].astype(np.float32)
    lo = train.reshape(-1, train.shape[-1]).min(0)
    span = train.reshape(-1, train.shape[-1]).max(0) - lo + 1e-12
    z = np.load(tracks_file())
    feats, lens = z["feats"], z["lens"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    out = {}
    for n, file in enumerate(NETS.values()):
        it = tf.lite.Interpreter(model_path=str(DEPLOY / file))
        it.allocate_tensors()
        inp, o = it.get_input_details()[0], it.get_output_details()[0]
        si, zi = inp["quantization"]
        q = np.clip(np.rint(((feats - lo) / span) / np.float32(si) + np.float32(zi)), -128, 127).astype(np.int8)
        res = np.empty((int((lens - IN_LEN + 1).sum()), 3), np.int8)
        w = 0
        for a, m in zip(starts, lens):
            for k in range(m - IN_LEN + 1):
                it.set_tensor(inp["index"], q[a + k:a + k + IN_LEN][None])
                it.invoke()
                res[w] = it.get_tensor(o["index"]).reshape(-1)
                w += 1
        out[f"n{n}"] = res
        print(f"{file}: {w} windows", flush=True)
    np.savez_compressed(OUTS, **out)
    print("OUTPUTS-DONE")


def phases(n: int, cross: list):
    """Per window of a track (ending at k = 9 .. n-1): phase, label, frames to the next crossing, frames
    since the previous one and its direction."""
    k = np.arange(IN_LEN - 1, n)
    cs = np.array([c for c, _ in cross], dtype=np.int64)
    ds = np.array([d for _, d in cross], dtype=np.int64)
    to_next = np.full(len(k), 10 ** 6)
    next_dir = np.zeros(len(k), np.int64)
    since = np.full(len(k), 10 ** 6)
    prev_dir = np.zeros(len(k), np.int64)
    for c, d in zip(cs, ds):
        ahead = (c > k) & (c - k < to_next)
        to_next[ahead], next_dir[ahead] = (c - k)[ahead], d
        past = (c <= k) & (k - c < since)
        since[past], prev_dir[past] = (k - c)[past], d
    holds = since <= IN_LEN - 2                              # k-9 < c <= k
    before = ~holds & (to_next <= HORIZON)
    early = ~holds & (to_next > HORIZON) & (to_next <= EARLY)
    free = ~holds & ~before & ~early
    phase = np.where(holds, 3, np.where(before, 1, np.where(early, 2, 0)))    # 0 free, 1 before, 2 early, 3 holds
    return k, phase, np.where(before, next_dir, 0), to_next, next_dir, since, prev_dir


def runs(mask: np.ndarray):
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)   # the lengths


def share(a, b):
    return None if b == 0 else round(float(a) / float(b), 4)


def measures_of(pred_all: np.ndarray, lens: np.ndarray, cross: list) -> dict:
    """The measures of a network from its class per window, the windows in track order."""
    w_starts = np.concatenate([[0], np.cumsum(lens - IN_LEN + 1)[:-1]])
    count = {key: np.zeros(2, np.int64) for key in
             ("free", "free_tracks_without_a_lane_change", "early", "early_coming_direction", "before_correct",
              "after_free_completed", "after_free_opposite", "all_correct")}
    holds = np.zeros((IN_LEN - 1, 3), np.int64)           # by frames since the crossing: completed, opposite, windows
    lead = {L: np.zeros(2, np.int64) for L in LEADS}
    eligible = protocol_like = 0
    run_all = run_long = free_windows = 0
    known = {key: np.zeros(2, np.int64) for key in
             ("lane_keeping", "lane_keeping_first_6_s", "lane_keeping_from_6_s", "lane_keeping_tracks_without_a_lane_change",
              "lane_keeping_tracks_without_a_lane_change_from_6_s", "opposite_without_a_return", "all_correct")}
    for t, (n_frames, cr) in enumerate(zip(lens, cross)):
        k, phase, label, to_next, next_dir, since, prev_dir = phases(int(n_frames), cr)
        pred = pred_all[w_starts[t]:w_starts[t] + len(k)]
        called = pred != 0
        free = phase == 0
        rem = (int(n_frames) - 1) - k                         # frames of the track after the window's last frame
        is_known = (to_next <= HORIZON) | (rem >= HORIZON)
        lk = is_known & (phase != 3) & (to_next > HORIZON)    # the protocol's label "lane keeping", known
        late = k >= 30
        for key, m in (("lane_keeping", lk), ("lane_keeping_first_6_s", lk & ~late), ("lane_keeping_from_6_s", lk & late)):
            known[key] += (int(called[m].sum()), int(m.sum()))
        if not cr:
            known["lane_keeping_tracks_without_a_lane_change"] += (int(called[lk].sum()), int(lk.sum()))
            known["lane_keeping_tracks_without_a_lane_change_from_6_s"] += (int(called[lk & late].sum()), int((lk & late).sum()))
        hk = (phase == 3) & is_known
        back = (to_next <= HORIZON) & (next_dir == 3 - prev_dir)
        m = hk & ~back
        known["opposite_without_a_return"] += (int((called[m] & (pred[m] != prev_dir[m])).sum()), int(m.sum()))
        known["all_correct"] += (int((pred[is_known] == label[is_known]).sum()), int(is_known.sum()))
        count["all_correct"] += (int((pred == label).sum()), len(k))
        count["free"] += (int(called[free].sum()), int(free.sum()))
        if not cr:
            count["free_tracks_without_a_lane_change"] += (int(called[free].sum()), int(free.sum()))
        e = phase == 2
        count["early"] += (int(called[e].sum()), int(e.sum()))
        count["early_coming_direction"] += (int((pred[e] == next_dir[e]).sum()), int(e.sum()))
        b = phase == 1
        count["before_correct"] += (int((pred[b] == label[b]).sum()), int(b.sum()))
        h = phase == 3
        for s in range(IN_LEN - 1):
            m = h & (since == s)
            holds[s] += (int((pred[m] == prev_dir[m]).sum()), int((called[m] & (pred[m] != prev_dir[m])).sum()), int(m.sum()))
        a = free & (since >= IN_LEN - 1) & (since <= 25)  # 1.8 to 5 s after a crossing, free
        count["after_free_completed"] += (int((pred[a] == prev_dir[a]).sum()), int(a.sum()))
        count["after_free_opposite"] += (int((called[a] & (pred[a] != prev_dir[a])).sum()), int(a.sum()))
        for seg in np.split(np.arange(len(k)), np.flatnonzero(np.diff(free.astype(np.int8)) != 0) + 1):
            if len(seg) and free[seg[0]]:
                r = runs(called[seg])
                run_all += len(r)
                run_long += int((r >= 3).sum())
        free_windows += int(free.sum())
        cs = [c for c, _ in cr]
        for c, d in cr:
            one_lane = lambda first: first >= 0 and not any(first < x < c for x in cs)   # noqa: E731
            if not one_lane(c - 19):
                continue
            eligible += 1
            protocol_like += int(one_lane(c - 35))
            for L in LEADS:
                if one_lane(c - L - 9):
                    lead[L] += (int(pred[(c - L) - (IN_LEN - 1)] == d), 1)
    hours = free_windows / FPS / 3600
    return {
        "windows_correct_with_the_labels_above": share(*count["all_correct"]),
        "lane_changes_with_19_frames_in_one_lane_before": eligible,
        "of_them_with_35_frames_in_one_lane_before": protocol_like,
        "correct_class_in_the_window_ending_before_the_crossing": {
            f"{L / FPS:g}_s": {"share": share(*lead[L]), "lane_changes": int(lead[L][1])} for L in LEADS},
        "windows_before_a_crossing_correct": {"share": share(*count["before_correct"]), "windows": int(count["before_correct"][1])},
        "free_windows_called_a_lane_change": {"share": share(*count["free"]), "windows": int(count["free"][1])},
        "free_windows_called_a_lane_change_tracks_without_a_lane_change": {
            "share": share(*count["free_tracks_without_a_lane_change"]), "windows": int(count["free_tracks_without_a_lane_change"][1])},
        "early_windows_called_a_lane_change": {"share": share(*count["early"]), "the_coming_direction": share(*count["early_coming_direction"]),
                                               "windows": int(count["early"][1])},
        "windows_holding_a_completed_lane_change": {
            "called_the_completed_direction": share(holds[:, 0].sum(), holds[:, 2].sum()),
            "called_the_opposite_direction": share(holds[:, 1].sum(), holds[:, 2].sum()), "windows": int(holds[:, 2].sum()),
            "by_time_since_the_crossing_s": {f"{s / FPS:g}": {"completed": share(holds[s, 0], holds[s, 2]),
                                                              "opposite": share(holds[s, 1], holds[s, 2]), "windows": int(holds[s, 2])}
                                             for s in range(IN_LEN - 1)}},
        "free_windows_1.8_to_5_s_after_a_crossing": {"called_the_completed_direction": share(*count["after_free_completed"]),
                                                     "called_the_opposite_direction": share(*count["after_free_opposite"]),
                                                     "windows": int(count["after_free_completed"][1])},
        "runs_of_free_windows_called_a_lane_change_per_vehicle_hour": {"any_length": round(run_all / hours, 1),
                                                                       "0.6_s_or_more": round(run_long / hours, 1),
                                                                       "free_vehicle_hours": round(hours, 2)},
        "label_known": {
            "note": "windows whose label is known: a crossing follows within 26 frames or the track goes on for 26 more frames",
            "windows_correct": share(*known["all_correct"]), "windows": int(known["all_correct"][1]),
            **{f"{key}_windows_called_a_lane_change": {"share": share(*v), "windows": int(v[1])}
               for key, v in known.items() if key.startswith("lane_keeping")},
            "windows_holding_a_completed_lane_change_opposite_called_without_a_return": {
                "share": share(*known["opposite_without_a_return"]), "windows": int(known["opposite_without_a_return"][1])}}}


def report():
    z = np.load(tracks_file())
    lens = z["lens"].astype(np.int64)
    cross = json.loads(str(z["cross"]))
    outs = np.load(OUTS)
    prot = np.load(PREP / "test.npz")["label"]
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "recordings": sorted({int(r) for r in z["rec"]}),
           "tracks": int(len(lens)), "tracks_with_a_lane_change": int(sum(1 for c in cross if c)),
           "tracks_left_out": {"shorter_than_a_window": int(z["short"]), "with_an_invalid_frame": int(z["invalid"])},
           "frames": int(lens.sum()), "windows": int((lens - IN_LEN + 1).sum()),
           "vehicle_hours": round(float(lens.sum()) / FPS / 3600, 2),
           "lane_changes": int(sum(len(c) for c in cross)),
           "lane_change_scenarios_of_the_protocol": int((prot != 0).sum()), "networks": {}}
    for n, name in enumerate(NETS):
        res["networks"][name] = measures_of(outs[f"n{n}"].astype(np.int16).argmax(-1), lens, cross)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "build":
        build(*sys.argv[2:3])
    else:
        {"run": run, "--report": report}.get(step, lambda: sys.exit("build [split] | run | --report"))()
