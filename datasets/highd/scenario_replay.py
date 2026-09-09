"""Replay one real highD lane change and record what each model would have said.

Picks a real lane-change scenario from the TEST recordings, exports the raw
25 Hz trajectory of the target vehicle and its neighbours (for drawing), and
runs our models over the scenario's 26 prediction windows exactly as in
evaluation, so the animation shows when each model first predicts the manoeuvre
and from when it stays correct. Nothing is simulated: positions come from the
tracks CSV, predictions from the trained models, timings from board
measurements (added by the renderer).

Step 1  python  scenario_replay.py select   scenario selection (pandas)
Step 2  .venv   scenario_replay.py hand     hand-CNN (torch) predictions
Step 3  WSL     scenario_replay.py keras    searched-model (keras) predictions

Outputs (local, gitignored: derived from highD data):
  datasets/highd/data/replay/scenario.json       trajectory + lanes + neighbours
  datasets/highd/data/replay/pred_hand_cnn.json  26 window predictions
  datasets/highd/data/replay/pred_searched.json  26 window predictions (step 2)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
# prepare_highd needs pandas, which on this machine lives in the Store Python
# (Smart App Control blocks it inside the venv), while torch lives in the venv.
# So the selection step imports it lazily and runs under a different
# interpreter from the prediction steps:  select -> python, hand -> .venv,
# keras -> WSL.
# The window helpers are re-stated here in numpy only, so that no step drags in
# torch (train_highd imports it at module level, which the Store Python and the
# WSL keras environment do not have).
IN_LEN, SEQ_LEN, FPS = 10, 35, 5
N_SLIDES = SEQ_LEN - IN_LEN + 1


def make_windows(feats, label, cross):
    S = len(feats)
    sw = np.lib.stride_tricks.sliding_window_view(feats, IN_LEN, axis=1)
    x = np.ascontiguousarray(sw.transpose(0, 1, 3, 2)).reshape(S * N_SLIDES, IN_LEN, feats.shape[2])
    y = np.repeat(label, N_SLIDES)
    s_idx = np.tile(np.arange(N_SLIDES), S)
    cx = np.repeat(cross.astype(np.float32), N_SLIDES)
    ttlc = (cx - (s_idx + IN_LEN) + 1) / FPS
    ttlc[cx < 0] = np.nan
    return x, y, ttlc


def norm_stats(x):
    flat = x.reshape(-1, x.shape[2])
    return flat.min(0), flat.max(0)


def apply_norm(x, lo, hi):
    return (x - lo) / (hi - lo + 1e-12)

OUT = HERE / "data" / "replay"
PREP = HERE / "data" / "prepared"
REC_ID = 56                     # first test recording
LABEL_NAME = {0: "lane keep", 1: "right lane change", 2: "left lane change"}


def candidates(rec, limit=60):
    """Lane changes in the recording with >=2 s of track after the crossing and a
    neighbour in the target lane. Yields (tid, first, c, lab, feats)."""
    import prepare_highd as ph
    n = 0
    for tid, tv in rec.T.items():
        d = rec.direction[tid]
        cr, labs = ph.crossings(tv["laneId"], d)
        for c, lab in zip(cr, labs):
            first = c - ph.SEQ_LEN
            if first < 0 or len(set(tv["laneId"][first:c])) > 1:
                continue
            if len(tv["laneId"]) - c < 10:
                continue
            f = rec.features(tid, first, c)
            if f is None:
                continue
            side = "rightAlongsideId" if lab == 1 else "leftAlongsideId"
            has_nb = any(int(tv[side][i]) or int(tv[side.replace("Alongside", "Preceding")][i])
                         for i in range(first, c))
            if not has_nb:
                continue
            yield tid, first, c, lab, f
            n += 1
            if n >= limit:
                return


def main_scan():
    """Export the candidate scenarios' evaluation inputs so a model can score
    them in one batch (keras-scan), then `select <tid>` the typical one."""
    import prepare_highd as ph
    rec = ph.Recording(REC_ID)
    metas, feats = [], []
    for tid, first, c, lab, f in candidates(rec):
        metas.append({"tid": int(tid), "first": int(first), "c": int(c), "label": int(lab)})
        feats.append(f)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "candidates.npz", feats=np.stack(feats), meta=json.dumps(metas))
    print(f"{len(metas)} candidate scenarios exported")


def main_keras_scan(target=4.53):
    import keras
    z = np.load(OUT / "candidates.npz")
    metas = json.loads(str(z["meta"]))
    m = keras.models.load_model(HERE / "results" / "models" / "highd_cls_tight_model_aaaaap.h5",
                                compile=False)

    def predict(xn):
        logits = m.predict(xn, batch_size=64, verbose=0)
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    scores = []
    for meta, f in zip(metas, z["feats"]):
        rec = window_predictions(f, predict)
        rec["label"] = meta["label"]
        summarize(rec, quiet=True)
        scores.append({**meta, "robust": rec["robust_from_ttlc_s"], "first": rec["first_correct_ttlc_s"]})
    ok = [s for s in scores if s["robust"] is not None]
    best = min(ok, key=lambda s: abs(s["robust"] - target))
    (OUT / "candidates_scores.json").write_text(json.dumps({"target": target, "chosen": best,
                                                            "scores": scores}, indent=1))
    print(f"chosen track {best['tid']} ({LABEL_NAME[best['label']]}): robust horizon "
          f"{best['robust']} s, closest to the test-set average {target} s")


def pick_scenario(tid_wanted=None):
    """The requested scenario (by track id) or, by default, the first candidate."""
    import prepare_highd as ph
    rec = ph.Recording(REC_ID)
    for tid, first, c, lab, f in candidates(rec):
        if tid_wanted is None or int(tid) == int(tid_wanted):
            return rec, tid, first, c, lab, f
    raise SystemExit("no suitable scenario found")


def _unused_pick_scenario_old():
    import prepare_highd as ph
    rec = ph.Recording(REC_ID)
    for tid, tv in rec.T.items():
        d = rec.direction[tid]
        cr, labs = ph.crossings(tv["laneId"], d)
        for c, lab in zip(cr, labs):
            first = c - ph.SEQ_LEN
            if first < 0 or len(set(tv["laneId"][first:c])) > 1:
                continue
            if len(tv["laneId"]) - c < 10:          # want >= 2 s after the crossing
                continue
            f = rec.features(tid, first, c)
            if f is None:
                continue
            side = "rightAlongsideId" if lab == 1 else "leftAlongsideId"
            has_nb = any(int(tv[side][i]) or int(tv[side.replace("Alongside", "Preceding")][i])
                         for i in range(first, c))
            if not has_nb:
                continue
            return rec, tid, first, c, lab, f
    raise SystemExit("no suitable scenario found")


def raw_track(rec_id, tid, f_start, f_end):
    """25 Hz rows of one vehicle between two absolute frames (inclusive)."""
    import pandas as pd
    import prepare_highd as ph
    df = pd.read_csv(ph.RAW / f"{rec_id:02d}_tracks.csv")
    g = df[(df.id == tid) & (df.frame >= f_start) & (df.frame <= f_end)].sort_values("frame")
    return g


def main_select(tid_wanted=None):
    import prepare_highd as ph
    rec, tid, first, c, lab, feats = pick_scenario(tid_wanted)
    tv = rec.T[tid]
    d = rec.direction[tid]
    f_first = int(tv["frame"][first])          # first scenario frame (5 Hz grid)
    f_cross = int(tv["frame"][c])              # first frame in the new lane
    f_end = int(tv["frame"][min(c + 10, len(tv["frame"]) - 1)])   # +2 s
    import pandas as pd
    df = pd.read_csv(ph.RAW / f"{REC_ID:02d}_tracks.csv")
    win = df[(df.frame >= f_first) & (df.frame <= f_end)]
    tv_rows = win[win.id == tid].sort_values("frame")
    # neighbours: any vehicle referenced by the TV during the window
    nb_ids = set()
    for col in ("precedingId", "followingId", "leftPrecedingId", "leftAlongsideId",
                "leftFollowingId", "rightPrecedingId", "rightAlongsideId", "rightFollowingId"):
        nb_ids |= set(int(v) for v in tv_rows[col].unique() if int(v) != 0)
    nbs = {}
    for nid in sorted(nb_ids):
        r = win[win.id == nid].sort_values("frame")
        if len(r):
            nbs[nid] = {"frame": r.frame.tolist(), "x": r.x.round(2).tolist(),
                        "y": r.y.round(2).tolist(),
                        "w": float(r.width.iloc[0]), "h": float(r.height.iloc[0])}
    markings = rec.upper if d == 1 else rec.lower
    scen = {
        "recording": REC_ID, "track_id": int(tid), "driving_direction": int(d),
        "label": int(lab), "label_name": LABEL_NAME[lab],
        "fps_raw": 25, "frame_first": f_first, "frame_cross": f_cross, "frame_end": f_end,
        "lane_markings_y": [float(v) for v in markings],
        "all_markings_y": {"upper": [float(v) for v in rec.upper],
                           "lower": [float(v) for v in rec.lower]},
        "tv": {"frame": tv_rows.frame.tolist(), "x": tv_rows.x.round(2).tolist(),
               "y": tv_rows.y.round(2).tolist(),
               "xVelocity": tv_rows.xVelocity.round(2).tolist(),
               "w": float(tv_rows.width.iloc[0]), "h": float(tv_rows.height.iloc[0])},
        "neighbours": nbs,
        "speed_at_cross_mps": float(abs(tv_rows[tv_rows.frame == f_cross].xVelocity.iloc[0])),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scenario.json").write_text(json.dumps(scen))
    np.save(OUT / "scenario_feats.npy", feats)     # (35, 18), the evaluation input
    print(f"scenario: rec {REC_ID} track {tid} {LABEL_NAME[lab]} | crossing at frame {f_cross} "
          f"| speed {scen['speed_at_cross_mps']:.1f} m/s | {len(nbs)} neighbours")
    return scen, feats


def window_predictions(feats, predict_fn):
    """Run a model over the 26 windows of one scenario, evaluation-style."""
    z = np.load(PREP / "train.npz")
    xtr, _, _ = make_windows(z["feats"], z["label"], z["cross_idx"])
    lo, hi = norm_stats(xtr)
    x, _, _ = make_windows(feats[None], np.array([0], np.int8), np.array([35], np.int16))
    xn = apply_norm(x, lo, hi).astype(np.float32)          # (26, 10, 18)
    probs = predict_fn(xn)                                   # (26, 3) softmax
    pred = probs.argmax(1)
    ttlc = [(N_SLIDES - s) / FPS for s in range(N_SLIDES)]  # their (26-s)/5 convention
    return {"ttlc_s": ttlc, "pred": pred.tolist(),
            "p_lc": probs[:, 1:].sum(1).round(4).tolist(), "probs": probs.round(4).tolist()}


def main_hand_cnn():
    import torch
    from src.models.baseline import BaselineDSCNN
    scen = json.loads((OUT / "scenario.json").read_text())
    feats = np.load(OUT / "scenario_feats.npy")
    m = BaselineDSCNN(n_features=18, n_outputs=3)
    m.load_state_dict(torch.load(HERE / "results" / "highd_baseline_cls_v2.pt", weights_only=True))
    m.eval()

    def predict(xn):
        with torch.no_grad():
            out = m(torch.from_numpy(np.ascontiguousarray(xn.transpose(0, 2, 1))))
            return torch.softmax(out, 1).numpy()
    rec = window_predictions(feats, predict)
    rec["model"] = "hand CNN, 8.4k (results/highd_baseline_cls_v2.pt)"
    rec["label"] = scen["label"]
    summarize(rec)
    (OUT / "pred_hand_cnn.json").write_text(json.dumps(rec))


def main_keras():
    import keras
    scen = json.loads((OUT / "scenario.json").read_text())
    feats = np.load(OUT / "scenario_feats.npy")
    m = keras.models.load_model(HERE / "results" / "models" / "highd_cls_tight_model_aaaaap.h5",
                                compile=False)

    def predict(xn):
        logits = m.predict(xn, batch_size=64, verbose=0)
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    rec = window_predictions(feats, predict)
    rec["model"] = "searched, 5,347 params (results/models/highd_cls_tight_model_aaaaap.h5)"
    rec["label"] = scen["label"]
    summarize(rec)
    (OUT / "pred_searched.json").write_text(json.dumps(rec))


def summarize(rec, quiet=False):
    lab = rec["label"]
    correct = [p == lab for p in rec["pred"]]
    first = next((rec["ttlc_s"][i] for i, c in enumerate(correct) if c), None)
    robust = None
    for i in range(len(correct)):
        if all(correct[i:]):
            robust = rec["ttlc_s"][i]; break
    rec["first_correct_ttlc_s"] = first
    rec["robust_from_ttlc_s"] = robust
    if not quiet:
        print(f"{rec['model']}: first correct at TTLC {first} s, robust from {robust} s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "select"
    if mode == "scan":            # Store python (pandas)
        main_scan()
    elif mode == "keras-scan":    # WSL
        main_keras_scan()
    elif mode == "select":        # Store python (pandas); optional track id
        main_select(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "hand":          # .venv (torch)
        main_hand_cnn()
    elif mode == "keras":         # WSL dmir_nas venv (keras)
        main_keras()
