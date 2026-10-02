"""Replay one real exiD exit and record what the same network says when trained on
exiD and when trained on highD.

The network is the hand-designed CNN of the transfer study (run_experiments.py),
seed 0: once trained on exiD (results/weights/exid_baseline_cls_seed0.pt, exiD input
scaling) and once trained on highD and applied as it is
(datasets/highd/results/source_weights/highd_source_cls_seed0.pt, highD input
scaling, as a model deployed on a highD-trained car would run). Both are run over
the 26 evaluation windows of every exit-to-off-ramp scenario of the exiD test split.

Selection rule, fixed before looking at the highD model's outputs: the exit scenario
whose robust prediction horizon under the exiD-trained model is closest to that
model's mean horizon over all test exits (a typical exit for the model that knows
junctions), among scenarios with at least 2 s of track after the crossing; ties go to
the first in the test file. What the highD model says on it is reported as it is.

Nothing is simulated: positions are the raw 25 Hz exiD tracks, the road is the
location's Lanelet2 map (WGS84 nodes projected to the recording's local UTM frame),
predictions come from the trained networks.

Run (.venv):  python datasets/exid/scenario_replay.py
Outputs (local, gitignored: derived from exiD data):
  datasets/exid/data/replay/scores.json    horizons of both models on every test exit
  datasets/exid/data/replay/scenario.json  chosen scenario: tracks, lanelets, predictions
"""
from __future__ import annotations

import json
import statistics as st
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src import lc_windows as W                                     # noqa: E402
from src.models.baseline import BaselineDSCNN                      # noqa: E402

PREP = HERE / "data" / "prepared"
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "replay"
HIGHD_PREP = ROOT / "datasets" / "highd" / "data" / "prepared"
W_EXID = HERE / "results" / "weights" / "exid_baseline_cls_seed0.pt"
W_HIGHD = ROOT / "datasets" / "highd" / "results" / "source_weights" / "highd_source_cls_seed0.pt"
KINDS = json.loads((PREP / "meta.json").read_text())["kinds"]
EXIT = KINDS.index("exit to off-ramp")
CLASS = {0: "lane keep", 1: "RIGHT lane change", 2: "LEFT lane change"}
FPS, SEQ_LEN, N_SLIDES = 25, 35, 26
AFTER_S = 2.0                       # track shown after the crossing
VIEW_M = 90.0                       # vehicles and lanelets kept within this distance of the target


# ------------------------------------------------------------------ WGS84 -> UTM 32N
def utm32(lat, lon):
    """Transverse Mercator (Krueger series, WGS84), zone 32N: central meridian 9 E."""
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    n = f / (2 - f)
    A = a / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
    e = np.sqrt(f * (2 - f))
    phi, lam = np.deg2rad(np.asarray(lat, float)), np.deg2rad(np.asarray(lon, float) - 9.0)
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi_p = np.arctan2(t, np.cos(lam))
    eta_p = np.arctanh(np.sin(lam) / np.sqrt(1 + t ** 2))
    xi = xi_p + sum(alpha[j] * np.sin(2 * (j + 1) * xi_p) * np.cosh(2 * (j + 1) * eta_p) for j in range(3))
    eta = eta_p + sum(alpha[j] * np.cos(2 * (j + 1) * xi_p) * np.sinh(2 * (j + 1) * eta_p) for j in range(3))
    return 500000.0 + k0 * A * eta, k0 * A * xi


def read_lanelets(loc: int, x0: float, y0: float):
    """Lanelets of one location in the recording's local frame: boundary polylines and tags."""
    path = next((RAW / "maps" / "lanelet2").glob(f"{loc}_*/*.osm"))
    root = ET.parse(path).getroot()
    ids, lat, lon = [], [], []
    for nd in root.iter("node"):
        ids.append(int(nd.get("id"))); lat.append(float(nd.get("lat"))); lon.append(float(nd.get("lon")))
    E, N = utm32(lat, lon)
    node = {i: (float(e - x0), float(n - y0)) for i, e, n in zip(ids, E, N)}
    ways = {}
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        pts = [node[int(r.get("ref"))] for r in w.findall("nd")]
        ways[int(w.get("id"))] = {"pts": pts, "type": tags.get("type", ""), "subtype": tags.get("subtype", "")}
    lanelets = {}
    for r in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        mem = {m.get("role"): int(m.get("ref")) for m in r.findall("member")}
        lanelets[int(r.get("id"))] = {"left": mem["left"], "right": mem["right"],
                                      "subtype": tags.get("subtype", ""),
                                      "offramp": tags.get("offramp") == "yes",
                                      "onramp": tags.get("onramp") == "yes"}
    return lanelets, ways


# ------------------------------------------------------------------ models
def load(path: Path) -> BaselineDSCNN:
    m = BaselineDSCNN(n_features=18, n_outputs=3)
    m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return m.eval()


def norm(prep: Path):
    x, _, _ = W.make_windows(*W.load_split(prep, "train"))
    return W.norm_stats(x)


def window_probs(model, feats, lo, hi) -> np.ndarray:
    """(S, 35, 18) scenarios -> (S, 26, 3) softmax outputs, evaluation-style windows."""
    S = len(feats)
    x, _, _ = W.make_windows(feats, np.zeros(S, np.int8), np.full(S, 35, np.int16))
    out = W.predict(model, W.to_model_input(x, lo, hi), "cpu")
    return torch.softmax(torch.from_numpy(out), -1).numpy().reshape(S, N_SLIDES, 3)


def horizons(pred: np.ndarray, label: int) -> tuple[float, float | None]:
    """Robust horizon (s; predictions stay correct from there to the crossing, 0 if the
    last window is wrong) and the earliest correct window (s), as in early_metrics()."""
    correct = pred == label
    robust = 0.0
    for i in range(N_SLIDES):
        if correct[i:].all():
            robust = (N_SLIDES - i) / 5
            break
    first = next(((N_SLIDES - i) / 5 for i in range(N_SLIDES) if correct[i]), None)
    return robust, first


def main():
    z = np.load(PREP / "test.npz")
    sel = np.nonzero(z["kind"] == EXIT)[0]
    feats = z["feats"][sel]
    m_e, m_h = load(W_EXID), load(W_HIGHD)
    p_e = window_probs(m_e, feats, *norm(PREP))
    p_h = window_probs(m_h, feats, *norm(HIGHD_PREP))
    label = z["label"][sel]
    acc_e = float((p_e.argmax(-1) == label[:, None]).mean())
    acc_h = float((p_h.argmax(-1) == label[:, None]).mean())
    print(f"{len(sel)} test exits; window accuracy seed 0: trained on exiD {acc_e:.4f}, highD as is {acc_h:.4f}")

    rows = []
    for k, i in enumerate(sel):
        re_, fe = horizons(p_e[k].argmax(-1), int(label[k]))
        rh, fh = horizons(p_h[k].argmax(-1), int(label[k]))
        rows.append({"test_index": int(i), "rec": int(z["rec"][i]), "loc": int(z["loc"][i]), "tv": int(z["tv"][i]),
                     "cross_frame": int(z["cross_frame"][i]), "robust_exid": re_, "first_exid": fe,
                     "robust_highd": rh, "first_highd": fh})
    mean_e = st.fmean(r["robust_exid"] for r in rows)
    mean_h = st.fmean(r["robust_highd"] for r in rows)

    # candidates with >= AFTER_S of track after the crossing (5 Hz grid frame)
    tracks_cache: dict[int, pd.DataFrame] = {}

    def tracks(rec: int) -> pd.DataFrame:
        if rec not in tracks_cache:
            tracks_cache.clear()
            tracks_cache[rec] = pd.read_csv(RAW / "data" / f"{rec:02d}_tracks.csv", low_memory=False,
                                            usecols=["trackId", "frame", "xCenter", "yCenter", "heading", "width",
                                                     "length", "lonVelocity", "laneletId", "latLaneCenterOffset"])
        return tracks_cache[rec]

    ranked = sorted(rows, key=lambda r: (abs(r["robust_exid"] - mean_e), r["test_index"]))
    chosen = None
    for r in ranked:
        f_cross = int(np.ceil(r["cross_frame"] / 5) * 5)      # first 5 Hz frame in the new lane
        last = tracks(r["rec"]).query("trackId == @r['tv']").frame.max()
        if last >= f_cross + AFTER_S * FPS:
            chosen = r
            break
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps({
        "rule": "exit closest to the exiD-trained model's mean robust horizon, >= 2 s of track after the crossing",
        "n_exits": len(rows), "window_acc_seed0": {"exid": acc_e, "highd_as_is": acc_h},
        "mean_robust_s": {"exid": mean_e, "highd_as_is": mean_h}, "chosen": chosen, "rows": rows}, indent=1))
    print(f"mean robust horizon over exits: exiD-trained {mean_e:.2f} s, highD as is {mean_h:.2f} s")
    print(f"chosen: test #{chosen['test_index']} rec {chosen['rec']} track {chosen['tv']}: exiD-trained robust "
          f"{chosen['robust_exid']} s, highD robust {chosen['robust_highd']} s")

    # ---- geometry of the chosen scenario
    rec, tv = chosen["rec"], chosen["tv"]
    meta = pd.read_csv(RAW / "data" / f"{rec:02d}_recordingMeta.csv").iloc[0]
    loc = int(meta.locationId)
    x0, y0 = float(meta.xUtmOrigin), float(meta.yUtmOrigin)
    f_cross = int(np.ceil(chosen["cross_frame"] / 5) * 5)
    f_first = f_cross - SEQ_LEN * 5
    f_end = int(f_cross + AFTER_S * FPS)
    t = tracks(rec)
    win = t[(t.frame >= f_first) & (t.frame <= f_end)]
    me = win[win.trackId == tv].sort_values("frame")
    assert me.frame.iloc[0] == f_first and me.frame.iloc[-1] == f_end, "track does not cover the scenario"
    path_xy = me[["xCenter", "yCenter"]].to_numpy()
    others = {}
    for oid, g in win[win.trackId != tv].groupby("trackId"):
        g = g.sort_values("frame")
        mine = me.set_index("frame").reindex(g.frame)
        d = np.hypot(g.xCenter.to_numpy() - mine.xCenter.to_numpy(), g.yCenter.to_numpy() - mine.yCenter.to_numpy())
        if np.nanmin(d) <= VIEW_M:
            others[int(oid)] = {k: g[c].round(3).tolist() for k, c in
                                (("frame", "frame"), ("x", "xCenter"), ("y", "yCenter"), ("heading", "heading"),
                                 ("w", "width"), ("l", "length"))}
    lanelets, ways = read_lanelets(loc, x0, y0)

    def near(pts):
        p = np.asarray(pts)
        return bool((np.min(np.hypot(p[:, None, 0] - path_xy[None, :, 0], p[:, None, 1] - path_xy[None, :, 1]))
                     <= VIEW_M + 40))
    keep = {i: l for i, l in lanelets.items() if near(ways[l["left"]]["pts"]) or near(ways[l["right"]]["pts"])}

    # check of the projection: the target's centre lies inside its own lanelet
    from matplotlib.path import Path as MPath
    inside, total = 0, 0
    for _, row in me.iterrows():
        lid = pd.to_numeric(str(row.laneletId).split(";")[0], errors="coerce")
        lan = lanelets.get(int(lid)) if np.isfinite(lid) else None
        if lan is None:
            continue
        poly = ways[lan["left"]]["pts"] + ways[lan["right"]]["pts"][::-1]
        total += 1
        inside += MPath(poly).contains_point((row.xCenter, row.yCenter))
    print(f"projection check: target centre inside its lanelet in {inside}/{total} frames")

    k = [r["test_index"] for r in rows].index(chosen["test_index"])
    scen = {
        "recording": rec, "location_id": loc, "track_id": tv,
        "location_map": next((RAW / "maps" / "lanelet2").glob(f"{loc}_*")).name,
        "city": next((RAW / "maps" / "lanelet2").glob(f"{loc}_*")).name.split("_")[1].capitalize(),
        "kind": "exit to off-ramp", "label": int(z["label"][chosen["test_index"]]), "fps_raw": FPS,
        "frame_first": f_first, "frame_cross": f_cross, "frame_end": f_end,
        "speed_at_cross_mps": float(abs(me[me.frame == f_cross].lonVelocity.iloc[0])),
        "projection_check": {"inside": inside, "frames": total},
        "tv": {k2: me[c].round(3).tolist() for k2, c in (("frame", "frame"), ("x", "xCenter"), ("y", "yCenter"),
                                                         ("heading", "heading"))}
              | {"w": float(me.width.iloc[0]), "l": float(me.length.iloc[0])},
        "others": others,
        "lanelets": {str(i): l for i, l in keep.items()},
        "ways": {str(w): ways[w] for l in keep.values() for w in (l["left"], l["right"])},
        "ttlc_s": [(N_SLIDES - s) / 5 for s in range(N_SLIDES)],
        "models": {
            "exid": {"name": "hand-designed CNN trained on exiD (seed 0)", "weights": W_EXID.name,
                     "probs": p_e[k].round(4).tolist(), "pred": p_e[k].argmax(-1).tolist(),
                     "robust_s": chosen["robust_exid"], "first_s": chosen["first_exid"]},
            "highd": {"name": "same CNN trained on highD, applied as it is (seed 0)", "weights": W_HIGHD.name,
                      "probs": p_h[k].round(4).tolist(), "pred": p_h[k].argmax(-1).tolist(),
                      "robust_s": chosen["robust_highd"], "first_s": chosen["first_highd"]},
        },
    }
    (OUT / "scenario.json").write_text(json.dumps(scen))
    print(f"scenario: {scen['location_map']} rec {rec} track {tv}, {scen['speed_at_cross_mps'] * 3.6:.0f} km/h, "
          f"{len(others)} other vehicles, {len(keep)} lanelets")
    for key in ("exid", "highd"):
        m = scen["models"][key]
        print(f"  {key}: classes {[CLASS[c][:5] for c in m['pred']]}")


if __name__ == "__main__":
    main()
