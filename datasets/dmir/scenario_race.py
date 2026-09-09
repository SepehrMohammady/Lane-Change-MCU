"""DMIR "race" scenario: one real lane change of a held-out test driver, replayed
for every model in the T4.5 table on identical inputs.

The prepared pickles hold shuffled 50-step windows, so a continuous scene has to
be rebuilt from the raw session (Materials/h5-samples/user{13,2}LC.h5, both
official test drivers). The provider's preprocessing is recovered, not guessed:

  * every pickle channel is an affine map of one raw signal; the map is fitted
    on windows located in the test pickle by correlation (r > 0.9995 on YawRate
    and LatAcceleration) and then checked to reproduce those windows exactly;
  * car1 (channels 16-21) is the vehicle nearest in |LongPosition| among object
    slots 1-10; car2 (22-27) is object slot 11, the scenario's dedicated "car on
    the left"; car2Present (30) is that slot's ID != -1; an absent car is zeros;
  * the turn-indicator channels (28, 29) are unscaled 0/1;
  * egoLaneWidth (7) is constant and maps to 0.

Windows built this way match the pickle windows to ~1e-13, so the model outputs
on the replay are exactly what the pipeline would produce. The windows are then
clipped to the training range, as in every evaluation in this repository.

Modes
  build    (.venv)  recover the normalisation, verify it, cut every lane-change
                    event of both drivers into 101 windows (t = -8 .. +2 s at
                    10 Hz) and save drawing data          -> data/race/*.npz|json
  predict  (WSL)    run the four table models on every window -> data/race/preds.npz
  select   (any)    pick the typical event and write data/race/scenario.json

Run:  .venv/Scripts/python datasets/dmir/scenario_race.py build
      (WSL) ~/dmir_nas/bin/python datasets/dmir/scenario_race.py predict
      .venv/Scripts/python datasets/dmir/scenario_race.py select
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / "data"
RACE = DATA / "race"
RACE.mkdir(parents=True, exist_ok=True)
H5DIR = ROOT / "Materials" / "h5-samples"
USERS = ("user13LC", "user2LC")            # official test drivers (verify_split.py)
W, FPS, LANE_W = 50, 10, 3.75
PRE, POST = 80, 20                         # frames before / after the crossing
CAR_FIELDS = ("LatPosition", "LatVelocity", "LongPosition", "LongVelocity", "YawAngle", "YawRate")
CLASS_NAME = {0: "no intention", 1: "right lane change", 2: "left lane change"}

MODELS = {   # key: (file, kind) ; numbers quoted on the slide come from datasets/dmir/docs/deployment.md
    "ref":  ("datasets/dmir/results/deploy/REF_cnn_multi.h5", "keras"),
    "best": ("datasets/dmir/results/deploy/cls_best.h5", "keras"),
    "qat":  ("datasets/dmir/results/qat/cls_best_qat_int8_io.tflite", "tflite_int8"),
    "tiny": ("datasets/dmir/results/deploy/cls_tiny.h5", "keras"),
}


# ----------------------------------------------------------------------------- raw session
def load_session(user):
    import h5py
    f = h5py.File(H5DIR / f"{user}.h5", "r")
    ego, ll, ob = f["egoVehicle"][()], f["laneLines"][()], f["objects"][()]
    meta = f.attrs["metaData"][0]
    return ego, ll, ob, {"length": float(meta["Car"]["VehicleLength"]), "width": float(meta["Car"]["VehicleWidth"])}


def raw_matrix(ego, ll, ob):
    """(N, 31) raw signals in the classification-layout order (feature-map.md)."""
    so, nobj, ids = ob["sObject"], ob["NumberOfObjects"], ob["sObject"]["ID"]
    N = len(ego)
    S = {fn: np.nan_to_num(so[fn].astype(float), nan=0.0) for fn in CAR_FIELDS}
    car1 = np.zeros((N, 6))
    for i in range(N):
        best = None
        for j in range(1, min(nobj[i], 11)):          # slots 1-10 = traffic; 0 = ego, 11 = left car
            d = abs(so["LongPosition"][i, j])
            if best is None or d < best[0]:
                best = (d, j)
        if best is not None:
            car1[i] = [S[fn][i, best[1]] for fn in CAR_FIELDS]
    car2 = np.stack([S[fn][:, 11] for fn in CAR_FIELDS], 1)
    present = (ids[:, 11] != -1).astype(float)
    dy, yaw, cur, cdx = (ll["sLaneLine"][k] for k in ("Dy", "YawAngle", "Curvature", "CurvatureDx"))
    di = ego["DirectionIndicator"].astype(float)
    cols = [ego["LatAcceleration"], ego["LongAcceleration"], ego["BrakePedalPos"], ego["SteeringAngle"],
            ego["ThrottlePedalPos"], ego["VehicleSpeed"], ego["YawRate"], ll["EgoLaneWidth"],
            yaw[:, 0], yaw[:, 1], cur[:, 0], cur[:, 1], cdx[:, 0], cdx[:, 1], dy[:, 0], dy[:, 1]]
    M = np.stack([np.asarray(c, float) for c in cols], 1)
    return np.concatenate([M, car1, car2, (di == 1)[:, None], (di == 2)[:, None], present[:, None]], 1)


def windows(a):
    n = len(a) - W + 1
    idx = np.arange(n)[:, None] + np.arange(W)[None, :]
    return a[idx]


# ----------------------------------------------------------------------------- normalisation
def match_pairs(M, x_test):
    """(pickle_idx, raw_start) pairs: same window, found by correlation on two channels."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def z(a):
        t = torch.from_numpy(np.asarray(a, np.float64)).float().to(dev)
        m, s = t.mean(1, keepdim=True), t.std(1, keepdim=True)
        return torch.where(s > 1e-8, (t - m) / s, torch.zeros_like(t))
    zr6, zr0 = z(windows(M[:, 6])), z(windows(M[:, 0]))
    zp6, zp0 = z(x_test[:, :, 6]), z(x_test[:, :, 0])
    pairs = []
    for i in range(0, len(x_test), 4096):
        c = torch.minimum(zp6[i:i + 4096] @ zr6.T, zp0[i:i + 4096] @ zr0.T) / (W - 1)
        m, arg = c.max(dim=1)
        for k in torch.where(m > 0.9995)[0]:
            pairs.append((i + int(k), int(arg[k])))
    return np.array(pairs)


def fit_affine(M, x_test, pairs):
    p_idx, r_idx = pairs[:, 0], pairs[:, 1]
    a, b, err = np.zeros(31), np.zeros(31), np.zeros(31)
    for c in range(31):
        xr = windows(M[:, c])[r_idx].reshape(-1)
        yp = x_test[p_idx][:, :, c].reshape(-1)
        if xr.std() < 1e-12:
            a[c], b[c] = 0.0, float(np.median(yp))
        else:
            (a[c], b[c]), *_ = np.linalg.lstsq(np.stack([xr, np.ones_like(xr)], 1), yp, rcond=None)
        err[c] = float(np.abs(a[c] * xr + b[c] - yp).max())
    return a, b, err


# ----------------------------------------------------------------------------- events
def crossings(ll):
    """(frame, direction) for every lane crossing: the right-line distance jumps
    by a lane width; it drops for a left change and rises for a right change."""
    dy0 = ll["sLaneLine"]["Dy"][:, 0]
    d = np.diff(dy0)
    out = []
    for i in np.where(np.abs(d) > 2.0)[0]:
        out.append((int(i + 1), 2 if d[i] < 0 else 1))
    return out


def event_frames(cross):
    return list(range(cross - PRE, cross + POST + 1))


def lane_index(ll):
    """Lane of the ego per frame (0 = right lane of the two-lane road), tracked by
    counting crossings from the first frame, where the missing right-adjacent
    line says which lane the session starts in."""
    dy = ll["sLaneLine"]["Dy"]
    lane = np.full(len(dy), 0 if np.isnan(dy[:20, 2]).mean() > 0.5 else 1)
    for f, direction in crossings(ll):
        lane[f:] += 1 if direction == 2 else -1
    return np.clip(lane, 0, 1)


def drawing_data(ego, ll, ob, frames, cross):
    so, nobj, ids = ob["sObject"], ob["NumberOfObjects"], ob["sObject"]["ID"]
    dy = ll["sLaneLine"]["Dy"]
    v = ego["VehicleSpeed"].astype(float)
    lane = lane_index(ll)
    x_abs = np.concatenate([[0.0], np.cumsum(v[frames[0]:frames[-1]] * (1 / FPS))])
    rows = []
    for k, f in enumerate(frames):
        objs = []
        for j in range(1, nobj[f]):
            if ids[f, j] == -1 or abs(so["LongPosition"][f, j]) > 120:
                continue
            objs.append({"id": int(ids[f, j]), "lat": round(float(so["LatPosition"][f, j]), 3),
                         "long": round(float(so["LongPosition"][f, j]), 2), "len": round(float(so["Length"][f, j]), 2),
                         "wid": round(float(so["Width"][f, j]), 2), "lvel": round(float(so["LongVelocity"][f, j]), 2),
                         "slot": j})
        rows.append({"t": round((f - cross) / FPS, 1), "frame": int(f), "x": round(float(x_abs[k]), 3),
                     "v": round(float(v[f]), 3), "lane": int(lane[f]),
                     "y": round(float(lane[f] * LANE_W + dy[f, 0]), 3),           # from the road's right edge
                     "ind": int(ego["DirectionIndicator"][f]), "objs": objs})
    return rows


# ----------------------------------------------------------------------------- modes
def main_build():
    x_test = pickle.load(open(DATA / "data-classification" / "x_test_multi.pkl", "rb")).astype(np.float64)
    y_test = np.asarray(pickle.load(open(DATA / "data-classification" / "y_test_multi.pkl", "rb"))).squeeze()
    rng = np.load(RACE / "cls_train_range.npz")
    lo, hi = rng["lo"], rng["hi"]
    sessions, pairs_all, norm = {}, {}, None
    for user in USERS:
        ego, ll, ob, car = load_session(user)
        M = raw_matrix(ego, ll, ob)
        pairs = match_pairs(M, x_test)
        a, b, err = fit_affine(M, x_test, pairs)
        print(f"[{user}] {len(pairs)} test-pickle windows located; affine fit max error per channel: "
              f"{err.max():.2e} (worst channel {int(err.argmax())})")
        if norm is None:
            norm = (a, b)
        else:                                       # the second driver must give the same map
            da = np.abs(a - norm[0]).max()
            db = np.abs(b - norm[1]).max()
            print(f"          map agrees with the first driver to {max(da, db):.2e}")
        sessions[user] = (ego, ll, ob, car, M)
        pairs_all[user] = pairs
    a, b = norm
    (RACE / "norm_cls.json").write_text(json.dumps({"a": a.tolist(), "b": b.tolist(),
                                                   "note": "pickle = a * raw + b per channel, classification layout"}, indent=1))

    events, wins, meta = [], [], []
    for user in USERS:
        ego, ll, ob, car, M = sessions[user]
        N = len(ego)
        Mn = M * a + b
        start_to_pickle = {int(r): int(p) for p, r in pairs_all[user]}
        xs = crossings(ll)
        for k, (cross, direction) in enumerate(xs):
            frames = event_frames(cross)
            if frames[0] - (W - 1) < 0 or frames[-1] >= N:
                continue
            others = [c for c, _ in xs if c != cross and frames[0] <= c <= frames[-1]]
            eid = len(events)
            max_diff, n_in_pickle, labels = 0.0, 0, []
            for f in frames:
                w = Mn[f - W + 1:f + 1]
                s = f - W + 1
                if s in start_to_pickle:
                    p = start_to_pickle[s]
                    max_diff = max(max_diff, float(np.abs(w - x_test[p]).max()))
                    n_in_pickle += 1
                    labels.append(int(y_test[p]))
                else:
                    labels.append(-1)
                wins.append(np.clip(w, lo, hi).astype(np.float32))
                meta.append((eid, f, (f - cross) / FPS))
            rows = drawing_data(ego, ll, ob, frames, cross)
            y_step = max(abs(r1["y"] - r0["y"]) for r0, r1 in zip(rows, rows[1:]))
            events.append({"id": eid, "user": user, "cross_frame": cross, "direction": direction,
                           "y_max_step_m": round(y_step, 3),
                           "label_name": CLASS_NAME[direction], "t_cross_s": round(float(ego["FileTime"][cross]), 1),
                           "speed_at_cross_mps": round(float(ego["VehicleSpeed"][cross]), 3),
                           "other_crossings_in_span": others, "windows_in_test_pickle": n_in_pickle,
                           "max_abs_diff_vs_pickle": max_diff, "pickle_labels": labels,
                           "ego": car, "frames": rows})
    wins = np.stack(wins)
    meta = np.array(meta)
    np.savez_compressed(RACE / "windows.npz", win=wins, event=meta[:, 0].astype(int),
                        frame=meta[:, 1].astype(int), t=meta[:, 2])
    (RACE / "events.json").write_text(json.dumps(events))
    jumps = max(abs(f1["y"] - f0["y"]) for e in events for f0, f1 in zip(e["frames"], e["frames"][1:]))
    print(f"largest frame-to-frame lateral step of the ego over all events: {jumps:.2f} m (must be small)")
    checked = [e for e in events if e["windows_in_test_pickle"]]
    print(f"{len(events)} lane-change events ({sum(e['direction'] == 2 for e in events)} left, "
          f"{sum(e['direction'] == 1 for e in events)} right); {wins.shape[0]} windows")
    print(f"windows also present in the test pickle: {sum(e['windows_in_test_pickle'] for e in events)}; "
          f"max |ours - pickle| over them: {max(e['max_abs_diff_vs_pickle'] for e in checked):.2e}")


def main_predict():
    import tensorflow as tf
    d = np.load(RACE / "windows.npz")
    X = d["win"]
    out = {}
    for key, (path, kind) in MODELS.items():
        if kind == "keras":
            m = tf.keras.models.load_model(ROOT / path, compile=False)
            p = m.predict(X, batch_size=512, verbose=0)
            if not np.allclose(p.sum(1), 1, atol=1e-3):       # logits -> probabilities
                p = np.exp(p - p.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        else:
            it = tf.lite.Interpreter(model_path=str(ROOT / path))
            it.allocate_tensors()
            inp, outp = it.get_input_details()[0], it.get_output_details()[0]
            s_in, z_in = inp["quantization"]
            s_out, z_out = outp["quantization"]
            p = np.zeros((len(X), 3), np.float32)
            for i, w in enumerate(X):
                q = np.clip(np.round(w / s_in) + z_in, -128, 127).astype(np.int8)
                it.set_tensor(inp["index"], q.reshape(inp["shape"]))
                it.invoke()
                o = (it.get_tensor(outp["index"]).astype(np.float32) - z_out) * s_out
                p[i] = o.reshape(-1)
            if not np.allclose(p.sum(1), 1, atol=1e-2):
                p = np.exp(p - p.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        out[key] = p.astype(np.float32)
        print(f"[{key}] {path.split('/')[-1]}: {len(X)} windows, mean max-prob {p.max(1).mean():.3f}")
    np.savez_compressed(RACE / "preds.npz", **out)


def first_robust(probs, t, direction):
    """Earliest window end t0 < 0 such that argmax == direction for every window
    in [t0, -0.1 s]. The window ending on the crossing frame itself already
    contains the lane-line jump, so it is not part of the prediction."""
    order = np.argsort(t)
    tt, pp = t[order], probs[order]
    pre = tt < 0
    hit = pp[pre].argmax(1) == direction
    if not hit[-1]:
        return None
    k = len(hit) - 1
    while k > 0 and hit[k - 1]:
        k -= 1
    return float(tt[pre][k])


def main_select():
    events = json.loads((RACE / "events.json").read_text())
    d = np.load(RACE / "windows.npz")
    P = np.load(RACE / "preds.npz")
    summary = []
    for e in events:
        m = d["event"] == e["id"]
        t = d["t"][m]
        flags = {k: first_robust(P[k][m], t, e["direction"]) for k in MODELS}
        summary.append({"id": e["id"], "user": e["user"], "direction": e["direction"], "flags": flags,
                        "others": len(e["other_crossings_in_span"]), "v": e["speed_at_cross_mps"],
                        "left_car": any(o["slot"] == 11 for fr in e["frames"] for o in fr["objs"])})
    # usable: every model flags before the crossing, no second crossing in the span, and the
    # lane-line record has no reset (a lateral step above 0.5 m in one frame is a session artefact)
    ok = [s for s in summary if all(v is not None for v in s["flags"].values()) and s["others"] == 0
          and events[s["id"]]["y_max_step_m"] < 0.5]
    med = float(np.median([s["flags"]["best"] for s in ok]))
    print(f"{len(events)} events; {len(ok)} with every model flagging before the crossing and no second crossing")
    print(f"searched-84k first robust flag: median {med:+.1f} s; per-model medians: "
          + ", ".join(f"{k} {np.median([s['flags'][k] for s in ok]):+.1f}" for k in MODELS))
    # typical = closest to the median flag time; prefer a scene with the left car and a nearby lead vehicle
    def score(s):
        e = events[s["id"]]
        near = min((abs(o["long"]) for fr in e["frames"] for o in fr["objs"]), default=999)
        return (abs(s["flags"]["best"] - med), 0 if s["left_car"] else 1, near)
    ok.sort(key=score)
    chosen = ok[0]
    e = events[chosen["id"]]
    m = d["event"] == e["id"]
    order = np.argsort(d["t"][m])
    scen = {k: e[k] for k in ("id", "user", "cross_frame", "direction", "label_name", "t_cross_s",
                              "speed_at_cross_mps", "ego", "frames", "windows_in_test_pickle", "max_abs_diff_vs_pickle")}
    scen["pickle_labels"] = e["pickle_labels"]
    scen["t_windows"] = d["t"][m][order].round(1).tolist()
    scen["probs"] = {k: P[k][m][order].round(4).tolist() for k in MODELS}
    scen["flags"] = chosen["flags"]
    scen["population"] = {"n_events": len(events), "n_ok": len(ok), "median_flag_best": med,
                          "flags_all": [s["flags"] for s in ok]}
    (RACE / "scenario.json").write_text(json.dumps(scen))
    print(f"chosen: event {e['id']} {e['user']} {e['label_name']} at {e['speed_at_cross_mps']*3.6:.0f} km/h, "
          f"t_cross {e['t_cross_s']} s; flags {chosen['flags']}; left car {chosen['left_car']}")


if __name__ == "__main__":
    {"build": main_build, "predict": main_predict, "select": main_select}[sys.argv[1]]()
