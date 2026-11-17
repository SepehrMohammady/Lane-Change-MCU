"""Closed loop A: an ACC that anticipates the lane change of a neighbouring vehicle, on recorded highD scenes,
with the lane-change predictor in the loop (2026-11-17).

The supervisor's point (2026-11-12): replaying drives through the board is not hardware in the loop; the
board's output has to act on a vehicle whose state comes back through the board's inputs. Here the ego is
a simulated car (point mass, 25 Hz) driven by an ACC; every recorded vehicle of highD's test recordings
(56-60) is replayed; every 5th frame the 18 protocol features of one neighbouring vehicle (TV) are computed
from the replayed vehicles plus the ego (geometrically, as unas/sim_features.py) and go to the predictor
(an int8 network on the PC, or the STM32 board: unas/acc_loop_board.py); while the predictor says that the
TV moves into the ego's lane, the ACC treats it as a second leader with bounded braking. Design, arms,
measures and criterion fixed beforehand (local notes, 2026-11-17 12:54):

  ACC        the ACC model of Kesting, Treiber and Helbing (2010): IDM with the constant-acceleration heuristic
             (a 1.4, b 2.0 m/s2, delta 4, s0 2 m, c 0.99), desired speed v_set, the leader's recorded acceleration;
             command limited to [-8, +1.4] m/s2; a command below -3.5 m/s2 (ISO 15622) exceeds the ACC's
             authority. Anticipation: a TV in the adjacent lane whose rear is ahead of the ego's front is a
             second leader, its demand bounded at -1.5 m/s2; in the ego's lane it is an ordinary leader.
             (Amended 13:03 before any network ran: the linear gap law of the first design demanded more than
             3.5 m/s2 for any cut-in closer than about 40 m, even at equal speed.)
  arms       reactive (no anticipation); oracle (from 2 s before the recorded crossing, part C); a network:
             anticipation while the class of its latest window is the lane change toward the ego's lane.
  part C     scripted cut-ins: the protocol's 463 lane-change test scenarios with the TV tracked 3 s after the
             crossing; the ego in the target lane follows a scripted path relative to the TV, s_front = s_TV_rear
             - g - dv (t_c - t) at v_TV + dv, from 7 s before the crossing (the predictor's history) until 3 s
             before it, when the ACC takes over (v_set = its speed then, T = 1.5 s); target-lane vehicles behind
             the TV at the crossing are removed; an episode whose scripted path overlaps a kept vehicle is left
             out. Outcomes: collision, authority exceeded, deceleration to avoid a crash, minimum TTC.
  part O     recorded cut-outs: the TV leaves the ego's lane (every lane change of the test recordings with
             the TV in one lane for the 3 s before and tracked 3 s after); the ego replaces the TV's recorded
             follower E (E's position and speed 3 s before the crossing, T = E's time gap then, clipped to
             0.8-2.2 s, 1.5 s without a leader; v_set = E's highest speed in the window), to 3 s after the
             crossing. Unnecessary braking: after the crossing the anticipation is the binding term and the
             command is -0.5 m/s2 or lower, and the TV does not cross back within those 3 s.

  .venv\\Scripts\\python datasets/highd/acc_loop.py build      # Windows (pandas): scenes and episodes
  .venv\\Scripts\\python datasets/highd/acc_loop.py check      # features without ego against the prepared scenarios
  .venv\\Scripts\\python datasets/highd/acc_loop.py step1      # reactive and oracle over the part-C grid
  ~/dmir_nas/bin/python .../datasets/highd/acc_loop.py run     # WSL (TensorFlow): the P and K networks
  .venv\\Scripts\\python datasets/highd/acc_loop.py --report
Scenes, episodes and per-episode values stay in firmware/build/probe/ (not versioned: licensed data);
aggregates go to datasets/highd/results/hil/acc_loop*.json(l).
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREP = HERE / "data" / "prepared"
PROBE = ROOT / "firmware" / "build" / "probe"
EPISODES = PROBE / "acc_episodes.json"
RES = HERE / "results" / "hil"
OUT = RES / os.environ.get("ACC_OUT", "acc_loop.jsonl")
KEEP = HERE / "data" / "track_train"
MODEL = "model_1793711684616882545_453917"
TEST_RECS = (56, 57, 58, 59, 60)

FPS, DIV = 25, 5                                             # recording rate; every 5th frame for the predictor
DT = 1.0 / FPS
IN_LEN, D_DEFAULT = 10, 400.0
A_ACC, B_ACC, DELTA, S0, COOL = 1.4, 2.0, 4.0, 2.0, 0.99    # ACC model (Kesting et al. 2010), amended 13:03
RANGE, A_MIN, AUTHORITY, A_ANT = 150.0, -8.0, -3.5, -1.5
T_SCRIPT, T_FREE, T_LO, T_HI = 1.5, 1.5, 0.8, 2.2
EGO_LEN = 4.6
PRE, SWITCH, POST_C, POST_C_MIN = 7 * FPS, 3 * FPS, 5 * FPS, 3 * FPS
PRE_O, POST_O = 3 * FPS, 3 * FPS                             # part O, amended 12:58 (counts only)
UNNEEDED = -0.5                                              # m/s2: a braking command attributed to the anticipation
GAPS, CLOSING = (5.0, 10.0, 20.0, 30.0), (0.0, 3.0, 6.0)     # step 1 grid of part C
STEP1_SHARE = 0.20                                           # cells kept: reactive collision or takeover in >= 20%
HALF, KEEP_PTS, BENEFIT = 0.5, 0.02, 0.25                    # the criterion


# ---------------------------------------------------------------------------------------------------- scenes
def scene_file(rec: int) -> Path:
    return PROBE / f"acc_scene_{rec:02d}.npz"


def build():
    """Windows: per test recording, every vehicle row at 25 Hz in along-road coordinates with the lane index
    and lateral values of prepare_highd/sim_features; and the episodes of parts C and O."""
    import pandas as pd

    sys.path.insert(0, str(HERE))
    import cutin_headroom as ch
    import prepare_highd as ph

    PROBE.mkdir(parents=True, exist_ok=True)
    test = np.load(PREP / "test.npz")
    episodes = {"C": [], "O": []}
    for rec in TEST_RECS:
        p = ph.RAW / f"{rec:02d}"
        tr = pd.read_csv(f"{p}_tracks.csv", usecols=["frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity",
                                                     "xAcceleration", "yAcceleration", "laneId"])
        meta = pd.read_csv(f"{p}_recordingMeta.csv").iloc[0]
        direction = pd.read_csv(f"{p}_tracksMeta.csv").set_index("id").drivingDirection
        upper = np.array([float(v) for v in meta.upperLaneMarkings.split(";")])
        lower = np.array([float(v) for v in meta.lowerLaneMarkings.split(";")])
        d = tr.id.map(direction).to_numpy()
        x, w = tr.x.to_numpy(), tr.width.to_numpy()
        yc = (tr.y + tr.height / 2.0).to_numpy()
        lane = np.full(len(tr), -1)
        d_left = np.zeros(len(tr))
        width = np.zeros(len(tr))
        n_lanes = np.zeros(len(tr), int)
        for dd, marks in ((1, upper), (2, lower)):
            m = d == dd
            ind = np.array([max([k for k in range(len(marks)) if y > marks[k]], default=0) for y in yc[m]])
            ok = ind + 1 < len(marks)
            nl = len(marks) - 1
            idx = np.flatnonzero(m)
            lane[idx[ok]] = (ind[ok] if dd == 2 else nl - 1 - ind[ok])
            width[idx[ok]] = marks[ind[ok] + 1] - marks[ind[ok]]
            d_left[idx[ok]] = (np.abs(yc[m][ok] - marks[ind[ok]]) if dd == 2 else np.abs(yc[m][ok] - marks[ind[ok] + 1]))
            n_lanes[idx] = nl
        sgn = np.where(d == 2, 1.0, -1.0)
        s_rear = np.where(d == 2, x, -(x + w))
        s_front = np.where(d == 2, x + w, -x)
        order = np.lexsort((tr.id.to_numpy(), tr.frame.to_numpy(), d))
        np.savez_compressed(scene_file(rec), frame=tr.frame.to_numpy()[order], id=tr.id.to_numpy()[order], d=d[order],
                            s_rear=s_rear[order], s_front=s_front[order], length=w[order],
                            v=(sgn * tr.xVelocity.to_numpy())[order], a=(sgn * tr.xAcceleration.to_numpy())[order],
                            v_left=(-sgn * tr.yVelocity.to_numpy())[order], a_left=(-sgn * tr.yAcceleration.to_numpy())[order],
                            lane=lane[order], d_left=d_left[order], lane_width=width[order], n_lanes=n_lanes[order])
        # the episodes: the protocol's lane-change scenarios of this recording, in prepare_highd's order
        R = ph.Recording(rec)
        scen = ch.lc_scenarios(R)
        want = test["tv"][(test["rec"] == rec) & (test["label"] != 0)]
        assert [s[0] for s in scen] == want.tolist(), f"recording {rec}: scenarios differ from test.npz"
        R25 = ch.Rec25(ph.RAW, rec)
        for tid, f_c, lab, _ in scen:
            tT = R25.T[tid]
            if tT["first"] > f_c - PRE:
                continue
            episodes["C"].append({"rec": rec, "tv": int(tid), "f_c": int(f_c), "label": int(lab),
                                  "tv_last": int(tT["last"])})
        # part O: every crossing of every track; the TV's follower in its original lane at the last 5 Hz frame
        for tid, tv in R.T.items():
            cr, labs = ph.crossings(tv["laneId"], R.direction[tid])
            for c, lab in zip(cr, labs):
                f_c = int(tv["frame"][c])
                tT = R25.T[tid]
                if c - PRE_O // DIV < 0 or len(set(tv["laneId"][c - PRE_O // DIV:c].tolist())) > 1:
                    continue
                if tT["first"] > f_c - PRE_O or tT["last"] < f_c + POST_O:
                    continue
                E = int(tv["followingId"][c - 1])
                if E == 0 or E not in R25.T:
                    continue
                tE = R25.T[E]
                if tE["first"] > f_c - PRE_O or tE["last"] < f_c + POST_O:
                    continue
                lanes = tE["laneId"][f_c - PRE_O - tE["first"]:f_c + POST_O - tE["first"] + 1]
                if len(set(lanes.tolist())) > 1:
                    continue
                episodes["O"].append({"rec": rec, "tv": int(tid), "f_c": f_c, "label": int(lab), "ego_replaces": E,
                                      "tv_last": int(tT["last"])})
        print(f"recording {rec}: {len(scen)} lane changes; episodes so far C {len(episodes['C'])}, O {len(episodes['O'])}",
              flush=True)
    EPISODES.write_text(json.dumps(episodes))


class Scene:
    """One recording, one driving direction, rows grouped by frame."""

    def __init__(self, rec: int, d: int):
        z = np.load(scene_file(rec))
        m = z["d"] == d
        self.c = {k: z[k][m] for k in z.files if k != "d"}
        f = self.c["frame"]
        self.f0 = int(f.min())
        starts = np.searchsorted(f, np.arange(self.f0, int(f.max()) + 2))
        self.starts = starts
        self.point = self.c["s_rear"] if d == 2 else self.c["s_front"]     # highD's x: rear (d=2) / front (d=1)
        self.d = d

    def rows(self, frame: int) -> slice:
        k = frame - self.f0
        return slice(int(self.starts[k]), int(self.starts[k + 1]))

    def row_of(self, frame: int, vid: int) -> int | None:
        sl = self.rows(frame)
        hit = np.flatnonzero(self.c["id"][sl] == vid)
        return int(sl.start + hit[0]) if len(hit) else None


# ---------------------------------------------------------------------------------------------------- features
def tv_features(S: Scene, frame: int, tv: int, removed: set, ego=None) -> np.ndarray:
    """The 18 protocol features of the TV at one frame: replayed vehicles minus `removed`, plus the ego
    (dict with s_rear, length, v, lane) if given; neighbours as sim_features.frame_features."""
    sl = S.rows(frame)
    c = S.c
    ids = c["id"][sl]
    keep = np.array([int(i) not in removed for i in ids], bool) if removed else np.ones(len(ids), bool)
    rear, length = c["s_rear"][sl][keep], c["length"][sl][keep]
    v, lane, point, vid = c["v"][sl][keep], c["lane"][sl][keep], S.point[sl][keep], ids[keep]
    if ego is not None:
        rear = np.append(rear, ego["s_rear"])
        length = np.append(length, ego["length"])
        v = np.append(v, ego["v"])
        lane = np.append(lane, ego["lane"])
        point = np.append(point, ego["s_rear"] if S.d == 2 else ego["s_rear"] + ego["length"])
        vid = np.append(vid, -1)
    i = int(np.flatnonzero(vid == tv)[0])
    r = int(sl.start + np.flatnonzero(ids == tv)[0])
    front, centre = rear + length, rear + length / 2.0

    def same_lane(j):
        sm = np.flatnonzero((lane == lane[j]) & (np.arange(len(lane)) != j))
        ah, be = sm[centre[sm] > centre[j]], sm[centre[sm] < centre[j]]
        return (int(ah[np.argmin(centre[ah])]) if len(ah) else -1), (int(be[np.argmax(centre[be])]) if len(be) else -1)

    def side(step):
        o = np.flatnonzero(lane == lane[i] + step)
        if not len(o):
            return -1, -1, -1
        over = o[(rear[o] < front[i]) & (front[o] > rear[i])]
        bef, aft = o[rear[o] >= front[i]], o[front[o] <= rear[i]]
        along = int(over[np.argmin(np.abs(centre[over] - centre[i]))]) if len(over) else -1
        c1 = int(bef[np.argmin(rear[bef])]) if len(bef) else -1
        c2 = int(aft[np.argmax(front[aft])]) if len(aft) else -1
        if along >= 0:
            return along, c1, c2
        if c1 < 0 and c2 < 0:
            return -1, -1, -1
        if c1 < 0:
            vv = c2
        elif c2 < 0:
            vv = c1
        else:
            vv = c1 if abs(point[c1] - point[i]) < abs(point[c2] - point[i]) else c2
        pv_, fv_ = same_lane(vv)
        return (vv, pv_, -1) if vv == c1 else (vv, -1, fv_)

    def dist(j):
        return abs(point[j] - point[i]) if j >= 0 else D_DEFAULT

    def rvx(j):
        return -(v[j] - v[i]) if j >= 0 else 0.0

    pv, fv = same_lane(i)
    rv, rpv, rfv = side(+1)
    lv, lpv, lfv = side(-1)
    n_l = int(c["n_lanes"][r])
    return np.array([c["v_left"][r], -c["v"][r], c["a_left"][r], -c["a"][r], c["d_left"][r],
                     rvx(pv), dist(pv), rvx(fv), dist(fv), dist(rpv), dist(rv), dist(rfv), dist(lpv), dist(lv), dist(lfv),
                     0.0 if lane[i] == 0 else 1.0, 0.0 if lane[i] == n_l - 1 else 1.0, c["lane_width"][r]], np.float32)


def check():
    """Features with no ego and nothing removed against the prepared test scenarios (the 463 lane changes)."""
    test = np.load(PREP / "test.npz")
    feats, label = test["feats"], test["label"]
    eps = json.loads(EPISODES.read_text())["C"]
    lc = np.flatnonzero(label != 0)
    equal = np.zeros(18)
    total = 0
    scenes = {}
    k = 0
    for rec in TEST_RECS:
        idx = lc[test["rec"][lc] == rec]
        es = [e for e in eps if e["rec"] == rec]
        assert len(es) == len(idx) and all(test["tv"][s_i] == e["tv"] for s_i, e in zip(idx, es)), rec
        for s_i, e in zip(idx, es):
            d = direction_of(rec, e["tv"])
            S = scenes.setdefault((rec, d), Scene(rec, d))
            for n in range(35):
                f = e["f_c"] - PRE + DIV * n
                got = tv_features(S, f, e["tv"], set())
                want = feats[s_i, n]
                equal += np.abs(got - want) <= 1e-3 * np.maximum(1.0, np.abs(want))
                total += 1
            k += 1
    share = equal / max(total, 1)
    names = json.loads((PREP / "meta.json").read_text())["features"]
    res = {"scenarios": k, "frames": total, "equal_share": {n: round(float(s), 5) for n, s in zip(names, share)},
           "all_values_equal_share": round(float(equal.sum() / (18 * max(total, 1))), 5)}
    print(json.dumps(res, indent=1))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "acc_loop_feature_check.json").write_text(json.dumps(res, indent=1))
    return res


_DIR = {}


def direction_of(rec: int, tv: int) -> int:
    if rec not in _DIR:
        z = np.load(scene_file(rec))
        _DIR[rec] = dict(zip(z["id"].tolist(), z["d"].tolist()))
    return int(_DIR[rec][tv])


# ---------------------------------------------------------------------------------------------------- the loop
def leader(S: Scene, frame: int, lane: int, s_front: float, removed: set):
    """Nearest replayed vehicle in the lane whose front is ahead of s_front: (gap to its rear, its speed, its
    acceleration, id) or None."""
    sl = S.rows(frame)
    c = S.c
    m = (c["lane"][sl] == lane) & (c["s_front"][sl] > s_front)
    if removed:
        m &= np.array([int(i) not in removed for i in c["id"][sl]], bool)
    if not m.any():
        return None
    j = np.flatnonzero(m)[np.argmin(c["s_rear"][sl][m])]
    return float(c["s_rear"][sl][j] - s_front), float(c["v"][sl][j]), float(c["a"][sl][j]), int(c["id"][sl][j])


def acc_free(v: float, v0: float) -> float:
    return A_ACC * (1.0 - (v / max(v0, 1.0)) ** DELTA)


def acc_follow(s: float, v: float, v_l: float, a_l: float, v0: float, T: float) -> float:
    """The ACC model of Kesting, Treiber and Helbing (2010): IDM with the constant-acceleration heuristic."""
    s = max(s, 0.1)
    dv = v - v_l
    s_star = S0 + max(0.0, v * T + v * dv / (2.0 * np.sqrt(A_ACC * B_ACC)))
    a_idm = A_ACC * (1.0 - (v / max(v0, 1.0)) ** DELTA - (s_star / s) ** 2)
    al = min(a_l, A_ACC)
    if v_l * dv <= -2.0 * s * al:
        a_cah = v * v * al / (v_l * v_l - 2.0 * s * al) if v_l * v_l - 2.0 * s * al != 0 else al
    else:
        a_cah = al - (dv * dv if dv > 0 else 0.0) / (2.0 * s)
    if a_idm >= a_cah:
        return a_idm
    return (1.0 - COOL) * a_idm + COOL * (a_cah + B_ACC * np.tanh((a_idm - a_cah) / B_ACC))


def setup(S: Scene, part: str, e: dict, g: float = 0.0, dv: float = 0.0):
    """Initial ego state, lane, set speed, time gap, removed vehicles and the episode's frames. Part C: the
    scripted path before the ACC takes over (None if it overlaps a vehicle kept in the ego's lane)."""
    f_c, tv = e["f_c"], e["tv"]
    if part == "C":
        f0, f_sw = f_c - PRE, f_c - SWITCH
        rT = S.row_of(f_c, tv)
        lane = int(S.c["lane"][rT])
        removed = set()                                      # in the target lane before the crossing, behind the TV at it
        for f in range(f0, f_c + 1, DIV):
            sl = S.rows(f)
            for vid, ln in zip(S.c["id"][sl], S.c["lane"][sl]):
                if ln == lane and int(vid) != tv and int(vid) not in removed:
                    r = S.row_of(f_c, int(vid))
                    if r is not None and S.c["s_rear"][r] < S.c["s_rear"][rT]:
                        removed.add(int(vid))
        path = {}
        for f in range(f0, f_sw + 1):
            r = S.row_of(f, tv)
            s_front = float(S.c["s_rear"][r]) - g - dv * (f_c - f) * DT
            path[f] = (s_front, float(S.c["v"][r]) + dv)
            sl = S.rows(f)
            m = (S.c["lane"][sl] == lane) & (S.c["s_rear"][sl] < s_front) & (S.c["s_front"][sl] > s_front - EGO_LEN)
            if m.any() and any(int(i) not in removed and int(i) != tv for i in S.c["id"][sl][m]):
                return None                                  # the scripted ego would overlap a vehicle of its lane
        s0, v0 = path[f_sw]
        f_end = min(f_c + POST_C, e["tv_last"])
        return dict(s_front=s0, v=v0, v_set=v0, T=T_SCRIPT, lane=lane, length=EGO_LEN, removed=removed,
                    f0=f0, f_sw=f_sw, f_end=f_end, path=path)
    E = e["ego_replaces"]
    f0 = f_c - PRE_O
    rE = S.row_of(f0, E)
    lane = int(S.c["lane"][rE])
    s_front0, v0 = float(S.c["s_front"][rE]), float(S.c["v"][rE])
    f_end = f_c + POST_O
    vmax = max(float(S.c["v"][S.row_of(f, E)]) for f in range(f0, f_end + 1, DIV))
    lead = leader(S, f0, lane, s_front0, {E})
    T = (min(T_HI, max(T_LO, (lead[0] - S0) / max(v0, 1.0))) if lead and lead[0] <= RANGE else T_FREE)
    return dict(s_front=s_front0, v=v0, v_set=vmax, T=T, lane=lane, length=float(S.c["length"][rE]), removed={E},
                f0=f0, f_sw=f0, f_end=f_end, path={})


def toward(ego_lane: int, tv_lane: int) -> int:
    """The class that means 'the TV moves into the ego's lane' (1 right, 2 left; lane 0 = leftmost), or 0."""
    return 2 if ego_lane == tv_lane - 1 else 1 if ego_lane == tv_lane + 1 else 0


def episode(S: Scene, part: str, e: dict, arm: str, predictor=None, g: float = 0.0, dv: float = 0.0,
            clock=None):
    """One closed-loop run, or None if part C's scripted path is not free. predictor(features (18,), first)
    -> class or None (window not full); its attribute last_out (the int8 outputs of the latest window) is
    kept per decision. clock(step) is called before every 25 Hz step (real-time pacing)."""
    p = setup(S, part, e, g, dv)
    if p is None:
        return None
    tv, f_c = e["tv"], e["f_c"]
    s_front, v, lane, T = p["s_front"], p["v"], p["lane"], p["T"]
    removed = p["removed"]
    cls, first = None, True
    classes, raw = [], []
    out = {"collision": False, "authority_exceeded": False, "peak_decel": 0.0, "max_drac": 0.0,
           "min_ttc": float("inf"), "min_gap": float("inf"), "unneeded_braking": False,
           "anticipation_s_after": 0.0, "anticipation_s_before": 0.0}
    returns = False
    for f in range(p["f0"], p["f_end"] + 1):
        if clock is not None:
            clock(f - p["f0"])
        scripted = f < p["f_sw"]
        if scripted:
            s_front, v = p["path"][f]
        rT = S.row_of(f, tv)
        tv_lane = int(S.c["lane"][rT]) if rT is not None else None
        if f > f_c and tv_lane == lane and part == "O":
            returns = True                                   # the TV came back into the ego's lane
        if predictor is not None and (f - p["f0"]) % DIV == 0 and rT is not None:
            ego = {"s_rear": s_front - p["length"], "length": p["length"], "v": v, "lane": lane}
            cls = predictor(tv_features(S, f, tv, removed, ego), first)
            first = False
            classes.append(-1 if cls is None else int(cls))
            if cls is not None and getattr(predictor, "last_out", None) is not None:
                raw.append([int(x) for x in predictor.last_out])
        if scripted:
            continue
        lead = leader(S, f, lane, s_front, removed)
        binding = "cruise"
        a = acc_free(v, p["v_set"])
        if lead and lead[0] <= RANGE:
            gap, v_l, a_l = lead[0], lead[1], lead[2]
            al = acc_follow(gap, v, v_l, a_l, p["v_set"], T)
            if al < a:
                a, binding = al, "leader"
            out["min_gap"] = min(out["min_gap"], gap)
            if gap <= 0:
                out["collision"] = True
            elif v > v_l:
                out["max_drac"] = max(out["max_drac"], (v - v_l) ** 2 / (2.0 * gap))
                if f >= f_c:
                    out["min_ttc"] = min(out["min_ttc"], gap / (v - v_l))
        anticipate = False
        if rT is not None and tv_lane is not None and abs(tv_lane - lane) == 1 and S.c["s_rear"][rT] > s_front:
            if arm == "oracle":
                anticipate = part == "C" and p["f_sw"] <= f < f_c
            elif arm != "reactive":
                anticipate = cls is not None and cls == toward(lane, tv_lane)
        if anticipate:
            gap_t = float(S.c["s_rear"][rT]) - s_front
            at = max(A_ANT, acc_follow(gap_t, v, float(S.c["v"][rT]), float(S.c["a"][rT]), p["v_set"], T))
            if at < a:
                a, binding = at, "anticipation"
            if f >= f_c:
                out["anticipation_s_after"] += DT
            else:
                out["anticipation_s_before"] += DT
        if a < AUTHORITY:
            out["authority_exceeded"] = True
        a = float(np.clip(a, A_MIN, A_ACC))
        if part == "O" and f >= f_c and binding == "anticipation" and a <= UNNEEDED:
            out["unneeded_braking"] = True
        out["peak_decel"] = max(out["peak_decel"], -a)
        v = max(0.0, v + a * DT)
        s_front += v * DT
    out["returns"] = returns
    if part == "O" and returns:
        out["unneeded_braking"] = False                      # counted apart
    out["end_speed"] = v
    out["decisions"] = classes
    out["outputs"] = raw
    return out


def summarize(part: str, rows: list) -> dict:
    left_out = sum(r is None for r in rows)
    rows = [r for r in rows if r is not None]
    n = len(rows)
    if part == "C":
        bad = sum(r["collision"] or r["authority_exceeded"] for r in rows)
        ttc = [r["min_ttc"] for r in rows if np.isfinite(r["min_ttc"])]
        return {"episodes": n, "left_out_scripted_path_not_free": left_out, "collision_or_authority_exceeded": bad,
                "share": round(bad / max(n, 1), 5), "collisions": sum(r["collision"] for r in rows),
                "authority_exceeded": sum(r["authority_exceeded"] for r in rows),
                "max_drac_mean": round(float(np.mean([r["max_drac"] for r in rows])), 4) if n else None,
                "max_drac_over_3.5": sum(r["max_drac"] > 3.5 for r in rows),
                "peak_decel_mean": round(float(np.mean([r["peak_decel"] for r in rows])), 4) if n else None,
                "min_ttc_p10": round(float(np.percentile(ttc, 10)), 3) if ttc else None}
    kept = [r for r in rows if not r["returns"]]
    un = sum(r["unneeded_braking"] for r in kept)
    return {"episodes": n, "tv_comes_back": n - len(kept), "unneeded_braking": un, "share": round(un / max(len(kept), 1), 5),
            "anticipation_s_after_mean": round(float(np.mean([r["anticipation_s_after"] for r in kept])), 4) if kept else None,
            "end_speed_mean": round(float(np.mean([r["end_speed"] for r in kept])), 4) if kept else None}


def episodes_of(part: str) -> list:
    return json.loads(EPISODES.read_text())[part]


def scenes_for(eps: list) -> dict:
    out = {}
    for e in eps:
        key = (e["rec"], direction_of(e["rec"], e["tv"]))
        if key not in out:
            out[key] = Scene(*key)
    return out


def step1():
    """Part C without networks: reactive and oracle over the grid; the cells for the study."""
    eps = episodes_of("C")
    eps = [e for e in eps if e["tv_last"] >= e["f_c"] + POST_C_MIN]
    sc = scenes_for(eps)
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "episodes": len(eps), "cells": []}
    for g in GAPS:
        for dv in CLOSING:
            cell = {"gap_m": g, "closing_m_s": dv}
            for arm in ("reactive", "oracle"):
                rows = [episode(sc[(e["rec"], direction_of(e["rec"], e["tv"]))], "C", e, arm, None, g, dv) for e in eps]
                cell[arm] = summarize("C", rows)
            res["cells"].append(cell)
            print(f"gap {g:4.0f} m, closing {dv:3.0f} m/s: {cell['reactive']['episodes']} episodes; reactive "
                  f"{cell['reactive']['share']:.3f} (coll {cell['reactive']['collisions']}), oracle {cell['oracle']['share']:.3f} "
                  f"(coll {cell['oracle']['collisions']}); DRAC>3.5 {cell['reactive']['max_drac_over_3.5']} / "
                  f"{cell['oracle']['max_drac_over_3.5']}", flush=True)
    res["kept_cells"] = [[c["gap_m"], c["closing_m_s"]] for c in res["cells"] if c["reactive"]["share"] >= STEP1_SHARE]
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "acc_loop_step1.json").write_text(json.dumps(res, indent=1))
    print("kept cells:", res["kept_cells"])


# ---------------------------------------------------------------------------------------------------- networks
def scaling():
    raw = np.load(PREP / "train.npz")["feats"].astype(np.float32)
    lo = raw.reshape(-1, raw.shape[-1]).min(0)
    return lo, raw.reshape(-1, raw.shape[-1]).max(0) - lo + 1e-12


class TFLitePredictor:
    """The int8 network with the board's arithmetic: float32 scaling, round half to even, saturation."""

    def __init__(self, path: Path, lo, span):
        import tensorflow as tf

        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.inp, self.out = self.it.get_input_details()[0], self.it.get_output_details()[0]
        self.si, self.zi = self.inp["quantization"]
        self.lo, self.span = lo, span
        self.win = []
        self.last_out = None

    def __call__(self, x: np.ndarray, first: bool):
        if first:
            self.win = []
        q = np.clip(np.rint(((x - self.lo) / self.span) / np.float32(self.si) + np.float32(self.zi)), -128, 127).astype(np.int8)
        self.win = (self.win + [q])[-IN_LEN:]
        if len(self.win) < IN_LEN:
            return None
        self.it.set_tensor(self.inp["index"], np.stack(self.win)[None])
        self.it.invoke()
        o = self.it.get_tensor(self.out["index"]).reshape(-1)
        self.last_out = o.copy()
        return int(o.astype(np.int16).argmax())


def cells_kept() -> list:
    return [tuple(c) for c in json.loads((RES / "acc_loop_step1.json").read_text())["kept_cells"]]


def run_arm(arm: str, predictor_of, label: str, keep_outputs: Path | None = None):
    """Both parts for one arm; one record per part (and cell) to OUT."""
    eps_c = [e for e in episodes_of("C") if e["tv_last"] >= e["f_c"] + POST_C_MIN]
    eps_o = episodes_of("O")
    sc = scenes_for(eps_c + eps_o)
    rec = {"arm": arm, "network": label, "utc": None, "parts": {}}
    per = {}
    pr = predictor_of() if predictor_of else None            # one predictor; its window restarts with every episode
    for g, dv in cells_kept():
        rows = []
        for e in eps_c:
            rows.append(episode(sc[(e["rec"], direction_of(e["rec"], e["tv"]))], "C", e, arm, pr, g, dv))
        rec["parts"][f"C_gap{g:g}_dv{dv:g}"] = summarize("C", rows)
        per[f"C_gap{g:g}_dv{dv:g}"] = rows
    rows = []
    for e in eps_o:
        rows.append(episode(sc[(e["rec"], direction_of(e["rec"], e["tv"]))], "O", e, arm, pr))
    rec["parts"]["O"] = summarize("O", rows)
    per["O"] = rows
    c_rows = [r for k, v in per.items() if k.startswith("C") for r in v]
    rec["parts"]["C_all_kept_cells"] = summarize("C", c_rows)
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if keep_outputs is not None:
        keep_outputs.parent.mkdir(parents=True, exist_ok=True)
        keep_outputs.write_text(json.dumps(per, default=float))
    return rec


def run():
    """WSL: reactive, oracle and the ten networks on the PC."""
    lo, span = scaling()
    done = {(r["arm"], r["network"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
    jobs = [("reactive", None, "none"), ("oracle", None, "none")]
    for arm in ("P", "K"):
        for seed in (100, 101, 102, 103, 104):
            jobs.append((arm, KEEP / f"{arm}_{MODEL}_seed{seed}_int8_io.tflite", f"{arm}_seed{seed}"))
    for arm, path, label in jobs:
        if (arm, label) in done:
            continue
        maker = (lambda p=path: TFLitePredictor(p, lo, span)) if path is not None else None
        r = run_arm(arm, maker, label, PROBE / f"acc_loop_pc_{label if path else arm}.json")
        print(f"{datetime.now():%H:%M} {arm} {label}: C {r['parts']['C_all_kept_cells']['share']:.3f}, "
              f"O {r['parts']['O']['share']:.3f}", flush=True)


def report():
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)

    def mean(arm, part):
        vals = [r["parts"][part]["share"] for r in by.get(arm, [])]
        return (round(st.mean(vals), 5), round(st.stdev(vals), 5) if len(vals) > 1 else None, vals) if vals else None
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cells": cells_kept(), "arms": {}}
    for arm in by:
        res["arms"][arm] = {"networks": [r["network"] for r in by[arm]],
                            "C_collision_or_takeover_share": mean(arm, "C_all_kept_cells"),
                            "O_unneeded_braking_share": mean(arm, "O"),
                            "parts": {r["network"]: r["parts"] for r in by[arm]}}
    if all(a in by for a in ("P", "K", "reactive")):
        P_o, K_o = mean("P", "O")[0], mean("K", "O")[0]
        P_c, K_c, R_c = mean("P", "C_all_kept_cells")[0], mean("K", "C_all_kept_cells")[0], mean("reactive", "C_all_kept_cells")[0]
        a_ok = K_o <= HALF * P_o + 1e-12
        b_ok = K_c <= P_c + KEEP_PTS + 1e-12 and P_c <= (1 - BENEFIT) * R_c + 1e-12 and K_c <= (1 - BENEFIT) * R_c + 1e-12
        res["criterion"] = {"rule": "(a) part O unnecessary braking: K at most half of P; (b) part C collision or takeover: K at most "
                                    "2 points above P, both at least 25% below reactive",
                            "a_O_P_K": [P_o, K_o], "a_met": bool(a_ok),
                            "b_C_reactive_P_K": [R_c, P_c, K_c], "b_met": bool(b_ok), "met": bool(a_ok and b_ok)}
    (RES / "acc_loop.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))
    for arm, a in res["arms"].items():
        print(arm, "C", a["C_collision_or_takeover_share"], "O", a["O_unneeded_braking_share"])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"build": build, "check": check, "step1": step1, "run": run, "--report": report}.get(
        step, lambda: sys.exit("build | check | step1 | run | --report"))()
