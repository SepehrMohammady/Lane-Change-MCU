"""exiD -> lane-change scenarios in the EarlyLCPred format used for highD.

exiD (Moers et al., IEEE IV 2022) was recorded with the highD drone method at
entries and exits of German motorways. This script writes it in the scenario
format of datasets/highd/prepare_highd.py, so the same models, training code and
metrics apply to both datasets and a model trained on one can be tested on the
other.

Kept from the highD protocol (Mozaffari et al., T-IV 2022):
- 25 Hz tracks downsampled to 5 Hz (frame % 5 == 0)
- LC scenario = 35 frames (7 s) ending at the first 5 Hz frame in the new lane,
  valid only if the vehicle crosses no other lane boundary in those frames
- labels 0 = lane keep, 1 = right LC, 2 = left LC
- one LK scenario per track, ending 25 frames before its first crossing (or the
  end of the track), undersampled to (RLC + LLC) / 2 per recording and
  preferring tracks with a later crossing
- the 18 highD features in the same units and signs: lateral quantities
  positive to the left, longitudinal velocity and acceleration negated as in
  the highD code, 400 m / 0 m/s for an absent vehicle

Adapted to exiD, whose roads curve and whose lanes merge and split:
- crossings come from exiD's laneChange flag, the first frame with the centroid
  in the new lanelet. The side is the side of the old lane the vehicle left
  through (sign of latLaneCenterOffset just before the flag); on the checked
  recordings this agrees with the Lanelet2 adjacency for every crossing.
- lateral velocity and acceleration are the first and second time derivatives
  of the lateral position relative to the lane centreline (latLaneCenterOffset
  with the step at each crossing or centreline switch removed), Savitzky-Golay
  filtered over 2 s. The map centrelines are polylines, and a vehicle on a
  curve sees their vertices as small offset steps every 1.6-2 s; a 1-s filter
  passes them into the acceleration. With 2 s, lateral velocity and
  acceleration in the last 3 s before a crossing match highD (medians 0.57 and
  0.21 against 0.59 m/s and 0.21 m/s2 on the training splits); the window was
  chosen on that comparison, not on model accuracy. exiD's latVelocity is given in the
  vehicle frame (side slip), so it does not show lateral motion across the lane.
- longitudinal velocity and acceleration: exiD lonVelocity and lonAcceleration
- distance to a surrounding vehicle: centre-to-centre distance projected on the
  heading of the target vehicle (highD: difference of x on a straight road)
- lane existence: a same-direction neighbouring lanelet of subtype highway
  that is not separated by a road border (hard shoulders do not count)
- crossings into or out of a hard shoulder (emergency_lane) are not used as
  scenarios, and no scenario may contain frames on a hard shoulder; highD has
  no shoulder lane in its lane markings
- lane-change kind from the Lanelet2 tags: merge (leaving an on-ramp lanelet),
  exit (entering an off-ramp lanelet), on a ramp (both lanelets on ramps), or
  main carriageway
- split by recording within each location: the latest recordings whose total
  duration is closest to 10% of that location's recorded time form the test
  set, the next ones the validation set, and the rest the training set

Output: datasets/exid/data/prepared/{train,val,test}.npz with the highD keys
  feats (S,35,18) f32 | label (S,) i8 | cross_idx (S,) i16 | rec (S,) i16 | tv (S,) i32
plus loc (S,) i8 | kind (S,) i8 | cls (S,) i8 | cross_frame (S,) i32 (25 Hz frame
of the laneChange flag, -1 for lane keeping), meta.json and index.csv.

Run: .venv\\Scripts\\python datasets/exid/prepare_exid.py
"""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "prepared"

FPS = 25
FPS_DIV = 5                 # 25 Hz -> 5 Hz
SEQ_LEN = 35
PRED_LEN = 25
SG_WINDOW, SG_ORDER = 51, 2  # 2 s at 25 Hz, for lateral velocity and acceleration
STEP = 0.15                  # m; a larger step of the offset at a lanelet change is a centreline switch
TEST_SHARE = 0.10
D_DEFAULT = 400.0

FEATURE_NAMES = [
    "lat_velocity", "long_velocity", "lat_acceleration", "long_acceleration",
    "lat_pos_left_marking", "rel_velo_pv", "dist_pv", "rel_velo_fv", "dist_fv",
    "dist_rpv", "dist_rv", "dist_rfv", "dist_lpv", "dist_lv", "dist_lfv",
    "left_lane_exists", "right_lane_exists", "lane_width",
]
KIND_NAMES = ["lane keep", "main carriageway", "merge from on-ramp", "exit to off-ramp", "on a ramp"]
CLASS_CODES = {"car": 0, "van": 1, "truck": 2, "motorcycle": 3}

COLS = ["trackId", "frame", "xCenter", "yCenter", "heading", "lonVelocity", "lonAcceleration",
        "latLaneCenterOffset", "laneWidth", "laneletId", "laneChange",
        "leadId", "rearId", "leftLeadId", "leftRearId", "leftAlongsideId",
        "rightLeadId", "rightRearId", "rightAlongsideId"]
ID_COLS = {"leadId": "lead", "rearId": "rear", "leftLeadId": "lL", "leftRearId": "lR",
           "rightLeadId": "rL", "rightRearId": "rR"}


# ------------------------------------------------------------------ Lanelet2 map
def read_map(loc: int) -> dict[int, dict]:
    """Lanelets of one location with neighbours, lane existence, ramp and shoulder flags."""
    path = next((RAW / "maps" / "lanelet2").glob(f"{loc}_*/*.osm"))
    root = ET.parse(path).getroot()
    ways = {int(w.get("id")): {t.get("k"): t.get("v") for t in w.findall("tag")} for w in root.iter("way")}
    ll: dict[int, dict] = {}
    for r in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        mem = {m.get("role"): int(m.get("ref")) for m in r.findall("member")}
        ll[int(r.get("id"))] = {"left": mem["left"], "right": mem["right"],
                                "subtype": tags.get("subtype", ""),
                                "onramp": tags.get("onramp") == "yes",
                                "offramp": tags.get("offramp") == "yes"}
    by_right: dict[int, list[int]] = {}
    by_left: dict[int, list[int]] = {}
    for i, lan in ll.items():
        by_right.setdefault(lan["right"], []).append(i)
        by_left.setdefault(lan["left"], []).append(i)
    for i, lan in ll.items():
        lan["left_nb"] = [j for j in by_right.get(lan["left"], []) if j != i]    # j.right is i.left
        lan["right_nb"] = [j for j in by_left.get(lan["right"], []) if j != i]   # j.left is i.right
        lan["left_lane"] = float(ways.get(lan["left"], {}).get("type") != "road_border"
                                 and any(ll[j]["subtype"] == "highway" for j in lan["left_nb"]))
        lan["right_lane"] = float(ways.get(lan["right"], {}).get("type") != "road_border"
                                  and any(ll[j]["subtype"] == "highway" for j in lan["right_nb"]))
        lan["shoulder"] = lan["subtype"] == "emergency_lane"
        lan["ramp"] = lan["onramp"] or lan["offramp"]
    return ll


def lc_kind(old: dict | None, new: dict | None) -> int:
    if old is None or new is None:
        return 1
    if old["onramp"] and not new["onramp"]:
        return 2
    if new["offramp"] and not old["offramp"]:
        return 3
    if old["ramp"] and new["ramp"]:
        return 4
    return 1


# ------------------------------------------------------------------ recordings
def first_value(s: pd.Series) -> np.ndarray:
    """First entry of exiD's semicolon lists (a vehicle can be in two lanelets)."""
    if s.dtype.kind in "if":
        return s.to_numpy(dtype=np.float64)
    return pd.to_numeric(s.astype(str).str.split(";").str[0], errors="coerce").to_numpy(dtype=np.float64)


def id_list(v) -> list[int]:
    if isinstance(v, str) and v.strip():
        return [int(float(x)) for x in v.split(";") if x.strip() != ""]
    if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v) and v >= 0:
        return [int(v)]
    return []


class Recording:
    """One recording at 5 Hz with lane-relative lateral kinematics and O(1) lookups."""

    def __init__(self, rec_id: int, lanelets: dict[int, dict], classes: dict[int, str]):
        self.rec_id = rec_id
        self.ll = lanelets
        self.cls = classes
        t = pd.read_csv(RAW / "data" / f"{rec_id:02d}_tracks.csv", usecols=COLS, low_memory=False)
        t = t.sort_values(["trackId", "frame"]).reset_index(drop=True)

        tid = t.trackId.to_numpy()
        off = first_value(t.latLaneCenterOffset)
        lane = first_value(t.laneletId)
        new_track = np.r_[True, tid[1:] != tid[:-1]]

        # continuous lateral position: remove the step of the offset where the lanelet changes
        d = np.diff(off, prepend=np.nan)
        d[new_track] = 0.0
        lane_switch = np.r_[False, lane[1:] != lane[:-1]] & ~new_track
        steps = lane_switch & (np.abs(d) > STEP)
        d[steps] = np.nan
        ds = pd.Series(d).groupby(tid).transform(lambda s: s.interpolate(limit_direction="both")).fillna(0.0)
        ypos = ds.groupby(tid).cumsum().to_numpy()
        vlat = np.zeros(len(t))
        alat = np.zeros(len(t))
        for idx in pd.Series(np.arange(len(t))).groupby(tid).indices.values():
            if len(idx) >= SG_WINDOW:
                y = ypos[idx]
                vlat[idx] = savgol_filter(y, SG_WINDOW, SG_ORDER, deriv=1, delta=1.0 / FPS)
                alat[idx] = savgol_filter(y, SG_WINDOW, SG_ORDER, deriv=2, delta=1.0 / FPS)

        # crossings at 25 Hz, before downsampling can drop the flagged frame
        lc = (t.laneChange.to_numpy() == 1) & ~new_track
        self.crossings: dict[int, list[tuple[int, int, int, bool]]] = {}
        for i in np.nonzero(lc)[0]:
            old, new = self.ll.get(int(lane[i - 1])), self.ll.get(int(lane[i]))
            side = 2 if off[i - 1] > 0 else 1
            shoulder = bool((old and old["shoulder"]) or (new and new["shoulder"]))
            self.crossings.setdefault(int(tid[i]), []).append(
                (int(t.frame.iat[i]), side, lc_kind(old, new), shoulder))

        keep = (t.frame.to_numpy() % FPS_DIV) == 0
        t = t[keep].reset_index(drop=True)
        vlat, alat, off, lane = vlat[keep], alat[keep], off[keep], lane[keep]
        width = first_value(t.laneWidth)
        self.T: dict[int, dict] = {}
        for track, idx in t.groupby("trackId").indices.items():
            g = t.iloc[idx]
            rec = {"frame": g.frame.to_numpy(), "x": g.xCenter.to_numpy(), "y": g.yCenter.to_numpy(),
                   "h": np.deg2rad(g.heading.to_numpy()), "vlon": g.lonVelocity.to_numpy(),
                   "alon": g.lonAcceleration.to_numpy(), "vlat": vlat[idx], "alat": alat[idx],
                   "off": off[idx], "width": width[idx], "lane": lane[idx]}
            for col, key in ID_COLS.items():
                rec[key] = pd.to_numeric(g[col], errors="coerce").fillna(-1).astype(int).to_numpy()
            rec["lA"] = [id_list(v) for v in g.leftAlongsideId.to_numpy()]
            rec["rA"] = [id_list(v) for v in g.rightAlongsideId.to_numpy()]
            rec["first"] = int(rec["frame"][0])
            self.T[int(track)] = rec

    def row(self, tid: int, frame: int) -> int | None:
        t = self.T.get(tid)
        if t is None:
            return None
        i = (frame - t["first"]) // FPS_DIV
        if i < 0 or i >= len(t["frame"]) or t["frame"][i] != frame:
            return None
        return int(i)

    def dist(self, tv: dict, i: int, vid: int, frame: int) -> float:
        """Centre distance to vid projected on the target vehicle's heading."""
        r = self.row(vid, frame) if vid >= 0 else None
        if r is None:
            return D_DEFAULT
        v = self.T[vid]
        h = tv["h"][i]
        return float(abs((v["x"][r] - tv["x"][i]) * np.cos(h) + (v["y"][r] - tv["y"][i]) * np.sin(h)))

    def rel_speed(self, tv: dict, i: int, vid: int, frame: int) -> float:
        """Target speed minus the other vehicle's speed (highD: sgn * (vx_other - vx_tv))."""
        r = self.row(vid, frame) if vid >= 0 else None
        return float(tv["vlon"][i] - self.T[vid]["vlon"][r]) if r is not None else 0.0

    def _side(self, tv: dict, i: int, frame: int, along: str, lead: str, rear: str):
        """(v, pv, fv) on one side, the highD alongside-fallback rule."""
        alongside = [a for a in tv[along][i] if self.row(a, frame) is not None]
        if alongside:
            a = min(alongside, key=lambda v: self.dist(tv, i, v, frame))
            return a, int(tv[lead][i]), int(tv[rear][i])
        c1, c2 = int(tv[lead][i]), int(tv[rear][i])
        if c1 < 0 and c2 < 0:
            return -1, -1, -1
        if c1 < 0:
            v = c2
        elif c2 < 0:
            v = c1
        else:
            v = c1 if self.dist(tv, i, c1, frame) < self.dist(tv, i, c2, frame) else c2
        r = self.row(v, frame)
        if r is None:
            return -1, -1, -1
        vd = self.T[v]
        if v == c1:
            return v, int(vd["lead"][r]), -1
        return v, -1, int(vd["rear"][r])

    def features(self, tid: int, first: int, last: int) -> np.ndarray | None:
        tv = self.T[tid]
        out = np.zeros((last - first, 18), dtype=np.float32)
        for k, i in enumerate(range(first, last)):
            frame = int(tv["frame"][i])
            lan = self.ll.get(int(tv["lane"][i])) if np.isfinite(tv["lane"][i]) else None
            w = tv["width"][i]
            if lan is None or lan["shoulder"] or not np.isfinite(w) or not np.isfinite(tv["off"][i]):
                return None
            pv, fv = int(tv["lead"][i]), int(tv["rear"][i])
            rv, rpv, rfv = self._side(tv, i, frame, "rA", "rL", "rR")
            lv, lpv, lfv = self._side(tv, i, frame, "lA", "lL", "lR")
            out[k] = (
                tv["vlat"][i], -tv["vlon"][i], tv["alat"][i], -tv["alon"][i],
                w / 2.0 - tv["off"][i],
                self.rel_speed(tv, i, pv, frame), self.dist(tv, i, pv, frame),
                self.rel_speed(tv, i, fv, frame), self.dist(tv, i, fv, frame),
                self.dist(tv, i, rpv, frame), self.dist(tv, i, rv, frame), self.dist(tv, i, rfv, frame),
                self.dist(tv, i, lpv, frame), self.dist(tv, i, lv, frame), self.dist(tv, i, lfv, frame),
                lan["left_lane"], lan["right_lane"], w,
            )
        return out


def extract_recording(rec_id: int, lanelets: dict, classes: dict):
    rec = Recording(rec_id, lanelets, classes)
    lc, lk_known, lk_unknown = [], [], []
    drop = {"shoulder": 0, "too early": 0, "other crossing": 0, "invalid": 0}
    for tid, tv in rec.T.items():
        frames = tv["frame"]
        cr = []
        for f_c, side, kind, shoulder in rec.crossings.get(tid, []):
            c = int(np.searchsorted(frames, f_c))       # first 5 Hz frame in the new lane
            if c < len(frames):
                cr.append((c, f_c, side, kind, shoulder))
        cr.sort()
        idxs = [c for c, *_ in cr]
        cls = CLASS_CODES.get(classes.get(tid, "car"), 0)

        for c, f_c, side, kind, shoulder in cr:
            if shoulder:
                drop["shoulder"] += 1
                continue
            first = c - SEQ_LEN
            if first < 0:
                drop["too early"] += 1
                continue
            if any(first < c2 < c for c2 in idxs):
                drop["other crossing"] += 1
                continue
            f = rec.features(tid, first, c)
            if f is None:
                drop["invalid"] += 1
                continue
            lc.append((f, side, SEQ_LEN, tid, kind, cls, f_c))

        # one LK scenario per track (their get_lk_scenarios)
        if idxs:
            last, known = idxs[0] - PRED_LEN, True
        else:
            last, known = len(frames) - PRED_LEN, False
        first = last - SEQ_LEN
        if first < 0:
            continue
        f = rec.features(tid, first, last)
        if f is None:
            continue
        (lk_known if known else lk_unknown).append((f, 0, SEQ_LEN + PRED_LEN if known else -1, tid, 0, cls, -1))

    rlc = sum(1 for s in lc if s[1] == 1)
    llc = sum(1 for s in lc if s[1] == 2)
    lk_count = (rlc + llc) // 2
    lk = lk_known[:lk_count]
    if len(lk) < lk_count:
        lk += lk_unknown[:lk_count - len(lk)]
    return lc + lk, rlc, llc, len(lk), drop


# ------------------------------------------------------------------ splits
def make_splits(meta: pd.DataFrame) -> dict[str, list[int]]:
    """Per location, the latest recordings closest to 10% of its time -> test, then val."""
    out = {"train": [], "val": [], "test": []}
    for _, g in meta.sort_values("rec").groupby("loc"):
        recs, durs = list(g.rec), list(g.duration)
        total = sum(durs)
        for split in ("test", "val"):
            best_k = min(range(1, len(recs) - 1),
                         key=lambda k: abs(sum(durs[-k:]) / total - TEST_SHARE))
            out[split] += recs[-best_k:]
            recs, durs = recs[:-best_k], durs[:-best_k]
        out["train"] += recs
    return {k: sorted(v) for k, v in out.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.concat([pd.read_csv(p) for p in sorted((RAW / "data").glob("*_recordingMeta.csv"))])
    meta = meta.rename(columns={"recordingId": "rec", "locationId": "loc"})
    loc_of = dict(zip(meta.rec, meta["loc"]))
    splits = make_splits(meta)
    maps = {loc: read_map(loc) for loc in sorted(meta["loc"].unique())}

    stats, index_rows, per_rec = {}, [], {}
    for split, recs in splits.items():
        acc = {k: [] for k in ("feats", "label", "cross_idx", "rec", "tv", "loc", "kind", "cls", "cross_frame")}
        t0 = time.time()
        drops = {"shoulder": 0, "too early": 0, "other crossing": 0, "invalid": 0}
        for r in recs:
            tm = pd.read_csv(RAW / "data" / f"{r:02d}_tracksMeta.csv")
            classes = dict(zip(tm.trackId, tm["class"]))
            scen, rlc, llc, lk, drop = extract_recording(r, maps[loc_of[r]], classes)
            for k in drops:
                drops[k] += drop[k]
            for f, lab, cx, tid, kind, cls, f_c in scen:
                for key, val in (("feats", f), ("label", lab), ("cross_idx", cx), ("rec", r), ("tv", tid),
                                 ("loc", loc_of[r]), ("kind", kind), ("cls", cls), ("cross_frame", f_c)):
                    acc[key].append(val)
                index_rows.append((split, r, loc_of[r], tid, lab, KIND_NAMES[kind], f_c))
            per_rec[r] = {"loc": int(loc_of[r]), "split": split, "RLC": rlc, "LLC": llc, "LK": lk}
            print(f"  {split} rec {r:02d} (loc {loc_of[r]}): RLC {rlc:4d}  LLC {llc:4d}  LK {lk:4d}", flush=True)
        y = np.array(acc["label"], dtype=np.int8)
        kind = np.array(acc["kind"], dtype=np.int8)
        np.savez_compressed(
            OUT / f"{split}.npz", feats=np.stack(acc["feats"]).astype(np.float32), label=y,
            cross_idx=np.array(acc["cross_idx"], dtype=np.int16), rec=np.array(acc["rec"], dtype=np.int16),
            tv=np.array(acc["tv"], dtype=np.int32), loc=np.array(acc["loc"], dtype=np.int8), kind=kind,
            cls=np.array(acc["cls"], dtype=np.int8), cross_frame=np.array(acc["cross_frame"], dtype=np.int32))
        stats[split] = {"scenarios": int(len(y)), "LK": int((y == 0).sum()), "RLC": int((y == 1).sum()),
                        "LLC": int((y == 2).sum()),
                        "by_kind": {KIND_NAMES[k]: int((kind == k).sum()) for k in range(len(KIND_NAMES))},
                        "dropped_crossings": drops, "recordings": len(recs),
                        "seconds": round(time.time() - t0, 1)}
        print(f"{split}: {stats[split]}", flush=True)

    pd.DataFrame(index_rows, columns=["split", "rec", "loc", "tv", "label", "kind", "cross_frame"]).to_csv(
        OUT / "index.csv", index=False)
    durations = dict(zip(meta.rec, meta.duration))
    total = sum(durations.values())
    (OUT / "meta.json").write_text(json.dumps({
        "dataset": "exiD v2.1", "protocol": "EarlyLCPred (Mozaffari et al., T-IV 2022), adapted to exiD",
        "fps": FPS // FPS_DIV, "seq_len": SEQ_LEN, "in_seq_len": 10, "pred_len": PRED_LEN,
        "labels": {"0": "LK", "1": "RLC", "2": "LLC"}, "kinds": KIND_NAMES, "features": FEATURE_NAMES,
        "lateral_kinematics": f"Savitzky-Golay derivative of the lateral position, window {SG_WINDOW}, order {SG_ORDER}",
        "splits": splits,
        "split_time_share": {k: round(sum(durations[r] for r in v) / total, 3) for k, v in splits.items()},
        "stats": stats, "per_recording": per_rec}, indent=1))
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
