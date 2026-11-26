r"""NGSIM US-101 and I-80 as complete vehicle tracks in the format of datasets/exid/track_stream_exid.py (2026-11-26).

A candidate fresh site for the online-learning study: US and congested (median speeds 4.6-9.3 m/s), against highD's and
exiD's German motorways. Every choice below mirrors the exiD adaptation (datasets/exid/prepare_exid.py) of the 18
features of the EarlyLCPred protocol; no network is involved in this file.

  tracks     NGSIM's rows (datasets/ngsim/explore_ngsim.py extract: US-101 and I-80, exact duplicates dropped) split
             into tracks wherever a vehicle's clock jumps (vehicle numbers restart every 15 min), on the 5 Hz grid
             (Global_Time a multiple of 200 ms).
  lanes      I-80 lanes 1-6, US-101 lanes 1-6 (6 = auxiliary lane); on- and off-ramp lanes (I-80 7, US-101 7 and 8) are
             invalid frames that cut a track, as exiD's hard shoulders. Lane lines halfway between the median lateral
             positions of neighbouring lanes; lane 1's left line half a median width left of its centre.
  features   lateral position positive to the left (-Local_X), lateral velocity and acceleration as Savitzky-Golay
             derivatives over 2 s at 10 Hz (exiD: 2 s); longitudinal speed and acceleration the same way from Local_Y
             (NGSIM's v_Vel and v_Acc are noisy), negated as in the highD code; distance to the left line; neighbours
             from positions at the same instant: same lane ahead and behind, and in each side lane the exiD/highD
             alongside-fallback rule (alongside = longitudinal overlap of the vehicle lengths); 400 m / 0 m/s for an
             absent vehicle; side lane exists where that lane holds traffic within 15 m along the road; lane width.
  crossings  a change of Lane_ID between valid neighbouring lanes, first 5 Hz frame in the new lane; left = towards
             lane 1. Kept only strictly inside a segment (the old lane must be seen).
  recordings 15-min parts of each recording period by a track's first time (a tail shorter than 5 min joins the
             part before): I-80 1-3, US-101 4-6; locations I-80 = 0, US-101 = 1; split train {1, 4}, val {2, 5},
             test {3, 6} (each location in time order).

  .venv\Scripts\python datasets/ngsim/prepare_ngsim.py build
Output: firmware/build/probe/ngsim_tracks_{split}.npz (not versioned) and datasets/ngsim/results/ngsim_prepare.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PKL = HERE / "data" / "motorways.pkl"
PROBE = ROOT / "firmware" / "build" / "probe"
RES = HERE / "results"
FT = 0.3048
IN_LEN = 10
SG = 21                                   # 2 s at 10 Hz
D_DEFAULT = 400.0
VALID = {"i-80": range(1, 7), "us-101": range(1, 7)}
LOC_ID = {"i-80": 0, "us-101": 1}
REC_BASE = {"i-80": 1, "us-101": 4}
SPLITS = {"train": (1, 4), "val": (2, 5), "test": (3, 6)}
EXISTS_M, EXISTS_MIN = 15.0, 20           # a side lane exists where it holds traffic within 15 m (20 frames at least)


def tracks_file(split: str = "test") -> Path:
    return PROBE / f"ngsim_tracks_{split}.npz"


def load() -> pd.DataFrame:
    d = pd.read_pickle(PKL).sort_values(["Location", "Vehicle_ID", "Global_Time"]).reset_index(drop=True)
    vid, t, loc = d.Vehicle_ID.to_numpy(), d.Global_Time.to_numpy(), d.Location.to_numpy()
    new = np.r_[True, (vid[1:] != vid[:-1]) | (loc[1:] != loc[:-1]) | (np.diff(t) != 100)]
    d["track"] = np.cumsum(new)
    # smoothed kinematics at 10 Hz, per track (positions in m)
    x, y = d.Local_X.to_numpy() * FT, d.Local_Y.to_numpy() * FT
    vlat = np.zeros(len(d)); alat = np.zeros(len(d)); vlon = np.zeros(len(d)); alon = np.zeros(len(d))
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], len(d)]
    for a, b in zip(starts, ends):
        if b - a >= SG:
            vlat[a:b] = -savgol_filter(x[a:b], SG, 2, deriv=1, delta=0.1)
            alat[a:b] = -savgol_filter(x[a:b], SG, 2, deriv=2, delta=0.1)
            vlon[a:b] = savgol_filter(y[a:b], SG, 2, deriv=1, delta=0.1)
            alon[a:b] = savgol_filter(y[a:b], SG, 2, deriv=2, delta=0.1)
        else:
            vlat[a:b] = np.nan
    d["x"], d["y"], d["vlat"], d["alat"], d["vlon"], d["alon"] = x, y, vlat, alat, vlon, alon
    d["len"] = d.v_length.to_numpy() * FT
    d["yc"] = d.y - d.len / 2.0                                  # Local_Y is the front centre
    d = d[(d.Global_Time % 200) == 0].reset_index(drop=True)     # 5 Hz grid
    return d


def geometry(d: pd.DataFrame) -> dict:
    geo = {}
    for L, g in d.groupby("Location"):
        med = g[g.Lane_ID.isin(VALID[L])].groupby("Lane_ID").x.median()
        c = med.to_numpy()
        w = float(np.median(np.diff(c)))
        lines = np.r_[c[0] - w / 2, (c[:-1] + c[1:]) / 2, c[-1] + w / 2]   # line k-1|k at index k-1 (lanes 1..n)
        geo[L] = {"centres": c.tolist(), "lines": lines.tolist(), "median_width": w}
    return geo


def recordings(d: pd.DataFrame) -> pd.Series:
    first = d.groupby("track").Global_Time.transform("min")
    rec = pd.Series(0, index=d.index)
    for L, g in d.groupby("Location"):
        t = np.sort(g.Global_Time.unique())
        gaps = np.flatnonzero(np.diff(t) > 60_000)               # recording periods
        cut = t[gaps + 1]
        per = np.searchsorted(cut, first[g.index].to_numpy(), side="right")
        bounds = np.r_[t[0], cut]
        ends = np.r_[t[gaps], t[-1]]                             # last instant of each period
        parts = []
        for p in range(len(bounds)):
            n = max(1, int((ends[p] - bounds[p]) // 900_000))
            if (ends[p] - bounds[p]) - n * 900_000 >= 300_000:
                n += 1
            parts.append(n)
        off = np.r_[0, np.cumsum(parts)[:-1]]
        k = np.minimum(((first[g.index].to_numpy() - bounds[per]) // 900_000).astype(int), np.array(parts)[per] - 1)
        rec.loc[g.index] = REC_BASE[L] + off[per] + k
    return rec


def neighbours(d: pd.DataFrame) -> dict:
    """Per row: same-lane lead and rear, side-lane lead, rear and nearest alongside (row indices, -1 if none)."""
    n = len(d)
    out = {k: np.full(n, -1, np.int64) for k in ("pv", "fv", "lL", "lR", "lA", "rL", "rR", "rA")}
    loc, t, lane = d.Location.to_numpy(), d.Global_Time.to_numpy(), d.Lane_ID.to_numpy()
    yc, ln = d.yc.to_numpy(), d.len.to_numpy()
    order = np.lexsort((yc, lane, t, loc))
    key = pd.factorize(pd.Series(loc[order]) + "_" + pd.Series(t[order]).astype(str))[0]
    cut = np.flatnonzero(np.diff(key)) + 1
    for g in np.split(order, cut):
        lanes = lane[g]
        per = {int(L): g[lanes == L] for L in np.unique(lanes)}
        for L, idx in per.items():
            ys = yc[idx]
            out["pv"][idx[:-1]] = idx[1:]                        # sorted by position: the next one is ahead
            out["fv"][idx[1:]] = idx[:-1]
            for side, Ls in (("l", L - 1), ("r", L + 1)):
                other = per.get(Ls)
                if other is None:
                    continue
                yo, lo = yc[other], ln[other]
                gap = np.abs(yo[None, :] - ys[:, None])
                over = gap < (lo[None, :] + ln[idx][:, None]) / 2.0
                ga = np.where(over, gap, np.inf)
                j = np.argmin(ga, axis=1)
                hit = np.isfinite(ga[np.arange(len(idx)), j])
                out[side + "A"][idx[hit]] = other[j[hit]]
                ah = np.where(~over & (yo[None, :] > ys[:, None]), yo[None, :], np.inf)
                j = np.argmin(ah, axis=1)
                hit = np.isfinite(ah[np.arange(len(idx)), j])
                out[side + "L"][idx[hit]] = other[j[hit]]
                be = np.where(~over & (yo[None, :] < ys[:, None]), yo[None, :], -np.inf)
                j = np.argmax(be, axis=1)
                hit = np.isfinite(be[np.arange(len(idx)), j])
                out[side + "R"][idx[hit]] = other[j[hit]]
    return out


def features(d: pd.DataFrame, nb: dict, geo: dict, exists: dict) -> np.ndarray:
    yc, v = d.yc.to_numpy(), d.vlon.to_numpy()

    def dist(i, j):
        return np.where(j >= 0, np.abs(yc[np.maximum(j, 0)] - yc[i]), D_DEFAULT)

    def rel(i, j):
        return np.where(j >= 0, v[i] - v[np.maximum(j, 0)], 0.0)

    i = np.arange(len(d))
    feats = np.zeros((len(d), 18), np.float32)
    lane = d.Lane_ID.to_numpy()
    for side, col in (("l", (12, 13, 14)), ("r", (9, 10, 11))):
        a, ld, rr = nb[side + "A"], nb[side + "L"], nb[side + "R"]
        vv = np.where(a >= 0, a, -1)
        pv = np.where(a >= 0, ld, -1)
        fv = np.where(a >= 0, rr, -1)
        none = (a < 0) & (ld < 0) & (rr < 0)
        only = (a < 0) & ~none
        closer_lead = only & (ld >= 0) & ((rr < 0) | (dist(i, ld) < dist(i, rr)))
        closer_rear = only & ~closer_lead
        vv = np.where(closer_lead, ld, np.where(closer_rear, rr, vv))
        pv = np.where(closer_lead, nb["pv"][np.maximum(ld, 0)], np.where(closer_rear, -1, pv))
        fv = np.where(closer_rear, nb["fv"][np.maximum(rr, 0)], np.where(closer_lead, -1, fv))
        feats[:, col[0]], feats[:, col[1]], feats[:, col[2]] = dist(i, pv), dist(i, vv), dist(i, fv)
    left_line = np.zeros(len(d)); width = np.zeros(len(d))
    for L, gm in geo.items():
        m = (d.Location == L).to_numpy()
        lines = np.array(gm["lines"])
        k = np.clip(lane[m] - 1, 0, len(lines) - 2)
        left_line[m] = lines[k]
        width[m] = lines[k + 1] - lines[k]
    feats[:, 0], feats[:, 1] = d.vlat, -d.vlon
    feats[:, 2], feats[:, 3] = d.alat, -d.alon
    feats[:, 4] = d.x.to_numpy() - left_line
    feats[:, 5], feats[:, 6] = rel(i, nb["pv"]), dist(i, nb["pv"])
    feats[:, 7], feats[:, 8] = rel(i, nb["fv"]), dist(i, nb["fv"])
    feats[:, 15] = (lane > 1).astype(np.float32)
    feats[:, 16] = exists["right"]
    feats[:, 17] = width
    return feats


def side_lane_exists(d: pd.DataFrame) -> dict:
    """Right lane present where it holds traffic within EXISTS_M along the road (any time of the location)."""
    right = np.zeros(len(d), np.float32)
    for L, g in d.groupby("Location"):
        bins = (g.yc // EXISTS_M).astype(int)
        occ = pd.Series(1, index=pd.MultiIndex.from_arrays([g.Lane_ID, bins])).groupby(level=[0, 1]).sum()
        occ = set(occ[occ >= EXISTS_MIN].index)
        r = [((ln + 1, b) in occ) or ((ln + 1, b - 1) in occ) or ((ln + 1, b + 1) in occ)
             for ln, b in zip(g.Lane_ID.to_numpy(), bins.to_numpy())]
        right[g.index] = np.array(r, np.float32)
    return {"right": right}


def build() -> None:
    d = load()
    geo = geometry(d)
    d["rec"] = recordings(d)
    nb = neighbours(d)
    ex = side_lane_exists(d)
    f = features(d, nb, geo, ex)
    valid = np.array([ln in VALID[L] for L, ln in zip(d.Location, d.Lane_ID)]) & np.isfinite(f).all(1)
    acc = {s: {"feats": [], "lens": [], "rec": [], "tv": [], "seg_start": [], "loc": [], "cross": [], "kinds": [],
               "t_first_ms": []}
           for s in SPLITS}
    stats = {"short": 0, "invalid_tracks": 0, "invalid_frames": 0, "crossings_dropped_ramp_or_jump": 0}
    for tr, idx in d.groupby("track").indices.items():
        n = len(idx)
        if n < IN_LEN:
            stats["short"] += 1
            continue
        L = d.Location.iat[idx[0]]
        r = int(d.rec.iat[idx[0]])
        split = next(s for s, rs in SPLITS.items() if r in rs)
        ok = valid[idx]
        if not ok.all():
            stats["invalid_tracks"] += 1
            stats["invalid_frames"] += int((~ok).sum())
        lane = d.Lane_ID.to_numpy()[idx]
        cr = []
        for c in np.flatnonzero(lane[1:] != lane[:-1]) + 1:
            a, b = int(lane[c - 1]), int(lane[c])
            if ok[c - 1] and ok[c] and abs(a - b) == 1:
                cr.append((int(c), 2 if b < a else 1))
            else:
                stats["crossings_dropped_ramp_or_jump"] += 1
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            if s1 - s0 < IN_LEN:
                continue
            inside = [[int(c - s0), int(dd)] for c, dd in cr if s0 < c < s1]
            A = acc[split]
            A["feats"].append(f[idx[s0:s1]])
            A["lens"].append(int(s1 - s0))
            A["rec"].append(r)
            A["tv"].append(int(tr))
            A["seg_start"].append(int(s0))
            A["t_first_ms"].append(int(d.Global_Time.iat[idx[s0]]))   # absolute time of the segment's first frame
            A["loc"].append(LOC_ID[L])
            A["cross"].append(inside)
            A["kinds"].append([1] * len(inside))                 # main carriageway (ramps are invalid here)
    PROBE.mkdir(parents=True, exist_ok=True)
    summary = {"geometry_m": geo, "stats": stats, "splits": {}}
    for s, A in acc.items():
        np.savez_compressed(tracks_file(s), feats=np.concatenate(A["feats"]).astype(np.float32),
                            lens=np.array(A["lens"]), rec=np.array(A["rec"]), tv=np.array(A["tv"]),
                            seg_start=np.array(A["seg_start"]), loc=np.array(A["loc"]),
                            t_first_ms=np.array(A["t_first_ms"], dtype=np.int64),
                            cross=json.dumps(A["cross"]), kinds=json.dumps(A["kinds"]),
                            short=stats["short"], invalid=stats["invalid_tracks"],
                            invalid_frames=stats["invalid_frames"])
        dirs = [dd for c in A["cross"] for _, dd in c]
        summary["splits"][s] = {"segments": len(A["lens"]), "frames": int(sum(A["lens"])),
                                "vehicle_hours": round(sum(A["lens"]) / 5 / 3600, 2),
                                "crossings": len(dirs), "left": dirs.count(2), "right": dirs.count(1),
                                "recordings": sorted(set(A["rec"]))}
        print(s, summary["splits"][s], flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "ngsim_prepare.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build()
    else:
        raise SystemExit("usage: prepare_ngsim.py build")
