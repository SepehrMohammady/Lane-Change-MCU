r"""NGSIM US-101 and I-80: data check for a fresh-site test (2026-11-26). No network is run here.

Source: FHWA's "Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data" (data.transportation.gov,
dataset 8ect-6jqj, CC BY-SA 3.0 per its catalog entry), downloaded into datasets/ngsim/data/raw/ (not versioned).
Units: feet, feet per second, 10 Hz frames, Global_Time in ms.

  .venv\Scripts\python datasets/ngsim/explore_ngsim.py extract     # US-101 and I-80 rows -> data/motorways.pkl
  .venv\Scripts\python datasets/ngsim/explore_ngsim.py check       # quality report -> results/ngsim_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw" / "ngsim_trajectories.csv"
PKL = HERE / "data" / "motorways.pkl"
RES = HERE / "results"
FT = 0.3048
COLS = ["Vehicle_ID", "Frame_ID", "Global_Time", "Local_X", "Local_Y", "v_length", "v_Width", "v_Class", "v_Vel",
        "v_Acc", "Lane_ID", "Preceding", "Following", "Space_Headway", "Location"]


def extract() -> None:
    parts = []
    for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=2_000_000, low_memory=False):
        parts.append(chunk[chunk.Location.isin(["us-101", "i-80"])])
    d = pd.concat(parts, ignore_index=True)
    n0 = len(d)
    d = d.drop_duplicates(["Location", "Vehicle_ID", "Frame_ID", "Global_Time"]).reset_index(drop=True)
    print(f"{n0} rows, {n0 - len(d)} exact duplicates of (location, vehicle, frame, time) dropped")
    d.to_pickle(PKL)
    print(d.groupby("Location").size())


def periods(d: pd.DataFrame) -> pd.Series:
    """Recording period of every row: per location, a new period wherever the frame clock jumps by more than 60 s."""
    out = pd.Series(0, index=d.index)
    for loc, g in d.groupby("Location"):
        t = np.sort(g.Global_Time.unique())
        cut = t[1:][np.diff(t) > 60_000]
        out.loc[g.index] = np.searchsorted(cut, g.Global_Time.to_numpy(), side="right")
    return out


def check() -> None:
    d = pd.read_pickle(PKL)
    d["period"] = periods(d)
    rep = {}
    for (loc, per), g in d.groupby(["Location", "period"]):
        g = g.sort_values(["Vehicle_ID", "Global_Time"])
        key = f"{loc}_p{per}"
        t0, t1 = g.Global_Time.min(), g.Global_Time.max()
        dup = int(g.duplicated(["Vehicle_ID", "Global_Time"]).sum())
        vid = g.Vehicle_ID.to_numpy()
        tt = g.Global_Time.to_numpy()
        same = vid[1:] == vid[:-1]
        gaps = int((same & (np.diff(tt) != 100)).sum())
        lane = g.Lane_ID.to_numpy()
        chg = same & (lane[1:] != lane[:-1])
        # a change followed by a change back to the old lane within 1 s / 3 s of the same vehicle
        idx = np.flatnonzero(chg) + 1
        back1 = back3 = 0
        for i in idx:
            old = lane[i - 1]
            for k, lim in ((10, "1"), (30, "3")):
                j = slice(i + 1, min(i + 1 + k, len(lane)))
                ok = (vid[j] == vid[i]) & (lane[j] == old)
                if ok.any():
                    if lim == "1":
                        back1 += 1
                    else:
                        back3 += 1
        rep[key] = {"start_utc_ms": int(t0), "minutes": round((t1 - t0) / 60000, 1), "rows": int(len(g)),
                    "vehicles": int(g.Vehicle_ID.nunique()), "vehicle_hours": round(len(g) / 10 / 3600, 2),
                    "duplicate_vehicle_time_rows": dup, "time_gaps_within_a_vehicle": gaps,
                    "lanes": sorted(int(v) for v in np.unique(lane)), "lane_id_changes": int(chg.sum()),
                    "changed_back_within_1_s": back1, "changed_back_within_3_s": back3,
                    "speed_m_s_median": round(float(g.v_Vel.median() * FT), 2),
                    "speed_m_s_p10_p90": [round(float(g.v_Vel.quantile(q) * FT), 2) for q in (0.1, 0.9)],
                    "classes": {str(int(k)): int(v) for k, v in g.groupby("v_Class").Vehicle_ID.nunique().items()}}
        # lateral position per lane: Local_X spread (ft) and the boundary between neighbouring lanes
        lx = g.groupby("Lane_ID").Local_X.describe(percentiles=[0.05, 0.5, 0.95])
        rep[key]["local_x_ft_by_lane"] = {str(int(k)): [round(float(r["5%"]), 1), round(float(r["50%"]), 1),
                                                        round(float(r["95%"]), 1)] for k, r in lx.iterrows()}
        print(key, json.dumps({k: v for k, v in rep[key].items() if k != "local_x_ft_by_lane"}), flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "ngsim_check.json").write_text(json.dumps(rep, indent=1))
    print("written", RES / "ngsim_check.json")


if __name__ == "__main__":
    {"extract": extract, "check": check}[sys.argv[1]]()
