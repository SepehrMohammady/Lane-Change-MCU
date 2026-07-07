"""Definitively name pickle channels by correlating against raw H5 signals,
and measure the turn-indicator label-leak for classification.

Uses user22 (a train driver) whose windows we can locate in the train pickle
(via YawRate correlation), then correlates every pickle channel against every
named raw egoVehicle signal + egoLaneWidth.
"""
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

H5 = Path(r"C:\Projects\PhD\DIMIR\Materials\h5-samples\user22LC.h5")
DATA = Path(r"C:\Projects\PhD\DIMIR\data")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
W = 50

EGO_FIELDS = ["LatAcceleration", "LongAcceleration", "BrakePedalPos",
              "DirectionIndicator", "SteeringAngle", "ThrottlePedalPos",
              "VehicleSpeed", "YawRate"]


def windows(series):
    n = len(series) - W + 1
    idx = np.arange(n)[:, None] + np.arange(W)[None, :]
    return series[idx]


def z(a):
    t = torch.from_numpy(np.asarray(a)).float().to(DEV)
    m = t.mean(dim=1, keepdim=True); s = t.std(dim=1, keepdim=True)
    return torch.where(s > 1e-8, (t - m) / s, torch.zeros_like(t))


def main():
    with h5py.File(H5, "r") as f:
        ego = {k: f["egoVehicle"][k].astype(np.float64) for k in EGO_FIELDS}
        lane_w = f["laneLines"]["EgoLaneWidth"].astype(np.float64)
        di = f["egoVehicle"]["DirectionIndicator"]
        print("DirectionIndicator raw unique values:", np.unique(di))
    raw = {k: windows(v) for k, v in ego.items()}
    raw["EgoLaneWidth"] = windows(lane_w)
    # derived indicator split candidates
    di_w = windows(ego["DirectionIndicator"])
    for val in np.unique(ego["DirectionIndicator"]):
        raw[f"DirInd=={val:g}"] = (di_w == val).astype(np.float64)

    with open(DATA / "data-classification" / "x_train_multi.pkl", "rb") as fp:
        x = pickle.load(fp).astype(np.float32)

    # locate user22 windows in the pickle via YawRate correlation
    zraw_yaw = z(raw["YawRate"])
    zp_yaw = z(x[:, :, 6].astype(np.float64))  # cls ch6 ~ yawRate hypothesis
    # for each pickle window, best corr to any raw yaw window
    matches = []  # (pickle_idx, raw_idx)
    for i in range(0, len(zp_yaw), 4096):
        corr = zp_yaw[i:i+4096] @ zraw_yaw.T / (W - 1)
        m, arg = corr.max(dim=1)
        for k in torch.where(m > 0.9999)[0]:
            matches.append((i + int(k), int(arg[k])))
    print(f"matched user22 windows in cls-train: {len(matches)}")
    if len(matches) < 50:
        print("too few matches; aborting naming"); return
    p_idx = np.array([m[0] for m in matches])
    r_idx = np.array([m[1] for m in matches])

    # name each pickle channel by best |corr| to a raw signal across matched windows
    print("\nchannel -> best raw signal (|corr|):")
    for c in range(x.shape[2]):
        pc = z(x[p_idx][:, :, c].astype(np.float64)).reshape(-1)
        best_name, best_abs = "?", 0.0
        for name, rw in raw.items():
            rc = z(rw[r_idx]).reshape(-1)
            sd = pc.std().item() * rc.std().item()
            if sd < 1e-9:
                # both constant -> match if equal
                if abs(x[p_idx][:, :, c].std()) < 1e-6 and rw[r_idx].std() < 1e-6:
                    best_name, best_abs = name + "(const)", 1.0
                continue
            r = abs((pc * rc).mean().item())
            if r > best_abs:
                best_abs, best_name = r, name
        print(f"  ch{c:>2}: {best_name:<22} |r|={best_abs:.4f}")


if __name__ == "__main__":
    main()
