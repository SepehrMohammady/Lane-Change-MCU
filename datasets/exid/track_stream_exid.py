"""exiD on complete vehicle tracks, for the shortcut study of 2026-11-18 (local notes, 16:30).

The highD study of 2026-11-17 (datasets/highd/track_stream.py, unas/track_train_check.py) found that a network
trained on the protocol's scenarios announces the opposite lane change right after a vehicle has changed lanes,
and that training on windows from complete tracks removes most of it. This module builds exiD's complete tracks
in the same file format, so the same measures (track_stream.measures_of) and the same training script
(unas/track_train_check.py with TRACK_DATASET=exid) apply.

A track at 5 Hz gets the 18 features of prepare_exid.Recording.features frame by frame. Where a frame has no
valid features (hard shoulder, no lanelet, missing lane width or offset) the track is cut, and every piece of at
least 10 frames counts as a track ("segment"). Crossings come from exiD's laneChange flag (first 5 Hz frame in
the new lane, side from the lane offset, as in the preparation); crossings into or out of a hard shoulder are
left out, as in the protocol, and only crossings strictly inside a segment are kept (the old lane must be seen).

  .venv\\Scripts\\python datasets/exid/track_stream_exid.py build [split ...]   # Windows; default: test val train
  .venv\\Scripts\\python datasets/exid/track_stream_exid.py --summary            # counts of the built files

Output: firmware/build/probe/exid_tracks_{split}.npz (not versioned: licensed data) with the keys of the highD
files (feats, lens, rec, tv, cross, short, invalid) plus seg_start, loc, kinds (kind of each crossing, as in
prepare_exid.KIND_NAMES) and invalid_frames.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREP = HERE / "data" / "prepared"
PROBE = ROOT / "firmware" / "build" / "probe"
IN_LEN = 10
WORKERS = 8
KIND_NAMES = ["lane keep", "main carriageway", "merge from on-ramp", "exit to off-ramp", "on a ramp"]  # prepare_exid's


def tracks_file(split: str = "test") -> Path:
    return PROBE / f"exid_tracks_{split}.npz"


def one_recording(args):
    """The segments of every track of one recording: features, lengths, crossings and their kinds."""
    r, loc = args
    sys.path.insert(0, str(HERE))
    import pandas as pd
    import prepare_exid as px

    lanelets = px.read_map(loc)
    tm = pd.read_csv(px.RAW / "data" / f"{r:02d}_tracksMeta.csv")
    rec = px.Recording(r, lanelets, dict(zip(tm.trackId, tm["class"])))
    out = {"feats": [], "lens": [], "tv": [], "seg_start": [], "cross": [], "kinds": []}
    short = invalid_tracks = invalid_frames = 0
    for tid, tv in rec.T.items():
        n = len(tv["frame"])
        if n < IN_LEN:
            short += 1
            continue
        rows = [rec.features(tid, i, i + 1) for i in range(n)]
        valid = np.array([f is not None for f in rows])
        if not valid.all():
            invalid_tracks += 1
            invalid_frames += int((~valid).sum())
        crossings = []
        for f_c, side, kind, shoulder in rec.crossings.get(tid, []):
            c = int(np.searchsorted(tv["frame"], f_c))
            if c < n and not shoulder:
                crossings.append((c, side, kind))
        crossings.sort()
        edges = np.flatnonzero(np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):                # maximal runs of valid frames [a, b)
            if b - a < IN_LEN:
                continue
            inside = [(c - a, side, kind) for c, side, kind in crossings if a < c < b]
            out["feats"].append(np.concatenate(rows[a:b]).astype(np.float32))
            out["lens"].append(int(b - a))
            out["tv"].append(int(tid))
            out["seg_start"].append(int(a))
            out["cross"].append([[int(c), int(d)] for c, d, _ in inside])
            out["kinds"].append([int(k) for _, _, k in inside])
    return r, loc, out, short, invalid_tracks, invalid_frames


def build(split: str):
    meta = json.loads((PREP / "meta.json").read_text())
    recs = meta["splits"][split]
    loc_of = {int(r): int(v["loc"]) for r, v in meta["per_recording"].items()}
    t0 = time.time()
    acc = {"feats": [], "lens": [], "rec": [], "tv": [], "seg_start": [], "loc": [], "cross": [], "kinds": []}
    short = invalid_tracks = invalid_frames = 0
    with Pool(WORKERS) as pool:
        for r, loc, out, s, it, fr in pool.imap(one_recording, [(r, loc_of[r]) for r in recs]):
            n = len(out["lens"])
            for key in ("feats", "lens", "tv", "seg_start", "cross", "kinds"):
                acc[key].extend(out[key])
            acc["rec"].extend([r] * n)
            acc["loc"].extend([loc] * n)
            short, invalid_tracks, invalid_frames = short + s, invalid_tracks + it, invalid_frames + fr
            print(f"{split} recording {r:02d} (location {loc}): {n} segments, {sum(out['lens'])} frames, "
                  f"{time.time() - t0:.0f} s", flush=True)
    PROBE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tracks_file(split), feats=np.concatenate(acc["feats"]).astype(np.float32),
                        lens=np.array(acc["lens"]), rec=np.array(acc["rec"]), tv=np.array(acc["tv"]),
                        seg_start=np.array(acc["seg_start"]), loc=np.array(acc["loc"]),
                        cross=json.dumps(acc["cross"]), kinds=json.dumps(acc["kinds"]),
                        short=short, invalid=invalid_tracks, invalid_frames=invalid_frames)
    print(f"{split}: {len(acc['lens'])} segments, {int(sum(acc['lens']))} frames, "
          f"{sum(len(c) for c in acc['cross'])} crossings; left out: {short} tracks shorter than a window, "
          f"{invalid_frames} invalid frames in {invalid_tracks} tracks; {time.time() - t0:.0f} s", flush=True)


def summary():
    for split in ("train", "val", "test"):
        f = tracks_file(split)
        if not f.exists():
            print(split, "not built")
            continue
        z = np.load(f)
        cross, kinds = json.loads(str(z["cross"])), json.loads(str(z["kinds"]))
        d = np.array([dd for c in cross for _, dd in c] or [0])
        k = np.array([kk for ks in kinds for kk in ks] or [0])
        print(f"{split}: {len(z['lens'])} segments, {int(z['lens'].sum())} frames, {len(d)} crossings "
              f"(right {int((d == 1).sum())}, left {int((d == 2).sum())}; by kind {np.bincount(k, minlength=5).tolist()}), "
              f"left out {int(z['short'])} short tracks, {int(z['invalid_frames'])} invalid frames in {int(z['invalid'])} tracks")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        for sp in (sys.argv[2:] or ["test", "val", "train"]):
            build(sp)
    elif len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summary()
    else:
        sys.exit("build [split ...] | --summary")
