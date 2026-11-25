"""How fast a CARLA-recorded drive ran compared with the clock (2026-11-25).

The lab's client takes one fixed simulation step (0.051 s) per drawn frame and has no clock of its own, so a drive runs
as fast as the PC draws. For every CARLA recorder file under the given folders this script reads the start time in the
file's header and the elapsed simulation time of its last frame, and takes the end on the clock from the file's date
(the recorder closes the file when the drive ends; copying keeps that date). Pace = simulated duration / wall duration.

  python recording_pace.py <folder> [<folder> ...]

Prints one line per recording (simulated and wall seconds, pace) and the medians of the drives longer than 2 minutes,
split at 750 s simulated (LCIR: lane-change drives about 15 min, baseline drives about 9 min). Copies of the same
recording (same start time) count once. Python 3.8 or newer, no packages.
"""
from __future__ import annotations

import datetime
import os
import statistics
import struct
import sys

MAGIC = b"CARLA_RECORDER"


def read_recording(path: str):
    """(start time, frames, elapsed seconds of the last frame) or None if the file is not a CARLA recording."""
    with open(path, "rb") as f:
        data = f.read()
    if data[4:4 + len(MAGIC)] != MAGIC:
        return None
    pos = 2 + 2 + struct.unpack_from("<H", data, 2)[0]          # version, then the string "CARLA_RECORDER"
    start = struct.unpack_from("<q", data, pos)[0]
    pos += 8
    pos += 2 + struct.unpack_from("<H", data, pos)[0]           # map file name
    frames, elapsed = 0, None
    while pos + 5 <= len(data):
        packet, size = data[pos], struct.unpack_from("<I", data, pos + 1)[0]
        if packet == 0 and size >= 24:                          # frame: uint64 id, double duration, double elapsed
            frames += 1
            elapsed = struct.unpack_from("<d", data, pos + 5 + 16)[0]
        pos += 5 + size
    return start, frames, elapsed


def main(folders):
    rows = {}
    for root in folders:
        for dirpath, _, names in os.walk(root):
            for name in names:
                path = os.path.join(dirpath, name)
                if os.path.getsize(path) < 100_000 or name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".h5")):
                    continue
                rec = read_recording(path)
                if rec is None or rec[2] is None:
                    continue
                start, frames, sim = rec
                wall = os.path.getmtime(path) - start
                print(f"{datetime.datetime.fromtimestamp(start):%Y-%m-%d %H:%M:%S}  frames {frames:6d}  simulated "
                      f"{sim:7.1f} s  wall {wall:7.1f} s  pace {sim / wall if wall > 0 else float('nan'):5.3f}  {name}")
                rows[start] = (sim, wall)
    for label, keep in (("drives over 750 s simulated", lambda s: s > 750), ("drives of 120-750 s", lambda s: 120 < s <= 750)):
        p = sorted(s / w for s, w in rows.values() if keep(s) and w > 0)
        if p:
            print(f"{label}: n = {len(p)}, pace median {statistics.median(p):.3f}, quartiles "
                  f"{p[len(p) // 4]:.3f}-{p[3 * len(p) // 4]:.3f}, range {p[0]:.3f}-{p[-1]:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
