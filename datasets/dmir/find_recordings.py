"""Find CARLA recordings by the date written inside them, not by file name or file date (2026-11-17).

The lab's client names a recording with the first free number in its logs folder, so the same name was used for
different drives once older files had been moved away, and copying can change a file's date. The recorder writes
the start of the recording into the file's header (CARLA 0.9.15: uint16 version, the string "CARLA_RECORDER",
int64 time), and this script reads it from every file under the given folders.

  python find_recordings.py <folder> [<folder> ...] [--from "2024-10-21 00:00"] [--to "2024-10-21 23:59"]

Prints the start time, size and path of every recording in the window (all recordings without a window),
oldest first. Only the first 64 bytes of each file are read. Python 3.8 or newer, no packages.
"""
from __future__ import annotations

import datetime
import os
import struct
import sys

MAGIC = b"CARLA_RECORDER"


def start_of(path: str):
    """The recording's start time from its header, or None if the file is not a CARLA recording."""
    try:
        with open(path, "rb") as f:
            b = f.read(64)
    except OSError:
        return None
    if len(b) < 30 or b[4:4 + len(MAGIC)] != MAGIC:
        return None
    try:
        return datetime.datetime.fromtimestamp(struct.unpack_from("<q", b, 4 + len(MAGIC))[0])
    except (OverflowError, OSError, ValueError):
        return None


def main(argv):
    folders, lo, hi = [], None, None
    i = 0
    while i < len(argv):
        if argv[i] in ("--from", "--to"):
            t = datetime.datetime.strptime(argv[i + 1], "%Y-%m-%d %H:%M")
            lo, hi = (t, hi) if argv[i] == "--from" else (lo, t)
            i += 2
        else:
            folders.append(argv[i])
            i += 1
    found = []
    for root in folders:
        for dp, _, names in os.walk(root):
            for n in names:
                if n.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".png", ".jpg")):
                    continue
                p = os.path.join(dp, n)
                t = start_of(p)
                if t and (lo is None or t >= lo) and (hi is None or t <= hi):
                    found.append((t, os.path.getsize(p), p))
    for t, size, p in sorted(found):
        print(f"{t:%Y-%m-%d %H:%M:%S}  {size / 1e6:7.1f} MB  {p}")
    print(f"{len(found)} recording(s)")


if __name__ == "__main__":
    main(sys.argv[1:])
