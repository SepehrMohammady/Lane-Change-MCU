"""Closed loop A with the STM32H7B3I-DK as the lane-change predictor (2026-11-17).

datasets/highd/acc_loop.py runs an ACC that anticipates the lane change of a neighbouring vehicle on recorded
highD scenes. Here the predictor is the board: every 5th frame (5 Hz) the PC computes the TV's 18 raw features
from the replayed vehicles and the simulated ego and sends them over the ST-LINK virtual COM port to the stream
firmware (firmware/h7b3, make STREAM=1), which scales, quantizes, keeps the 10-frame window and runs the int8
network; its outputs decide the ACC's anticipation, which moves the ego, whose position enters the next
features. Fixed beforehand (local notes, 2026-11-17 12:54, amended 12:58 and 13:03): P seed 100 and K seed 100
(no choice) over every episode of both parts at top speed, decisions compared with the same loop on the PC
(acc_loop.py run), and one set of 20 part-C episodes in real time (the ego at 25 Hz, frames at 5 Hz).

  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/acc_loop_board.py prepare     # WSL: headers
  .venv\\Scripts\\python unas/acc_loop_board.py run                                   # Windows, board connected
  .venv\\Scripts\\python unas/acc_loop_board.py realtime
  .venv\\Scripts\\python unas/acc_loop_board.py --report
Output: datasets/highd/results/hil/acc_loop_board.jsonl (one record per network and run) and acc_loop_board.json.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("ACC_OUT", "acc_loop_board.jsonl")
sys.path.insert(0, str(REPO / "unas"))
sys.path.insert(0, str(REPO / "datasets" / "highd"))
import acc_loop as al                                        # noqa: E402  (numpy only at import)

NETS = {"P": al.KEEP / f"P_{al.MODEL}_seed100_int8_io.tflite", "K": al.KEEP / f"K_{al.MODEL}_seed100_int8_io.tflite"}
SUMMARY = al.RES / "acc_loop_board.json"
REAL_TIME = ("K", 20)                                        # network and number of part-C episodes in real time


class BoardPredictor:
    """One frame out, one reply in (the stream firmware's protocol, unas/hil_stream.py)."""

    def __init__(self, port, n_out: int = 3, in_len: int = al.IN_LEN):
        self.port, self.n_out, self.in_len = port, n_out, in_len
        self.seq = self.n = 0
        self.last_out = None
        self.cycles_net, self.cycles_pre, self.rtt = [], [], []

    def __call__(self, x: np.ndarray, first: bool):
        if first:
            self.n = 0
        pkt = struct.pack("<BBH", 0xA5, 1 if first else 0, self.seq & 0xFFFF) + np.asarray(x, "<f4").tobytes()
        t0 = time.perf_counter()
        self.port.write(pkt)
        rep = self.port.read(12 + self.n_out)
        self.rtt.append(time.perf_counter() - t0)
        if len(rep) != 12 + self.n_out or rep[0] != 0x5A or struct.unpack_from("<H", rep, 2)[0] != (self.seq & 0xFFFF):
            raise RuntimeError(f"bad reply at frame {self.seq}: {rep!r}")
        self.seq += 1
        self.n += 1
        ran = rep[1] == 1
        if ran != (self.n >= self.in_len):
            raise RuntimeError(f"frame {self.seq}: the board ran={ran} with {self.n} frames in its window")
        c_pre, c_inf = struct.unpack_from("<II", rep, 4)
        self.cycles_pre.append(c_pre)
        if not ran:
            self.last_out = None
            return None
        self.cycles_net.append(c_inf)
        self.last_out = np.frombuffer(rep[12:], np.int8).copy()
        return int(self.last_out.astype(np.int16).argmax())


def prepare():
    """WSL: the stream header (training scaling of highD) of the two networks."""
    import hil_prepare as hp

    for tfl in NETS.values():
        hp.main(str(tfl.relative_to(REPO)))


def flashed(tfl: Path):
    """Build the stream firmware for one network, flash it, check its status; returns (serial number, size, core Hz)."""
    import hil_stream as hs
    import local_bench as lb

    elf = hs.build(tfl)
    size = lb.elf_size(elf)
    sn = lb.find_stlink("STM32H7B3I-DK")
    lb.run([lb.CLI, "-c", "port=SWD", f"sn={sn}", "mode=UR", "-w", elf, "-v", "-rst"])
    res = lb.read_result(elf, sn)
    if res[6] != 0:
        raise RuntimeError(f"firmware reported status {res[6]:#x}")
    return sn, size, res[1]


def timing(pr: BoardPredictor, hz: float) -> dict:
    c = np.array(pr.cycles_net, float)
    return {"network_cycles_mean": round(float(c.mean()), 1), "network_ms_mean": round(float(c.mean()) / hz * 1e3, 4),
            "host_round_trip_ms": {"mean": round(float(np.mean(pr.rtt)) * 1e3, 3), "p99": round(float(np.percentile(pr.rtt, 99)) * 1e3, 3),
                                   "max": round(float(np.max(pr.rtt)) * 1e3, 3)},
            "frames": len(pr.rtt), "decisions": len(pr.cycles_net)}


def run():
    import hil_stream as hs

    done = {r["network"] for r in map(json.loads, al.OUT.read_text().splitlines())} if al.OUT.exists() else set()
    for arm, tfl in NETS.items():
        label = f"{arm}_seed100_board"
        if label in done:
            continue
        sn, size, hz = flashed(tfl)
        with hs.open_port(sn) as port:
            pr = BoardPredictor(port)
            t0 = time.perf_counter()
            rec = al.run_arm(arm, lambda pr=pr: pr, label, al.PROBE / f"acc_loop_board_{arm}_seed100.json")
            wall = time.perf_counter() - t0
        extra = {"network": label, "kind": "top speed", "firmware": size, "core_hz": hz, "timing": timing(pr, hz),
                 "wall_s": round(wall, 1)}
        with open(al.OUT, "a") as f:
            f.write(json.dumps({**extra, "kind_record": "board"}) + "\n")
        print(f"{datetime.now():%H:%M} {label}: C {rec['parts']['C_all_kept_cells']['share']:.3f}, O {rec['parts']['O']['share']:.3f}, "
              f"network {extra['timing']['network_ms_mean']} ms, {wall:.0f} s", flush=True)


def realtime():
    """REAL_TIME[1] part-C episodes of the first fixed cell with the board, paced at 25 Hz."""
    import hil_stream as hs

    arm, n = REAL_TIME
    tfl = NETS[arm]
    sn, size, hz = flashed(tfl)
    g, dv = al.cells_kept()[0]
    eps = [e for e in al.episodes_of("C") if e["tv_last"] >= e["f_c"] + al.POST_C_MIN]
    sc = al.scenes_for(eps)
    rows, late = [], []
    with hs.open_port(sn) as port:
        pr = BoardPredictor(port)
        for e in eps:
            if len(rows) >= n:
                break
            S = sc[(e["rec"], al.direction_of(e["rec"], e["tv"]))]
            if al.setup(S, "C", e, g, dv) is None:
                continue
            start = [None]

            def clock(step, start=start):
                if start[0] is None:
                    start[0] = time.perf_counter()
                due = start[0] + step * al.DT
                now = time.perf_counter()
                if now < due:
                    time.sleep(due - now)
                late.append(time.perf_counter() - due)
            rows.append(al.episode(S, "C", e, arm, pr, g, dv, clock))
    lat = np.array(late) * 1e3
    rec = {"network": f"{arm}_seed100_board", "kind": "real time", "cell": [g, dv], "episodes": len(rows),
           "steps": len(late), "step_lateness_ms": {"mean": round(float(lat.mean()), 3), "p99": round(float(np.percentile(lat, 99)), 3),
                                                     "max": round(float(lat.max()), 3)},
           "timing": timing(pr, hz), "outcome": al.summarize("C", rows),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(al.OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    (al.PROBE / "acc_loop_board_realtime.json").write_text(json.dumps(rows, default=float))
    print(json.dumps(rec, indent=1))


def report():
    """The board's loops against the same loops on the PC: decisions, outputs, outcomes; timing."""
    recs = [json.loads(line) for line in al.OUT.read_text().splitlines()]
    res = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "networks": {}}
    for arm in NETS:
        b = al.PROBE / f"acc_loop_board_{arm}_seed100.json"
        pc = al.PROBE / f"acc_loop_pc_{arm}_seed100.json"
        if not b.exists() or not pc.exists():
            continue
        B, P = json.loads(b.read_text()), json.loads(pc.read_text())
        same_dec = same_out = total_dec = total_out = episodes = identical_eps = 0
        max_lsb = 0
        first_div = []
        for part in B:
            for rb, rp in zip(B[part], P[part]):
                if rb is None or rp is None:
                    continue
                episodes += 1
                db, dp = rb["decisions"], rp["decisions"]
                k = next((i for i, (x, y) in enumerate(zip(db, dp)) if x != y), None)
                identical_eps += k is None and len(db) == len(dp)
                if k is not None:
                    first_div.append(k)
                n = len(db) if k is None else k                  # until the loops part, the inputs are the same
                total_dec += n
                same_dec += n
                ob, op = rb["outputs"], rp["outputs"]
                m = min(len(ob), len(op), sum(1 for x in db[:n] if x >= 0))
                for x, y in zip(ob[:m], op[:m]):
                    total_out += 1
                    d = int(np.abs(np.array(x, int) - np.array(y, int)).max())
                    same_out += d == 0
                    max_lsb = max(max_lsb, d)
        board_rec = next((r for r in recs if r.get("network") == f"{arm}_seed100_board" and r.get("kind_record") == "board"), {})
        loop_rec = next((r for r in recs if r.get("network") == f"{arm}_seed100_board" and "parts" in r), {})
        res["networks"][arm] = {"episodes_compared": episodes, "episodes_with_identical_decisions": identical_eps,
                                "outputs_identical_until_a_divergence": [same_out, total_out], "max_abs_lsb": max_lsb,
                                "episodes_whose_decisions_part": len(first_div), "timing": board_rec.get("timing"),
                                "firmware": board_rec.get("firmware"), "outcomes_board": loop_rec.get("parts")}
    rt = [r for r in recs if r.get("kind") == "real time"]
    if rt:
        res["real_time"] = rt[-1]
    SUMMARY.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"prepare": prepare, "run": run, "realtime": realtime, "--report": report}.get(
        step, lambda: sys.exit("prepare | run | realtime | --report"))()
