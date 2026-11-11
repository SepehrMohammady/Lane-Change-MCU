"""Hardware in the loop, step 1 (2026-11-10): replay the test split frame by frame to the STM32H7B3I-DK at the
protocol's sensor rate and check the board against the host.

The streaming firmware (firmware/h7b3, make STREAM=1) takes one frame of raw features at a time over the
ST-LINK virtual COM port, scales and quantizes it, keeps the last 10 frames and runs the network once the
window is full: 26 decisions per 35-frame scenario, as the protocol's windows. This script generates the
network's C code (stedgeai, as unas/local_bench.py), builds that firmware with the header of
unas/hil_prepare.py (run it first, in WSL), flashes it, reads the timed-run block over SWD, then sends every
test scenario at --rate frames per second (5 = the protocol's rate, 25 = highD's camera rate, 0 = as fast
as the link allows) and compares the board's raw int8 outputs with the host reference.

One JSON line per run goes to datasets/<ds>/results/hil/hil_stream.jsonl: exact agreement with the host,
the protocol's metrics from the board's outputs (src/lc_windows.early_metrics for the classifiers), cycles
per frame on the board (scaling, quantization and window copy; network) and the host's round trip.

LCIR networks (datasets/dmir/results/deploy/lcir_*_int8_io.tflite): every test window is one 50-frame
sequence with one decision (10 Hz in the recordings, so --rate 10 is real time); the record carries the
accuracy or the RMSE of the board's outputs.

  .venv\\Scripts\\python unas/hil_stream.py datasets/highd/results/deploy/<stem>.tflite [--rate 5] [--limit N] [--save]
  .venv\\Scripts\\python unas/hil_stream.py <tflite> --build-only        # no board needed
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_bench as lb                                    # noqa: E402  (tool paths, run, SWD helpers)

ROOT = lb.ROOT
sys.path.insert(0, str(ROOT))
from src.lc_windows import early_metrics                    # noqa: E402

BAUD = 921600
IN_LEN, SEQ_LEN = 10, 35                                    # highD and exiD; other lengths come with the reference


def build(tfl: Path) -> Path:
    tag = tfl.stem
    cfg = lb.FW / "build" / "stream" / tag
    if not (cfg / "stream_cfg.h").exists():
        sys.exit(f"run unas/hil_prepare.py {tfl.relative_to(ROOT).as_posix()} in WSL first")
    gen = lb.FW / "build" / "gen" / tag
    gen.mkdir(parents=True, exist_ok=True)
    lb.run([lb.STEDGEAI, "generate", "--model", tfl, "--target", "stm32", "--name", "network", "--c-api", "legacy",
            "--output", gen, "--workspace", lb.FW / "build" / "ws" / tag])
    lb.run([lb.MAKE, "STREAM=1", f"GEN=../build/gen/{tag}", f"CFG=../build/stream/{tag}"], cwd=lb.FW / "h7b3")
    return lb.FW / "build" / "h7b3stream" / tag / "fw.elf"


def open_port(sn: str):
    import serial
    import serial.tools.list_ports as lp
    st = [p for p in lp.comports() if p.vid == 0x0483]
    ports = [p for p in st if (p.serial_number or "").upper() == sn.upper()] or (st if len(st) == 1 else [])
    if not ports:
        sys.exit(f"no ST-LINK virtual COM port for the board's probe ({len(st)} ST ports found)")
    return serial.Serial(ports[0].device, BAUD, timeout=2.0)


def stream(port, feats: np.ndarray, rate: float, n_out: int, limit: int | None, in_len: int = IN_LEN):
    """Send each sequence frame by frame; returns board outputs (S, windows per sequence, n_out) int8,
    cycles and round trips."""
    S = len(feats) if limit is None else min(limit, len(feats))
    seq_len = feats.shape[1]
    reply_len = 12 + n_out
    outs = np.zeros((S, seq_len - in_len + 1, n_out), np.int8)
    pre, inf, rtt, late = [], [], [], []
    port.reset_input_buffer()
    period = 1.0 / rate if rate > 0 else 0.0
    t_next = time.perf_counter()
    seq = 0
    for s in range(S):
        for f in range(seq_len):
            if period:
                now = time.perf_counter()
                if now < t_next:
                    time.sleep(t_next - now)
                late.append(time.perf_counter() - t_next)
                t_next += period
            pkt = struct.pack("<BBH", 0xA5, 1 if f == 0 else 0, seq & 0xFFFF) + feats[s, f].astype("<f4").tobytes()
            t0 = time.perf_counter()
            port.write(pkt)
            rep = port.read(reply_len)
            rtt.append(time.perf_counter() - t0)
            if len(rep) != reply_len or rep[0] != 0x5A or struct.unpack_from("<H", rep, 2)[0] != (seq & 0xFFFF):
                raise RuntimeError(f"bad reply at scenario {s} frame {f}: {rep!r}")
            ran = rep[1] == 1
            if ran != (f >= in_len - 1):
                raise RuntimeError(f"scenario {s} frame {f}: the board ran={ran}")
            c_pre, c_inf = struct.unpack_from("<II", rep, 4)
            pre.append(c_pre)
            if ran:
                inf.append(c_inf)
                outs[s, f - in_len + 1] = np.frombuffer(rep[12:], np.int8)
            seq += 1
    return outs, np.array(pre), np.array(inf), np.array(rtt), np.array(late)


def stats(v: np.ndarray, scale: float = 1.0, nd: int = 4) -> dict:
    v = v * scale
    return {"mean": round(float(v.mean()), nd), "p50": round(float(np.median(v)), nd),
            "p99": round(float(np.percentile(v, 99)), nd), "max": round(float(v.max()), nd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tflite")
    ap.add_argument("--rate", type=float, default=5.0, help="frames per second; 0 = as fast as the link allows")
    ap.add_argument("--limit", type=int, default=None, help="first N test scenarios only")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--save", action="store_true",
                    help="keep the board's raw outputs and timings next to the reference (firmware/build/stream)")
    a = ap.parse_args()
    tfl = (ROOT / a.tflite).resolve()
    ds = tfl.relative_to(ROOT).parts[1]
    elf = build(tfl)
    print(f"built {elf.relative_to(ROOT).as_posix()}: {lb.elf_size(elf)}")
    if a.build_only:
        return
    ref = np.load(lb.FW / "build" / "stream" / tfl.stem / "reference.npz")
    sn = lb.find_stlink("STM32H7B3I-DK")
    lb.run([lb.CLI, "-c", "port=SWD", f"sn={sn}", "mode=UR", "-w", elf, "-v", "-rst"])
    magic, hz, runs, cmin, cmean, cmax, status, _ = lb.read_result(elf, sn)
    if status != 0:
        raise RuntimeError(f"firmware reported status {status:#x}")
    n_out = ref["ref"].shape[-1]
    in_len = int(ref["in_len"]) if "in_len" in ref else IN_LEN
    seq_len = ref["feats"].shape[1]
    n_slides = seq_len - in_len + 1
    with open_port(sn) as port:
        outs, pre, inf, rtt, late = stream(port, ref["feats"], a.rate, n_out, a.limit, in_len)
    S = len(outs)
    if a.save:
        tag = f"rate{a.rate:g}" + (f"_n{S}" if S < len(ref["feats"]) else "")
        np.savez_compressed(lb.FW / "build" / "stream" / tfl.stem / f"board_out_{tag}.npz",
                            outs=outs, pre=pre, inf=inf, rtt=rtt, late=late)
    want = ref["ref"][:S]
    label = ref["label"][:S]
    so, zo = float(ref["out_scale"]), int(ref["out_zp"])
    deq = (outs.astype(np.float32) - zo) * so
    rec = {"model": tfl.name, "board": "STM32H7B3I-DK", "core_hz": hz, "baud": BAUD,
           "rate_hz": a.rate, "scenarios": S, "frames": int(S * seq_len), "windows": int(S * n_slides),
           "agreement": {"identical_windows": round(float((outs == want).all(-1).mean()), 6),
                         "differing_windows": int((outs != want).any(-1).sum()),
                         "max_abs_lsb": int(np.abs(outs.astype(np.int16) - want.astype(np.int16)).max())},
           "timed_runs_cycles_mean": cmean,
           "board_cycles": {"preprocess": stats(pre, nd=1), "network": stats(inf, nd=1)},
           "board_ms": {"preprocess": stats(pre, 1e3 / hz), "network": stats(inf, 1e3 / hz)},
           "host_round_trip_ms": stats(rtt, 1e3, 3)}
    if a.rate > 0:
        rec["frame_lateness_ms"] = stats(late, 1e3, 3)
        # share of the frame period the core spends computing (the polling UART wait is not counted)
        rec["compute_share_of_period"] = round(float((pre.mean() + inf.mean() * n_slides / seq_len) / hz * a.rate), 8)
        # the same with a decision on every frame, as on a continuous stream
        rec["compute_share_every_frame"] = round(float((pre.mean() + inf.mean()) / hz * a.rate), 8)
    if ds == "dmir":
        if n_out == 3:
            rec["accuracy"] = round(float((deq[:, 0].argmax(-1) == label).mean()), 5)
        else:
            err = deq[:, 0, 0] - label
            rec["rmse_s"] = round(float(np.sqrt((err ** 2).mean())), 5)
            rec["mae_s"] = round(float(np.abs(err).mean()), 5)
    elif n_out == 3:
        z = deq - deq.max(-1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        m = early_metrics(probs, label)
        rec["protocol_metrics"] = {k: round(float(v), 4) for k, v in m.items() if not isinstance(v, (list, dict))}
    else:
        lc = label != 0
        cross = ref["cross"][:S].astype(np.float64)
        gt = (cross[:, None] - (np.arange(n_slides)[None, :] + in_len) + 1) / 5.0   # unas/highd_dataset.py
        err = deq[lc, :, 0] - gt[lc]
        rec["ttlc_rmse_s"] = round(float(np.sqrt((err ** 2).mean())), 4)
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(os.environ.get("HIL_OUT", ROOT / "datasets" / ds / "results" / "hil" / "hil_stream.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
