"""On-device learning on the STM32H7B3I-DK (2026-11-19): the update of unas/online_site_pilot.py run by the board.

The highD network (arm K, one seed) up to its Flatten layer is converted to int8 (int8 input and output, calibrated
as the deployed network: input_scaling_check.convert_int8_io on arm K's training windows) and generated with ST Edge
AI; the last layer, the class memories (int8) and the gradient step run in C on the board (firmware/h7b3, make
ONLINE=1; protocol in main.c). The host replays exiD locations as the site pilot's device sees them: per frame, first
the labels that became known (26 frames after their window, within the recording), then the windows of every vehicle
in view. The update is the setting the site pilot selected. A host copy of the board's algorithm (int8 features from
the TFLite file, float32, the same xorshift32 draws) is the reference the board is compared with.

  WSL:     ~/dmir_nas/bin/python unas/online_board.py prepare [--seed 100]   # feature file, header, streams, reference
  Windows: .venv\\Scripts\\python unas/online_board.py build                  # generate, build, flash, timed runs
           .venv\\Scripts\\python unas/online_board.py replay --loc 5         # one location through the board
           .venv\\Scripts\\python unas/online_board.py --report
Streams and references stay in firmware/build/online/<tag>/ (exiD features: not versioned). Records:
datasets/exid/results/hil/online_board.jsonl and online_board_reference.json.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
OUTDIR = ROOT / "firmware" / "build" / "online"
RES = ROOT / "datasets" / "exid" / "results" / "hil"
RECORDS = RES / "online_board.jsonl"
REFERENCE = RES / "online_board_reference.json"
MEM, PER, MIN_CLASS, RING, SEED = 128, 4, 20, 4096, 2463534242
OFFSET = 130                                                 # 26 frames at 5 Hz, in 25 Hz frame numbers
BAUD = 921600


def tag_of(seed: int) -> str:
    return f"highd_K_seed{seed}_features_int8_io"


def xorshift(x: int) -> int:
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def messages(w: np.ndarray):
    """The device's messages for one location (rows of online_site_pilot.windows_by_location): per frame, the
    window indices whose labels are released (in release order) and the window indices of the frame."""
    from collections import deque
    known, t_abs, rec = w[:, 4].astype(bool), w[:, 5], w[:, 6]
    cut = np.flatnonzero((np.diff(t_abs) != 0) | (np.diff(rec) != 0)) + 1
    pend = deque()
    for g0, g1 in zip(np.concatenate([[0], cut]), np.concatenate([cut, [len(w)]])):
        r, t = rec[g0], t_abs[g0]
        while pend and rec[pend[0]] != r:
            pend.popleft()
        rel = []
        while pend and t_abs[pend[0]] + OFFSET <= t:
            rel.append(pend.popleft())
        yield rel, list(range(g0, g1))
        pend.extend(i for i in range(g0, g1) if known[i])


def emulate(fq: np.ndarray, w: np.ndarray, w0, b0, scale: float, zp: int, eta: float, lam: float):
    """The board's algorithm on the host: int8 features fq, float32 arithmetic, the same draws."""
    f32 = np.float32
    W0, B0 = w0.astype(f32), b0.astype(f32)
    Wt, Bt = W0.copy(), B0.copy()
    deq = lambda q: (q.astype(f32) - f32(zp)) * f32(scale)          # noqa: E731
    mem = np.zeros((3, MEM, fq.shape[1]), np.int8)
    cnt, pos = [0, 0, 0], [0, 0, 0]
    xr, steps = SEED, 0
    label = w[:, 3]
    onehot = np.repeat(np.eye(3, dtype=f32), PER, axis=0)
    out = np.empty(len(fq), np.int8)
    for rel, frame in messages(w):
        for i in rel:
            c = int(label[i])
            mem[c, pos[c]] = fq[i]
            pos[c] = (pos[c] + 1) % MEM
            cnt[c] = min(cnt[c] + 1, MEM)
            if min(cnt) >= MIN_CLASS:
                idx = []
                for k in range(3):
                    for _ in range(PER):
                        xr = xorshift(xr)
                        idx.append((k, xr % cnt[k]))
                xs = deq(np.stack([mem[k, j] for k, j in idx]))
                z = Bt + xs @ Wt
                z = np.exp(z - z.max(1, keepdims=True))
                g = (z / z.sum(1, keepdims=True) - onehot) / f32(len(xs))
                Wt = Wt - f32(eta) * (xs.T @ g + f32(2 * lam) * (Wt - W0))
                Bt = Bt - f32(eta) * (g.sum(0) + f32(2 * lam) * (Bt - B0))
                steps += 1
        if frame:
            out[frame[0]:frame[-1] + 1] = np.argmax(Bt + deq(fq[frame[0]:frame[-1] + 1]) @ Wt, axis=1)
    return out, steps


# ------------------------------------------------------------------------------------------------- WSL: prepare
def prepare(seed: int) -> None:
    import input_scaling_check as isc
    import online_site_pilot as site
    import track_train_check as ttc                          # highD (TRACK_DATASET unset)
    import track_stream as ts

    keras, tf = isc.keras, isc.tf
    sel = site.selection(site.summaries(site.records()))["selected"]
    setting = next(s for s in site.SETTINGS if s["id"] == sel)
    keep, arm, prep = site.NETS["highD_K"]
    feat, w0, b0 = site.network(keras, keep, arm, seed)
    lo, span = ttc.scaling()
    xtr, _, _ = ttc.track_windows("train", lo, span, np.random.default_rng(0), True)
    tfl = isc.convert_int8_io(feat, xtr)
    tag = tag_of(seed)
    d = OUTDIR / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.tflite").write_bytes(tfl)
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    (si, zi), (so, zo) = it.get_input_details()[0]["quantization"], it.get_output_details()[0]["quantization"]
    D = int(w0.shape[0])
    rows = ",\n".join("    {" + ", ".join(f"{v:.9e}f" for v in r) + "}" for r in w0.astype(np.float32))
    (d / "online_cfg.h").write_text(
        f"/* Generated by unas/online_board.py prepare ({tag}, setting {sel}); not versioned. */\n"
        f"#define ONL_D {D}\n#define ONL_C 3\n#define ONL_MEM {MEM}\n#define ONL_PER {PER}\n#define ONL_MIN {MIN_CLASS}\n"
        f"#define ONL_RING {RING}\n#define ONL_SEED {SEED}u\n#define ONL_ETA {setting['eta']:.9e}f\n"
        f"#define ONL_LAM {setting['lam']:.9e}f\n#define ONL_FEAT_SCALE {so:.9e}f\n#define ONL_FEAT_ZP {int(zo)}\n"
        f"static const float onl_w0[ONL_D][ONL_C] = {{\n{rows}}};\n"
        f"static const float onl_b0[ONL_C] = {{" + ", ".join(f"{v:.9e}f" for v in b0.astype(np.float32)) + "};\n",
        encoding="utf-8")
    meta = {"tag": tag, "seed": seed, "setting": setting, "in_quant": [float(si), int(zi)], "feat_quant": [float(so), int(zo)],
            "features": D}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))

    by_loc = site.windows_by_location()
    views = site.scaled_views(prep)
    n_feat = views["train"][0].shape[1]
    preds = {"fixed": {}, "adapted": {}}
    for loc, w in sorted(by_loc.items()):
        t0 = time.perf_counter()
        q = np.empty((len(w), ts.IN_LEN, n_feat), np.int8)
        for rank, split in enumerate(site.SPLITS):
            sel_rows = np.flatnonzero(w[:, 0] == rank)
            view, first = views[split]
            for a in range(0, len(sel_rows), 200_000):
                s = sel_rows[a:a + 200_000]
                x = np.ascontiguousarray(view[first[w[s, 1]]].transpose(0, 2, 1))
                q[s] = np.clip(np.rint(x / np.float32(si) + np.float32(zi)), -128, 127).astype(np.int8)
        fq = np.empty((len(w), D), np.int8)
        inp, out = it.get_input_details()[0], it.get_output_details()[0]
        B = 8192
        it.resize_tensor_input(inp["index"], [B, ts.IN_LEN, n_feat])
        it.allocate_tensors()
        for a in range(0, len(q), B):
            chunk = q[a:a + B]
            if len(chunk) < B:
                chunk = np.concatenate([chunk, np.zeros((B - len(chunk),) + chunk.shape[1:], np.int8)])
            it.set_tensor(inp["index"], chunk)
            it.invoke()
            fq[a:a + B] = it.get_tensor(out["index"]).reshape(B, D)[:min(B, len(q) - a)]
        fixed = np.argmax(np.float32(so) * (fq.astype(np.float32) - np.float32(zo)) @ w0.astype(np.float32)
                          + b0.astype(np.float32), axis=1).astype(np.int8)
        adapted, steps = emulate(fq, w, w0, b0, float(so), int(zo), setting["eta"], setting["lam"])
        np.savez(d / f"loc{loc}.npz", q=q, w=w, fixed=fixed, adapted=adapted)
        for key, arr in (("fixed", fixed), ("adapted", adapted)):
            preds[key][loc] = arr
        print(f"location {loc}: {len(w)} windows, {steps} steps, {time.perf_counter() - t0:.0f} s", flush=True)
    rank = site.SPLITS.index("test")
    n_test = int((np.load(site.tsx.tracks_file("test"))["lens"] - ts.IN_LEN + 1).sum())
    res = {"tag": tag, "seed": seed, "setting": sel, "note": "host copy of the board's algorithm: int8 features of the "
           "TFLite feature file, float32 last layer and update, xorshift32 draws; exiD test recordings, prequential",
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key in ("fixed", "adapted"):
        full = np.full(n_test, -1, np.int8)
        for loc, w in by_loc.items():
            sc = w[:, 0] == rank
            full[w[sc, 1]] = preds[key][loc][sc]
        assert (full >= 0).all()
        res[key] = site.split_measures("test", full)
    RES.mkdir(parents=True, exist_ok=True)
    REFERENCE.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k]["all"] for k in ("fixed", "adapted")}, indent=1))


# ------------------------------------------------------------------------------------------------- Windows: board
def build(seed: int) -> None:
    import local_bench as lb
    tag = tag_of(seed)
    d = OUTDIR / tag
    gen = lb.FW / "build" / "gen" / tag
    gen.mkdir(parents=True, exist_ok=True)
    lb.run([lb.STEDGEAI, "generate", "--model", d / f"{tag}.tflite", "--target", "stm32", "--name", "network",
            "--c-api", "legacy", "--output", gen, "--workspace", lb.FW / "build" / "ws" / tag])
    info = json.loads((gen / "network_c_info.json").read_text(encoding="utf-8"))
    lb.run([lb.MAKE, "ONLINE=1", f"GEN=../build/gen/{tag}", f"OCFG=../build/online/{tag}"], cwd=lb.FW / "h7b3")
    elf = lb.FW / "build" / "h7b3online" / tag / "fw.elf"
    sn = lb.find_stlink("STM32H7B3I-DK")
    lb.run([lb.CLI, "-c", "port=SWD", f"sn={sn}", "mode=UR", "-w", elf, "-v", "-rst"])
    magic, hz, runs, cmin, cmean, cmax, status, loops = lb.read_result(elf, sn)
    if status:
        raise RuntimeError(f"firmware status {status:#x}")
    nodes = info["graphs"][0]["nodes"]
    mem = info.get("memory_footprint", {})
    rec = {"kind": "build", "tag": tag, "feature_network_ms": round(cmean / hz * 1e3, 5), "cycles_mean": cmean,
           "core_hz": hz, "macc": sum(n.get("macc", 0) for n in nodes), "weights_B": mem.get("weights"),
           "activations_B": mem.get("activations"), **lb.elf_size(elf),
           "online_state_B": {"class_memories_int8": 3 * MEM * json.loads((OUTDIR / tag / "meta.json").read_text())["features"],
                              "pending_ring_int8": RING * json.loads((OUTDIR / tag / "meta.json").read_text())["features"]},
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


def replay(seed: int, loc: int) -> None:
    import hil_stream as hs
    import local_bench as lb
    import online_site_pilot as site
    tag = tag_of(seed)
    z = np.load(OUTDIR / tag / f"loc{loc}.npz")
    q, w, ref = z["q"], z["w"], z["adapted"]
    sn = lb.find_stlink("STM32H7B3I-DK")
    port = hs.open_port(sn)
    port.timeout = 5.0
    port.reset_input_buffer()
    label = w[:, 3]
    board = np.full(len(w), -1, np.int8)
    cyc_l, cyc_w, n_l, n_w, rtt = [], [], [], [], []
    seq, first, t_start = 0, True, time.perf_counter()
    in_bytes = q.shape[1] * q.shape[2]
    for rel, frame in messages(w):
        for i in rel:
            if frame and frame[0] - i >= RING:
                raise RuntimeError(f"window {i} left the ring before its label")
        parts = [(rel[a:a + 255], []) for a in range(0, len(rel), 255)] or [([], [])]
        parts[-1] = (parts[-1][0], frame[:255])
        for a in range(255, len(frame), 255):
            parts.append(([], frame[a:a + 255]))
        for lab, win in parts:
            msg = bytearray(struct.pack("<BBBBH", 0xA6, len(lab), len(win), 1 if first else 0, seq & 0xFFFF))
            for i in lab:
                msg += struct.pack("<HB", i % 65536, int(label[i]))
            for i in win:
                msg += struct.pack("<H", i % 65536) + q[i].tobytes()
            t0 = time.perf_counter()
            port.write(msg)
            rep = port.read(12 + len(win))
            rtt.append(time.perf_counter() - t0)
            if len(rep) != 12 + len(win) or rep[0] != 0x5B or struct.unpack_from("<H", rep, 2)[0] != (seq & 0xFFFF):
                raise RuntimeError(f"bad reply at message {seq}: {rep[:16]!r}")
            cl, cw = struct.unpack_from("<II", rep, 4)
            cyc_l.append(cl), cyc_w.append(cw), n_l.append(len(lab)), n_w.append(len(win))
            if win:
                board[win[0]:win[-1] + 1] = np.frombuffer(rep[12:], np.int8)
            seq, first = seq + 1, False
        if seq % 2000 == 0:
            print(f"  {seq} messages, {int((board >= 0).sum())} windows, {time.perf_counter() - t_start:.0f} s", flush=True)
    port.close()
    assert (board >= 0).all()
    hz = 280e6
    cyc_l, cyc_w, n_l, n_w = map(np.array, (cyc_l, cyc_w, n_l, n_w))
    test = w[:, 0] == site.SPLITS.index("test")
    rec = {"kind": "replay", "tag": tag, "location": loc, "windows": int(len(w)), "messages": int(seq),
           "labels": int(n_l.sum()), "agreement_with_host_copy": round(float((board == ref).mean()), 6),
           "agreement_on_test_windows": round(float((board[test] == ref[test]).mean()), 6) if test.any() else None,
           "test_windows": int(test.sum()),
           "board_ms_per_window": round(float(cyc_w.sum() / max(n_w.sum(), 1) / hz * 1e3), 5),
           "board_ms_per_label": round(float(cyc_l.sum() / max(n_l.sum(), 1) / hz * 1e3), 5),
           "wall_s": round(time.perf_counter() - t_start, 1), "round_trip_ms_mean": round(float(np.mean(rtt)) * 1e3, 3),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    np.savez(OUTDIR / tag / f"loc{loc}_board.npz", board=board)
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


def measures(seed: int, loc: int) -> None:
    """The site measures of the board's classes on the location's test recordings, next to the int8 fixed network
    and the host copy (track order of exiD's test tracks; only the location's tracks)."""
    import online_site_pilot as site
    ts, tsx = site.ts, site.tsx
    tag = tag_of(seed)
    z = np.load(OUTDIR / tag / f"loc{loc}.npz")
    w = z["w"]
    board = np.load(OUTDIR / tag / f"loc{loc}_board.npz")["board"]
    tt = np.load(tsx.tracks_file("test"))
    lens, cross = tt["lens"].astype(np.int64), json.loads(str(tt["cross"]))
    w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
    test = w[:, 0] == site.SPLITS.index("test")
    tracks = np.unique(w[test, 2])
    rec = {"kind": "replay_measures", "tag": tag, "location": loc,
           "recordings": sorted({int(r) for r in tt["rec"][tracks]}), "tracks": int(len(tracks))}
    for key, cls in (("fixed_int8", z["fixed"]), ("host_copy", z["adapted"]), ("board", board)):
        full = np.full(int(w_start[-1] + lens[-1] - ts.IN_LEN + 1), -1, np.int8)
        full[w[test, 1]] = cls[test]
        p = np.concatenate([full[w_start[t]:w_start[t] + lens[t] - ts.IN_LEN + 1] for t in tracks])
        assert (p >= 0).all()
        rec[key] = site.compact(ts.measures_of(p, lens[tracks], [cross[t] for t in tracks]))
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


def report() -> None:
    rows = [json.loads(line) for line in RECORDS.read_text().splitlines()] if RECORDS.exists() else []
    for r in rows:
        print(json.dumps(r))
    if REFERENCE.exists():
        ref = json.loads(REFERENCE.read_text())
        print("host copy, exiD test recordings:", json.dumps({k: ref[k]["all"] for k in ("fixed", "adapted")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("prepare", "build", "replay", "measures"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--loc", type=int, default=5)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.cmd == "prepare":
        prepare(a.seed)
    elif a.cmd == "build":
        build(a.seed)
    elif a.cmd == "replay":
        replay(a.seed, a.loc)
    elif a.cmd == "measures":
        measures(a.seed, a.loc)
    else:
        ap.error("prepare | build | replay | --report")
