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
           .venv\\Scripts\\python unas/online_board.py measures --loc 5       # its test recordings: board, host, fixed
           .venv\\Scripts\\python unas/online_board.py measures-all           # all test recordings, once all replayed
           .venv\\Scripts\\python unas/online_board.py curve                  # windows correct against time deployed
           .venv\\Scripts\\python unas/online_board.py --report
With --net lolo --loc L (2026-11-24): exiD's network trained without location L (unas/online_lolo.py) at location L, with
the check of unas/online_switch.py compiled in (ONL_CHECK_N): the board answers with the class the check chooses, and
the host copy gives the update's and the check's classes.
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


def tag_of(seed: int, net: str = "highd", loc: int | None = None) -> str:
    """highd: the highD arm K network (all exiD locations); lolo: exiD's network trained without location loc."""
    return f"highd_K_seed{seed}_features_int8_io" if net == "highd" else f"lolo_loc{loc}_seed{seed}_features_int8_io"


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
def int8_streams(it, views: dict, w: np.ndarray, si: float, zi: int, D: int):
    """The int8 input windows of one location's stream (rows of windows_by_location, device order) and the int8
    features the TFLite feature file gives for them."""
    import online_site_pilot as site
    import track_stream as ts
    n_feat = views["train"][0].shape[1]
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
    return q, fq


def header(tag: str, note: str, D: int, setting: dict, so: float, zo: int, w0, b0, check_n: int | None = None) -> str:
    rows = ",\n".join("    {" + ", ".join(f"{v:.9e}f" for v in r) + "}" for r in w0.astype(np.float32))
    return (f"/* Generated by unas/online_board.py prepare ({tag}, {note}); not versioned. */\n"
            f"#define ONL_D {D}\n#define ONL_C 3\n#define ONL_MEM {MEM}\n#define ONL_PER {PER}\n#define ONL_MIN {MIN_CLASS}\n"
            f"#define ONL_RING {RING}\n#define ONL_SEED {SEED}u\n#define ONL_ETA {setting['eta']:.9e}f\n"
            f"#define ONL_LAM {setting['lam']:.9e}f\n#define ONL_FEAT_SCALE {so:.9e}f\n#define ONL_FEAT_ZP {int(zo)}\n"
            + (f"#define ONL_CHECK_N {check_n}\n" if check_n else "")
            + f"static const float onl_w0[ONL_D][ONL_C] = {{\n{rows}}};\n"
            f"static const float onl_b0[ONL_C] = {{" + ", ".join(f"{v:.9e}f" for v in b0.astype(np.float32)) + "};\n")


def prepare_lolo(seed: int, loc: int) -> None:
    """exiD's network trained without location loc (unas/online_lolo.py) at that location, with the update and the
    check of unas/online_switch.py (its selected N): feature file, header with ONL_CHECK_N, the location's stream and
    the host copy of the board's algorithm (update, check) in loc<loc>.npz."""
    import input_scaling_check as isc
    import online_lolo as lolo
    import online_site_pilot as site
    import online_switch as osw

    keras, tf = isc.keras, isc.tf
    check_n = osw.selection(osw.records())["selected"]
    setting = osw.SETTING
    feat, w0, b0 = lolo.network(keras, loc, seed)
    sc = np.load(lolo.KEEP / f"scaling_without_loc{loc}.npz")
    lo, span = sc["lo"], sc["span"]
    xtr, _, _ = lolo.track_windows_without("train", lo, span, np.random.default_rng(0), loc)
    tfl = isc.convert_int8_io(feat, xtr)
    tag = tag_of(seed, "lolo", loc)
    d = OUTDIR / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.tflite").write_bytes(tfl)
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    (si, zi), (so, zo) = it.get_input_details()[0]["quantization"], it.get_output_details()[0]["quantization"]
    D = int(w0.shape[0])
    (d / "online_cfg.h").write_text(header(tag, f"setting {setting['id']}, check N {check_n}", D, setting, so, zo, w0, b0,
                                           check_n), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"tag": tag, "seed": seed, "location": loc, "setting": setting,
                                             "check_n": check_n, "in_quant": [float(si), int(zi)],
                                             "feat_quant": [float(so), int(zo)], "features": D}, indent=1))
    t0 = time.perf_counter()
    w = site.windows_by_location()[loc]
    q, fq = int8_streams(it, lolo.views_with(lo, span), w, si, zi, D)
    fixed = np.argmax(np.float32(so) * (fq.astype(np.float32) - np.float32(zo)) @ w0.astype(np.float32)
                      + b0.astype(np.float32), axis=1).astype(np.int8)
    adapted, steps = emulate(fq, w, w0, b0, float(so), int(zo), setting["eta"], setting["lam"])
    check, used = osw.switched(w, fixed, adapted, check_n)
    np.savez(d / f"loc{loc}.npz", q=q, w=w, fixed=fixed, adapted=adapted, check=check)
    known = w[:, 4].astype(bool)
    acc = {k: round(float((a[known] == w[known, 3]).mean()), 5) for k, a in (("fixed", fixed), ("update", adapted), ("check", check))}
    print(f"location {loc}: {len(w)} windows, {steps} steps, update used on {used:.3f} of the frames, "
          f"windows correct (all splits, label known) {acc}, {time.perf_counter() - t0:.0f} s", flush=True)


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
    (d / "online_cfg.h").write_text(header(tag, f"setting {sel}", D, setting, so, zo, w0, b0), encoding="utf-8")
    meta = {"tag": tag, "seed": seed, "setting": setting, "in_quant": [float(si), int(zi)], "feat_quant": [float(so), int(zo)],
            "features": D}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))

    by_loc = site.windows_by_location()
    views = site.scaled_views(prep)
    preds = {"fixed": {}, "adapted": {}}
    for loc, w in sorted(by_loc.items()):
        t0 = time.perf_counter()
        q, fq = int8_streams(it, views, w, si, zi, D)
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
def build(seed: int, net: str = "highd", loc: int | None = None) -> None:
    import local_bench as lb
    tag = tag_of(seed, net, loc)
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


def replay(seed: int, loc: int, net: str = "highd") -> None:
    import hil_stream as hs
    import local_bench as lb
    import online_site_pilot as site
    tag = tag_of(seed, net, loc)
    z = np.load(OUTDIR / tag / f"loc{loc}.npz")
    reference = "check" if "check" in z.files else "update"
    q, w, ref = z["q"], z["w"], z["check" if reference == "check" else "adapted"]
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
    rec = {"kind": "replay", "tag": tag, "location": loc, "host_copy_of": reference, "windows": int(len(w)), "messages": int(seq),
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


def measures(seed: int, loc: int, net: str = "highd") -> None:
    """The site measures of the board's classes on the location's test recordings, next to the int8 fixed network
    and the host copy (track order of exiD's test tracks; only the location's tracks)."""
    import online_site_pilot as site
    ts, tsx = site.ts, site.tsx
    tag = tag_of(seed, net, loc)
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
    sets = (("fixed_int8", z["fixed"]), ("host_copy", z["adapted"]), ("board", board))
    if "check" in z.files:
        sets = (("fixed_int8", z["fixed"]), ("host_update", z["adapted"]), ("host_check", z["check"]), ("board", board))
    for key, cls in sets:
        full = np.full(int(w_start[-1] + lens[-1] - ts.IN_LEN + 1), -1, np.int8)
        full[w[test, 1]] = cls[test]
        p = np.concatenate([full[w_start[t]:w_start[t] + lens[t] - ts.IN_LEN + 1] for t in tracks])
        assert (p >= 0).all()
        rec[key] = site.compact(ts.measures_of(p, lens[tracks], [cross[t] for t in tracks]))
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


def measures_all(seed: int) -> None:
    """The site measures over the test recordings of every replayed location at once (exiD test track order), for
    the board's classes, the host copy and the int8 fixed network; written only when all locations are replayed."""
    import online_site_pilot as site
    ts, tsx = site.ts, site.tsx
    tag = tag_of(seed)
    tt = np.load(tsx.tracks_file("test"))
    lens, cross = tt["lens"].astype(np.int64), json.loads(str(tt["cross"]))
    w_start = np.concatenate([[0], np.cumsum(lens - ts.IN_LEN + 1)[:-1]])
    n_win = int(w_start[-1] + lens[-1] - ts.IN_LEN + 1)
    full = {k: np.full(n_win, -1, np.int8) for k in ("fixed_int8", "host_copy", "board")}
    locs, tracks = [], []
    for f in sorted((OUTDIR / tag).glob("loc[0-9]*_board.npz")):
        loc = int(f.stem[3:].split("_")[0])
        z = np.load(OUTDIR / tag / f"loc{loc}.npz")
        w, board = z["w"], np.load(f)["board"]
        test = w[:, 0] == site.SPLITS.index("test")
        for key, cls in (("fixed_int8", z["fixed"]), ("host_copy", z["adapted"]), ("board", board)):
            full[key][w[test, 1]] = cls[test]
        locs.append(loc)
        tracks += [int(t) for t in np.unique(w[test, 2])]
    tracks = np.array(sorted(set(tracks)))
    if len(tracks) != len(lens):
        raise RuntimeError(f"{len(tracks)} of {len(lens)} test tracks replayed (locations {locs})")
    rec = {"kind": "replay_measures_all", "tag": tag, "locations": locs,
           "recordings": sorted({int(r) for r in tt["rec"][tracks]}), "tracks": int(len(tracks))}
    for key in ("fixed_int8", "host_copy", "board"):
        p = np.concatenate([full[key][w_start[t]:w_start[t] + lens[t] - ts.IN_LEN + 1] for t in tracks])
        assert (p >= 0).all()
        rec[key] = site.compact(ts.measures_of(p, lens[tracks], [cross[t] for t in tracks]))
    rec["test_windows_equal_board_host"] = round(float((full["board"] == full["host_copy"]).mean()), 6)
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


def curve(seed: int, block_min: float = 5.0) -> None:
    """How fast the board adapts after deployment (analysis added after the results): per location, device time runs
    through its recordings in stream order (each recording's 25 Hz frames, recordings back to back); windows correct
    among label-known windows per block of device time, for the board and the int8 fixed network, and the time of the
    first update (every class with MIN_CLASS labels released, a label 26 frames after its window)."""
    tag = tag_of(seed)
    edges_min = (0, 2, 10, 30, 60, 120, float("inf"))
    per_loc, pooled = {}, {k: np.zeros((len(edges_min) - 1, 3)) for k in ("all",)}
    for f in sorted((OUTDIR / tag).glob("loc[0-9]*_board.npz")):
        loc = int(f.stem[3:].split("_")[0])
        z = np.load(OUTDIR / tag / f"loc{loc}.npz")
        w, fixed, board = z["w"], z["fixed"], np.load(f)["board"]
        rec, t_abs, known, label = w[:, 6], w[:, 5], w[:, 4].astype(bool), w[:, 3]
        order = list(dict.fromkeys(rec.tolist()))                        # recordings in stream order
        offset, t0 = {}, 0
        for r in order:
            offset[r] = t0
            t0 += int(t_abs[rec == r].max()) + 1
        t_dev = np.array([offset[r] for r in rec]) + t_abs               # 25 Hz frames since deployment
        rel = t_dev[known] + OFFSET
        cls = label[known][np.argsort(rel, kind="stable")]
        cnt = np.zeros((len(cls), 3), np.int64)
        cnt[np.arange(len(cls)), cls] = 1
        warm = np.flatnonzero((np.cumsum(cnt, axis=0) >= MIN_CLASS).all(axis=1))
        t_warm = float(np.sort(rel)[warm[0]]) / 25 / 60 if len(warm) else None
        blk = (t_dev[known] / 25 / 60 // block_min).astype(int)
        nb = int(blk.max()) + 1
        n = np.bincount(blk, minlength=nb)
        okf = np.bincount(blk, weights=(fixed[known] == label[known]), minlength=nb)
        okb = np.bincount(blk, weights=(board[known] == label[known]), minlength=nb)
        mins = t_dev[known] / 25 / 60
        iv = {}
        for a, b in zip(edges_min[:-1], edges_min[1:]):
            s = (mins >= a) & (mins < b)
            key = f"{a:g}-{b:g}_min"
            iv[key] = {"windows": int(s.sum()),
                       "fixed": round(float((fixed[known][s] == label[known][s]).mean()), 4) if s.any() else None,
                       "board": round(float((board[known][s] == label[known][s]).mean()), 4) if s.any() else None}
        per_loc[str(loc)] = {"device_hours": round(t0 / 25 / 3600, 3), "minutes_to_first_update": round(t_warm, 2),
                             "intervals": iv,
                             "blocks": {"windows": n.tolist(), "fixed": np.round(okf / np.maximum(n, 1), 4).tolist(),
                                        "board": np.round(okb / np.maximum(n, 1), 4).tolist()}}
        for j, (a, b) in enumerate(zip(edges_min[:-1], edges_min[1:])):
            s = (mins >= a) & (mins < b)
            pooled["all"][j] += [s.sum(), (fixed[known][s] == label[known][s]).sum(), (board[known][s] == label[known][s]).sum()]
    pi = {f"{a:g}-{b:g}_min": {"windows": int(p[0]), "fixed": round(float(p[1] / p[0]), 4), "board": round(float(p[2] / p[0]), 4)}
          for (a, b), p in zip(zip(edges_min[:-1], edges_min[1:]), pooled["all"]) if p[0] > 0}
    rec_out = {"kind": "adaptation_curve", "tag": tag, "block_min": block_min, "measure": "windows correct, label known, "
               "all splits as the device streams them", "per_location": per_loc, "pooled_intervals": pi,
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(RECORDS, "a") as fh:
        fh.write(json.dumps(rec_out) + "\n")
    print(json.dumps({k: rec_out[k] for k in ("pooled_intervals",)}, indent=1))
    for loc, d in per_loc.items():
        print(loc, d["device_hours"], "h; first update after", d["minutes_to_first_update"], "min;",
              {k: (v["fixed"], v["board"]) for k, v in d["intervals"].items()})


def report() -> None:
    rows = [json.loads(line) for line in RECORDS.read_text().splitlines()] if RECORDS.exists() else []
    for r in rows:
        print(json.dumps(r))
    if REFERENCE.exists():
        ref = json.loads(REFERENCE.read_text())
        print("host copy, exiD test recordings:", json.dumps({k: ref[k]["all"] for k in ("fixed", "adapted")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("prepare", "build", "replay", "measures", "measures-all", "curve"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--loc", type=int, default=5)
    ap.add_argument("--net", choices=("highd", "lolo"), default="highd")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.cmd == "prepare":
        prepare_lolo(a.seed, a.loc) if a.net == "lolo" else prepare(a.seed)
    elif a.cmd == "build":
        build(a.seed, a.net, a.loc if a.net == "lolo" else None)
    elif a.cmd == "replay":
        replay(a.seed, a.loc, a.net)
    elif a.cmd == "measures":
        measures(a.seed, a.loc, a.net)
    elif a.cmd == "measures-all":
        measures_all(a.seed)
    elif a.cmd == "curve":
        curve(a.seed)
    else:
        ap.error("prepare | build | replay | measures | measures-all | curve | --report")
