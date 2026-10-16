"""Interactive power session with a USB meter between the laptop and the board (run it yourself).

The meter (JT-UM120 here) measures the whole board at USB VBUS. For each step the script
flashes a firmware over SWD, lets the board settle, and asks you to type the meter's P-BUS
reading. Steps: core asleep (WFI) and core spinning (idle baselines, same clock and
peripherals, no network), then each model running inference in a loop (its latency is
measured in the same flash), then core asleep again to check drift.

    .venv\\Scripts\\python unas\\energy_session.py --board f401
    .venv\\Scripts\\python unas\\energy_session.py --board f401 datasets\\highd\\results\\deploy\\x.tflite ...

Per model, appended to datasets/<dataset>/results/deploy/energy_local.jsonl:
  p_run_W         mean P-BUS while the network runs back to back
  energy_board_mJ p_run_W x latency (whole board, upper bound)
  energy_mcu_mJ   (p_run_W - p_sleep_W) x latency (power above the asleep baseline)
Baselines are written to the same files as records named idle-sleep / idle-busy.
"""
from __future__ import annotations

import json
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_bench as lb  # noqa: E402

METER = "JT-UM120 USB meter, P-BUS at the board's USB input (laptop -> meter -> board)"
DEFAULT = {
    "h7b3axi": [
        "datasets/dmir/results/deploy/REF_cnn_multi_float32.tflite",
        "datasets/dmir/results/deploy/cls_best_float32.tflite",
        "datasets/dmir/results/qat/cls_best_qat_int8_io.tflite",
        "datasets/dmir/results/deploy/cls_tiny_float32.tflite",
        "datasets/dmir/results/deploy/lcir_dscnn_classification_float32.tflite",
        "datasets/dmir/results/deploy/lcir_dscnn_classification_int8_io.tflite",
        "datasets/dmir/results/deploy/lcir_v4_regression_lcl_float32.tflite",
        "datasets/dmir/results/deploy/transformer_lcl_fp32.tflite",
        "datasets/dmir/results/deploy/transformer_lcr_fp32.tflite",
        "datasets/highd/results/deploy/highd_cls_model_1790282740507531593_621856_f32.tflite",
        "datasets/highd/results/deploy/highd_cls_model_1790282740507531593_621856_int8_io.tflite",
        "datasets/highd/results/deploy/highd_cls_model_aaaaap_int8_io.tflite",
        "datasets/highd/results/deploy/highd_dscnn_cls_f32.tflite",
        "datasets/highd/results/deploy/highd_dscnn_cls_int8_io.tflite",
    ],
    "f401": [
        "datasets/dmir/results/deploy/cls_best_float32.tflite",
        "datasets/dmir/results/qat/cls_best_qat_int8_io.tflite",
        "datasets/dmir/results/deploy/cls_tiny_float32.tflite",
        "datasets/dmir/results/deploy/lcir_dscnn_classification_float32.tflite",
        "datasets/dmir/results/deploy/lcir_dscnn_classification_int8_io.tflite",
        "datasets/dmir/results/deploy/lcir_v4_regression_lcl_float32.tflite",
        "datasets/dmir/results/deploy/transformer_lcl_fp32.tflite",
        "datasets/highd/results/deploy/highd_cls_model_1790282740507531593_621856_f32.tflite",
        "datasets/highd/results/deploy/highd_cls_model_1790282740507531593_621856_int8_io.tflite",
        "datasets/highd/results/deploy/highd_cls_model_aaaaap_int8_io.tflite",
        "datasets/highd/results/deploy/highd_dscnn_cls_f32.tflite",
        "datasets/highd/results/deploy/highd_dscnn_cls_int8_io.tflite",
    ],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ask_power(label: str) -> list[float]:
    print(f"\n>>> {label}")
    print("    Wait until the meter settles (about 5 s). Then type three P-BUS readings in watts,")
    print("    a few seconds apart, separated by spaces (example: 0.37952 0.37960 0.37948).")
    while True:
        raw = input("    P-BUS [W]: ").replace(";", " ").split()
        try:
            vals = [float(t.strip(",").replace(",", ".")) for t in raw]
        except ValueError:
            vals = []
        if vals and all(0.01 < v < 5.0 for v in vals):
            return vals
        print("    Please type numbers in watts, for example: 0.37952 0.37960 0.37948")


def flash_idle(board: str, sn: str, mode: int) -> None:
    folder = lb.BOARDS[board][1]
    lb.run([lb.MAKE, f"IDLE={mode}"], cwd=lb.FW / folder)  # (H7: idle builds via the same Makefile option)
    elf = lb.FW / "build" / folder / f"idle{mode}" / "fw.elf"
    lb.run([lb.CLI, "-c", "port=SWD", f"sn={sn}", "mode=UR", "-w", elf, "-v", "-rst"])
    if mode == 1:
        return  # a sleeping core does not answer SWD reads; the flash was verified (-v)
    words = lb.read_result(elf, sn)
    if words[6] != (0x300 | mode):
        raise RuntimeError(f"idle firmware {mode} reported status {words[6]:#x}")


def append(ds: str, rec: dict) -> None:
    out = lb.ROOT / "datasets" / ds / "results" / "deploy" / "energy_local.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def plan(files: list[Path]) -> list[dict]:
    return ([{"kind": "idle", "mode": 1, "label": "idle-sleep"}, {"kind": "idle", "mode": 2, "label": "idle-busy"}]
            + [{"kind": "model", "file": str(f)} for f in files]
            + [{"kind": "idle", "mode": 1, "label": "idle-sleep-end"}])


def model_record(name: str, sn: str, f: Path, rec: dict, vals: list[float], base: dict) -> dict:
    p_run = st.mean(vals)
    t_s = rec["duration_ms"] / 1e3
    return {"model": f.name, "board": name, "board_sn": sn, "meter": METER, "readings_W": vals,
            "p_run_W": round(p_run, 6), "p_sleep_W": round(base["idle-sleep"], 6),
            "p_busy_W": round(base["idle-busy"], 6), "latency_ms": rec["duration_ms"],
            "cycles_mean": rec["cycles_mean"], "core_hz": rec["core_hz"],
            "energy_board_mJ": round(p_run * t_s * 1e3, 6),
            "energy_mcu_mJ": round((p_run - base["idle-sleep"]) * t_s * 1e3, 6),
            "stedgeai": rec["stedgeai"], "utc": now()}


def stepwise(board: str, cmd: str, rest: list[str]) -> None:
    """Non-interactive use, one step per call, readings typed by whoever reads the meter:
         energy_session.py --board f401 step <i> [model files for step 0 only]
         energy_session.py --board f401 record <i> <W> <W> <W>
    State between calls: firmware/build/energy_state_<board>.json (not versioned)."""
    path = lb.FW / "build" / f"energy_state_{board}.json"
    name = lb.BOARDS[board][0]
    i = int(rest[0])
    if cmd == "step" and i == 0:
        files = [str(Path(a).resolve()) for a in (rest[1:] or DEFAULT.get(board, []))]
        state = {"board": board, "files": files, "base": {}, "done": [], "current": None}
    else:
        state = json.loads(path.read_text(encoding="utf-8"))
    files = [Path(f) for f in state["files"]]
    steps = plan(files)
    datasets = sorted({f.relative_to(lb.ROOT / "datasets").parts[0] for f in files})
    sn = lb.find_stlink(name)
    s = steps[i]
    if cmd == "step":
        if s["kind"] == "idle":
            flash_idle(board, sn, s["mode"])
            state["current"] = {"step": i, **s}
            what = f"baseline {s['label']} (core {'asleep' if s['mode'] == 1 else 'spinning'}, no network)"
        else:
            rec = lb.measure(Path(s["file"]), board, sn)
            state["current"] = {"step": i, **s, "rec": rec}
            what = f"{Path(s['file']).name} running back to back ({rec['duration_ms']:.4f} ms per inference)"
        path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"STEP {i}/{len(steps) - 1} ready: {what}")
        return
    vals = [float(v) for v in rest[1:]]
    cur = state["current"]
    if not cur or cur["step"] != i:
        sys.exit(f"step {i} is not the flashed step ({cur and cur['step']})")
    if s["kind"] == "idle":
        label = s["label"]
        rec = {"model": "idle-sleep" if label == "idle-sleep-end" else label, "board": name, "board_sn": sn,
               "meter": METER, "readings_W": vals, "p_W": round(st.mean(vals), 6), "utc": now()}
        if label == "idle-sleep-end":
            rec["drift_W"] = round(st.mean(vals) - state["base"]["idle-sleep"], 6)
        else:
            state["base"][label] = st.mean(vals)
        for ds in datasets:
            append(ds, rec)
        print(f"recorded {label}: {st.mean(vals):.5f} W" + (f", drift {rec['drift_W']:+.5f} W" if "drift_W" in rec else ""))
    else:
        f = Path(s["file"])
        out = model_record(name, sn, f, cur["rec"], vals, state["base"])
        append(f.relative_to(lb.ROOT / "datasets").parts[0], out)
        print(f"recorded {f.name}: {out['p_run_W']:.5f} W, {out['p_run_W'] - out['p_sleep_W']:+.5f} W above asleep, "
              f"{out['energy_mcu_mJ']:.4f} mJ above asleep, {out['energy_board_mJ']:.4f} mJ whole board")
    state["done"].append(i)
    state["current"] = None
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")


def main():
    args = sys.argv[1:]
    board = "f401"
    if "--board" in args:
        i = args.index("--board")
        board = args[i + 1]
        del args[i:i + 2]
    if args and args[0] in ("step", "record"):
        return stepwise(board, args[0], args[1:])
    files = [Path(a).resolve() for a in (args or DEFAULT.get(board, []))]
    if not files or board not in lb.BOARDS:
        sys.exit(__doc__)
    name = lb.BOARDS[board][0]
    sn = lb.find_stlink(name)
    datasets = sorted({f.relative_to(lb.ROOT / "datasets").parts[0] for f in files})
    print(f"Board {name} (ST-LINK {sn}); {len(files)} models; meter: {METER}")

    base = {}
    for mode, label in ((1, "idle-sleep"), (2, "idle-busy")):
        print(f"\nFlashing baseline {label} ...")
        flash_idle(board, sn, mode)
        time.sleep(2)
        vals = ask_power(f"Baseline {label}: the core is {'asleep' if mode == 1 else 'spinning'}, no network")
        base[label] = st.mean(vals)
        for ds in datasets:
            append(ds, {"model": label, "board": name, "board_sn": sn, "meter": METER, "readings_W": vals,
                        "p_W": round(base[label], 6), "utc": now()})

    for f in files:
        ds = f.relative_to(lb.ROOT / "datasets").parts[0]
        print(f"\nMeasuring {f.name}: generate, build, flash, time 200 runs ...")
        rec = lb.measure(f, board, sn)
        time.sleep(2)
        vals = ask_power(f"{f.name} runs back to back ({rec['duration_ms']:.3f} ms per inference)")
        p_run = st.mean(vals)
        t_s = rec["duration_ms"] / 1e3
        out = {"model": f.name, "board": name, "board_sn": sn, "meter": METER, "readings_W": vals,
               "p_run_W": round(p_run, 6), "p_sleep_W": round(base["idle-sleep"], 6),
               "p_busy_W": round(base["idle-busy"], 6), "latency_ms": rec["duration_ms"],
               "cycles_mean": rec["cycles_mean"], "core_hz": rec["core_hz"],
               "energy_board_mJ": round(p_run * t_s * 1e3, 6),
               "energy_mcu_mJ": round((p_run - base["idle-sleep"]) * t_s * 1e3, 6),
               "stedgeai": rec["stedgeai"], "utc": now()}
        append(ds, out)
        print(f"    {p_run:.5f} W, {p_run - base['idle-sleep']:+.5f} W above asleep -> "
              f"{out['energy_mcu_mJ']:.4f} mJ per inference above asleep, {out['energy_board_mJ']:.4f} mJ whole board")

    print("\nFlashing baseline idle-sleep again (drift check) ...")
    flash_idle(board, sn, 1)
    time.sleep(2)
    vals = ask_power("Baseline idle-sleep again")
    for ds in datasets:
        append(ds, {"model": "idle-sleep", "board": name, "board_sn": sn, "meter": METER, "readings_W": vals,
                    "p_W": round(st.mean(vals), 6), "drift_W": round(st.mean(vals) - base["idle-sleep"], 6),
                    "utc": now()})
    print(f"\nDone. Drift of the asleep baseline: {st.mean(vals) - base['idle-sleep']:+.5f} W. "
          f"Records in datasets/{{{','.join(datasets)}}}/results/deploy/energy_local.jsonl")


if __name__ == "__main__":
    main()
