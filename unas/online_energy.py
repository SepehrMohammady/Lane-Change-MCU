"""Power session for on-device learning (USB meter between the laptop and the board).

Same method as unas/energy_session.py: the meter (JT-UM120) reads P-BUS at the board's USB input, three readings a few
seconds apart once the value is steady. Steps, each flashed over SWD: the core asleep (WFI) and the core spinning
(baselines, no network), the int8 feature network running back to back (the default benchmark firmware of
firmware/h7b3), the on-device update running back to back (make ONLINE=1 OLOOP=1, memories filled with pseudo-random
features), and the core asleep again (drift check). Each step's own timing is read over SWD.

Interactive (you type the readings at the prompts):
    .venv\\Scripts\\python unas\\online_energy.py [--seed 100]
One step per call (readings typed by whoever reads the meter; state in firmware/build/online/<tag>/energy_state.json):
    .venv\\Scripts\\python unas\\online_energy.py step <i>               # i = 0..4, flashes step i
    .venv\\Scripts\\python unas\\online_energy.py record <i> <W> <W> <W>  # the three P-BUS readings of step i
    .venv\\Scripts\\python unas\\online_energy.py report                 # per window, per label, per exiD location

Per step, appended to datasets/exid/results/hil/online_energy.jsonl; the feature-network and update records also carry
energy_mcu_uJ = (P - P_sleep) x time per run and energy_board_uJ = P x time per run. The report (online_energy.json)
adds the energy per window and per label at the board's replay times and the average power above the asleep board
that inference and learning add at each exiD location.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_bench as lb  # noqa: E402

OUT = lb.ROOT / "datasets" / "exid" / "results" / "hil" / "online_energy.jsonl"
METER = "JT-UM120 USB meter, P-BUS at the board's USB input (laptop -> meter -> board)"
STEPS = ("idle-sleep", "idle-busy", "feature-network", "update-loop", "idle-sleep-end")


def read_words(elf: Path, sn: str, symbol: str, n: int, timeout_s: float = 60.0) -> list[int]:
    nm = lb.run([lb.GCC_BIN / "arm-none-eabi-nm.exe", elf])
    addr = next(int(l.split()[0], 16) for l in nm.splitlines() if l.endswith(" " + symbol))
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = lb.run([lb.CLI, "-c", "port=SWD", f"sn={sn}", "mode=HOTPLUG", "-r32", hex(addr), str(4 * n)])
        words = []
        for line in out.splitlines():
            m = re.match(r"\s*0x[0-9A-Fa-f]{8}\s*:\s*((?:[0-9A-Fa-f]{8}\s*)+)$", line)
            if m:
                words += [int(w, 16) for w in m.group(1).split()]
        if len(words) >= n and words[0] == lb.MAGIC:
            return words[:n]
        time.sleep(1.0)
    raise TimeoutError(f"{symbol}: no result after {timeout_s:.0f} s")


def tag_of(seed: int) -> str:
    return f"highd_K_seed{seed}_features_int8_io"


def flash(step: str, tag: str, sn: str):
    """Build and flash one step; return its time per run in seconds (None for the baselines)."""
    gen, ocfg = f"GEN=../build/gen/{tag}", f"OCFG=../build/online/{tag}"
    if step in ("idle-sleep", "idle-busy", "idle-sleep-end"):
        mode = 2 if step == "idle-busy" else 1
        args, elf = [f"IDLE={mode}", gen], lb.FW / "build" / "h7b3" / f"idle{mode}" / "fw.elf"
    elif step == "feature-network":
        args, elf = [gen], lb.FW / "build" / "h7b3" / tag / "fw.elf"
    else:
        args, elf = ["ONLINE=1", "OLOOP=1", gen, ocfg], lb.FW / "build" / "h7b3onlineloop" / tag / "fw.elf"
    lb.run([lb.MAKE, *args], cwd=lb.FW / "h7b3")
    lb.run([lb.CLI, "-c", "port=SWD", f"sn={sn}", "mode=UR", "-w", elf, "-v", "-rst"])
    if step in ("idle-sleep", "idle-sleep-end"):
        return None                                  # a sleeping core does not answer SWD reads; the flash was verified
    if step == "idle-busy":
        words = lb.read_result(elf, sn)
        if words[6] != 0x302:
            raise RuntimeError(f"idle firmware 2 reported status {words[6]:#x}")
        return None
    if step == "feature-network":
        r = lb.read_result(elf, sn)
        return r[4] / r[1]
    g = read_words(elf, sn, "g_online", 4)
    return g[1] / 280e6


def record(step: str, tag: str, vals: list[float], t_s, base: dict) -> dict:
    p = st.mean(vals)
    rec = {"step": "idle-sleep" if step == "idle-sleep-end" else step, "tag": tag, "readings_W": vals,
           "p_bus_W": round(p, 6), "time_per_run_ms": None if t_s is None else round(t_s * 1e3, 5), "meter": METER}
    if step == "idle-sleep-end":
        rec["drift_W"] = round(p - base["idle-sleep"], 6)
    if t_s is not None:
        rec["p_sleep_W"] = round(base["idle-sleep"], 6)
        rec["p_busy_W"] = round(base["idle-busy"], 6)
        rec["energy_board_uJ"] = round(p * t_s * 1e6, 4)
        rec["energy_mcu_uJ"] = round((p - base["idle-sleep"]) * t_s * 1e6, 4)
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def parse_readings(raw: list[str]) -> list[float]:
    vals = [float(t.strip(",;").replace(",", ".")) for t in raw]
    if not vals or not all(0.01 < v < 5.0 for v in vals):
        raise ValueError("readings in watts between 0.01 and 5, for example 0.41210 0.41220 0.41215")
    return vals


def stepwise(cmd: str, i: int, rest: list[str], seed: int) -> None:
    tag = tag_of(seed)
    path = lb.FW / "build" / "online" / tag / "energy_state.json"
    state = ({"tag": tag, "base": {}, "done": [], "current": None} if cmd == "step" and i == 0
             else json.loads(path.read_text(encoding="utf-8")))
    step = STEPS[i]
    if cmd == "step":
        sn = lb.find_stlink("STM32H7B3I-DK")
        t_s = flash(step, tag, sn)
        state["current"] = {"step": i, "name": step, "t_s": t_s}
        path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        what = {"idle-sleep": "core asleep, no network", "idle-busy": "core spinning, no network",
                "feature-network": "int8 feature network back to back", "update-loop": "on-device update back to back",
                "idle-sleep-end": "core asleep again (drift check)"}[step]
        print(f"STEP {i}/{len(STEPS) - 1} ready: {step} ({what})" + ("" if t_s is None else f", {t_s * 1e3:.4f} ms per run"))
        return
    cur = state["current"]
    if not cur or cur["step"] != i:
        sys.exit(f"step {i} is not the flashed step ({cur and cur['step']})")
    vals = parse_readings(rest)
    rec = record(step, state["tag"], vals, cur["t_s"], state["base"])
    if step in ("idle-sleep", "idle-busy"):
        state["base"][step] = st.mean(vals)
    state["done"].append(i)
    state["current"] = None
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print("recorded:", json.dumps(rec))


def interactive(seed: int) -> None:
    tag = tag_of(seed)
    sn = lb.find_stlink("STM32H7B3I-DK")
    base = {}
    for step in STEPS:
        t_s = flash(step, tag, sn)
        print(f"\n{step}: flashed{'' if t_s is None else f', {t_s * 1e3:.4f} ms per run'}; wait until the meter is steady.")
        while True:
            try:
                vals = parse_readings(input("  three P-BUS readings in W: ").replace(";", " ").split())
                break
            except ValueError as e:
                print(" ", e)
        rec = record(step, tag, vals, t_s, base)
        if step in ("idle-sleep", "idle-busy"):
            base[step] = st.mean(vals)
        print("  recorded:", json.dumps(rec))


def report(seed: int) -> None:
    """Energy per window and per label on the board, and the average power above the asleep board that inference and
    learning add at each exiD location (the board's replay counts and device hours, online_board.jsonl); the last
    session of the tag. Written to datasets/exid/results/hil/online_energy.json."""
    tag = tag_of(seed)
    rows = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    last = {}
    for r in rows:
        if r.get("tag") == tag:
            last["idle-sleep-end" if "drift_W" in r else r["step"]] = r
    p0, pf, pu = last["idle-sleep"]["p_bus_W"], last["feature-network"]["p_bus_W"], last["update-loop"]["p_bus_W"]
    board = [json.loads(line) for line in (OUT.parent / "online_board.jsonl").read_text().splitlines() if line.strip()]
    rep = {r["location"]: r for r in board if r["kind"] == "replay" and r["tag"] == tag}
    curve = [r for r in board if r["kind"] == "adaptation_curve" and r["tag"] == tag][-1]
    per_loc, tot = {}, {"windows": 0, "labels": 0, "seconds": 0.0, "inference_J": 0.0, "learning_J": 0.0}
    for loc, r in sorted(rep.items()):
        e_w = (pf - p0) * r["board_ms_per_window"] * 1e-3                 # J per window above the asleep board
        e_l = (pu - p0) * r["board_ms_per_label"] * 1e-3                  # J per label above the asleep board
        sec = curve["per_location"][str(loc)]["device_hours"] * 3600
        inf_j, lrn_j = r["windows"] * e_w, r["labels"] * e_l
        per_loc[str(loc)] = {"windows": r["windows"], "labels": r["labels"], "device_hours": round(sec / 3600, 3),
                             "energy_per_window_uJ": round(e_w * 1e6, 3), "energy_per_label_uJ": round(e_l * 1e6, 3),
                             "inference_mW": round(inf_j / sec * 1e3, 3), "learning_mW": round(lrn_j / sec * 1e3, 3)}
        for k, v in (("windows", r["windows"]), ("labels", r["labels"]), ("seconds", sec), ("inference_J", inf_j),
                     ("learning_J", lrn_j)):
            tot[k] += v
    out = {"tag": tag, "meter": METER, "p_sleep_W": p0, "p_busy_W": last["idle-busy"]["p_bus_W"], "p_feature_network_W": pf,
           "p_update_loop_W": pu, "drift_W": last.get("idle-sleep-end", {}).get("drift_W"),
           "feature_network": {k: last["feature-network"][k] for k in ("time_per_run_ms", "energy_mcu_uJ", "energy_board_uJ")},
           "update": {k: last["update-loop"][k] for k in ("time_per_run_ms", "energy_mcu_uJ", "energy_board_uJ")},
           "per_location": per_loc,
           "all_locations": {"device_hours": round(tot["seconds"] / 3600, 3),
                             "inference_mW": round(tot["inference_J"] / tot["seconds"] * 1e3, 3),
                             "learning_mW": round(tot["learning_J"] / tot["seconds"] * 1e3, 3),
                             "learning_share_of_the_asleep_board": round(tot["learning_J"] / tot["seconds"] / p0, 5)},
           "note": "above the asleep board (core in WFI, 0.93 W for the whole kit with its display); window and label "
                   "times are the board's own from the replays, at the powers of the back-to-back steps",
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT.parent / "online_energy.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("step", "record", "report"))
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "report":
        report(a.seed)
    elif a.cmd:
        stepwise(a.cmd, int(a.rest[0]), a.rest[1:], a.seed)
    else:
        interactive(a.seed)


if __name__ == "__main__":
    main()
