"""Power session for on-device learning (run it yourself, with the USB meter between the laptop and the board).

Same method as unas/energy_session.py: the meter (JT-UM120) reads P-BUS at the board's USB input. Three steps, each
flashed over SWD: the core asleep (WFI, the baseline), the int8 feature network running back to back (the default
benchmark firmware of firmware/h7b3), and the on-device update running back to back (make ONLINE=1 OLOOP=1, memories
filled with pseudo-random features). The script reads each step's own timing over SWD and asks for the meter reading
once the value is steady (about 10 s after the prompt).

    .venv\\Scripts\\python unas\\online_energy.py [--seed 100]

Per step, appended to datasets/exid/results/hil/online_energy.jsonl; the update record also carries
energy_mcu_uJ = (P_update - P_sleep) x time per update and energy_board_uJ = P_update x time per update, and the same for
one inference of the feature network.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_bench as lb  # noqa: E402

OUT = lb.ROOT / "datasets" / "exid" / "results" / "hil" / "online_energy.jsonl"
METER = "JT-UM120 USB meter, P-BUS at the board's USB input (laptop -> meter -> board)"


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


def ask(prompt: str) -> float:
    while True:
        s = input(f"{prompt} P-BUS in W (e.g. 0.412): ").strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            print("  a number, please")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    tag = f"highd_K_seed{a.seed}_features_int8_io"
    gen, ocfg = f"GEN=../build/gen/{tag}", f"OCFG=../build/online/{tag}"
    sn = lb.find_stlink("STM32H7B3I-DK")
    steps = [("idle-sleep", ["IDLE=1", gen], lb.FW / "build" / "h7b3" / "idle1" / "fw.elf"),
             ("feature-network", [gen], lb.FW / "build" / "h7b3" / tag / "fw.elf"),
             ("update-loop", ["ONLINE=1", "OLOOP=1", gen, ocfg], lb.FW / "build" / "h7b3onlineloop" / tag / "fw.elf")]
    res = {}
    for name, make_args, elf in steps:
        lb.run([lb.MAKE, *make_args], cwd=lb.FW / "h7b3")
        lb.run([lb.CLI, "-c", "port=SWD", f"sn={sn}", "mode=UR", "-w", elf, "-v", "-rst"])
        if name == "feature-network":
            r = lb.read_result(elf, sn)
            t_s = r[4] / r[1]
        elif name == "update-loop":
            g = read_words(elf, sn, "g_online", 4)
            t_s = g[1] / 280e6
        else:
            time.sleep(3)
            t_s = None
        print(f"\n{name}: flashed{'' if t_s is None else f', {t_s * 1e3:.4f} ms per run'}; wait until the meter is steady.")
        p = ask(f"  {name}:")
        res[name] = (p, t_s)
        rec = {"step": name, "tag": tag, "p_bus_W": p, "time_per_run_ms": None if t_s is None else round(t_s * 1e3, 5),
               "meter": METER, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if t_s is not None and "idle-sleep" in res:
            p0 = res["idle-sleep"][0]
            rec["energy_board_uJ"] = round(p * t_s * 1e6, 3)
            rec["energy_mcu_uJ"] = round((p - p0) * t_s * 1e6, 3)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("  recorded:", json.dumps(rec))


if __name__ == "__main__":
    main()
