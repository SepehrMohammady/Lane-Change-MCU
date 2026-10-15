"""Measure TFLite files on our own boards (local ST Edge AI Core, bare-metal firmware).

For each file: `stedgeai generate` (legacy C API, default options as on the Developer Cloud:
balanced, inputs and outputs in the activation buffer) into firmware/build/gen/<stem>, build
firmware/<board> against it, flash it over SWD, and read the result block g_result from RAM over
SWD (works while another program holds the virtual COM port). One JSON line per file is
appended to datasets/<dataset>/results/deploy/local_bench.jsonl.

    python unas/local_bench.py [--board f401|h7b3] datasets/highd/results/deploy/a.tflite [b.tflite ...]

The ST-LINK is chosen by the board name it reports, so the right board is flashed when both
are connected. Timing: DWT cycle counter around ai_network_run, 5 warm-up runs and 200 timed
runs with fresh pseudo-random inputs, no interrupts. F401RE: 84 MHz, flash prefetch and caches
on. H7B3I-DK: ST's template (direct SMPS, VOS0, 280 MHz, I- and D-cache on, data in DTCM).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FW = ROOT / "firmware"
STEDGEAI = Path(r"C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe")
CLI = Path(r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe")
IDE_PLUGINS = Path(r"C:\ST\STM32CubeIDE_2.1.1\STM32CubeIDE\plugins")
MAKE = next(IDE_PLUGINS.glob("com.st.stm32cube.ide.mcu.externaltools.make.*")) / "tools" / "bin" / "make.exe"
GCC_BIN = next(IDE_PLUGINS.glob("com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.*")) / "tools" / "bin"
MAGIC = 0x42454E43
BOARDS = {  # option -> (board name reported by the ST-LINK, firmware folder, runtime library)
    "f401": ("NUCLEO-F401RE", "f401", "NetworkRuntime1200_CM4_GCC"),
    "h7b3": ("STM32H7B3I-DK", "h7b3", "NetworkRuntime1200_CM7_GCC"),
}


def run(cmd, cwd=None, timeout=600) -> str:
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def find_stlink(board_name: str) -> str:
    """Serial number of the ST-LINK whose board reports board_name."""
    out = run([CLI, "-l", "st-link"])
    sns = re.findall(r"ST-LINK SN\s*:\s*(\S+)", out)
    names = re.findall(r"Board Name\s*:\s*(\S+)", out)
    for sn, name in zip(sns, names):
        if name == board_name:
            return sn
    raise RuntimeError(f"no ST-LINK reports board {board_name}; connected: {list(zip(sns, names))}")


def read_result(elf: Path, sn: str, timeout_s: float = 120.0) -> list[int]:
    nm = run([GCC_BIN / "arm-none-eabi-nm.exe", elf])
    addr = next(int(l.split()[0], 16) for l in nm.splitlines() if l.endswith(" g_result"))
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        out = run([CLI, "-c", "port=SWD", f"sn={sn}", "mode=HOTPLUG", "-r32", hex(addr), "32"])
        words = []
        for line in out.splitlines():
            m = re.match(r"\s*0x[0-9A-Fa-f]{8}\s*:\s*((?:[0-9A-Fa-f]{8}\s*)+)$", line)
            if m:
                words += [int(w, 16) for w in m.group(1).split()]
        if len(words) >= 8 and words[0] == MAGIC:
            return words[:8]
        time.sleep(1.0)
    raise TimeoutError(f"no result block at {hex(addr)} after {timeout_s:.0f} s")


def elf_size(elf: Path) -> dict:
    out = run([GCC_BIN / "arm-none-eabi-size.exe", elf]).splitlines()[1].split()
    text, data, bss = (int(x) for x in out[:3])
    return {"flash_B": text + data, "ram_static_B": data + bss}


def measure(tfl: Path, board: str, sn: str) -> dict:
    name, folder, runtime = BOARDS[board]
    tag = tfl.stem
    gen = FW / "build" / "gen" / tag
    gen.mkdir(parents=True, exist_ok=True)
    run([STEDGEAI, "generate", "--model", tfl, "--target", "stm32", "--name", "network", "--c-api", "legacy",
         "--output", gen, "--workspace", FW / "build" / "ws" / tag])
    info = json.loads((gen / "network_c_info.json").read_text(encoding="utf-8"))
    run([MAKE, f"GEN=../build/gen/{tag}"], cwd=FW / folder)
    elf = FW / "build" / folder / tag / "fw.elf"
    run([CLI, "-c", "port=SWD", f"sn={sn}", "mode=UR", "-w", elf, "-v", "-rst"])
    magic, hz, runs, cmin, cmean, cmax, status, loops = read_result(elf, sn)
    if status != 0:
        raise RuntimeError(f"{tag}: firmware reported status {status:#x}")
    nodes = info["graphs"][0]["nodes"]
    mem = info.get("memory_footprint", {})
    tools = info.get("environment", {}).get("tools")
    return {"model": tfl.name, "board": name, "board_sn": sn, "source": "local",
            "duration_ms": round(cmean / hz * 1e3, 5), "cycles_mean": cmean, "cycles_min": cmin,
            "cycles_max": cmax, "runs": runs, "core_hz": hz,
            "macc": sum(n.get("macc", 0) for n in nodes), "weights_B": mem.get("weights"),
            "activations_B": mem.get("activations"),
            "stedgeai": tools[0].get("version") if isinstance(tools, list) and tools else None,
            "compiler": f"arm-none-eabi-gcc 14.3.1 -O2, nano specs, {runtime}",
            **elf_size(elf), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main():
    args = sys.argv[1:]
    board = "f401"
    if "--board" in args:
        i = args.index("--board")
        board = args[i + 1]
        del args[i:i + 2]
    files = [Path(a).resolve() for a in args]
    if not files or board not in BOARDS:
        sys.exit(__doc__)
    sn = find_stlink(BOARDS[board][0])
    for f in files:
        ds = f.relative_to(ROOT / "datasets").parts[0]
        out = ROOT / "datasets" / ds / "results" / "deploy" / "local_bench.jsonl"
        try:
            rec = measure(f, board, sn)
        except Exception as e:  # keep going; report at the end of the line
            print(f"{f.name}: FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
            continue
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{rec['model']} on {rec['board']}: {rec['duration_ms']:.4f} ms ({rec['cycles_mean']} cycles, "
              f"min {rec['cycles_min']}, max {rec['cycles_max']}) -> {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
