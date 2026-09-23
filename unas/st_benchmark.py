"""Measure TFLite files on the ST Edge AI Developer Cloud board farm.

Uses ST's official Python client (stm32ai-modelzoo-services/common/stm32ai_dc,
kept under Materials/). The MyST credentials are read from the environment
variables STM32AI_USERNAME and STM32AI_PASSWORD and are never written to disk.
Each file is uploaded once and benchmarked on each board with the default
command-line options of ST Edge AI Core (balanced optimization, inputs and
outputs allocated in the activation buffer); one JSON line per file and board
is appended to <out.jsonl> in the format of the existing benchmarks_api.jsonl.

    STM32AI_USERNAME=... STM32AI_PASSWORD=... python unas/st_benchmark.py \
        datasets/highd/results/deploy/benchmarks_api.jsonl a.tflite b.tflite

Options: --boards STM32H7B3I-DK,NUCLEO-F401RE   --version 4.0.1
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Materials" / "stm32ai-modelzoo-services"))

from common.stm32ai_dc import CliParameters, CloudBackend, Stm32Ai  # noqa: E402


def option(name: str, default: str) -> str:
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def record(path: Path, board: str, res, wall_s: float) -> dict:
    return {"model": path.name, "board": board, "duration_ms": res.duration_ms, "cycles": res.cycles,
            "cycles_by_macc": res.cycles_by_macc, "macc": res.macc, "weights_B": res.weights,
            "activations_B": res.activations_size, "rom_B": res.rom_size, "ram_B": res.ram_size,
            "ram_io_B": res.total_ram_io_size, "lib_flash_est_B": res.estimated_library_flash_size,
            "cli_version": res.cli_version_str, "device": res.device, "wall_s": round(wall_s, 1),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main() -> None:
    skip = {"--boards", "--version"}
    args, i = [], 1
    while i < len(sys.argv):
        if sys.argv[i] in skip:
            i += 2
            continue
        args.append(sys.argv[i])
        i += 1
    out, files = Path(args[0]), [Path(a) for a in args[1:]]
    boards = option("--boards", "STM32H7B3I-DK,NUCLEO-F401RE").split(",")
    version = option("--version", "4.0.1")
    user, pwd = os.environ.get("STM32AI_USERNAME"), os.environ.get("STM32AI_PASSWORD")
    if not user or not pwd:
        sys.exit("set STM32AI_USERNAME and STM32AI_PASSWORD in the environment")
    ai = Stm32Ai(CloudBackend(user, pwd, version))
    del pwd
    names = {b.name for b in ai.get_benchmark_boards()}
    missing = [b for b in boards if b not in names]
    if missing:
        sys.exit(f"boards not offered by the farm: {missing}")
    for f in files:
        ai.upload_model(str(f))
        for board in boards:
            t0 = time.time()
            try:
                res = ai.benchmark(CliParameters(model=f.name), board)
                rec = record(f, board, res, time.time() - t0)
            except Exception as e:                      # e.g. the build does not fit the board
                rec = {"model": f.name, "board": board, "error": f"{type(e).__name__}: {e}",
                       "wall_s": round(time.time() - t0, 1),
                       "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            shown = {k: rec.get(k) for k in ("model", "board", "duration_ms", "rom_B", "ram_B", "cli_version", "error")}
            print(json.dumps(shown), flush=True)


if __name__ == "__main__":
    main()
