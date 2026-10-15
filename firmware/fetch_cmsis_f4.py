"""Fetch ST's CMSIS device files for the STM32F401 into firmware/third_party (not versioned).

    python firmware/fetch_cmsis_f4.py
"""
from pathlib import Path
from urllib.request import urlopen

BASE = "https://raw.githubusercontent.com/STMicroelectronics/cmsis-device-f4/master/"
FILES = ["Include/stm32f4xx.h", "Include/stm32f401xe.h", "Include/system_stm32f4xx.h",
         "Source/Templates/system_stm32f4xx.c", "Source/Templates/gcc/startup_stm32f401xe.s", "LICENSE.md"]
DEST = Path(__file__).resolve().parent / "third_party" / "cmsis-device-f4"

for f in FILES:
    out = DEST / f
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(urlopen(BASE + f, timeout=60).read())
    print("ok", f)
