"""Hardware in the loop, step 1 (2026-11-11): where the board and the TFLite reference differ, ask ST's C
model on the host.

For a network replayed with `hil_stream.py --rate 0 --save`, the windows whose board outputs differ from
the reference of unas/hil_prepare.py (TFLite interpreter) are run through the C model that ST Edge AI
generates, compiled for the PC (`stedgeai validate --mode host`), with the reference's int8 inputs. The
same number of windows on which board and reference agree goes along as a control (at least 32). If the
C model on the host returns the board's values, a difference lies between ST's kernels and the TFLite
interpreter, not in the board, its preprocessing or the link.

  .venv\\Scripts\\python unas/hil_host_check.py datasets/<ds>/results/deploy/<stem>_int8_io.tflite [...]

One JSON line per network goes to datasets/<ds>/results/hil/hil_host_check.jsonl (counts only).
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_bench as lb                                    # noqa: E402  (tool paths, run)

ROOT = lb.ROOT


def check(tfl: Path) -> dict:
    ds = tfl.relative_to(ROOT).parts[1]
    folder = lb.FW / "build" / "stream" / tfl.stem
    ref = np.load(folder / "reference.npz")
    board = np.load(folder / "board_out_rate0.npz")["outs"]
    want = ref["ref"][:len(board)]
    in_len = int(ref["in_len"]) if "in_len" in ref else 10
    differ = np.argwhere((board != want).any(-1))
    same = np.argwhere((board == want).all(-1))
    rng = np.random.default_rng(0)
    control = same[rng.choice(len(same), size=min(len(same), max(32, len(differ))), replace=False)]
    rows = np.concatenate([differ, control]) if len(differ) else control
    x = np.stack([ref["q"][s, k:k + in_len] for s, k in rows]).astype(np.int8)

    work = lb.FW / "build" / "hostcheck" / tfl.stem
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    np.savez(work / "in.npz", m_inputs_1=x)
    lb.run([lb.STEDGEAI, "validate", "--model", tfl, "--target", "stm32", "--mode", "host", "--name", "network",
            "--valinput", work / "in.npz", "--batches", str(len(x)),
            "--output", work / "out", "--workspace", work / "ws"])
    z = np.load(work / "out" / "network_val_io.npz")
    c = z["c_outputs_1"].reshape(len(z["c_outputs_1"]), -1)
    if len(c) != len(x) or not (z["m_inputs_1"].reshape(x.shape) == x).all():
        raise RuntimeError(f"{tfl.name}: the host run used {len(c)} of {len(x)} windows")
    b = np.stack([board[s, k] for s, k in rows])
    t = np.stack([want[s, k] for s, k in rows])
    nd = len(differ)
    rec = {"model": tfl.name, "windows_replayed": int(board.shape[0] * board.shape[1]),
           "differing_windows": nd,
           "differing": {"host_c_equals_board": int((c[:nd] == b[:nd]).all(-1).sum()),
                         "host_c_equals_tflite": int((c[:nd] == t[:nd]).all(-1).sum()),
                         "max_abs_lsb_board_vs_tflite": int(np.abs(b[:nd].astype(int) - t[:nd].astype(int)).max()) if nd else 0},
           "control": {"windows": int(len(rows) - nd),
                       "host_c_equals_board": int((c[nd:] == b[nd:]).all(-1).sum())},
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out = ROOT / "datasets" / ds / "results" / "hil" / "hil_host_check.jsonl"
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    return rec


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        check((ROOT / arg).resolve())
