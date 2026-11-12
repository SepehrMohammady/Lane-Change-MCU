"""Does a saturating input calibration remove the int8 loss on LCIR? (2026-11-11)

The LCIR features arrive standardized (1st to 99th percentile within about -5 to +5) with rare extremes up
to 431 in absolute value. TFLite sets the range of the int8 input from the minimum and maximum of the
calibration windows, so one int8 step spans 1 to 2 standard deviations (the input quantization loss found
on 2026-11-05). This script trains the three corrected LCIR choices on raw inputs (search recipe, as
input_scaling_check.py), then converts every trained network to full int8 with int8 I/O once as before
and once per bound B with the calibration windows clipped to [-B, B]. Nothing is retrained and the
networks stay as searched. Each int8 file is scored on the unclipped validation and test windows through
its own input quantization, which saturates as the board's quantizer does (firmware/h7b3, STREAM_MODE);
the float network is also scored on inputs clipped to [-B, B]. The rule was fixed beforehand (local
notes, 2026-11-11): per head, B is chosen on the mean int8 validation result and the test result of that
B is compared with float32.

Run in the WSL dmir_nas venv, from the fork (as select_by_seeds.py):
  DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data SEEDS=100,101,102,103,104 \
    ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/input_clip_check.py [task ...]
  .venv\\Scripts\\python unas/input_clip_check.py --report        # selection and rule from the result file

Output: datasets/dmir/results/seeds/input_clip.jsonl (one line per task, seed and bound; bound null = the
calibration as before; resumable). The trained networks are kept in datasets/dmir/data/clip_check/ (not
versioned). SEEDS and EPOCH_CAP as in seed_variance.py (smoke test: EPOCH_CAP=1 SEEDS=0
OUT_NAME=input_clip_smoke.jsonl).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BOUNDS = (3, 4, 6, 8, 12, 16, 24, 32)
TOL = {"classification": 0.01, "regression_lcr": 0.03, "regression_lcl": 0.03}   # 1.0 point, 0.03 s
OUT = REPO / "datasets" / "dmir" / "results" / "seeds" / os.environ.get("OUT_NAME", "input_clip.jsonl")
KEEP = REPO / "datasets" / "dmir" / "data" / "clip_check"


def predict_int8(tf, tfl: bytes, x: np.ndarray, batch: int = 1024, as_int8: bool = False):
    """Outputs of the int8 file for float windows x, quantized by the file's own input scale with
    saturation; batched when the interpreter gives the per-window values for a resized batch."""
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    (si, zi), (so, zo) = inp["quantization"], out["quantization"]
    xq = np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)
    n_out = int(np.prod(out["shape"][1:]))
    one = np.empty((len(x), n_out), np.int8)

    def loop(rows):
        for i in rows:
            it.set_tensor(inp["index"], xq[i:i + 1])
            it.invoke()
            one[i] = it.get_tensor(out["index"]).reshape(-1)

    head = range(min(64, len(x)))
    loop(head)
    raw = one.copy()
    try:
        bt = tf.lite.Interpreter(model_content=tfl)
        bt.resize_tensor_input(inp["index"], [batch, *x.shape[1:]])
        bt.allocate_tensors()
        for a in range(0, len(x), batch):
            chunk = xq[a:a + batch]
            pad = np.zeros((batch, *x.shape[1:]), np.int8)
            pad[:len(chunk)] = chunk
            bt.set_tensor(inp["index"], pad)
            bt.invoke()
            raw[a:a + len(chunk)] = bt.get_tensor(bt.get_output_details()[0]["index"]).reshape(batch, -1)[:len(chunk)]
        if not (raw[:len(head)] == one[:len(head)]).all():
            raise ValueError("batched outputs differ from single-window outputs")
    except Exception as e:                                   # noqa: BLE001  (fall back to one window at a time)
        print(f"  batched int8 inference not used ({type(e).__name__}: {e})", flush=True)
        loop(range(len(head), len(x)))
        raw = one
    if as_int8:                                              # the int8 outputs and their quantization
        return raw, (float(so), int(zo))
    return (raw.astype(np.float32) - zo) * so, (float(si), int(zi)), float((np.abs(xq.astype(np.int16)) >= 127).mean())


def run(tasks):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import input_scaling_check as isc                        # sets up TF, keras, the fork's adapters

    tf, keras, sv = isc.tf, isc.keras, isc.sv
    os.environ["DMIR_SCALE"] = ""
    KEEP.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["seed"], r["bound"]))
    for task in tasks:
        search, model = isc.CHOICES[task]
        cls = task == "classification"
        todo = [s for s in sv.SEEDS if any((task, s, b) not in done for b in (None, *BOUNDS))]
        if not todo:
            continue
        ds = sv.DMIR_Dataset(task=task)
        xtr = ds._data["train"][0]
        rng = np.random.default_rng(42)                      # the 500 calibration windows of convert_int8_io
        calib = xtr[rng.choice(len(xtr), size=min(500, len(xtr)), replace=False)]
        for seed in todo:
            t0 = time.perf_counter()
            kept = KEEP / f"{task}_{model}_seed{seed}{'_smoke' if sv.EPOCH_CAP else ''}.h5"
            if kept.exists():
                net, epochs = keras.models.load_model(str(kept), compile=False), None
            else:
                net, epochs = isc.train(str(isc.NV / search / f"{model}.h5"), ds, cls, seed)
                net.save(str(kept))
            for bound in (None, *BOUNDS):
                if (task, seed, bound) in done:
                    continue
                clip = (lambda v: v) if bound is None else (lambda v, b=bound: np.clip(v, -b, b))
                conv = tf.lite.TFLiteConverter.from_keras_model(net)
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                s = clip(calib).astype(np.float32)
                conv.representative_dataset = lambda s=s: ([s[i:i + 1]] for i in range(len(s)))
                conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                conv.inference_input_type = tf.int8
                conv.inference_output_type = tf.int8
                tfl = conv.convert()
                rec = {"task": task, "seed": seed, "bound": bound, "search": search, "model": model,
                       "params": int(net.count_params()), "recipe": "search recipe, raw inputs",
                       "epochs_run": epochs, "int8_io_bytes": len(tfl)}
                for split in ("val", "test"):
                    x, y = ds._data[split]
                    rec[f"float_{split}"] = isc.metric(net.predict(clip(x), batch_size=4096, verbose=0), y, cls)
                    p, quant, sat = predict_int8(tf, tfl, x)
                    rec[f"int8_{split}"] = isc.metric(p, y, cls)
                    rec[f"saturated_share_{split}"] = round(sat, 6)
                rec["in_quant"] = [quant[0], quant[1]]
                rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{task}] seed {seed} bound {bound}: float test {rec['float_test']} "
                      f"int8 test {rec['int8_test']}", flush=True)
            print(f"[{task}] seed {seed} done in {time.perf_counter() - t0:.0f} s", flush=True)
            keras.backend.clear_session()
        del ds


def report():
    """Selection on validation and the rule on test, from the result file."""
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    out = {}
    for task in sorted({r["task"] for r in rows}):
        key = "acc" if task == "classification" else "rmse"
        sign = 1.0 if key == "acc" else -1.0                 # higher is better for accuracy
        by = {}
        for r in rows:
            if r["task"] == task:
                by.setdefault(r["bound"], {})[r["seed"]] = r
        seeds = sorted(by[None])
        col = lambda b, f: np.array([by[b][s][f][key] for s in seeds])   # noqa: E731
        complete = [b for b in by if sorted(by[b]) == seeds]
        val = {b: float(col(b, "int8_val").mean()) for b in complete if b is not None}
        best = max(val.values()) if sign > 0 else min(val.values())
        sel = max(b for b, v in val.items() if v == best)
        f32 = col(None, "float_test")
        q = col(sel, "int8_test")
        gap = float(sign * (f32.mean() - q.mean()))          # what int8 loses against float32, test
        out[task] = {
            "metric": key, "seeds": seeds, "selected_bound": sel,
            "float32_test": [round(float(f32.mean()), 4), round(float(f32.std(ddof=1)), 4)],
            "int8_test_selected": [round(float(q.mean()), 4), round(float(q.std(ddof=1)), 4)],
            "int8_test_as_before": [round(float(col(None, "int8_test").mean()), 4),
                                    round(float(col(None, "int8_test").std(ddof=1)), 4)],
            "float_on_clipped_test": round(float(col(sel, "float_test").mean()), 4),
            "int8_loss_test": round(gap, 4), "tolerance": TOL[task], "rule_met": bool(gap <= TOL[task]),
            "paired_loss_per_seed": [round(float(v), 4) for v in sign * (f32 - q)],
            "saturated_share_test": round(float(np.mean([by[sel][s]["saturated_share_test"] for s in seeds])), 5),
            "by_bound": {str(b): {"int8_val": round(float(col(b, "int8_val").mean()), 4),
                                  "int8_test": round(float(col(b, "int8_test").mean()), 4),
                                  "float_val": round(float(col(b, "float_val").mean()), 4),
                                  "float_test": round(float(col(b, "float_test").mean()), 4)}
                         for b in sorted(complete, key=lambda v: (v is not None, v or 0))},
        }
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        run(sys.argv[1:] or ["classification", "regression_lcr", "regression_lcl"])
