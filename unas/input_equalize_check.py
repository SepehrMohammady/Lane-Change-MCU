"""Does equalizing the first convolution's weights remove the int8 loss on exiD? (2026-11-11)

The hand-designed CNN trained on exiD loses 3 points in int8, and the loss sits in the weights of its
first convolution alone: after folding the batch normalization, the largest weight on one input feature
is 135 times the largest weight on another, so in every filter the small weights get 1 to 4 of the 127
int8 steps (per-channel quantization scales a whole filter by its largest weight).

Repair without retraining, exact in float: per input feature f a gain g_f = (W_f / max W)^alpha <= 1,
with W_f the largest folded |weight| on that feature. The network is fed x' = g_f x and the first
convolution's weights on feature f are divided by g_f: the float outputs do not change, the weights of
a filter become more alike, and a feature with small weights uses a smaller part of the int8 input
range. alpha = 0 is the network as deployed. On a board the gain is a different span in the scaling
table (unas/hil_prepare.py): no extra arithmetic.

Input: the per-seed bundles of scripts/export_hand_cnn.py exid5 (weights, 500 calibration windows,
validation and test windows). Every variant is converted to full int8 with int8 I/O and scored on
validation and test. The rule was fixed beforehand (local notes, 2026-11-11): alpha is chosen on the
mean int8 validation accuracy, and the test accuracy at that alpha is compared with float32.

Run in the WSL dmir_nas venv:
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/input_equalize_check.py
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/input_equalize_check.py --report     (any Python)
Output: datasets/exid/results/seeds/input_equalize.jsonl (one line per seed and alpha; resumable).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2, 3, 4)
TOL = 0.01                                                  # 1.0 point
OUT = REPO / "datasets" / "exid" / "results" / "seeds" / "input_equalize.jsonl"


def gains(model, alpha: float) -> np.ndarray:
    """g_f per input feature from the BN-folded weights of the first convolution."""
    w = model.get_layer("stem_conv").get_weights()[0]       # (k, in, out)
    g, _, _, var = model.get_layer("stem_bn").get_weights()
    folded = w * (g / np.sqrt(var + 1e-5))
    per_in = np.abs(folded).max(axis=(0, 2))
    return (per_in / per_in.max()) ** alpha, per_in


def run():
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import deploy_hand_cnn as d
    from input_clip_check import predict_int8

    tf = d.tf
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["seed"], r["alpha"]))
    for seed in SEEDS:
        if all((seed, a) in done for a in ALPHAS):
            continue
        z = np.load(REPO / "datasets" / "exid" / "data" / "hand_export" / f"exid_dscnn_cls_seed{seed}.npz")
        xs = {"val": z["x_val"].astype(np.float32), "test": z["x_test"].astype(np.float32)}
        ys = {"val": z["y_val"].astype(np.int64), "test": z["y_test"].astype(np.int64)}
        calib = z["x_calib"].astype(np.float32)
        base = d.build(z)
        ref = {s: base.predict(xs[s], batch_size=8192, verbose=0) for s in xs}
        for alpha in ALPHAS:
            if (seed, alpha) in done:
                continue
            m = d.build(z)
            g, per_in = gains(m, alpha)
            layer = m.get_layer("stem_conv")
            layer.set_weights([layer.get_weights()[0] / g[None, :, None]])
            rec = {"task": "exid_cls", "network": "hand-designed CNN", "seed": seed, "alpha": alpha,
                   "source": str(z["source"]).replace("\\", "/"), "params": int(z["params"]),
                   "weight_spread_before": round(float(per_in.max() / per_in.min()), 1),
                   "smallest_gain": round(float(g.min()), 5)}
            tfl = d.convert(m, calib * g, "int8", int8_io=True)
            for s in xs:
                p = m.predict(xs[s] * g, batch_size=8192, verbose=0)
                rec[f"float_change_{s}"] = float(np.abs(p - ref[s]).max())
                rec[f"float_{s}"] = {"acc": float((ref[s].argmax(1) == ys[s]).mean())}
                q, quant, _ = predict_int8(tf, tfl, xs[s] * g)
                rec[f"int8_{s}"] = {"acc": float((q.argmax(1) == ys[s]).mean())}
            rec["in_quant"] = [quant[0], quant[1]]
            rec["int8_io_bytes"] = len(tfl)
            rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"seed {seed} alpha {alpha}: float test {rec['float_test']['acc']:.4f} int8 test "
                  f"{rec['int8_test']['acc']:.4f} (float change {rec['float_change_test']:.1e})", flush=True)


def report():
    rows = [json.loads(line) for line in OUT.read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["alpha"], {})[r["seed"]] = r
    seeds = sorted(by[0.0])
    col = lambda a, f, ss=seeds: np.array([by[a][s][f]["acc"] for s in ss])       # noqa: E731
    val = {a: float(col(a, "int8_val").mean()) for a in by if sorted(by[a]) == seeds}
    sel = min(a for a, v in val.items() if v == max(val.values()))
    f32, q = col(0.0, "float_test"), col(sel, "int8_test")
    fresh = [s for s in seeds if s != 0]                    # seed 0 was seen in the exploration
    out = {"seeds": seeds, "selected_alpha": sel,
           "float32_test": [round(float(f32.mean()), 4), round(float(f32.std(ddof=1)), 4)],
           "int8_test_selected": [round(float(q.mean()), 4), round(float(q.std(ddof=1)), 4)],
           "int8_test_as_deployed": [round(float(col(0.0, "int8_test").mean()), 4),
                                     round(float(col(0.0, "int8_test").std(ddof=1)), 4)],
           "int8_loss_test": round(float(f32.mean() - q.mean()), 4), "tolerance": TOL,
           "rule_met": bool(f32.mean() - q.mean() <= TOL),
           "seeds_1_to_4": {"float32_test": round(float(col(0.0, "float_test", fresh).mean()), 4),
                            "int8_test_selected": round(float(col(sel, "int8_test", fresh).mean()), 4),
                            "int8_test_as_deployed": round(float(col(0.0, "int8_test", fresh).mean()), 4)},
           "paired_loss_per_seed": [round(float(v), 4) for v in f32 - q],
           "largest_float_change": max(max(r["float_change_val"], r["float_change_test"]) for r in rows),
           "weight_spread_before": [by[0.0][s]["weight_spread_before"] for s in seeds],
           "by_alpha": {str(a): {"int8_val": round(val[a], 4), "int8_test": round(float(col(a, "int8_test").mean()), 4)}
                        for a in sorted(val)}}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
