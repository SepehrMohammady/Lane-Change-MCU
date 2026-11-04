"""Would board-measured latency choose differently from MACs? Headroom check for Part A (2026-11-04).

For each corrected search, the 12 shortlisted networks of unas/select_by_seeds.py (five-seed validation
accuracy known) are converted to full-int8 TFLite with int8 I/O and timed on the STM32H7B3I-DK
(unas/local_bench.py). No training. Rules fixed before any latency was measured (paper/NOTES.md, 2026-11-04):
  tied set  networks whose five-seed validation accuracy is not distinguishable from the best one's
            (Welch p > 0.05; the selector's one-standard-error rule leaves one network in two searches)
  headroom  latency of the network MACs would pick in the tied set (fewest MACs, the search's own cost
            count) over the latency of the fastest network in the tied set
  decision  build the latency-aware search if the headroom is at least 1.3 in at least two of the three
            searches
Also reported: the network the parameter count would pick, the accuracy-first choice, and Spearman
(MACs, latency) over the twelve.

  convert (WSL dmir_nas venv, CPU):
    CUDA_VISIBLE_DEVICES= HIGHD_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/highd/data/prepared \
    DMIR_DATA_ROOT=/mnt/c/Projects/PhD/DIMIR/datasets/dmir/data \
      ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/latency_headroom.py convert
  measure (Windows, board connected; writes datasets/<ds>/results/headroom/latency_h7b3.jsonl):
    python unas/latency_headroom.py measure
  analyse (Windows):
    python unas/latency_headroom.py analyse
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEARCHES = [("highd", "highd_cls_v4_k3_single_s42"), ("highd", "highd_cls_v4_k3_single_s43"),
            ("dmir", "dmir_cls_v4_k3_single_s42")]
ALPHA, GO_RATIO, GO_COUNT = 0.05, 1.3, 2


def out_dir(ds: str, search: str) -> Path:
    return ROOT / "datasets" / ds / "results" / "headroom" / search


def table(ds: str, search: str) -> list[dict]:
    sel = json.loads((ROOT / "datasets" / ds / "results" / "nas-v2" / search / "selection.json").read_text())
    return sorted(sel["table"], key=lambda r: -r["val"]["mean"])


def convert() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import select_by_seeds as sel                           # sets up TF, keras and the dataset adapters
    tf, keras = sel.tf, sel.keras
    for ds_name, search in SEARCHES:
        ds = sel.sv.DMIR_Dataset(task="classification") if ds_name == "dmir" else sel.sv.HighD_Dataset(task="highd_cls")
        xtr = ds._data["train"][0]
        calib = xtr[np.random.default_rng(42).choice(len(xtr), size=500, replace=False)].astype(np.float32)
        out = out_dir(ds_name, search)
        out.mkdir(parents=True, exist_ok=True)
        for r in table(ds_name, search):
            dst = out / f"{r['model']}_int8_io.tflite"
            if dst.exists():
                continue
            net = keras.models.load_model(ROOT / "datasets" / ds_name / "results" / "nas-v2" / search / f"{r['model']}.h5",
                                          compile=False)
            conv = tf.lite.TFLiteConverter.from_keras_model(net)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = lambda: ([calib[i:i + 1]] for i in range(len(calib)))
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
            dst.write_bytes(conv.convert())
            keras.backend.clear_session()
            print(f"{search}: {dst.name} ({dst.stat().st_size} B)", flush=True)


def measure() -> None:
    for ds_name, search in SEARCHES:
        files = sorted(out_dir(ds_name, search).glob("*_int8_io.tflite"))
        done = {json.loads(l)["model"] for p in [out_dir(ds_name, search).parent / "latency_h7b3.jsonl"]
                if p.exists() for l in p.read_text().splitlines() if l.strip()}
        todo = [str(f) for f in files if f.name not in done]
        if not todo:
            continue
        env = dict(os.environ, BENCH_OUT=str(out_dir(ds_name, search).parent / "latency_h7b3.jsonl"))
        subprocess.run([sys.executable, str(ROOT / "unas" / "local_bench.py"), "--board", "h7b3", *todo], env=env,
                       check=True)


def analyse() -> None:
    from scipy import stats
    go = 0
    summary: dict[str, list] = {}
    for ds_name, search in SEARCHES:
        lat = {}
        p = out_dir(ds_name, search).parent / "latency_h7b3.jsonl"
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                lat[r["model"].replace("_int8_io.tflite", "")] = r["duration_ms"]
        t = [dict(r, ms=lat.get(r["model"])) for r in table(ds_name, search)]
        best = t[0]
        se = best["val"]["std"] / math.sqrt(best["val"]["n"])
        for r in t:
            r["p"] = 1.0 if r is best else stats.ttest_ind_from_stats(
                r["val"]["mean"], r["val"]["std"], r["val"]["n"],
                best["val"]["mean"], best["val"]["std"], best["val"]["n"], equal_var=False).pvalue
            r["tied"] = r["p"] > ALPHA
            r["one_se"] = best["val"]["mean"] - r["val"]["mean"] <= se
        missing = [r["model"] for r in t if r["ms"] is None]
        if missing:
            print(f"{search}: no latency for {len(missing)} networks: {missing[:3]}")
            continue
        tied = [r for r in t if r["tied"]]
        fastest = min(tied, key=lambda r: r["ms"])
        by_macs = min(tied, key=lambda r: (r["macs"], r["ms"]))
        by_params = min(tied, key=lambda r: (r["params"], r["ms"]))
        rho = stats.spearmanr([r["macs"] for r in t], [r["ms"] for r in t])[0]
        head = by_macs["ms"] / fastest["ms"]
        go += head >= GO_RATIO
        print(f"\n{search}: {len(tied)} tied of {len(t)}; Spearman(MACs, latency) over 12 = {rho:.2f}")
        for r in t:
            tag = " ".join(k for k, m in (("best", best), ("fastest", fastest), ("by-MACs", by_macs),
                                          ("by-params", by_params)) if r is m)
            print(f"  {'T' if r['tied'] else ' '} {r['model'][:34]:34s} {r['params']:6d} p {r['macs']:7d} MACs "
                  f"val {100 * r['val']['mean']:.2f} (p {r['p']:.3f})  {r['ms']:.4f} ms  "
                  f"{r['ms'] * 1e6 / r['macs'] / 1e3 * 280:.2f} cyc/MAC  {tag}")
        print(f"  headroom (by-MACs / fastest) {head:.2f}; by-params / fastest {by_params['ms'] / fastest['ms']:.2f}; "
              f"best / fastest {best['ms'] / fastest['ms']:.2f}")
        summary.setdefault(ds_name, []).append({"search": search, "tied": len(tied), "spearman_macs_latency": rho, "headroom": head,
                        "by_params_over_fastest": by_params["ms"] / fastest["ms"],
                        "best_over_fastest": best["ms"] / fastest["ms"],
                        "fastest": fastest["model"], "by_macs": by_macs["model"], "best": best["model"]})
    if sum(len(v) for v in summary.values()) == len(SEARCHES):
        print(f"\ndecision: headroom >= {GO_RATIO} in {go} of {len(SEARCHES)} searches -> "
              f"{'GO: build the latency-aware search' if go >= GO_COUNT else 'NO GO'}")
    for ds_name, rows in summary.items():
        (ROOT / "datasets" / ds_name / "results" / "headroom" / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    {"convert": convert, "measure": measure, "analyse": analyse}[sys.argv[1] if len(sys.argv) > 1 else ""]()
