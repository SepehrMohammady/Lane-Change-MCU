"""Add the board runs of the hand-designed CNNs to the measurement registries.

The builds come from unas/deploy_hand_cnn.py (accuracy of each file in
<stem>_deploy.json) and were measured with unas/st_benchmark.py (records in
benchmarks_api.jsonl). The registry entries point at those records by file name
({"api": ...}), as the searched models' API runs do, so dashboard/build.py reads the
raw numbers. Idempotent: rerunning replaces the entries it wrote.

    python scripts/register_hand_cnn.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARDS = ("STM32H7B3I-DK", "NUCLEO-F401RE")
NOTE = ("Hand-designed depthwise-separable CNN (PyTorch), rebuilt layer by layer in Keras with the "
        "same weights (unas/deploy_hand_cnn.py, outputs agree within 1e-4) and converted and measured like "
        "the searched models.")
VARIANTS = (("fp32", "float32", None), ("int8 PTQ", "float32", "int8"), ("int8 PTQ", "int8", "int8_io"))


def measured(api_file: Path) -> set[tuple[str, str]]:
    ok = set()
    for line in api_file.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "error" not in r or r["error"] is None:
            ok.add((r["model"], r["board"]))
    return ok


def variants(deploy_dir: Path, stem: str, f32: str, api: set, keep_metric: dict | None = None) -> list[dict]:
    dj = json.loads((deploy_dir / f"{stem}_deploy.json").read_text())
    out = []
    for precision, io, tag in VARIANTS:
        tag = tag or f32
        name = f"{stem}_{tag}.tflite"
        boards = {b: {"api": name} for b in BOARDS if (name, b) in api}
        if not boards:
            continue
        metric = {k: v for k, v in dj["variants"][tag].items() if k != "bytes"}
        if keep_metric and tag == f32:
            metric = {**keep_metric, **metric}
        # the explorer shows the first key: same order as the searched models' entries
        order = ["acc", "f1", "auc", "tau_c_s", "mae", "rmse"]
        metric = {k: metric[k] for k in order if k in metric} | {k: v for k, v in metric.items() if k not in order}
        out.append({"precision": precision, "io": io, "metric": metric, "boards": boards,
                    "source": f"{deploy_dir.relative_to(ROOT).as_posix()}/{stem}_deploy.json; benchmarks_api.jsonl"})
    return out, dj["params"]


def update(ds: str, entries: list[dict]) -> None:
    reg_path = ROOT / f"datasets/{ds}/results/deploy/measurements.json"
    text = reg_path.read_text(encoding="utf-8")
    second = text.splitlines()[1]
    indent = len(second) - len(second.lstrip())          # keep the file's own layout
    reg = json.loads(text)
    ids = {e["id"] for e in entries}
    new = {e["id"]: e for e in entries}
    existing = [m["id"] for m in reg["models"]]
    reg["models"] = [new.get(m["id"], m) for m in reg["models"]] + [e for e in entries if e["id"] not in existing]
    reg_path.write_text(json.dumps(reg, indent=indent, ensure_ascii=False) + ("\n" if text.endswith("\n") else ""),
                        encoding="utf-8")
    print(f"{reg_path.relative_to(ROOT)}: {', '.join(sorted(ids))}")


def main() -> None:
    d = ROOT / "datasets/dmir/results/deploy"
    api = measured(d / "benchmarks_api.jsonl")
    entries = []
    for mid, label, task, stem in (("dscnn_cls", "Hand-designed DSCNN", "classification", "lcir_dscnn_classification"),
                                   ("dscnn_lcr", "Hand-designed, time to right lane change", "regression_lcr",
                                    "lcir_dscnn_regression_lcr"),
                                   ("dscnn_lcl", "Hand-designed, time to left lane change", "regression_lcl",
                                    "lcir_dscnn_regression_lcl")):
        v, params = variants(d, stem, "float32", api)
        entries.append({"id": mid, "label": label, "role": "baseline", "task": task, "params": params,
                        "note": NOTE + " Weights: scripts/run_baseline.py <task> --save (one run).", "variants": v})
    update("dmir", entries)

    h = ROOT / "datasets/highd/results/deploy"
    api = measured(h / "benchmarks_api.jsonl")
    reg = json.loads((h / "measurements.json").read_text(encoding="utf-8"))
    old = {m["id"]: m for m in reg["models"]}
    entries = []
    for mid, label, task, stem in (("baseline_cls", "Hand-designed CNN", "highd_cls", "highd_dscnn_cls"),
                                   ("baseline_ttlc", "Hand-designed CNN, time to lane change", "highd_ttlc",
                                    "highd_dscnn_ttlc")):
        prev = old.get(mid, {}).get("variants", [{}])[0].get("metric", {})
        v, params = variants(h, stem, "f32", api, keep_metric=prev)
        entries.append({"id": mid, "label": label, "role": "baseline", "task": task, "params": params,
                        "note": NOTE + " Weights: the committed checkpoints scored in their_protocol_eval.json.",
                        "variants": v})
    update("highd", entries)


if __name__ == "__main__":
    main()
