"""Add the board runs of the v2-v4 searches' chosen models to the measurement registries.

The chosen model of each search is named in datasets/<ds>/results/nas-v2/<search>/
selection.json (rule "best": best five-seed validation mean). Its builds come from
unas/prepare_deploy_highd.py or unas/prepare_deploy_lcir.py (accuracy of each file in
<stem>_deploy.json) and were measured with unas/st_benchmark.py (records in
benchmarks_api.jsonl). Idempotent.

    python scripts/register_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOARDS = ("STM32H7B3I-DK", "NUCLEO-F401RE")
# per dataset: (precision, io, file tag) of each build, and
# (search, registry id, label, task, file stem or None for "<task>_<chosen model>")
DATASETS = {
    "highd": {
        "variants": (("fp32", "float32", "f32"), ("int8 PTQ", "float32", "int8"), ("int8 PTQ", "int8", "int8_io")),
        "searches": (
            ("highd_cls_v2", "cls_v2", "Searched classifier (v2 search)", "highd_cls", None),
            ("highd_ttlc_v2_p", "ttlc_v2", "Searched regressor (v2 search)", "highd_ttlc", None),
            ("highd_cls_v3", "cls_v3", "Searched classifier (v3 search, budgets at the hand-designed cost)",
             "highd_cls", None),
            ("highd_cls_v4", "cls_v4", "Searched classifier (v4 search, faithful cost count)", "highd_cls", None),
            ("highd_ttlc_v4", "ttlc_v4", "Searched regressor (v4 search, faithful cost count)", "highd_ttlc", None),
        ),
    },
    "dmir": {
        "variants": (("fp32", "float32", "float32"), ("int8 PTQ", "float32", "int8"), ("int8 PTQ", "int8", "int8_io")),
        "searches": (
            ("dmir_cls_v4", "cls_v4", "Searched, v4 search (faithful cost count)", "classification",
             "lcir_v4_classification"),
            ("dmir_lcr_v4", "lcr_v4", "Searched, v4 search, time to right lane change", "regression_lcr",
             "lcir_v4_regression_lcr"),
            ("dmir_lcl_v4", "lcl_v4", "Searched, v4 search, time to left lane change", "regression_lcl",
             "lcir_v4_regression_lcl"),
        ),
    },
}
# the explorer shows the first metric key: same order as the other entries
ORDER = ["acc", "f1", "auc", "tau_c_s", "mae", "rmse"]


def register(ds: str) -> None:
    res = ROOT / "datasets" / ds / "results"
    spec = DATASETS[ds]
    api = set()
    for line in (res / "deploy/benchmarks_api.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("error"):
            api.add((r["model"], r["board"]))
    reg_path = res / "deploy/measurements.json"
    text = reg_path.read_text(encoding="utf-8")
    second = text.splitlines()[1]
    indent = len(second) - len(second.lstrip())
    reg = json.loads(text)
    for search, mid, label, task, fixed_stem in spec["searches"]:
        sel_path = res / "nas-v2" / search / "selection_permuted.json"      # shuffled-order re-selection
        if not sel_path.exists():
            sel_path = res / "nas-v2" / search / "selection.json"
        if not sel_path.exists():
            continue
        sel = json.loads(sel_path.read_text())
        stem = fixed_stem or f"{task}_{sel['best']}"
        dj_path = res / "deploy" / f"{stem}_deploy.json"
        if not dj_path.exists():
            continue
        dj = json.loads(dj_path.read_text())
        if fixed_stem and dj.get("source_h5") != f"{sel['best']}.h5":
            raise SystemExit(f"{stem}_deploy.json was not built from the chosen model {sel['best']}.h5")
        variants = []
        for precision, io, tag in spec["variants"]:
            name = f"{stem}_{tag}.tflite"
            boards = {b: {"api": name} for b in BOARDS if (name, b) in api}
            if boards:
                metric = {k: v for k, v in dj["variants"][tag].items() if k != "bytes"}
                metric = {k: metric[k] for k in ORDER if k in metric} | {k: v for k, v in metric.items() if k not in ORDER}
                variants.append({"precision": precision, "io": io, "metric": metric, "boards": boards,
                                 "source": f"datasets/{ds}/results/deploy/{stem}_deploy.json; benchmarks_api.jsonl"})
        entry = {"id": mid, "label": label, "role": "searched", "task": task, "params": dj["params"],
                 "note": (f"Chosen by its five-seed validation mean among the 12 best of {sel['saved_candidates']} saved "
                          f"candidates of the {search} search (unas/select_by_seeds.py); file {sel['best']}.h5."),
                 "variants": variants}
        ids = [m["id"] for m in reg["models"]]
        reg["models"] = [entry if m["id"] == mid else m for m in reg["models"]] + ([entry] if mid not in ids else [])
        print(f"{ds}: registered {mid}: {len(variants)} builds")
    reg_path.write_text(json.dumps(reg, indent=indent, ensure_ascii=False) + ("\n" if text.endswith("\n") else ""),
                        encoding="utf-8")


def main() -> None:
    for ds in DATASETS:
        register(ds)


if __name__ == "__main__":
    main()
