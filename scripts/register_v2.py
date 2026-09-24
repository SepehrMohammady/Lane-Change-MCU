"""Add the board runs of the v2 searches' chosen models to the highD measurement registry.

The chosen model of each v2 search is named in datasets/highd/results/nas-v2/<search>/
selection.json (rule "best": best five-seed validation mean). Its builds come from
unas/prepare_deploy_highd.py (accuracy of each file in <stem>_deploy.json) and were
measured with unas/st_benchmark.py (records in benchmarks_api.jsonl). Idempotent.

    python scripts/register_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H = ROOT / "datasets/highd/results"
BOARDS = ("STM32H7B3I-DK", "NUCLEO-F401RE")
VARIANTS = (("fp32", "float32", "f32"), ("int8 PTQ", "float32", "int8"), ("int8 PTQ", "int8", "int8_io"))
SEARCHES = (("highd_cls_v2", "cls_v2", "Searched classifier (v2 search)", "highd_cls"),
            ("highd_ttlc_v2", "ttlc_v2", "Searched regressor (v2 search)", "highd_ttlc"))


def main() -> None:
    api = set()
    for line in (H / "deploy/benchmarks_api.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if not r.get("error"):
            api.add((r["model"], r["board"]))
    reg_path = H / "deploy/measurements.json"
    text = reg_path.read_text(encoding="utf-8")
    second = text.splitlines()[1]
    indent = len(second) - len(second.lstrip())
    reg = json.loads(text)
    for search, mid, label, task in SEARCHES:
        sel_path = H / "nas-v2" / search / "selection_permuted.json"      # shuffled-order re-selection
        if not sel_path.exists():
            sel_path = H / "nas-v2" / search / "selection.json"
        if not sel_path.exists():
            continue
        sel = json.loads(sel_path.read_text())
        stem = f"{task}_{sel['best']}"
        dj_path = H / "deploy" / f"{stem}_deploy.json"
        if not dj_path.exists():
            continue
        dj = json.loads(dj_path.read_text())
        variants = []
        for precision, io, tag in VARIANTS:
            name = f"{stem}_{tag}.tflite"
            boards = {b: {"api": name} for b in BOARDS if (name, b) in api}
            if boards:
                metric = {k: v for k, v in dj["variants"][tag].items() if k != "bytes"}
                variants.append({"precision": precision, "io": io, "metric": metric, "boards": boards,
                                 "source": f"datasets/highd/results/deploy/{stem}_deploy.json; benchmarks_api.jsonl"})
        entry = {"id": mid, "label": label, "role": "searched", "task": task, "params": dj["params"],
                 "note": (f"Chosen by its five-seed validation mean among the 12 best of {sel['saved_candidates']} saved "
                          f"candidates of the v2 search (unas/select_by_seeds.py); file {sel['best']}.h5."),
                 "variants": variants}
        ids = [m["id"] for m in reg["models"]]
        reg["models"] = [entry if m["id"] == mid else m for m in reg["models"]] + ([entry] if mid not in ids else [])
        print(f"registered {mid}: {len(variants)} builds")
    reg_path.write_text(json.dumps(reg, indent=indent, ensure_ascii=False) + ("\n" if text.endswith("\n") else ""),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
