"""Retrain the network a search run chose with five fresh seeds (Part A, 2026-11-02).

Every variant of unas/run_noise_aware.sh ends with one chosen network: single and random through
unas/select_by_seeds.py (selection.json, rule "best"), fixed and adaptive through their own rule
(<name>_final.json). Their own estimates come from the runs that chose them, so they are biased
upwards; this script trains each choice again with seeds the search never used (100-104 unless
FINAL_SEEDS says otherwise), under the search recipe of select_by_seeds.py, and records validation
and test metrics per seed. These runs are not part of any variant's budget.

Run in the WSL dmir_nas venv, from the fork (as select_by_seeds.py):
  ~/dmir_nas/bin/python /mnt/c/Projects/PhD/DIMIR/unas/final_eval.py highd_cls_v4_adaptive_s42

Names without a variant suffix (highd_cls_v4, dmir_cls_v4) are the final searches of the paper,
i.e. run 1 of the single variant with seed 42.

Two-domain pilot (2026-11-04): the choices of the twod_* searches (unas/run_twod.sh, arms pooled and
worst, rule "best" of select_by_seeds.py) are trained on highD and exiD pooled, and FINAL_DATA=pooled
does the same for the choice of a highD-only search (e.g. highd_cls_v4_k3_single_s42). Every choice
then gets the same recipe, early stopping on the pooled validation accuracy, so the arms differ only
in the network they chose; each domain's validation and test scores are recorded.

Output: datasets/<highd|dmir>/results/nas-noise/final_eval.jsonl, and for pooled training
datasets/exid/results/nas-noise/final_eval_pooled.jsonl (one line per search and seed; resumable).
"""
from __future__ import annotations

import json
import os
import re
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_by_seeds as sel                               # noqa: E402  (sets up TF, keras, adapters)

keras = sel.keras
REPO = sel.REPO
SEEDS = [int(x) for x in os.environ.get("FINAL_SEEDS", "100,101,102,103,104").split(",")]
MODES = "single|random|fixed|adaptive|pooled|worst|costfirst"
# Cost-first search (2026-11-10): FINAL_SEL_TAG=_costfirst evaluates the network a search's selection_costfirst.json
# chose (select_by_seeds.py SEL_RESCORE=1, the uNAS baseline with the cost-first choice); pair it with
# FINAL_TAG=_costfirst so its runs go to their own file.
SEL_FILE = f"selection{os.environ.get('FINAL_SEL_TAG', '')}.json"


def chosen(search: str, folder: str) -> tuple[str, Path]:
    results = REPO / "datasets" / folder / "results"
    m = re.search(rf"_({MODES})_s(\d+)$", search)
    mode = m.group(1) if m else "single"
    if mode in ("single", "random", "pooled", "worst", "costfirst"):
        sel_json = json.loads((results / "nas-v2" / search / SEL_FILE).read_text())
        name = sel_json.get("chosen", sel_json["best"])      # files before 2026-11-10 hold "best" only
        return name, results / "nas-v2" / search / f"{name}.h5"
    final = json.loads((results / "nas-noise" / search / f"{search}_final.json").read_text())
    return f"candidate_{final['chosen']['candidate']}", results / "nas-noise" / search / f"{search}_final.h5"


def main(search: str) -> None:
    ds_name, folder = sel.dataset_of(search)
    if "_cls" not in search:
        sys.exit("only the classification searches are set up")
    m = re.search(rf"_({MODES})_s(\d+)$", search)
    mode, search_seed = (m.group(1), int(m.group(2))) if m else ("single", 42)
    pooled = ds_name == "highd_exid" or os.environ.get("FINAL_DATA") == "pooled"
    if pooled and ds_name == "dmir":
        sys.exit("pooled training is highD + exiD; LCIR searches are not part of it")
    if pooled:
        ds, train_ds, out_folder, out_name = sel.sv.TwoDomain_Dataset(task="highd_cls"), "highd_exid", "exid", "final_eval_pooled"
    else:
        ds = sel.sv.DMIR_Dataset(task="classification") if ds_name == "dmir" else sel.sv.HighD_Dataset(task="highd_cls")
        train_ds, out_folder, out_name = ds_name, ds_name, "final_eval"
    model, h5 = chosen(search, folder)
    params = int(keras.models.load_model(h5, compile=False).count_params())
    out = REPO / "datasets" / out_folder / "results" / "nas-noise" / f"{out_name}{os.environ.get('FINAL_TAG', '')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[(r["search"], r["seed"])] = r
    print(f"{search}: chosen {model}, {params:,} parameters, seeds {SEEDS}"
          + (" (trained on highD + exiD)" if pooled else ""), flush=True)
    for seed in SEEDS:
        if (search, seed) in done:
            continue
        res = sel.train_one(str(h5), ds, "cls", seed, train_ds, "pooled")
        rec = {"search": search, "mode": mode, "search_seed": search_seed, "model": model, "params": params,
               "seed": seed, "recipe": "search recipe", "train_order": getattr(ds, "train_order", "as stored"),
               **res, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        done[(search, seed)] = rec
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{model}] seed {seed}: val {res['val']} test {res['test']}", flush=True)
    rows = [done[(search, s)] for s in SEEDS if (search, s) in done]
    for split in ("val", "test"):
        v = [r[split]["acc"] for r in rows]
        print(f"{search} {split} accuracy mean {st.fmean(v):.4f} ± {st.stdev(v) if len(v) > 1 else 0:.4f} "
              f"(n={len(v)})", flush=True)
        for d in (rows[0].get(f"{split}_by_domain") or {}):
            v = [r[f"{split}_by_domain"][d]["acc"] for r in rows]
            print(f"  {d} {split} accuracy mean {st.fmean(v):.4f} ± {st.stdev(v) if len(v) > 1 else 0:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
