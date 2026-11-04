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

Output: datasets/<highd|dmir>/results/nas-noise/final_eval.jsonl (one line per search and seed;
resumable).
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


def chosen(search: str, ds_name: str) -> tuple[str, Path]:
    results = REPO / "datasets" / ds_name / "results"
    m = re.search(r"_(single|random|fixed|adaptive)_s(\d+)$", search)
    mode = m.group(1) if m else "single"
    if mode in ("single", "random"):
        sel_json = json.loads((results / "nas-v2" / search / "selection.json").read_text())
        return sel_json["best"], results / "nas-v2" / search / f"{sel_json['best']}.h5"
    final = json.loads((results / "nas-noise" / search / f"{search}_final.json").read_text())
    return f"candidate_{final['chosen']['candidate']}", results / "nas-noise" / search / f"{search}_final.h5"


def main(search: str) -> None:
    ds_name = "dmir" if search.startswith("dmir") else "highd"
    if "_cls" not in search:
        sys.exit("only the classification searches are set up")
    m = re.search(r"_(single|random|fixed|adaptive)_s(\d+)$", search)
    mode, search_seed = (m.group(1), int(m.group(2))) if m else ("single", 42)
    ds = sel.sv.DMIR_Dataset(task="classification") if ds_name == "dmir" else sel.sv.HighD_Dataset(task="highd_cls")
    model, h5 = chosen(search, ds_name)
    params = int(keras.models.load_model(h5, compile=False).count_params())
    out = REPO / "datasets" / ds_name / "results" / "nas-noise" / f"final_eval{os.environ.get('FINAL_TAG', '')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[(r["search"], r["seed"])] = r
    print(f"{search}: chosen {model}, {params:,} parameters, seeds {SEEDS}", flush=True)
    for seed in SEEDS:
        if (search, seed) in done:
            continue
        res = sel.train_one(str(h5), ds, "cls", seed, ds_name)
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


if __name__ == "__main__":
    main(sys.argv[1])
