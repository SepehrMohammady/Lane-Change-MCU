"""exiD experiments with the hand-designed DSCNN, five seeds each.

1. scratch     exiD model trained from scratch on exiD (exiD normalization)
2. source      highD model trained on highD, weights kept (highD normalization)
3. zero-shot   the highD model applied to exiD as it is (highD normalization)
4. fine-tune   the highD model trained further on exiD with the same recipe;
               only the initial weights and the normalization (highD's) differ
5. fraction    scratch and fine-tune with 10% and 25% of the exiD training
               scenarios (same random subset per seed for both)
6. reverse     the exiD model applied to highD (exiD normalization)

Every trained model is evaluated on the exiD and highD test sets with window
metrics, the early-prediction metrics of Mozaffari et al. (classification) and
a breakdown by exiD lane-change kind. Training runs are logged to the datasets'
experiments.jsonl; evaluations go to datasets/exid/results/transfer.jsonl.
Resume-safe: an evaluation already in transfer.jsonl is skipped.

Run: .venv\\Scripts\\python datasets/exid/run_experiments.py [stage ...]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from src import lc_windows as W                                     # noqa: E402
from src.models.baseline import BaselineDSCNN                      # noqa: E402

EXID = {"prep": HERE / "data" / "prepared", "logs": HERE / "logs", "weights": HERE / "results" / "weights",
        "dataset": "exiD", "protocol": "EarlyLCPred T-IV 2022, adapted to exiD"}
HIGHD = {"prep": ROOT / "datasets/highd/data/prepared", "logs": ROOT / "datasets/highd/logs",
         "weights": ROOT / "datasets/highd/results/source_weights",
         "dataset": "highD", "protocol": "EarlyLCPred T-IV 2022"}
OUT = HERE / "results" / "transfer.jsonl"
SEEDS = [0, 1, 2, 3, 4]
FRACTIONS = [0.10, 0.25]
KIND_NAMES = json.loads((EXID["prep"] / "meta.json").read_text())["kinds"]


def done() -> set:
    if not OUT.exists():
        return set()
    return {(r["method"], r["task"], r["seed"], r.get("fraction", 1.0), r["eval_on"])
            for r in map(json.loads, OUT.read_text(encoding="utf-8").splitlines())}


def test_split(ds: dict):
    feats, label, cross, extra = W.load_split(ds["prep"], "test", extras=("kind",))
    return feats, label, cross, extra.get("kind")


def norm_of(ds: dict):
    x, _, _ = W.make_windows(*W.load_split(ds["prep"], "train"))
    return W.norm_stats(x)


def write(rec: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    m = rec["metrics"]
    short = {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}
    print(f"[{rec['method']} {rec['task']} seed {rec['seed']} frac {rec.get('fraction', 1.0)} on {rec['eval_on']}] {short}", flush=True)


def evaluate_all(model, device, task, lo, hi, meta: dict, tests: dict, skip: set):
    for name, (feats, label, cross, kind) in tests.items():
        key = (meta["method"], task, meta["seed"], meta.get("fraction", 1.0), name)
        if key in skip:
            continue
        groups = kind if name == "exiD" else None
        m = W.evaluate_scenarios(model, feats, label, cross, lo, hi, task, device,
                                 groups=groups, group_names=KIND_NAMES if groups is not None else None)
        write({**meta, "task": task, "eval_on": name, "metrics": m})


def load_model(path: Path, task: str, device):
    model = BaselineDSCNN(n_features=18, n_outputs=3 if task == "cls" else 1).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def main(stages: set):
    skip = done()
    tests = {"exiD": test_split(EXID), "highD": test_split(HIGHD)}
    lo_ex, hi_ex = norm_of(EXID)
    lo_hd, hi_hd = norm_of(HIGHD)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def need(method, task, seed, fraction=1.0):
        return any((method, task, seed, fraction, n) not in skip for n in tests)

    for task in ("cls", "ttlc"):
        for seed in SEEDS:
            # 1. exiD from scratch
            w = EXID["weights"] / f"exid_baseline_{task}_seed{seed}.pt"
            if "scratch" in stages and (need("scratch", task, seed) or not w.exists()):
                if not w.exists():
                    W.train_dscnn(task, f"exid_baseline_{task}_seed{seed}", prep=EXID["prep"], logs=EXID["logs"],
                                  results=EXID["weights"], dataset=EXID["dataset"], protocol=EXID["protocol"],
                                  seed=seed, save_weights=True, verbose=False)
                evaluate_all(load_model(w, task, device), device, task, lo_ex, hi_ex,
                             {"method": "scratch", "trained_on": "exiD", "seed": seed}, tests, skip)

            # 2-3. highD source model and its zero-shot transfer
            src = HIGHD["weights"] / f"highd_source_{task}_seed{seed}.pt"
            if "transfer" in stages:
                if not src.exists():
                    W.train_dscnn(task, f"highd_source_{task}_seed{seed}", prep=HIGHD["prep"], logs=HIGHD["logs"],
                                  results=HIGHD["weights"], dataset=HIGHD["dataset"], protocol=HIGHD["protocol"],
                                  seed=seed, save_weights=True, verbose=False,
                                  extra_config={"purpose": "source model for highD -> exiD transfer"})
                evaluate_all(load_model(src, task, device), device, task, lo_hd, hi_hd,
                             {"method": "zero-shot", "trained_on": "highD", "seed": seed}, tests, skip)

                # 4. fine-tune on exiD, same recipe, highD normalization
                if need("fine-tune", task, seed):
                    r = W.train_dscnn(task, f"exid_finetune_{task}_seed{seed}", prep=EXID["prep"], logs=EXID["logs"],
                                      results=EXID["weights"], dataset=EXID["dataset"], protocol=EXID["protocol"],
                                      seed=seed, save_weights=True, verbose=False,
                                      init_state=torch.load(src, weights_only=True), norm=(lo_hd, hi_hd),
                                      extra_config={"init": f"highd_source_{task}_seed{seed}",
                                                    "normalization": "minmax-train of highD"})
                    evaluate_all(r["model"], device, task, lo_hd, hi_hd,
                                 {"method": "fine-tune", "trained_on": "highD then exiD", "seed": seed}, tests, skip)

            # 5. less exiD data: scratch against fine-tune on the same subset
            if "fraction" in stages:
                for frac in FRACTIONS:
                    if need("scratch", task, seed, frac):
                        r = W.train_dscnn(task, f"exid_scratch_{task}_frac{int(frac * 100)}_seed{seed}",
                                          prep=EXID["prep"], logs=EXID["logs"], results=None,
                                          dataset=EXID["dataset"], protocol=EXID["protocol"], seed=seed,
                                          train_fraction=frac, verbose=False)
                        evaluate_all(r["model"], device, task, *r["norm"],
                                     {"method": "scratch", "trained_on": "exiD", "seed": seed, "fraction": frac},
                                     tests, skip)
                    if need("fine-tune", task, seed, frac) and src.exists():
                        r = W.train_dscnn(task, f"exid_finetune_{task}_frac{int(frac * 100)}_seed{seed}",
                                          prep=EXID["prep"], logs=EXID["logs"], results=None,
                                          dataset=EXID["dataset"], protocol=EXID["protocol"], seed=seed,
                                          train_fraction=frac, verbose=False,
                                          init_state=torch.load(src, weights_only=True), norm=(lo_hd, hi_hd),
                                          extra_config={"init": f"highd_source_{task}_seed{seed}",
                                                        "normalization": "minmax-train of highD"})
                        evaluate_all(r["model"], device, task, lo_hd, hi_hd,
                                     {"method": "fine-tune", "trained_on": "highD then exiD", "seed": seed,
                                      "fraction": frac}, tests, skip)
        skip = done()
    print("EXPERIMENTS FINISHED", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]) or {"scratch", "transfer", "fraction"})
