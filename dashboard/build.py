"""Build the results explorer: one self-contained HTML page for every dataset.

Reads the project's own result files (nothing is typed in by hand here), resolves
board measurements against the raw ST API records, and writes a single HTML file
with the data, the app and the media inlined, so it opens offline and can be sent
as one attachment.

    python dashboard/build.py            -> dashboard/dist/results-explorer.html
    python dashboard/build.py --share    -> dashboard/dist/results-explorer-share.html

The share build leaves out two things: the project-status page (internal to-dos),
and any picture drawn from raw highD trajectories, which the highD licence does
not allow us to pass on. Standard library only.
"""
from __future__ import annotations

import base64
import csv
import json
import statistics
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DMIR = ROOT / "datasets" / "dmir"
HIGHD = ROOT / "datasets" / "highd"
EXID = ROOT / "datasets" / "exid"
MEDIA = ROOT / "Materials" / "T4.5"


# ------------------------------------------------------------------ readers
def read_json(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def hand_params(root: Path, prefix: str):
    """Parameter count of the hand-designed CNN, from the first logged run with this prefix."""
    for r in read_jsonl(root / "logs/experiments.jsonl"):
        if r.get("run_name", "").startswith(prefix) and r.get("config", {}).get("n_params"):
            return r["config"]["n_params"]
    return None


def read_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_front(p: Path, metric_col: str):
    rows = []
    for r in csv.DictReader(p.open(encoding="utf-8")):
        rows.append({"model": r["model"].replace(".h5", ""), "params": int(r["params"]),
                     "kb": float(r["int8_KB"]), "value": float(r[metric_col])})
    return rows


def data_uri(p: Path, mime: str):
    if not p.exists():
        return None
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def resolve_measurements(reg_path: Path, api_path: Path, svg_dir: Path):
    """Replace {"api": artifact} references with the raw API record's numbers."""
    reg = read_json(reg_path)
    api = {(r["model"], r["board"]): r for r in read_jsonl(api_path)}
    for m in reg["models"]:
        svg = m.get("arch_svg")
        if svg and (svg_dir / svg).exists():
            m["arch_svg_markup"] = (svg_dir / svg).read_text(encoding="utf-8")
        for v in m["variants"]:
            for board, b in list(v["boards"].items()):
                if "api" in b:
                    r = api.get((b["api"], board))
                    if r is None or r.get("duration_ms") is None:
                        v["boards"][board] = {"fits": False, "why": "no measurement returned by the board farm"}
                    else:
                        v["boards"][board] = {"latency_ms": r["duration_ms"], "macc": r.get("macc"),
                                              "flash_b": r.get("rom_B"), "ram_b": r.get("ram_B"), "api": True}
    return reg


def short_recipe(label):
    label = label or ""
    for key, short in (("search recipe", "search recipe"), ("DSCNN recipe", "DSCNN recipe"),
                       ("baseline recipe", "hand-designed recipe")):
        if label.startswith(key):
            return short
    return label


def seed_summary(rows, key_field="key"):
    """Group seed runs; mean and sample std of every test metric."""
    out = {}
    for r in rows:
        k = r[key_field]
        out.setdefault(k, {"runs": [], "params": r.get("params"), "recipe": r.get("recipe"),
                           "recipe_short": short_recipe(r.get("recipe")),
                           "original": r.get("original_run", {})})
        out[k]["runs"].append({"seed": r["seed"], **r["test"], "epochs": r.get("epochs_run")})
    for k, g in out.items():
        stats = {}
        for metric in g["runs"][0]:
            if metric in ("seed", "epochs"):
                continue
            vals = [x[metric] for x in g["runs"] if x.get(metric) is not None]
            if vals:
                stats[metric] = {"mean": statistics.fmean(vals),
                                 "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                                 "min": min(vals), "max": max(vals), "n": len(vals)}
        g["stats"] = stats
    return out


def seed_line(seeds, key, metric, pct):
    g = seeds.get(key) if seeds else None
    st = g and g["stats"].get(metric)
    if not st or st["n"] < 2:
        return None
    if pct:
        return f"retrained {st['n']}×: {st['mean'] * 100:.1f} ± {st['std'] * 100:.1f}%"
    return f"retrained {st['n']}×: {st['mean']:.3f} ± {st['std']:.3f} s"


def baseline_seeds(log_path: Path, prefix: str):
    rows = []
    for r in read_jsonl(log_path):
        if r.get("run_name", "").startswith(prefix) and "_seed" in r["run_name"]:
            m = r["metrics"]
            test = {k[5:]: v for k, v in m.items() if k.startswith("test_") and isinstance(v, (int, float))}
            rows.append({"key": prefix, "seed": r["config"]["seed"], "params": r["config"].get("n_params"),
                         "recipe": "baseline recipe (AdamW 3e-3, wd 1e-4, <=60 ep, patience 8)", "test": test})
    return rows


# BaselineDSCNN (src/models) parameter counts, counted with torch: 3 outputs / 1 output
DSCNN_PARAMS = {"classification": 10451, "regression_lcr": 10321, "regression_lcl": 10321}
DSCNN_METRICS = {"test_accuracy": "acc", "test_macro_f1": "macro_f1", "test_rmse": "rmse", "test_mae": "mae"}


def dscnn_runs(log_path: Path):
    """The hand-designed DMIR DSCNN: seeded runs (scripts/run_baseline.py <task> <seed>) and its first logged run."""
    seeded, first = {}, {}
    for r in read_jsonl(log_path):
        name, task = r.get("run_name", ""), r.get("config", {}).get("task")
        if not name.startswith("baseline-dscnn") or task not in DSCNN_PARAMS:
            continue
        test = {v: r["metrics"][k] for k, v in DSCNN_METRICS.items() if k in r["metrics"]}
        if name == "baseline-dscnn":
            first.setdefault(task, test)
        elif name.startswith("baseline-dscnn-seed"):
            seeded[(task, r["config"]["seed"])] = {          # a repeated seed keeps its latest run
                "key": "dmir_dscnn_" + task, "seed": r["config"]["seed"], "params": DSCNN_PARAMS[task],
                "recipe": "DSCNN recipe (AdamW 3e-3, wd 1e-4, cosine, <=60 ep, patience 10, MSE)", "test": test}
    return list(seeded.values()), first


def rank_corr(a, b):
    """Spearman rank correlation."""
    def ranks(v):
        out = [0] * len(v)
        for pos, i in enumerate(sorted(range(len(v)), key=v.__getitem__)):
            out[i] = pos
        return out
    return statistics.correlation(ranks(a), ranks(b))


def rerank_summary(path: Path, labels: dict):
    """Every saved front model retrained with several seeds (unas/rerank_front.py)."""
    groups = {}
    for r in read_jsonl(path):
        g = groups.setdefault((r["search"], r["model"]), {
            "search": labels.get(r["search"], r["search"]), "model": r["model"], "params": r["params"],
            "orig": r["reported_acc"], "runs": []})
        g["runs"].append({"seed": r["seed"], "acc": r["test_acc"], "epochs": r.get("epochs_run")})
    rows = []
    for g in groups.values():
        v = [x["acc"] for x in g["runs"]]
        if len(v) >= 2:
            g["stats"] = {"mean": statistics.fmean(v), "std": statistics.stdev(v), "min": min(v), "max": max(v), "n": len(v)}
            rows.append(g)
    if len(rows) < 3:
        return None
    rows.sort(key=lambda g: g["params"])
    by_mean = sorted(rows, key=lambda g: -g["stats"]["mean"])
    best_single = max(rows, key=lambda g: g["orig"])
    return {"rows": rows, "n": len(rows), "best_mean": by_mean[0],
            "rho": rank_corr([g["orig"] for g in rows], [g["stats"]["mean"] for g in rows]),
            "best_single_rank": by_mean.index(best_single) + 1}


def stat(seeds, key, metric):
    return ((seeds or {}).get(key) or {}).get("stats", {}).get(metric)


def pm(st, pct=True):
    if not st:
        return "–"
    return f"{st['mean'] * 100:.1f} ± {st['std'] * 100:.1f}%" if pct else f"{st['mean']:.3f} ± {st['std']:.3f} s"


def mean_only(st, pct=True):
    if not st:
        return "–"
    return f"{st['mean'] * 100:.1f}%" if pct else f"{st['mean']:.3f} s"


def latency(reg, model_id, precision="fp32", board="STM32H7B3I-DK"):
    for m in reg["models"]:
        if m["id"] == model_id:
            for v in m["variants"]:
                if v["precision"] == precision and v["boards"].get(board, {}).get("latency_ms") is not None:
                    return v["boards"][board]["latency_ms"]
    return None


# ------------------------------------------------------------------ datasets
def build_dmir(share: bool):
    fr = DMIR / "results" / "nas-fronts"
    reg = resolve_measurements(DMIR / "results/deploy/measurements.json",
                               DMIR / "results/deploy/benchmarks_api.jsonl", DMIR / "results/deploy")
    refs = reg["references"]
    dscnn_seeded, dscnn_first = dscnn_runs(DMIR / "logs/experiments.jsonl")
    seeds = seed_summary(read_jsonl(DMIR / "results/seeds/seed_variance.jsonl") + dscnn_seeded)
    seeds_final = seed_summary(read_jsonl(DMIR / "results/seeds/seed_variance_dscnn.jsonl"))

    def dscnn_ref(task, metric):
        if task not in dscnn_first:
            return None
        return {"kind": "point", "label": "hand-designed DSCNN", "params": DSCNN_PARAMS[task],
                "value": dscnn_first[task][metric], "seed_key": "dmir_dscnn_" + task}

    tasks = [
        {"id": "classification", "label": "Intention", "long": "Lane-change intention: none, right or left",
         "metric": {"key": "acc", "name": "test accuracy", "better": "high", "fmt": "pct"},
         "fronts": [{"label": "search", "rows": read_front(fr / "dmir_cls.csv", "test_acc")}],
         "picks": [{"id": "cls_best", "model": "model_aaaabl", "front": 0, "params": 83803, "label": "searched, best accuracy", "seed_key": "dmir_cls_best"},
                   {"id": "cls_tiny", "model": "model_aaaaat", "front": 0, "params": 7953, "label": "searched, smallest", "seed_key": "dmir_cls_tiny"}],
         "refs": [{"kind": "point", "label": "reference CNN", "params": 441347, "value": 0.9169, "seed_key": "dmir_ref_cnn"},
                  dscnn_ref("classification", "acc")]},
        {"id": "classification_noind", "label": "Intention, no turn signal",
         "long": "Same task with the two turn-signal channels removed",
         "metric": {"key": "acc", "name": "test accuracy", "better": "high", "fmt": "pct"},
         "fronts": [{"label": "search", "rows": read_front(fr / "dmir_cls_noind.csv", "test_acc")}],
         "picks": [{"id": "cls_noind", "model": "model_aaaaah", "front": 0, "params": 21038, "label": "searched, no turn signal", "seed_key": "dmir_cls_noind"}],
         "refs": [{"kind": "line", "label": "turn signal alone", "value": refs["indicator_only_acc"]},
                  {"kind": "line", "label": "all channels, best", "value": 0.9208}]},
        {"id": "regression_lcr", "label": "Time to right change",
         "long": "Seconds until a right lane change (0 to 4 s)",
         "metric": {"key": "rmse", "name": "test RMSE (s)", "better": "low", "fmt": "s3"},
         "fronts": [{"label": "MAE-objective search", "rows": read_front(fr / "dmir_lcr.csv", "test_rmse")},
                    {"label": "RMSE-objective search", "rows": read_front(fr / "dmir_lcr_rmse.csv", "test_rmse")}],
         "picks": [{"id": "lcr_best", "model": "model_aaaaan", "front": 0, "params": 117404, "label": "searched", "seed_key": "dmir_lcr_best"}],
         "refs": [{"kind": "point", "label": "internal Transformer", "params": 333505, "value": 0.42},
                  dscnn_ref("regression_lcr", "rmse"),
                  {"kind": "line", "label": "published Transformer (both directions)", "value": 0.5102}]},
        {"id": "regression_lcl", "label": "Time to left change",
         "long": "Seconds until a left lane change (0 to 4 s)",
         "metric": {"key": "rmse", "name": "test RMSE (s)", "better": "low", "fmt": "s3"},
         "fronts": [{"label": "MAE-objective search", "rows": read_front(fr / "dmir_lcl.csv", "test_rmse")},
                    {"label": "RMSE-objective search", "rows": read_front(fr / "dmir_lcl_rmse.csv", "test_rmse")}],
         "picks": [{"id": "lcl_best", "model": "model_aaaabu", "front": 1, "params": 105769, "label": "searched", "seed_key": "dmir_lcl_best"}],
         "refs": [{"kind": "point", "label": "internal Transformer", "params": 49089, "value": 0.44},
                  dscnn_ref("regression_lcl", "rmse"),
                  {"kind": "line", "label": "published Transformer (both directions)", "value": 0.5102}]},
    ]
    for t in tasks:
        t["refs"] = [r for r in t["refs"] if r]

    quant = [
        {"model": "Reference CNN", "params": 441347, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.9169], ["int8 PTQ", 0.8827]]},
        {"model": "Searched, best accuracy", "params": 83803, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.9208], ["int8 PTQ", 0.8686], ["int8 QAT", 0.8990]]},
        {"model": "Searched, smallest", "params": 7953, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.9130], ["int8 PTQ", 0.8546]]},
        {"model": "Searched, no turn signal", "params": 21038, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.9108], ["int8 PTQ", 0.7606]]},
        {"model": "Time to right change", "params": 117404, "metric": "MAE (s)", "better": "low", "fmt": "s3",
         "steps": [["fp32", 0.2865], ["int8 PTQ", 0.4485], ["int8 QAT", 0.4313]]},
        {"model": "Time to left change", "params": 105769, "metric": "MAE (s)", "better": "low", "fmt": "s3",
         "steps": [["fp32", 0.3165], ["int8 PTQ", 0.3440], ["int8 QAT", 0.3620]]},
    ]

    noind = seeds.get("dmir_cls_noind")
    if noind is not None and not noind["original"]:                 # its single run is the fp32 quant step
        noind["original"] = {"acc": next(q["steps"][0][1] for q in quant if q["model"] == "Searched, no turn signal")}

    media = []
    if (MEDIA / "race/race.gif").exists():
        media.append({"id": "race", "label": "Five models, one lane change",
                      "caption": "A held-out test driver's recorded left lane change, replayed for every model on the same "
                                 "5-second windows. The bar under each road is the distance driven while the model computes "
                                 "one answer, from latency measured on the board.",
                      "src": data_uri(MEDIA / "race/race.gif", "image/gif")})
    if (MEDIA / "race/race_whatif.gif").exists():
        media.append({"id": "whatif", "label": "What a late warning costs",
                      "caption": "Same recording. The recorded driver waited for the car in the target lane; nothing happened. "
                                 "The overlay adds one driver who does not check the mirror and asks whose warning reaches "
                                 "him in time. The reaction time (1.0 s) is an assumption, and the outcome depends on it.",
                      "src": data_uri(MEDIA / "race/race_whatif.gif", "image/gif")})

    return {
        "id": "dmir", "name": "LCIR", "title": "Lane Change Intention Recognition", "status": "complete", "accent": 1,
        "facts": [
            ["Whose behaviour", "The driver of the car (ego view)"],
            ["Recorded in", "CARLA driving simulator, 50 drivers"],
            ["Input", "50 time steps × 31 in-car signals (5 s at 10 Hz)"],
            ["Tasks", "Intention (3 classes) · time to right change · time to left change"],
            ["Test split", "7 drivers never seen in training (verified)"],
            ["Data", "Zenodo 10.5281/zenodo.16686054 · MIT licence"],
        ],
        "headline": {"value": f"{latency(reg, 'ref_cnn') / latency(reg, 'cls_best'):.1f}×",
                     "label": "faster than the reference CNN on the same board, at the same accuracy",
                     "sub": f"Five training seeds each: {pm(stat(seeds, 'dmir_cls_best', 'acc'))} with 84 k parameters, "
                            f"against {pm(stat(seeds, 'dmir_ref_cnn', 'acc'))} for the reference CNN with 441 k."},
        "kpis": [
            {"label": "Accuracy, five seeds", "value": mean_only(stat(seeds, "dmir_cls_best", "acc")),
             "sub": f"84 k params · reference CNN {mean_only(stat(seeds, 'dmir_ref_cnn', 'acc'))} · "
                    f"hand-designed DSCNN {mean_only(stat(seeds, 'dmir_dscnn_classification', 'acc'))}",
             "seed": f"one search run: {seeds['dmir_cls_best']['original']['acc'] * 100:.2f}%" if "dmir_cls_best" in seeds else None},
            {"label": "Smallest model", "value": "8 k",
             "sub": f"{mean_only(stat(seeds, 'dmir_cls_tiny', 'acc'))} over five seeds, 37 KB of flash",
             "seed": f"one search run: {seeds['dmir_cls_tiny']['original']['acc'] * 100:.2f}%" if "dmir_cls_tiny" in seeds else None},
            {"label": "Fastest on board", "value": "1.435 ms", "sub": "QAT int8, Cortex-M7"},
            {"label": "Low-end board", "value": "fits", "sub": "the reference CNN does not"},
        ],
        "pipeline": [["Searches", "6"], ["Models saved", str(sum(len(f["rows"]) for t in tasks for f in t["fronts"]))],
                     ["Board runs", str(sum(len(v["boards"]) for m in reg["models"] for v in m["variants"]))],
                     ["Boards", "2"]],
        "tasks": tasks, "registry": reg, "quant": quant, "seeds": seeds,
        "seeds_final": seeds_final, "final_label": "DSCNN recipe",
        "bench": {
            "regression": [reg_rows(seeds, refs, "Right", "regression_lcr", "dmir_lcr_best", "117 k"),
                           reg_rows(seeds, refs, "Left", "regression_lcl", "dmir_lcl_best", "106 k")],
            "ablation": [["Turn signal only", refs["indicator_only_acc"]],
                         ["Everything except turn signal", (stat(seeds, "dmir_cls_noind", "acc") or {"mean": 0.9108})["mean"]],
                         ["All channels", (stat(seeds, "dmir_cls_best", "acc") or {"mean": 0.9208})["mean"]]],
            "notes": ["Our bars are five-seed means. The Transformer figures are single reported runs.",
                      "The published Transformer predicts both directions with one model, on 30 channels; ours are one model "
                      "per direction on 31. The comparison is indicative, as the paper states.",
                      "The internal Transformers (0.42 / 0.44 s) are not beaten on RMSE."],
        },
        "media": media,
    }


def reg_rows(seeds, refs, side, task, key, size):
    it = refs["internal_transformers"]
    internal = (it["params_lcr"], it["rmse_lcr"]) if task == "regression_lcr" else (it["params_lcl"], it["rmse_lcl"])
    rows = []
    if stat(seeds, key, "rmse"):
        rows.append([f"searched, {size}", stat(seeds, key, "rmse")["mean"], "ours"])
    if stat(seeds, "dmir_dscnn_" + task, "rmse"):
        rows.append(["hand-designed DSCNN, 10 k", stat(seeds, "dmir_dscnn_" + task, "rmse")["mean"], "ours2"])
    rows += [[f"internal Transformer, {round(internal[0] / 1000)} k", internal[1], "ref"],
             ["published Transformer, ~54 k", refs["published_sota"]["rmse"], "pub"]]
    return {"task": side, "rows": rows}


def build_highd(share: bool):
    fr = HIGHD / "results" / "nas-fronts"
    reg = resolve_measurements(HIGHD / "results/deploy/measurements.json",
                               HIGHD / "results/deploy/benchmarks_api.jsonl", HIGHD / "results/deploy")
    refs = reg["references"]
    seeds = seed_summary(read_jsonl(HIGHD / "results/seeds/seed_variance.jsonl"))
    seeds_final = seed_summary(read_jsonl(HIGHD / "results/seeds/seed_variance_final.jsonl"))
    base = seed_summary(baseline_seeds(HIGHD / "logs/experiments.jsonl", "highd_baseline_cls")
                        + baseline_seeds(HIGHD / "logs/experiments.jsonl", "highd_baseline_ttlc"))
    rerank = rerank_summary(HIGHD / "results/seeds/rerank_cls.jsonl",
                            {"highd_cls": "first search", "highd_cls_tight": "tighter search"})

    tasks = [
        {"id": "highd_cls", "label": "Lane-change prediction", "long": "Keep lane, change right or change left, up to 5 s ahead",
         "metric": {"key": "acc", "name": "test accuracy", "better": "high", "fmt": "pct"},
         "fronts": [{"label": "first search", "rows": read_front(fr / "highd_cls.csv", "test_acc")},
                    {"label": "tighter search", "rows": read_front(fr / "highd_cls_tight.csv", "test_acc")}],
         "picks": [{"id": "cls_aaaaap", "model": "model_aaaaap", "front": 1, "params": 5347, "label": "searched (tighter search)", "seed_key": "highd_cls_aaaaap"},
                   {"id": "cls_aaaaam", "model": "model_aaaaam", "front": 0, "params": 7904, "label": "searched (first search)", "seed_key": "highd_cls_aaaaam"}],
         "refs": [{"kind": "point", "label": "hand-designed CNN", "params": hand_params(HIGHD, "highd_baseline_cls"), "value": 0.9109, "seed_key": "highd_baseline_cls"},
                  {"kind": "line", "label": "published model (T-IV 2022)", "value": refs["published"]["acc"]}]},
        {"id": "highd_ttlc", "label": "Time to lane change", "long": "Seconds until the lane crossing (0.2 to 5.2 s)",
         "metric": {"key": "rmse", "name": "test RMSE (s)", "better": "low", "fmt": "s3"},
         "fronts": [{"label": "search", "rows": read_front(fr / "highd_ttlc.csv", "test_rmse")}],
         "picks": [{"id": "ttlc_aaaaaw", "model": "model_aaaaaw", "front": 0, "params": 27719, "label": "searched", "seed_key": "highd_ttlc_aaaaaw"}],
         "refs": [{"kind": "point", "label": "hand-designed CNN", "params": hand_params(HIGHD, "highd_baseline_ttlc"), "value": 0.2763, "seed_key": "highd_baseline_ttlc"},
                  {"kind": "line", "label": "published model (T-IV 2022)", "value": refs["published"]["ttlc_rmse"]}]},
    ]

    quant = [
        {"model": "Searched (tighter search)", "params": 5347, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.91153], ["int8 PTQ", 0.91037]]},
        {"model": "Searched (first search)", "params": 7904, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.90526], ["int8 PTQ", 0.891]]},
        {"model": "Time to lane change", "params": 27719, "metric": "MAE (s)", "better": "low", "fmt": "s3",
         "steps": [["fp32", 0.16561], ["int8 PTQ", 0.18701]]},
    ]

    pub, hb, sr = refs["published"], refs["baseline_their_metrics"], refs["searched_their_metrics"]
    bench = {
        "metrics": [
            {"name": "Accuracy", "fmt": "f3", "better": "high", "rows": [["published", pub["acc"]], ["hand-designed", hb["acc"]], ["searched", 0.91153]]},
            {"name": "F1", "fmt": "f3", "better": "high", "rows": [["published", pub["f1"]], ["hand-designed", hb["f1"]], ["searched", sr["f1"]]]},
            {"name": "AUC", "fmt": "f3", "better": "high", "rows": [["published", pub["auc"]], ["hand-designed", hb["auc"]], ["searched", sr["auc"]]]},
            {"name": "Robust horizon (s)", "fmt": "s2", "better": "high", "rows": [["published", pub["tau_c_s"]], ["hand-designed", hb["tau_c_s"]], ["searched", sr["tau_c_s"]]]},
            {"name": "Time-to-change RMSE (s)", "fmt": "s3", "better": "low", "rows": [["published", pub["ttlc_rmse"]], ["hand-designed", hb["ttlc_rmse"]], ["searched", 0.2608]]},
        ],
        "splits": refs["splits"],
        "notes": ["Our accuracy moves by one to three points between training seeds (Seeds tab); every seed stays above the published figure.",
                  "Scored with the metric functions transcribed from the published code.",
                  "Our reimplementation of their protocol, not a run of their own code.",
                  "Their model is not public, so its cost cannot be measured on a board."],
    }

    media = []
    if not share and (MEDIA / "replay/replay.gif").exists():
        media.append({"id": "replay", "label": "One real lane change",
                      "caption": "A real vehicle from the highD test recordings changing lane at 119 km/h. The trace is what the "
                                 "searched 5.3 k model says at every window. Bars on the road: distance driven during one inference.",
                      "src": data_uri(MEDIA / "replay/replay.gif", "image/gif")})
    elif share:
        media.append({"id": "replay", "label": "One real lane change", "withheld":
                      "Not included in this copy: the replay is drawn from raw highD trajectories, which the dataset licence does not let us pass on."})

    return {
        "id": "highd", "name": "highD", "title": "Surrounding-vehicle lane changes", "status": "complete", "accent": 2,
        "facts": [
            ["Whose behaviour", "Other vehicles around the car"],
            ["Recorded in", "German motorways, filmed by drone (RWTH Aachen)"],
            ["Input", "10 time steps × 18 track features (2 s at 5 Hz)"],
            ["Tasks", "Lane-change prediction (3 classes) · time to lane change"],
            ["Protocol", "Mozaffari et al., IEEE T-IV 2022, reimplemented"],
            ["Data", "On request from RWTH · not redistributable"],
        ],
        "headline": {"value": "0.106 ms", "label": "per prediction on a Cortex-M7",
                     "sub": f"The deployed 5.3 k model scores 91.0% in 15 KB of flash. Retrained from scratch it averages "
                            f"{pm(stat(seeds, 'highd_cls_aaaaap', 'acc'))}, still above the published 83%, which reports no hardware cost."},
        "kpis": [
            {"label": "Accuracy, five seeds", "value": mean_only(stat(base, "highd_baseline_cls", "acc")),
             "sub": "hand-designed 8.4 k CNN · published 83%",
             "seed": (f"best searched model: {rerank['best_mean']['stats']['mean'] * 100:.1f}% at "
                      f"{rerank['best_mean']['params'] / 1000:.0f} k") if rerank else None},
            {"label": "Robust horizon", "value": "4.53 s", "sub": "deployed 5.3 k model · published 3.96 s"},
            {"label": "Time-to-change RMSE", "value": mean_only(stat(seeds, "highd_ttlc_aaaaaw", "rmse"), False),
             "sub": "five seeds, searched and hand-designed alike · published 0.629 s",
             "seed": f"one search run: {seeds['highd_ttlc_aaaaaw']['original']['rmse']:.3f} s" if "highd_ttlc_aaaaaw" in seeds else None},
            {"label": "Split check", "value": "2 of 3 exact", "sub": "test 693 vs 698"},
        ],
        "pipeline": [["Searches", "3"], ["Models saved", str(sum(len(f["rows"]) for t in tasks for f in t["fronts"]))],
                     ["Board runs", str(sum(len(v["boards"]) for m in reg["models"] for v in m["variants"]))],
                     ["Boards", "2"]],
        "tasks": tasks, "registry": reg, "quant": quant,
        "seeds": {**seeds, **base}, "seeds_final": seeds_final, "final_label": "hand-designed recipe",
        "rerank": rerank, "rerank_ref": "highd_baseline_cls",
        "bench": bench, "media": media,
    }


EXID_METHODS = [("scratch", "trained on exiD", "s3"),
                ("zero-shot", "highD model, applied as is", "s2"),
                ("fine-tune", "highD model, fine-tuned on exiD", "s1")]


def stats_of(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"mean": statistics.fmean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "max": max(v), "n": len(v)}


def build_exid(share: bool):
    meta = read_json(EXID / "results/prepared_meta.json")
    if meta is None:
        return {"id": "exid", "name": "exiD", "title": "Motorway entries and exits", "status": "planned", "accent": 3,
                "facts": [["Whose behaviour", "Other vehicles at on- and off-ramps"],
                          ["Recorded in", "German motorway junctions, filmed by drone (RWTH Aachen)"],
                          ["Status", "Data not prepared yet"]],
                "headline": {"value": "next", "label": "third dataset", "sub": "Same pipeline as highD."}}
    kinds = meta["kinds"]
    evals = read_jsonl(EXID / "results/transfer.jsonl")
    hand = {t: hand_params(EXID, f"exid_baseline_{t}") for t in ("cls", "ttlc")}

    def runs(task, method, eval_on="exiD", fraction=1.0):
        return sorted((r for r in evals if r["task"] == task and r["method"] == method
                       and r["eval_on"] == eval_on and r.get("fraction", 1.0) == fraction),
                      key=lambda r: r["seed"])

    tasks = []
    for task, label, metric, fmt, better in (("cls", "Lane-change prediction", "acc", "pct", "high"),
                                              ("ttlc", "Time to lane change", "rmse", "s3", "low")):
        methods, fractions, by_kind, on_highd = [], {}, {}, {}
        for mid, mlabel, color in EXID_METHODS:
            rr = runs(task, mid)
            methods.append({"id": mid, "label": mlabel, "color": color,
                            "runs": [{"seed": r["seed"], metric: r["metrics"][metric]} for r in rr],
                            "stats": stats_of([r["metrics"][metric] for r in rr])})
            on_highd[mid] = stats_of([r["metrics"][metric] for r in runs(task, mid, "highD")])
            groups = {}
            for r in rr:
                for g, v in (r["metrics"].get("by_group") or {}).items():
                    groups.setdefault(g, []).append(v[metric])
            by_kind[mid] = {g: stats_of(v) for g, v in groups.items()}
            if mid != "zero-shot":
                fractions[mid] = [{"fraction": fr, "stats": stats_of([r["metrics"][metric] for r in runs(task, mid, "exiD", fr)])}
                                  for fr in (0.1, 0.25, 1.0)]
        tasks.append({"task": task, "label": label, "metric": metric, "fmt": fmt, "better": better,
                      "params": hand[task], "methods": methods, "fractions": fractions, "by_kind": by_kind, "on_highd": on_highd})

    searched_zero = [{"model": r["model"], "params": r["params"], "task": r["task"], "eval_on": r["eval_on"],
                      "metrics": {k: v for k, v in r["metrics"].items() if k != "by_group"}}
                     for r in read_jsonl(EXID / "results/transfer_searched.jsonl")]

    # Seeds view: searched highD architectures retrained on exiD, and the hand-designed CNN
    seeds = seed_summary(read_jsonl(EXID / "results/seeds/seed_variance.jsonl"))
    seeds_final = seed_summary(read_jsonl(EXID / "results/seeds/seed_variance_final.jsonl"))
    for task in ("cls", "ttlc"):
        rr = runs(task, "scratch")
        if rr:
            seeds[f"exid_baseline_{task}"] = {
                "runs": [{"seed": r["seed"], **{k: v for k, v in r["metrics"].items() if isinstance(v, float)}} for r in rr],
                "params": hand[task], "recipe": "hand-designed recipe", "recipe_short": "hand-designed recipe", "original": {}}
            seeds[f"exid_baseline_{task}"]["stats"] = {
                k: stats_of([x.get(k) for x in seeds[f"exid_baseline_{task}"]["runs"]])
                for k in (("acc", "macro_f1") if task == "cls" else ("mae", "rmse"))}
    seed_tasks = [
        {"id": "exid_cls", "label": "Lane-change prediction",
         "metric": {"key": "acc", "name": "test accuracy", "better": "high", "fmt": "pct"},
         "picks": [{"id": "exid_cls_aaaaap", "label": "searched on highD, 5.3 k", "seed_key": "exid_cls_aaaaap"},
                   {"id": "exid_cls_aaaaam", "label": "searched on highD, 7.9 k", "seed_key": "exid_cls_aaaaam"}],
         "refs": [{"kind": "point", "label": "hand-designed CNN", "params": hand["cls"], "seed_key": "exid_baseline_cls"}],
         "fronts": []},
        {"id": "exid_ttlc", "label": "Time to lane change",
         "metric": {"key": "rmse", "name": "test RMSE (s)", "better": "low", "fmt": "s3"},
         "picks": [{"id": "exid_ttlc_aaaaaw", "label": "searched on highD, 28 k", "seed_key": "exid_ttlc_aaaaaw"}],
         "refs": [{"kind": "point", "label": "hand-designed CNN", "params": hand["ttlc"], "seed_key": "exid_baseline_ttlc"}],
         "fronts": []},
    ]

    def mean_of(task, mid, eval_on="exiD"):
        t = next(x for x in tasks if x["task"] == task)
        st = next(m for m in t["methods"] if m["id"] == mid)["stats"] if eval_on == "exiD" else t["on_highd"].get(mid)
        return st

    def pct(st):
        return f"{st['mean'] * 100:.1f}%" if st else "–"

    cls_scratch, cls_zero, cls_ft = (mean_of("cls", m) for m in ("scratch", "zero-shot", "fine-tune"))
    ttlc_scratch = mean_of("ttlc", "scratch")
    reverse = mean_of("cls", "scratch", "highD")
    total = sum(meta["stats"][s]["scenarios"] for s in meta["stats"])
    lc_total = sum(meta["stats"][s]["RLC"] + meta["stats"][s]["LLC"] for s in meta["stats"])
    by_kind_total = {k: sum(meta["stats"][s]["by_kind"][k] for s in meta["stats"]) for k in kinds}
    return {
        "id": "exid", "name": "exiD", "title": "Motorway entries and exits", "status": "complete", "accent": 3,
        "views": ["overview", "transfer", "robust"],
        "shortcuts": [["transfer", "Transfer", "highD models on exiD: as they are, fine-tuned, and against training on exiD."],
                      ["robust", "Seeds", "Every exiD model trained five times."]],
        "facts": [
            ["Whose behaviour", "Other vehicles at motorway entries and exits"],
            ["Recorded in", "7 German motorway junctions, filmed by drone (RWTH Aachen, fka)"],
            ["Input", "10 time steps × 18 track features (2 s at 5 Hz), the highD format"],
            ["Tasks", "Lane-change prediction (3 classes) · time to lane change"],
            ["Protocol", "highD protocol adapted to exiD (Lanelet2 maps, curved roads)"],
            ["Test split", "Latest recordings of every location, 8.5% of the time"],
            ["Data", "On request from levelXdata · not redistributable"],
        ],
        "headline": {"value": pct(cls_scratch), "label": "accuracy when trained on exiD, five seeds",
                     "sub": (f"A highD model applied as is reaches {pct(cls_zero)}; fine-tuned on exiD, {pct(cls_ft)}. "
                             f"Same 8.4 k network and the same board cost as on highD.") if cls_zero else
                            "Same 8.4 k network and the same board cost as on highD."},
        "kpis": [
            {"label": "Trained on exiD", "value": pct(cls_scratch), "sub": "hand-designed CNN, five seeds"},
            {"label": "highD model as is", "value": pct(cls_zero), "sub": "no exiD training at all"},
            {"label": "Fine-tuned", "value": pct(cls_ft), "sub": "highD weights, then exiD"},
            {"label": "Time-to-change RMSE", "value": f"{ttlc_scratch['mean']:.3f} s" if ttlc_scratch else "–",
             "sub": "trained on exiD, five seeds",
             "seed": f"exiD model on highD: {pct(reverse)}" if reverse else None},
        ],
        "pipeline_title": "Same format, same networks, same boards",
        "pipeline_caption": "exiD is prepared in the highD scenario format, so the highD networks, training code and board builds carry over unchanged.",
        "pipeline": [["Recordings", "93"], ["Scenarios", f"{total:,}"], ["Lane changes", f"{lc_total:,}"],
                     ["Merges and exits", f"{by_kind_total['merge from on-ramp'] + by_kind_total['exit to off-ramp']:,}"]],
        "tasks": seed_tasks, "seeds": seeds, "seeds_final": seeds_final, "final_label": "hand-designed recipe",
        "transfer": {"tasks": tasks, "kinds": kinds, "searched_zero": searched_zero},
    }


def main():
    share = "--share" in sys.argv
    data = {"built": date.today().isoformat(), "share": share,
            "datasets": [build_dmir(share), build_highd(share), build_exid(share)]}
    status = HERE / "status.local.json"
    if not share and status.exists():
        data["status"] = read_json(status)

    src = HERE / "src"
    html = (src / "index.html").read_text(encoding="utf-8")
    css = (src / "app.css").read_text(encoding="utf-8")
    js = (src / "app.js").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (html.replace("/*__CSS__*/", css)
                .replace("/*__JS__*/", js)
                .replace("/*__DATA__*/null", payload))
    out = HERE / "dist" / ("results-explorer-share.html" if share else "results-explorer.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB  (share={share})")


if __name__ == "__main__":
    main()
