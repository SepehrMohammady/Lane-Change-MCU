"""Build the results explorer: one self-contained HTML page for every dataset.

Reads the project's own result files (nothing is typed in by hand here), resolves
board measurements against the raw ST API records, and writes a single HTML file
with the data, the app and the media inlined, so it opens offline and can be sent
as one attachment.

    python dashboard/build.py            -> dashboard/dist/results-explorer.html
    python dashboard/build.py --lab      -> dashboard/dist/results-explorer-lab.html
    python dashboard/build.py --share    -> dashboard/dist/results-explorer-share.html

The lab build, for colleagues in the lab, leaves out the project-status page
(internal to-dos). The share build, for anyone outside the lab, also leaves out
every picture drawn from raw highD or exiD trajectories, which the dataset
licences do not allow us to pass on. Standard library only.
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
MEDIA_NEW = MEDIA / "2026-10-09"          # renders for the 9 October talk (labels: ours, sizes)


def media_file(rel: str) -> Path:
    """Newest render of a replay when it exists, else the September one."""
    p = MEDIA_NEW / rel
    return p if p.exists() else MEDIA / rel


# Builds measured before the API log existed carry no file name in the registry; these are their .tflite
# names in the own-board (unas/local_bench.py) and energy (unas/energy_session.py) records.
BUILD_FILES = {
    ("dmir", "ref_cnn", "fp32", "float32"): "REF_cnn_multi_float32.tflite",
    ("dmir", "cls_best", "fp32", "float32"): "cls_best_float32.tflite",
    ("dmir", "cls_best", "int8 QAT", "int8"): "cls_best_qat_int8_io.tflite",
    ("dmir", "cls_tiny", "fp32", "float32"): "cls_tiny_float32.tflite",
    ("highd", "cls_aaaaam", "fp32", "float32"): "highd_cls_model_aaaaam_f32.tflite",
    ("highd", "cls_aaaaam", "int8 PTQ", "int8"): "highd_cls_model_aaaaam_int8_io.tflite",
}


# ------------------------------------------------------------------ readers
def read_json(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def hand_params(root: Path, prefix: str):
    """Parameter count of the hand-designed CNN, from the first logged run with this prefix."""
    for r in read_jsonl(root / "logs/experiments.jsonl"):
        if r.get("run_name", "").startswith(prefix) and r.get("config", {}).get("n_params"):
            return r["config"]["n_params"]
    return None


def permuted(p: Path) -> Path:
    """The re-run with the training windows shuffled (2026-09-23) when it exists.

    Keras runs on highD and exiD made before that date trained on the stored scenario
    order (found 2026-09-23); their re-runs live next to them with "_permuted"
    in the name and replace them here."""
    q = p.with_name(p.stem + "_permuted" + p.suffix)
    return q if q.exists() else p


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


def resolve_measurements(reg_path: Path, api_path: Path, svg_dir: Path, ds_id: str | None = None):
    """Replace {"api": artifact} references with the raw API record's numbers. With a dataset id (the
    explorer), also add what our own boards measured for the same file: latency (200 timed runs) and
    energy per inference; paper/figures/make_figures.py calls it without one and gets the cloud numbers."""
    reg = read_json(reg_path)
    api = {(r["model"], r["board"]): r for r in read_jsonl(api_path)}
    own = {(r["model"], r["board"]): r["duration_ms"] for r in read_jsonl(svg_dir / "local_bench.jsonl")}
    energy = {(r["model"], r["board"]): r for r in read_jsonl(svg_dir / "energy_local.jsonl") if "energy_mcu_mJ" in r}
    for m in reg["models"]:
        svg = m.get("arch_svg")
        if svg and (svg_dir / svg).exists():
            m["arch_svg_markup"] = (svg_dir / svg).read_text(encoding="utf-8")
        for v in m["variants"]:
            names = {b["api"] for b in v["boards"].values() if "api" in b}
            file = names.pop() if names else BUILD_FILES.get((ds_id, m["id"], v["precision"], v["io"]))
            for board, b in list(v["boards"].items()):
                if "api" in b:
                    r = api.get((b["api"], board))
                    if r is None or r.get("duration_ms") is None:
                        v["boards"][board] = {"fits": False, "why": "no measurement returned by the board farm"}
                    else:
                        v["boards"][board] = {"latency_ms": r["duration_ms"], "macc": r.get("macc"),
                                              "flash_b": r.get("rom_B"), "ram_b": r.get("ram_B"), "api": True}
            if ds_id is None:
                continue
            for board in ("STM32H7B3I-DK", "NUCLEO-F401RE"):
                e = energy.get((file, board))
                own_ms = own.get((file, board), e["latency_ms"] if e else None)
                if own_ms is None:
                    continue
                b = v["boards"].setdefault(board, {})
                b["own_ms"] = own_ms
                if e:
                    b["energy_mj"] = e["energy_mcu_mJ"]
                    b["power_mw"] = (e["p_run_W"] - e["p_sleep_W"]) * 1e3
    return reg


def own_counts(deploy: Path):
    """Own-board latency runs and energy readings of one dataset."""
    return (len(read_jsonl(deploy / "local_bench.jsonl")),
            sum(1 for r in read_jsonl(deploy / "energy_local.jsonl") if "energy_mcu_mJ" in r))


def kept_candidates(path: Path, key: str):
    """A later search (second, third, final) kept its 12 best candidates and retrained each five times with
    the search recipe (unas/select_by_seeds.py); the one with the best validation mean was deployed.
    Returns the candidates as a front (test metric averaged over the runs), the deployed model's name, and
    its runs as seed rows under `key`."""
    rows = read_jsonl(path)
    if not rows:
        return [], None, []
    metric = "acc" if "acc" in rows[0]["test"] else "rmse"
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    sign = 1 if metric == "acc" else -1
    chosen = max(by, key=lambda mdl: sign * statistics.fmean(r["val"][metric] for r in by[mdl]))
    front = [{"model": mdl, "params": rr[0]["params"], "kb": rr[0]["ms"] / 1024,
              "value": statistics.fmean(r["test"][metric] for r in rr), "runs": len(rr)} for mdl, rr in by.items()]
    seeds = [{"key": key, "seed": r["seed"], "params": r["params"], "recipe": r["recipe"], "test": r["test"],
              "epochs_run": r.get("epochs_run")} for r in by[chosen]]
    return front, chosen, seeds


def cloud_runs(reg):
    return sum(1 for m in reg["models"] for v in m["variants"] for b in v["boards"].values() if b.get("latency_ms") is not None)


def relabel(reg, labels):
    for m in reg["models"]:
        if m["id"] in labels:
            m["label"] = labels[m["id"]]


def kparams(n):
    return f"{round(n / 1000)} k" if n >= 1e4 else f"{n / 1000:.1f} k"


def quant_from_registry(reg, short):
    """Float32 against int8 for the registry models named in `short`: accuracy for classifiers, MAE for
    regressors, the int8 build with int8 input and output where there is one."""
    out = []
    for m in reg["models"]:
        if m["id"] not in short:
            continue
        steps, metric = [], None
        for prec in ("fp32", "int8 PTQ", "int8 QAT"):
            vs = [v for v in m["variants"] if v["precision"] == prec]
            v = next((x for x in vs if x["io"] == "int8"), vs[0] if vs else None)
            if v is None:
                continue
            key = "acc" if "acc" in v["metric"] else "mae" if "mae" in v["metric"] else None
            if key is None or (metric and key != metric):
                continue
            metric = key
            steps.append([prec, v["metric"][key]])
        if len(steps) >= 2:
            out.append({"model": short[m["id"]], "params": m["params"], "metric": "acc" if metric == "acc" else "MAE (s)",
                        "better": "high" if metric == "acc" else "low", "fmt": "pct" if metric == "acc" else "s3",
                        "steps": steps})
    return out


def fastest(reg, ids, board="STM32H7B3I-DK"):
    """Fastest cloud-measured build among the given models: (latency, model, variant)."""
    best = None
    for m in reg["models"]:
        if m["id"] not in ids:
            continue
        for v in m["variants"]:
            lat = v["boards"].get(board, {}).get("latency_ms")
            if lat is not None and (best is None or lat < best[0]):
                best = (lat, m, v)
    return best


def lowest_energy(reg, ids, board="STM32H7B3I-DK"):
    """Build with the least energy per inference among the given models: (mJ, model, variant)."""
    best = None
    for m in reg["models"]:
        if m["id"] not in ids:
            continue
        for v in m["variants"]:
            e = v["boards"].get(board, {}).get("energy_mj")
            if e is not None and (best is None or e < best[0]):
                best = (e, m, v)
    return best


def energy_of(reg, model_id, precision="fp32", io="float32", board="STM32H7B3I-DK"):
    for m in reg["models"]:
        if m["id"] == model_id:
            for v in m["variants"]:
                if v["precision"] == precision and v["io"] == io:
                    return v["boards"].get(board, {}).get("energy_mj")
    return None


def build_name(v):
    return v["precision"].replace("fp32", "float32") + (", int8 I/O" if v["io"] == "int8" and v["precision"] != "fp32" else "")


def mj(v):
    return f"{v:.3f} mJ" if v < 0.1 else f"{v:.2f} mJ" if v < 10 else f"{v:.1f} mJ"


def ms(v):
    return f"{v:.3f} ms" if v < 1 else f"{v:.2f} ms" if v < 10 else f"{v:.1f} ms"


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
    """Cloud latency of one build; for int8, the deployed build with int8 input and output when there is one."""
    for m in reg["models"]:
        if m["id"] == model_id:
            vs = [v for v in m["variants"] if v["precision"] == precision and v["boards"].get(board, {}).get("latency_ms") is not None]
            vs.sort(key=lambda v: v["io"] != "int8")
            if vs:
                return vs[0]["boards"][board]["latency_ms"]
    return None


# ------------------------------------------------------------------ datasets
def build_dmir(share: bool):
    fr = DMIR / "results" / "nas-fronts"
    reg = resolve_measurements(DMIR / "results/deploy/measurements.json",
                               DMIR / "results/deploy/benchmarks_api.jsonl", DMIR / "results/deploy", "dmir")
    relabel(reg, {"cls_best": "Searched, first search, best run", "cls_tiny": "Searched, first search, smallest",
                  "lcr_best": "Searched, first search, right", "lcl_best": "Searched, first search, left",
                  "cls_v4": "Searched, final search", "lcr_v4": "Searched, final search, right",
                  "lcl_v4": "Searched, final search, left"})
    refs = reg["references"]
    dscnn_seeded, dscnn_first = dscnn_runs(DMIR / "logs/experiments.jsonl")
    kept = {t: kept_candidates(DMIR / f"results/seeds/select_dmir_{t}_v4.jsonl", f"dmir_{t}_v4_best") for t in ("cls", "lcr", "lcl")}
    seeds = seed_summary(read_jsonl(DMIR / "results/seeds/seed_variance.jsonl") + dscnn_seeded
                         + [r for front, chosen, rows in kept.values() for r in rows])
    seeds_final = seed_summary(read_jsonl(DMIR / "results/seeds/seed_variance_dscnn.jsonl"))
    final_front = lambda t: {"label": "final search: 12 kept, mean of five runs", "final": True, "rows": kept[t][0]}

    def dscnn_ref(task, metric):
        if task not in dscnn_first:
            return None
        return {"kind": "point", "label": "hand-designed DSCNN", "params": DSCNN_PARAMS[task],
                "value": dscnn_first[task][metric], "seed_key": "dmir_dscnn_" + task}

    tasks = [
        {"id": "classification", "label": "Intention", "long": "Lane-change intention: none, right or left",
         "metric": {"key": "acc", "name": "test accuracy", "better": "high", "fmt": "pct"},
         "fronts": [{"label": "first search", "rows": read_front(fr / "dmir_cls.csv", "test_acc")}, final_front("cls")],
         "picks": [{"id": "cls_best", "model": "model_aaaabl", "front": 0, "params": 83803, "label": "first search, 84 k", "seed_key": "dmir_cls_best"},
                   {"id": "cls_tiny", "model": "model_aaaaat", "front": 0, "params": 7953, "label": "first search, 8.0 k", "seed_key": "dmir_cls_tiny"},
                   {"id": "cls_v4", "model": kept["cls"][1], "front": 1, "params": 6528, "label": "final search, 6.5 k", "seed_key": "dmir_cls_v4_best"}],
         "refs": [{"kind": "point", "label": "reference CNN", "params": 441347, "value": 0.9169, "seed_key": "dmir_ref_cnn"},
                  dscnn_ref("classification", "acc")]},
        {"id": "classification_noind", "label": "Intention, no turn signal",
         "long": "Same task with the two turn-signal channels removed",
         "metric": {"key": "acc", "name": "test accuracy", "better": "high", "fmt": "pct"},
         "fronts": [{"label": "first search", "rows": read_front(fr / "dmir_cls_noind.csv", "test_acc")}],
         "picks": [{"id": "cls_noind", "model": "model_aaaaah", "front": 0, "params": 21038, "label": "searched, no turn signal", "seed_key": "dmir_cls_noind"}],
         "refs": [{"kind": "line", "label": "turn signal alone", "value": refs["indicator_only_acc"]},
                  {"kind": "line", "label": "all channels, best", "value": 0.9208}]},
        {"id": "regression_lcr", "label": "Time to right change",
         "long": "Seconds until a right lane change (0 to 4 s)",
         "metric": {"key": "rmse", "name": "test RMSE (s)", "better": "low", "fmt": "s3"},
         "fronts": [{"label": "first search, MAE objective", "rows": read_front(fr / "dmir_lcr.csv", "test_rmse")},
                    {"label": "first search, RMSE objective", "rows": read_front(fr / "dmir_lcr_rmse.csv", "test_rmse")},
                    final_front("lcr")],
         "picks": [{"id": "lcr_best", "model": "model_aaaaan", "front": 0, "params": 117404, "label": "first search, 117 k", "seed_key": "dmir_lcr_best"},
                   {"id": "lcr_v4", "model": kept["lcr"][1], "front": 2, "params": 4663, "label": "final search, 4.7 k", "seed_key": "dmir_lcr_v4_best"}],
         "refs": [{"kind": "point", "label": "internal Transformer (reported)", "params": 333505, "value": 0.42},
                  dscnn_ref("regression_lcr", "rmse"),
                  {"kind": "line", "label": "published Transformer (both directions)", "value": 0.5102}]},
        {"id": "regression_lcl", "label": "Time to left change",
         "long": "Seconds until a left lane change (0 to 4 s)",
         "metric": {"key": "rmse", "name": "test RMSE (s)", "better": "low", "fmt": "s3"},
         "fronts": [{"label": "first search, MAE objective", "rows": read_front(fr / "dmir_lcl.csv", "test_rmse")},
                    {"label": "first search, RMSE objective", "rows": read_front(fr / "dmir_lcl_rmse.csv", "test_rmse")},
                    final_front("lcl")],
         "picks": [{"id": "lcl_best", "model": "model_aaaabu", "front": 1, "params": 105769, "label": "first search, 106 k", "seed_key": "dmir_lcl_best"},
                   {"id": "lcl_v4", "model": kept["lcl"][1], "front": 2, "params": 3652, "label": "final search, 3.7 k", "seed_key": "dmir_lcl_v4_best"}],
         "refs": [{"kind": "point", "label": "internal Transformer (reported)", "params": 49089, "value": 0.44},
                  dscnn_ref("regression_lcl", "rmse"),
                  {"kind": "line", "label": "published Transformer (both directions)", "value": 0.5102}]},
    ]
    for t in tasks:
        t["refs"] = [r for r in t["refs"] if r]

    quant = [
        {"model": "Reference CNN", "params": 441347, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.9169], ["int8 PTQ", 0.8827]]},
        {"model": "First search, best run", "params": 83803, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.9208], ["int8 PTQ", 0.8686], ["int8 QAT", 0.8990]]},
        {"model": "First search, smallest", "params": 7953, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.9130], ["int8 PTQ", 0.8546]]},
        {"model": "Searched, no turn signal", "params": 21038, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.9108], ["int8 PTQ", 0.7606]]},
        {"model": "First search, right", "params": 117404, "metric": "MAE (s)", "better": "low", "fmt": "s3",
         "steps": [["fp32", 0.2865], ["int8 PTQ", 0.4485], ["int8 QAT", 0.4313]]},
        {"model": "First search, left", "params": 105769, "metric": "MAE (s)", "better": "low", "fmt": "s3",
         "steps": [["fp32", 0.3165], ["int8 PTQ", 0.3440], ["int8 QAT", 0.3620]]},
    ] + quant_from_registry(reg, {"dscnn_cls": "Hand-designed DSCNN", "cls_v4": "Final search",
                                  "dscnn_lcr": "Hand-designed, right", "lcr_v4": "Final search, right",
                                  "lcl_v4": "Final search, left"})

    noind = seeds.get("dmir_cls_noind")
    if noind is not None and not noind["original"]:                 # its single run is the fp32 quant step
        noind["original"] = {"acc": next(q["steps"][0][1] for q in quant if q["model"] == "Searched, no turn signal")}

    media = []
    if media_file("race/race.gif").exists():
        media.append({"id": "race", "label": "Five models, one lane change",
                      "caption": "A held-out test driver's recorded left lane change, replayed for every model on the same "
                                 "5-second windows. The bar under each road is the distance driven while the model computes "
                                 "one answer, from latency measured on the board.",
                      "src": data_uri(media_file("race/race.gif"), "image/gif")})
    if media_file("race/race_whatif.gif").exists():
        media.append({"id": "whatif", "label": "What a late warning costs",
                      "caption": "Same recording. The recorded driver waited for the car in the target lane; nothing happened. "
                                 "The overlay adds one driver who does not check the mirror and asks whose warning reaches "
                                 "him in time. The reaction time (1.0 s) is an assumption, and the outcome depends on it.",
                      "src": data_uri(media_file("race/race_whatif.gif"), "image/gif")})

    cls_ids = ("ref_cnn", "cls_best", "cls_tiny", "dscnn_cls", "cls_v4")
    seed_of = {"ref_cnn": "dmir_ref_cnn", "cls_best": "dmir_cls_best", "cls_tiny": "dmir_cls_tiny",
               "dscnn_cls": "dmir_dscnn_classification", "cls_v4": "dmir_cls_v4_best"}
    fast, low = fastest(reg, cls_ids), lowest_energy(reg, cls_ids)
    ref_e = energy_of(reg, "ref_cnn")
    n_own, n_energy = own_counts(DMIR / "results/deploy")

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
            {"label": "Fastest on board", "value": ms(fast[0]),
             "sub": f"{kparams(fast[1]['params'])} model, {build_name(fast[2])}, Cortex-M7 · "
                    f"{latency(reg, 'ref_cnn') / fast[0]:.0f}× faster than the reference CNN",
             "seed": f"five seeds: {mean_only(stat(seeds, seed_of[fast[1]['id']], 'acc'))}"},
            {"label": "Energy per prediction", "value": mj(low[0]),
             "sub": f"{kparams(low[1]['params'])} model, {build_name(low[2])}, H7B3I-DK · reference CNN {mj(ref_e)}",
             "seed": "above the core asleep, measured on our board"},
            {"label": "Low-end board", "value": "fits", "sub": "the reference CNN does not"},
        ],
        "pipeline": [["Searches", str(sum(len(t["fronts"]) for t in tasks))],
                     ["Models saved or kept", str(sum(len(f["rows"]) for t in tasks for f in t["fronts"]))],
                     ["Board runs, cloud and own", str(cloud_runs(reg) + n_own)],
                     ["Energy readings", str(n_energy)]],
        "tasks": tasks, "registry": reg, "quant": quant, "seeds": seeds,
        "seeds_final": seeds_final, "final_label": "DSCNN recipe",
        "bench": {
            "regression": [reg_rows(seeds, refs, "Right", "regression_lcr", [("first search, 117 k", "dmir_lcr_best"),
                                                                               ("final search, 4.7 k", "dmir_lcr_v4_best")]),
                           reg_rows(seeds, refs, "Left", "regression_lcl", [("first search, 106 k", "dmir_lcl_best"),
                                                                              ("final search, 3.7 k", "dmir_lcl_v4_best")])],
            "ablation": [["Turn signal only", refs["indicator_only_acc"]],
                         ["Everything except turn signal", (stat(seeds, "dmir_cls_noind", "acc") or {"mean": 0.9108})["mean"]],
                         ["All channels", (stat(seeds, "dmir_cls_best", "acc") or {"mean": 0.9208})["mean"]]],
            "notes": ["Our bars are five-seed means: the searched models with the search recipe, the DSCNN with its own. "
                      "The Transformer figures are single reported runs.",
                      "The published Transformer predicts both directions with one model, on 30 channels; ours are one model "
                      "per direction on 31. The comparison is indicative, as the paper states.",
                      "The internal Transformers' 0.42 and 0.44 s are reported figures. Their graphs, rebuilt with the "
                      "original weights, match the parameter counts but not these errors, so they count for cost only."],
        },
        "media": media,
    }


def reg_rows(seeds, refs, side, task, picks):
    it = refs["internal_transformers"]
    internal = (it["params_lcr"], it["rmse_lcr"]) if task == "regression_lcr" else (it["params_lcl"], it["rmse_lcl"])
    rows = [[label, stat(seeds, key, "rmse")["mean"], "ours"] for label, key in picks if stat(seeds, key, "rmse")]
    if stat(seeds, "dmir_dscnn_" + task, "rmse"):
        rows.append(["hand-designed DSCNN, 10 k", stat(seeds, "dmir_dscnn_" + task, "rmse")["mean"], "ours2"])
    rows += [[f"internal Transformer, {round(internal[0] / 1000)} k, reported", internal[1], "ref"],
             ["published Transformer, ~54 k", refs["published_sota"]["rmse"], "pub"]]
    return {"task": side, "rows": rows}


def build_highd(share: bool):
    fr = HIGHD / "results" / "nas-fronts"
    reg = resolve_measurements(HIGHD / "results/deploy/measurements.json",
                               HIGHD / "results/deploy/benchmarks_api.jsonl", HIGHD / "results/deploy", "highd")
    relabel(reg, {"ttlc_aaaaaw": "Searched regressor (first search)", "cls_v3": "Searched classifier (third search)",
                  "cls_v4": "Searched classifier (final search)", "ttlc_v4": "Searched regressor (final search)"})
    refs = reg["references"]
    sd = HIGHD / "results/seeds"
    kept = {"cls_v2": kept_candidates(permuted(sd / "select_highd_cls_v2.jsonl"), "highd_cls_v2p_best"),
            "cls_v3": kept_candidates(sd / "select_highd_cls_v3.jsonl", "highd_cls_v3_best"),
            "cls_v4": kept_candidates(sd / "select_highd_cls_v4.jsonl", "highd_cls_v4_best"),
            "ttlc_v2": kept_candidates(sd / "select_highd_ttlc_v2_p.jsonl", "highd_ttlc_v2p_best"),
            "ttlc_v4": kept_candidates(sd / "select_highd_ttlc_v4.jsonl", "highd_ttlc_v4_best")}
    seeds = seed_summary(read_jsonl(permuted(HIGHD / "results/seeds/seed_variance.jsonl"))
                         + [r for front, chosen, rows in kept.values() for r in rows])
    seeds_final = seed_summary(read_jsonl(permuted(HIGHD / "results/seeds/seed_variance_final.jsonl")))
    base = seed_summary(baseline_seeds(HIGHD / "logs/experiments.jsonl", "highd_baseline_cls")
                        + baseline_seeds(HIGHD / "logs/experiments.jsonl", "highd_baseline_ttlc"))
    rerank = rerank_summary(permuted(HIGHD / "results/seeds/rerank_cls.jsonl"),
                            {"highd_cls": "first search", "highd_cls_tight": "tighter search"})

    tasks = [
        {"id": "highd_cls", "label": "Lane-change prediction", "long": "Keep lane, change right or change left, up to 5 s ahead",
         "metric": {"key": "acc", "name": "test accuracy", "better": "high", "fmt": "pct"},
         "fronts": [{"label": "first search", "rows": read_front(fr / "highd_cls.csv", "test_acc")},
                    {"label": "tighter search", "rows": read_front(fr / "highd_cls_tight.csv", "test_acc")},
                    {"label": "final search: 12 kept, mean of five runs", "final": True, "rows": kept["cls_v4"][0]}],
         "picks": [{"id": "cls_aaaaam", "model": "model_aaaaam", "front": 0, "params": 7904, "label": "first search, 7.9 k", "seed_key": "highd_cls_aaaaam"},
                   {"id": "cls_aaaaap", "model": "model_aaaaap", "front": 1, "params": 5347, "label": "tighter search, 5.3 k", "seed_key": "highd_cls_aaaaap"},
                   {"id": "cls_v2", "model": kept["cls_v2"][1], "front": None, "params": 24191, "label": "second search, 24 k", "seed_key": "highd_cls_v2p_best"},
                   {"id": "cls_v3", "model": kept["cls_v3"][1], "front": None, "params": 7919, "label": "third search, 7.9 k", "seed_key": "highd_cls_v3_best"},
                   {"id": "cls_v4", "model": kept["cls_v4"][1], "front": 2, "params": 3968, "label": "final search, 4.0 k", "seed_key": "highd_cls_v4_best"}],
         "refs": [{"kind": "point", "label": "hand-designed CNN", "params": hand_params(HIGHD, "highd_baseline_cls"), "value": 0.9109, "seed_key": "highd_baseline_cls"},
                  {"kind": "line", "label": "published model (T-IV 2022)", "value": refs["published"]["acc"]}]},
        {"id": "highd_ttlc", "label": "Time to lane change", "long": "Seconds until the lane crossing (0.2 to 5.2 s)",
         "metric": {"key": "rmse", "name": "test RMSE (s)", "better": "low", "fmt": "s3"},
         "fronts": [{"label": "first search", "rows": read_front(fr / "highd_ttlc.csv", "test_rmse")},
                    {"label": "final search: 12 kept, mean of five runs", "final": True, "rows": kept["ttlc_v4"][0]}],
         "picks": [{"id": "ttlc_aaaaaw", "model": "model_aaaaaw", "front": 0, "params": 27719, "label": "first search, 28 k", "seed_key": "highd_ttlc_aaaaaw"},
                   {"id": "ttlc_v2", "model": kept["ttlc_v2"][1], "front": None, "params": 25883, "label": "second search, 26 k", "seed_key": "highd_ttlc_v2p_best"},
                   {"id": "ttlc_v4", "model": kept["ttlc_v4"][1], "front": 1, "params": 4940, "label": "final search, 4.9 k", "seed_key": "highd_ttlc_v4_best"}],
         "refs": [{"kind": "point", "label": "hand-designed CNN", "params": hand_params(HIGHD, "highd_baseline_ttlc"), "value": 0.2763, "seed_key": "highd_baseline_ttlc"},
                  {"kind": "line", "label": "published model (T-IV 2022)", "value": refs["published"]["ttlc_rmse"]}]},
    ]

    quant = [
        {"model": "Tighter search", "params": 5347, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.91153], ["int8 PTQ", 0.91037]]},
        {"model": "First search", "params": 7904, "metric": "acc", "better": "high", "fmt": "pct",
         "steps": [["fp32", 0.90526], ["int8 PTQ", 0.891]]},
        {"model": "First search, time to change", "params": 27719, "metric": "MAE (s)", "better": "low", "fmt": "s3",
         "steps": [["fp32", 0.16561], ["int8 PTQ", 0.18701]]},
    ] + quant_from_registry(reg, {"baseline_cls": "Hand-designed CNN", "cls_v2": "Second search", "cls_v3": "Third search",
                                  "cls_v4": "Final search", "baseline_ttlc": "Hand-designed, time to change",
                                  "ttlc_v2": "Second search, time to change", "ttlc_v4": "Final search, time to change"})

    pub, hb, sr = refs["published"], refs["baseline_their_metrics"], refs["searched_their_metrics"]
    # the 4.0 k classifier of the searches with the faithful cost count, scored 2026-10-02
    # (datasets/highd/eval_their_protocol_keras.py; TTLC from its 4.9 k regressor)
    s4 = read_json(HIGHD / "results/their_protocol_eval_model_1790282740507531593_621856.json", None)
    def rows(p, h, s53, key4):
        r = [["published", p], ["hand-designed", h], ["searched 5.3 k", s53]]
        return r + [["searched 4.0 k", s4[key4]]] if s4 else r
    bench = {
        "metrics": [
            {"name": "Accuracy", "fmt": "f3", "better": "high", "rows": rows(pub["acc"], hb["acc"], 0.91153, "accuracy")},
            {"name": "F1", "fmt": "f3", "better": "high", "rows": rows(pub["f1"], hb["f1"], sr["f1"], "f1")},
            {"name": "AUC", "fmt": "f3", "better": "high", "rows": rows(pub["auc"], hb["auc"], sr["auc"], "auc")},
            {"name": "Robust horizon (s)", "fmt": "s2", "better": "high", "rows": rows(pub["tau_c_s"], hb["tau_c_s"], sr["tau_c_s"], "tau_c_s")},
            {"name": "Time-to-change RMSE (s)", "fmt": "s3", "better": "low", "rows": rows(pub["ttlc_rmse"], hb["ttlc_rmse"], 0.2608, "ttlc_rmse_s")},
        ],
        "splits": refs["splits"],
        "notes": ["Single runs of the deployed weights. Our accuracy moves by one to three points between training seeds (Seeds tab); "
                  "the means stay above the published figure, and one hand-recipe run of the 5.3 k model falls to 81.4%.",
                  "Scored with the metric functions transcribed from the published code.",
                  "Our reimplementation of their protocol, not a run of their own code.",
                  "Their trained weights are not public, so their accuracy cannot be re-run in our pipeline; the cost of "
                  "their architecture was measured with random weights (cost does not depend on the weights)."],
    }

    media = []
    if not share and media_file("replay/replay.gif").exists():
        newest = media_file("replay/replay.gif").parent.parent == MEDIA_NEW
        media.append({"id": "replay", "label": "One real lane change",
                      "caption": ("A real vehicle from the highD test recordings changing lane at 119 km/h. The traces are what "
                                  "our searched 4.0 k and 5.3 k models and our hand-designed 8.4 k CNN say at every window. "
                                  "Bars on the road: distance driven during one inference.") if newest else
                                 ("A real vehicle from the highD test recordings changing lane at 119 km/h. The trace is what the "
                                  "searched 5.3 k model says at every window. Bars on the road: distance driven during one inference."),
                      "src": data_uri(media_file("replay/replay.gif"), "image/gif")})
    elif share:
        media.append({"id": "replay", "label": "One real lane change", "withheld":
                      "Not included in this copy: the replay is drawn from raw highD trajectories, which the dataset licence does not let us pass on."})
    n_own, n_energy = own_counts(HIGHD / "results/deploy")

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
        "headline": {"value": f"{latency(reg, 'baseline_cls') / latency(reg, 'cls_v4'):.1f}×",
                     "label": "faster than the hand-designed CNN on the Cortex-M7, at the same accuracy",
                     "sub": f"The final search found a 4.0 k classifier: {pm(stat(seeds, 'highd_cls_v4_best', 'acc'))} over five runs, "
                            f"against {pm(stat(base, 'highd_baseline_cls', 'acc'))} for the hand-designed 8.4 k CNN, "
                            f"{ms(latency(reg, 'cls_v4'))} in float32 and {ms(latency(reg, 'cls_v4', 'int8 PTQ'))} in int8. "
                            f"The fastest build is the 5.3 k model in int8, {ms(latency(reg, 'cls_aaaaap', 'int8 PTQ'))}."},
        "kpis": [
            {"label": "Accuracy, five runs", "value": mean_only(stat(seeds, "highd_cls_v4_best", "acc")),
             "sub": f"final search, 4.0 k · hand-designed 8.4 k CNN {mean_only(stat(base, 'highd_baseline_cls', 'acc'))} · "
                    f"published {pub['acc'] * 100:.0f}%",
             "seed": (f"best saved classifier of the first two searches: {rerank['best_mean']['stats']['mean'] * 100:.1f}% at "
                      f"{rerank['best_mean']['params'] / 1000:.0f} k") if rerank else None},
            {"label": "Energy per prediction", "value": mj(energy_of(reg, "cls_v4", "int8 PTQ", "int8")),
             "sub": f"4.0 k model, int8, H7B3I-DK · hand-designed CNN {mj(energy_of(reg, 'baseline_cls', 'int8 PTQ', 'int8'))} · "
                    f"5.3 k model {mj(energy_of(reg, 'cls_aaaaap', 'int8 PTQ', 'int8'))}",
             "seed": "above the core asleep, measured on our board"},
            {"label": "Robust horizon", "value": f"{s4['tau_c_s']:.2f} s" if s4 else "–",
             "sub": f"final 4.0 k model · 5.3 k model {sr['tau_c_s']:.2f} s · published {pub['tau_c_s']:.2f} s"},
            {"label": "Time-to-change RMSE", "value": mean_only(stat(seeds, "highd_ttlc_v4_best", "rmse"), False),
             "sub": f"final search, 4.9 k · hand-designed {mean_only(stat(base, 'highd_baseline_ttlc', 'rmse'), False)} · "
                    f"published {pub['ttlc_rmse']} s",
             "seed": f"five runs each, {(latency(reg, 'baseline_ttlc') / latency(reg, 'ttlc_v4')):.1f}× faster than the hand-designed CNN"},
        ],
        "pipeline": [["Searches", str(sum(len(t["fronts"]) for t in tasks) + sum(1 for t in tasks for p in t["picks"] if p["front"] is None))],
                     ["Models saved or kept", str(sum(len(f["rows"]) for t in tasks for f in t["fronts"])
                                                  + sum(len(kept[k][0]) for k in ("cls_v2", "cls_v3", "ttlc_v2")))],
                     ["Board runs, cloud and own", str(cloud_runs(reg) + n_own)],
                     ["Energy readings", str(n_energy)]],
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
    seeds = seed_summary(read_jsonl(permuted(EXID / "results/seeds/seed_variance.jsonl")))
    seeds_final = seed_summary(read_jsonl(permuted(EXID / "results/seeds/seed_variance_final.jsonl")))
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
         "picks": [{"id": "exid_cls_aaaaam", "label": "highD first search, 7.9 k", "seed_key": "exid_cls_aaaaam"},
                   {"id": "exid_cls_aaaaap", "label": "highD tighter search, 5.3 k", "seed_key": "exid_cls_aaaaap"},
                   {"id": "exid_cls_v2", "label": "highD second search, 24 k", "seed_key": "exid_cls_v2p_best"},
                   {"id": "exid_cls_v3", "label": "highD third search, 7.9 k", "seed_key": "exid_cls_v3_best"},
                   {"id": "exid_cls_v4", "label": "highD final search, 4.0 k", "seed_key": "exid_cls_v4_best"}],
         "refs": [{"kind": "point", "label": "hand-designed CNN", "params": hand["cls"], "seed_key": "exid_baseline_cls"}],
         "fronts": []},
        {"id": "exid_ttlc", "label": "Time to lane change",
         "metric": {"key": "rmse", "name": "test RMSE (s)", "better": "low", "fmt": "s3"},
         "picks": [{"id": "exid_ttlc_aaaaaw", "label": "highD first search, 28 k", "seed_key": "exid_ttlc_aaaaaw"},
                   {"id": "exid_ttlc_v2", "label": "highD second search, 26 k", "seed_key": "exid_ttlc_v2p_best"},
                   {"id": "exid_ttlc_v4", "label": "highD final search, 4.9 k", "seed_key": "exid_ttlc_v4_best"}],
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
    exid_media = []
    if not share and media_file("exid/exid_exit.gif").exists():
        exid_media.append({"id": "exit", "label": "One real exit",
                           "caption": "A real exit to the off-ramp from the exiD test recordings (Cologne, 58 km/h). Green: our "
                                      "hand-designed CNN trained on exiD; orange: the same network trained on highD (seed 0).",
                           "src": data_uri(media_file("exid/exid_exit.gif"), "image/gif")})
    elif share:
        exid_media.append({"id": "exit", "label": "One real exit", "withheld":
                           "Not included in this copy: the replay is drawn from raw exiD trajectories, which the dataset licence does not let us pass on."})

    return {
        "id": "exid", "name": "exiD", "title": "Motorway entries and exits", "status": "complete", "accent": 3,
        "views": ["overview", "transfer", "robust", "scenario"],
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
        "media": exid_media,
    }


# Input rate of each dataset as prepared: LCIR windows at 10 Hz, highD and exiD tracks downsampled to 5 Hz by the
# protocol of Mozaffari et al.; one prediction per input sample, so the period is the deadline of a prediction.
RATE_HZ = {"dmir": 10, "highd": 5, "exid": 5}


def size_text(b):
    return f"{b // 1048576} MiB" if b % 1048576 == 0 else f"{b // 1024:,} KiB"


def build_study(dm, hd, ex):
    """The first page: the problem, the idea and the method, then a few results that link into the datasets."""
    boards = dm["registry"]["boards"]
    h7, m4 = "STM32H7B3I-DK", "NUCLEO-F401RE"
    reg = {m["id"]: m for m in dm["registry"]["models"]}

    def board(mid, b):
        v = reg[mid]["variants"][0]["boards"].get(b, {})
        return v.get("latency_ms"), v.get("fits", True) and v.get("latency_ms") is not None
    problem = []
    for mid, label in (("tr_lcr", "Reference Transformer"), ("tr_lcl", "Reference Transformer"), ("ref_cnn", "Reference CNN")):
        lat7, _ = board(mid, h7)
        lat4, fits4 = board(mid, m4)
        problem.append({"label": label, "params": reg[mid]["params"], "m7": lat7, "m4": lat4 if fits4 else None})
    slowest = max(problem, key=lambda p: p["m7"])

    final = {"dmir": ("dscnn_cls", "dscnn_lcr", "dscnn_lcl", "cls_v4", "lcr_v4", "lcl_v4"),
             "highd": ("baseline_cls", "baseline_ttlc", "cls_v4", "ttlc_v4")}
    share = max(v["boards"][h7]["latency_ms"] * RATE_HZ[d["id"]] / 10 for d in (dm, hd) for m in d["registry"]["models"]
                if m["id"] in final[d["id"]] for v in m["variants"] if v["boards"].get(h7, {}).get("latency_ms") is not None)
    e8, eref = energy_of(dm["registry"], "cls_tiny"), energy_of(dm["registry"], "ref_cnn")
    rb = read_json(HIGHD / "results/resource_bias.json", {})
    undercount = max((v["macs_ratio"]["max"] for k, v in rb.items() if "n" in v and not k.endswith("_v4")), default=None)
    cls = next(t for t in ex["transfer"]["tasks"] if t["task"] == "cls")
    exm = {m["id"]: m["stats"] for m in cls["methods"]}
    rr = hd.get("rerank")

    return {
        "title": "Lane-change prediction on the microcontroller of a vehicle",
        "lead": ("Driver assistance can warn a driver, or prepare a controller, only if a lane change is predicted "
                 "before it happens. In a vehicle the predictor runs on a small microcontroller next to other "
                 "functions, once for every input sample. This study builds compact neural networks for that "
                 "setting and measures them on two STM32 boards."),
        "requirements": [
            {"label": "Finish before the next input", "value": f"{1000 // RATE_HZ['dmir']} ms",
             "sub": f"between inputs at {RATE_HZ['dmir']} Hz (LCIR); {1000 // RATE_HZ['highd']} ms at "
                    f"{RATE_HZ['highd']} Hz (highD, exiD)"},
            {"label": "Predict early enough", "value": "0.7–1.5 s",
             "sub": "for a driver to respond, depending on how expected the event is (Green, 2000)"},
            {"label": "Fit the board", "value": f"{size_text(boards[h7]['flash_b'])} · {size_text(boards[m4]['flash_b'])}",
             "sub": f"flash of the {boards[h7]['core']} and {boards[m4]['core']} boards, shared with the other functions"},
            {"label": "Add little power", "value": "energy × rate",
             "sub": "it runs at every input, so its energy per prediction sets the power it adds"},
        ],
        "uses": [
            ["Lane change decision aid", "Warns the driver of vehicles alongside or approaching from behind in the "
             "adjacent lanes (ISO 17387). Predicting the driver's intention lets it warn before the maneuver starts."],
            ["Cruise control and automated lane keeping", "Have to anticipate the lane changes of the vehicles around. "
             "Automated lane keeping may also change lanes itself since 2023 (UN Regulation No. 157)."],
        ],
        "gap": ("Two recent surveys count 76 and more than 150 lane-change prediction studies. They report prediction "
                "quality on recorded data. Few validate a predictor in simulation or in a vehicle, and fewer measure its "
                "time and energy on vehicle hardware."),
        "problem": problem, "period_ms": 1000 // RATE_HZ["dmir"],
        "problem_caption": (f"Reference models from earlier work on the LCIR data, timed on the {boards[h7]['core']} board. "
                            f"Dashed line: the {1000 // RATE_HZ['dmir']} ms between two inputs at {RATE_HZ['dmir']} Hz. "
                            f"At 100 km/h the car moves {100 / 3.6 * slowest['m7'] / 1000:.1f} m during one prediction "
                            f"of the larger reference Transformer."),
        "idea": [
            ["Search within the board's limits", "A constrained architecture search (µNAS) proposes one-dimensional "
             "CNNs and keeps those within budgets on peak RAM, model size and multiply-accumulate operations."],
            ["Compare under the same training", "Small hand-designed CNNs and reference models get the same training "
             "recipes, and every final model is retrained with five seeds, because one training run can mislead."],
            ["Measure on the boards", "Every build is timed on two STM32 boards, its energy per prediction is read with "
             "a USB meter, and both are set against the input period."],
        ],
        "datasets": [{"id": d["id"], "name": d["name"], "accent": d["accent"], "title": d["title"],
                      "who": d["facts"][0][1], "input": next((v for k, v in d["facts"] if k == "Input"), "")}
                     for d in (dm, hd, ex)],
        "stages": [
            ["Prepare", "Cut the recordings into windows, derive the labels, split by driver or by recording."],
            ["Search", "Aging evolution trains 150 candidates and scores error and cost against the budgets."],
            ["Select", "Retrain the 12 best five times each; keep the best mean on validation data."],
            ["Quantize", "Export float32 and int8 builds and score each on the test windows."],
            ["Compile", "ST Edge AI Core turns each build into C code and reports flash, RAM and operations."],
            ["Measure", "Time each build on both boards and read the board's power for the energy."],
        ],
        "data_use": [  # label, first stage, last stage (1-based, inclusive), style
            ["Training windows: search, retraining, int8 calibration", 2, 4, "train"],
            ["Validation windows: every choice", 2, 3, "val"],
            ["Test windows: the final score only", 4, 4, "test"],
            ["Boards: latency and energy, on random inputs", 5, 6, "board"],
        ],
        "changes": [
            ["Head with global average pooling and dropout", "the space could not express the small hand-designed CNNs"],
            ["Cost counted on the padded network that is trained",
             f"the fork counted up to {undercount:.1f} times too few operations, so its budgets did not bind"
             if undercount else "the fork counted too few operations, so its budgets did not bind"],
            ["Saver with unique file names and no test data",
             "resumed runs and parallel workers overwrote saved models, and the saver filtered on test error"],
            ["Choice made on five training runs",
             f"one run does not rank the candidates: rank correlation {rr['rho']:.2f} over {rr['n']} highD models"
             if rr else "one run does not rank the candidates"],
            ["Training windows shuffled in full",
             "a buffer of 2,048 windows over data stored scenario by scenario could fill a batch with one recording"],
        ],
        "changes_note": ("µNAS was built for image and audio classifiers; a fork of it adds one-dimensional convolutions "
                         "and regression outputs. None of these changes is a new search method; each fixed a failure "
                         "we measured, as the searches exposed it."),
        "boards": [{"name": b, "core": boards[b]["core"], "mhz": boards[b]["mhz"], "flash": size_text(boards[b]["flash_b"]),
                    "ram": size_text(boards[b]["ram_b"])} for b in (h7, m4)],
        "measure_note": ("Latency from the ST Edge AI Developer Cloud, repeated on our own boards over 200 timed runs. "
                         "Energy per prediction: the power drawn above the core asleep at the same clock, read with a "
                         "USB meter at the board input, times the latency."),
        "found": [
            {"value": mj(e8), "label": f"per prediction for an 8 k LCIR intention model, against {mj(eref)} for the "
                                       f"441 k reference CNN at the same accuracy", "link": "dmir/hardware"},
            {"value": f"≤ {share:.1f}%", "label": "of the input period on the Cortex-M7 for the hand-designed models "
                                                  "and the final searched models", "link": "compare"},
            {"value": hd["headline"]["value"], "label": "faster than the hand-designed CNN on highD at the same "
                                                        "accuracy, found by the final search", "link": "highd/overview"},
            {"value": f"−{(exm['scratch']['mean'] - exm['zero-shot']['mean']) * 100:.1f} points",
             "label": f"accuracy a highD model loses at motorway entries and exits; trained there it reaches "
                      f"{exm['scratch']['mean'] * 100:.1f}%", "link": "exid/transfer"},
        ],
        "sources": [
            "ISO 17387:2026, Lane change decision aid systems (LCDAS): performance requirements and test procedures.",
            "UN Regulation No. 157, Automated Lane Keeping Systems, 01 series of amendments, in force January 2023.",
            "M. Green, “How long does it take to stop?”, Transportation Human Factors 2(3), 2000.",
            "F. De Cristofaro et al., IEEE Trans. Intell. Veh. 11(4), 2026; M. A. Manzour et al., IET Intell. Transp. "
            "Syst. 20(1), 2026: surveys of lane-change prediction.",
            "E. Liberis, Ł. Dudziak, N. D. Lane, µNAS: constrained neural architecture search for microcontrollers, "
            "EuroMLSys 2021.",
        ],
    }


def main():
    copy = "share" if "--share" in sys.argv else "lab" if "--lab" in sys.argv else "full"
    share = copy == "share"
    datasets = [build_dmir(share), build_highd(share), build_exid(share)]
    data = {"built": date.today().isoformat(), "share": share, "copy": copy,
            "study": build_study(*datasets), "datasets": datasets}
    status = HERE / "status.local.json"
    if copy == "full" and status.exists():
        data["status"] = read_json(status)

    src = HERE / "src"
    html = (src / "index.html").read_text(encoding="utf-8")
    css = (src / "app.css").read_text(encoding="utf-8")
    js = (src / "app.js").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = (html.replace("/*__CSS__*/", css)
                .replace("/*__JS__*/", js)
                .replace("/*__DATA__*/null", payload))
    out = HERE / "dist" / {"share": "results-explorer-share.html", "lab": "results-explorer-lab.html",
                           "full": "results-explorer.html"}[copy]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.0f} KB  ({copy} copy, status page: {'status' in data})")


if __name__ == "__main__":
    main()
