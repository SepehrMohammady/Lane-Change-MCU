"""Addendum to the comparison at exiD sites (2026-11-19, added after its results; listed, no claims).

Fixed in the local notes (2026-11-19 22:05) before this script existed: what part of the gain a class-prior correction
gives. Same devices, streams, networks (highD arm K, seeds 100-104), measures, selection rule on the validation
recordings and single scoring of the 14 test recordings as unas/online_site_compare.py.

  bias_only      the update of ours (delayed labels, class-balanced memories, start at 20, 4 per class) on the bias
                 alone; eta 0.003 / 0.01 / 0.03
  prior_em       label-free class-prior correction of the fixed layer (online EM in the manner of Saerens et al. 2002):
                 logits + log(pi); every frame pi moves towards the mean corrected posterior of the frame's windows by
                 a factor alpha (1e-4 / 1e-3 / 1e-2); pi starts uniform (the networks were trained on balanced windows)
  prior_labels   the same correction with pi the running class frequency of the delayed labels (alpha as above)

  WSL: SEEDS=100 OUT_NAME=online_addendum_s100.jsonl ~/dmir_nas/bin/python unas/online_site_addendum.py tune | final
  .venv\\Scripts\\python unas/online_site_addendum.py --report
Output: datasets/exid/results/seeds/online_addendum*.jsonl, datasets/exid/results/hil/online_addendum.json.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_site_compare as oc                             # noqa: E402
import online_site_pilot as site                             # noqa: E402
import track_stream as ts                                    # noqa: E402  (path set by online_site_pilot)

op = site.op
OUT = site.RESULTS / "seeds" / os.environ.get("OUT_NAME", "online_addendum.jsonl")
SUMMARY = site.RESULTS / "hil" / "online_addendum.json"
GRIDS = {"bias_only": [{"eta": e} for e in (0.003, 0.01, 0.03)],
         "prior_em": [{"alpha": a} for a in (1e-4, 1e-3, 1e-2)],
         "prior_labels": [{"alpha": a} for a in (1e-4, 1e-3, 1e-2)]}


def run_method(method: str, cfg: dict, f: np.ndarray, w: np.ndarray, w0, b0, rng) -> np.ndarray:
    label, known, t_abs, rec = w[:, 3], w[:, 4].astype(bool), w[:, 5], w[:, 6]
    n, d = f.shape
    Bt = b0.copy()
    eye, rep = np.eye(3), np.repeat(np.arange(3), op.PER_CLASS)
    mem = np.zeros((3, op.MEM, d))
    count, pos = np.zeros(3, np.int64), np.zeros(3, np.int64)
    pi = np.full(3, 1 / 3)
    pred = np.empty(n, np.int8)
    pend = collections.deque()
    cut = np.flatnonzero((np.diff(t_abs) != 0) | (np.diff(rec) != 0)) + 1
    for g0, g1 in zip(np.concatenate([[0], cut]), np.concatenate([cut, [n]])):
        r, t = rec[g0], t_abs[g0]
        if method != "prior_em":
            while pend and rec[pend[0]] != r:
                pend.popleft()
            while pend and t_abs[pend[0]] + site.OFFSET <= t:
                i = pend.popleft()
                c = int(label[i])
                if method == "prior_labels":
                    pi = (1 - cfg["alpha"]) * pi + cfg["alpha"] * eye[c]
                    continue
                mem[c, pos[c]] = f[i]
                pos[c] = (pos[c] + 1) % op.MEM
                count[c] = min(count[c] + 1, op.MEM)
                if (count >= op.MIN_CLASS).all():
                    x = np.concatenate([mem[k, rng.integers(0, count[k], op.PER_CLASS)] for k in range(3)])
                    gr = (op.softmax(x @ w0 + Bt) - eye[rep]) / len(x)
                    Bt = Bt - cfg["eta"] * gr.sum(0)
        z = f[g0:g1] @ w0 + (Bt if method == "bias_only" else b0)
        if method != "bias_only":
            z = z + np.log(np.clip(pi, 1e-12, None))
        pred[g0:g1] = np.argmax(z, axis=1)
        if method == "prior_em":
            pi = (1 - cfg["alpha"]) * pi + cfg["alpha"] * op.softmax(z).mean(0)
        else:
            pend.extend(int(i) for i in range(g0, g1) if known[i])
    return pred


def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_addendum*.jsonl")):
        if "smoke" not in path.name:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def run(phase: str) -> None:
    import input_scaling_check as isc

    keras = isc.keras
    seeds = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]
    smoke = os.environ.get("SMOKE") == "1"
    if smoke and "OUT_NAME" not in os.environ:
        raise SystemExit("SMOKE=1 needs its own OUT_NAME")
    if phase == "tune":
        jobs, scored, stream_splits = [(m, c) for m, grid in GRIDS.items() for c in grid], "val", 2
    else:
        sel = selection(records())
        if sel.get("complete") is False:
            raise SystemExit("tuning incomplete")
        jobs, scored, stream_splits = [(m, sel[m]["cfg"]) for m in GRIDS], "test", 3
    done = {(r["phase"], r["config"], r["seed"]) for r in records()}
    by_loc = {loc: w[w[:, 0] < stream_splits] for loc, w in site.windows_by_location().items()}
    if smoke:
        small = min(by_loc, key=lambda k: len(by_loc[k]))
        by_loc, seeds = {small: by_loc[small]}, seeds[:1]
    rank = site.SPLITS.index(scored)
    n_scored = int((np.load(site.tsx.tracks_file(scored))["lens"] - ts.IN_LEN + 1).sum())
    keep, arm, prep_h = site.NETS["highD_K"]
    views = site.scaled_views(prep_h)
    for seed in seeds:
        todo = [(m, c) for m, c in jobs if (phase, oc.cfg_id(m, c), seed) not in done]
        if not todo:
            continue
        t1 = time.perf_counter()
        feat, w0, b0 = site.network(keras, keep, arm, seed)
        outs = {oc.cfg_id(m, c): np.full(n_scored, -1, np.int8) for m, c in todo}
        for loc, w in by_loc.items():
            sc = w[:, 0] == rank
            f = site.features(feat, views, w)
            for m, c in todo:
                pr = run_method(m, c, f, w, w0, b0, np.random.default_rng([seed, loc, 7]))
                outs[oc.cfg_id(m, c)][w[sc, 1]] = pr[sc]
            del f
        keras.backend.clear_session()
        for m, c in todo:
            key = oc.cfg_id(m, c)
            if smoke:
                print(f"  smoke {key}: {int((outs[key] >= 0).sum())} windows scored", flush=True)
                continue
            assert (outs[key] >= 0).all()
            rec = {"phase": phase, "method": m, "cfg": c, "config": key, "seed": seed, "split": scored,
                   "measures": site.split_measures(scored, outs[key]), "float32": True,
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"{phase} seed {seed}: {len(todo)} configurations, {time.perf_counter() - t1:.0f} s", flush=True)


def selection(rows: list) -> dict:
    ref = oc.pilot("tune")["fixed"]
    sel = {}
    for m, grid in GRIDS.items():
        cands = []
        for c in grid:
            got = [r for r in rows if r["phase"] == "tune" and r["config"] == oc.cfg_id(m, c)]
            if len(got) < 5:
                return {"complete": False}
            mm = oc.means(got)
            ok = (mm["correct_2_s_before"] >= ref["correct_2_s_before"] - site.EARLY - 1e-12
                  and mm["runs_0.6_s_per_vehicle_hour"] <= ref["runs_0.6_s_per_vehicle_hour"] + 1e-12)
            cands.append((mm["windows_correct"] - ref["windows_correct"], ok, c, mm))
        qual = [x for x in cands if x[1]]
        best = max(qual, key=lambda x: x[0]) if qual else max(cands, key=lambda x: x[0])
        sel[m] = {"cfg": best[2], "qualified": bool(qual),
                  "validation": [{"cfg": c, "gain": round(g, 4), "qualifies": ok, **mm} for g, ok, c, mm in cands]}
    return sel


def report() -> None:
    rows = records()
    sel = selection(rows)
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes 2026-11-19 22:05",
           "selection": sel}
    if sel.get("complete") is not False:
        pt = oc.pilot("final")
        test = {"fixed": pt["fixed"], "ours": pt["adapted"]}
        for m in GRIDS:
            got = [r for r in rows if r["phase"] == "final" and r["config"] == oc.cfg_id(m, sel[m]["cfg"])]
            if len(got) == 5:
                test[m] = oc.means(got)
        out["test"] = test
        for m, s in sel.items():
            print(f"{m}: selected {s['cfg']} (qualified: {s['qualified']})")
        for m, mm in test.items():
            print(f"test {m:14s} " + "; ".join(f"{k} {v}" for k, v in mm.items()))
    SUMMARY.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] in ("tune", "final"):
        run(sys.argv[1])
    else:
        raise SystemExit("usage: online_site_addendum.py tune | final | --report")
