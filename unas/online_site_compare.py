"""Comparison of approaches at exiD sites (2026-11-19): what the self-labelled on-device update is better than.

Same devices, streams, networks (highD arm K, seeds 100-104), measures and protocol as unas/online_site_pilot.py: one
device per exiD location running through its training, validation and test recordings; settings selected on the
validation recordings by the pilot's rule; the 14 test recordings scored once. Fixed before the script existed
(local notes, 2026-11-19 18:11). Methods:

  site_scaling   label-free: the network as deployed, inputs scaled with exiD's training range
  entropy        label-free (TENT's objective): one gradient step per frame on the mean prediction entropy of that
                 frame's windows, last layer [step 1e-4, 1e-3, 1e-2]
  self_training  the pilot's update with the network's own class as the label, used at once, memories balanced by
                 the predicted class [eta 0.003, 0.01]
  naive          true delayed labels, one gradient step on the new window alone, no memory [eta 0.003, 0.01]
  rls            recursive least squares on the last layer from the true delayed labels (AGen's RLS-PAA family):
                 target 4 (onehot - softmax(fixed logits)) added to the fixed logits, no class balance, P0 = I, P reset
                 to I when its trace exceeds 1e6 (amendment 18:18, after the smoke run overflowed),
                 forgetting factor [0.99, 0.999]
  instant        the pilot's update (eta 0.003, lambda 0) with each label used at once, after its window's prediction
  offline_refit  the last layer refitted offline on all windows of exiD's training recordings with known labels,
                 classes balanced by subsampling (rng 0), softmax cross-entropy, 200 full-batch steps of 0.1 from the
                 fixed layer; then fixed (labels a deployed device does not have)
The pilot's own records give "fixed" and "ours". Claims on the test recordings (windows correct, label known):
(C1) ours >= best of site_scaling, entropy, self_training + 10 points; (C2) ours >= naive + 3 points, or naive fails
the constraints on the validation recordings; (C3) instant <= ours + 3 points; (C4) ours >= rls + 3 points.

  WSL: SEEDS=100 OUT_NAME=online_compare_s100.jsonl ~/dmir_nas/bin/python unas/online_site_compare.py tune | final
  .venv\\Scripts\\python unas/online_site_compare.py --report
Output: datasets/exid/results/seeds/online_compare*.jsonl, datasets/exid/results/hil/online_compare.json.
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import online_site_pilot as site                             # noqa: E402
import track_stream as ts                                    # noqa: E402  (path set by online_site_pilot)

op = site.op
RES = site.RESULTS
OUT = RES / "seeds" / os.environ.get("OUT_NAME", "online_compare.jsonl")
SUMMARY = RES / "hil" / "online_compare.json"
GRIDS = {"entropy": [{"step": s} for s in (1e-4, 1e-3, 1e-2)],
         "self_training": [{"eta": e} for e in (0.003, 0.01)],
         "naive": [{"eta": e} for e in (0.003, 0.01)],
         "rls": [{"mu": m} for m in (0.99, 0.999)]}
FIXED_CFG = {"site_scaling": {}, "instant": {"eta": 0.003, "lam": 0.0}, "offline_refit": {"steps": 200, "lr": 0.1}}
OURS = "G_eta0.003_lam0.0"


def cfg_id(method: str, cfg: dict) -> str:
    return method + "".join(f"_{k}{v}" for k, v in sorted(cfg.items()))


# ------------------------------------------------------------------------------------------------- the methods
def run_method(method: str, cfg: dict, f: np.ndarray, w: np.ndarray, w0, b0, rng) -> np.ndarray:
    label, known, t_abs, rec = w[:, 3], w[:, 4].astype(bool), w[:, 5], w[:, 6]
    n, d = f.shape
    Wt, Bt = w0.copy(), b0.copy()
    eye, rep = np.eye(3), np.repeat(np.arange(3), op.PER_CLASS)
    mem = np.zeros((3, op.MEM, d))
    count, pos = np.zeros(3, np.int64), np.zeros(3, np.int64)
    if method == "rls":
        V, P = np.zeros((d + 1, 3)), np.eye(d + 1)
        r_all = 4.0 * (eye[label] - op.softmax(f @ w0 + b0))
    pred = np.empty(n, np.int8)
    pend = collections.deque()
    delayed = method in ("naive", "rls")

    def g_step(eta, lam=0.0):
        nonlocal Wt, Bt
        x = np.concatenate([mem[k, rng.integers(0, count[k], op.PER_CLASS)] for k in range(3)])
        gr = (op.softmax(x @ Wt + Bt) - eye[rep]) / len(x)
        Wt = Wt - eta * (x.T @ gr + 2 * lam * (Wt - w0))
        Bt = Bt - eta * (gr.sum(0) + 2 * lam * (Bt - b0))

    def remember(c, x):
        mem[c, pos[c]] = x
        pos[c] = (pos[c] + 1) % op.MEM
        count[c] = min(count[c] + 1, op.MEM)

    cut = np.flatnonzero((np.diff(t_abs) != 0) | (np.diff(rec) != 0)) + 1
    for g0, g1 in zip(np.concatenate([[0], cut]), np.concatenate([cut, [n]])):
        r, t = rec[g0], t_abs[g0]
        while pend and rec[pend[0]] != r:
            pend.popleft()
        while pend and t_abs[pend[0]] + site.OFFSET <= t:
            i = pend.popleft()
            c = int(label[i])
            if method == "naive":
                gr = op.softmax(f[i] @ Wt + Bt) - eye[c]
                Wt = Wt - cfg["eta"] * np.outer(f[i], gr)
                Bt = Bt - cfg["eta"] * gr
            else:                                            # rls
                x = np.append(f[i], 1.0)
                px = P @ x
                k = px / (cfg["mu"] + x @ px)
                V = V + np.outer(k, r_all[i] - x @ V)
                P = (P - np.outer(k, px)) / cfg["mu"]
                if not np.isfinite(P).all() or np.trace(P) > 1e6:   # covariance resetting (amendment 18:18)
                    P = np.eye(d + 1)
        fr = f[g0:g1]
        logits = fr @ Wt + Bt
        if method == "rls":
            logits = logits + np.hstack([fr, np.ones((len(fr), 1))]) @ V
        cls = np.argmax(logits, axis=1)
        pred[g0:g1] = cls
        if method == "entropy":
            p = op.softmax(logits)
            lp = np.log(np.clip(p, 1e-12, None))
            h = -(p * lp).sum(1, keepdims=True)
            gz = -p * (lp + h) / len(fr)                     # d(mean entropy)/d(logits)
            Wt = Wt - cfg["step"] * (fr.T @ gz)
            Bt = Bt - cfg["step"] * gz.sum(0)
        elif method == "self_training":
            for j in range(len(fr)):
                remember(int(cls[j]), fr[j])
                if (count >= op.MIN_CLASS).all():
                    g_step(cfg["eta"])
        elif method == "instant":
            for j in range(g0, g1):
                if known[j]:
                    remember(int(label[j]), f[j])
                    if (count >= op.MIN_CLASS).all():
                        g_step(cfg["eta"], cfg["lam"])
        if delayed:
            pend.extend(int(i) for i in range(g0, g1) if known[i])
    return pred


def offline_refit(feat, views: dict, w_by_loc: dict, w0, b0, cfg: dict) -> tuple:
    """Balanced subsample of the known windows of the training recordings (rng 0), features computed for it only."""
    rank = site.SPLITS.index("train")
    rows = {loc: np.flatnonzero((w[:, 0] == rank) & (w[:, 4] == 1)) for loc, w in w_by_loc.items()}
    locs = sorted(rows)
    y_all = np.concatenate([w_by_loc[loc][rows[loc], 3] for loc in locs])
    where = np.concatenate([np.stack([np.full(len(rows[loc]), loc), rows[loc]], 1) for loc in locs])
    rng = np.random.default_rng(0)
    counts = np.bincount(y_all, minlength=3)
    pick = np.concatenate([rng.choice(np.flatnonzero(y_all == c), size=int(counts.min()), replace=False) for c in range(3)])
    xs, ys = [], []
    for loc in locs:
        r = np.sort(where[pick][where[pick][:, 0] == loc][:, 1])
        if len(r):
            xs.append(site.features(feat, views, w_by_loc[loc][r]))
            ys.append(w_by_loc[loc][r, 3])
    x, y = np.concatenate(xs).astype(np.float64), np.concatenate(ys)
    Wt, Bt = w0.copy(), b0.copy()
    oh = np.eye(3)[y]
    for _ in range(cfg["steps"]):
        g = (op.softmax(x @ Wt + Bt) - oh) / len(x)
        Wt, Bt = Wt - cfg["lr"] * (x.T @ g), Bt - cfg["lr"] * g.sum(0)
    return Wt, Bt, int(counts.min())


# ------------------------------------------------------------------------------------------------- the runs
def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_compare*.jsonl")):
        if "smoke" not in path.name:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def run(phase: str) -> None:
    import input_scaling_check as isc

    keras = isc.keras
    seeds = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]
    if phase == "tune":
        jobs = [(m, c) for m, grid in GRIDS.items() for c in grid]
        scored, stream_splits = "val", 2
    else:
        sel = selection(records())
        jobs = [(m, sel[m]["cfg"]) for m in GRIDS if sel.get(m, {}).get("cfg") is not None]
        jobs += [(m, c) for m, c in FIXED_CFG.items()]
        scored, stream_splits = "test", 3
    smoke = os.environ.get("SMOKE") == "1"
    if smoke and "OUT_NAME" not in os.environ:
        raise SystemExit("SMOKE=1 needs its own OUT_NAME")
    done = {(r["phase"], r["config"], r["seed"]) for r in records()}
    by_loc = {loc: w[w[:, 0] < stream_splits] for loc, w in site.windows_by_location().items()}
    if smoke:                                                # the smallest location, one seed: a code check
        small = min(by_loc, key=lambda k: len(by_loc[k]))
        by_loc, seeds = {small: by_loc[small]}, seeds[:1]
    rank = site.SPLITS.index(scored)
    n_scored = int((np.load(site.tsx.tracks_file(scored))["lens"] - ts.IN_LEN + 1).sum())
    keep, arm, prep_h = site.NETS["highD_K"]
    views = {"highD": site.scaled_views(prep_h)}
    for seed in seeds:
        todo = [(m, c) for m, c in jobs if (phase, cfg_id(m, c), seed) not in done]
        if not todo:
            continue
        t1 = time.perf_counter()
        feat, w0, b0 = site.network(keras, keep, arm, seed)
        if any(m == "site_scaling" for m, _ in todo):
            views.setdefault("exiD", site.scaled_views(site.NETS["exiD_K"][2]))
        outs = {cfg_id(m, c): np.full(n_scored, -1, np.int8) for m, c in todo}
        extra = {cfg_id(m, c): {} for m, c in todo}
        refit = {}
        for m, c in todo:
            if m == "offline_refit":
                Wr, Br, per_class = offline_refit(feat, views["highD"], by_loc, w0, b0, c)
                refit[cfg_id(m, c)] = (Wr, Br)
                extra[cfg_id(m, c)]["training_windows_per_class"] = per_class
        for loc, w in by_loc.items():                        # one location's features at a time
            sc = w[:, 0] == rank
            f = site.features(feat, views["highD"], w)
            fs = site.features(feat, views["exiD"], w) if "exiD" in views and any(m == "site_scaling" for m, _ in todo) else None
            for m, c in todo:
                key = cfg_id(m, c)
                if m == "site_scaling":
                    pr = np.argmax(fs @ w0 + b0, axis=1)
                elif m == "offline_refit":
                    pr = np.argmax(f @ refit[key][0] + refit[key][1], axis=1)
                else:
                    rng = np.random.default_rng([seed, loc, 7])
                    pr = run_method(m, c, f, w, w0, b0, rng)
                outs[key][w[sc, 1]] = pr[sc]
            del f, fs
        keras.backend.clear_session()
        for m, c in todo:
            key = cfg_id(m, c)
            if smoke:
                print(f"  smoke {key}: {int((outs[key] >= 0).sum())} windows scored", flush=True)
                continue
            assert (outs[key] >= 0).all()
            rec = {"phase": phase, "method": m, "cfg": c, "config": key, "seed": seed, "split": scored,
                   "measures": site.split_measures(scored, outs[key]), **extra[key], "float32": True,
                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"{phase} seed {seed}: {len(todo)} configurations, {time.perf_counter() - t1:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------- the report
def means(rows: list) -> dict:
    out = {}
    for key in site.KEYS:
        vals = [r["measures"]["all"][key] for r in rows]
        out[key] = None if any(v is None for v in vals) else round(st.mean(vals), 5)
    return out


def pilot(phase: str) -> dict:
    """fixed and ours from the site pilot's records (val = tune phase, test = final phase)."""
    rows = [r for r in site.records() if r["phase"] == phase and r["net"] == "highD_K" and r["setting"] == OURS]
    return {k: {key: round(st.mean(r[k]["all"][key] for r in rows), 4) for key in site.KEYS} for k in ("fixed", "adapted")}


def selection(rows: list) -> dict:
    ref = pilot("tune")["fixed"]
    sel = {}
    for m, grid in GRIDS.items():
        cands = []
        for c in grid:
            got = [r for r in rows if r["phase"] == "tune" and r["config"] == cfg_id(m, c)]
            if len(got) < 5:
                return {"complete": False}
            mm = means(got)
            ok = (mm["correct_2_s_before"] >= ref["correct_2_s_before"] - site.EARLY - 1e-12
                  and mm["runs_0.6_s_per_vehicle_hour"] <= ref["runs_0.6_s_per_vehicle_hour"] + 1e-12)
            cands.append((mm["windows_correct"] - ref["windows_correct"], ok, c, mm))
        qual = [x for x in cands if x[1]]
        best = max(qual, key=lambda x: x[0]) if qual else max(cands, key=lambda x: x[0])
        sel[m] = {"cfg": best[2], "qualified": bool(qual),          # not qualified: best gain, marked (note 18:40)
                  "validation": [{"cfg": c, "gain": round(g, 4), "qualifies": ok, **mm} for g, ok, c, mm in cands]}
    return sel


def report() -> None:
    rows = records()
    sel = selection(rows)
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes 2026-11-19 18:11",
           "selection": sel}
    if sel.get("complete") is False:
        print("tuning incomplete")
        SUMMARY.write_text(json.dumps(out, indent=1))
        return
    for m, s in sel.items():
        print(f"{m}: selected {s['cfg']} (qualified: {s['qualified']})")
    test = {}
    pt = pilot("final")
    test["fixed"], test["ours"] = pt["fixed"], pt["adapted"]
    for m in list(GRIDS) + list(FIXED_CFG):
        c = sel[m]["cfg"] if m in GRIDS else FIXED_CFG[m]
        got = [r for r in rows if r["phase"] == "final" and r["config"] == cfg_id(m, c)]
        if len(got) == 5:
            test[m] = means(got)
    out["test"] = test
    for m, mm in test.items():
        print(f"test {m:14s} " + "; ".join(f"{k} {v}" for k, v in mm.items()))
    if all(k in test for k in ("site_scaling", "entropy", "self_training", "naive", "rls", "instant")):
        wc = {m: test[m]["windows_correct"] for m in test}
        best_free = max(wc["site_scaling"], wc["entropy"], wc["self_training"])
        out["claims"] = {
            "C1_ours_minus_best_label_free": round(wc["ours"] - best_free, 4), "C1_met": wc["ours"] - best_free >= 0.10 - 1e-12,
            "C2_ours_minus_naive": round(wc["ours"] - wc["naive"], 4), "C2_naive_qualified_on_validation": sel["naive"]["qualified"],
            "C2_met": (wc["ours"] - wc["naive"] >= 0.03 - 1e-12) or not sel["naive"]["qualified"],
            "C3_instant_minus_ours": round(wc["instant"] - wc["ours"], 4), "C3_met": wc["instant"] - wc["ours"] <= 0.03 + 1e-12,
            "C4_ours_minus_rls": round(wc["ours"] - wc["rls"], 4), "C4_met": wc["ours"] - wc["rls"] >= 0.03 - 1e-12}
        print(json.dumps(out["claims"], indent=1))
    SUMMARY.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] in ("tune", "final"):
        run(sys.argv[1])
    else:
        raise SystemExit("usage: online_site_compare.py tune | final | --report")
