"""A guard against false lane-change labels: on-device learning at exiD sites when the tracker's lateral position is noisy
(2026-11-26).

Fixed before this script existed (local notes, 2026-11-26 17:52). The noise study of 2026-11-25 (unas/online_noise.py)
found that random wrong labels fill the class-balanced lane-change memories. Here the label errors come from a model
of the device's tracker instead: the lateral position of each vehicle in its lane is seen with an error e_t, AR(1) at
5 Hz (e_t = 0.8 e_(t-1) + 0.6 sigma eta_t, stationary standard deviation sigma), and the device assigns the vehicle to
the dataset's lane, one lane further left while e_t exceeds the distance to the left marking (lat_pos_left_marking
clipped to [0, lane width]) and one further right while -e_t exceeds the distance to the right marking. A crossing is a
change of the assigned lane; with guard g it counts only once the new lane has held for g frames (hysteresis). Window
classes and "label known" follow from these crossings as in the site study; lane keeping is known only if the track
goes on for 26 + g - 1 frames, and labels are used 26 + g - 1 frames after the window (causal). Only the labels the
device learns from and the check compares on change; inputs and scoring use the dataset's values.

  sigma  0 / 0.1 / 0.2 / 0.3 / 0.5 m (one unit draw per seed, split and track, generator [seed, split, 17], scaled)
  g      1 (no guard, the current labeller) / 3 / 5 / 8
  devices, networks (highD arm K, seeds 100-104), update (eta 0.003, lambda 0), check (N = 4096): the site study's;
  every device runs through train, validation and test recordings; validation selects g, test is scored once.

  .venv\\Scripts\\python unas/online_guard.py stats            # label statistics of the error model, no network
  WSL: SEEDS=100 OUT_NAME=online_guard_s100.jsonl ~/dmir_nas/bin/python unas/online_guard.py run
  .venv\\Scripts\\python unas/online_guard.py --report
Output: datasets/exid/results/seeds/online_guard*.jsonl, datasets/exid/results/hil/online_guard*.json.
"""
from __future__ import annotations

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

ts, tsx = site.ts, site.tsx
RES = site.RESULTS
OUT = RES / "seeds" / os.environ.get("OUT_NAME", "online_guard.jsonl")
SUMMARY = RES / "hil" / "online_guard.json"
STATS = RES / "hil" / "online_guard_labels.json"
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.5)
GUARDS = (1, 3, 5, 8)
RHO = 0.8
CHECK_N = 4096
KEEP_NOISY, KEEP_CLEAN, EARLY = 0.75, 0.95, 0.05             # the criterion
SEEDS = [int(v) for v in os.environ.get("SEEDS", "100,101,102,103,104").split(",")]
SETTING = next(s for s in site.SETTINGS if s["id"] == "G_eta0.003_lam0.0")
SIDX = site.SETTINGS.index(SETTING)


# ------------------------------------------------------------------------------------------------- the error model
def unit_noise(lens: np.ndarray, rng) -> np.ndarray:
    """AR(1) noise with unit stationary variance, restarted at every track (one value per 5 Hz frame)."""
    eta = rng.standard_normal(int(lens.sum()))
    z = np.empty_like(eta)
    s = np.sqrt(1.0 - RHO ** 2)
    a = 0
    for n in lens:
        seg = eta[a:a + n]
        out = z[a:a + n]
        out[0] = seg[0]
        for i in range(1, n):
            out[i] = RHO * out[i - 1] + s * seg[i]
        a += n
    return z


def true_lane(lens: np.ndarray, cross: list) -> np.ndarray:
    """The dataset's lane of every frame as an index relative to the track's first lane (+1 per left crossing)."""
    lane = np.zeros(int(lens.sum()), np.int64)
    a = 0
    for n, cr in zip(lens, cross):
        for c, d in cr:
            lane[a + c:a + n] += 1 if d == 2 else -1
        a += n
    return lane


def detect(assigned: np.ndarray, g: int) -> list:
    """Crossings [frame, direction] of one track from its assigned lanes, with hysteresis g (direction 2 = left)."""
    change = np.flatnonzero(np.diff(assigned)) + 1
    if len(change) == 0:
        return []
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(assigned)]])
    vals, lens = assigned[starts], ends - starts
    stable = np.flatnonzero(lens >= g)
    if len(stable) == 0:
        return []
    conf = vals[stable[0]] if g > 1 else vals[0]
    out = []
    for j in range(len(starts)):
        if lens[j] >= g and vals[j] != conf:
            out.append([int(starts[j]), 2 if vals[j] > conf else 1])
            conf = vals[j]
    return out


def relabel(split: str, sigma: float, g: int, z: np.ndarray, base: dict) -> tuple:
    """Per split-local window: class and label known under the error model, plus the detected crossings per track."""
    lens, cross, dl, w, lane = base["lens"], base["cross"], base["dl"], base["w"], base["lane"]
    e = sigma * z
    assigned = lane + (e > dl).astype(np.int64) - (-e > (w - dl)).astype(np.int64)
    labels, known, det = [], [], []
    a = 0
    for n, cr in zip(lens, cross):
        crs = cr if sigma == 0 and g == 1 else detect(assigned[a:a + n], g)
        k, _, _, to_next, next_dir, _, _ = ts.phases(int(n), crs)
        labels.append(np.where(to_next <= ts.HORIZON, next_dir, 0))
        known.append((to_next <= ts.HORIZON) | ((n - 1) - k >= ts.HORIZON + g - 1))
        det.append(crs)
        a += n
    return np.concatenate(labels), np.concatenate(known), det


def split_base(split: str) -> dict:
    zf = np.load(tsx.tracks_file(split))
    lens = zf["lens"].astype(np.int64)
    cross = json.loads(str(zf["cross"]))
    w = zf["feats"][:, 17].astype(np.float64)
    dl = np.clip(zf["feats"][:, 4].astype(np.float64), 0.0, w)
    return {"lens": lens, "cross": cross, "w": w, "dl": dl, "lane": true_lane(lens, cross)}


def noisy_labels(seed: int, configs: list) -> dict:
    """For every (sigma, g) and split: the device's class (int8) and label known (bool) per split-local window."""
    lab = {cfg: {} for cfg in configs}
    for rank, split in enumerate(site.SPLITS):
        base = split_base(split)
        z = unit_noise(base["lens"], np.random.default_rng([seed, rank, 17]))
        for sigma, g in configs:
            lb, kn, _ = relabel(split, sigma, g, z, base)
            lab[(sigma, g)][rank] = (lb.astype(np.int8), kn.astype(bool))
    return lab


def with_labels(w: np.ndarray, lab_cfg: dict) -> np.ndarray:
    """A copy of one location's stream with the device's labels in columns 3 and 4."""
    wn = w.copy()
    for rank, (lb, kn) in lab_cfg.items():
        s = wn[:, 0] == rank
        wn[s, 3] = lb[wn[s, 1]]
        wn[s, 4] = kn[wn[s, 1]]
    return wn


def cfg_id(sigma: float, g: int) -> str:
    return f"s{sigma:g}_g{g}"


# ------------------------------------------------------------------------------------------------- label statistics
def stats() -> None:
    """Label statistics of the error model per split (seed 100's draw), no network."""
    res = {}
    for rank, split in enumerate(site.SPLITS):
        base = split_base(split)
        z = unit_noise(base["lens"], np.random.default_rng([100, rank, 17]))
        hours = base["lens"].sum() / 5 / 3600
        n_true = sum(len(c) for c in base["cross"])
        lab0, kn0, _ = relabel(split, 0.0, 1, z, base)
        res[split] = {"vehicle_hours": round(float(hours), 2), "true_crossings": n_true,
                      "label_known_windows": int(kn0.sum()), "lane_change_labels": int((kn0 & (lab0 > 0)).sum())}
        for sigma in SIGMAS:
            for g in GUARDS:
                lb, kn, det = relabel(split, sigma, g, z, base)
                both = kn0 & kn
                n_det = sum(len(c) for c in det)
                # a detected crossing matches a true one of the same direction within 5 frames
                matched = 0
                for cr, dc in zip(base["cross"], det):
                    used = set()
                    for c, d in dc:
                        hit = [j for j, (ct, dt) in enumerate(cr) if dt == d and abs(ct - c) <= 5 and j not in used]
                        if hit:
                            used.add(hit[0])
                            matched += 1
                res[split][cfg_id(sigma, g)] = {
                    "label_known": int(kn.sum()),
                    "changed_share_of_known": round(float((lb[both] != lab0[both]).mean()), 5),
                    "lane_keeping_to_lane_change_share": round(float(((lab0[both] == 0) & (lb[both] > 0)).mean()), 5),
                    "lane_change_labels": int((kn & (lb > 0)).sum()),
                    "lane_change_labels_wrong": int((both & (lb > 0) & (lb != lab0)).sum()),
                    "crossings_detected": n_det, "matched_to_true": matched,
                    "spurious_per_vehicle_hour": round((n_det - matched) / hours, 1),
                    "true_found_share": round(matched / max(n_true, 1), 4)}
        print(split, json.dumps({k: v for k, v in res[split].items() if k.startswith("s0.2") or k.startswith("s0.5")}),
              flush=True)
    STATS.write_text(json.dumps({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "rule": "local notes 2026-11-26 17:52", "seed": 100, "results": res}, indent=1))
    print("written", STATS)


# ------------------------------------------------------------------------------------------------- the runs
def records() -> list:
    rows = []
    for path in sorted(OUT.parent.glob("online_guard*.jsonl")):
        if "smoke" not in path.name:
            rows += [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def run() -> None:
    import input_scaling_check as isc
    import online_switch as osw

    keras = isc.keras
    configs = [(s, g) for s in SIGMAS for g in GUARDS]
    if os.environ.get("SMOKE"):
        configs = [(0.0, 1), (0.2, 5)]
    done = {(r["config"], r["seed"]) for r in records()}
    t0 = time.perf_counter()
    by_loc = site.windows_by_location()
    if os.environ.get("SMOKE"):
        by_loc = {min(by_loc, key=lambda k: len(by_loc[k])): by_loc[min(by_loc, key=lambda k: len(by_loc[k]))]}
    print(f"streams built in {time.perf_counter() - t0:.0f} s", flush=True)
    n_split = {sp: int((np.load(tsx.tracks_file(sp))["lens"] - ts.IN_LEN + 1).sum()) for sp in ("val", "test")}
    ranks = {sp: site.SPLITS.index(sp) for sp in ("val", "test")}
    keep, arm, prep = site.NETS["highD_K"]
    views = site.scaled_views(prep)
    for seed in SEEDS:
        todo = [c for c in configs if (cfg_id(*c), seed) not in done]
        if not todo:
            continue
        t1 = time.perf_counter()
        labs = noisy_labels(seed, todo)
        feat, w0, b0 = site.network(keras, keep, arm, seed)
        fixed = {sp: np.full(n, -1, np.int8) for sp, n in n_split.items()}
        outs = {cfg_id(*c): {sp: {"adapted": np.full(n, -1, np.int8), "check": np.full(n, -1, np.int8)}
                             for sp, n in n_split.items()} for c in todo}
        info = {cfg_id(*c): {} for c in todo}
        for loc, w in by_loc.items():
            f = site.features(feat, views, w)
            fx = np.argmax(f @ w0 + b0, axis=1).astype(np.int8)
            for sp, rk in ranks.items():
                s = w[:, 0] == rk
                fixed[sp][w[s, 1]] = fx[s]
            for sigma, g in todo:
                key = cfg_id(sigma, g)
                wn = with_labels(w, labs[(sigma, g)])
                site.OFFSET = (ts.HORIZON + g - 1) * 5            # labels used 26 + g - 1 frames later
                adapted, _ = site.adapt_site(f, wn, w0, b0, SETTING, np.random.default_rng([seed, loc, SIDX]))
                check, used = osw.switched(wn, fx, adapted, CHECK_N)
                site.OFFSET = ts.HORIZON * 5
                for sp, rk in ranks.items():
                    s = w[:, 0] == rk
                    outs[key][sp]["adapted"][w[s, 1]], outs[key][sp]["check"][w[s, 1]] = adapted[s], check[s]
                kn0, kn = w[:, 4].astype(bool), wn[:, 4].astype(bool)
                both = kn0 & kn
                info[key][str(loc)] = {"labels_known": int(kn.sum()), "labels_changed": int((wn[both, 3] != w[both, 3]).sum()),
                                       "share_of_frames_with_the_update": round(used, 4)}
            del f
        keras.backend.clear_session()
        partial = bool(os.environ.get("SMOKE"))
        assert partial or all((a >= 0).all() for a in fixed.values())
        fm = {sp: site.split_measures(sp, fixed[sp]) for sp in n_split} if not partial else {}
        for sigma, g in todo:
            key = cfg_id(sigma, g)
            rec = {"config": key, "sigma": sigma, "g": g, "seed": seed, "setting": SETTING["id"], "check_n": CHECK_N,
                   "info": info[key], "float32": True, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if partial:
                for sp in n_split:
                    m = fixed[sp] >= 0
                    rec[f"{sp}_windows_scored"] = int(m.sum())
                    rec[f"{sp}_agree_fixed_adapted"] = round(float((fixed[sp][m] == outs[key][sp]["adapted"][m]).mean()), 4)
            else:
                assert all((a >= 0).all() for d in outs[key].values() for a in d.values())
                for sp in n_split:
                    rec[sp] = {"fixed": fm[sp], "adapted": site.split_measures(sp, outs[key][sp]["adapted"]),
                               "check": site.split_measures(sp, outs[key][sp]["check"])}
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        print(f"seed {seed}: {len(todo)} configurations, {time.perf_counter() - t1:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------- the report
def report() -> None:
    rows = records()
    res = {}
    for sp in ("val", "test"):
        res[sp] = {}
        for sigma in SIGMAS:
            for g in GUARDS:
                key = cfg_id(sigma, g)
                got = [r for r in rows if r["config"] == key and sp in r]
                if len(got) < 5:
                    continue
                mean = lambda which, k: round(st.mean(r[sp][which]["all"][k] for r in got), 5)   # noqa: E731
                res[sp][key] = {k: {which: mean(which, k) for which in ("fixed", "adapted", "check")}
                                for k in ("windows_correct", "correct_2_s_before", "runs_0.6_s_per_vehicle_hour")}
                res[sp][key]["labels_changed_share"] = round(
                    sum(v["labels_changed"] for r in got for v in r["info"].values())
                    / sum(v["labels_known"] for r in got for v in r["info"].values()), 5)
    out = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rule": "local notes 2026-11-26 17:52",
           "results": res}
    full = all(cfg_id(s, g) in res["val"] and cfg_id(s, g) in res["test"] for s in SIGMAS for g in GUARDS)
    if full:
        def ok(sp, key):
            r = res[sp][key]
            return (r["correct_2_s_before"]["adapted"] >= r["correct_2_s_before"]["fixed"] - EARLY - 1e-12
                    and r["runs_0.6_s_per_vehicle_hour"]["adapted"] <= r["runs_0.6_s_per_vehicle_hour"]["fixed"] + 1e-9)
        cand = []
        for g in GUARDS[1:]:
            m = st.mean(res["val"][cfg_id(s, g)]["windows_correct"]["adapted"] for s in SIGMAS)
            cand.append({"g": g, "mean_windows_correct": round(m, 5), "qualifies": all(ok("val", cfg_id(s, g)) for s in SIGMAS)})
        pool = [c for c in cand if c["qualifies"]] or cand
        g_sel = max(pool, key=lambda c: c["mean_windows_correct"])["g"]
        out["selection"] = {"candidates": cand, "selected": g_sel, "qualified": any(c["qualifies"] for c in cand)}
        wc = lambda key: res["test"][key]["windows_correct"]                                     # noqa: E731
        gain = lambda key: wc(key)["adapted"] - wc(key)["fixed"]                                  # noqa: E731
        clean = gain(cfg_id(0.0, 1))
        harmful = [s for s in SIGMAS[1:] if gain(cfg_id(s, 1)) < KEEP_NOISY * clean]
        g1 = {str(s): {"raw_share": round(gain(cfg_id(s, 1)) / clean, 4), "guard_share": round(gain(cfg_id(s, g_sel)) / clean, 4),
                       "met": gain(cfg_id(s, g_sel)) >= KEEP_NOISY * clean - 1e-12 and ok("test", cfg_id(s, g_sel))}
              for s in harmful}
        g2_share = gain(cfg_id(0.0, g_sel)) / clean
        crit = {"clean_gain": round(clean, 5), "harmful_sigmas": harmful, "G1": g1,
                "G1_met": bool(harmful) and all(v["met"] for v in g1.values()),
                "G2_share": round(g2_share, 4), "G2_met": g2_share >= KEEP_CLEAN - 1e-12 and ok("test", cfg_id(0.0, g_sel)),
                "applies": bool(harmful)}
        crit["met"] = bool(crit["applies"] and crit["G1_met"] and crit["G2_met"])
        out["criterion"] = crit
    SUMMARY.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        stats()
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        run()
    else:
        raise SystemExit("usage: online_guard.py stats | run | --report")
