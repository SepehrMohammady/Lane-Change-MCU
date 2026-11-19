"""Paired statistics for on-device learning at exiD sites (analysis added 2026-11-19 after the results; no new claim).

Reads the committed records only: the site pilot's final records (fixed and the selected update, five seeds), the
comparison's final records and, for the location of each test recording, the board's per-location records. Per test
recording the measures are averaged over the five seeds; differences are then paired by recording (n = 14) and, as the
more conservative unit, by location (the mean over a location's test recordings, n = 7): mean difference, bootstrap
95% interval (10,000 resamples of the units, generator seed 0), units with a gain, and the exact Wilcoxon signed-rank
test (two-sided). The reverse direction (highD test recordings) is added when its records exist.

    .venv\\Scripts\\python unas/online_stats.py
Output: datasets/exid/results/hil/online_stats.json
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
EXID = ROOT / "datasets" / "exid" / "results"
HIGHD = ROOT / "datasets" / "highd" / "results"
OUT = EXID / "hil" / "online_stats.json"
SETTING = "G_eta0.003_lam0.0"
MEASURES = ("windows_correct", "correct_2_s_before", "runs_0.6_s_per_vehicle_hour")
METHODS = ("site_scaling", "entropy", "self_training", "naive", "rls", "instant", "offline_refit")


def rows(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(pattern))):
        out += [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
    return out


def seed_mean(per_seed: list[dict]) -> dict:
    """{recording: {measure: mean over seeds}} from a list of per_recording dicts (one per seed)."""
    recs = sorted(per_seed[0], key=int)
    assert all(sorted(p, key=int) == recs for p in per_seed)
    return {r: {m: float(np.mean([p[r][m] for p in per_seed])) for m in MEASURES} for r in recs}


def paired(a: dict, b: dict, measure: str, groups: dict | None = None) -> dict:
    """b - a per unit (recordings, or locations when groups maps recording -> location)."""
    keys = sorted(a, key=int)
    d = np.array([b[k][measure] - a[k][measure] for k in keys])
    if groups:
        locs = sorted({groups[k] for k in keys})
        d = np.array([np.mean([b[k][measure] - a[k][measure] for k in keys if groups[k] == g]) for g in locs])
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
    res = {"n": int(len(d)), "mean": round(float(d.mean()), 5), "ci95": [round(float(np.percentile(boot, 2.5)), 5),
           round(float(np.percentile(boot, 97.5)), 5)], "units_with_a_gain": int((d > 0).sum()),
           "min": round(float(d.min()), 5), "max": round(float(d.max()), 5)}
    if np.any(d != 0):
        res["wilcoxon_two_sided_p"] = float(f"{wilcoxon(d, method='exact').pvalue:.3g}")
    return res


def block(a: dict, b: dict, groups: dict) -> dict:
    return {m: {"by_recording": paired(a, b, m), "by_location": paired(a, b, m, groups)} for m in MEASURES}


def main() -> None:
    board = rows(EXID / "hil" / "online_board.jsonl")
    groups = {str(r): str(x["location"]) for x in board if x["kind"] == "replay_measures" for r in x["recordings"]}
    site = [r for r in rows(EXID / "seeds" / "online_site_s*.jsonl") if r["phase"] == "final" and r["setting"] == SETTING]
    out = {"note": "analysis added after the results (2026-11-19); differences are the second minus the first, in "
                   "fractions (runs per vehicle hour as they are); seeds averaged per recording first",
           "recording_location": groups}
    by_net = defaultdict(list)
    for r in site:
        by_net[r["net"]].append(r)
    fixed, ours = {}, {}
    for net, rs in by_net.items():
        assert sorted(r["seed"] for r in rs) == [100, 101, 102, 103, 104], net
        fixed[net] = seed_mean([r["fixed"]["per_recording"] for r in rs])
        ours[net] = seed_mean([r["adapted"]["per_recording"] for r in rs])
    assert set(fixed["highD_K"]) == set(groups), "every test recording needs its location from the board records"
    out["site_highD_K_fixed_to_update"] = block(fixed["highD_K"], ours["highD_K"], groups)
    out["site_highD_P_fixed_to_update"] = block(fixed["highD_P"], ours["highD_P"], groups)
    out["own_exiD_K_fixed_to_update"] = block(fixed["exiD_K"], ours["exiD_K"], groups)
    out["highD_K_update_to_exiD_K_fixed"] = block(ours["highD_K"], fixed["exiD_K"], groups)
    comp = [r for r in rows(EXID / "seeds" / "online_compare_s*.jsonl") if r["phase"] == "final"]
    out["comparison_method_minus_ours"] = {}
    for m in METHODS:
        rs = [r for r in comp if r["method"] == m]
        assert sorted(r["seed"] for r in rs) == [100, 101, 102, 103, 104], m
        out["comparison_method_minus_ours"][m] = block(ours["highD_K"], seed_mean([r["measures"]["per_recording"] for r in rs]), groups)
    rev = rows(HIGHD / "seeds" / "online_reverse_s*.jsonl")
    if rev:
        out["reverse"] = {}
        for net in sorted({r["net"] for r in rev}):
            rs = [r for r in rev if r["net"] == net]
            f = seed_mean([r["fixed"]["per_recording"] for r in rs])
            a = seed_mean([r["adapted"]["per_recording"] for r in rs])
            out["reverse"][f"{net}_fixed_to_update"] = {m: {"by_recording": paired(f, a, m)} for m in MEASURES}
    OUT.write_text(json.dumps(out, indent=1))
    s = out["site_highD_K_fixed_to_update"]["windows_correct"]
    print("site, highD K, windows correct:", json.dumps(s))
    for m in METHODS:
        x = out["comparison_method_minus_ours"][m]
        print(f"{m:>14} - ours: windows {json.dumps(x['windows_correct']['by_recording'])}")
        print(f"{'':>14}        2 s before {json.dumps(x['correct_2_s_before']['by_recording'])}")
    print("written", OUT)


if __name__ == "__main__":
    main()
