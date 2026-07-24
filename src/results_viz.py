"""Result visualizations: NAS Pareto fronts, quantization, measured deployment.

Every number in this module traces to a repo artifact — no synthetic values:

- ``results/nas-fronts/*.csv``  — harvested Pareto fronts, independently
  re-evaluated on the test split (unas/harvest_fronts.py, LOGBOOK 2026-07-09/10).
- ``results/deploy/*.tflite``   — the actual deployment artifacts; file sizes
  are read from disk at call time.
- ``MEASURED_H7B3`` / ``MEASURED_F401`` — on-device measurements from the
  ST Edge AI Developer Cloud (Core 4.0.1-20581, optimization "balanced",
  allocate inputs/outputs true), boards STM32H7B3I-DK (Cortex-M7 @ 280 MHz)
  and NUCLEO-F401RE (Cortex-M4 @ 84 MHz), measured 2026-07-13/14.
  Recorded in docs/research/deployment.md and LOGBOOK.md.
- ``QUANT_ACC`` — our own test-set evaluation of each .tflite artifact
  (docs/research/deployment.md, "Artifacts" table).

Chart style follows the dataviz tokens already used by the course site
(course/assets/course.css): series-1 blue, series-2 green, muted gray for the
reference/baseline, red reserved for does-not-fit status.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# --- dataviz tokens (course/assets/course.css) ------------------------------
BLUE = "#2a78d6"      # series-1: our models
GREEN = "#1baf7a"     # series-2: second series (ablation / LCL)
MUTED = "#898781"     # reference / baseline
INK = "#0b0b0b"
GRID = "#e1e0d9"
CRITICAL = "#d03b3b"  # status only: "does not fit"

ROOT = Path(__file__).resolve().parents[1]
FRONTS_DIR = ROOT / "results" / "nas-fronts"
DEPLOY_DIR = ROOT / "results" / "deploy"

# --- measured on-device numbers (ST Edge AI Developer Cloud, real boards) ---
# STM32H7B3I-DK, Cortex-M7 @ 280 MHz, 1184 KB RAM / 2048 KB flash.
# Source: docs/research/deployment.md "MEASURED" table (2026-07-13/14).
MEASURED_H7B3 = pd.DataFrame([
    # model,                who,   quality label,        lat_ms,  MACC,      flash_B,   ram_B
    ("REF_cnn_multi f32",   "ref", "acc 91.69%",          33.52,  1_965_360, 1_769_882, 39_168),
    ("cls_best f32",        "ours", "acc 92.08%",          3.628,   158_094,   343_254,  9_456),
    ("cls_tiny f32",        "ours", "acc 91.30%",          0.7931,   31_742,    37_954,  9_412),
    ("lcr_best f32",        "ours", "MAE 0.287 s",        14.06,     860_407,   474_522, 20_772),
    ("lcl_best f32",        "ours", "MAE 0.317 s",        28.77,   1_658_927,   423_494, 28_264),
    ("cls_best int8",       "ours", "acc 86.86%",          1.885,     158_336,   106_738,  8_096),
], columns=["model", "who", "quality", "latency_ms", "macc", "flash_B", "ram_B"])

# NUCLEO-F401RE, Cortex-M4 @ 84 MHz, 512 KB flash / 96 KB RAM (2026-07-14).
F401_FLASH_B = 512 * 1024
MEASURED_F401 = {"cls_tiny f32": 4.376}  # ms; REF does not fit (1,769,882 B = 3.38x flash)

# Our test-set evaluation of each deployment artifact
# (docs/research/deployment.md "Artifacts" table; higher acc / lower MAE better).
QUANT_ACC = pd.DataFrame([
    ("REF_cnn_multi", "acc %", 91.69, 88.27),
    ("cls_best",      "acc %", 92.08, 86.86),
    ("cls_tiny",      "acc %", 91.30, 85.46),
    ("cls_noind",     "acc %", 91.08, 76.06),
    ("lcr_best",      "MAE s", 0.2865, 0.4485),
    ("lcl_best",      "MAE s", 0.3165, 0.3440),
], columns=["model", "metric", "float32", "int8"])

# Reference points for the front plots (published + internal + measured ref).
PUBLISHED_TTLC_RMSE = 0.510   # Forneris et al., SPL 2026 — single combined TTLC
INTERNAL_RMSE = {"LCR": 0.42, "LCL": 0.44}   # internal reference transformers
REF_CNN_ACC = 91.69           # 441k reference CNN, our measured test accuracy


def _style(ax):
    ax.grid(True, color=GRID, alpha=0.6, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --- Pareto fronts -----------------------------------------------------------

def load_fronts(fronts_dir: Path = FRONTS_DIR) -> dict[str, pd.DataFrame]:
    """Load every harvested front CSV, keyed by search name."""
    return {p.stem.removeprefix("dmir_"): pd.read_csv(p)
            for p in sorted(fronts_dir.glob("*.csv"))}


def front_summary(fronts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per search: how many Pareto models and the best/smallest points."""
    rows = []
    for name, df in fronts.items():
        if "test_acc" in df:
            best = df.loc[df.test_acc.idxmax()]
            small = df.loc[df.int8_KB.idxmin()]
            rows.append((name, len(df), f"{best.test_acc:.2%} @ {best.int8_KB:.0f} KB",
                         f"{small.test_acc:.2%} @ {small.int8_KB:.1f} KB"))
        else:
            best = df.loc[df.test_rmse.idxmin()]
            small = df.loc[df.int8_KB.idxmin()]
            rows.append((name, len(df), f"RMSE {best.test_rmse:.3f} @ {best.int8_KB:.0f} KB",
                         f"RMSE {small.test_rmse:.3f} @ {small.int8_KB:.1f} KB"))
    return pd.DataFrame(rows, columns=["search", "models on front",
                                       "best metric", "smallest model"])


def _nondominated(df: pd.DataFrame, ycol: str, maximize: bool) -> pd.DataFrame:
    """Points not beaten on both axes (min size, best metric): the true frontier.

    The saved fronts are Pareto-optimal *at save time*; later candidates can
    dominate earlier ones, so the CSV also contains dominated points."""
    df = df.sort_values("int8_KB")
    best = None
    keep = []
    for _, r in df.iterrows():
        v = r[ycol]
        if best is None or (v > best if maximize else v < best):
            best = v
            keep.append(r)
    return pd.DataFrame(keep)


def plot_classification_front(fronts: dict[str, pd.DataFrame]):
    """Accuracy vs int8 size for the two classification searches."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for key, label, color in (("cls", "with turn indicators", BLUE),
                              ("cls_noind", "no indicators (ablation)", GREEN)):
        df = fronts[key].sort_values("int8_KB")
        ax.scatter(df.int8_KB, df.test_acc * 100, color=color, s=28, alpha=0.45,
                   label=None)
        nd = _nondominated(df, "test_acc", maximize=True)
        ax.plot(nd.int8_KB, nd.test_acc * 100, color=color, linewidth=2,
                drawstyle="steps-post", label=label)
    ax.axhline(REF_CNN_ACC, color=MUTED, linewidth=1.5, linestyle="--")
    ax.annotate(f"reference CNN, 441 k params ({REF_CNN_ACC}%)",
                xy=(0.02, REF_CNN_ACC), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=9, color=INK)
    cls = fronts["cls"]
    for name, marker_label, offset in (("model_aaaabl.h5", "cls_best", (-74, -28)),
                                       ("model_aaaaat.h5", "cls_tiny", (8, -24))):
        row = cls[cls.model == name].iloc[0]
        ax.scatter([row.int8_KB], [row.test_acc * 100], color=BLUE, s=70, zorder=3)
        ax.annotate(f"{marker_label}\n{row.test_acc:.2%} @ {row.int8_KB:.0f} KB",
                    xy=(row.int8_KB, row.test_acc * 100), xytext=offset,
                    textcoords="offset points", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("model size, int8 [KB, log]")
    ax.set_ylabel("test accuracy [%]")
    ax.set_title("Classification Pareto fronts — every dot is a searched architecture")
    ax.legend(loc="lower right", frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def plot_regression_fronts(fronts: dict[str, pd.DataFrame]):
    """Test RMSE vs int8 size for both directions (MAE- and RMSE-objective runs
    combined per direction — all are real searched models)."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for keys, label, color in ((("lcr", "lcr_rmse"), "LCR", BLUE),
                               (("lcl", "lcl_rmse"), "LCL", GREEN)):
        df = pd.concat([fronts[k] for k in keys]).sort_values("int8_KB")
        ax.plot(df.int8_KB, df.test_rmse, "o", color=color, label=label,
                markersize=5, alpha=0.75)
    ax.axhline(PUBLISHED_TTLC_RMSE, color=MUTED, linewidth=1.5, linestyle="--")
    ax.annotate("published SOTA (single TTLC): 0.510 s",
                xy=(0.98, PUBLISHED_TTLC_RMSE), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=9, color=INK)
    for direction, y, color in (("LCR", INTERNAL_RMSE["LCR"], BLUE),
                                ("LCL", INTERNAL_RMSE["LCL"], GREEN)):
        ax.axhline(y, color=color, linewidth=1, linestyle=":")
        ax.annotate(f"internal ref {direction}: {y:.2f} s",
                    xy=(0.02, y), xycoords=("axes fraction", "data"),
                    ha="left", va="top", fontsize=8.5, color=INK)
    best = {"LCR": ("lcr", "test_rmse"), "LCL": ("lcl_rmse", "test_rmse")}
    for direction, (key, col) in best.items():
        row = fronts[key].loc[fronts[key][col].idxmin()]
        ax.annotate(f"{direction} best: {row.test_rmse:.3f} s",
                    xy=(row.int8_KB, row.test_rmse), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("model size, int8 [KB, log]")
    ax.set_ylabel("test RMSE [s]")
    ax.set_title("Regression Pareto fronts — beats the published record, not yet the internal one")
    ax.legend(loc="upper right", frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


# --- quantization ------------------------------------------------------------

def tflite_sizes() -> pd.DataFrame:
    """Actual on-disk sizes of every deployment artifact (KB)."""
    rows = []
    for f in sorted(DEPLOY_DIR.glob("*.tflite")):
        base, _, variant = f.stem.rpartition("_")
        rows.append((base, variant, f.stat().st_size / 1024))
    df = pd.DataFrame(rows, columns=["model", "variant", "KB"])
    return (df.pivot(index="model", columns="variant", values="KB")
              .loc[:, ["float32", "int16x8", "int8"]].round(1))


def plot_quantization_effect():
    """float32 vs int8 quality per artifact — int8 costs real accuracy here."""
    acc = QUANT_ACC[QUANT_ACC.metric == "acc %"]
    mae = QUANT_ACC[QUANT_ACC.metric == "MAE s"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6),
                                   gridspec_kw={"width_ratios": [4, 2]})
    for ax, df, unit in ((ax1, acc, "test accuracy [%]"), (ax2, mae, "test MAE [s]")):
        y = range(len(df))
        for yi, (_, r) in zip(y, df.iterrows()):
            ax.plot([r.float32, r.int8], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.scatter(df.float32, y, color=BLUE, s=55, zorder=2, label="float32")
        ax.scatter(df.int8, y, color=MUTED, s=55, zorder=2, label="int8 PTQ")
        ax.set_yticks(list(y), df.model)
        ax.set_xlabel(unit)
        _style(ax)
    ax1.invert_yaxis(); ax2.invert_yaxis()
    ax1.legend(loc="center left", frameon=False, fontsize=9)
    fig.suptitle("Full-int8 PTQ costs real quality on these wide-dynamic-range inputs "
                 "(int16x8 would fix it, but ST silently dequantizes it — measured)",
                 fontsize=10)
    fig.tight_layout()
    return fig


# --- measured deployment ------------------------------------------------------

def plot_measured_deployment():
    """Measured latency and flash on the STM32H7B3I-DK (ST Edge AI, real board)."""
    df = MEASURED_H7B3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = [MUTED if w == "ref" else BLUE for w in df.who]
    y = range(len(df))
    ax1.barh(y, df.latency_ms, color=colors, height=0.62)
    for yi, v in zip(y, df.latency_ms):
        ax1.annotate(f"{v:g} ms", xy=(v, yi), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=9, color=INK)
    ax1.set_xscale("log")
    ax1.set_xlabel("inference latency [ms, log] — lower is better")
    ax2.barh(y, df.flash_B / 1024, color=colors, height=0.62)
    ax2.axvline(2048, color=CRITICAL, linewidth=1.2, linestyle="--")
    ax2.annotate("H7B3 flash (2 MB)", xy=(2048, 0.02), xycoords=("data", "axes fraction"),
                 rotation=90, fontsize=8.5, va="bottom", ha="right", color=CRITICAL)
    ax2.set_xlabel("flash [KB]")
    for ax in (ax1, ax2):
        ax.set_yticks(list(y), df.model)
        ax.invert_yaxis()
        _style(ax)
    fig.suptitle("Measured on the STM32H7B3I-DK — Cortex-M7 @ 280 MHz, ST Edge AI Core 4.0.1",
                 fontsize=10.5)
    fig.tight_layout()
    return fig


def plot_f401_fit():
    """The categorical result: the reference CNN cannot run on the F401 at all."""
    rows = [("REF_cnn_multi", 1_769_882), ("lcr_best", 474_522),
            ("lcl_best", 423_494), ("cls_best", 343_254), ("cls_tiny", 37_954)]
    names = [r[0] for r in rows]
    pct = [r[1] / F401_FLASH_B * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    colors = [CRITICAL if p > 100 else BLUE for p in pct]
    ax.barh(range(len(rows)), pct, color=colors, height=0.62)
    ax.axvline(100, color=INK, linewidth=1.4)
    ax.annotate("100% = F401's entire 512 KB flash", xy=(100, 0.03),
                xycoords=("data", "axes fraction"), rotation=90,
                ha="right", va="bottom", fontsize=8.5, color=INK)
    for yi, p in enumerate(pct):
        label = f"{p:.1f}%" + ("  — does not fit" if p > 100 else "")
        ax.annotate(label, xy=(min(p, 330), yi), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), names)
    ax.invert_yaxis()
    ax.set_xlabel("flash needed, % of NUCLEO-F401RE (Cortex-M4)")
    ax.set_title("Smallest-board check: every searched model fits; the reference cannot "
                 f"(cls_tiny measured there: {MEASURED_F401['cls_tiny f32']} ms)")
    _style(ax)
    fig.tight_layout()
    return fig


# --- headline ------------------------------------------------------------------

def records_table() -> pd.DataFrame:
    """Final scoreboard — mirrors the course and paper tables (all traceable)."""
    return pd.DataFrame([
        ("Classification", "92.08% @ 84k params", "92% @ 441k (internal)",
         "beaten with 5x fewer params"),
        ("Classification tiny", "91.30% @ 8k params", "—",
         "55x smaller than the reference CNN"),
        ("LCR vs published record", "MAE 0.287 / RMSE 0.447", "MAE 0.298 / RMSE 0.510",
         "record beaten"),
        ("LCR/LCL vs internal ref", "RMSE 0.447 / 0.466", "RMSE 0.42 / 0.44",
         "not yet — reported honestly"),
        ("No-indicator ablation", "91.08%", "—",
         "the blinker is worth only ~1 point"),
        ("On-device (H7B3I-DK)", "3.63 ms / 335 KiB (cls_best f32)",
         "33.52 ms / 1.69 MiB (ref CNN)", "9.2x faster, 5.2x smaller, measured"),
        ("Smallest board (F401RE)", "cls_tiny 4.376 ms, 7.2% flash",
         "reference needs 3.38x the flash", "categorical: ref cannot run there"),
    ], columns=["problem", "ours (measured/verified)", "reference", "status"])
