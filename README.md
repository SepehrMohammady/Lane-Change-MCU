# LC-Intention-NAS — Lane-Change Intention Prediction under MCU Budgets

PhD research project (ELIOS Lab, University of Genoa, SYNERGIES project).
Constrained neural architecture search on the Lane Change Intention
Recognition driving time-series dataset (codename DMIR); goal: beat
the published LC-Intention baseline with models small enough for deployment
on the **STM32H7B3I-DK**, and publish in a Q1 venue. The H7B3I-DK was chosen
because the published baseline deployed on the same platform — making the
on-device comparison like-for-like — and the board is physically available
in the ELIOS lab.

📚 A trilingual course website (Farsi/English/Italian) documenting this project
A-to-Z, and the LaTeX manuscript, are kept **local only** and are not published
in this repository.

## Tasks and targets

Input: windows of 50 timesteps × 31 features (prepared and normalised by the
data provider). Published baseline
([LC-Intention framework](https://elios-lab.github.io/LC-Intention-Framework/),
[IEEE 11271346](https://ieeexplore.ieee.org/document/11271346)). Final NAS
results (test-set evaluation of the searched models; details in
`datasets/dmir/docs/nas-results.md`):

| Task | Metric | Published SOTA¹ | Internal ref.² | Ours (NAS) |
|---|---|---|---|---|
| 3-class intention (none/LCR/LCL) | accuracy | — | 92% @ 441 k params | **92.08% @ 84 k** (5× smaller) |
| — tiny variant | accuracy | — | — | 91.30% @ ~8 k (55× smaller) |
| — no-turn-indicator ablation | accuracy | — | — | 91.1% |
| Time-to-LC regression, LCR | RMSE / MAE (s) | 0.510 / 0.298 | 0.42 / — | **0.447 / 0.287** |
| Time-to-LC regression, LCL | RMSE / MAE (s) | 0.510 / 0.298¹ | 0.44 / — | 0.466 / 0.317 |

¹ Published SOTA (Forneris et al., SPL 2026, Transformer) reports a *single*
combined TTLC, not per-direction: LCR beats it on both RMSE and MAE; for LCL
our RMSE 0.466 is below 0.510 but the comparison is directionally valid
rather than strict, and LCL MAE 0.317 does not beat 0.298 (LCL is the harder
direction).
² Internal unpublished reference uses a different train/threshold protocol
(soft comparison); its RMSE 0.42 (LCR) / 0.44 (LCL) is **not yet beaten**.

## Measured on-device (ST Edge AI Developer Cloud, Core 4.0.1)

Float32, optimization *balanced*, board **STM32H7B3I-DK** (Cortex-M7 @
280 MHz); full tables and analysis in `datasets/dmir/docs/deployment.md`:

| Model | quality | latency | flash | RAM |
|---|--:|--:|--:|--:|
| reference CNN (441 k) | 91.69% | 33.52 ms | 1,769,882 B | 39,168 B |
| cls_best (84 k) | **92.08%** | 3.628 ms | 343,254 B | 9,456 B |
| cls_tiny (8 k) | 91.30% | 0.793 ms | 37,954 B | 9,412 B |
| lcr_best (117 k) | MAE 0.287 s | 14.06 ms | 474,522 B | 20,772 B |
| lcl_best (106 k) | MAE 0.317 s | 28.77 ms | 423,494 B | 28,264 B |

On the low-end **NUCLEO-F401RE** (Cortex-M4 @ 84 MHz, 512 KB flash) the
reference CNN needs 3.38× the board's entire flash and cannot run at all,
while every one of our architectures does: cls_tiny 4.376 ms (7.2% of flash),
cls_best int8-QAT 7.381 ms (25.0%), the full float32 cls_best — the 92.08%
headline model — 18.35 ms (65.5%), and lcl_best 162.5 ms (80.8%). The one
build that returned no measured time was `lcr_best` in float32 (90.5% of
flash, under 50 KB left for the runtime) — yet the **int8 build of that same
network runs in 28.10 ms** at 28.7% of flash. Same architecture, same board,
only the numeric format differs, so the limit here is flash *headroom* for the
runtime rather than model size or supported operators: on this board
quantization is what makes the widest model deployable at all.

**Quantization.** Full-int8 PTQ costs accuracy on these wide-dynamic-range
inputs (cls 92.08 → 86.86%); **quantization-aware training recovers it to
89.82%**, measured at **1.558 ms** on the H7B3I-DK (the fastest operating point)
and 7.381 ms on the F401RE. int16×8 preserves accuracy offline but ST Edge AI
silently dequantizes it, so it is not deployable. See `unas/qat_finetune.py` and
`datasets/dmir/docs/deployment.md`.

## Repository layout

```
src/                             shared, dataset-agnostic logic (train, logging, env, EDA)
unas/                            µNAS fork adapters, search launchers, export/quantization tools
notebooks/dmir_pipeline.ipynb    main DMIR pipeline — all knobs in its Config cell
scripts/check_pipeline.py        3-task smoke test on real data; run after every change
docs/research/                   shared literature and toolchain notes (µNAS, ST Edge AI, venue)
LOGBOOK.md                       dated journal of decisions and results (all datasets)

datasets/dmir/                   everything specific to the DMIR dataset
  ├── data/                      prepared pickles (gitignored)
  ├── docs/                      DATA.md + dataset/results/deployment notes
  ├── logs/experiments.jsonl     one JSON line per run (feeds the paper's tables)
  └── results/                   nas-fronts/, deploy/, qat/ artifacts
datasets/highd/                  second dataset: highD lane-change prediction
  (same shape; data gitignored — highD licence forbids redistribution)
```

A second dataset gets its own `datasets/<name>/` with the same shape; shared
code stays in `src/` and `unas/` rather than being copied per dataset.

Two directories exist locally but are gitignored and not published here:
`paper/` (LaTeX manuscript, built with `scripts/build_paper.ps1`) and `course/`
(trilingual course website).

## Setup (Windows, Python 3.13)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

Note: on machines with Windows Smart App Control enabled, some compiled wheels
(numpy.random, pandas, scikit-learn) may be blocked inside a venv. Workaround
used here: set `include-system-site-packages = true` in `.venv/pyvenv.cfg`,
uninstall those packages from the venv, and install them into the (Microsoft
Store) system Python instead.

Data: extract `Materials/data-*.zip` into `datasets/dmir/data/` (folders
`data-classification/`, `data-regression-lcl/`, `data-prepared-lcr/`).
The archives are not part of this repository.

## Working rules

1. Every experiment goes through the notebook or scripts — never untracked
   one-offs; every run appends to `datasets/dmir/logs/experiments.jsonl`.
2. After any change to `src/`: `python scripts/check_pipeline.py` must pass.
3. Paper numbers only from logged runs or cited sources; no placeholder data
   anywhere in the pipeline.
4. `LOGBOOK.md` records decisions; the local `course/` and `paper/` are updated
   as milestones land, but are not published in this repository.
