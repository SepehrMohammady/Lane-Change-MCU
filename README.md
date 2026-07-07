# DIMIR × µNAS — Lane-Change Intention Prediction under MCU Budgets

PhD research project (ELIOS Lab, University of Genoa, SYNERGIES project).
Constrained neural architecture search on the DIMIR driving time-series
dataset; goal: beat the published LC-Intention baseline on all three tasks
with models small enough for STM32 deployment, and publish in a Q1 venue.

## Tasks and targets

Input: windows of 50 timesteps × 31 features (prepared and normalised by the
data provider). Published baseline to beat
([LC-Intention framework](https://elios-lab.github.io/LC-Intention-Framework/),
[IEEE 11271346](https://ieeexplore.ieee.org/document/11271346)):

| Task | Metric | Baseline | Ours (first untuned DSCNN) |
|---|---|---|---|
| 3-class intention (none/LCR/LCL) | accuracy | 92% | 91.5% |
| Time-to-LC regression, LCR | RMSE | 0.42 s | 0.439 s |
| Time-to-LC regression, LCL | RMSE | 0.44 s | 0.459 s |

## Repository layout

```
notebooks/dimir_pipeline.ipynb   main pipeline — all knobs in its Config cell
src/                             importable logic (data, models, train, eda, logging)
scripts/check_pipeline.py        3-task smoke test on real data; run after every change
logs/experiments.jsonl           one JSON line per run (feeds the paper's tables)
docs/DATA.md                     dataset facts, quirks, open questions
docs/research/                   literature and toolchain notes
paper/                           LaTeX draft (IEEE journal format) + notes + style guard
course/                          Farsi step-by-step course following the project
LOGBOOK.md                       dated journal of decisions and results
```

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

Data: extract `Materials/data-*.zip` into `data/` (folders
`data-classification/`, `data-regression-lcl/`, `data-prepared-lcr/`).
The archives are not part of this repository.

## Working rules

1. Every experiment goes through the notebook or scripts — never untracked
   one-offs; every run appends to `logs/experiments.jsonl`.
2. After any change to `src/`: `python scripts/check_pipeline.py` must pass.
3. Paper numbers only from logged runs or cited sources; no placeholder data
   anywhere in the pipeline.
4. `LOGBOOK.md` records decisions; `course/` is updated as milestones land.
