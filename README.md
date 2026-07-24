# LC-Intention-NAS — Lane-Change Intention Prediction under MCU Budgets

PhD research project (ELIOS Lab, University of Genoa, SYNERGIES project).
Constrained neural architecture search on the Lane Change Intention
Recognition driving time-series dataset (codename DMIR); goal: beat
the published LC-Intention baseline with models small enough for deployment
on the **STM32H7B3I-DK**, and publish in a Q1 venue. The H7B3I-DK was chosen
because the published baseline deployed on the same platform — making the
on-device comparison like-for-like — and the board is physically available
in the ELIOS lab.

📚 Course website (A-to-Z of this project), being extended to three
languages: Farsi at [`course/`](course/), English at `course/en/`, Italian
at `course/it/`. Enable GitHub Pages (Settings → Pages → main branch, root)
or open `course/index.html` locally.

## Tasks and targets

Input: windows of 50 timesteps × 31 features (prepared and normalised by the
data provider). Published baseline
([LC-Intention framework](https://elios-lab.github.io/LC-Intention-Framework/),
[IEEE 11271346](https://ieeexplore.ieee.org/document/11271346)). Final NAS
results (test-set evaluation of the searched models; details in
`docs/research/nas-results.md`):

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
280 MHz); full tables and analysis in `docs/research/deployment.md`:

| Model | quality | latency | flash | RAM |
|---|--:|--:|--:|--:|
| reference CNN (441 k) | 91.69% | 33.52 ms | 1,769,882 B | 39,168 B |
| cls_best (84 k) | **92.08%** | 3.628 ms | 343,254 B | 9,456 B |
| cls_tiny (8 k) | 91.30% | 0.793 ms | 37,954 B | 9,412 B |
| lcr_best (117 k) | MAE 0.287 s | 14.06 ms | 474,522 B | 20,772 B |
| lcl_best (106 k) | MAE 0.317 s | 28.77 ms | 423,494 B | 28,264 B |

On the low-end **NUCLEO-F401RE** (Cortex-M4 @ 84 MHz, 512 KB flash),
cls_tiny runs at 4.376 ms in 7.2% of the flash, while the reference CNN
needs 3.38× the board's entire flash and cannot run on it at all.

## Repository layout

```
notebooks/dmir_pipeline.ipynb   main pipeline — all knobs in its Config cell
src/                             importable logic (data, models, train, eda, logging)
scripts/check_pipeline.py        3-task smoke test on real data; run after every change
logs/experiments.jsonl           one JSON line per run (feeds the paper's tables)
docs/DATA.md                     dataset facts, quirks, open questions
docs/research/                   literature and toolchain notes
paper/                           LaTeX draft + main.pdf (rebuilt via scripts/build_paper.ps1) + style guard
course/                          trilingual course website (Farsi at root, en/, it/)
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
