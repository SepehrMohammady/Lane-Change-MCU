# Lane-Change-MCU

Lane-change prediction on STM32 microcontrollers. Searched (µNAS) and hand-designed
1D CNNs for driver intention on the Lane Change Intention Recognition dataset
(LCIR; `dmir` is the internal codename used in folder names) and for lane changes
of surrounding vehicles on motorway sections (highD) and at motorway entries and
exits (exiD). Every final model is retrained with five seeds, and latency, flash
and RAM are measured on a Cortex-M7 (STM32H7B3I-DK) and a Cortex-M4 (NUCLEO-F401RE)
board.

PhD research project (ELIOS Lab, University of Genoa, SYNERGIES project). The
H7B3I-DK was chosen because the published LCIR baseline deployed on the same
platform, which makes the on-device comparison like-for-like, and because the
board is available in the ELIOS lab. The repository was called LC-Intention-NAS
until September 2026; GitHub redirects the old address.

A trilingual course website (Farsi/English/Italian) documenting this project
A-to-Z, and the LaTeX manuscript, are kept **local only** and are not published
in this repository.

## Tasks and targets

Input: windows of 50 timesteps × 31 features (prepared and normalised by the
data provider). Published baseline
([LC-Intention framework](https://elios-lab.github.io/LC-Intention-Framework/),
[IEEE 11271346](https://ieeexplore.ieee.org/document/11271346)). Final NAS
results (test-set evaluation of the searched models; details in
`datasets/dmir/docs/nas-results.md`):

| Task | Metric | Published SOTA¹ | Internal ref.² | Ours, search run | Ours, retrained 5×³ |
|---|---|---|---|---|---|
| 3-class intention (none/LCR/LCL) | accuracy | — | 92% @ 441 k params | 92.08% @ 84 k (5× smaller) | 91.45 ± 0.59% (reference CNN, same recipe: 91.33 ± 0.50%) |
| Intention, tiny variant | accuracy | — | — | 91.30% @ ~8 k (55× smaller) | 91.19 ± 0.32% |
| Intention without turn signal | accuracy | — | — | 91.1% | 89.99 ± 0.66% |
| Time-to-LC regression, LCR | RMSE / MAE (s) | 0.510 / 0.298 | 0.42 / — | 0.447 / 0.287 | 0.483 ± 0.021 / 0.326 ± 0.016 |
| Time-to-LC regression, LCL | RMSE / MAE (s) | 0.510 / 0.298¹ | 0.44 / — | 0.466 / 0.317 | 0.496 ± 0.012 / 0.351 ± 0.013 |
| Hand-designed DSCNN (10 k), intention | accuracy | — | — | 91.51% | 91.50 ± 0.47% |
| Hand-designed DSCNN (10 k), LCR / LCL | RMSE (s) | 0.510 | 0.42 / 0.44 | 0.439 / 0.459 | 0.454 ± 0.008 / 0.469 ± 0.010 |

¹ Published SOTA (Forneris et al., SPL 2026, Transformer) reports a *single*
combined TTLC, not per-direction, so the comparison is directional rather than
strict. Retrained 5×, both searched RMSE means stay below 0.510 (one LCR seed,
0.518, does not), every DSCNN run is below it, and no MAE mean beats 0.298.
² Internal unpublished reference uses a different train/threshold protocol
(soft comparison); its RMSE 0.42 (LCR) / 0.44 (LCL) is **not yet beaten**.
³ Same architecture re-initialised and retrained with five seeds
(`unas/seed_variance.py`, `scripts/run_baseline.py <task> <seed>`; records in
`datasets/dmir/results/seeds/` and `datasets/dmir/logs/experiments.jsonl`). The
search trains each candidate once and keeps the best, so its single runs lean
optimistic; quote the seed means. Under identical training the searched classifiers
match the 441 k reference rather than beat it, and the hand-designed 10 k DSCNN
matches them on intention and has lower mean RMSE on both regression tasks, also
when the searched models are trained with the DSCNN's recipe (details and tests in
`datasets/dmir/docs/nas-results.md`). Latency, flash and RAM below depend only on
the graph and are unaffected.

## Measured on-device (ST Edge AI Developer Cloud, Core 4.0.1)

Float32, optimization *balanced*, board **STM32H7B3I-DK** (Cortex-M7 @
280 MHz); full tables and analysis in `datasets/dmir/docs/deployment.md`:

| Model | quality (5-seed mean) | latency | flash | RAM |
|---|--:|--:|--:|--:|
| reference CNN (441 k) | 91.33% | 33.52 ms | 1,769,882 B | 39,168 B |
| cls_best (84 k) | 91.45% | 3.628 ms | 343,254 B | 9,456 B |
| cls_tiny (8 k) | 91.19% | 0.793 ms | 37,954 B | 9,412 B |
| hand-designed DSCNN (10.5 k) | 91.50% | 3.272 ms | 51,382 B | 11,632 B |
| lcr_best (117 k) | MAE 0.326 s | 14.06 ms | 474,522 B | 20,772 B |
| lcl_best (106 k) | MAE 0.351 s | 28.77 ms | 423,494 B | 28,264 B |
| hand-designed DSCNN, LCR and LCL (10.3 k) | MAE 0.324 / 0.338 s | 3.265 ms | 50,862 B | 11,632 B |

At the same five-seed accuracy on intention, the searched 8 k model runs 4.1 times
faster than the hand-designed DSCNN, whose strided first convolution over all 31
channels needs most of its 171,971 MACs. For time to lane change the hand-designed
network is both more accurate and faster than the searched regressors.

On the low-end **NUCLEO-F401RE** (STM32F401RE, Cortex-M4 @ 84 MHz, 512 KB flash)
the reference CNN needs 3.38× the board's entire flash and cannot run at all,
while every one of our architectures does: cls_tiny 4.376 ms (7.2% of flash),
cls_best int8-QAT 7.381 ms (25.0%), the full float32 cls_best 18.35 ms (65.5%),
and lcl_best 162.5 ms (80.8%). The one build that returned no measured time was
`lcr_best` in float32 (90.5% of flash, under 50 KB left for the runtime), yet the
**int8 build of that same network runs in 28.10 ms** at 28.7% of flash. Same
architecture, same board, only the numeric format differs, so the limit here is
flash *headroom* for the runtime rather than model size or supported operators:
on this board quantization is what makes the widest model deployable at all.

**Quantization.** Full-int8 PTQ costs accuracy on these wide-dynamic-range
inputs (deployed cls_best weights, 92.08 → 86.86%); **quantization-aware training
recovers it to 89.82%**, measured at **1.558 ms** on the H7B3I-DK and 7.381 ms on
the F401RE. int16×8 preserves accuracy offline but ST Edge AI silently dequantizes
it, so it is not deployable. See `unas/qat_finetune.py` and
`datasets/dmir/docs/deployment.md`.

## highD and exiD: lane changes of surrounding vehicles

highD and exiD were recorded from a drone on German motorways, highD on straight
sections and exiD at entries and exits. highD follows the protocol of Mozaffari et
al. (IEEE T-IV 2022), reimplemented with its training and validation splits
reproduced exactly. exiD is prepared in the same scenario format
(`datasets/exid/prepare_exid.py`), so the same networks and board builds apply to
both. Five seeds each, test sets:

| model | highD accuracy | highD TTLC RMSE | exiD accuracy | exiD TTLC RMSE |
|---|--:|--:|--:|--:|
| published (Mozaffari et al.) | 83% | 0.629 s | – | – |
| hand-designed CNN (8.4 k, TTLC 8.2 k), trained on that dataset | 92.11 ± 0.71% | 0.276 ± 0.009 s | 89.93 ± 0.24% | 0.406 ± 0.008 s |
| searched classifier, 7.9 k¹ | 92.16 ± 0.71% | – | 89.50 ± 0.31% | – |
| searched classifier, 5.3 k¹ | 90.83 ± 1.57% | – | 87.40 ± 0.14% | – |
| searched TTLC regressor, 28 k¹ | – | 0.282 ± 0.016 s | – | 0.394 ± 0.008 s |
| v2 search (global-pooling space), 24 k² | 91.09 ± 0.66% | – | 89.31 ± 0.08% | – |
| hand-designed CNN trained on highD, applied as is | – | – | 61.24 ± 2.32% | 0.867 ± 0.062 s |
| the same, fine-tuned on exiD | – | – | 90.13 ± 0.22% | 0.393 ± 0.002 s |

- On highD every retrained model beats the published figures on average, and the
  deployed 5.3 k int8 classifier predicts in 0.106 ms on the Cortex-M7.
- A highD model at motorway junctions gets exits right in 33% of the windows and
  lane keeping in 45%; trained on exiD, 94% and 85%.
- Starting from highD weights adds 1.3 points with 10% of the exiD training data,
  and nothing measurable with all of it.
- ¹ Architectures found by the first highD searches, retrained with five seeds under
  the search recipe (the table in `datasets/highd/docs/baseline-results.md` has the
  hand recipe too). These Keras runs permute the training windows once: the prepared
  splits are stored scenario by scenario and the search tool shuffles with a
  2,048-window buffer, which cost one architecture 12.7 points on exiD
  (`unas/shuffle_check.py`). Runs before 2026-09-23 used the stored order and made
  the searched architectures look 11 points worse on exiD.
- ² The v2 search adds a global-average-pooling head, zero hidden layers and dropout
  to the space (`unas/cnn1d_gap.py`) and chooses on the five-seed validation mean
  (`unas/select_by_seeds.py`); under the hand-designed CNN's recipe its choice
  reaches 92.51 ± 0.41%.
- On the boards the hand-designed highD CNN takes 0.669 ms (float32) and 0.350 ms
  (int8) on the Cortex-M7: as accurate as the 7.9 k searched classifier and 12%
  faster, 4.3 times slower than the 5.3 k one, which is 1.3 points less accurate
  over five seeds. The v2 choice needs 2.891 ms. For time to lane change the
  hand-designed CNN matches the searched 28 k regressor at 0.667 against 1.038 ms.
- exiD models have the board cost of the highD builds: same graphs, other weights.

Details: `datasets/highd/README.md`, `datasets/exid/README.md`,
`datasets/exid/docs/DATA.md`.

## Repository layout

```
src/                             shared, dataset-agnostic logic (train, logging, env, EDA)
unas/                            µNAS fork adapters, search launchers, export/quantization tools
notebooks/dmir_pipeline.ipynb    main LCIR pipeline; all knobs in its Config cell
scripts/run_baseline.py          train the baseline on one LCIR task and log the run
scripts/check_pipeline.py        3-task smoke test on real data; run after every change
docs/research/                   shared literature and toolchain notes (µNAS, ST Edge AI, venue)
dashboard/                       results explorer: one interactive page for every dataset (build.py)
LOGBOOK.md                       dated journal of decisions and results (all datasets)

datasets/dmir/                   everything specific to LCIR (codename dmir)
  ├── data/                      prepared pickles (gitignored)
  ├── docs/                      DATA.md + dataset/results/deployment notes
  ├── logs/experiments.jsonl     one JSON line per run (feeds the paper's tables)
  └── results/                   nas-fronts/, deploy/ (+ measurements.json), qat/, seeds/
datasets/highd/                  second dataset: highD lane-change prediction
  (same shape; data gitignored because the highD licence forbids redistribution)
datasets/exid/                   third dataset: exiD, prepared in the highD format
  (same shape; data gitignored because the exiD licence forbids redistribution)
```

A new dataset gets its own `datasets/<name>/` with the same shape; shared code
stays in `src/` and `unas/` rather than being copied per dataset.

To browse every result in one place, build the results explorer and open the page
it writes (standard library only, opens offline):

```
python dashboard/build.py            # dashboard/dist/results-explorer.html
python dashboard/build.py --share    # copy for outside the lab (no raw-trajectory media)
```

Kept locally and not published here: `paper/` (the LaTeX manuscript, built with
`scripts/build_paper.ps1`), `course/` (the trilingual course website), and
`CLAUDE.md` (working notes for the coding agent).

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

## Licence and data

The code and documentation in this repository are released under the MIT
licence (`LICENSE`). The datasets are **not** covered by it and are not
redistributed here:

- **LCIR, Lane Change Intention Recognition** (codename DMIR): MIT, Zenodo
  [10.5281/zenodo.16686054](https://doi.org/10.5281/zenodo.16686054).
- **highD**: free for academic use on request from
  [levelxdata.com/highd-dataset](https://levelxdata.com/highd-dataset/);
  redistribution is not permitted, so obtain your own copy.
- **exiD**: free for non-commercial use on request from
  [levelxdata.com/exid-dataset](https://levelxdata.com/exid-dataset/);
  redistribution is not permitted, so obtain your own copy.

Cite this work through `CITATION.cff`, and cite the dataset papers separately
when you use the data.

## Working rules

1. Every experiment goes through the notebook or scripts, never through untracked
   one-offs; every run appends to `datasets/<name>/logs/experiments.jsonl`.
2. After any change to `src/`: `python scripts/check_pipeline.py` must pass.
3. Paper numbers only from logged runs or cited sources; no placeholder data
   anywhere in the pipeline.
4. `LOGBOOK.md` records decisions; the local `course/` and `paper/` are updated
   as milestones land, but are not published in this repository.
