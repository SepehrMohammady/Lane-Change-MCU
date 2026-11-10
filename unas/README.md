# LCIR (DMIR) × µNAS integration

> **Search names.** The searches ran in four versions: v1, the first searches (the tool's own space, saver and cost count); v2, a global-average-pooling head and a five-seed choice on validation data; v3, budgets at the hand-designed CNN's cost as the tool counts it; v4, the final searches, which count the cost on the network that is trained. Configs and result folders carry these names (`*_v2` to `*_v4`, `results/nas-v2/`). The paper and the results explorer call them the first, second, third and final searches.


We run the constrained architecture search with the **ELIOS lab µNAS fork**
(https://github.com/Elios-Lab/uNAS) rather than reimplementing it: the colleague's
directive ("take µNAS code"), and the fork already provides what we need:
1D multi-channel CNN search space, regression (`num_classes=1`), QAT during
search, and INT8 TFLite output compatible with the ST Edge AI tools.

**Why not vendor it here:** the fork has no license file (as does upstream), so we
keep it as an external checkout and version only *our* adapter files (this folder,
MIT with the rest of the repo).

## What "efficiency" and "threshold" mean (the colleague's words)

- **efficiency** = the resource objectives µNAS minimises: peak SRAM
  (`peak_mem_bound`), INT8 weight storage (`model_size_bound`), and MACs
  (`mac_bound`), computed statically per candidate by `to_resource_graph`.
- **threshold** = the four `BoundConfig` bounds. Aging evolution's fitness is
  `-max_i( normalise(feature_i, bound_i) / lambda_i )` over
  `[val_error, peak_mem, model_size, mac]` with random `lambda_i` per round, so a
  candidate that exceeds any bound is normalised to > 1 and penalised first.
  Setting the thresholds = setting our STM32H7B3I-DK budget.

`val_error` per the fork's trainer: classification `1 - max(val_accuracy)`;
regression `min(val_mae)` (the loss is MAE, so optimise and threshold on **MAE**, not RMSE).

## Files here

| File | Drop into the fork at | Purpose |
|---|---|---|
| `dmir_dataset.py` | `dataset/dmir_dataset.py` | serves the real DMIR pickles (all 3 tasks), train-range clipping, optional indicator drop |
| `dmir_config.py` | `configs/dmir_config.py` | search configs + STM32H7B3I-DK thresholds |
| `highd_dataset.py`, `highd_config.py` | `dataset/`, `configs/` | highD (and exiD) windows; highD search configs, including the `*_v2` ones |
| `cnn1d_gap.py` | `configs/cnn1d_gap.py` | the fork's 1D search space plus a global-average-pooling head, zero to three hidden Dense layers and a searched dropout rate (used by the `*_v2` configs); `FaithfulGapCnn1DSearchSpace` also builds the resource graph with the Keras model's shapes (used by the `*_v4` configs) |
| `safe_saver.py` | `configs/safe_saver.py` | model saver with unique file names and a JSON sidecar per model (validation error, resources); keeps every candidate within the resource bounds and never uses test error (used by the `*_v2` configs) |
| `two_domain.py` | `configs/two_domain.py` | per-domain validation accuracy after every epoch for searches on highD and exiD pooled (`twod_cls_v4` config) |

`run_chunked_highd.sh` copies the highD files into the fork and registers the configs (`run_twod.sh` for
the two-domain config).

Register in the fork:
- `dataset/__init__.py`: `from .dmir_dataset import DMIR_Dataset`
- `driver.py` `_CONFIGS`:
  ```python
  "dmir_lcr":       ("configs.dmir_config", "get_dmir_lcr_setup"),
  "dmir_lcl":       ("configs.dmir_config", "get_dmir_lcl_setup"),
  "dmir_cls":       ("configs.dmir_config", "get_dmir_cls_setup"),
  "dmir_cls_noind": ("configs.dmir_config", "get_dmir_cls_noind_setup"),
  ```
- set `DATA_ROOT` in `dmir_dataset.py` to this repo's `data/`.

## Run

```bash
python driver.py -c dmir_lcr     # LCR time-to-lane-change regression (primary)
python driver.py -c dmir_lcl
python driver.py -c dmir_cls          # classification (report with ablation)
python driver.py -c dmir_cls_noind    # classification without turn indicators
# then per the fork's 3-step pipeline: train.py (QAT) -> test.py (INT8 TFLite)
```

Output: Pareto-front `.h5` models + a state `.pickle` under
`artifacts/<name>/`. QAT fine-tune (`train.py`) → INT8 TFLite (`test.py`) →
feed to ST Edge AI / STM32Cube.AI for the STM32H7B3I-DK flash/RAM/latency
numbers (see docs/research/stm32-toolchain.md).

**Known issue in the fork's saver.** `uNAS/model_saver.py` starts every process
with an empty model list and a file counter at zero, and each Ray worker holds its
own saver. Resumed chunks (`run_chunked*.sh`) and `*_PARALLEL=2` therefore write the
same file names into one folder, so saved `.h5` files overwrite each other and
`metadatas.json` stops describing them. Never quote the fork's metadata or console
`test_error`; evaluate the saved files (`harvest_fronts.py`, `harvest_highd.py`) and
fix the saver before the next search.

**Known issue in the fork's resource model (found 2026-09-24).** The 1-D port builds
each candidate's Keras model with `padding="same"` but its resource graph with
`padding="valid"` (`uNAS/cnn1d/cnn1d_architecture.py`), and its optional pre-pooling
rounds the sequence length up where Keras rounds it down. The graph therefore
shortens the sequence after every wide kernel and under-counts the MACs, weights and
activations of the network that is actually trained, most on short windows: over all
evaluated candidates the true MACs are a median 1.25-1.52 times the counted ones on
LCIR and 1.18-2.41 times on highD, up to 11.5 times. Budgets and cost objectives of
every search before v4 were applied to these counts; measured board numbers are not
affected. `resource_bias.py` recomputes the gap for every search from its state file;
the `*_v4` configs search with the faithful graph.

**Known issue in the fork's trainer (found 2026-11-03).** `uNAS/model_trainer.py` imports
tensorflow-model-optimization, which sets `TF_USE_LEGACY_KERAS=1` when imported. In the Ray workers this
happens before `tf.keras` is first used, so there `tf.keras` is the Keras 2 package while the search spaces
build Keras 3 models, and Keras 3 never resets the Keras 2 metric the trainer compiles. The validation
accuracy (or MAE) a search records is then a running value over every training and validation batch of the
run, the test score after training is the same value, and early stopping never fires. Every search before
the fix ran on that score; five-seed retrainings (`select_by_seeds.py`, `seed_variance.py`, `final_eval.py`
run outside Ray) and board numbers are not affected. `patch_fork.py` (patch 3) builds the loss and metric with
the model's own Keras; `path_check.py` compares the fork's trainer with `select_by_seeds.train_one` outside Ray.

## Seed studies

| script | what it does | output |
|---|---|---|
| `seed_variance.py` | retrains each final architecture from scratch with seeds 0-4; `RECIPE=search` (the fork's trainer), `final` (hand-designed highD recipe) or `dscnn` (hand-designed LCIR recipe) | `datasets/*/results/seeds/seed_variance*.jsonl` |
| `rerank_front.py` | retrains every saved highD classifier above `RERANK_FLOOR` with five seeds and ranks them by mean | `datasets/highd/results/seeds/rerank_cls.jsonl` |
| `seed_variance.py exid_*` | trains the architectures found by the highD searches on exiD (both recipes) | `datasets/exid/results/seeds/seed_variance*.jsonl` |
| `transfer_eval.py` | applies the deployed highD searched models to exiD with the highD input scaling | `datasets/exid/results/transfer_searched.jsonl` |
| `select_by_seeds.py <search>` | v2 searches: shortlists the 12 candidates with the best single-run validation metric, retrains each with five seeds, picks the best mean validation metric and the smallest model within one standard error of it; test data never enter the choice | `datasets/highd/results/seeds/select_<search>.jsonl`, `datasets/highd/results/nas-v2/<search>/` |
| `input_scaling_check.py [task ...]` | trains the corrected LCIR choices with five seeds on raw and on per-feature min-max scaled inputs and converts every run to full int8 (int8 I/O): with raw inputs one int8 input scale leaves up to half the features fewer than 8 steps; `CHOICES_FROM=<search>,...` scores the choices of named searches instead, on the inputs each was searched on | `datasets/dmir/results/seeds/input_scaling.jsonl` |
| `run_mm_searches.sh` | the three corrected seed-42 LCIR searches again on min-max scaled inputs (runs named `*_mm*`; `DMIR_SCALE=minmax` in `dmir_dataset.py`, set from the name by the runners, `select_by_seeds.py` and `final_eval.py`), their selections, then float and int8 scores on fresh seeds 100-104 for the new and the raw-input choices | `datasets/dmir/results/seeds/input_scaling_mm.jsonl` |
| `resource_bias.py [search ...]` | for every evaluated candidate of a search: the fork's stored peak memory, size and MACs against the Keras model's weights and MACs and the faithful graph (CPU, from the fork's state files) | `datasets/highd/results/resource_bias.json` |

## Spending training runs under noise (2026-11-02)

One training run does not rank the candidates (Spearman 0.17 between the run the search kept and
the five-seed mean, 17 highD candidates). `noise_aware_search.py` (dropped into the fork as
`configs/noise_aware_search.py`) keeps the fork's space, mutations, fitness and resource model and
changes only how training runs are spent; the budget counts training runs, 210 for every variant.

| variant | training runs | final choice |
|---|---|---|
| `single` | the fork's aging evolution, 150 runs | `select_by_seeds.py`: 12 best single runs x 5 seeds (60) |
| `random` | random architectures, 150 runs | the same selection (60) |
| `fixed` | 70 candidates x 3 runs, scored by the mean | best mean among candidates within the resource bounds |
| `adaptive` | one run per new candidate; extra runs, up to five, for candidates within the bounds whose interval still reaches the best one | best mean among those with five runs |
| `costfirst` (2026-11-10, `cost_first_search.py`) | the fork's aging evolution, 150 runs; in parent selection, candidates within the bounds whose validation error is within `CF_TAU` of the best one compete on MACs and beat all others | the 12 cheapest of those x 5 seeds (60); the cheapest whose mean is within `CF_MARGIN` of the best mean |

`SEL_RESCORE=1 SEL_TAG=_costfirst select_by_seeds.py <search>` applies the `costfirst` final choice to a search's
existing shortlist and runs without training (written to `selection_costfirst.json`); `FINAL_SEL_TAG=_costfirst
FINAL_TAG=_costfirst final_eval.py <search>` trains that choice with the fresh seeds.

Outcome of the `costfirst` pilot (highD classifier and LCIR intention, search seeds 42 and 43, fresh seeds
100-104; `final_eval.jsonl` mode `costfirst` and `final_eval_costfirst.jsonl`): its choices use 1.08x (highD,
11,970 and 5,841 MACs) and 1.47x (LCIR, 44,900 and 19,908 MACs) the MACs of the `single` searches with the same
final choice, so the change in parent selection does not pay off. In the search histories: on highD, the plateau
networks cheaper than the final choice lose a median 0.32 points when retrained with five seeds (their single run
was lucky); on LCIR seed 42, a 107,200-MAC network reached the plateau first and plateau-first parent selection
only walked down from it (44,900 MACs after 150 runs).

```bash
bash /mnt/c/Projects/PhD/DIMIR/unas/run_noise_aware.sh highd_cls|dmir_cls single|random|fixed|adaptive|costfirst SEED
SMOKE=1 bash .../run_noise_aware.sh highd_cls adaptive 42      # mechanics only, separate files
```

`NAME_TAG=_k3` names a series of runs (`highd_cls_v4_k3_<mode>_s<seed>`); the `_k3` series is the
first one after the trainer fix above, the runs of 2026-11-02 used the accumulated score.
The same `SEED` gives every variant the same first random architectures. Each run ends with
`final_eval.py`, which retrains the chosen network with five seeds no variant used (100-104).
Outputs: `datasets/*/results/nas-noise/<search>_runs.jsonl` (one line per training run),
`nas-noise/<search>/` (choice of `fixed` and `adaptive`), `nas-noise/final_eval.jsonl`.

`speed_check.py` times one network's training per epoch for several `steps_per_execution` values and checks,
with TF op determinism on, that per-epoch loss and validation accuracy stay the same (2026-11-03: identical,
1.3x faster). A step of these networks takes about 2 ms at batch 256 or 1024, so the GPU waits on per-step
latency; more throughput comes from trainers running side by side, and host RAM limits how many.

## Searching on two road types (2026-11-04)

The corrected highD classifier choices lose 1.5-2.3 points of exiD accuracy against the hand-like network
even when all three are trained on highD and exiD pooled (`seed_variance.py pooled_cls_*`). `run_twod.sh`
runs the final highD classifier search on the pooled data (`TwoDomain_Dataset` in `highd_dataset.py`) with
one of two validation objectives; space, resource model, budgets, error bound, epochs and workers are those
of the highD-only search.

| arm | objective (early stopping, the search's validation error, selection ranking) |
|---|---|
| `pooled` | accuracy on the pooled validation windows, the fork's own |
| `worst` | the lower of the two domains' validation accuracies |

`two_domain.py` (dropped into the fork as `configs/two_domain.py`) scores each domain's validation windows
after every epoch in both arms, so both spend the same compute; patch 4 of `patch_fork.py` lets the fork's
trainer read the validation error from the key in `NAS_VAL_KEY`. Each run ends with `select_by_seeds.py`
and `final_eval.py`, which trains every arm's choice on the pooled data with the same recipe and records
each domain's scores; `FINAL_DATA=pooled final_eval.py highd_cls_v4_k3_single_s42` does the same for the
highD-only choice.

```bash
bash /mnt/c/Projects/PhD/DIMIR/unas/run_twod.sh pooled|worst SEED
SMOKE=1 bash .../run_twod.sh worst 42      # mechanics only, separate files
```

Outputs: `datasets/exid/results/nas-v2/twod_cls_v4_k3_<arm>_s<seed>/`, `seeds/select_twod_*.jsonl`,
`nas-noise/final_eval_pooled.jsonl`.

## Hand-designed CNNs on the boards

| script | what it does |
|---|---|
| `../scripts/export_hand_cnn.py` | exports the PyTorch DSCNN weights in Keras layout, with reference outputs, calibration and test windows (Windows venv) |
| `deploy_hand_cnn.py` | rebuilds the DSCNN in Keras, checks it against PyTorch, writes float32 / int8 / int8-I/O TFLite files and scores each on the test set (WSL) |
| `st_benchmark.py` | measures TFLite files on the ST Edge AI Developer Cloud boards with ST's client; credentials from `STM32AI_USERNAME` / `STM32AI_PASSWORD`, never written to disk |
| `../scripts/register_hand_cnn.py` | adds the measured builds to `datasets/*/results/deploy/measurements.json` |

Both run in the WSL `dmir_nas` venv and resume from their output files. The
hand-built baselines take a seed argument: `scripts/run_baseline.py <task> <seed>`
(DMIR) and `datasets/highd/train_highd.py <task> <run> <seed>`.

## Hardware in the loop, step 1 (2026-11-10)

The board takes the test split one frame at a time, as a sensor would deliver it: the streaming build of
`firmware/h7b3` (`make STREAM=1`) reads a frame of raw features from the ST-LINK virtual COM port, scales
and quantizes it with the training statistics and the network's input quantization, keeps the last 10
frames and runs the network on every frame once the window is full (26 decisions per 35-frame highD or
exiD scenario, the protocol's windows).

| script | what it does |
|---|---|
| `hil_prepare.py <tflite>` | WSL: the firmware header (window, per-feature offset and range, input quantization) and the host reference (int8 frames and TFLite outputs for every test window), both under `firmware/build/stream/` (not versioned: training statistics and test data) |
| `hil_stream.py <tflite> [--rate 5] [--limit N]` | Windows: generates, builds and flashes the streaming firmware, replays the test scenarios at `--rate` frames per second (0 = as fast as the link allows) and compares the board's int8 outputs with the host reference; writes agreement, the protocol's metrics from the board's outputs, cycles per frame and the host round trip to `datasets/<ds>/results/hil/hil_stream.jsonl` |

## Thresholds (first pass, in `dmir_config.py`)

| Bound | Value | Rationale |
|---|---|---|
| `peak_mem_bound` | 256 KB | headroom under H7B3 1.4 MB SRAM |
| `model_size_bound` | 256 KB | INT8 weights, headroom under 2 MB flash |
| `mac_bound` | 2 M | baseline DSCNN ~0.17 M → generous first pass |
| `error_bound` (reg) | 0.30 MAE | ≤ baseline (0.318/0.333), near SOTA 0.298 |
| `error_bound` (cls) | 0.10 | ≥ 90% accuracy |

Tighten `peak_mem`/`model_size` after the first front to push toward the
STM32F401 low-end stretch (96 KB / 512 KB).

## Environment (see docs/research/unas-integration.md)

The fork is **TensorFlow**, not PyTorch. On Windows it pins `tensorflow<2.11`
+ `numpy==1.23.5` + Python ≤3.10, and that old TF cannot use the RTX 5070
(Blackwell needs CUDA 12.8), so on Windows the search runs on the CPU only. GPU needs Linux/WSL2
with `tensorflow[and-cuda]` (2.18), and Blackwell support there is unverified.
Options and recommendation in the integration note.
