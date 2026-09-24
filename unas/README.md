# LCIR (DMIR) × µNAS integration

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

`run_chunked_highd.sh` copies the highD files into the fork and registers the configs.

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

## Seed studies

| script | what it does | output |
|---|---|---|
| `seed_variance.py` | retrains each final architecture from scratch with seeds 0-4; `RECIPE=search` (the fork's trainer), `final` (hand-designed highD recipe) or `dscnn` (hand-designed LCIR recipe) | `datasets/*/results/seeds/seed_variance*.jsonl` |
| `rerank_front.py` | retrains every saved highD classifier above `RERANK_FLOOR` with five seeds and ranks them by mean | `datasets/highd/results/seeds/rerank_cls.jsonl` |
| `seed_variance.py exid_*` | trains the architectures found by the highD searches on exiD (both recipes) | `datasets/exid/results/seeds/seed_variance*.jsonl` |
| `transfer_eval.py` | applies the deployed highD searched models to exiD with the highD input scaling | `datasets/exid/results/transfer_searched.jsonl` |
| `select_by_seeds.py <search>` | v2 searches: shortlists the 12 candidates with the best single-run validation metric, retrains each with five seeds, picks the best mean validation metric and the smallest model within one standard error of it; test data never enter the choice | `datasets/highd/results/seeds/select_<search>.jsonl`, `datasets/highd/results/nas-v2/<search>/` |
| `resource_bias.py [search ...]` | for every evaluated candidate of a search: the fork's stored peak memory, size and MACs against the Keras model's weights and MACs and the faithful graph (CPU, from the fork's state files) | `datasets/highd/results/resource_bias.json` |

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
