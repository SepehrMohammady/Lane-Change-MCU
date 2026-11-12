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
| `input_clip_check.py [task ...]` | trains the corrected LCIR choices with five seeds on raw inputs and converts every run to full int8 (int8 I/O) with the calibration windows as they are and clipped to [-B, B] for eight bounds B: the features are standardized with rare extremes of several hundred, which set the int8 input range; `--report` picks B per head on validation and applies the rule on test | `datasets/dmir/results/seeds/input_clip.jsonl` |
| `input_equalize_check.py` | the hand-designed exiD classifier of seeds 0-4 (bundles of `scripts/export_hand_cnn.py exid5`) with a per-feature input gain folded into the first convolution, exact in float, for five strengths; full int8 (int8 I/O) on validation and test; `--report` picks the strength on validation and applies the rule on test | `datasets/exid/results/seeds/input_equalize.jsonl` |
| `stream_train_check.py [arm ...]` | the LCIR intention network of the raw-input search trained with five seeds on the lab's windows and on windows taken from the whole drives of the training drivers (`datasets/dmir/session_stream.py build train`, `build more`), converted to int8 as deployed and scored on the whole drives of the validation and test drivers with the measures of `datasets/dmir/decision_rule.py`; `--report` gives the means and the criterion (see Hardware in the loop); `--six` scores the kept networks again on the six test drivers that are not a copy of a training drive | `datasets/dmir/results/seeds/stream_windows.jsonl`, `stream_windows_six_test_drivers.jsonl`, `datasets/dmir/results/hil/stream_windows.json` |
| `stream_cv_check.py [L] [D]` | the same comparison as a cross-validation over the 48 different LCIR drives, six folds of eight: the lab's windows of the training drives against windows taken from their whole drives, five seeds, int8 as deployed, every held-out drive scored on its whole session; criterion fixed beforehand on the 34 drives that were never a test or validation drive; `--report` gives the per-drive table and the criterion | `datasets/dmir/results/seeds/stream_cv.jsonl`, `datasets/dmir/results/hil/stream_cv.json` |
| `resource_bias.py [search ...]` | for every evaluated candidate of a search: the fork's stored peak memory, size and MACs against the Keras model's weights and MACs and the faithful graph (CPU, from the fork's state files) | `datasets/highd/results/resource_bias.json` |

Outcome of `run_mm_searches.sh` (2026-11-11, test, fresh seeds 100-104): searched on scaled inputs, the LCR and
LCL choices lose almost nothing in full int8 (RMSE 0.456 and 0.458 s, against 0.452 and 0.441 s in float32 for
the raw-input choices, whose int8 builds reach 0.688 and 0.490 s); the intention choice does not keep the float
accuracy (87.95% int8 against 91.30% float). On the H7B3I-DK the LCR choice in int8 runs 3.2 times faster
than the raw-input choice in float32 (1.012 against 3.229 ms); the LCL choice has 7.2 times the MACs and is
2.6 times slower (1.321 against 0.507 ms), so float32 stays the LCL build. `prepare_deploy_lcir.py` applies
the search's scaling to models from `*_mm*` folders; results and deploy files in
`datasets/dmir/docs/deployment.md`.

Outcome of `input_clip_check.py` (2026-11-11, test, fresh seeds 100-104, rule fixed before the run): with the
calibration windows clipped to ±4 (intention, LCR) or ±6 (LCL) standard deviations, the raw-input choices keep
their float32 accuracy in full int8 without retraining: intention 91.25% against 91.00% (85.24% with the
calibration as before), LCR 0.449 against 0.451 s (0.659 s), LCL 0.434 against 0.438 s (0.503 s). On the H7B3I-DK
the int8 builds take 0.520, 1.183 and 0.243 ms against 0.805, 3.229 and 0.507 ms in float32, so every LCIR head
is deployed in int8 (`CALIB_CLIP=<B> prepare_deploy_lcir.py`, files `lcir_v4k3sat_*`). The same extremes set
the ranges of the min-max scaling used by the scaled-input searches, where the central 98% of the values of 4
to 7 features fills under 5% of the unit interval; whether that causes their float loss was not tested.

Outcome of `input_equalize_check.py` (2026-11-11, exiD, hand-designed classifier, seeds 0-4, rule fixed before
the run): in int8 this network loses 2.4 points on average (89.93% to 87.49%, up to 4.9 for one seed), and the
whole loss sits in the weights of its first convolution, where the largest weight on one input feature is 66 to
143 times the largest on another. A gain g_f = (W_f / max W)^0.25 per input feature, folded into that layer
(float outputs equal to 1e-4), brings int8 to 89.50%, 0.43 points under float32. On a board the gain is part of
the scaling table. `EQUALIZE_ALPHA=0.25 deploy_hand_cnn.py` writes the files (`exid_dscnn_cls_eq_*`: 89.44% in
int8 against 86.82% before, seed 0).

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
exiD scenario, the protocol's windows). An LCIR test window is one sequence of 50 frames at 10 Hz with one
decision; there the board first limits every feature to its training range, as every LCIR evaluation does.

| script | what it does |
|---|---|
| `hil_prepare.py <tflite>` | WSL: the firmware header (window, per-feature offset and range, input quantization) and the host reference (int8 frames and TFLite outputs for every test window), both under `firmware/build/stream/` (not versioned: training statistics and test data) |
| `hil_stream.py <tflite> [--rate 5] [--limit N]` | Windows: generates, builds and flashes the streaming firmware, replays the test scenarios at `--rate` frames per second (0 = as fast as the link allows) and compares the board's int8 outputs with the host reference; writes agreement, the protocol's metrics from the board's outputs, cycles per frame and the host round trip to `datasets/<ds>/results/hil/hil_stream.jsonl` |
| `hil_host_check.py [--board] <tflite> [...]` | Windows, after `hil_stream.py --rate 0 --save`: runs ST's generated C model on the PC (`stedgeai validate --mode host`) on the windows where board and reference differ and on control windows; with `--board` it also gives those int8 windows to the board directly (`make CHECK=1`, outputs read over SWD), which separates the board's preprocessing from its network code; counts to `datasets/<ds>/results/hil/hil_host_check.jsonl` |

Outcome on the STM32H7B3I-DK (2026-11-11, `datasets/highd/results/hil/hil_stream.jsonl`; 693 test scenarios,
18,018 windows per network, at top speed): the searched classifier, the searched TTLC regressor and the
hand-designed classifier return the int8 outputs of the TFLite reference in every window. The hand-designed
TTLC network differs by 1 LSB in one window, and there ST's C model run on the host (`stedgeai validate --mode
host`) differs from TFLite in the same way. Per frame the board spends 6.4 µs on scaling, quantization and the
window copy and 0.107 to 0.316 ms on the network; the host round trip over the virtual COM port takes 1.4 to
1.6 ms. `--save` keeps the board's raw outputs next to the reference.

In real time (searched classifier, full test split, 24,255 frames per run): at the protocol's 5 Hz and at
highD's camera rate of 25 Hz every window again equals the reference and the metrics are those of the deploy
record (92.76% accuracy, τ_c 4.78 s). The core computes for 0.07% of the frame period at 5 Hz and 0.36% at
25 Hz. The host sends a frame a mean 0.47 ms and 0.37 ms after its due time (99th percentile 1.0 and 0.9 ms,
worst case 30 and 99 ms: the PC's scheduling, not the board).

exiD (hand-designed CNN trained on exiD, seed 0, files from `scripts/export_hand_cnn.py exid` and
`deploy_hand_cnn.py`; 1,757 test scenarios, 45,682 windows per network, top speed): the classifier's outputs
equal the reference in all but 6 windows (at most 2 LSB) and the TTLC network's in all but 1 (1 LSB). From the
board's outputs: 86.82% accuracy, τ_c 4.65 s, RMSE 0.417 s, the values of the deploy records; the network takes
0.315 ms per frame. In float32 this classifier reaches 89.94%, so int8 costs it 3.1 points on exiD against 0.2
on highD; `input_equalize_check.py` (above) finds the cause and removes the loss. `hil_host_check.py` on the 8 differing windows of all replays (6 + 1 on exiD, 1 on
highD) and 32 control windows per network: ST's C model on the PC returns the board's values in every one of
them (a later replay has three windows where it does not: last paragraph of this section).

LCIR (the three int8 files with the clipped calibration, every test window as one 50-frame sequence, top speed):
intention 19,722 of 19,722 windows equal to the reference and 91.26% from the board's
outputs; LCR 7,073 of 7,073, RMSE 0.450 s; LCL 8,096 of 8,097 (1 LSB), RMSE 0.433 s: the values of
the deploy records. Per frame the board spends 12 µs on limiting, quantizing and storing the 31 features and
0.521, 1.187 and 0.243 ms on the networks. At LCIR's 10 Hz (the first 120 windows of each test set, 6,000
frames per network) all outputs are again equal; with a decision on every frame the core would compute for 0.53,
1.20 and 0.25% of the frame period.

More replays at top speed: the seed-43 choices of the corrected highD searches (18,010 and 18,018 of 18,018 windows
equal; 92.57%, τ_c 4.76 s; TTLC RMSE 0.338 s), the equalized exiD classifier (45,674 of 45,682;
89.44%, τ_c 4.60 s) and the simulated scenarios of `datasets/carla/` given to the highD classifier (7,436
of 7,436 windows equal, with inputs far outside the training range), and the three LCIR networks searched on
scaled inputs, where the board also applies the min-max scaling (intention 19,722 of 19,722 windows equal, 87.91%;
LCR 7,072 of 7,073, RMSE 0.459 s; LCL 8,096 of 8,097, RMSE 0.467 s). Over all replays of the three datasets:
15 networks, 314,938 windows, 27 of them different from TFLite by at most 2 LSB. `hil_host_check.py --board` on
every replay with a difference: given the reference's int8 window directly, without its own preprocessing, the board
returns the streamed value in all 27 differing windows and all 64 control windows, so its preprocessing and the
link are not involved in any difference. ST's C model compiled for the PC returns the board's value in 23 of the 27
and TFLite's value in the other 4 (three in the equalized exiD classifier, one in the LCL network searched on scaled
inputs): the differences lie in the network arithmetic of ST's Cortex-M7 kernels against the TFLite interpreter and,
rarely, against ST's own PC build.

Whole LCIR drives (2026-11-11). The prepared test sets are shuffled, class-balanced windows; a deployed network
gets every frame of a drive. `datasets/dmir/session_stream.py` rebuilds the complete raw sessions of the seven
official test drivers as frame streams (14 sessions, 175 minutes, 543 lane changes) and recovers
the provider's normalization for the three prepared datasets: one affine map per channel, which reproduces every
window of the test pickles. A copy of a network's file under `datasets/dmir/data/sessions/sessions_<file name>` makes
`hil_prepare.py` and `hil_stream.py` replay those drives, a decision on every frame from the 50th on;
`datasets/dmir/session_metrics.py` scores the board's outputs with measures fixed before the first output existed.
Six networks, 104,288 windows each: 50 of 625,728 windows differ from TFLite by at most 2 LSB, and the board
given those int8 windows directly returns the streamed value in all 50. One drive at the recording's 10 Hz
(9,422 frames): every output equal, the core computes for 0.54% of the frame period, the host is a mean
0.52 ms late (99th percentile 1.0 ms). On the frames of the test pickles the stream returns the deploy
records (91.26%, 0.450 s, 0.433 s). On the whole drives, intention network searched on raw inputs:

| measure | result |
|---|---|
| lane changes recognized at least 2 s before the crossing | 90.4% of 543 (median lead 5.0 s, 2.4% missed) |
| the same, lane changes held by the lab's test pickles | 95.0% of 382 |
| the same, lane changes not in the pickles (145 left out, 16 made in the baseline sessions) | 79.5% of 161 |
| baseline sessions, 69 minutes of free driving (they hold 16 lane changes): frames called a lane change | 4.2%; 52 false alarms of 0.5 s or more per hour |
| sessions with lane changes: free-driving frames called a lane change | 46.6% |
| of those frames, window still holding a completed lane change (post hoc split) | 84.0% called a lane change, against 21.4% otherwise |
| LCR / LCL RMSE on test-pickle frames | 0.450 / 0.433 s |
| LCR / LCL RMSE before lane changes left out of the pickles | 0.835 / 0.747 s |

The lab's test pickles hold 382 of the 527 lane changes that its own rule finds in the seven sessions with lane
changes, and their
free-driving windows never contain a completed lane change, while a stream delivers such windows after every lane
change; there the network keeps announcing a lane change. An accuracy on balanced windows therefore does not
describe the deployed behaviour. The network searched on scaled inputs: 85.1% at 2 s (median 3.3 s),
37 false alarms per hour in the baseline sessions. All numbers: `datasets/dmir/results/hil/session_metrics.json`.

A decision rule for the alarm (2026-11-12, `datasets/dmir/decision_rule.py`). A deployed system would not show
the frame-by-frame output but an alarm. Eighty rules were fixed before any validation session was built: a
probability threshold (none, 0.6, 0.8, 0.9), a hold-off after the car's own lane-line crossing (0, 1.5, 3, 5,
6.4 s) and a persistence (the last 1, 3, 5 or 10 frames agree). The rule is chosen on the whole drives of the seven
validation drivers (`session_stream.py build val`: 14 sessions, 170 minutes, 540 lane changes; the map recovered
on the test drivers reproduces every window of the validation pickles) as the one with the fewest false alarms
among those that lose at most 5 points of the lane changes announced 2 s early, and it is adopted if on the
test drivers it at least halves the false alarms per hour of free driving within the same 5 points. A false
alarm here is every run of alarm frames that reaches no frame labelled as before a lane change, of any length,
all 14 sessions pooled; the outputs are the board's.

| intention network | chosen on the validation drivers | test drivers, raw output | test drivers, with the rule | adopted |
|---|---|---|---|---|
| searched on raw inputs | 5 frames of persistence, no threshold, no hold-off | 90.4% of 543 early, 301 false alarms per hour | 87.8%, 200 per hour (34% fewer) | no |
| searched on scaled inputs | 3 frames of persistence, no threshold, no hold-off | 85.1%, 631 per hour | 80.8%, 284 per hour (55% fewer) | yes |

For the network searched on raw inputs no rule of this family meets the criterion. In its baseline sessions the
chosen rule lowers the alarms from 145 to 66 per hour (52 to 39 per hour for those of 0.5 s or more). Only 5 of
the 80 rules stay within the 5 points on the validation drivers, and none of them has a hold-off: a hold-off of
6.4 s leaves 48.2% of the lane changes announced early on the test drivers. The reason is in the drives
(`datasets/dmir/lane_change_pairs.py`, not planned beforehand): on this two-lane road every lane change is the
opposite of the one before it, the median gap between them is 8.6 s for the test drivers, 49% follow within
8.4 s and 78% of those are the return to the right after an overtake. The prepared windows carry this pairing as
a shortcut: no free-driving window of the classification test pickle holds a completed lane change, while 48.1%
of its windows before a right lane change and 7.4% of those before a left one do. On whole drives 31.2% of the
free-driving frames of the sessions with lane changes have such a window. There the network announces the
opposite lane change (72.8% of those frames between 1.5 and 6.3 s after a lane change, validation drivers); that
lane change follows within 10 s of the completed one in 48% of the cases. Numbers:
`datasets/dmir/results/hil/decision_rule.json`, `lane_change_pairs.json`, `session_stream_val.json`.

Training windows taken from whole drives (2026-11-12, `unas/stream_train_check.py`). If the shortcut is in the
windows, the windows have to change: every frame of the training drivers' sessions with lane changes, from the
50th on and not ignored, gives a window with the class of its last frame, whatever else the window holds; the
classes are balanced by subsampling. The architecture (the intention network searched on raw inputs), the recipe
and the five seeds are the same in every arm, the networks are scored in int8 as deployed, and the arms and the
criterion were fixed beforehand: adopted if, on the test drivers and for the raw output, the false alarms per
hour are at most half of those of the networks trained on the lab's windows, within 5 points of the lane changes
announced 2 s early. The lab's training pickles hold 36 drivers: the 22 of the archive used here and users 37-50
of the public record (`session_stream.py build train`, `build more`; every training window is located).

| test drivers, mean of five seeds | lab's pickle, 36 drivers | lab's windows, 22 drivers | whole drives, 22 drivers | whole drives and baseline sessions, 22 drivers | whole drives, 36 drivers |
|---|---|---|---|---|---|
| training windows | 94,620 | 60,969 | 82,695 | 84,906 | 139,797 |
| lane changes announced 2 s early, raw output | 89.6% | 88.5% | 87.3% | 87.3% | 87.1% |
| false alarms per hour, raw output | 301 ± 39 | 295 ± 16 | 358 ± 55 | 350 ± 37 | 322 ± 49 |
| free-driving frames in alarm, raw output | 22.8% | 21.5% | 10.4% | 11.4% | 9.4% |
| with the rule chosen on the validation drivers: announced early, false alarms per hour | 83.4%, 178 | 84.2%, 221 | 82.2%, 106 | 83.4%, 114 | 81.9%, 87 |
| correct frames on whole drives | 80.1% ± 0.6 | 80.8% ± 0.8 | 89.6% ± 1.4 | 89.1% ± 1.4 | 90.5% ± 0.7 |
| frames after a lane change on which the opposite one is announced | 74.8% | 67.2% | 20.4% | 27.6% | 21.5% |
| false alarms per hour in the baseline sessions, raw output | 149 | 115 | 190 | 88 | 147 |
| accuracy on the lab's balanced test set, int8 | 91.25% | 90.82% | 89.99% | 89.13% | 90.70% |

The criterion is not met: the number of alarm runs does not fall, it rises (358 against 301 per hour at 87.3%
against 89.6%). With all 36 training drivers, the comparison with equal drivers, it is 322 per hour at 87.1%:
not met either. The other measures fixed beforehand show what did change. After a lane change the network
announces the opposite one on 20% of the frames instead of 75%, the free-driving frames in alarm fall from 22.8%
to 10.4%, and the share of correct frames on whole drives rises from 80.1% to 89.6% (90.5% with 36 drivers), for
1.3 points on the lab's balanced test set (0.6 with 36 drivers). The alarms that remain are shorter and more
numerous, which a count of runs of any length does not reward. With the persistence rule chosen for each network
on the validation drivers, the windows from whole drives give 106 false alarms per hour at 82.2% (87 at 81.9%
with 36 drivers), and 114 at 83.4% when the baseline sessions are added, against 178 at 83.4% for the lab's
windows; none of these is within 5 points of the 89.6%. The networks trained on the lab's windows are right on
91.25% of the lab's balanced test windows and on 80.1% of the frames of whole drives, and the windows are the
reason. Per seed: `datasets/dmir/results/seeds/stream_windows.jsonl`; means and the criterion:
`datasets/dmir/results/hil/stream_windows.json`.

One of the seven test drivers is a copy of a training drive (2026-11-12, `datasets/dmir/duplicate_sessions.py`;
found while scanning the raw sessions, not looked for). The session files of user 25 hold, in every signal of
every frame, the sessions of user 9, and the session with lane changes of user 24 holds that of user 23; the
public record shows the same (one checksum for `user9LC.csv` and `user25LC.csv`, one for `user23LC.csv` and
`user24LC.csv`). The dataset has 48 different drives with lane changes. User 25 is an official test driver,
user 9 a training driver: 19.0% of the intention test windows and 15.1% / 13.2% of the LCR / LCL test windows
come from a drive that is in the training sets (details in `datasets/dmir/docs/DATA.md`). All numbers above
were measured on the official seven; on the six test drivers that are not in the training set:

| | official seven test drivers | six drivers not in the training set |
|---|---|---|
| deployed networks on the board: intention accuracy; LCR, LCL RMSE (test sets) | 91.26%; 0.450 s, 0.433 s | 90.63%; 0.462 s, 0.445 s |
| deployed intention network, whole drives: announced 2 s early, false alarms per hour | 90.4%, 301 | 89.4%, 317 |
| the same with the rule chosen on the validation drivers | 87.8%, 200 | 86.8%, 218 |
| correct frames on whole drives: lab's windows (36 drivers) against windows from whole drives (36 drivers), five seeds | 80.1% against 90.5% | 79.1% against 89.8% |
| the same, sessions with lane changes only | 68.9% against 85.8% | 67.6% against 85.0% |
| frames after a lane change on which the opposite one is announced | 74.8% against 21.5% | 73.6% against 20.7% |
| false alarms per hour, raw output | 301 against 322 | 321 against 346 |
| accuracy on the lab's balanced test set, int8 | 91.25% against 90.70% | 90.50% against 89.66% |

The six-driver numbers were computed after the studies (`stream_train_check.py --six`,
`datasets/dmir/results/seeds/stream_windows_six_test_drivers.jsonl`); the pre-set criteria were judged on the
official seven and stay as reported. Nothing changes in what the comparisons say; the absolute numbers on the
official split are 0.6 points and 0.012 s better than on drivers the networks have not seen.

Windows from whole drives on drivers nobody had looked at (2026-11-12, `unas/stream_cv_check.py`). The rise in
correct frames above was a listed measure of a study whose criterion failed, on test drivers that had been
looked at several times. This is the test with the measure fixed first: a cross-validation over the 48
different drives, six folds of eight (eight held out, eight for early stopping, 32 for training), the same
architecture, recipe and five seeds in both arms, int8 as deployed. Arm L trains on the lab's prepared windows of
the training drives (`session_stream.py rows` gives the drive of every prepared window), arm D on windows taken
from their whole sessions with lane changes. The criterion was written down before the script existed, for the
34 drives that had never been a test or validation drive: D at least 8 points above L in the share of correct
frames of the held-out sessions with lane changes, above L for at least 30 of the 34 drives, and at most 2
points below L on the lab's own windows of those drives.

| per held-out drive, mean over drives | lab's windows (L) | windows from whole drives (D) | |
|---|---|---|---|
| correct frames, sessions with lane changes, 34 drives not looked at before | 69.7% | 85.8% | +16.1 points (95% interval over drives 14.4 to 17.7), D above L in 34 of 34 |
| the same, all 48 drives | 69.7% | 86.3% | +16.6 points, 48 of 48 |
| accuracy on the lab's windows of the held-out drives, 34 drives | 90.79% | 90.09% | -0.70 points |
| frames after a lane change on which the opposite one is announced | 70.9% | 20.1% | lower in 34 of 34 |
| lane changes announced over the whole last 2 s | 88.9% | 87.4% | |
| alarm runs of 0.5 s or more per hour of free driving | 248 | 101 | lower in 34 of 34 |
| alarm runs of any length per hour of free driving | 525 | 482 | |
| free-driving frames in alarm | 43.9% | 17.4% | lower in 34 of 34 |
| correct frames in the baseline session (30 of the 34 drives have one here) | 93.7% | 95.0% | |

The criterion is met in all three parts. The smallest gain of a drive is 4.9 points and the largest 23.0.
The event rates are per hour of free driving inside the sessions with lane changes and were reported without
a criterion; the two earlier criteria on the number of alarm runs of any length stay failed. Per fold, arm
and seed: `datasets/dmir/results/seeds/stream_cv.jsonl`; per drive and the criterion:
`datasets/dmir/results/hil/stream_cv.json`.

The dataset's simulator as the source of the frames (2026-11-12). The simulator behind LCIR is public
(LC-Intention-Framework: a packaged CARLA 0.9.15 with the dataset's road, the driving client and the data
preparation). On the development laptop the unmodified client drove one minute on autopilot, the simulator's
recorder file went through the lab's `generate_CDF_Carla_log_to_H5.py`, and
`datasets/dmir/session_stream.py own <name>` turned that H5 into a frame stream: 573 frames at 10 Hz,
every channel inside the training range. `hil_prepare.py` with `LCIR_SPLIT=own` and
`hil_stream.py --split own --rate 10` sent it to the board in real time: 524 decisions, all identical to
TFLite, 0.54% of the frame period (`datasets/dmir/results/hil/hil_own.jsonl`). This checks the chain from
the simulator to the board with a recording; the autopilot changes no lane, so it says nothing about
recognition. Two things were needed to run the package on Windows 11: the DirectX runtime of June 2010 (the
package has no installer for it) and a start of the client that waits until the simulator answers (5 to 7 s
here, the client connects after 3 s).

Before closing a loop around the board (2026-11-11): `datasets/highd/cutin_headroom.py` replaces the follower of
each test lane change by a car with an ACC (point mass, the car-following model of Milanés and Shladover, 2014)
and asks what that car gains from knowing the lane change k seconds before the crossing; recorded data only.
Rule fixed before the run: at least 30 runs above 2.0 m/s² deceleration without early knowledge, at most half as
many with 2 s of it. Not met: 86 against 76 of 91 eligible scenarios. The braking comes from the controller's gap
rule, not from the traffic: the recorded followers never exceed 1.33 m/s² on highD and 1.28 m/s² on exiD
(`datasets/exid/cutin_followers.py`, 272 lane changes), because the vehicle that cuts in is usually faster than
its new follower. These recordings hold no conflict that an earlier prediction could resolve for a following
car, so that closed loop is not built. Results: `datasets/*/results/hil/cutin_*.json`.

A simulator as the source of the frames (2026-11-11, `datasets/carla/`): `sim_features.py` computes the 18
features of the highD protocol from positions, velocities and lane geometry alone, with the neighbours found
geometrically; on highD's five test recordings it equals `prepare_highd.py` in at least 99.87% of the vehicle
frames for every feature. `run_carla.sh` and `carla_record.py` record CARLA Traffic Manager vehicles on a
straight three-lane carriageway (no rendering, 8 times real time with 42 vehicles). Rule fixed before the
network saw that traffic: usable if 80% of the simulated lane changes are recognized 2 s before the crossing
with at most 10% of the lane-keeping windows called a lane change. Not met: 11.5% and 40.3% (highD test split:
100% and 9.4%). The Traffic Manager's lane changes have no lead-up: no lateral motion until 0.8 s before the
crossing, then 4.1 m/s sideways against 0.87 m/s on highD. A study of lane-change prediction in CARLA therefore
needs human drivers or replayed recordings; details in `datasets/carla/README.md`.

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
