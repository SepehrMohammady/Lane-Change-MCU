# NAS results (completed searches, independently verified)

Date 2026-07-09. Source: four aging-evolution searches (RTX 5070, WSL2, ELIOS
µNAS fork), each run to its **full 150-round target** via chunked resume
(`unas/run_all_chunked.sh`) after the OOM was defeated.

**All numbers are our own test-set evaluation of the saved models**
(`unas/harvest_fronts.py`), not the fork's console output (its `val_error` is an
optimistic `min(val_mae)`). Footprint = int8 weight bytes (= parameter count);
peak RAM, MACs, and measured latency come from ST Edge AI at deployment.
Front CSVs: `datasets/dmir/results/nas-fronts/*.csv`.

Reference points: published SOTA (Forneris et al., SPL 2026, Transformer
~54k params, FP32): single TTLC MAE 0.298 / RMSE 0.510. Internal unpublished
reference: RMSE 0.42 (LCR) / 0.44 (LCL), acc 92%. Our PyTorch DSCNN (~10.5k
params): LCR MAE 0.318 / RMSE 0.439; LCL MAE 0.333 / RMSE 0.459; acc 91.5%.

## Search cost

All four 150-round searches ran as a single overnight queue on one RTX 5070
laptop GPU (launched 2026-07-08, completed via chunked resume on 2026-07-09 —
see `LOGBOOK.md`): **about one GPU-night for all four searches in total**. No
finer per-search wall-clock was logged, so no per-task figure is quoted.

Literature context (different tasks, datasets, and hardware — cite as context
only, never as a direct comparison): NASNet's RL search ~2000 GPU-days
(Zoph & Le), AmoebaNet's evolution ~3150 GPU-days (Real et al., 2019),
DARTS' gradient search ~1–4 GPU-days (Liu et al., 2019).

- [ ] TODO: log wall-clock per chunk on any future re-run so a per-search
      figure can be reported.

## Regression — LCR (time to right lane change)

| Model | params | int8 KB | test MAE | test RMSE |
|---|--:|--:|--:|--:|
| best MAE | 117,404 | 114.7 | **0.287** | 0.447 |
| compact | 65,863 | 64.3 | 0.290 | 0.454 |

Beats the published SOTA on MAE (0.287 < 0.298) and RMSE (0.447 < 0.510), and
still beats it at 64 KB.

## Regression — LCL (time to left lane change)

| Model | params | int8 KB | test MAE | test RMSE |
|---|--:|--:|--:|--:|
| best MAE | 84,743 | 82.8 | 0.325 | 0.501 |
| compact | 64,255 | 62.7 | 0.331 | 0.503 |

Beats our DSCNN MAE (0.325 < 0.333) and marginally beats SOTA RMSE
(0.501 < 0.510). MAE 0.325 is above SOTA MAE 0.298 — LCL is the harder
direction (see caveats on the MAE-vs-RMSE objective).

## Classification (3-class intention)

| Variant | best acc | @ KB | tiny acc | @ KB |
|---|--:|--:|--:|--:|
| with turn indicators | **0.921** | 81.8 | 0.913 | 7.8 |
| without indicators (ablation) | 0.911 | 20.5 | 0.902 | 11.4 |

The full-budget search reaches **92.1%** with the turn signal (at 82 KB,
vs the published Transformer's ~216 KB FP32) — matching the internal reference
(92%) and beating our DSCNN (91.5%). A **7.8 KB** model still gives 91.3%.

## The key finding: the model anticipates, it does not just read the blinker

- turn-signal channels **alone** → 81.5% (the leak we flagged);
- everything **except** the turn signal → **91.1%**;
- everything → 92.1%.

Removing the blinker costs ~1 point. The classifier's accuracy comes from
vehicle dynamics and surrounding traffic, not a declared turn signal. An 11 KB
no-indicator model reaches 90.2% and a 20 KB one 91.1% — both fit the tiny
STM32F401 (96 KB RAM). This is the honest, defensible headline for the paper.

## Quantization (settled empirically on hardware)

`unas/quantize_eval.py` / `quantize_compare.py` / `unas/qat_finetune.py`. The
DMIR inputs have very different per-channel scales, so **full-int8 PTQ degrades
badly** (LCR MAE 0.287→0.449, cls 92.1%→86.9%, cls-noind 91.1%→76.1%). **int16x8**
(int8 weights + int16 activations) preserves accuracy *offline* — but ST Edge AI
does **not** deploy it: it silently dequantizes int16x8 back to float32
(hardware-verified, see `deployment.md`). So the table below is an **offline
bound, not a deployment option** (float → int16x8):

| Deployment model | int16x8 TFLite | metric (float → int16x8) |
|---|--:|---|
| LCR aaaaan | 160.7 KB | MAE 0.2865 → **0.2860** |
| LCR aaaaav (compact) | 108.8 KB | MAE 0.290 → 0.288 |
| LCL aaaaas | 123.0 KB | MAE 0.3249 → **0.3214** |
| cls aaaabl | 117.9 KB | acc 92.08 → **92.15**% |
| cls aaaaat (tiny) | 19.0 KB | acc 91.30 → 91.16% |
| cls-noind aaaaah | 44.6 KB | acc 91.09 → 91.08% |
| cls-noind aaaaaw (tiny) | 36.3 KB | acc 90.16 → 90.15% |

The real deployable quantized point is **QAT int8**: quantization-aware
fine-tuning recovers the classifier to **89.82%** (from 86.9% PTQ) and was
measured on-device at **1.558 ms** on the H7B3I-DK, 7.381 ms on the F401RE
(`unas/qat_finetune.py`; see `deployment.md`). This supersedes the earlier
"int16x8, no QAT needed" plan — the Keras-3/tfmot obstacle was solved (width-1
2D re-expression) and QAT was in fact completed.

## Caveats / TODO

- [ ] **Save policy**: searches used `save_criteria="pareto"`, and chunked
      resume gives each chunk a fresh model-saver, so a good `.h5` from an
      earlier chunk can be pruned (LCL's best is marginally behind a transient
      earlier model). For the definitive front, re-run with
      `save_criteria="all"` (env `DMIR_SAVE_CRITERIA=all`) so no model is
      dropped, then re-harvest. Current fronts are strong but not guaranteed
      globally optimal.
- [ ] Internal-reference RMSE 0.42/0.44 not beaten. The fork trains/thresholds
      on **MAE**; consider an RMSE-aware objective or report MAE as primary
      (where we beat the *published* SOTA). Decide framing with the team.
- [x] Peak RAM, MACs, int8 accuracy drop, and measured latency on the
      STM32H7B3I-DK / F401 — DONE (deployment.md): cls_best float32 3.628 ms,
      int8 PTQ 1.885 ms / 86.9%, int8 QAT 1.558 ms / 89.82%; all fit both boards,
      the reference CNN fits neither the F401. int16x8 hardware-verified as
      non-deployable (silently dequantized).

## Seed study: what survives five retrainings (2026-09-16)

Every number above is one training run of the model the search kept. Each final
architecture was re-initialised and retrained five times (seeds 0-4) under two
recipes, and the hand-built DSCNN baseline (`src/models`, 10.5 k params, global
average pooling → dropout 0.2 → linear head) was retrained with the same seeds:

- **search recipe**: the fork's trainer as the search ran it (Adam, the search
  config's callbacks); `unas/seed_variance.py`, `results/seeds/seed_variance.jsonl`.
- **DSCNN recipe**: the hand-built baseline's recipe (AdamW 3e-3, weight decay 1e-4,
  per-epoch cosine over 60 epochs, patience 10, MSE loss for every regressor);
  `RECIPE=dscnn`, `results/seeds/seed_variance_dscnn.jsonl`. Searched, reference and
  hand-built models then differ only in architecture.
- DSCNN runs: `scripts/run_baseline.py <task> <seed>`, logged as
  `baseline-dscnn-seed<n>` in `logs/experiments.jsonl`.

The reference CNN's original 91.69% came from the colleague's unknown recipe, so its
rows here are controlled retrains.

Three-class intention, test accuracy, mean ± sample std [min, max]:

| model | params | search recipe | DSCNN recipe | single run reported |
|---|--:|--:|--:|--:|
| searched, best accuracy | 83,803 | 91.45 ± 0.59% [90.68, 92.29] | 91.47 ± 0.46% [90.95, 92.19] | 92.08% |
| searched, smallest | 7,953 | 91.19 ± 0.32% [90.74, 91.61] | 91.27 ± 0.32% [90.90, 91.64] | 91.30% |
| reference CNN | 441,347 | 91.33 ± 0.50% [90.99, 92.19] | 91.21 ± 0.73% [90.07, 92.04] | 91.69% |
| hand-built DSCNN | 10,451 | — | 91.50 ± 0.47% [90.91, 91.96] | 91.51% |
| searched, no turn signal | 21,038 | 89.99 ± 0.66% [89.51, 91.11] | — | 91.08% |

Time to lane change, test RMSE / MAE in seconds:

| model | params | search recipe | DSCNN recipe | single run reported |
|---|--:|--:|--:|--:|
| searched, right | 117,404 | 0.483 ± 0.021 / 0.326 ± 0.016 | 0.493 ± 0.019 / 0.344 ± 0.019 | 0.447 / 0.287 |
| hand-built DSCNN, right | 10,321 | — | **0.454 ± 0.008** / 0.324 ± 0.008 | 0.439 / 0.318 |
| searched, left | 105,769 | 0.496 ± 0.012 / 0.351 ± 0.013 | 0.483 ± 0.019 / 0.334 ± 0.013 | 0.466 / 0.317 |
| hand-built DSCNN, left | 10,321 | — | **0.469 ± 0.010** / 0.338 ± 0.005 | 0.459 / 0.333 |

Published Transformer (one model, both directions): RMSE 0.510 / MAE 0.298.

What this changes (Welch's t-test on the five runs, two-sided):

1. **Classification: equal, not better.** Searched 84 k, searched 8 k, the 441 k
   reference and the 10 k hand-built DSCNN all sit between 91.2% and 91.5% under
   either recipe (no pair of these nine groups separates: smallest p = 0.27). The defensible claim is reference-level
   accuracy at a fraction of the deployed cost (84 k: 9.2× faster, 5.2× less flash
   than the reference on the H7B3I-DK; 8 k: 0.79 ms, 37 KB). The search did not find
   a more accurate classifier than a simple hand-built one.
2. **Regression: the hand-built DSCNN is better on RMSE at a tenth of the size.**
   Right: 0.454 against 0.483 (p = 0.035) and 0.493 (p = 0.008); left: 0.469 against
   0.496 (p = 0.005) and 0.483 (p = 0.21, not resolved). MAE does not separate them
   under either recipe (p > 0.07). The recipe is not the explanation: the searched regressors stay behind
   under the DSCNN's own recipe.
3. **Against the published 0.510 s RMSE:** all ten DSCNN runs are below it (worst
   0.480). The searched regressors' means are below it, but one right-lane run per
   recipe is above (0.518, 0.524). On MAE the published 0.298 is lower than every mean.
   The comparison stays indicative (one published model for both directions, 30
   channels, free-ride windows included).
4. **The turn-signal ablation holds:** 91.45 → 89.99% without the two channels
   (p = 0.006), against 81.5% for the turn signal alone.
5. **Why the searched models do not win.** The µNAS 1D space ends in an optional
   pool of size 2-6, Flatten, at least one hidden dense layer and the output layer,
   with no dropout; the DSCNN's global-pooling head is outside it (see
   `datasets/highd/docs/baseline-results.md`, seed study, item 3). Single-run scoring
   adds selection noise on top: on highD the reported run and the five-seed mean of
   17 saved classifiers have rank correlation −0.19.
6. **The saved fronts are not guaranteed Pareto sets.** The fork's saver restarts
   its file counter in every resumed chunk, so later files overwrite earlier ones and
   the metadata stops describing the files (verified on highD, same `run_chunked`
   pattern here). All numbers in this document come from evaluating or retraining
   the saved files, so they stand; the word "front" should be read as "saved models".

Deployment measurements (latency, flash, RAM) depend on the graph, not the weights,
and are unaffected. The hand-built DSCNN has not been measured on the boards yet.

## v4 searches with the faithful cost count (2026-09-25)

The first searches optimized the fork's resource count, which leaves out convolution
padding (unas/README.md; on LCIR the true MACs were a median 1.25-1.52 times the counted
ones, up to 6.3). The v4 searches (`dmir_cls_v4`, `dmir_lcr_v4`, `dmir_lcl_v4` in
`unas/dmir_config.py`) use the global-average-pooling space with the faithful count, budgets at
the DSCNN layer sequence's cost under that count (10,387 B and 169,872 MACs for intention,
10,257 B and 169,744 MACs for TTLC), error bounds at the DSCNN's validation level (0.06, and
0.44 / 0.48 s RMSE with an RMSE objective), 150 candidates each (86, 81 and 69 saved), and the
five-seed validation choice (`unas/select_by_seeds.py`). Test results over five seeds:

| task | v4 choice | search recipe | DSCNN recipe | DSCNN (10 k) | first searched model |
|---|---|--:|--:|--:|--:|
| intention | 6,528 params | 91.43 ± 0.32% | 91.21 ± 0.24% | 91.50 ± 0.47% | 8 k: 91.19 ± 0.32% |
| LCR | 4,663 params | RMSE 0.476 ± 0.012 s | 0.462 ± 0.005 s | 0.454 ± 0.008 s | 117 k: 0.483 ± 0.021 s |
| LCL | 3,652 params | RMSE 0.463 ± 0.007 s | 0.479 ± 0.020 s | 0.469 ± 0.010 s | 106 k: 0.496 ± 0.012 s |

Welch tests against the DSCNN: intention p = 0.79; LCL p = 0.30 (as accurate); LCR p = 0.013
under the search recipe and 0.10 under the DSCNN recipe. Against the first searched models:
LCL better (p = 0.002, search recipe), LCR better under the DSCNN recipe (p = 0.022). The
chosen regressors are one or two layers: LCR a stride-2 convolution (42 filters) and average
pooling; LCL a stride-2 depthwise convolution, a stride-2 convolution (33 filters) and average
pooling. The intention model starts with a 1x1 convolution of 87 filters over all 50 steps,
which holds most of its 146 k MACs. Board results are in deployment.md.

