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
