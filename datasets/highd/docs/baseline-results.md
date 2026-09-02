# highD baseline results (2026-09-01)

Model: `src.models.baseline.BaselineDSCNN` (n_features=18), **8,371 params**,
input 10x18 windows, AdamW 3e-3, batch 256, early stopping on val. Trained on
RTX 5070 laptop, ~30 s per run. Runs logged in `logs/experiments.jsonl`
(`highd_baseline_cls_v2`, `highd_baseline_ttlc_v2`).

## Under their exact metric definitions (eval_their_protocol.py)

vs Table III of Mozaffari et al., T-IV 2022 (test split; theirs 698 scenarios,
ours 693 — see DATA.md):

| metric | ours (8.4k, state features) | their proposed (BEV+attention CNN) | their best baseline |
|---|--:|--:|--:|
| accuracy | **0.911** | 0.83 | 0.79 (LSTM1) |
| recall | **0.938** | 0.85 | 0.90 (LSTM1) |
| precision | **0.923** | 0.85 | 0.94 (MLP1) |
| F1 | **0.930** | 0.85 | 0.82 |
| AUC | **0.959** | 0.88 | 0.86 |
| tau_f (first pred) | **4.93 s** | 4.75 s | 4.43 s |
| tau_c (robust pred) | **4.79 s** | 3.96 s | 3.76 s |
| TTLC RMSE | **0.276 s** | 0.629 s | 0.841 s (LSTM1) |

Early-prediction profile (per-TTLC-bucket accuracy on LC windows, our eval):
99.5% (0-1 s), 99.9% (1-2 s), 99.9% (2-3 s), 98.6% (3-4 s), 80.6% (4-5 s).

## Read this before quoting

- **These are our reimplementation's numbers, not a re-run of their code.**
  Scenario extraction is validated exactly on two of three splits; metric
  formulas are transcribed from their `utils.py`; still, the honest claim is
  "under our faithful reimplementation of their protocol", not "on their
  benchmark harness".
- The margin is large partly because their baselines are 2021-era MLP/LSTM
  training setups; a small temporal CNN with BN + AdamW + longer early-stopping
  patience is simply a stronger function class on 18-feature windows. Their
  proposed model's advantage was BEV imagery over weak state baselines.
- Run-to-run variance: two identical cls runs gave 91.65% / 91.09% test acc
  (cuDNN nondeterminism). Quote ~±0.3 pt.
- The 5-s horizon boundary is intrinsically ambiguous: a window exactly 5 s
  before a crossing appears as LK (from the LK scenario of the same track) and
  as LC (from the LC scenario) — an irreducible error floor shared with their
  setup.

## What this sets up

The baseline alone already exceeds every published Table III number at 8.4k
params. The NAS searches (`unas/run_chunked_highd.sh`, configs highd_cls /
highd_ttlc, bounds 32 KB / 32 KB / 500k MACs) then map the accuracy-size
frontier below that, and the winners go through the existing int8/QAT + ST
Edge AI measurement pipeline — the deployment-cost axis nobody in this
literature reports.

## NAS search results (2026-09-02, overnight, 150 rounds each)

Both aging-evolution searches ran to completion on the RTX 5070 (pop 50,
sample 15, bounds 32 KB peak-mem / 32 KB int8 size / 500k MACs; fronts in
`results/nas-fronts/`, re-evaluated independently on real test windows —
the fork's console test_error tracks val_error and is never quoted).

**Classification** (13 Pareto models, 7.9k–38k params): best searched point
90.53% window accuracy @ 7,904 params. The hand baseline (91.1% @ 8,371)
still leads: with the 0.07 error bound satisfied at ~93% val accuracy, fitness
pressure goes entirely to size, so the front tilts small instead of accurate.
Lever for a follow-up: tighten `HIGHD_CLS_ERROR_BOUND`.

**TTLC regression** (25 models after the salvage resume): the first 150-round
run saved zero models — every candidate's val MAE (best 0.1622) sat above the
0.16 error bound and `model_saver.py` drops out-of-bound models. Salvage
resume (+bound 0.20, save-all) captured the population. Best searched points
on test: MAE 0.1624 / RMSE 0.2614 @ 80k params, and MAE 0.1656 / RMSE 0.2608
@ 27.7k. The 8.4k baseline (MAE 0.169 / RMSE 0.276) is edged on accuracy at
3–10x the size — per-parameter the baseline still wins.

Context: every model in these fronts, baseline included, sits far under the
published TTLC RMSE of 0.629 s.

Full 150-candidate ttlc history (recovered from the search state after the
runner overwrote chunk-1 logs): `results/nas-fronts/highd_ttlc_history.csv`.

## Deployment artifacts (2026-09-02, ready for ST Edge AI measurement)

`results/deploy/` — search winners exported to full-int8 PTQ TFLite in both
tensor interfaces, each variant evaluated on real test windows with the
interface-honouring evaluator:

| artifact | bytes | test metric |
|---|--:|---|
| cls model_aaaaam f32 | 38,440 | acc 90.53% |
| cls model_aaaaam int8 | 26,280 | acc 89.10% |
| cls model_aaaaam **int8-I/O** | 25,952 | acc 89.10% |
| ttlc model_aaaaaw f32 | 125,320 | MAE 0.166 / RMSE 0.261 |
| ttlc model_aaaaaw int8 | 55,992 | MAE 0.187 / RMSE 0.280 |
| ttlc model_aaaaaw **int8-I/O** | 55,648 | MAE 0.187 / RMSE 0.280 |

Observations vs the DMIR campaign: PTQ costs the classifier only 1.4 pt here
(DMIR: 5.2) — the min-max-normalized inputs are far gentler on int8 than
DMIR's wide-dynamic-range channels — and the interface change is again
accuracy-neutral. Regression pays relatively more (+13% MAE), the same task
asymmetry seen on DMIR, so QAT is deferred: unnecessary for cls at 1.4 pt,
and it failed to rescue regression on DMIR.

Upload list for the boards (H7B3I-DK first, F401 fits trivially):
`highd_cls_model_aaaaam_int8_io.tflite`, `highd_ttlc_model_aaaaaw_int8_io.tflite`,
plus the f32 pair for like-for-like float measurements.

## Measured on-device (2026-09-02, ST Edge AI Developer Cloud)

`highd_cls_model_aaaaam_f32.tflite` (7,904 params, test acc 90.53%):

| board | measured inference |
|---|--:|
| STM32H7B3I-DK (M7 @ 280 MHz) | **0.7614 ms** |
| NUCLEO-F401RE (M4 @ 84 MHz) | **4.161 ms** |

Per-layer profile (both boards): the first conv (conv2d_1) dominates runtime
by a wide margin while conv2d_7 holds the most weights — the usual
early-layer-compute / late-layer-memory split. Report SVG:
`results/deploy/highd_cls_model_aaaaam_f32.tflite.svg`.
Footprint (Core 4.0.1-20581, balanced, allocate inputs/outputs true):
**MACC 34,220 · flash 35,010 B** (weights 28.14 KiB + ~6 KiB library) ·
**RAM 5,328 B** (activations 4.92 KiB + ~288 B library; I/O buffers 0/0).
Badge STAI_FORMAT_FLOAT, input 10x18 float32.

Context: DMIR cls_tiny (8k params, 50x31 input) measured 0.7931 ms on the same
M7; this model (7.9k params, 10x18 input) lands at 0.7614 ms — consistent scale.

`highd_cls_model_aaaaam_int8.tflite` (int8 weights, float32 I/O — badge
STAI_FORMAT_FLOAT; test acc 89.10%):

| board | measured inference |
|---|--:|
| STM32H7B3I-DK | **0.4181 ms** (1.82x vs f32) |
| NUCLEO-F401RE | **2.168 ms** (1.92x vs f32) |

Footprint: MACC 34,002 · flash 27,438 B (weights 7.94 KiB — 3.54x smaller —
but library grows ~6 -> ~19 KiB, the int8-kernel overhead that dominates tiny
models, same pattern as DMIR) · **RAM 8,616 B** (activations 7.51 KiB + 924 B
library). Note RAM is HIGHER than the f32 build's 5,328 B: at this scale the
int8 kernels' scratch outweighs the activation-precision saving, and the
float32 interface still pays the conversion_0/conversion_19 casts visible in
the per-layer chart. The int8-I/O variant is where the interface cost goes
away. SVG: `results/deploy/highd_cls_model_aaaaam_int8.tflite.svg`.
