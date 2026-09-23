# highD baseline results (2026-09-01)

Model: `src.models.baseline.BaselineDSCNN` (n_features=18), **8,371 params**,
input 10x18 windows, AdamW 3e-3, batch 256, early stopping on val. Trained on
RTX 5070 laptop, ~30 s per run. Runs logged in `logs/experiments.jsonl`
(`highd_baseline_cls_v2`, `highd_baseline_ttlc_v2`).

## Under their exact metric definitions (eval_their_protocol.py)

vs Table III of Mozaffari et al., T-IV 2022 (test split; theirs 698 scenarios,
ours 693 — see DATA.md):

| metric | ours (8.4k, TTLC 8.2k, state features) | their proposed (BEV+attention CNN) | their best baseline |
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
Edge AI measurement pipeline, an axis their Table III does not report.

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
@ 27.7k. The 8.2k TTLC baseline (MAE 0.169 / RMSE 0.276) is edged on accuracy at
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
against 5.2 pt on DMIR, because the min-max-normalized inputs quantize better
than DMIR's wide-dynamic-range channels. The interface change is again
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

Footprint: MACC 34,002 · flash 27,438 B (weights 7.94 KiB, 3.54x smaller; but the library grows ~6 -> ~19 KiB, the int8-kernel overhead that dominates tiny
models, same pattern as DMIR) · **RAM 8,616 B** (activations 7.51 KiB + 924 B
library). Note RAM is HIGHER than the f32 build's 5,328 B: at this scale the
int8 kernels' scratch outweighs the activation-precision saving, and the
float32 interface still pays the conversion_0/conversion_19 casts visible in
the per-layer chart. The int8-I/O variant is where the interface cost goes
away. SVG: `results/deploy/highd_cls_model_aaaaam_int8.tflite.svg`.

`highd_cls_model_aaaaam_int8_io.tflite` (int8 tensor interface — badge
STAI_FORMAT_S8, 10x18 8-bit in / 3 8-bit out; test acc 89.10%):

| board | measured inference |
|---|--:|
| STM32H7B3I-DK | **0.4012 ms** (fastest model measured in the project) |
| NUCLEO-F401RE | **2.111 ms** |

Footprint: MACC 33,636 (both conversion casts gone, absent from the per-layer
chart) · flash 27,114 B · **RAM 8,616 B, unchanged vs the float32-interface
build.** On DMIR the int8 interface cut RAM
8,096 -> 5,444 B because the 6,200 B float input buffer *was* the binding
allocation; here the input is only 10x18 (720 B float), the peak is set by the
7.51 KiB of internal activations, and shrinking the input buffer moves nothing.
Confirms the max-over-live-tensors model of ST's allocator from the DMIR
campaign: **int8 I/O saves RAM only when the input buffer binds the peak** —
it still saves latency (-4%) and the casts regardless.

## Measured via the ST Edge AI REST API (2026-09-02, batch of 8)

Automated through ST's official Developer Cloud Python client (same engine:
Core 4.0.1, balanced, allocate I/O true). Raw records:
`results/deploy/benchmarks_api.jsonl`.

**Re-verification of the manual measurements** — the API re-ran two artifacts
measured by hand earlier the same day:

| artifact | manual | API | delta |
|---|--:|--:|--:|
| cls f32 @ H7B3 | 0.7614 ms | 0.7641 ms | +0.35% |
| cls int8-IO @ H7B3 | 0.4012 ms | 0.4016 ms | +0.09% |

Board-farm repeatability is under 0.4%, and the manually pasted numbers are
confirmed accurate.

**TTLC winner (model_aaaaaw, 27.7k params), all variants, both boards:**

| variant | H7B3I-DK | F401RE | ROM (H7B3) | RAM (H7B3, API) | MACC |
|---|--:|--:|--:|--:|--:|
| f32 | 1.0382 ms | 5.0578 ms | 117,438 B | 3,004 B | 47,070 |
| int8 | 0.6744 ms | 2.9922 ms | 55,233 B | 11,040 B | 46,777 |
| int8-IO | **0.6576 ms** | **2.9396 ms** | 54,933 B | 10,740 B | 46,415 |

Same shapes as the classifier: int8 1.54-1.72x faster, the int8 interface adds
a further ~2.5% and removes the casts, and **int8 RAM exceeds f32 RAM**
(3,004 -> ~11 kB) — the int8-scratch-dominates-at-small-scale pattern, even
more pronounced here. fp32 cycles/MAC = 6.2, consistent with every fp32
measurement in the project. Accounting note: the API's ram_B differs slightly
from the web UI's RAM total for the same artifact (e.g. cls int8-IO 9,924 vs
8,616 B) — different inclusion of I/O/library buffers; the jsonl keeps the
API's separate ram_io fields, and cross-tool comparisons should stick to one
source.

## Final classification winner: model_aaaaap (tight-bound search, 5,347 params)

Measured via the ST API (Core 4.0.1, balanced), records in benchmarks_api.jsonl:

| variant | acc | H7B3I-DK | F401RE | ROM | RAM | MACC |
|---|--:|--:|--:|--:|--:|--:|
| f32 | 91.15% | 0.1547 ms | 0.6960 ms | 25,096 B | 1,368 B | 5,855 |
| **int8-IO** | **91.04%** | **0.1063 ms** | **0.4670 ms** | **15,329 B** | 3,088 B | 5,635 |

The search also cut compute: this architecture needs **5.9k MACs**, 6x fewer
than model_aaaaam (34k), at higher accuracy. Combined
statement: a classifier that beats the published Table III proposed model by
8 accuracy points runs at ~9,400 inferences/s on a Cortex-M7 and ~2,100/s on
an entry Cortex-M4, in 15 KB of flash. It is 13.5x faster than the fastest
DMIR deployment point (1.435 ms) and the fastest artifact in the project.

The int8-vs-f32 RAM inversion appears a third time (1,368 -> ~3 kB).

## The searched classifier under their metric definitions (2026-09-08)

`eval_their_protocol.py` scored only the hand baseline. Applying the same
transcribed formulas to the tight-search winner (`results/models/
highd_cls_tight_model_aaaaap.h5`, 5,347 params; record in
`results/their_protocol_eval_model_aaaaap.json`):

| metric | hand CNN 8.4k | searched 5.3k | their proposed |
|---|--:|--:|--:|
| accuracy | 0.911 | 0.912 | 0.83 |
| precision | 0.923 | 0.966 | 0.85 |
| recall | 0.938 | 0.896 | 0.85 |
| F1 | 0.930 | 0.930 | 0.85 |
| AUC | 0.959 | 0.962 | 0.88 |
| τ_f (first correct) | 4.93 s | 4.73 s | 4.75 s |
| τ_c (robust) | 4.79 s | 4.53 s | 3.96 s |

Same F1 by a different route: the searched model trades recall for precision
(fewer false lane-change alarms, later first correct call), so its robust
prediction horizon is 0.26 s shorter than the baseline's. Both remain well
above the published 3.96 s.

## Seed study: the searched classifier does not beat the hand-built CNN (2026-09-16)

Every number above comes from one training run. The final models were retrained
from scratch with five seeds (`unas/seed_variance.py` for the searched Keras
models, `train_highd.py <task> <run> <seed>` for the hand-built PyTorch CNN), and
the searched models a second time under the hand-built model's recipe (AdamW
3e-3, weight decay 1e-4, at most 60 epochs, early stopping patience 8), so that
searched and hand-built differ only in architecture. Records:
`results/seeds/seed_variance.jsonl`, `results/seeds/seed_variance_final.jsonl`,
`logs/experiments.jsonl` (`highd_baseline_*_seed*`).

Test accuracy, mean ± sample std over five seeds [min, max]:

| model | params | search recipe | hand-built's recipe | single run reported |
|---|--:|--:|--:|--:|
| searched, tighter search (model_aaaaap) | 5,347 | 88.89 ± 2.90% [84.41, 91.72] | 89.19 ± 2.20% [86.59, 92.27] | 91.15% |
| searched, first search (model_aaaaam) | 7,904 | 90.80 ± 0.79% [89.72, 91.56] | 90.84 ± 0.75% [89.66, 91.63] | 90.53% |
| hand-built CNN | 8,371 | — | **92.11 ± 0.71%** [91.28, 92.99] | 91.09% |

Time-to-lane-change test RMSE (s):

| model | params | search recipe | hand-built's recipe | single run reported |
|---|--:|--:|--:|--:|
| searched regressor (model_aaaaaw) | 27,719 | 0.276 ± 0.011 [0.259, 0.286] | 0.316 ± 0.074 [0.270, 0.447] | 0.261 |
| hand-built CNN | 8,241 | — | 0.276 ± 0.009 [0.270, 0.292] | 0.276 |

What this changes:

1. **Every model and every seed still beats the published model** (accuracy 0.83,
   RMSE 0.629). That claim holds.
2. **The claim that the search beats the hand-built CNN does not hold.** The
   tighter-search winner's 91.15% was a favourable run: its five-seed mean is
   2.3 points lower, and its spread is four times the hand-built model's. The
   search scores each candidate with one run, so the candidate it keeps is partly
   the one whose run went well. The same recipe gives the same gap, so the recipe
   is not the explanation. On the regression task the searched model only ties
   the hand-built one, at 3.3 times the parameters.
3. **A likely structural cause: the hand-built model lies outside the search
   space.** The µNAS 1D space (`cnn1d_schema.py`, `cnn1d_architecture.py`) ends in
   an optional average or max pool of size 2, 4 or 6, a Flatten, at least one
   hidden Dense layer of 10-256 units and the output layer, with no dropout. The
   pool acts globally only when the sequence is already down to the pool size or
   shorter, and a hidden Dense layer always follows it. The hand-built CNN ends in
   global average pooling and dropout straight into the output layer, so the
   search could not have produced it. The dense head is also what dominates
   runtime in the profiled DMIR models (82-93% for the classifiers).
4. Two further observations. The model kept by the tighter search differs from
   a candidate that scored 83.0% (model_aaaaan, same 5,347 parameters) only in
   one pooling layer (max against average). And repeating a seed does not repeat
   the result exactly (seed 0 gave 84.4% and 86.0% in two runs): GPU kernels are
   not deterministic, so seeds fix initialisation and data order, not the run.

### Re-ranking every saved classifier by its seed mean

`unas/rerank_front.py` retrained all 17 saved classifiers from both searches whose
reported test accuracy was at least 88.5%, five seeds each under the search recipe
(`results/seeds/rerank_cls.jsonl`, 85 runs).

| params | search | reported | 5-seed mean ± std | [min, max] |
|--:|---|--:|--:|--:|
| 5,347 | tighter (model_aaaaap) | 91.15% | 88.66 ± 1.98% | [85.99, 90.82] |
| 7,904 | first (model_aaaaam) | 90.53% | 90.65 ± 1.12% | [89.26, 91.82] |
| 9,010 | first (model_aaaaap) | 88.96% | 90.70 ± 0.54% | [90.17, 91.33] |
| 11,849 | first (model_aaaaar) | 88.78% | 89.92 ± 1.03% | [88.38, 90.99] |
| 15,073 | tighter (model_aaaaag) | 89.43% | 90.11 ± 0.82% | [88.98, 91.21] |
| 28,805 | tighter (model_aaaaaq) | 88.68% | 89.25 ± 0.47% | [88.54, 89.66] |
| 38,166 | tighter (model_aaaabk) | 89.34% | 88.43 ± 1.42% | [85.99, 89.61] |
| 44,117 | tighter (model_aaaabs) | 89.55% | 89.75 ± 0.57% | [88.82, 90.34] |
| 62,991 | tighter (model_aaaaak) | 88.95% | 90.64 ± 0.80% | [89.72, 91.53] |
| 63,741 | tighter (model_aaaaae) | 89.28% | 91.11 ± 0.69% | [90.07, 92.00] |
| 64,067 | tighter (model_aaaaaz) | 90.67% | 90.20 ± 1.41% | [87.90, 91.50] |
| 69,344 | tighter (model_aaaaas) | 88.81% | 90.36 ± 0.76% | [89.37, 91.45] |
| 74,649 | tighter (model_aaaaay) | 88.82% | 91.26 ± 0.38% | [90.86, 91.76] |
| 79,987 | tighter (model_aaaaah) | 89.52% | **91.28 ± 0.90%** | [90.18, 92.10] |
| 80,379 | tighter (model_aaaaau) | 89.26% | 90.78 ± 1.58% | [88.66, 92.45] |
| 88,546 | tighter (model_aaaaam) | 90.45% | 90.11 ± 0.67% | [89.22, 91.05] |
| 89,850 | tighter (model_aaaaac) | 88.54% | 90.72 ± 1.19% | [89.62, 92.38] |
| 8,371 | hand-built CNN (its recipe) | 91.09% | **92.11 ± 0.71%** | [91.28, 92.99] |

1. **The single run carried no ranking information.** Spearman correlation between
   reported accuracy and five-seed mean is −0.19 over the 17 models; the model with
   the best single run (5,347 params, 91.15%) ranks 16th of 17 by its mean.
2. **No saved classifier reaches the hand-built CNN.** The best means (91.3% at
   80 k and 75 k) are 0.8 points below it at nine to ten times its parameters.
   Five seeds do not resolve that gap (Welch p = 0.145 and 0.055), so the defensible
   statement is that the search found no classifier more accurate than the
   hand-built one, not that all are worse. The small tighter-search winner is
   clearly behind (88.66 ± 1.98%). The smallest model with a stable mean is
   9,010 params at 90.70 ± 0.54%.
3. On average the retrains score 0.8 points *above* their reported runs, the
   opposite of what selection alone would produce. The search recipe and the retrain
   recipe are the same code path, so this is not settled; see the next section for
   why the reported runs are not reliable references for the saved files.

### The saved search files do not match the search's own records

Evaluating saved files against the fork's `metadatas.json` showed mismatches:
`highd_cls_tight/model_aaaaan.h5` scores 83.5% on validation while its metadata row
says 90.3%; `model_aaaaap.h5` scores 92.4% against 90.5%. Two causes in the fork's
`ModelSaver`:

- it restarts its file counter in every process (`self.iteration = 0`, empty
  `stored_models`), so a resumed chunk writes `model_aaaaaa.h5` again over the first
  chunk's file (the `highd_cls` chunk 2 log restarts at "Stored models: 1");
- with `HIGHD_PARALLEL=2` each Ray worker holds its own saver, so two workers write
  the same names into one folder, and `metadatas.json` keeps whichever list was
  flushed last.

Separately, the fork's logged `test_error` agrees with `val_error` to about 1e-4 for
every candidate, although the saved files differ by one to four points between the
two splits. It was not traced further.

What this does and does not affect: every accuracy, error and board number in this
project comes from evaluating or retraining the saved files themselves
(`harvest_highd.py`, `seed_variance.py`, `rerank_front.py`), so those numbers hold.
What does not hold is that the saved files form the search's Pareto front, or that
the fork's metadata describes them. The DMIR searches ran through the same chunked
saver (`run_chunked.sh`), so the same caveat applies there. Fix before any new
search: persist the counter and the stored list across chunks, give each worker
unique file names, and store the evaluated validation accuracy inside each file.

Open decisions for the paper: report seed means throughout; decide whether to extend
the search space with a global-pooling head (optional hidden layer, dropout) and
re-run the searches with the saver fixed and the final front scored over seeds.

## Note on the Transformer latency used in the highD replay

The replay (T4.5 deck) quotes 368.82 ms for "a reference-size Transformer". That
is the DMIR regression Transformer, built for 50 time steps × 31 features. The
same architecture on the highD input (10 × 18) needs 5.0× fewer floating-point
operations (29.7 M against 5.9 M, counted with the TensorFlow profiler on the
rebuilt graph; parameters 333,505 against 325,025). Board latency does not follow
operation counts exactly, so the highD figure is unmeasured: do not publish the
12 m distance for highD without measuring a highD-shaped Transformer on the board.
