# Research logbook — DMIR × µNAS

Dated journal of decisions, findings, and results. Machine-readable run
records live in `datasets/dmir/logs/experiments.jsonl`; this file records the *why*.

## 2026-07-07 — Project kickoff: environment, data audit, first baseline

**Environment.** Windows 11, Python 3.13 (Microsoft Store), RTX 5070 Laptop
(8 GB, Blackwell) → PyTorch 2.11.0+cu128. Windows Smart App Control blocked
several compiled wheels inside the venv (numpy.random, pandas, scikit-learn);
resolved by enabling `include-system-site-packages` and installing the blocked
packages into the system Python. Full details in CLAUDE.md.

**Data audit (before any modelling).** Extracted the three prepared archives
(classification, regression-LCL, regression-LCR); shapes and split sizes match
the provider's description (50×31 windows; balanced 3-class splits;
regression targets 0.0–4.0 s, step 0.1). Two findings:

1. Test-only extreme spikes (up to ~5×10⁶) on feature pairs (12,13) for
   classification/LCL and (14,15) for LCR — 56/37/21 affected samples,
   train/val stay within ≈[−431, 140]. Likely a division-by-near-zero
   preprocessing artefact. Mitigation: clip val/test to per-feature train
   range (`Config.clip_to_train_range`); training data untouched.
   Question sent upstream — see datasets/dmir/docs/DATA.md.
2. Feature 7 constant zero in classification train split.

**Pipeline.** Built `src/` (config, data, baseline DSCNN, train/eval, EDA,
logging), `scripts/check_pipeline.py` (3-task smoke test on real data —
passing), and the main notebook `notebooks/dmir_pipeline.ipynb` (executes
end-to-end headless, including ONNX export with PyTorch↔ONNX parity check).

**First baseline result (logged run `baseline-dscnn`).** Generic
depthwise-separable 1D CNN, no tuning: **test accuracy 91.51%, macro-F1
0.9153** (SOTA: 92%), best epoch 3, training time 27.5 s. Per-class recall
[0.904, 0.960, 0.881] — LCL intention is the hardest class. Early convergence
suggests the schedule (lr 3e-3, cosine) peaks too fast; not tuning further by
hand — that is the NAS's job.

**Regression baselines (same untuned DSCNN).** LCL: test RMSE 0.459 s / MAE
0.333 s (SOTA 0.44). LCR: test RMSE 0.439 s / MAE 0.318 s (SOTA 0.42). Both
within ~0.02 s of the published numbers with zero tuning — the search has a
realistic shot at passing them while shrinking the model.

**Research sweep (6 parallel agents; notes in docs/research/).** Key
corrections and facts:

1. *The published baseline is not what we assumed.* Forneris et al., IEEE SPL
   vol. 33, 2026 (DOI 10.1109/LSP.2025.3638676) reports a single
   time-to-lane-change regression — Transformer RMSE **0.5102 s** (~54k
   params) — and FP32-only deployment on STM32H7B3/F401. It publishes no
   3-class accuracy and no per-direction RMSE. The 92% / 0.42 / 0.44 numbers
   from the team are internal, unpublished results on our prepared pickles.
   Both now tracked separately in paper/NOTES.md.
2. *DMIR is the project codename.* The data is the "Lane Change Intention
   Recognition Dataset" (Zenodo 10.5281/zenodo.16686054, MIT, CARLA, 50
   drivers, 10 Hz; DMIR = Driver Maneuver Intention Recognition, the acronym
   from the ApplePies 2024 precursor). Official split is driver-wise; whether
   our pickles follow it is an open (blocking) question.
   The 31st channel is probably fileTime (official count: 30 features) —
   must confirm and drop.
3. *µNAS method mapped* (aging evolution, morphisms, 4-objective random
   scalarisation over acc/RAM/flash/MACs; MACs↔latency R²=0.975 on STM32).
   Official repo is TF2.3, 2D-only, unlicensed → we implement a 1D PyTorch
   version. Closest related work to differentiate from: MicroNAS for time
   series (Sci. Reports 2025) and TinyTNAS.
4. *STM32 toolchain*: ST Edge AI Core 2.2 (`stedgeai`) accepts int8 QDQ ONNX
   (quantize via onnxruntime `quantize_static`); `analyze` gives flash/RAM/
   MACC offline → usable inside the NAS constraint evaluator; real latency
   free via ST Edge AI Developer Cloud board farm. Plan: mirror the baseline
   boards (H7B3 + F401).
5. *Venue*: SPL is Q1 but caps at 4 pages + references; IEEE IoT Journal
   (IF 8.7, 8 pages, ~7-week first decision) recommended as primary
   alternative. Decision deferred to supervisor discussion.

**Next.** Colleague answers on split/column order → drop fileTime if
confirmed → design 1D search space + aging-evolution loop with stedgeai-based
constraint evaluation → NAS smoke run.

## 2026-07-07 (later) — GitHub, hardware target, course website, PDF pipeline

- Repository live: github.com/SepehrMohammady/LC-Intention-NAS (author
  identity rewritten to Sepehr Mohammady on all commits before first push).
- Deployment target confirmed by the team: **STM32H7B3I-DK** (Cortex-M7 @
  280 MHz, 1.4 MB SRAM, 2 MB flash) — same H7B3 family as the baseline
  paper's high-end board, so deployment tables stay directly comparable.
  NAS budgets updated in docs/research/stm32-toolchain.md; F401 kept as an
  optional low-end stretch target.
- The Farsi course became a static website (course/index.html + lessons
  00–04): RTL layout, light/dark themes, SVG diagrams, CSS bar charts on the
  dataviz reference palette, and an interactive quiz per lesson. Markdown
  lessons removed; HTML is canonical. Serving: enable GitHub Pages (main,
  root) — the root index.html redirects into the course.
- Paper now builds to PDF after every .tex change (MiKTeX pdflatex via
  scripts/build_paper.ps1; main.pdf committed). Author set to Sepehr
  Mohammady; H7B3I-DK written into the deployment section.

## 2026-07-07 (evening) — Blocking data questions answered from raw materials

New materials supplied by the team: `user46(04.12.24).zip` (raw CARLA logs),
`Overtaking2.zip` (the data-prep repo: 73 per-user H5 sessions in L3Pilot CDF,
windowing notebook, schema), and the ApplePies 2024 precursor paper PDF
(DMIR = Driver Maneuver Intention Recognition; CARLA 0.9.15; Logitech G920;
60 km 2-lane 11.5 m highway, mean curve radius 500 m; Hi-Drive grant
101006664). Two analyses (`scripts/analysis/`) resolved everything that was
blocking without waiting for colleagues:

1. **Channel fingerprinting** (`fingerprint_channels.py`): no fileTime /
   timestamp channel exists in any pickle (no monotonic near-unique channel);
   classification feature 7 = egoLaneWidth (constant 3.75 m → 0 after
   scaling); the test-split spike pairs are exactly-equal right/left pairs at
   the curvatureDx positions — (12,13) cls/LCL, (14,15) LCR — confirming the
   derivative-artefact hypothesis. Classification normalization ≈
   StandardScaler on train; LCR uses a different scaler fit and a shifted
   channel layout (egoLaneWidth at 9, indicators mid-block).
2. **Split verification** (`verify_split.py`): matched raw per-user H5
   sessions to pickle windows via per-window Pearson correlation (invariant
   to the normalization). Official test users 13 & 2 hit ONLY the test split,
   val users 10 & 5 ONLY val, train user 22 ONLY train (YawRate /
   LatAcceleration, r > 0.9999; SteeringAngle cross-hits are quantization
   false-positives). **The pickles follow the official driver-wise protocol —
   our results are directly comparable to the published RMSE 0.5102, with no
   window leakage.** This claim went into the paper's Data section.

Remaining for the team (non-blocking): provenance of the internal 92%/0.42/0.44
reference numbers (colleague: 92% is a CNN; others are Transformers — asking).

## 2026-07-07 (night) — Feature identities confirmed + turn-indicator leak

Colleague sent the authoritative `feature_description` doc and confirmed:
driver-wise split; `fileTime` is not in the arrays; the 31st channel is
"is the left car present?" (`car2Present`); `egoLaneWidth` is constant and
kept deliberately. The raw `DirectionIndicator` is ternary {0 off, 1 left,
2 right}, split into two binaries in the prepared data.

Verified the full channel map (`scripts/analysis/name_channels.py`: ego
channels 0-7 matched to raw H5 signals at r ≈ 0.98) and wrote it up in
datasets/dmir/docs/feature-map.md + machine-readable src/features.py. The two task
layouts differ: regression keeps indicators inline (ch 3-4); classification
relocates them to the end (ch 28-29), shifting egoLaneWidth to 7 and the
curvatureDx spike pair to 12-13.

**Turn-indicator label leak (paper-shaping).** The feature set includes the
driver's turn signal. `indicator_leak.py`: the two indicator channels ALONE
give 81.5% test accuracy (full DSCNN 91.5%, internal ref 92%); blinker-on
rate 92%/71% for LCR/LCL vs 6.5% for no-intent. So the 3-class headline is
largely "read the blinker." The regression is clean, though
(`indicator_leak_regression.py`: indicator-only RMSE 0.72/0.90, far worse
than our 0.439/0.459). Decision: lead the paper with the time-to-lane-change
regression + STM32 deployment; treat classification with a with/without-
indicator ablation and a no-signal-subset accuracy. Recorded in paper/NOTES.md.

## 2026-07-08 — NAS tool decided: reuse the ELIOS µNAS fork

Colleague pointed to github.com/Elios-Lab/uNAS (their keywords: "take µNAS
code", "efficiency", "setting threshold"). Inspected it: the fork adds, over
upstream, a **1D multi-channel CNN search space** (takes our (50,31) directly),
**regression** (num_classes=1, MAE loss), **QAT + INT8 TFLite** output for the
ST tools, and CLI-overridable **BoundConfig** thresholds. Aging-evolution
fitness = -max_i( normalise(feature_i, bound_i)/lambda_i ) over
[val_error, peak_mem, model_size, MACs] — so "efficiency" = the resource
objectives and "setting threshold" = the BoundConfig bounds (our STM32 budget).

Decision: **reuse the fork** (matches the colleague's directive; gives a direct
TFLite→ST deployment path) instead of reimplementing µNAS in PyTorch. Wrote our
adapters in `unas/` (DMIR dataset serving the real pickles for all 3 tasks with
train-range clipping + optional indicator-drop ablation; configs with
STM32H7B3I-DK thresholds) — MIT, kept separate from the unlicensed fork.
Decision + plan in docs/research/unas-integration.md.

Compute path chosen (user): WSL2/Linux GPU.

## 2026-07-08 — WSL2 GPU env stood up; RTX 5070 (Blackwell) runs the search

Built the NAS environment under WSL2 Ubuntu 22.04 (driver 610.62, GPU
passthrough, Python 3.10.12). Fresh venv `~/dmir_nas` with
`tensorflow[and-cuda]` → **TF 2.21.0 + CUDA 12.9 + cuDNN 9.24 + Keras 3.12**;
TF registers the GPU as "compute capability 12.0a" (sm_120) and runs matmul +
Conv1D on our (N,50,31) shape. The Blackwell question is settled: it works.
Gotcha captured — the pip TF wheel needs `LD_LIBRARY_PATH` at the pip
`nvidia/*/lib` dirs (written to `~/dmir_nas/env.sh`).

Fork deps installed (ray 2.56, dragonfly-opt via --no-build-isolation, sklearn,
scipy, tqdm, matplotlib, tfmot 0.8.1 + tf_keras). Cloned the fork to `~/uNAS`,
registered our adapters (setup_fork.sh). Reproducible setup scripts committed
in `unas/` (setup_wsl_env.sh, setup_fork.sh, run_smoke.sh).

Smoke-run debugging on real DMIR data (dmir_lcr):
1. adapter import fused onto the last line of dataset/__init__.py → fixed
   setup_fork.sh to prepend a newline.
2. `model_trainer.py` imports tfmot at module top even though QAT is off →
   installed tfmot (0.8.1 imports cleanly under Keras 3).
3. Keras-version mix: models are built with bare `keras` (Keras 3) but tfmot
   flips `tf.keras` to legacy Keras 2, so `tf.keras.callbacks.ReduceLROnPlateau`
   (Keras 2, reads optimizer.lr) crashed against the Keras-3 Adam
   (learning_rate). Fixed by building our callbacks from `keras` (Keras 3) in
   dmir_config.py — same Keras as the models. The chain already trained a real
   128 k-param 1D CNN on GPU before this callback fired, so everything upstream
   (ray actor, data, model build, GPU training) is confirmed working.
4. Keras 3 EarlyStopping needs explicit `mode=` for `val_mae` → added mode to
   all callbacks (max for accuracy, min for mae/loss).

**Smoke SUCCESS (dmir_lcr, 4 rounds, 3 epochs each).** The full NAS chain runs
end-to-end on the RTX 5070: aging evolution builds real 1D CNNs from the DMIR
(50,31) windows, trains them on GPU (~3 ms/step after the first-epoch compile),
computes resource features [peak_mem, model_size, MACs] via to_resource_graph,
and completes. Threshold mechanism behaves correctly — with `error_bound`=0.30
MAE and only 3 epochs, candidates land at val_mae ≈ 0.41 (above the bound) so
"pareto" saves 0 models. Encouraging signal: a 72,729-param (284 KB) candidate
reached **val_mae 0.41 / test 0.405 in just 3 epochs** (resource features
peak_mem 9 KB, MACs 0.84 M) — with the full 120-epoch schedule this class of
tiny model should reach the SOTA MAE (0.298) region while staying MCU-sized.
Env + adapters + run scripts are reproducible in `unas/`.

Model saving confirmed separately (loosened bound so candidates pass): 3
Pareto `.h5` models written to `artifacts/dmir_lcr/models/` (Keras warns that
`.h5` is legacy, but the fork's `test.py` expects `.h5` for TFLite conversion,
so we keep it). So the complete pipeline is validated end-to-end on the RTX
5070: search → GPU training → resource_features → Pareto `.h5` save.

Next: full-budget searches per task (regression first — the un-leaky, primary
result), then QAT → INT8 TFLite → ST Edge AI for the H7B3I-DK numbers.

## 2026-07-08 — First real LCR search: OOM fixed, results beat SOTA

First full LCR run (100 rounds, 60 epochs) trained ~40 candidates and found
models with reported test MAE down to 0.143, then **crashed on a ray host-RAM
OOM**: the persistent GPUTrainer actor leaked ~200 MB per candidate (TF not
freeing graphs between fits) and hit the ~15 GB WSL cap. Fix: patch the actor
to `keras.backend.clear_session()` + `gc.collect()` after each candidate is
saved (in setup_fork.sh, idempotent; also TF_FORCE_GPU_ALLOW_GROWTH). Validated
with a 25-round run: **50 candidates, zero OOM, 16 Pareto models saved** —
memory now bounded.

**Independent verification (unas/verify_models.py).** Do not take the fork's
reported numbers on faith — I re-evaluated the saved `.h5` models on the test
set with our own metrics. Reading model_trainer.py confirms the fork's
`test_error` (regression) is `model.evaluate(test)` MAE on the
restore_best_weights model — i.e. honest — so it matches our eval for the same
model. On the short (20-epoch) run the best saved model is **test MAE 0.287,
RMSE 0.443** (predict-mean MAE ~1.01). That already beats the published SOTA
(MAE 0.298, RMSE 0.510) and matches our DSCNN baseline (RMSE 0.439), from a
tiny search; the 0.143 seen mid-run came from a longer-trained (60-epoch)
candidate lost in the OOM. Full searches should land ~0.14–0.20 test MAE.
Policy: always re-verify final best models with verify_models.py before
quoting a number in the paper.

## 2026-07-08 — Full-search queue finished (all 4 crashed on OOM but harvested)

The overnight queue ran all four searches; each stopped before its 150-round
target on a **ray host-RAM OOM** (LCR 68 candidates / graph-error, LCL 93,
cls 62, cls_noind 59). The session-clear patch slowed the leak (40 -> 60-90
candidates) but did not stop it, and ray's OOM-kill is a SIGKILL the
safe-evaluate try/except cannot catch. **Every task saved its Pareto models
incrementally**, so the fronts survive (28/32/25/27 models).

Harvested and INDEPENDENTLY VERIFIED all four fronts (unas/harvest_fronts.py,
our own test-set eval; CSVs in datasets/dmir/results/nas-fronts/). Verification again proved
necessary: the fork's optimistic val_error (a candidate showed val MAE 0.113)
does not hold on test (0.29). Real, verified results:
- **LCR** best test MAE 0.290 / RMSE 0.447 @ 216 KB (int8) — beats published
  SOTA (0.298 / 0.510); still 0.294 @ 109 KB; 0.333 @ 10 KB.
- **LCL** best MAE 0.320 / RMSE 0.484 @ 34 KB — beats SOTA RMSE and our DSCNN.
- **Classification** 91.6% @ 216 KB, 91.5% @ 21 KB (with turn signal).
- **Ablation (no turn signal): 90.8% @ 93 KB, 90.1% @ 5.2 KB.** Removing the
  blinker costs <1% — with the everything-except-blinker measurement (90.8%)
  vs blinker-alone (81.5%), this shows the model genuinely anticipates. Big
  honesty win for the paper.

Recorded in docs/research/nas-results-prelim.md, paper/NOTES.md, and the
results table in paper/main.tex (marked preliminary). Next: robustly fix the
OOM (self-healing actor / chunked resume) and re-run all four to completion;
then QAT -> INT8 TFLite -> ST Edge AI for on-device numbers.

## 2026-07-09 — OOM defeated by chunked resume; all searches completed

Traced the leak: model_saver pops the model before storing (not it); the
accumulation is TF/XLA internal state clear_session() can't release in a
long-lived process, and ray's OOM-kill is a SIGKILL the safe-evaluate can't
catch. Fix that works: **chunked resume** — each search runs in fresh-process
chunks resuming from the aging-evolution checkpoint (the loop counts
len(history), which load_state restores, so --rounds TARGET continues toward
TARGET). Validated on LCR (Loaded 60 → +30 → Search done, no OOM), then
completed all four to 150 rounds (2 chunks for the classification runs).

Final verified fronts (datasets/dmir/docs/nas-results.md; results doc renamed from
-prelim; CSVs in datasets/dmir/results/nas-fronts/): LCR MAE 0.287/RMSE 0.447 @ 115 KB
(0.290 @ 64 KB) — beats SOTA; LCL 0.325/0.501 @ 83 KB; **classification 92.1%
@ 82 KB (91.3% @ 7.8 KB) — matches internal ref 92%**; no-indicator 91.1% @
20 KB (90.2% @ 11 KB). All better than the partial run. Paper table + NOTES +
PDF updated.

Known caveat: pareto-save + chunked resume can prune a good .h5 from an earlier
chunk (LCL slightly behind a transient earlier model). For a guaranteed-optimal
front, re-run with DMIR_SAVE_CRITERIA=all. Next: QAT -> INT8 TFLite -> ST Edge
AI on the STM32H7B3I-DK for real flash/RAM/latency.

## 2026-07-09 — Quantization + deployment footprints (int16x8; no QAT needed)

Quantized the best models to TFLite. **Full int8 PTQ degrades badly** on the
DMIR inputs (wide per-channel dynamic range): LCR MAE 0.287->0.449, cls
92.1%->86.9%, cls-noind 91.1%->76.1%. **int16x8** (int8 weights, int16
activations) preserves or slightly improves accuracy (cls 92.15%, LCR MAE
0.286, cls-noind 91.08%) — so no QAT, avoiding the Keras-3/tfmot incompat.
Deployment .tflite in datasets/dmir/results/tflite/; quantize_eval.py / quantize_compare.py.

Footprints (compute_footprint.py; flash = tflite size, MACs + peak RAM from the
arch): classification 92.15% @ 118 KB flash / 4.5 KB RAM / 152 k MACs (91.2% @
19 KB / 4.4 KB / 30 k); LCR MAE 0.286 @ 161 KB / 11.3 KB / 852 k; cls-noind
91.1% @ 45 KB / 4.2 KB / 36 k. **Every model fits even the STM32F401 (96 KB
RAM), where the baseline Transformer did not fit** — SOTA-beating accuracy at a
fraction of the size. datasets/dmir/docs/deployment.md; paper deployment section +
course lessons 09/10 updated.

Blocked on user: measured on-device latency needs a myST account (ST Edge AI
Developer Cloud) — X-CUBE-AI 10.2 pack is installed but the stedgeai CLI
binaries are not extracted locally. Handoff steps in deployment.md. Everything
else (accuracy, flash, RAM, MACs) is done and verified.

## 2026-07-10 — Reference models verified; RMSE re-run; honest regression stance

Colleague sent the reference models (Materials/Models/, legacy HDF5).
Evaluated (unas/eval_reference.py): cnn_multi 441k params = 91.5% test acc (the
"92%"); transformer_lcr 333k / transformer_lcl 49k (RMSE 0.42/0.44 as reported;
couldn't re-run — custom TransformerEncoder not in the public repo). Head-to-
head (datasets/dmir/docs/reference-comparison.md): **classification is a clean win**
(ours 92.1% @ 84k vs 441k @ 91.5%; 8k matches at 55x smaller); regression NOT a
win vs the internal transformers on RMSE.

Ran RMSE-objective regression searches (patched model_trainer to MSE loss +
val_rmse; DMIR_REG_METRIC=rmse, save_criteria=all, 150 rounds each). Result:
**LCL improved 0.50→0.466** (and MAE 0.325→0.317); **LCR stayed 0.447** (RMSE
run found smaller 62k @ 0.464 but not lower). Still behind the internal
transformers (0.42/0.44). Final paper stance: claim beating the PUBLISHED SOTA
(0.51) at 2-3x fewer params + deployability; do NOT claim regression RMSE win
over the internal reference. Headlines: classification + deployment + the
turn-signal ablation. Paper table/NOTES updated honestly; PDF rebuilt.

## 2026-07-14 23:07 — Course overhaul: figures, quizzes, measured results

User flagged: text overlap/crop in 5 SVG figures; quizzes trivially easy (longest option always correct); missing depth on correlation/ablation; aging-evolution oldest-member ambiguity; no k-fold CV justification; lesson-8 formula unverified; new spike-provenance intel (crash-heavy drivers).

Root cause of ALL 5 figure bugs: dir=rtl flips SVG text-anchor semantics (end extends RIGHT for Farsi) -> labels ran into boxes / off-canvas. Fix: text-anchor=middle for Farsi labels; verified geometrically (0 crops/overlaps).

Facts verified from fork source before writing: fitness = -max_i(min(f_i/bound_i, 10)/lambda_i), lambdas U(0,1) fresh per parent selection; population is a FIFO queue (pop(0), fitness-blind) -> oldest well-defined from round 1 (initial population appended one-by-one). Fixed lesson-07 step-numbering bug (said step 5, is step 6).

Spike provenance: DMIR Test Reports.xlsx corroborates User34 (4 collisions+2 accidents) and User1 (5+1); User43 shows 0/0. Crash -> pose reset -> curvatureDx derivative explosion = plausible, unproven. Recorded in DATA.md + dataset-provenance.md; paper keeps neutral wording.

Lesson 06 expanded (Pearson r explainer, ablation definition, blinker-detection methodology, why-no-k-fold). Lessons 07-11 refreshed with measured results: int16x8 silent-dequantize story, PTQ not QAT as actual path, float32 headline, H7B3 measured table, F401 categorical result, int8 point, 7-11.6 cycles/MAC.

All 49 quiz questions rewritten. First rewrite FAILED its own audit (49/49 still longest-option-correct); second pass -> 33% (chance 25%), gaps within a few chars, indices spread. Agent sweep died on session limit; replaced with deterministic checks (JSON parse, geometric SVG audit, staleness grep).

## 2026-07-24 15:52 — Course trilingual; notebook covers full arc; results narrative unified

Three deliverables in one session (all workflow-verified):
1) Results narrative sync: lesson 07 gets the honest 2x2 comparison (LCR/LCL x
   published-SOTA/internal-ref, both caveats), NAS-algorithm families card,
   Pareto-is-N-dimensional note, cls_tiny provenance (same 33-model front as
   cls_best, not a pruned copy), and search-cost card (~1 GPU-night for all
   four 150-round searches on the RTX 5070 vs literature GPU-days, context
   only). README results table finally replaced the first-DSCNN numbers with
   final verified results + measured on-device section. Paper: search setup
   sentence (150 rounds, pop 50, sample 15 per launch-script defaults) +
   Zoph/Real citations; board-choice rationale recorded (same platform as the
   published baseline = like-for-like; unit available in ELIOS lab).
2) Notebook dmir_pipeline.ipynb now covers the whole research arc: new §8-§11
   (verified Pareto fronts from datasets/dmir/results/nas-fronts, quantization artifact
   sizes read from disk, measured ST Edge AI deployment, scoreboard) via new
   src/results_viz.py; executed headlessly end-to-end, 0 errors.
3) Course is trilingual: full English and Italian editions (course/en/,
   course/it/), 13 pages each, language switcher everywhere, i18n quiz
   engine. Deterministic checks: zero Farsi residue, 49 quiz questions per
   language, longest-option-correct 4%/2%, all links resolve; two editorial
   audit agents passed both languages and unified terminology.

## 2026-07-24 16:36 — QAT recovers the int8 drop: cls_best 86.9% -> 89.8% (same footprint)

User asked to try QAT after the ~5-point int8 PTQ drop (92.08->86.86). Did it, honestly. tfmot 8-bit only registers 2D layers, so searched 1D cls_best re-expressed with width-1 kernels (Conv1D->Conv2D(k,1), Pool1D->Pool2D(p,1), depthwise strides (s,1)->(s,s)=no-op at width 1). Re-expression proven exact: float-2D 0.9208 (=1D orig), PTQ-2D 0.8686 (=measured 1D int8), so PTQ-vs-QAT is single-variable. QAT fine-tune (Adam 2e-4, 22/40 epochs, val early-stop restore-best) -> int8 89.82%, +2.96 pts (57% of gap) at same footprint (101,616 B tflite, float32 I/O). Ran in WSL dmir_nas (TF 2.21/tf_keras/tfmot 0.8.1); needed Keras-3->tf_keras port (rebuild from adjacency + per-layer weight transfer). Code unas/qat_finetune.py; artifact datasets/dmir/results/qat/cls_best_qat_int8.tflite. NOT yet on-device (expected ~int8 PTQ 1.885 ms / 104 KB). deployment.md, paper, course L09 fa/en/it, experiments.jsonl all updated.

## 2026-07-24 16:55 — QAT int8 measured on-device: 89.82% @ 1.558 ms (fastest point)

User uploaded ST Edge AI results for cls_best_qat_int8.tflite (H7B3I-DK, Core 4.0.1, balanced). Measured: 1.558 ms, MACC 161,570, flash 131,008 B (128 KiB; weights 83.28 KiB byte-identical to the PTQ int8 + ~43 KiB library), RAM 8,404 B (6.05 KiB act). Graph confirms genuine int8. Two findings: (1) FASTER than the int8 PTQ point (1.558 vs 1.885 ms) though the 1D->2D change confounds a clean QAT-vs-PTQ latency read; the width-1 Conv2D kernels ST emits look better-optimized than 1D. (2) BIGGER flash (128 vs 104 KiB) - weights identical, the +24 KiB is ST library code for the 2D re-expression (Conv2D+Reshape+extra conversions). Net operating point: 89.82% @ 1.558 ms @ 128 KiB, fastest of the three, 2.6x under float32 flash. deployment.md measured table + QAT section, paper quantization sentence (+PDF), NOTES (QAT measure done; native-1D-QAT optional to reclaim the 24 KiB), course L09 fa/en/it, results_viz MEASURED_H7B3, experiments.jsonl all updated.

## 2026-07-24 16:58 — QAT int8 also on NUCLEO-F401RE: 7.381 ms @ 84 MHz

User measured cls_best_qat_int8 on the F401RE too: 7.381 ms @ 84 MHz. Flash/RAM platform-level, unchanged from H7B3 (128 KiB = 25.6% of 512 KB flash; 8,404 B RAM). So the full classifier, quantized to int8 via QAT, runs on the Cortex-M4 at 89.82%. Cross-board 4.7x slower than H7B3 M7 (1.558 ms) ~ 3.3x clock x 1.4x M4-vs-M7 IPC, consistent with cls_tiny scaling. deployment.md F401 section + fits table + experiments.jsonl updated.

## 2026-07-27 15:14 — cls_best float32 on NUCLEO-F401RE: 18.35 ms @ 84 MHz

User measured the remaining F401 gap: cls_best_float32 = 18.35 ms @ 84 MHz (flash 343,254 B = 65.5% of 512 KB; RAM 9,456 B = 9.6% of 96 KB). Significance: the ACCURACY-HEADLINE model (92.08%, the one that beats the 441k reference CNN) runs on the 0 Cortex-M4 - a board the reference cannot run on at all (3.38x over flash). Cross-board scaling now has three models: M4/M7 wall ratio 5.52x (cls_tiny), 5.06x (cls_best fp32), 4.74x (QAT int8) against a 3.33x clock ratio -> IPC factor 1.4-1.7x, shrinking as the kernels get more efficient. New finding from the cycles/MAC table: int8 runs at 2.70 cyc/MAC vs float32 6.43 on the M7 (~2.4x more efficient per MAC) - that is the real source of the QAT model speed win, since it executes MORE MACs (161,570 vs 158,094) in LESS time. deployment.md F401 section + fits table + cross-board tables + experiments.jsonl updated.

## 2026-07-27 15:21 — lcr_best fp32 on F401: NO measured result (flash headroom limit?)

User ran lcr_best_float32 on the NUCLEO-F401RE: the Developer Cloud returned a DASH, no inference time, no error reason. Honest handling: recorded as a negative result, not converted into a does-not-fit claim. Context: lcr_best is 474,522 B = 90.5% of the 512 KB flash, leaving only 49,766 B for the validation application and runtime, whereas cls_tiny/QAT/cls_best (which all returned times) had 486/393/181 KB headroom. So flash HEADROOM, not model size, looks like the real F401 limit - but that is a hypothesis: a dash can also mean the run was not executed or timed out. IMPORTANT correction: our F401 fits table claimed lcr_best and lcl_best fit based on model-size arithmetic alone; that yes is now downgraded to "no result returned" / "not established", and the README + paper claims were narrowed from "every searched model" to "the three classifiers" (which were actually measured). lcl_best (80.8%, 100 KB headroom) is the discriminating next test. deployment.md fits table + new headroom table, README, paper, results_viz F401 chart (gray = not established) all updated.

## 2026-07-27 15:23 — F401 headroom ceiling bracketed: lcl_best runs (162.5 ms), lcr_best does not

User ran the discriminating test. lcl_best_float32 on NUCLEO-F401RE = 162.5 ms @ 84 MHz (80.8% of flash, 100,794 B headroom) - it RUNS. lcr_best (90.5%, 49,766 B headroom) returns a dash, no time. So the practical F401 ceiling is flash HEADROOM and sits between 80.8% and 90.5% occupancy: the validation app and runtime need flash on top of the weights. Model-size arithmetic (474 KB < 512 KB therefore fits) is NOT a deployability test - a point now made in the paper, since size arithmetic alone would have called all five models deployable. Caveat kept: the Cloud reported no reason for the dash, so insufficient-headroom is the supported explanation rather than a reported error. Second finding: cycles/MAC now falls monotonically with model size on the float32 path - cls_tiny 7.00, cls_best 6.43, lcl_best 4.86 (M7) - bigger models amortize the fixed per-op overhead, which is direct support for the overhead-bound reading (previously asserted from two points). Cross-board IPC factor 1.42-1.69x across four models. deployment.md (fits + headroom + both cross-board tables), README, paper (+PDF), NOTES (item closed), results_viz + F401 chart, experiments.jsonl updated.

## 2026-07-27 15:33 — Headroom cause isolated: lcr_best int8 runs on F401 (28.10 ms) where its fp32 build could not

The decisive control. lcr_best_int8 on NUCLEO-F401RE: 28.10 ms @ 84 MHz, MACC 858,209, flash 150,504 B (147 KiB; weights 115.46 KiB + ~29 KiB lib) = 28.7% of flash, RAM 14,352 B (13.18 KiB act). The fp32 build of the SAME network (474,522 B, 90.5% flash, 49,766 B headroom) returns no measured time. Same architecture, same operators, same board, same toolchain - only the numeric format differs - so the fp32 failure is isolated to the FLASH FOOTPRINT, not the model structure or an unsupported op. This upgrades the headroom account from plausible explanation to demonstrated cause (ST still reports no reason for the dash, so the precise internal failure mode stays unreported). Paper now makes the sharper claim: on this board quantization is not merely a size/speed optimization for lcr_best, it is the difference between a model that cannot be benchmarked and one that runs in 28.10 ms. HONESTY CAVEAT recorded everywhere: that int8 build is PTQ and costs the regressor a lot (MAE 0.287 -> 0.4485), so the deployable-on-F401 claim for lcr_best currently comes at a large accuracy price; QAT for the regression heads is the natural fix and is now a NOTES TODO. Efficiency note: 2.75 cyc/MAC on the M4, the best of any build measured, continuing the pattern that bigger models and int8 both amortize per-op overhead better. deployment.md (fits table + new controlled-experiment table), README, paper (+PDF), NOTES, results_viz + F401 chart, experiments.jsonl.

## 2026-07-27 15:56 — QAT rescues the classifier but NOT the regressors (negative result)

Generalized unas/qat_finetune.py to any task (+ new unas/export_graph.py for the Keras-3 -> tf_keras graph export) and ran QAT on both regression heads. Anchors exact in both cases (LCR float 0.2865 / PTQ 0.4485; LCL float 0.3165 / PTQ 0.3440 - all reproduce the 1D numbers to 4dp), so these are clean single-variable comparisons. RESULT: LCR MAE 0.4485 -> 0.4313 (recovers only ~11% of the gap to float32, vs the classifier 57%); LCL 0.3440 -> 0.3620, i.e. QAT is WORSE than plain PTQ. Re-ran both at lr 2e-5 to rule out a hyper-parameter artifact: worse again (LCR +0.0115, LCL -0.0321), so the negative result is robust across two learning rates. INTERPRETATION (consistent with the data): classification only needs the argmax of three logits and tolerates coarse activations, while regression needs the continuous value, so the same activation-resolution loss lands on the output - int8 PTQ costs the classifier 5.7% relative accuracy but raises LCR error 57%. Fine-tuning adapts weights and cannot restore information lost in the activation representation. CONSEQUENCE: operating-point table is asymmetric - classification has three usable points (float32 / int8 QAT / int8 PTQ), regression has float32 as the only accurate one with int8 as a deployability fallback (which is exactly the role it plays on the F401, where int8 is what makes lcr_best run at all). deployment.md new section, paper quantization paragraph (+PDF), NOTES item closed, course lesson 09 fa/en/it got a why-it-differs callout, experiments.jsonl.

## 2026-07-27 16:24 — int8 I/O artifact ready: cls_best_int8_io.tflite (predicts RAM 8,096 -> ~3,446 B)

Built unas/export_int8_io.py and produced datasets/dmir/results/deploy/cls_best_int8_io.tflite (112,568 B, int8 in / int8 out, input scale 1.10578 zero_point 38). Verified accuracy-neutral: 0.868624 through a quantizing interpreter, identical to the float32-I/O build to six decimals and matching the measured 0.8686. The float32-I/O control rebuilt BYTE-FOR-BYTE identical to the committed cls_best_int8.tflite (112,912 B), confirming the converter is deterministic and the script reproduces prepare_deploy.py exactly. Found and documented a latent trap: eval_tflite in prepare_deploy.py/quantize_eval.py only CASTS (.astype), a no-op for float32 interfaces (so all existing numbers are unaffected) but it would silently destroy an int8 input - export_int8_io.py carries its own quantize-in/dequantize-out evaluator. Insight worth keeping: the float32 interface never protected accuracy, because the float32-I/O build already quantizes the input internally with the same scale (the visible conversion_0 op); the int8 interface merely moves that cast off the device. It was costing RAM and buying nothing. PENDING: one ST upload to confirm RAM 8,096 -> ~3,446 B (-4,650 B float32 input buffer) and that conversion_0 disappears; flash/latency should be unchanged. deployment.md + NOTES updated.

## 2026-07-27 16:35 — int8 I/O measured: RAM 8,096 -> 5,444 B; badge puzzle solved; new board (H573I-DK)

Uploaded cls_best_int8_io.tflite. RESULT: RAM 8,096 -> 5,444 B (1.49x; activations 6.05 -> 3.46 KiB), flash 106,738 -> 106,446 B (weights identical), MACC 158,336 -> 155,230, and BOTH cast ops are gone - the per-layer charts now run pool_1..gemm_26 where the float32-I/O build ran conversion_0..conversion_14. So int8 is finally the clear RAM-efficient point: 1.74x under float32, up from a marginal 1.17x. SELF-CORRECTION: my predicted 3,446 B was 2 KB optimistic. I subtracted the 4,650 B buffer saving from the measured total, but peak RAM is a MAX over simultaneously-live tensors plus kernel scratch - the very principle already written in our own RAM-regimes section. Shrinking the input from 6,200 to 1,550 B did not subtract 4,650 B; it removed the input as the BINDING CONSTRAINT, after which the widest internal activation set the new floor at 3,543 B. Generalizable lesson now in the docs and paper: once you are no longer input-bound, further input shrinking buys nothing. BADGE PUZZLE RESOLVED: this build reports STAI_FORMAT_S8 where every previous one reported STAI_FORMAT_FLOAT, so the badge tracks the I/O tensor dtype, NOT weight precision - which vindicates the earlier retraction of that badge as int16x8 evidence and explains why a genuinely-int8 build had shown FLOAT. CAVEAT RECORDED: this was benchmarked on a THIRD board, STM32H573I-DK (Cortex-M33 @ 250 MHz, 2.462 ms, 3.97 cyc/MAC), so its latency is NOT comparable to the 1.885 ms Cortex-M7 number; flash/RAM come from the platform-level optimize step and do compare. An H7B3 run of this artifact remains optional if we want to claim the interface change is latency-neutral. deployment.md (measured table + new H573 section), paper RAM paragraph (+PDF), NOTES, results_viz, experiments.jsonl.

## 2026-07-27 16:44 — CORRECTION + completion: int8 I/O measured on H7B3 = 1.752 ms (7.1% faster), not H573

The earlier H573I-DK screenshot was a mix-up (user flagged it); the real cls_best_int8_io benchmark is on the STM32H7B3I-DK, Cortex-M7 @ 280 MHz: 1.752 ms. Removed the third-board section and the non-comparability caveat I had written on the bad data, and added the int8-IO row to the main H7B3 measured table. This makes it a CLEAN SINGLE-VARIABLE comparison - same board, settings and weights, only the tensor interface differs: RAM 8,096 -> 5,444 B (1.49x) AND latency 1.885 -> 1.752 ms (7.1% faster), accuracy identical at 86.86%. So the float32 interface was costing both memory and TIME: casting 1,550 elements per inference is real work. Earlier I could only claim the memory effect; now the paper states both, and notes that default converter settings produce exactly this misleading configuration. Revised int8 operating points on H7B3: PTQ+float32-IO 86.86% @ 1.885 ms / 8,096 B; PTQ+int8-IO 86.86% @ 1.752 ms / 5,444 B; QAT+float32-IO 89.82% @ 1.558 ms / 8,404 B. NEW TODO: QAT + int8 I/O has never been built and should dominate all three (accuracy of QAT, memory and cast saving of int8 I/O). The RAM self-correction (predicted 3,446 B vs measured 5,444 B; max-over-live-tensors not subtraction) and the STAI_FORMAT_S8 badge finding both stand unchanged. deployment.md, paper (+PDF), NOTES, results_viz, experiments.jsonl.

## 2026-07-28 11:58 — Built QAT + int8 I/O: 89.90%, fully integer, ready to upload

Combined the two independent wins. unas/qat_finetune.py now (a) is seeded (SEED=42), (b) saves the trained fake-quant model so future interface variants need no retraining, and (c) emits BOTH the float32-I/O and int8-I/O int8 builds from the SAME trained model, so the pair differs only by interface. Result: datasets/dmir/results/qat/cls_best_qat_int8_io.tflite, 101,064 B, int8 in/out (scale 1.149315, zp 32), tensor inventory 22 int8 + 9 int32 and ZERO float32 - fully integer end to end, where the float32-I/O builds still carried two float32 tensors. Accuracy 89.90%, and the same-run float32-I/O build also scores 89.90%, so the interface is accuracy-neutral for QAT exactly as it was for PTQ. Anchors exact again (float 0.9208, PTQ 0.8686). REPRODUCIBILITY: the seeded re-run scores 89.90% where the original unseeded run scored 89.82% - a 0.08 pt fine-tuning stochastic spread, now eliminated by seeding. Deliberately restored the committed cls_best_qat_int8.tflite to its ORIGINAL bytes (the re-run had overwritten it) because the measured 1.558 ms / 8,404 B row belongs to those exact bytes; provenance preserved. Pending one upload: expect ~89.90% accuracy, RAM well under 8,404 B, latency under 1.558 ms, flash ~128 KiB. If it lands it is the best classifier operating point measured. deployment.md, NOTES, experiments.jsonl.

## 2026-07-28 12:32 — QAT + int8 I/O measured: 89.90% @ 1.435 ms - best configuration in the project

Combining the two independent wins landed exactly as predicted, and it is now the best classifier operating point we have. STM32H7B3I-DK: 1.435 ms, MACC 158,710, flash 131,562 B, RAM 6,204 B (3.39 KiB act), badge STAI_FORMAT_S8, accuracy 89.90%. Against the float32-I/O QAT build: 7.9% faster (1.558 -> 1.435 ms), 1.35x less RAM (8,404 -> 6,204 B), flash unchanged. At 2.53 cycles/MAC it is the most efficient kernel path measured (previous best 2.70). Against the float32 headline model: 2.53x faster, 2.61x less flash, 1.52x less RAM, for 2.18 accuracy points. Note the I/O casts are gone but one INTERNAL requantize (conversion_8, a Quantize node between the pooling and the FC head) remains - it is not an I/O cast and the float32-I/O build had it too, alongside conversion_0 and conversion_14 at the boundaries. The deployment story now has THREE Pareto-optimal points, none dominating: float32 92.08% (accuracy, and the like-for-like comparison with the FP32 baseline), QAT+int8-IO 89.90% @ 1.435 ms (speed), PTQ+int8-IO 86.86% @ 104 KiB / 5,444 B (size). The QAT rows carry ~22 KiB more library than the PTQ rows purely because of the width-1 2D re-expression - a tooling cost, not a quantization cost, removable by a native-1D QAT. deployment.md (measured table + final operating-points table), paper (+PDF), NOTES, results_viz, course lesson 09 fa/en/it (a default-setting-cost-you-memory card), experiments.jsonl.

## 2026-07-28 13:04 — Paper audit: fixed an inverted claim, 2 contradictions and ~20 numeric/scope errors

Ran a 6-agent audit of paper/main.tex against the source-of-truth docs. It caught things I had missed. BLOCKERS: (1) the regression claim was INVERTED - the paper said we beat the published SOTA at "2-3x fewer parameters", but our regressors are 117k/106k against the published Transformer ~54k, i.e. ~2x LARGER; rewritten as an accuracy+deployability claim at comparable parameter count, with the task-mismatch caveat (their single combined-direction TTLC on 30 channels vs our per-direction models on 31). (2) the contributions claimed we beat the published baseline "on all three tasks" - the published baseline has only ONE task, and the three-task comparison is against the internal reference, which we LOSE on both regression RMSEs. (3)+(4) abstract placeholders filled with real measured numbers, and its promise of int8-headline corrected to float32-headline to match the tables. CONTRADICTIONS: LCL RMSE printed as 0.50 in the footnote and 0.47 in the body two paragraphs apart (0.466 is right); reference CNN accuracy 91.5 in one table and 91.7 in the other (91.7 clipped is protocol-consistent). NUMERIC FIXES: F401 QAT flash 25.6%->25.0% (also wrong in deployment.md), board SRAM 1.4MB->1184KB, reference FC share 95%->93%, head reduction 6x->5.9x, LCL MAE 0.33->0.32 (double-rounded AND from the superseded pre-RMSE-rerun model), DSCNN 10k->10.5k, flash 1729->1728 KiB (tflite size vs ST-measured flash), 47x->46.6x, LCL RAM 26.5->27.6 KB (activations quoted where the others were totals), LCR RAM 20.8->20.3, param gap 9x->10.5x, headroom 100->98 KB. SCOPE FIXES: the caption claimed both reference models were re-evaluated (only the CNN was; the Transformers could not be re-run); "our search optimizes MAE" is no longer blanket-true since the LCL model comes from the RMSE-objective re-run; the per-layer-time claim was float32-only and documented for one classifier (int8 inverts it); the LCL liveness arithmetic did not reach its own stated peak (missing the 12.0 KB live branch output); "allocator fragmentation" asserted a mechanism the source declines to assign; "2D-only" mis-attributed a tfmot QAT restriction to ST. TODOs: 6 resolved and deleted (abstract results, memory ratio, STM32 part, fileTime/channel-31, RAM+MACC+latency columns, RMSE framing), search-space ranges filled from the fork schema (which also proved unas-method.md wrong: kernels are {3,5,7} not {3,5,7,9}), Setup section written, ST Edge AI documentation citation added and the previously-uncited forneris2024dmir now cited. STYLE: the one "not only X but also Y" removed; em-dashes cut 16->8 prose instances to meet the <=2/page budget; blocklist clean. 8 TODOs remain, all genuinely open (co-authors/supervisor/grant - now tracked in NOTES - plus the unwritten Introduction, Related Work and Conclusion). PDF rebuilds clean at 4 pages with no undefined references.

## 2026-09-01 16:55 — highD: protocol reimplemented, baseline beats published Table III

Access granted + downloaded (879 MB, gitignored, non-redistributable).
Reimplemented the EarlyLCPred scenario protocol (Mozaffari et al., T-IV 2022):
35-frame scenarios @5 Hz ending at lane crossing, 18 state_ours features,
balanced LK undersampling, split by recording 1-50/51-55/56-60. Validation
against the paper: train 7,487 EXACT, val 932 EXACT, test 693 vs 698 (0.7%).

Baseline DSCNN (8,371 params) under THEIR exact metric formulas
(eval_their_protocol.py transcribes utils.py):
acc 0.911 / F1 0.930 / AUC 0.959 / tau_c 4.79 s / TTLC RMSE 0.276 s
vs their proposed BEV-attention model 0.83 / 0.85 / 0.88 / 3.96 / 0.629.
Every Table III entry exceeded. Caveats in datasets/highd/docs/baseline-results.md
(reimplementation, not their harness; cuDNN variance ~0.3 pt; 5-s boundary
ambiguity). TTLC label uses their (26-s)/5 convention.

Launched overnight aging-evolution searches highd_cls + highd_ttlc (150 rounds
each, pop 50 sample 15, bounds 32 KB peak-mem / 32 KB size / 500k MACs) via
unas/run_chunked_highd.sh; smoke 3/3 candidates passed first.

## 2026-09-02 10:47 — highD searches complete: cls 13 models, ttlc salvaged to 25

Both 150-round searches finished overnight. cls front 7.9k-38k params, best
90.53% @ 7,904 (baseline 91.1% @ 8.4k still leads — 0.07 bound tilts fitness
to size). ttlc first run saved 0 models (all val MAE > 0.16 bound; saver drops
out-of-bound models) — salvage resume with bound 0.20 + save-all captured 25;
best test MAE 0.1624/RMSE 0.2614 @ 80k, 0.1656/0.2608 @ 27.7k vs baseline
0.169/0.276 @ 8.4k. All dominate published RMSE 0.629. Gotchas recorded:
saver bound-gating, fork test_error==val_error, chunk-log overwrite (history
recovered from state pickle). One Ray OOM absorbed by chunked runner.

## 2026-09-02 11:04 — highD deploy artifacts built; tight-bound cls search running

Six TFLite artifacts in datasets/highd/results/deploy (f32 / int8 / int8-IO
for cls model_aaaaam 7.9k and ttlc model_aaaaaw 27.7k), all evaluated on real
test windows. PTQ drop: cls -1.4 pt (much milder than DMIR 5.2 — min-max
inputs), ttlc +13% relative MAE (task asymmetry again). QAT deferred.
Second cls search highd_cls_tight (error bound 0.045) launched; runner now
timestamps chunk logs so histories survive re-invocations. CRLF gotcha: python
Path.write_text on Windows converts .sh files to CRLF — rewrite with
newline=chr(10).

## 2026-09-02 13:18 — ST board farm driven via official API; manual numbers verified

User granted credentials + bypass permissions; benchmarks now run through
ST's official Developer Cloud Python client (Materials/stm32ai-modelzoo-services
common/stm32ai_dc). 8-job batch, 0 failures: ttlc winner measured in all three
variants on both boards (f32 1.038/5.058 ms; int8-IO 0.658/2.940 ms; int8 RAM
> f32 RAM again — scratch-dominated at small scale), plus re-verification of
two manually pasted cls numbers: deltas +0.35% and +0.09% — manual process
confirmed accurate, farm repeatability <0.4%. Credentials never persisted to
disk or repo; user to rotate password. Also: HIGHD_PARALLEL=2 live on the
tight search (GPU 21->25%, throughput to be compared at completion — tiny-model
NAS is not GPU-bound).

## 2026-09-02 15:15 — highD campaign complete: tight-search winner deployed and measured

model_aaaaap (5,347 params, 91.15% test) measured: f32 0.1547/0.6960 ms,
int8-IO 0.1063/0.4670 ms (H7B3/F401) at 91.04%, 15.3 KB ROM, 3 KB RAM, 5,635
MACC — 6x fewer MACs than the first search winner at higher accuracy; fastest
artifact in the project (13.5x faster than DMIR best). int8 cost 0.12 pt.
highD story now complete end to end: protocol validated vs the paper, searched
front beats published Table III and the hand baseline, three-variant deployment
measured on two boards with API-verified repeatability <0.4%.

## 2026-09-03 12:28 — AI-tell audit across the whole project; course lesson 12; public-repo cleanup

AUDIT. 23-agent sweep of every prose surface (docs, docstrings, notebook
markdown, 39 course pages, manuscript) against paper/STYLE.md. Verifiers
rejected 55 of 85 candidates as legitimate technical usage; 30 confirmed and
fixed. Substantive ones, not just style: the paper claimed "automotive-grade"
hardware when neither board is an automotive part; a paragraph restated
Section III-C verbatim; unas/quantize_eval.py promised a uint8-I/O artifact it
never writes; highD docs claimed "nobody in this literature reports" the
deployment axis, narrowed to their Table III. Register fix in deployment.md
(a first-person-singular post-mortem inside a we-document, closing on an
aphorism addressed to "you"). Em-dashes 96 -> 82 and 22 -> 14; a mechanical
first pass created three comma splices, repaired by hand. STYLE.md gained
section E with the 13 tells this project actually exhibits.

COURSE. Lesson 12 written in all three languages: the second dataset, why
DMIR and highD are different tasks (ego driver vs surrounding vehicles), the
protocol-reproduction discipline (7,487/932 exact, 693 vs 698 declared), the
results, and the three findings only a second dataset could produce
(data-dependent quantization damage, the conditional int8-I/O RAM rule, the
search being overhead-bound not GPU-bound). Indexes and lesson-11 navigation
updated in fa/en/it. Quiz distractors length-balanced so the correct option is
never the longest. 42 pages validated: 0 tag errors, 0 broken links, 39 quizzes
parse.

REPO. MIT LICENSE added with a README section separating code terms from
dataset terms; CITATION.cff; CLAUDE.md and the dead root index.html moved out
of version control; run_baseline.py and prepare_deploy_highd.py documented.

## 2026-09-08 16:56 — T4.5 SYNERGIES update deck; searched highD classifier scored with their metrics

Built the T4.5 meeting deck on the SYNERGIES template (Materials/T4.5, local):
opening, 4 content slides (chain + two datasets, DMIR measured results, highD
protocol + results, compression findings / October questions / next), closing;
speaker notes on every slide; charts in the template palette (validated pair:
accent blue vs neutral baseline). For the highD table the tight-search winner
was scored with the transcribed T-IV metrics: acc 0.912, precision 0.966, recall
0.896, F1 0.930, AUC 0.962, tau_c 4.53 s (baseline 4.79 s). Deck left local; the
metric JSON and doc note are committed.

## 2026-09-08 17:17 — Reference Transformers measured on the boards: the deployment cost is now data

Partners are pushing Transformers; Luca flagged the latency objection. The
project had never deployed one, so unas/transformer_deploy.py rebuilds both
internal reference Transformers from the configuration stored inside their .keras
files (three custom layer classes exist in no public repo, so load_model fails)
and loads the original weights.

Scope, stated in every record: architecture is exact (params match to the unit,
333,505 / 49,089) but the rebuilt RMSE does not reproduce the reported
0.42/0.44, so accuracy is never claimed. Latency/flash/RAM depend on the
operator graph, not weight values, so those are reported.

Measured (ST Edge AI 4.0.1, balanced, fp32):
  Transformer LCR 333k: 368.82 ms H7B3; does not fit F401 (1,368,946 B flash vs
    524,288; 143,300 B RAM vs 98,304). 95.2 M MACs.
  ours lcr_best 117k:  14.06 ms  -> 26.2x faster, 111x fewer MACs
  Transformer LCL  49k:  45.64 ms H7B3, 248.37 ms F401. 33.9 M MACs.
  ours lcl_best 106k:  28.77 ms  -> 1.59x faster with 2.2x MORE parameters

Headline finding: the LCL Transformer has 0.46x our parameters and 20x our MACs.
Parameter count badly understates Transformer cost because attention recomputes
across the sequence. Both convert with TFLite builtin ops only, so the honest
claim is cost, not infeasibility. At 100 km/h the LCR Transformer spends 10.2 m
of road per inference against 0.39 m for ours.

Briefing in qub/docs/transformer-on-mcu.md; backup slide added to the T4.5 deck
(v2, since the original was open).

## 2026-09-09 12:28 — highD: scenario replay for the T4.5 extra slide

Request: one extra slide that shows the same real lane change under the
different methods, so the comparison is visible rather than tabular.

datasets/highd/scenario_replay.py picks the scenario and runs the models:
candidates are test-split lane changes with at least 2 s of track after the
crossing and a neighbour in the target lane; the chosen one is the candidate
whose searched-model robust horizon is closest to the model's test-set average
(4.53 s), i.e. a typical case, not a flattering one. Chosen: recording 56,
vehicle 221, left lane change at 119 km/h, crossing frame 2760. Searched 5.3k
classifier: first correct call at 5.0 s, robust from 4.6 s. Hand CNN 8.4k:
robust from 4.8 s. Of the 60 candidates, 22 are called from the first window
(5.2 s). Predictions come from the trained models over the scenario's 26
evaluation windows; nothing is synthesised.

datasets/highd/render_replay.py draws it from the raw 25 Hz positions and the
recording's lane markings: animated GIF (113 frames, 12.5 fps, for slideshow
mode), a still at the 4.6 s flag, and a self-contained interactive HTML page
(slider, play, toggles). Traces and the flag are revealed as the replay
advances. A band under the road shows road covered during ONE inference at the
scenario speed: 4 mm for the int8 searched model (0.1063 ms on H7B3I-DK),
12.2 m for the DMIR reference Transformer (368.82 ms, same board). Two items
are averages and are labelled as such on the picture: the T-IV model's robust
horizon (3.96 s, test-set average, model not public) and the Transformer
latency (no highD Transformer exists).

Outputs stay local (Materials/T4.5/replay, data/replay under the gitignored
highD data folder); the slide is a separate one-slide file on the SYNERGIES
template because the user has edited the main deck by hand.

## 2026-09-09 13:06 — DMIR: five-model race replay for the T4.5 extra slide

Request: one more visual, DMIR this time, with one car per model of the
deck's table on the same scenario, all starting together, and a crash for the
slow model to show what latency costs.

The pickles are shuffled windows, so datasets/dmir/scenario_race.py rebuilds a
continuous scene from the raw H5 sessions of the two official test drivers.
The provider's preprocessing was recovered rather than guessed: windows located
in the test pickle by correlation, one affine map per channel fitted and then
checked to reproduce those windows (6,063 windows, max deviation 1.4e-7; the
map agrees between drivers to 2e-10). Findings recorded in feature-map.md:
indicator channels unscaled, car1 = nearest object among slots 1-10, car2 =
object slot 11 (the scenario's left car), absent car = zeros.

164 lane-change events cut (t = -8 .. +2 s); the four table models run on all
16,564 windows in WSL (REF_cnn_multi, cls_best, cls_best QAT int8 I/O,
cls_tiny). First robust flag before the crossing, medians over the 101 usable
events: reference CNN -3.7 s, searched 84k -3.5 s, QAT int8 -2.7 s, 8k -3.3 s.
The QAT model flags noticeably later than its float parent; worth a look.
Chosen scene = closest to the searched model's median: driver 2, left lane
change at 90 km/h, braking behind a slower car while a faster car passes on
the left; flags -3.3/-3.5/-3.3/-3.3 s.

datasets/dmir/render_race.py draws five rows (four table models + a
reference-size Transformer using the measured 368.82 ms of the DMIR regression
reference, assumed to flag with the reference CNN), answer bars in road metres
(94 cm, 10 cm, 4 cm, 2 cm, 10.3 m), roof light when the answer is ready, and an
optional what-if hazard one car length ahead of the question point. The crash
is an illustration of the measured latency and is labelled so on the picture,
in the caption and in the notes; the plain version is delivered alongside.
Outputs local: Materials/T4.5/race/{race.gif, race_whatif.gif, race_still*.png,
race.html} and Materials/T4.5/T4.5_extra-slide_DMIR-race.pptx (two slides).

## 2026-09-09 22:47 — DMIR race: the what-if rebuilt around the recorded car in the target lane

Feedback on the first what-if: the hazard was an invented box in the ego's own
lane, but DMIR is about warning the driver, and the situation the dataset
actually contains is a lane change into a car the driver may not have seen.

Rebuilt on that. A four-lens review (claims, visuals, counterfactual maths,
skeptical audience) of the first version found the old crash was an artefact:
the counterfactual driver was anchored on the frame where the recorded lateral
displacement first exceeds 0.4 m (-1.2 s), and any anchor at -1.5 s or earlier
produced no crash at all. The margin was also measured centre-to-centre while
the crash test used bumper overlap.

The recorded scene now drives the whole slide: the driver closes on a slower
car (the brake is never pressed; he lifts off, 33 -> 25 m/s), signals at
-3.5 s, and a car in the target lane is 22 m behind closing at 15 m/s at the
searched model's flag. He waited for it and steered at -1.2 s, after it passed.
The recording contains no conflict, and the slide says so.

The counterfactual is now one driver, the same in all five rows, with one
parameter taken from the recording: he commits at the indicator and crosses at
the recording's own top lateral rate (1.26 m/s), reaching the recorded car at
-2.2 s. Each row stops him if its warning arrives more than TAU before that.
Margins: searched 84k 1.30 s, QAT int8 1.10 s, 8k 1.10 s, reference CNN 1.07 s,
reference-size Transformer 0.73 s. At TAU 0.75 s and 1.00 s the Transformer row
is the only one too late; at 1.25 s only the searched float model still stops;
at 1.5 s none do. That sensitivity is printed on the picture and the HTML has a
slider for it.

Other corrections from the review, all applied: the QAT row's two board figures
are different builds (int8-I/O on the H7B3, float-I/O on the F401) and are now
marked; the indicator time was 3.5 s not 3.6; the hazard figures belong to the
-3.5 s flag; the scenario is one of two events at the median and was picked
between them by the presence of the target-lane car; the header speed (90 km/h
at the crossing) differs from the speed the answer bars use (101 km/h at the
flag). Predictions were re-run and match the previous file (argmax 100%).

## 2026-09-16 15:56 — Seed study: the search matches, hand-built CNNs match or win

Audit of the DMIR and highD results before the Q1 paper, triggered by the question
of whether each headline number is a property of the architecture or of one run.

Seed study, about 200 retrainings, seeds 0-4 throughout:
- DMIR final models (`unas/seed_variance.py`, search recipe and `RECIPE=dscnn`) and
  the hand-built DSCNN (`scripts/run_baseline.py <task> <seed>`, new seed argument).
  Intention: searched 84 k 91.45 ± 0.59%, 8 k 91.19 ± 0.32%, reference CNN 91.33 ±
  0.50%, DSCNN 10.5 k 91.50 ± 0.47%; no pair separates (p ≥ 0.27). Regression RMSE:
  DSCNN 0.454 / 0.469 s against searched 0.483 / 0.496 s (search recipe) and
  0.493 / 0.483 s (DSCNN recipe); MAE ties. Turn-signal ablation holds (89.99%).
- highD: final models under both recipes; hand-built CNN 92.11 ± 0.71%; all 17 saved
  classifiers re-ranked (`unas/rerank_front.py`): best mean 91.28% at 80 k, rank
  correlation between reported run and mean −0.19, the best single run ranks 16/17.

Causes found: the µNAS 1D head (pool → Flatten → hidden Dense, no dropout) cannot
express the hand-built global-pooling head; single-run candidate scoring keeps lucky
runs; the fork's ModelSaver restarts its file counter per chunk and per Ray worker,
so saved .h5 files overwrite each other and no longer match `metadatas.json`
(numbers measured on the files stand, "Pareto front" does not).

Deliverables: `datasets/*/results/deploy/measurements.json` registries (every board
number, API runs referenced into `benchmarks_api.jsonl`); `dashboard/` results
explorer (one self-contained page per build, `--share` copy without status page and
highD trajectory media); seed tables in both dataset docs; README headline table
with seed means and the DSCNN rows; paper draft switched to seed means and the
corrected search-space description (PDF rebuilt); NOTES audit with options for the
supervisor. Side note recorded in highD docs: the highD replay's Transformer latency
is the DMIR-shaped model; on 10 × 18 it needs 5× fewer FLOPs, so it must be measured.

## 2026-09-16 17:48 — Target venue: IEEE Transactions on Intelligent Vehicles

Decided by Sepehr. The paper grows from the 4-page SPL-style DMIR draft into a full journal paper; highD goes in, exiD if access and time allow. T-IV is also where the highD protocol we reproduce was published (Mozaffari et al., 2022). Still open: which story leads (search-led after fixing and re-running the search, or deployment-led with searched and hand-built models compared on the boards). exiD access: the levelXdata non-commercial application is being prepared.

## 2026-09-16 18:23 — Manuscript rewritten for IEEE T-IV

The paper draft (local, paper/main.tex) now targets IEEE Transactions on Intelligent Vehicles: 9 pages covering DMIR and highD, with exiD as planned work. Framing follows the seed study: measured cost on the STM32H7B3I-DK and NUCLEO-F401RE, a seed-controlled comparison of searched, hand-designed and reference models, the reimplemented highD protocol, and the toolchain findings. Data figures are generated from the result files (paper/figures/make_figures.py) and a check script recomputes the table values from the same files; it caught one wrong standard deviation (searched LCL under the DSCNN recipe is 0.483 +/- 0.019 s, not 0.020), now corrected in nas-results.md, the notes and course lesson 13. scripts/build_paper.ps1 no longer stops pdflatex early: piping the compiler into Select-Object -First ended the process once 20 warning lines had matched.

## 2026-09-17 15:18 — Dataset name LCIR; manuscript moved to the official IEEE template

The Lane Change Intention Recognition data is now called LCIR in the paper, the README and the results explorer; DMIR stays the internal codename and folder name. The manuscript (local) now follows the official IEEE Transactions LaTeX template downloaded through the IEEE template selector: its class options and package list, hline tables, and the TikZ diagrams compiled as standalone PDFs and included as graphics. It is 7 pages. The IEEE ITSS page for T-IV gives a suggested length of 10 pages for regular papers with a charge per extra page; the draft stays at 8 pages or fewer so exiD results and biographies fit. Author, funding and acknowledgment items are placeholders for now. The results explorer tab formerly called Robustness is now Seeds, and the READMEs were edited to remove em-dash asides.

## 2026-09-17 15:30 — Repository renamed to Lane-Change-MCU

The GitHub repository is now github.com/SepehrMohammady/Lane-Change-MCU (formerly LC-Intention-NAS); GitHub redirects the old address. The old name no longer described the project: highD and exiD predict lane changes of surrounding vehicles rather than driver intention, and the work now compares searched with hand-designed models and measures them on two microcontrollers. Updated the git remote, README title and opening paragraph, CITATION.cff (title, URL, abstract), and the local course pages and notes. The paper keeps the repository citation as a placeholder until the author details are settled.

## 2026-09-23 13:09 — exiD prepared in the highD format; transfer study; paper and explorer updated

exiD v2.1 (levelXdata, non-commercial, no redistribution) arrived and is prepared in the
highD scenario format by `datasets/exid/prepare_exid.py`: 93 recordings at 7 motorway
junctions, 20,820 scenarios (train/val/test 16,446 / 2,617 / 1,757), 13,896 lane changes,
of which 4,657 merges from on-ramps and 2,128 exits. Adaptations, all in
`datasets/exid/docs/DATA.md`: crossings from the laneChange flag; side from the sign of
latLaneCenterOffset (agrees with the Lanelet2 adjacency on all 1,603 checked crossings);
exiD's latVelocity is in the vehicle frame, so lateral velocity and acceleration are
2-s Savitzky-Golay derivatives of the lane-relative offset (window chosen to match highD's
lane-change kinematics, not on accuracy); distances projected on the heading; lane
existence from the Lanelet2 maps; hard-shoulder crossings dropped; the latest recordings
of every location form the test split.

The highD training loop moved to `src/lc_windows.py` and is shared by both datasets (the
highD wrapper gives bit-identical results on CPU). It now uses deterministic cuDNN
kernels, so a repeated seed repeats its result; older highD runs predate this.

Results, hand-designed CNN (8,371 parameters, 8,241 for TTLC), five seeds, exiD test set:
trained on exiD 89.93 +/- 0.24% (TTLC RMSE 0.406 +/- 0.008 s); the highD model as is 61.24 +/- 2.32% (0.867 +/- 0.062 s);
the same weights with exiD input scaling 38.93 +/- 4.66%, so scaling is not the cause;
highD weights fine-tuned on exiD 90.13 +/- 0.22% (0.393 +/- 0.002 s). A highD start helps
only with little data: +1.3 points with 10% of the training scenarios (p = 0.02), +0.6
with 25% (p = 0.12), +0.2 with all (p = 0.22). The highD model gets exits right in 33.2%
of the windows and lane keeping in 45.4% (exiD-trained: 94.2% and 84.7%). The exiD model
scores 84.64 +/- 0.67% on highD. The highD-searched architectures retrained on exiD reach
76.7-78.6% under both recipes, at least 11 points below the hand-designed CNN. Board cost
of exiD models equals the highD builds (same graphs).

Paper (local): Section III-C and V-D, Table IV and Fig. 6, title covering the three
datasets, still 8 pages; `check_numbers.py` covers the exiD values. Results explorer:
exiD with Overview, Transfer and Seeds views; the Compare page's log-axis labels no longer
print float noise (0.30000000000000004 ms); project name Lane-Change-MCU. READMEs: exiD
section, layout and licence lines.

Open: board runs of the hand-designed CNNs (needs an ST login session); architecture
search on exiD only after the saver fix and a pooling-head option; QUB needs its own exiD
access.

## 2026-09-23 14:04 — Correction: size of the hand-designed TTLC CNN

The hand-designed CNN has 8,371 parameters as a classifier and 8,241 with its single TTLC
output (config.n_params in the highD and exiD run logs). Several places called the TTLC
model 8.4 k: the paper's seed table and highD metrics table, the exiD table note, the
section text, the README tables, the highD baseline notes, and the results explorer, which
had 8,371 hard-coded for both tasks. All now give 8.2 k (8,241) for TTLC; the explorer reads
both counts from the run logs. No result changes. The paper's transfer paragraph now also
says that the five highD source models are new runs with deterministic kernels (92.61% on
highD, against 92.11% for the Table III runs; Welch p = 0.29).

## 2026-09-23 16:56 — Hand-designed CNNs measured on both boards; v2 search set up; QUB code checked

Board runs. The hand-designed CNNs (PyTorch) were rebuilt in Keras with the same weights
(`unas/deploy_hand_cnn.py`, outputs within 5.2e-5 of PyTorch) and converted exactly like
the searched models; 26 runs through the ST API (`unas/st_benchmark.py`, Core 4.0.1,
balanced), none failed. The LCIR DSCNN weights had never been kept, so one run per task
was retrained with `scripts/run_baseline.py <task> --save` (90.96%, RMSE 0.439 / 0.459 s).

- LCIR intention, 10.5 k: 3.272 ms on the M7, 18.70 ms on the M4, 51,382 B flash,
  171,971 MACC. The searched 8 k model has the same five-seed accuracy and runs in
  0.793 ms, 4.1 times faster: the one place where the search wins on cost.
- LCIR time to lane change, 10.3 k: 3.265 ms against 14.06 / 28.77 ms for the searched
  regressors, with lower RMSE.
- highD classifier, 8.4 k: 0.669 ms FP32 and 0.350 ms int8 on the M7; more accurate and
  12% faster than the searched 7.9 k model, 4.3 times slower than the 5.3 k one, which
  is about 3 points less accurate. highD TTLC, 8.2 k: 0.667 ms against 1.038 ms at the
  same RMSE.
- PTQ: LCIR intention 90.96% to 87.14%, LCR RMSE 0.439 to 0.652 s; highD 0.2 points and
  0.022 s.

Search. The fork's ModelSaver restarts its file names in every process and applied its
error bound and Pareto test to the test error. `unas/safe_saver.py` saves every
candidate within the resource bounds under a unique name with a JSON sidecar and never
uses test error; `unas/cnn1d_gap.py` adds a global-average-pooling head, zero to three
hidden Dense layers and a searched dropout rate, so the hand-designed DSCNN's layer
sequence is inside the space; `unas/select_by_seeds.py` retrains the 12 best single-run
candidates with five seeds and chooses on the validation mean. Smoke-tested over two
chunks; the highD v2 searches (classifier, then TTLC; same budgets and recipe as the
first searches) are running.

QUB. TrajPred holds their later trajectory predictors, not the T-IV 2022 model; no
trained weights are public in any of the author's repositories. The T-IV model is
ATTCNN3 in EarlyLCPred (3.22 M parameters, 38.1 M MACs, bird's-eye-view image input
10 x 80 x 200; exports to standard ONNX). EarlyLCPred has no licence. To measure it
like-for-like we need their trained checkpoint, the exact scenario lists and their
permission.

Paper (local): Table V has the hand-designed rows, and the cost section, discussion and
conclusion give the comparison; it is 9 pages until the v2 results are in and the text
is cut. Explorer: hand-designed builds are drawn as rings.
