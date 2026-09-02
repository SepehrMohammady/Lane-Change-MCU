# highD — reproducing these results

Everything here follows the problem definition of **Mozaffari et al., "Early
Lane Change Prediction for Automated Driving Systems Using Multi-Task
Attention-Based Convolutional Neural Networks", IEEE T-IV 7(3), 2022**, using
their public reference implementation
([github.com/SajjadMzf/EarlyLCPred](https://github.com/SajjadMzf/EarlyLCPred))
as the specification.

**The dataset itself is not redistributable.** Obtain highD from
[levelxdata.com/highd-dataset](https://levelxdata.com/highd-dataset/) (free for
academic use, request form) and extract it so that
`datasets/highd/data/data/XX_tracks.csv` exists. Everything else needed to
check our numbers is in this repository.

## 1. Build the scenarios (~80 s)

```bash
python datasets/highd/prepare_highd.py
```

Writes `data/prepared/{train,val,test}.npz`. **Verification checkpoint:** the
printed scenario counts must be **7,487 / 932 / 693**. The first two match the
paper exactly; our test split is 5 scenarios (0.7%) short of the 698 reported,
a discrepancy we have documented but not resolved — see `docs/DATA.md`.

## 2. Reproduce the headline comparison (~10 s, CPU is fine)

```bash
python datasets/highd/eval_their_protocol.py
```

Loads the committed baseline checkpoints in `results/` and scores them with the
metric definitions transcribed from their `utils.py` (per-slide averaging,
their double-counted false positives, their 101-point AUC sweep, their
τ_f / τ_c prediction times, their TTLC ground truth `(26-i)/5`). Reproduces
`results/their_protocol_eval.json`:

| metric | ours (8.4k params) | their Table III proposed |
|---|--:|--:|
| accuracy | 0.911 | 0.83 |
| F1 | 0.930 | 0.85 |
| AUC | 0.959 | 0.88 |
| τ_c | 4.79 s | 3.96 s |
| TTLC RMSE | 0.276 s | 0.629 s |

**This is our reimplementation of their protocol, not their harness.** The
metric code is a transcription and the extraction matches on two of three
splits; treat it as such.

## 3. Retrain the baselines from scratch (~30 s each, GPU)

```bash
python datasets/highd/train_highd.py cls
python datasets/highd/train_highd.py ttlc
```

Expect ~±0.3 accuracy points of run-to-run variation (cuDNN nondeterminism).

## 4. Architecture search (optional, ~1.5 h on one RTX 5070)

Needs the [ELIOS µNAS fork](https://github.com/elios-lab) with the adapters in
`unas/` (`highd_dataset.py`, `highd_config.py`) installed by the runner:

```bash
bash unas/run_chunked_highd.sh highd_cls_tight 150 50   # HIGHD_PARALLEL=2 for 2 workers
python unas/harvest_highd.py ~/uNAS/artifacts <out> highd_cls_tight
```

Search outputs are already committed, so this step is only needed to re-derive
them: fronts in `results/nas-fronts/*.csv`, the three winning Keras models in
`results/models/`, and every candidate's metrics in
`results/nas-fronts/highd_ttlc_history.csv`.

## 5. Deployment artifacts and on-device measurements

`results/deploy/` holds each winner as float32 / int8 / int8-interface TFLite,
with the accuracy of every variant in `*_deploy.json`. On-device numbers come
from the ST Edge AI Developer Cloud board farm (Core 4.0.1, balanced) and are
recorded per run in `results/deploy/benchmarks_api.jsonl`.

Final classifier, **5,347 parameters, 91.15% (float) / 91.04% (int8)**:

| variant | STM32H7B3I-DK | NUCLEO-F401RE | flash | RAM |
|---|--:|--:|--:|--:|
| float32 | 0.1547 ms | 0.6960 ms | 25,096 B | 1,368 B |
| int8 (int8 I/O) | 0.1063 ms | 0.4670 ms | 15,329 B | 3,088 B |

Repeatability: two artifacts were measured both by hand through the web UI and
again through the API; the two paths agreed to +0.35% and +0.09%.

Full commentary, caveats and negative results: `docs/baseline-results.md`.
