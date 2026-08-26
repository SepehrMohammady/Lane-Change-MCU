# The published baseline (the paper we must beat)

Researched 2026-07-07 (web + IEEE stamp PDF + official repo).

## Citation

L. Forneris, R. Berta, M. Fresta, L. Lazzaroni, H. Rojhan, C. Oh, A. Pighetti,
H. Ballout, F. Tango, F. Bellotti, "A Deployment-Oriented Simulation Framework
for Deep Learning-Based Lane Change Prediction," *IEEE Signal Processing
Letters*, vol. 33, pp. 136–140, 2026. DOI 10.1109/LSP.2025.3638676
(IEEE doc 11271346). CC BY-NC-ND. Funded by EU H2020 Hi-Drive (101006664).
Code: https://github.com/Elios-Lab/LaneChangeIntentionRecognition

## ⚠ Critical correction to our project premise

The published paper does **not** report 3-class classification accuracy and
does **not** report separate LCL/LCR RMSE. It frames the task as a **single
time-to-lane-change (TTLC) regression**: labels 0–4.0 s (0.1 s step), 4.1 =
free ride, frames within 1.5 s after a lane change discarded. Input windows
(50, **30**) — our prepared pickles have **31** channels.

The numbers our colleague quoted (92% accuracy 3-class; 0.42 s RMSE LCR;
0.44 s RMSE LCL) are **internal, unpublished results** on the prepared
pickles — a different task formulation than the paper. Both targets matter:

| Reference point | Task | Metric |
|---|---|---|
| Published (SPL paper, Transformer) | single TTLC regression, user-independent | **RMSE 0.5102 s** (MAE 0.2978, R² 0.8641) |
| Internal (colleague) | 3-class intention | accuracy 92% |
| Internal (colleague) | TTLC regression LCR / LCL | RMSE 0.42 s / 0.44 s |

## Published models (Keras, 50-trial Bayesian optimization, seed 42)

| Model | Params | Test RMSE (s) |
|---|---|---|
| Transformer (2 enc. blocks, 1 head, head 128, FF 128, GAP, dense 160) | ~54k | **0.5102** |
| LSTM (192 units) | 171,457 | 0.5492 |
| GRU (192/32/32) | 157,089 | 0.5681 |
| 1DCNN (conv 64+32, k=3, dense 32) | ~24.4k | 0.5746 |
| XGBoost (100 trees, depth 6) | — | highest MAE |

Training: Adam lr 1e-3, MSE, early stop patience 15, ≤200 epochs, batch 256
(64 for Transformer), StandardScaler. Prior HighD-based TTLC literature:
RMSE 0.629–0.7.

## Published deployment (our comparison axis)

- Toolchain: STM32CubeMX + STM32Cube.AI, **FP32 only — no quantization**.
- High-end: **STM32H7B3** (280 MHz, 1.4 MB RAM, 2 MB flash). Exact per-model
  flash/RAM/latency/power are in Table III (rasterized image; needs
  institutional access — TO-DO). Qualitative: 1DCNN most compact and ~10×
  faster/lower-energy than the other DL models; Transformer small flash but
  RAM-heavy; XGBoost best latency/RAM/energy but big model file.
- Low-end: **STM32F401** (84 MHz, 96 KiB RAM, 512 KiB flash): XGBoost 1.15 ms
  / 50 mW / 0.06 mJ; 1DCNN 40.8 ms / 90 mW / 3.7 mJ; Transformer does not fit.

Our angle: they deploy FP32 hand-picked architectures; we search architectures
under int8/RAM/flash/latency constraints — better accuracy AND better
deployment numbers, ideally on the same two boards.

## Reproducibility anchors

- User-independent split — validation users {5,8,10,12,16,19,27}, test users
  {2,7,13,18,25,31,36}, train = remaining 36 (of 50).
- Their window counts (binary LC-vs-FR balanced): 165,182 / 36,602 / 36,100 —
  different preparation than our pickles (94,620 / 16,332 / 19,722 3-class).

## Open questions

- [ ] Table III exact values (get PDF via institutional access).
- [x] Driver-wise split confirmed — `scripts/analysis/verify_split.py`
      reproduces the official val/test user sets exactly (see DATA.md and
      dataset-provenance.md), so the comparison to 0.5102 is apples-to-apples.
- [ ] Which 1DCNN config is in Table III (paper says 64+32; repo ships 32+32).
