# DMIR × µNAS integration

We run the constrained architecture search with the **ELIOS lab µNAS fork**
(https://github.com/Elios-Lab/uNAS) rather than reimplementing it: the colleague's
directive ("take µNAS code"), and the fork already provides exactly what we need —
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
regression `min(val_mae)` (loss is MAE — optimise/threshold on **MAE**, not RMSE).

## Files here

| File | Drop into the fork at | Purpose |
|---|---|---|
| `dmir_dataset.py` | `dataset/dmir_dataset.py` | serves the real DMIR pickles (all 3 tasks), train-range clipping, optional indicator drop |
| `dmir_config.py` | `configs/dmir_config.py` | search configs + STM32H7B3I-DK thresholds |

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

## Environment (the open decision — see docs/research/unas-integration.md)

The fork is **TensorFlow**, not PyTorch. On Windows it pins `tensorflow<2.11`
+ `numpy==1.23.5` + Python ≤3.10, and that old TF cannot use the RTX 5070
(Blackwell needs CUDA 12.8) — so Windows = CPU-only search. GPU needs Linux/WSL2
with `tensorflow[and-cuda]` (2.18), and Blackwell support there is unverified.
Options and recommendation in the integration note.
