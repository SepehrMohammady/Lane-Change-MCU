"""DMIR search configs for the ELIOS uNAS fork (all three tasks).

Drop into the fork's `configs/` directory and register in `driver.py`'s
`_CONFIGS`:

    "dmir_lcr": ("configs.dmir_config", "get_dmir_lcr_setup"),
    "dmir_lcl": ("configs.dmir_config", "get_dmir_lcl_setup"),
    "dmir_cls": ("configs.dmir_config", "get_dmir_cls_setup"),

Then, e.g.:  python driver.py -c dmir_lcr

The four BoundConfig values ARE the "thresholds": aging evolution's fitness is
-max_i( normalise(feature_i, bound_i) / lambda_i ), so a model exceeding any
bound is penalised first. Defaults below target the STM32H7B3I-DK
(Cortex-M7, 1.4 MB SRAM, 2 MB flash) with generous headroom for a first
Pareto front; tighten peak_mem/model_size afterwards to push toward the
STM32F401 low-end (96 KB / 512 KB) stretch target.
"""
import os

import tensorflow as tf
# Build callbacks from the SAME Keras the fork's models use. The 1D search
# space builds models with bare `keras` (Keras 3), but importing tfmot flips
# tf.keras to legacy Keras 2; mixing the two makes Keras-2 callbacks read a
# Keras-3 optimizer (e.g. ReduceLROnPlateau -> optimizer.lr) and crash. Using
# keras.callbacks (Keras 3) keeps callbacks and model on the same Keras.
import keras

from uNAS.config import (TrainingConfig, BoundConfig, AgingEvoConfig,
                         ModelSaverConfig)
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch
from dataset.dmir_dataset import DMIR_Dataset

# STM32H7B3I-DK first-pass thresholds (bytes / MACs). INT8 weights.
PEAK_MEM_BOUND = 256 * 1024      # SRAM activations budget (headroom under 1.4 MB)
MODEL_SIZE_BOUND = 256 * 1024    # INT8 weight storage (headroom under 2 MB flash)
MAC_BOUND = 2_000_000            # MACs; baseline DSCNN ~0.17 M, so generous
# error_bound is task-specific. The fork defines val_error as:
#   classification: 1 - max(val_accuracy)
#   regression:     min(val_mae)   <-- MAE, not RMSE (loss = MeanAbsoluteError)
# Our DSCNN baseline test MAE: LCR 0.318, LCL 0.333; published SOTA MAE 0.2978.
CLS_ERROR_BOUND = float(os.environ.get("DMIR_CLS_ERROR_BOUND", "0.10"))  # >=90% acc
REG_ERROR_BOUND = float(os.environ.get("DMIR_REG_ERROR_BOUND", "0.30"))  # MAE target
SAVE_CRITERIA = os.environ.get("DMIR_SAVE_CRITERIA", "pareto")

# Search budget. rounds=2000 is the paper default; start smaller for a first
# end-to-end validation, then scale up. Override for a quick smoke test with
#   export DMIR_ROUNDS=20
ROUNDS = int(os.environ.get("DMIR_ROUNDS", "300"))
POPULATION = int(os.environ.get("DMIR_POPULATION", "100"))
SAMPLE = int(os.environ.get("DMIR_SAMPLE", "25"))
EPOCHS = int(os.environ.get("DMIR_EPOCHS", "120"))  # set small (e.g. 3) for smoke
# Regression objective: "mae" (default) or "rmse". Requires the model_trainer
# patch in patch_fork.py; the callbacks below monitor the matching val metric.
REG_METRIC = os.environ.get("DMIR_REG_METRIC", "mae")
# Model save policy: "pareto" (default) or "all" (keep every model — avoids
# chunked-resume pruning a good candidate).
SAVE_CRITERIA = os.environ.get("DMIR_SAVE_CRITERIA", "pareto")


def _training_config(dataset, classification):
    # Keras 3 requires explicit mode= for monitors it can't classify by name.
    if classification:
        cbs = lambda: [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                          patience=12, restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
        ]
    else:
        mon = "val_rmse" if REG_METRIC == "rmse" else "val_mae"
        cbs = lambda: [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                              factor=0.5, patience=15, min_lr=1e-6),
            keras.callbacks.EarlyStopping(monitor=mon, mode="min", patience=20,
                                          min_delta=0.005, restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
        ]
    return TrainingConfig(dataset=dataset, optimizer="adam", callbacks=cbs,
                          epochs=EPOCHS, batch_size=256)


def _setup(task, name, error_bound, drop_indicators=False):
    classification = task == "classification"
    dataset = DMIR_Dataset(task=task, drop_indicators=drop_indicators)
    config = {
        "training_config": _training_config(dataset, classification),
        "bound_config": BoundConfig(
            error_bound=error_bound,
            peak_mem_bound=PEAK_MEM_BOUND,
            model_size_bound=MODEL_SIZE_BOUND,
            mac_bound=MAC_BOUND,
        ),
        "search_algorithm": AgingEvoSearch,
        "search_config": AgingEvoConfig(
            search_space=Cnn1DSearchSpace(),
            checkpoint_dir=f"artifacts/{name}",
            rounds=ROUNDS, population_size=POPULATION, sample_size=SAMPLE,
        ),
        "model_saver_config": ModelSaverConfig(save_criteria=SAVE_CRITERIA),
        "serialized_dataset": False,
    }
    return {"config": config, "name": name, "load_from": None,
            "save_every": 10, "seed": 42}


def get_dmir_lcr_setup(**_):
    return _setup("regression_lcr", "dmir_lcr", REG_ERROR_BOUND)


def get_dmir_lcl_setup(**_):
    return _setup("regression_lcl", "dmir_lcl", REG_ERROR_BOUND)


# RMSE-objective variants (set DMIR_REG_METRIC=rmse). Distinct names -> fresh
# artifacts dirs, so they do not collide with the MAE runs. The error_bound is
# an RMSE target here (published SOTA 0.51; internal reference 0.42/0.44).
def get_dmir_lcr_rmse_setup(**_):
    return _setup("regression_lcr", "dmir_lcr_rmse", 0.44)


def get_dmir_lcl_rmse_setup(**_):
    return _setup("regression_lcl", "dmir_lcl_rmse", 0.44)


def get_dmir_cls_setup(**_):
    return _setup("classification", "dmir_cls", CLS_ERROR_BOUND)


# Ablation variant (no turn-signal channels) — see the label-leak finding.
def get_dmir_cls_noind_setup(**_):
    return _setup("classification", "dmir_cls_noind", CLS_ERROR_BOUND,
                  drop_indicators=True)
