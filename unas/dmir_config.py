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
# Parallel candidate evaluations (Ray workers); the first searches ran one at a time.
PARALLEL = int(os.environ.get("DMIR_PARALLEL", "1"))


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


def _setup(task, name, error_bound, drop_indicators=False, v4=False, model_size_bound=None, mac_bound=None):
    classification = task == "classification"
    dataset = DMIR_Dataset(task=task, drop_indicators=drop_indicators)
    space = Cnn1DSearchSpace
    if v4:
        from configs.cnn1d_gap import FaithfulGapCnn1DSearchSpace
        from configs.safe_saver import install
        install()                       # fork builds SafeModelSaver for this run
        space = FaithfulGapCnn1DSearchSpace
    config = {
        "training_config": _training_config(dataset, classification),
        "bound_config": BoundConfig(
            error_bound=error_bound,
            peak_mem_bound=PEAK_MEM_BOUND,
            model_size_bound=model_size_bound or MODEL_SIZE_BOUND,
            mac_bound=mac_bound or MAC_BOUND,
        ),
        "search_algorithm": AgingEvoSearch,
        "search_config": AgingEvoConfig(
            search_space=space(),
            checkpoint_dir=f"artifacts/{name}",
            rounds=ROUNDS, population_size=POPULATION, sample_size=SAMPLE,
            max_parallel_evaluations=PARALLEL,
        ),
        # v4: keep every candidate within the resource bounds; the final model is chosen
        # afterwards on validation data (unas/select_by_seeds.py)
        "model_saver_config": ModelSaverConfig(save_criteria="boundaries" if v4 else SAVE_CRITERIA),
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


# v4 (2026-09-25): the searches of unas/highd_config.py *_v4 on LCIR. The global-average-
# pooling space with the resource graph of the trained Keras model
# (configs/cnn1d_gap.py FaithfulGapCnn1DSearchSpace; the fork's own graph pads no
# convolution and under-counted LCIR candidates a median 1.25-1.52 times), the unique-name
# saver, and budgets at the hand-designed DSCNN's cost under that graph
# (datasets/highd/results/resource_bias.json, "hand_like_in_v2_space": 10,387 B and 169,872
# MACs for intention, 10,257 B and 169,744 MACs for TTLC; board MACC 171,971 and 171,841).
# Error bounds: the DSCNN's five-seed validation level rounded up (accuracy 94.64% -> 0.06
# error; RMSE 0.439 s LCR -> 0.44, 0.479 s LCL -> 0.48). TTLC searches optimize RMSE, so they
# need DMIR_REG_METRIC=rmse (MSE loss, RMSE metric; patch_fork.py). DMIR_V4_SUFFIX renames the
# artifact folder (smoke tests). The final model: unas/select_by_seeds.py (five-seed validation).
V4_SUFFIX = os.environ.get("DMIR_V4_SUFFIX", "")


def _v4_regression(task, name, bound):
    if REG_METRIC != "rmse":
        raise RuntimeError("the v4 TTLC searches optimize RMSE: set DMIR_REG_METRIC=rmse")
    return _setup(task, name + V4_SUFFIX, bound, v4=True, model_size_bound=10_257, mac_bound=169_744)


def get_dmir_cls_v4_setup(**_):
    return _setup("classification", "dmir_cls_v4" + V4_SUFFIX, 0.06, v4=True,
                  model_size_bound=10_387, mac_bound=169_872)


def get_dmir_lcr_v4_setup(**_):
    return _v4_regression("regression_lcr", "dmir_lcr_v4", 0.44)


def get_dmir_lcl_v4_setup(**_):
    return _v4_regression("regression_lcl", "dmir_lcl_v4", 0.48)
