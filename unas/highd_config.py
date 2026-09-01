"""highD search configs for the ELIOS uNAS fork.

Register in driver.py _CONFIGS:
    "highd_cls":  ("configs.highd_config", "get_highd_cls_setup"),
    "highd_ttlc": ("configs.highd_config", "get_highd_ttlc_setup"),

Bounds target a *small* MCU point from the start: the hand-built DSCNN
baseline (8.4k params, ~0.9 MACs-per-window x 10 frames) already reaches
val acc 94.3% / val MAE 0.146 s, so the search must find models that are
smaller and/or better within tight budgets.
"""
import os

import keras

from uNAS.config import (TrainingConfig, BoundConfig, AgingEvoConfig,
                         ModelSaverConfig)
from uNAS.cnn1d import Cnn1DSearchSpace
from uNAS.search_algorithms import AgingEvoSearch
from dataset.highd_dataset import HighD_Dataset

PEAK_MEM_BOUND = int(os.environ.get("HIGHD_PEAK_MEM_BOUND", 32 * 1024))
MODEL_SIZE_BOUND = int(os.environ.get("HIGHD_MODEL_SIZE_BOUND", 32 * 1024))
MAC_BOUND = int(os.environ.get("HIGHD_MAC_BOUND", 500_000))
# val_error: classification 1 - val_acc; regression val MAE (seconds).
CLS_ERROR_BOUND = float(os.environ.get("HIGHD_CLS_ERROR_BOUND", "0.07"))
REG_ERROR_BOUND = float(os.environ.get("HIGHD_REG_ERROR_BOUND", "0.16"))

ROUNDS = int(os.environ.get("HIGHD_ROUNDS", "150"))
POPULATION = int(os.environ.get("HIGHD_POPULATION", "50"))
SAMPLE = int(os.environ.get("HIGHD_SAMPLE", "15"))
EPOCHS = int(os.environ.get("HIGHD_EPOCHS", "50"))
SAVE_CRITERIA = os.environ.get("HIGHD_SAVE_CRITERIA", "pareto")


def _training_config(dataset, classification):
    if classification:
        cbs = lambda: [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                          patience=8, restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
        ]
    else:
        cbs = lambda: [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min",
                                              factor=0.5, patience=6, min_lr=1e-6),
            keras.callbacks.EarlyStopping(monitor="val_mae", mode="min",
                                          patience=10, min_delta=0.002,
                                          restore_best_weights=True),
            keras.callbacks.TerminateOnNaN(),
        ]
    return TrainingConfig(dataset=dataset, optimizer="adam", callbacks=cbs,
                          epochs=EPOCHS, batch_size=256)


def _setup(task, name, error_bound):
    classification = task == "highd_cls"
    dataset = HighD_Dataset(task=task)
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


def get_highd_cls_setup(**_):
    return _setup("highd_cls", "highd_cls", CLS_ERROR_BOUND)


def get_highd_ttlc_setup(**_):
    return _setup("highd_ttlc", "highd_ttlc", REG_ERROR_BOUND)
