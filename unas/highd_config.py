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
# Parallel candidate evaluations (Ray workers; each claims GPU:0.2 + 6 CPUs).
# MCU-scale candidates leave the GPU mostly idle per worker, so 2 roughly
# doubles search throughput; keep <=2 on this 15 GB WSL (OOM history).
PARALLEL = int(os.environ.get("HIGHD_PARALLEL", "1"))


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


def _setup(task, name, error_bound, v2=False, model_size_bound=None, mac_bound=None):
    classification = task == "highd_cls"
    dataset = HighD_Dataset(task=task)
    if v2:
        from configs.cnn1d_gap import GapCnn1DSearchSpace
        from configs.safe_saver import install
        install()                       # fork builds SafeModelSaver for this run
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
            search_space=GapCnn1DSearchSpace() if v2 else Cnn1DSearchSpace(),
            checkpoint_dir=f"artifacts/{name}",
            rounds=ROUNDS, population_size=POPULATION, sample_size=SAMPLE,
            max_parallel_evaluations=PARALLEL,
        ),
        # v2: keep every candidate within the resource bounds; selection happens
        # afterwards on validation data (unas/select_by_seeds.py)
        "model_saver_config": ModelSaverConfig(save_criteria="boundaries" if v2 else SAVE_CRITERIA),
        "serialized_dataset": False,
    }
    return {"config": config, "name": name, "load_from": None,
            "save_every": 10, "seed": 42}


def get_highd_cls_setup(**_):
    return _setup("highd_cls", "highd_cls", CLS_ERROR_BOUND)


# Tighter accuracy demand: the 0.07-bound run satisfied accuracy early and
# spent all fitness pressure on size (front peaked at 90.5%). A 0.045 bound
# (val acc >= 95.5%) keeps accuracy binding for longer.
def get_highd_cls_tight_setup(**_):
    return _setup("highd_cls", "highd_cls_tight",
                  float(os.environ.get("HIGHD_CLS_TIGHT_BOUND", "0.045")))


def get_highd_ttlc_setup(**_):
    return _setup("highd_ttlc", "highd_ttlc", REG_ERROR_BOUND)


# v2 (2026-09-23): search space with a global-average-pooling head, zero to three
# hidden Dense layers and a searched dropout rate (configs/cnn1d_gap.py), and the saver
# of configs/safe_saver.py. Same recipe, budgets and error bounds as the first searches,
# so the head is the only change to what the search can find. HIGHD_V2_SUFFIX renames the
# artifact folder (used for smoke tests).
V2_SUFFIX = os.environ.get("HIGHD_V2_SUFFIX", "")


def get_highd_cls_v2_setup(**_):
    return _setup("highd_cls", "highd_cls_v2" + V2_SUFFIX,
                  float(os.environ.get("HIGHD_CLS_TIGHT_BOUND", "0.045")), v2=True)


def get_highd_ttlc_v2_setup(**_):
    return _setup("highd_ttlc", "highd_ttlc_v2" + V2_SUFFIX, REG_ERROR_BOUND, v2=True)


# v3 (2026-09-23): the v2 space and saver with budgets at the hand-designed CNN's cost. The
# v2 classifier search ran with loose budgets and an error bound (0.045) that no candidate
# reaches, so the fitness gave cost little weight (each objective is divided by a random
# weight, so cost can still decide a draw): its chosen model matches the hand-designed CNN's
# accuracy but runs 4.3 times slower. Here model size and MACs are
# bounded at what the fork's resource model gives for the hand-designed CNN's layer sequence
# (unas/build_hand_like.py: 8,051 B, 13,648 MACs), rounded up to 8 KiB and 14,000 MACs, and
# the error bound is the hand-designed CNN's validation level (0.06). Question: can the search
# match the hand-designed CNN's accuracy at equal or lower cost?
def get_highd_cls_v3_setup(**_):
    return _setup("highd_cls", "highd_cls_v3" + V2_SUFFIX, float(os.environ.get("HIGHD_V3_ERROR_BOUND", "0.06")),
                  v2=True, model_size_bound=8 * 1024, mac_bound=14_000)
