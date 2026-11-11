"""Verified channel names and roles for the prepared windows.

Two layouts (see datasets/dmir/docs/feature-map.md): the LCR pickle keeps the turn
indicators inline (channels 3-4); the classification and the LCL pickles have them
at the end (channels 28-29). Names for channels 0-7 (classification) were verified by
correlating each channel against the raw H5 signals; the rest follow the data
provider's feature list plus the fingerprint signatures.

Corrected 2026-11-11: until then the LCL pickle was listed with the LCR layout. Its
test windows are found in the raw sessions with the classification layout only, and
the affine map of every channel then fits to 1e-12 (datasets/dmir/session_stream.py).

Use these to build feature masks (e.g. drop the leaky indicators for an
ablation, or drop the constant egoLaneWidth) without magic indices in the
pipeline.
"""
from __future__ import annotations

# LCR layout: provider order minus fileTime, + car2Present.
REGRESSION_FEATURES = [
    "latAcceleration", "longAcceleration", "brakePedalPos",
    "leftDirectionIndicator", "rightDirectionIndicator", "steeringAngle",
    "throttlePedalPos", "vehicleSpeed", "yawRate", "egoLaneWidth",
    "yawAngleRightLine", "yawAngleLeftLine", "curvatureRightLine",
    "curvatureLeftLine", "curvatureDxRightLine", "curvatureDxLeftLine",
    "rightLaneLine", "leftLaneLine",
    "car1LatPosition", "car1LatVelocity", "car1LongPosition",
    "car1LongVelocity", "car1YawAngle", "car1YawRate",
    "car2LatPosition", "car2LatVelocity", "car2LongPosition",
    "car2LongVelocity", "car2YawAngle", "car2YawRate", "car2Present",
]

# Classification and LCL layout: indicators relocated to the end (28, 29).
CLASSIFICATION_FEATURES = [
    "latAcceleration", "longAcceleration", "brakePedalPos", "steeringAngle",
    "throttlePedalPos", "vehicleSpeed", "yawRate", "egoLaneWidth",
    "yawAngleRightLine", "yawAngleLeftLine", "curvatureRightLine",
    "curvatureLeftLine", "curvatureDxRightLine", "curvatureDxLeftLine",
    "rightLaneLine", "leftLaneLine",
    "car1LatPosition", "car1LatVelocity", "car1LongPosition",
    "car1LongVelocity", "car1YawAngle", "car1YawRate",
    "car2LatPosition", "car2LatVelocity", "car2LongPosition",
    "car2LongVelocity", "car2YawAngle", "car2YawRate",
    "leftDirectionIndicator", "rightDirectionIndicator", "car2Present",
]

FEATURES_BY_TASK = {
    "classification": CLASSIFICATION_FEATURES,
    "regression_lcl": CLASSIFICATION_FEATURES,
    "regression_lcr": REGRESSION_FEATURES,
}

# Channels that carry the driver's turn-signal state. Reaching 81.5% test
# accuracy from these alone, they nearly announce the classification label;
# use this for the with/without-indicators ablation.
INDICATOR_FEATURES = ("leftDirectionIndicator", "rightDirectionIndicator")

# Constant on the 2-lane highway (egoLaneWidth = 3.75 m -> 0 after scaling).
CONSTANT_FEATURES = ("egoLaneWidth",)

# Numerical-derivative channels with test-only spikes (handled by clipping).
SPIKE_FEATURES = ("curvatureDxRightLine", "curvatureDxLeftLine")


def feature_names(task: str) -> list[str]:
    return list(FEATURES_BY_TASK[task])


def indices_of(task: str, names) -> list[int]:
    feats = FEATURES_BY_TASK[task]
    return [feats.index(n) for n in names if n in feats]
