"""Verified channel names and roles for the prepared windows.

Two layouts (see docs/research/feature-map.md): the regression tasks keep the
turn indicators inline (channels 3-4); classification moves them to the end
(channels 28-29). Names for channels 0-7 (classification) were verified by
correlating each channel against the raw H5 signals; the rest follow the data
provider's feature list plus the fingerprint signatures.

Use these to build feature masks (e.g. drop the leaky indicators for an
ablation, or drop the constant egoLaneWidth) without magic indices in the
pipeline.
"""
from __future__ import annotations

# Regression layout (LCL, LCR): provider order minus fileTime, + car2Present.
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

# Classification layout: indicators relocated to the end (28, 29).
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
    "regression_lcl": REGRESSION_FEATURES,
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
