# Feature map — the 31 channels per task

Confirmed 2026-07-07 from the colleague's `feature_description` doc (authoritative
names) cross-checked against the data: ego channels named by correlation to the
raw H5 signals (`scripts/analysis/name_channels.py`, r ≈ 0.98, each channel to a
distinct signal), the rest fixed by the fingerprint signatures
(`scripts/analysis/fingerprint_channels.py`).

Colleague's confirmations:
- split is by user (driver-wise) — matches our `verify_split.py`;
- the raw `fileTime` timestamp is **not** in the arrays;
- the 31st channel is **"is the car on the left present?"** (binary), i.e.
  `car2Present`;
- `egoLaneWidth` is constant (3.75 m) and kept "in case it changes in future".

The raw `DirectionIndicator` is a single ternary signal {0 = off, 1 = left,
2 = right}; the prepared data splits it into two binary channels
(`leftDirectionIndicator`, `rightDirectionIndicator`).

## Regression layout (LCL, LCR) — 31 channels

Matches the doc's order (minus `fileTime`) with `car2Present` appended.

| idx | name | notes |
|--:|---|---|
| 0 | latAcceleration | |
| 1 | longAcceleration | |
| 2 | brakePedalPos | mostly 0 |
| 3 | leftDirectionIndicator | binary — **label-leaky** |
| 4 | rightDirectionIndicator | binary — **label-leaky** |
| 5 | steeringAngle | |
| 6 | throttlePedalPos | ~97 discrete levels |
| 7 | vehicleSpeed | |
| 8 | yawRate | |
| 9 | egoLaneWidth | constant → 0 after scaling |
| 10 | yawAngleRightLine | = ch11 (parallel lane lines) |
| 11 | yawAngleLeftLine | |
| 12 | curvatureRightLine | = ch13 |
| 13 | curvatureLeftLine | |
| 14 | curvatureDxRightLine | = ch15 — **test-only spikes (≤5e6)** |
| 15 | curvatureDxLeftLine | **spike pair** |
| 16 | rightLaneLine | = −ch17 (distance to right line) |
| 17 | leftLaneLine | |
| 18–23 | car1{LatPos,LatVel,LongPos,LongVel,YawAngle,YawRate} | nearest right |
| 24–29 | car2{LatPos,LatVel,LongPos,LongVel,YawAngle,YawRate} | nearest left |
| 30 | car2Present | binary (left car present?) |

(LCR pickle stats deviate from mean 0 / std 1 → its StandardScaler was fit on
a different subset than the classification one.)

## Classification layout — 31 channels

Same signals, but the **two indicator binaries are moved to the end**
(28, 29); everything between `brakePedalPos` and the car block shifts down by 2,
so `egoLaneWidth` lands at 7 and the `curvatureDx` spike pair at 12–13.
Ego channels 0–7 verified by correlation (r ≈ 0.98 each to a distinct signal).

| idx | name | notes |
|--:|---|---|
| 0 | latAcceleration | verified |
| 1 | longAcceleration | verified |
| 2 | brakePedalPos | verified |
| 3 | steeringAngle | verified |
| 4 | throttlePedalPos | verified |
| 5 | vehicleSpeed | verified |
| 6 | yawRate | verified |
| 7 | egoLaneWidth | verified constant |
| 8 | yawAngleRightLine | = ch9 |
| 9 | yawAngleLeftLine | |
| 10 | curvatureRightLine | = ch11 |
| 11 | curvatureLeftLine | |
| 12 | curvatureDxRightLine | = ch13 — **spike pair** |
| 13 | curvatureDxLeftLine | **spike pair** |
| 14 | rightLaneLine | = −ch15 |
| 15 | leftLaneLine | |
| 16–21 | car1{LatPos,LatVel,LongPos,LongVel,YawAngle,YawRate} | |
| 22–27 | car2{LatPos,LatVel,LongPos,LongVel,YawAngle,YawRate} | |
| 28 | leftDirectionIndicator | binary — **label-leaky** (corr to DirInd==left) |
| 29 | rightDirectionIndicator | binary — **label-leaky** (corr to DirInd==right) |
| 30 | car2Present | binary |

Class mapping (per colleague): 0 = no intention, 1 = LCR, 2 = LCL.

## The turn-indicator label leak (important)

`scripts/analysis/indicator_leak.py`: the two indicator channels **alone**
reach **81.5% test accuracy** (full DSCNN 91.5%; internal reference 92%).
Blinker-active-in-window rate: no-intent 6.5%, LCR 91.7%, LCL 70.7%. So the
classification headline is largely "read the blinker."

The regression tasks are NOT leaky (`indicator_leak_regression.py`):
indicator-only test RMSE 0.72 (LCR) / 0.90 (LCL), far above our DSCNN
(0.439/0.459) and the SOTA (0.42/0.44). The leak is specific to classification.

Consequences for the paper (see paper/NOTES.md):
- keep indicators for a like-for-like comparison with the baseline, **but**
  add a with/without-indicators ablation to show genuine anticipation;
- lead the contribution with the **time-to-lane-change regression** (no
  trivial leak) and the **deployment efficiency**, not raw 3-class accuracy;
- report a no-signal-subset accuracy for classification.
