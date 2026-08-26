# Dataset provenance — DMIR (Driver Maneuver Intention Recognition)

Researched 2026-07-07.

## Identity

The public dataset is not published under the name "DMIR". Our data is, with
high confidence, the ELIOS lab **"Lane Change Intention Recognition Dataset"**:

- Zenodo DOI **10.5281/zenodo.16686054**, v1.0.0, published 2025-08-01,
  **MIT license**, 486.7 MB (50 CSV + 50 H5, one per driver).
- **DMIR** = Driver Maneuver Intention Recognition, the acronym used in the
  precursor paper and as this project's codename: Forneris et al., "Setting Up
  a Lightweight
  Simulation Environment for Automated Driving Dataset Collection,"
  ApplePies 2024, Springer LNEE, DOI 10.1007/978-3-031-84100-2_19.
- Project page: https://elios-lab.github.io/LC-Intention-Framework/

## Collection (confirmed from the ApplePies 2024 precursor paper, in hand)

Human drivers (50 user files, >3,400 annotated lane changes) drove in the
**CARLA 0.9.15 simulator** (synchronous mode, fixed_delta_seconds=0.051,
~20 FPS, **Logitech G920** wheel + pedals, 29" ultra-wide curved screen,
105° FOV, cockpit with mirrors/tachometer/odometer, RPM-linked engine sound)
on a custom RoadRunner 60 km one-way **2-lane highway, 11.5 m wide** (mean
curve angle 35.77°, mean radius 500 m, bridge instead of junctions); max ego
speed 140 km/h; procedural traffic (NEVs respawned ahead of the ego).
Sessions logged with CARLA logger → converted to the **L3Pilot Common Data
Format** H5 (`generate_l3pilot_from_Carla.py`); features at 10 Hz; 5 s
windows → 50 timesteps. Two phases per participant: baseline (light traffic,
no overtaking) and treatment (traffic, lane changes allowed). Precursor
funding: EU H2020 **Hi-Drive, grant 101006664**. Pilot test: 4 participants,
~6 lane changes/minute.

Raw materials now in the repo's `Materials/` (not versioned):
`user46(04.12.24).zip` (raw CARLA session logs), `Overtaking2.zip` (the full
data-prep repo: 73 per-user H5 sessions, windowing notebook
`h5_to_CSV.ipynb`, CDF schema `structure_definitions.py`, per-window sample
CSVs with the raw ~120-column L3Pilot header).

## The 31 columns (from the repo's labeling script)

- 10 ego: fileTime, latAcceleration, longAcceleration, brakePedalPos,
  leftDirectionIndicator, rightDirectionIndicator, steeringAngle,
  throttlePedalPos, vehicleSpeed, yawRate
- 9 lane geometry: egoLaneWidth, yawAngleRight/LeftLine,
  curvatureRight/LeftLine, curvatureDxRight/LeftLine, right/leftLaneLine
- 12 surrounding traffic: {car1, car2} × {LatPosition, LatVelocity,
  LongPosition, LongVelocity, YawAngle, YawRate}

The official README counts **30 features** (excludes fileTime); our windows have
**31** channels. This was investigated and **resolved** (see the checklist
below): there is **no fileTime channel** in any pickle — the 31st is a real
feature (`car2Present`), not a timestamp, so nothing needs dropping. The
test-split spikes on feature pairs (12,13)/(14,15) are confirmed as the
`curvatureDx` derivative pair (division-by-near-zero) and the lane-line pair,
handled by `clip_to_train_range`.

## Labeling

Lane change detected when the right lane-line lateral position shifts by more
than half the ego-lane width; the 4 s (40 timesteps) before the LC are labeled
with time-to-LC (0.0–4.0 s, 0.1 steps); 4.1 = free ride; 1.5 s after each LC
excluded.

## Official split (driver-wise)

validation users {5,8,10,12,16,19,27}; test users {2,7,13,18,25,31,36};
train = remaining 36 users.

## Empirically resolved (2026-07-07, scripts/analysis/ — see datasets/dmir/docs/DATA.md)

- [x] **Driver-wise split confirmed** by window-matching per-user H5 sessions
      against the pickles (official val/test user sets reproduced exactly).
- [x] **No fileTime channel** in any pickle; classification layout:
      7 ego channels, egoLaneWidth (constant→0) at 7, yawAngle pair (8,9),
      curvature pair (10,11), **curvatureDx pair (12,13) = the spike pair**,
      lane-line pair (14,15, r = −1), 12 car channels, 3 trailing binaries.
      LCR layout shifted (egoLaneWidth at 9, indicators mid-block,
      curvatureDx = spike pair at (14,15)).
- [x] Normalization ≈ StandardScaler fit on train for classification; LCR
      scaler fit on a different subset (stats off 0/1).

## Feature identities (resolved 2026-07-07)

Colleague supplied the authoritative `feature_description` doc and confirmed
the split-by-user, no-fileTime-channel, `car2Present` as the 31st channel, and
constant `egoLaneWidth`. Full verified channel map for both layouts:
[feature-map.md](feature-map.md); machine-readable `src/features.py`. Includes
the turn-indicator label-leak analysis (81.5% classification accuracy from the
blinker alone).

## Spike provenance lead (2026-07-14, from colleague + test reports)

A colleague reported some drivers **crashed** in the simulator ("many
accidents", naming users 34 and 43 from memory) and the team deliberately kept
their data. `Materials/DMIR Test Reports.xlsx` (50 per-driver sheets with
Collisions/Accidents counts) corroborates part of it:

| user | collisions | accidents |
|---|--:|--:|
| User1 | 5 | 1 |
| User34 | 4 | 2 |
| User43 | 0 | 0 |

So **User34 confirmed, User43 not** (User1 is actually the other heavy
crasher). Mechanism is consistent with the test-only 5×10⁶ spikes on the
curvatureDx pairs: a collision teleports/resets the ego pose, and a numerical
derivative explodes at the discontinuity. NOT proven — we have no
window→driver mapping for the prepared pickles, so we cannot check whether the
spiking windows belong to a crashing driver, nor whether User1/34 are in the
test split. Status: plausible hypothesis; the paper keeps the neutral
"division-by-near-zero artefact" wording, and clipping handles it either way.

## Open questions (for colleague — non-blocking)

- [ ] Provenance of the internal 92%/0.42/0.44 reference results. Colleague:
      92% is a CNN, the others are Transformers — exact configs pending.
- [ ] Are User1/User34 (the crash-heavy drivers) in the official test split?
      Would settle the spike-provenance hypothesis above. (Also: colleague
      named User43, but the xlsx shows 0 collisions/accidents for them —
      worth double-checking which users she meant.)
