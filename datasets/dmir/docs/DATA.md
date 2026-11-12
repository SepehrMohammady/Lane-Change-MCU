# DMIR prepared dataset — facts and findings

Verified on 2026-07-07 from the pickles in `Materials/*.zip` (extracted to `data/`).
Data was prepared and normalised by the data provider (ELIOS lab colleague).

## Layout

| Task | Folder | Files | Window | Target |
|---|---|---|---|---|
| Classification | `data-classification` | `{x,y}_{train,val,test}_multi.pkl` | 50 × 31 | {0: none, 1: LCR, 2: LCL} |
| Regression LCL | `data-regression-lcl` | `{x,y}_{train,val,test}_lcl.pkl` | 50 × 31 | time-to-LC ∈ [0, 4.0] s, step 0.1 |
| Regression LCR | `data-prepared-lcr` | `{x,y}_{train,val,test}_lcr.pkl` | 50 × 31 | time-to-LC ∈ [0, 4.0] s, step 0.1 |

## Split sizes

| Task | train | val | test |
|---|---|---|---|
| Classification | 94,620 | 16,332 | 19,722 |
| Regression LCL | 39,313 | 7,742 | 8,097 |
| Regression LCR | 33,201 | 5,828 | 7,073 |

Classification splits are exactly class-balanced (1/3 per class in every split).
Regression targets take the 41 discrete values 0.0, 0.1, …, 4.0 with a near-uniform
spread (mean ≈ 2.0).

## Data quality findings

1. **Test-only extreme spikes.** Feature pairs (12, 13) — classification and LCL —
   and (14, 15) — LCR — contain values up to ~5.0 × 10⁶ in the *test* splits only.
   Affected: 56/19,722 (cls), 37/8,097 (LCL), 21/7,073 (LCR) samples; the same value
   appears in both features of the pair. Train/val ranges stay within roughly
   [−431, 140]. The p99.9 of the affected features is < 0.5, so these are isolated
   spikes, most likely a division-by-near-zero artefact (TTC-like or ratio signal)
   in the preprocessing.
   **Provenance lead (2026-07-14, unproven):** a colleague reported some drivers
   crashed in the simulator and the team deliberately kept their data (she named
   users 34 and 43 from memory). `Materials/DMIR Test Reports.xlsx` corroborates
   **User34** (4 collisions + 2 accidents — worst combined with User1's 5 + 1)
   but shows **0/0 for User43**. Mechanism fits: a collision teleports/resets the
   ego position, and a derivative-built channel (curvatureDx) explodes at the
   discontinuity. Consistent with spikes being rare and test-only *if* an
   affected driver landed in the test split — which we cannot verify (no
   window→driver mapping in the prepared pickles). Recorded as a plausible
   hypothesis, not a conclusion.
   **Handling:** `Config.clip_to_train_range=True` clips val/test features to the
   per-feature train min/max. Training data is never modified.
2. **Feature 7 is constant zero** in the classification training split (std = 0).
   It carries no information for that task; kept in place to preserve the published
   50 × 31 input contract, may be pruned by NAS automatically.
3. No NaNs in any split.

## What the prepared pickles hold of the drives (found 2026-11-11)

`datasets/dmir/session_stream.py` rebuilds the raw sessions of the seven test drivers and locates every
test-pickle window in them (results: `datasets/dmir/results/hil/session_stream.json`):

- All test windows come from the sessions with lane changes (`userNLC`); the baseline sessions (`userNN`)
  are in no pickle.
- One affine map per channel and task reproduces every test window (19,722, 7,073 and 8,097) to 5e-7: the
  three scalers are recovered. The LCL pickle has the classification channel layout.
- Targets: the time to the lane change is 0.0 s on the last frame before the lane-line distance jumps;
  a lane change is labelled back to the previous lane change, not past it.
- The lab's rule finds 527 lane changes with enough history in these sessions; the test pickles hold 382
  of them (driver 13: 44 of 80, driver 31: 27 of 111). The frames before the others are in no class. The
  reason is not known to us.
- Free-driving windows never contain a completed lane change (a window with an ignored frame is dropped),
  so they end at least 6.4 s after the previous lane change. A stream delivers such windows after every
  lane change; see the whole-drive results in `unas/README.md`.

Lane changes come in pairs (found 2026-11-12, `datasets/dmir/lane_change_pairs.py`, results
`datasets/dmir/results/hil/lane_change_pairs.json`; `session_stream.py build val` gives the same for the seven
validation drivers, whose pickles the map recovered on the test drivers reproduces to 2e-11):

- On the two-lane road every lane change is the opposite of the one before it (520 of 520 for the test
  drivers, 531 of 531 for the validation drivers). The median gap is 8.6 s (validation: 7.7 s); 49% follow
  the previous one within 8.4 s (56%), and 78% of those are the return to the right after an overtake (81%).
- The windows before a lane change may contain the previous lane change, the free-driving windows never do.
  In the classification test pickle a completed lane change is inside 0 of the 6,574 free-driving windows,
  3,162 (48.1%) of the windows before a right lane change and 487 (7.4%) of those before a left one
  (validation: 0, 59.1%, 8.8%). In the prepared data a window that holds a lane-line jump therefore always
  means that the opposite lane change is coming.
- On whole drives 31.2% of the free-driving frames of the sessions with lane changes have a window that
  holds a completed lane change (validation: 28.6%).

Where the pairing shortcut comes from (read 2026-11-12 in the lab's public code, LC-Intention-Framework,
`data_preparation/machine_learning/CSV_to_windows_by_users_5s.ipynb`): a window before a lane change is kept
whenever its last row is labelled 0.0 to 4.0 s, whatever the window contains; a free-driving window is kept only
if none of its 50 rows is an ignored one (`"-" not in window['labels'].values`), and free-driving windows are
then drawn at random to match the count. The public record's per-driver CSV files (`userNLC.csv`, all 50
drivers) are the input of that notebook: the 30 features per frame and a `labels` column with the time to the
lane change, 4.1 for free driving and "-" for the 15 rows after a lane change. They are not exactly what the
prepared pickles were cut from: for test drivers 13 and 31 the CSV labels 78 and 99 lane changes where the rule
that reproduces every pickle target finds 80 and 111, and the notebook's rule would give 2,854 and 3,207
windows before a lane change where the LCR and LCL test pickles hold 1,690 and 999 of these drivers. Which
further selection made the pickles is the open question behind the 382 of 527 lane changes above.

Which drivers the training pickles hold (found 2026-11-12, `session_stream.py build train` and `build more`,
results `session_stream_train.json` and `session_stream_more.json`): exactly 36, the 22 drivers of the local
archive that are neither test nor validation drivers and users 37-50, whose sessions with lane changes are
in the public record (Zenodo, DOI 10.5281/zenodo.16686054; the record has no baseline sessions). Every
window is accounted for: classification 60,969 + 33,651 = 94,620, LCR 20,599 + 12,602 = 33,201, LCL
23,822 + 15,491 = 39,313, each reproduced by the map recovered on the test drivers to 2e-11. The frame
rule gives another class than the pickle for 12 of the 60,969 local classification windows: frames
between two lane-line jumps 5 to 8 frames apart, where the car straddles the line.

## Provenance (resolved 2026-07-07 — details in datasets/dmir/docs/dataset-provenance.md)

The data is the ELIOS "Lane Change Intention Recognition Dataset" (Zenodo DOI
10.5281/zenodo.16686054, MIT, CARLA simulator, 50 drivers, 10 Hz → the window
is 5 s). Project codename DMIR (Driver Maneuver Intention Recognition,
ApplePies 2024 precursor paper).

## Feature identities

The exact ordered channel names for both layouts are in
[datasets/dmir/docs/feature-map.md](feature-map.md) and machine-readable in
[`src/features.py`](../../../src/features.py). Confirmed from the provider's
`feature_description` doc + correlation to raw H5 signals. Key point: the
**turn-signal state is a feature** (channels 28/29 in the classification and LCL pickles, 3/4 in
the LCR pickle; corrected 2026-11-11) and is heavily label-leaky — see below.

## The turn-indicator label leak (⚠ affects paper framing)

`scripts/analysis/indicator_leak.py`: the two indicator channels alone reach
**81.5% test accuracy** (full DSCNN 91.5%; internal reference 92%).
Blinker-active-in-window rate — no-intent 6.5%, LCR 91.7%, LCL 70.7%. The
3-class headline is largely "read the blinker." Plan: keep indicators for a
like-for-like baseline comparison, add a with/without ablation, and lead the
paper with the regression + deployment story. Details in feature-map.md.

## Verified answers (2026-07-07, scripts/analysis/ — no colleague needed)

1. **Driver-wise split CONFIRMED** (`scripts/analysis/verify_split.py`).
   Raw per-user H5 sessions (Materials/Overtaking2.zip) matched against the
   pickles by per-window Pearson correlation (invariant to the normalization):
   official test users 13 & 2 appear ONLY in the pickle test split, val users
   10 & 5 ONLY in val, train user 22 ONLY in train (YawRate/LatAcceleration
   channels, r > 0.9999; SteeringAngle cross-hits are quantization
   false-positives). The comparison against the published RMSE 0.5102 is
   therefore apples-to-apples, and there is no window leakage across drivers.
2. **No fileTime/timestamp channel** in any pickle
   (`scripts/analysis/fingerprint_channels.py`): no channel is monotonic
   within windows with near-unique values. No timestamp leakage.
3. **Spike pairs = `curvatureDx{Right,Left}Lane`.** The spiky pairs are
   exactly-equal right/left pairs sitting where the curvature-derivative
   columns fall in the layout: (12,13) for classification/LCL, (14,15) for
   LCR. Numerical derivative → division-by-near-zero explains the ~5×10⁶
   magnitudes. Clipping to train range (our default) remains the handling.
4. **Feature 7 (classification) = `egoLaneWidth`**: constant 3.75 m on the
   2-lane highway → 0 after standardization. Harmless; NAS may prune it.
5. **Normalization**: classification stats ≈ StandardScaler fit on train
   (per-channel mean ≈ 0, std ≈ 1). The LCR set deviates (std up to 3.9,
   means to −1.2) — its scaler was fit on a different subset; also its
   channel layout differs (egoLaneWidth at index 9, indicators mid-block,
   vs index 7 / end-block for classification). Treat tasks as separate
   input spaces.

## Still open (nice-to-have, non-blocking)

- [x] Exact identity of all 31 classification channels, including the two
      neighbour-car blocks: recovered and verified against the pickle on
      2026-09-09 (feature-map.md, last section).
- [ ] Provenance of the internal 92%/0.42/0.44 reference results (model,
      spike handling) — needed only for the paper's "internal reference" row.
