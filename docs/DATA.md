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
   **Handling:** `Config.clip_to_train_range=True` clips val/test features to the
   per-feature train min/max. Training data is never modified.
2. **Feature 7 is constant zero** in the classification training split (std = 0).
   It carries no information for that task; kept in place to preserve the published
   50 × 31 input contract, may be pruned by NAS automatically.
3. No NaNs in any split.

## Provenance (resolved 2026-07-07 — details in docs/research/dataset-provenance.md)

The data is the ELIOS "Lane Change Intention Recognition Dataset" (Zenodo DOI
10.5281/zenodo.16686054, MIT, CARLA simulator, 50 drivers, 10 Hz → the window
is 5 s). Project codename DMIR (Driver Maneuver Intention Recognition,
ApplePies 2024 precursor paper).

## Feature identities

The exact ordered channel names for both layouts are in
[docs/research/feature-map.md](research/feature-map.md) and machine-readable in
[`src/features.py`](../src/features.py). Confirmed from the provider's
`feature_description` doc + correlation to raw H5 signals. Key point: the
**turn-signal state is a feature** (channels 28/29 classification, 3/4
regression) and is heavily label-leaky — see below.

## The turn-indicator label leak (⚠ affects paper framing)

`scripts/analysis/indicator_leak.py`: the two indicator channels alone reach
**81.5% test accuracy** (full DSCNN 91.5%; internal reference 92%).
Blinker-active-in-window rate — no-intent 6.5%, LCR 91.7%, LCL 70.7%. The
3-class headline is largely "read the blinker." Plan: keep indicators for a
like-for-like baseline comparison, add a with/without ablation, and lead the
paper with the regression + deployment story. Details in feature-map.md and
paper/NOTES.md.

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

- [ ] Exact name list of all 31 channels per task (ego-block order inferred,
      not confirmed; the 3 trailing binaries in classification unidentified —
      likely indicators + one flag).
- [ ] Provenance of the internal 92%/0.42/0.44 reference results (model,
      spike handling) — needed only for the paper's "internal reference" row.
