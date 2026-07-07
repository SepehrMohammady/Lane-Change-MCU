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
is 5 s). The official feature list has 30 channels + fileTime = 31 CSV
columns; feature identities (ego dynamics, lane geometry, two nearest
vehicles) are listed in the provenance note. The spiky feature pairs are
plausibly `curvatureDx*` (numerical derivative → division-by-near-zero).

## Open questions for the data provider (TO-DO)

- [ ] ⛔ Exact column order of the pickles; is channel 31 fileTime? (drop it
      if so — no physical meaning, leakage risk).
- [ ] ⛔ Is the pickle split driver-wise per the official protocol (val users
      {5,8,10,12,16,19,27}, test {2,7,13,18,25,31,36})? Required for a fair
      comparison against the published RMSE 0.5102.
- [ ] Are the test-split spikes on features 12–15 known artefacts (curvatureDx)?
      How were they handled in the internal 92%/0.42/0.44 runs?
- [ ] What normalization was applied (StandardScaler fit on train only?).
- [ ] Provenance of the internal 92%/0.42/0.44 results (model, protocol).
