# Reference models vs our NAS models

Date 2026-07-09. The colleague supplied the internal reference models
(`Materials/Models/*.keras`, actually legacy HDF5). We evaluated them on our
test set with our own metrics (`unas/eval_reference.py`).

## Reference models

| Model | Task | params | metric | source |
|---|---|--:|---|---|
| cnn_multi | classification | 441,347 | **91.5%** (raw) / 91.7% (clipped) test acc — **verified** | our eval |
| transformer_lcr | LCR regression | 333,345 | RMSE 0.42 | reported (see below) |
| transformer_lcl | LCL regression | 49,025 | RMSE 0.44 | reported (see below) |

- The "92%" internal reference is `cnn_multi`; our test eval gives 91.5–91.7%
  (rounds to 92%). Provenance confirmed.
- The Transformers use a custom `TransformerEncoder` layer that is **not in the
  public LCIR repo** we cloned (only a functional `transformer_encoder` and
  `PositionalEncoding` are there), so we could not faithfully re-run them; we
  use their reported RMSE (0.42 / 0.44) and the param counts read from the H5.

## Head-to-head (our NAS, independently verified)

| Task | Reference | Ours | Verdict |
|---|---|---|---|
| Classification | 441k @ 91.5% | **83k @ 92.08%** (or 8k @ 91.3%) | **win**: 5.3× smaller *and* more accurate; 8k model matches the 441k CNN at 55× smaller |
| LCR regression | 333k @ RMSE 0.42 | 117k @ RMSE 0.447 (MAE 0.286) | mixed: 2.8× smaller, RMSE slightly worse (we optimize MAE) |
| LCL regression | **49k @ RMSE 0.44** | 106k @ RMSE 0.466 (MAE 0.317) | **lose**: their Transformer is smaller *and* better on RMSE |

## Honest takeaways for the paper

1. **Classification is a clean, strong win** — smaller and more accurate than the
   441k reference CNN, plus the turn-signal ablation and the F401-fits story.
2. **vs the published SOTA** (0.51 RMSE single-TTLC, FP32 Transformer): our LCR
   beats it; our models also quantize (int8 QAT, 89.82% @ 1.558 ms measured) and
   deploy where theirs did not.
3. **Regression vs the internal-reference Transformers is not yet a win on RMSE**
   — LCL in particular: their 49k Transformer (RMSE 0.44) beats our 85k
   (RMSE 0.501). Do not claim a regression win over the internal reference.
   To close it: (a) re-run with an RMSE-aware objective (the search optimizes
   MAE), (b) `save_criteria=all` so no good model is pruned, (c) push the
   footprint lower. Track before finalizing the regression claims.

This reframes the paper to lead with **classification + deployment + honesty**,
and to present regression as competitive-at-smaller-size against the *published*
SOTA while being transparent about the internal Transformers.

## RMSE-objective re-run (2026-07-10) — gap narrowed, not closed

Re-ran both regressions with an RMSE objective (MSE loss, `val_rmse`,
`save_criteria=all`, 150 rounds; `DMIR_REG_METRIC=rmse`). Best RMSE across both
the MAE and RMSE runs, independently verified:

| Task | Ours best RMSE | @ params | Reference Transf.\ | Published SOTA |
|---|--:|--:|--:|--:|
| LCR | 0.447 | 117 k | **0.42** @ 333 k | 0.51 |
| LCL | **0.466** | 103 k | **0.44** @ 49 k | 0.51 |

- LCL improved from 0.50 → **0.466** with the RMSE objective (and its best MAE
  0.317 also beats the MAE run's 0.325). LCR did not improve — the MAE run's
  117 k model (RMSE 0.447) stayed best; the RMSE run found smaller models
  (0.464 @ 62 k) but not lower RMSE.
- **We still do not beat the internal-reference Transformers on regression RMSE**
  (0.447 vs 0.42; 0.466 vs 0.44). We beat the *published* SOTA (0.51) on both,
  at 2–3× fewer parameters, and the LCL reference (49 k @ 0.44) is a genuinely
  strong small model.

**Final regression stance for the paper:** claim (i) beating the published SOTA
RMSE at a fraction of the size, (ii) a full accuracy/footprint Pareto front, and
(iii) deployability (float32 headline; int8 QAT operating point, both fit the
F401) — but do **not** claim beating the
internal-reference Transformers on regression RMSE. The headline wins are
classification and deployment.
