# exiD: reproducing these results

exiD (Moers et al., "The exiD Dataset: A Real-World Trajectory Dataset of Highly
Interactive Highway Scenarios in Germany", IEEE IV 2022) was recorded with the
highD drone method at entries and exits of German motorways. We prepare it in the
highD scenario format (Mozaffari et al., IEEE T-IV 2022), so the highD networks,
training code and board builds apply unchanged, and a model trained on one
dataset can be tested on the other. There is no published lane-change prediction
result on exiD to compare with; the comparisons here are between our models and
between the two datasets.

**The dataset itself is not redistributable.** Request exiD from
[levelxdata.com/exid-dataset](https://levelxdata.com/exid-dataset/) (free for
non-commercial use) and unpack `data/` and `maps/lanelet2/` of the release zip
(v2.1) into `datasets/exid/data/raw/`, so that
`datasets/exid/data/raw/data/00_tracks.csv` and
`datasets/exid/data/raw/maps/lanelet2/0_cologne_butzweiler/location0.osm` exist.

## 1. Build the scenarios (about 5 min)

```bash
python datasets/exid/prepare_exid.py
```

Writes `data/prepared/{train,val,test}.npz`, `meta.json` and `index.csv`.
**Checkpoint:** 16,446 / 2,617 / 1,757 scenarios for train / val / test
(committed copy of the counts: `results/prepared_meta.json`). How the highD
protocol was adapted to exiD's curved roads, merging lanes and Lanelet2 maps is
described in `docs/DATA.md`.

## 2. Train and compare (about 1.5 h on one RTX 5070)

```bash
python datasets/exid/run_experiments.py
```

Hand-designed DSCNN (8,371 parameters, shared training loop in
`src/lc_windows.py`), seeds 0-4: trained on exiD; a highD model applied to exiD
as it is; the same highD model fine-tuned on exiD with the same recipe; both
again with 10% and 25% of the exiD training data. Every model is also tested on
highD. Evaluations go to `results/transfer.jsonl`, training runs to
`logs/experiments.jsonl`, weights to `results/weights/`. The runs use
deterministic cuDNN kernels, so a repeated seed repeats the result.

The architectures found by the highD searches are retrained on exiD with
`unas/seed_variance.py` (WSL, `exid_*` entries, both recipes; results in
`results/seeds/`), and their deployed highD weights are applied to exiD by
`unas/transfer_eval.py` (`results/transfer_searched.jsonl`).

## 3. Results (test set, five seeds)

| model | trained on | accuracy | TTLC RMSE |
|---|---|--:|--:|
| hand-designed CNN, 8.4 k | exiD | 89.93 ± 0.24% | 0.406 ± 0.008 s |
| hand-designed CNN, 8.4 k | highD, applied as is | 61.24 ± 2.32% | 0.867 ± 0.062 s |
| same weights, exiD input scaling | highD, applied as is | 38.93 ± 4.66% | 1.209 ± 0.067 s |
| hand-designed CNN, 8.4 k | highD, then fine-tuned on exiD | 90.13 ± 0.22% | 0.393 ± 0.002 s |
| searched on highD, 5.3 k | exiD (search / hand recipe) | 76.67 ± 3.52 / 77.99 ± 0.84% | – |
| searched on highD, 7.9 k | exiD (search / hand recipe) | 77.91 ± 1.11 / 78.59 ± 1.59% | – |
| searched on highD, 28 k | exiD (search / hand recipe) | – | 0.487 ± 0.074 / 0.588 ± 0.069 s |

The model trained on exiD scores 84.64 ± 0.67% on the highD test set; the highD
model scores 61.24% on exiD, so exiD carries over to highD much better than the
reverse. By scenario kind (hand-designed CNN trained on exiD, and in brackets the
highD model applied as is): lane keeping 84.7% (45.4%), main carriageway 89.8%
(76.6%), merge from an on-ramp 95.4% (74.1%), exit to an off-ramp 94.2% (33.2%),
on a ramp 97.3% (79.3%). With 10% of the exiD training scenarios, a highD start
adds 1.3 points (84.22 against 82.94%, Welch p = 0.02); with 25% it adds 0.6
(p = 0.12) and with all of them 0.2 (p = 0.22). For time to lane change the highD
start lowers RMSE slightly at every data size (0.393 against 0.406 s with all data,
p = 0.02). Trained on exiD, the model keeps its predictions correct from 4.71 s
before the crossing on average (τ_c, metric of Mozaffari et al.), AUC 0.956.

## 4. Board cost

exiD uses the highD input (10 × 18) and the same networks, so every build has
the latency, flash and RAM measured for highD in
`datasets/highd/results/deploy/`; board cost depends on the graph, not on the
weights.
