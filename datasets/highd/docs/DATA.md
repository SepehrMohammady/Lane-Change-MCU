# highD data — facts and protocol

Source: highD dataset (Krajewski et al., ITSC 2018), access granted 2026-09-01,
non-redistributable — the zip and everything derived from it live under the
gitignored `datasets/highd/data/`. Cite the ITSC 2018 paper in any output.

Raw: 60 drone recordings of German highways, 25 Hz, per recording
`XX_tracks.csv` (per-frame x/y, velocities, accelerations, DHW/THW/TTC,
resolved surrounding-vehicle IDs), `XX_tracksMeta.csv` (per-vehicle class,
direction, numLaneChanges), `XX_recordingMeta.csv` (lane-marking y-positions).
879 MB zipped, 4.5 GB extracted.

## Task protocol — EarlyLCPred (Mozaffari et al., T-IV 2022)

We follow their public reference implementation
(github.com/SajjadMzf/EarlyLCPred, cloned to `Materials/EarlyLCPred/`) so
results are comparable:

- downsample to 5 Hz (`frame % 5 == 0`)
- **LC scenario**: 35 frames (7 s) ending exactly at the lane-crossing frame;
  valid only if the target vehicle stays in one lane for all 35 frames
- labels 0=LK, 1=RLC, 2=LLC, direction-normalized via `drivingDirection`
- **LK scenarios** undersampled to (RLC+LLC)/2 per recording, preferring
  tracks that do cross later (known TTLC), ending 25 frames before the
  crossing / recording end
- **features**: their 18-dim `state_ours` vector (lat/long velocity+accel,
  lateral distance to left marking, PV/FV relative velocity + distance, six
  more surrounding-vehicle distances with the alongside-fallback rule, lane
  existence flags, lane width); absent vehicle -> 400 m / 0 m/s
- **windows**: input 10 frames (2 s) slid over the scenario at 26 positions;
  TTLC label = (26 - s)/5 s in [0.2, 5.2] (their exact convention)
- split by recording: train 1-50, val 51-55, test 56-60
- normalization: per-feature min-max fitted on train

## Validation against the paper

| split | ours | paper | |
|---|--:|--:|---|
| train | 7,487 | 7,487 | exact |
| val | 932 | 932 | exact |
| test | 693 | 698 | −5 scenarios (0.7%), edge-case validity |

Two exact splits show the event extraction is identical; the small test gap is
documented, unresolved, and quoted with every result.

## Deviations from their code (all documented, all benign)

1. Lateral features computed in metres; their code mixes image-scaled units
   (scaleH=4 + margin). Per-feature affine difference — removed exactly by
   min-max normalization.
2. We train per-window models (classification + TTLC separately, like the
   DMIR pair); theirs is a multi-task model over the full 35-frame sequence.
   Metrics are computed per-window with their exact formulas either way
   (`eval_their_protocol.py`).

## Files

- `prepare_highd.py` — raw CSVs -> `data/prepared/{train,val,test}.npz` + meta
- `train_highd.py` — DSCNN baseline, cls or ttlc; logs to `logs/experiments.jsonl`
- `eval_their_protocol.py` — our models under their exact Table III metrics
