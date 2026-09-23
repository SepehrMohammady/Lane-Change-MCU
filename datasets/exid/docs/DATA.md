# exiD data: facts and protocol

Source: exiD dataset v2.1 (Moers et al., "The exiD Dataset: A Real-World Trajectory
Dataset of Highly Interactive Highway Scenarios in Germany", IEEE IV 2022,
pp. 958-964, doi 10.1109/IV51971.2022.9827305). Access granted by fka / levelXdata
on 2026-09-23 for non-commercial use. The data may not be redistributed: the zip
stays in `Materials/` and everything extracted or derived from it lives under the
gitignored `datasets/exid/data/`. Trained models and aggregate results may be
shared (licence: "abstract representations ... such as models trained on it").
Cite the IV 2022 paper in any output.

Raw: 93 drone recordings (00-92) at 7 motorway entry/exit locations near Cologne
and Aachen, 25 Hz; 2.0 GB zipped, 5.4 GB unpacked without the 3D scene files. Per recording `XX_tracks.csv`,
`XX_tracksMeta.csv`, `XX_recordingMeta.csv`, `XX_background.png`; per location a
Lanelet2 map (`maps/lanelet2/*/location*.osm`) and an OpenDRIVE map. The 3D
scene files (`.fbx`, `.osgb`) are not extracted. The official column description
is `data/raw/exiD-Format_2_1.pdf` (downloaded from levelxdata.com).

| location (map folder) | recordings | minutes | speedLimit in recordingMeta |
|---|---|--:|---|
| 0 cologne_butzweiler | 00-18 | 85 | 100 km/h |
| 1 cologne_fortiib | 19-38 | 144 | 100 km/h |
| 2 aachen_brand | 39-52 | 239 | not given (-1) |
| 3 bergheim_roemer | 53-60 | 158 | not given (-1) |
| 4 cologne_klettenberg | 61-72 | 102 | 120 km/h |
| 5 aachen_laurensberg | 73-77 | 90 | not given (-1) |
| 6 merzenich_rather | 78-92 | 149 | not given (-1) |

Tracks: 69,430 vehicles (52,762 cars, 3,940 vans, 12,651 trucks, 77 motorcycles);
no pedestrians or bicycles.

## Protocol: the highD scenario format, adapted

`prepare_exid.py` writes exiD in the scenario format of `datasets/highd/prepare_highd.py`
(EarlyLCPred, Mozaffari et al., IEEE T-IV 2022), so the same models, training code
(`src/lc_windows.py`) and metrics apply, and a model trained on one dataset can be
tested on the other.

Kept exactly: 5 Hz (frame % 5 == 0); lane-change scenario = 35 frames (7 s) ending
at the first frame in the new lane, valid only without another crossing in those
frames; labels 0 lane keep, 1 right change, 2 left change; one lane-keeping
scenario per track ending 25 frames before its first crossing (or the track end),
undersampled to (right + left) / 2 per recording; windows of 10 frames at 26
positions, TTLC (26 - s) / 5 s; the 18 features in the same units and signs
(lateral positive to the left, longitudinal velocity and acceleration negated
as in the highD code, 400 m / 0 m/s for an absent vehicle); min-max normalization
on the training split.

Adapted, because exiD roads curve and its lanes merge and split:

1. **Crossings** come from exiD's `laneChange` flag (first frame with the centroid
   in the new lanelet). The side is the side of the old lane the vehicle left
   through (sign of `latLaneCenterOffset` before the flag, positive = left of the
   centreline). Checked against the Lanelet2 adjacency (new lanelet shares the old
   one's left or right boundary): agreement on every one of 1,603 crossings in
   recordings 0, 39, 66 and 80 (four locations). The offset jumps by 3.5-3.6 m on
   average at a crossing, one lane width.
2. **Lateral velocity and acceleration.** exiD's `latVelocity` is in the vehicle
   frame (side slip): 0.03 m/s on average just before a crossing, against
   0.84 m/s for highD's lateral velocity. We therefore differentiate the lateral
   position relative to the lane centreline (`latLaneCenterOffset` with the step
   at each crossing or centreline switch removed) with a Savitzky-Golay filter.
   The map centrelines are polylines; on curves a vehicle sees their vertices as
   offset steps of about 5 cm every 1.6-2 s, which a 1-s filter passes into the
   acceleration (lane-keeping |lat acc| p95 0.95 m/s² on a sample of seven
   recordings, against 0.45 m/s² with 2 s). A
   2-s window (51 frames, order 2) was chosen because it makes the lateral
   kinematics of the last 3 s before a crossing match highD, and it was chosen
   on that comparison, not on model accuracy:

   | last 3 s before a crossing | exiD | highD |
   |---|--:|--:|
   | median lateral velocity | 0.57 m/s | 0.59 m/s |
   | median lateral acceleration | 0.21 m/s² | 0.21 m/s² |

   Lane keeping stays livelier than on highD (|lat velocity| p95 0.37 against
   0.17 m/s, |lat acc| p95 0.40 against 0.10 m/s²): ramps, widening lanes and the
   drift of merging vehicles are real lateral motion relative to the lane, and
   some polyline noise remains.
3. **Longitudinal velocity and acceleration**: `lonVelocity`, `lonAcceleration`.
4. **Distances to surrounding vehicles**: centre-to-centre distance projected on
   the target vehicle's heading (highD: difference of x on a straight road). The
   eight neighbours use exiD's lead/rear/left/right IDs with the highD
   alongside-fallback rule; of several alongside vehicles the nearest is used.
5. **Lane existence**: a same-direction neighbouring lanelet of subtype `highway`
   that is not separated by a `road_border`. Hard shoulders (`emergency_lane`)
   do not count.
6. **Hard shoulders**: highD's lane markings contain no shoulder lane, so
   crossings into or out of an `emergency_lane` are not used and no scenario may
   contain frames on one (1,119 crossings excluded).
7. **Lane-change kind** from the Lanelet2 tags: merge (leaving an `onramp`
   lanelet), exit (entering an `offramp` lanelet), on a ramp (both lanelets on
   ramps), main carriageway (the rest).
8. **Split by recording within each location**: the latest recordings whose total
   duration is closest to 10% of that location's recorded time form the test set,
   the next ones the validation set. Every location appears in every split.

| split | recordings | share of time | scenarios | lane keep | right | left |
|---|--:|--:|--:|--:|--:|--:|
| train | 69 | 78.8% | 16,446 | 5,471 | 3,525 | 7,450 |
| val | 10 | 12.7% | 2,617 | 870 | 535 | 1,212 |
| test | 14 | 8.5% | 1,757 | 583 | 418 | 756 |

Test recordings: 15-18, 37-38, 51-52, 60, 70-72, 77, 92. Validation: 14, 35-36,
49-50, 59, 69, 76, 90-91.

Lane changes by kind (all splits): main carriageway 6,664, merge from an on-ramp
4,657, exit to an off-ramp 2,128, on a ramp 447. Left changes outnumber right
ones about 2:1 because merges go left; location 5 is almost only merges (1,836
left, 81 right). For comparison, highD has 9,112 scenarios with right and left
changes in similar numbers.

Crossings not used as scenarios: 9,987 less than 7 s after the vehicle entered
the recorded area (the same rule as highD; many merges happen soon after a ramp
vehicle appears), 2,432 with another crossing inside the 7 s, 1,119 involving a
hard shoulder, 1 with a missing lane value.

## Feature ranges (training split, p1 / median / p99)

| feature | exiD | highD |
|---|---|---|
| lat_velocity | -0.86 / 0.05 / 1.02 | -1.04 / -0.01 / 1.00 |
| long_velocity | -38.18 / -23.03 / -11.12 | -40.01 / -30.09 / -8.31 |
| lat_acceleration | -0.59 / 0.01 / 0.70 | -0.41 / -0.01 / 0.41 |
| long_acceleration | -1.43 / -0.13 / 0.76 | -0.91 / -0.01 / 1.02 |
| lat_pos_left_marking | 0.12 / 2.06 / 4.20 | 0.19 / 1.94 / 3.72 |
| rel_velo_pv | -9.32 / 0.00 / 9.60 | -9.64 / 0.15 / 11.23 |
| dist_pv | 13.30 / 68.77 / 400.00 | 13.89 / 55.31 / 400.00 |
| rel_velo_fv | -8.08 / 0.00 / 8.53 | -8.19 / 0.00 / 8.27 |
| dist_fv | 13.19 / 68.17 / 400.00 | 14.81 / 62.91 / 400.00 |
| dist_rpv | 29.40 / 400.00 / 400.00 | 26.20 / 400.00 / 400.00 |
| dist_rv | 1.30 / 400.00 / 400.00 | 0.50 / 29.01 / 400.00 |
| dist_rfv | 25.02 / 400.00 / 400.00 | 23.30 / 400.00 / 400.00 |
| dist_lpv | 18.15 / 400.00 / 400.00 | 20.43 / 400.00 / 400.00 |
| dist_lv | 0.51 / 51.48 / 400.00 | 0.50 / 47.45 / 400.00 |
| dist_lfv | 16.60 / 400.00 / 400.00 | 21.00 / 400.00 / 400.00 |
| left_lane_exists | 0 / 1 / 1 | 0 / 1 / 1 |
| right_lane_exists | 0 / 0 / 1 | 0 / 1 / 1 |
| lane_width | 3.25 / 3.87 / 5.97 | 3.56 / 3.85 / 4.32 |

exiD traffic is slower (median 83 against 108 km/h), more vehicles drive in the
rightmost lane or on a ramp (no right lane in at least half of the frames), and
where lanes merge or split they reach 6 m in width (p99).

## Files

- `prepare_exid.py`: raw CSVs and Lanelet2 maps -> `data/prepared/{train,val,test}.npz`,
  `meta.json`, `index.csv` (scenario list: split, recording, track, crossing frame);
  a copy of `meta.json` is committed as `results/prepared_meta.json`
- `train_exid.py`: hand-designed DSCNN on exiD (shared loop in `src/lc_windows.py`)
- `run_experiments.py`: five-seed exiD training, highD -> exiD transfer and
  fine-tuning; evaluations in `results/transfer.jsonl`
