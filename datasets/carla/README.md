# Simulated motorway traffic (CARLA)

This folder holds a check, not a dataset about drivers. The traffic is produced by the
[CARLA](https://carla.org) simulator (0.9.16) with its Traffic Manager: scripted vehicles on a straight
three-lane carriageway. It was made to test the chain from a simulator to the microcontroller and to see
what a network trained on highD does on such traffic, before building a closed loop around the board.

## What is here

| file | what it does |
|---|---|
| `../../unas/run_carla.sh` | WSL: starts the CARLA server without rendering, runs one client script, stops the server |
| `../../unas/carla_record.py` | WSL, CARLA client: Traffic Manager vehicles (cars and trucks, desired speeds spread like motorway traffic) on an OpenDRIVE road with 3.75 m lanes, synchronous 25 Hz physics; state of every vehicle at 5 Hz to `data/tm_seed<seed>.npz` |
| `../../unas/sim_features.py` | the 18 features of the highD protocol from positions, velocities and lane geometry alone, with the neighbours found geometrically |
| `prepare_carla.py` | features and scenarios in the highD format (`data/prepared/test.npz`), lateral kinematics next to highD's (`results/kinematics.json`) |
| `transfer_check.py` | the rule below applied to the highD classifier's outputs (`results/transfer_check.json`) |

`data/` is not versioned; the recordings are reproduced with
`bash unas/run_carla.sh unas/carla_record.py --seed 1 --duration 300 --lane-change 5`.

## The feature extraction is checked on highD

`unas/sim_features.py <recording>` computes the features of every vehicle of a highD recording from its
positions and compares them with `datasets/highd/prepare_highd.py`, which reads the neighbours from the
dataset's id columns. On the five test recordings (442,901 vehicle frames) every feature is equal in at
least 99.87% of the vehicle frames of each recording
(`datasets/highd/results/hil/sim_features_check.json`).

## Result (2026-11-11): the Traffic Manager's lane changes cannot be predicted

The Traffic Manager changes lanes only when told to at random (`--lane-change`, percent per step); with
overtaking alone, 42 vehicles made no lane change in 60 s. One recording of 300 s (seed 1, 42 vehicles)
gives 191 lane-change scenarios (91 to the right, 100 to the left) and 95 lane-keeping scenarios.

| lane-change scenarios, 7 s before the crossing | simulated (191) | highD test (463) |
|---|--:|--:|
| peak lateral speed, median (10th to 90th percentile) | 4.10 (3.48 to 4.97) m/s | 0.87 (0.60 to 1.18) m/s |
| lateral speed 2 s before the crossing, median | 0.00 m/s | 0.47 m/s |
| moved toward the new lane 2 s before, median | 0.00 m | 0.46 m |
| moved toward the new lane 1 s before, median | 0.00 m | 1.09 m |
| lateral speed first above 0.2 m/s, median | 0.8 s before | 3.0 s before |

A simulated vehicle keeps its lane to the centimetre until 0.8 s before the crossing and then swerves at
more than four times the lateral speed of a driver on highD. There is nothing to recognize 2 s ahead.

Rule fixed before the network saw this traffic: the highD classifier (the deployed int8 file with highD's
input statistics) is usable in a CARLA loop if at least 80% of the lane-change scenarios carry the correct
class in the window that ends 2 s before the crossing and at most 10% of the lane-keeping windows are
called a lane change. Outcome: 11.5% and 40.3%, not met (on highD's test split: 100% and 9.4%). The network
recognizes the simulated lane changes in the last 0.8 s only (97% in the last window), and τ_c falls from
4.78 s to 0.61 s.

So a study of lane-change prediction in CARLA needs lane changes with a human lead-up: human drivers in a
driving simulator, or recorded trajectories replayed in the simulator. The result says nothing about real
traffic.
