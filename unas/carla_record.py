"""Simulated motorway traffic for the hardware-in-the-loop link (2026-11-11).

CARLA Traffic Manager vehicles on a straight three-lane carriageway built from an OpenDRIVE string (3.75 m
lanes, as German motorways); no rendering, synchronous mode, 25 Hz physics. The state of every vehicle is
written at 5 Hz, the frame rate of the highD protocol: position, velocity, lane. Features and scenarios
are made from it by datasets/carla/prepare_carla.py with unas/sim_features.py. The traffic is simulated
and scripted (the Traffic Manager decides the lane changes); it is not data about real drivers.

Runs in WSL with the CARLA client module (Python 3.10 venv), the server started by unas/run_carla.sh:
  bash unas/run_carla.sh unas/carla_record.py --seed 1 [--vehicles 36] [--trucks 6] [--duration 300]
      [--lane-change 0.0] [--out datasets/carla/data/tm_seed1.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import carla

ROOT = Path(__file__).resolve().parent.parent
LENGTH = 12000.0            # m of road; the generated mesh ends near 10.5 km, where the simulator removes
                            # a vehicle (its rows are NaN from then on; prepare_carla.py skips them)
LANE_W = 3.75
DT = 0.04                   # physics step, s
EVERY = 5                   # record every 5th step: 5 Hz
CARS = ["vehicle.lincoln.mkz_2020", "vehicle.audi.a2", "vehicle.tesla.model3", "vehicle.bmw.grandtourer",
        "vehicle.mercedes.coupe_2020", "vehicle.nissan.patrol_2021", "vehicle.audi.tt", "vehicle.seat.leon"]
TRUCKS = ["vehicle.carlamotors.european_hgv", "vehicle.carlamotors.carlacola"]


def lane(i, kind, width, mark):
    return (f'<lane id="{i}" type="{kind}" level="false"><link/><width sOffset="0" a="{width}" b="0" c="0" d="0"/>'
            f'<roadMark sOffset="0" type="{mark}" weight="standard" color="standard" width="0.15"/></lane>')


XODR = f'''<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
 <header revMajor="1" revMinor="4" name="straight" version="1.00" north="0" south="0" east="0" west="0"/>
 <road name="motorway" length="{LENGTH}" id="1" junction="-1">
  <link/>
  <type s="0.0" type="motorway"><speed max="130" unit="km/h"/></type>
  <planView><geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="{LENGTH}"><line/></geometry></planView>
  <elevationProfile><elevation s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/></elevationProfile>
  <lateralProfile/>
  <lanes><laneSection s="0.0">
   <left>{lane(1, "shoulder", 1.0, "none")}</left>
   <center><lane id="0" type="none" level="false"><link/><roadMark sOffset="0" type="solid" weight="standard" color="standard" width="0.15"/></lane></center>
   <right>{lane(-1, "driving", LANE_W, "broken")}{lane(-2, "driving", LANE_W, "broken")}{lane(-3, "driving", LANE_W, "solid")}{lane(-4, "shoulder", 2.5, "none")}</right>
  </laneSection></lanes>
 </road>
</OpenDRIVE>'''


def connect(port: int = 2000):
    t0 = time.time()
    while True:
        try:
            client = carla.Client("127.0.0.1", port)       # a client made before the server listens never connects
            client.set_timeout(5.0)
            client.get_server_version()
            client.set_timeout(120.0)
            return client
        except RuntimeError as e:
            if time.time() - t0 > 180:
                sys.exit(f"CARLA server not reachable after 180 s: {e}")
            time.sleep(3)


def build_world(client):
    world = client.generate_opendrive_world(XODR, carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=500.0, wall_height=1.5, additional_width=0.6,
        smooth_junctions=True, enable_mesh_visibility=True))               # walls: nobody falls off the mesh
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds, st.no_rendering_mode = True, DT, True
    world.apply_settings(st)
    return world


def spawn_traffic(client, world, rng, n_cars: int, n_trucks: int, lane_change: float, tm_port: int = 8000):
    """Vehicles on a 50 m grid over the first stretch of road, trucks in the right lane; returns
    (actors, desired speeds in km/h, traffic manager)."""
    m = world.get_map()
    lib = world.get_blueprint_library()
    have = {b.id for b in lib.filter("vehicle.*")}
    cars = [c for c in CARS if c in have]
    trucks = [t for t in TRUCKS if t in have]
    slots = [(lane_id, 60.0 + 50.0 * k) for k in range(40) for lane_id in (-1, -2, -3)]
    right = [s for s in slots if s[0] == -3]
    truck_slots = [right[i] for i in rng.choice(len(right), size=n_trucks, replace=False)]
    rest = [s for s in slots if s not in truck_slots]
    car_slots = [rest[i] for i in rng.choice(len(rest), size=n_cars, replace=False)]
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(int(rng.integers(1, 1 << 30)))
    actors, desired = [], []
    for kind, places in (("truck", truck_slots), ("car", car_slots)):
        for lane_id, s in places:
            bp = lib.find(str(rng.choice(trucks if kind == "truck" else cars)))
            tf = m.get_waypoint_xodr(1, lane_id, s).transform
            tf.location.z += 0.3
            a = world.try_spawn_actor(bp, tf)
            if a is None:
                continue
            # trucks near 88 km/h; cars spread like the passing lanes of a motorway
            kmh = float(rng.normal(88.0, 4.0)) if kind == "truck" else float(np.clip(rng.normal(122.0, 14.0), 95.0, 160.0))
            actors.append(a)
            desired.append(kmh)
    world.tick()
    for a, kmh in zip(actors, desired):
        a.set_target_velocity(carla.Vector3D(kmh / 3.6 * 0.95, 0.0, 0.0))     # start near the desired speed
        a.set_autopilot(True, tm_port)
        tm.auto_lane_change(a, True)
        tm.set_desired_speed(a, kmh)
        tm.random_left_lanechange_percentage(a, lane_change)
        tm.random_right_lanechange_percentage(a, lane_change)
        tm.ignore_lights_percentage(a, 100.0)
    return actors, np.array(desired), tm


def record(world, actors, duration: float, on_frame=None):
    """Step the simulation; every 5th step store the state of all vehicles. on_frame(k, state) is called
    with each stored frame (the live link uses it)."""
    m = world.get_map()
    n = len(actors)
    half_len = np.array([a.bounding_box.extent.x for a in actors])
    half_wid = np.array([a.bounding_box.extent.y for a in actors])
    off = np.array([a.bounding_box.location.x for a in actors])
    frames = int(duration / (DT * EVERY))
    out = {k: np.zeros((frames, n), np.float32) for k in ("x", "y", "yaw", "vx", "vy")}
    out["lane_id"] = np.zeros((frames, n), np.int8)
    t0 = time.time()
    k = 0
    for step in range(frames * EVERY):
        world.tick()
        if step % EVERY != EVERY - 1:
            continue
        snap = world.get_snapshot()
        for i, a in enumerate(actors):
            s = snap.find(a.id)
            if s is None:                                   # the simulator removed the vehicle
                for key in ("x", "y", "yaw", "vx", "vy"):
                    out[key][k, i] = np.nan
                continue
            tf, vel = s.get_transform(), s.get_velocity()
            out["x"][k, i], out["y"][k, i], out["yaw"][k, i] = tf.location.x + off[i], tf.location.y, tf.rotation.yaw
            out["vx"][k, i], out["vy"][k, i] = vel.x, vel.y
            out["lane_id"][k, i] = m.get_waypoint(tf.location, project_to_road=True).lane_id
        if on_frame is not None:
            on_frame(k, {key: v[k] for key, v in out.items()})
        k += 1
        if k % 250 == 0:
            print(f"  {k * DT * EVERY:.0f} s simulated, {time.time() - t0:.0f} s wall", flush=True)
        if np.nanmax(out["x"][k - 1]) > LENGTH - 300.0:     # never reached: see LENGTH
            break
    wall = time.time() - t0
    out = {key: v[:k] for key, v in out.items()}
    out.update(half_len=half_len.astype(np.float32), half_wid=half_wid.astype(np.float32))
    return out, {"frames": k, "sim_s": round(k * DT * EVERY, 1), "wall_s": round(wall, 1),
                 "real_time_factor": round(k * DT * EVERY / wall, 1)}


def shutdown(client, world, actors, tm, tm_port: int = 8000):
    """Hand the vehicles back before destroying them: the Traffic Manager aborts the client when a
    vehicle it still drives disappears."""
    for x in actors:
        x.set_autopilot(False, tm_port)
    world.tick()
    client.apply_batch_sync([carla.command.DestroyActor(x) for x in actors], True)
    tm.set_synchronous_mode(False)
    st = world.get_settings()
    st.synchronous_mode = False
    world.apply_settings(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vehicles", type=int, default=36, help="cars")
    ap.add_argument("--trucks", type=int, default=6)
    ap.add_argument("--duration", type=float, default=300.0, help="simulated seconds")
    ap.add_argument("--lane-change", type=float, default=0.0,
                    help="Traffic Manager's random lane-change percentage per side (0 = overtaking only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    client = connect()
    world = build_world(client)
    actors, desired, tm = spawn_traffic(client, world, rng, a.vehicles, a.trucks, a.lane_change)
    state, timing = record(world, actors, a.duration)
    m = world.get_map()
    centres = {lid: m.get_waypoint_xodr(1, lid, 100.0).transform.location.y for lid in (-1, -2, -3)}
    meta = {"seed": a.seed, "cars": a.vehicles, "trucks": a.trucks, "spawned": len(actors),
            "lane_change_percentage": a.lane_change, "dt": DT, "fps": 1.0 / (DT * EVERY), "lane_width": LANE_W,
            "lane_centre_y": {str(k): round(float(v), 3) for k, v in centres.items()},
            "blueprints": [x.type_id for x in actors], "carla": client.get_server_version(), **timing}
    meta["vehicles_lost"] = int(np.isnan(state["x"][-1]).sum())
    out = Path(a.out) if a.out else ROOT / "datasets" / "carla" / "data" / f"tm_seed{a.seed}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, desired_kmh=desired.astype(np.float32), meta=json.dumps(meta), **state)
    # kinematic summary only: no network is involved here
    lane = state["lane_id"]
    changes = int((np.diff(lane.astype(int), axis=0) != 0).sum())
    speed = np.hypot(state["vx"], state["vy"])
    speed = speed[~np.isnan(speed)].reshape(-1, 1)
    print(json.dumps({**meta, "blueprints": sorted(set(meta["blueprints"])), "lane_changes": changes,
                      "speed_kmh_mean": round(float(speed.mean() * 3.6), 1),
                      "speed_kmh_p5_p95": [round(float(np.percentile(speed, 5) * 3.6), 1),
                                           round(float(np.percentile(speed, 95) * 3.6), 1)],
                      "min_speed_kmh_last_frame": round(float(speed[-1].min() * 3.6), 1)}))
    print(f"wrote {out}", flush=True)
    shutdown(client, world, actors, tm)


if __name__ == "__main__":
    main()
