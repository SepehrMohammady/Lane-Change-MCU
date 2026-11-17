"""LCIR: the 31 raw features of a frame from the live simulator, as the lab's converter computes them (2026-11-17).

The LCIR sessions were recorded with the lab's CARLA client and turned into features afterwards: the recorder
file is read frame by frame (generate_CDF_Carla_log_to_H5.py of the public LC-Intention-Framework, MIT licence),
every second frame is kept (20 Hz -> 10 Hz), and datasets/dmir/scenario_race.raw_matrix takes the 31 raw signals
in the classification layout from the session file. For a closed loop with a driver the same numbers are
needed while the drive goes on. LiveFeatures.step(world) computes them from the running simulator with the
converter's formulas, copied here (the angle difference, the ego base vector, the velocity difference over
deltaTime = 0.1 s for the acceleration, the lane lines from the map's waypoints with the converter's curvature
helpers, the objects in creation order), and assembles the row as raw_matrix does (car1 = the nearest of the
object slots 1-10 by straight-line distance, car2 = slot 11, the scenario's fast car on the left).

The converter reads the recording's text (client.show_recorder_file_info): positions in centimetres,
rotations, velocities and pedal values stored as float32 and printed with 6 significant digits. On a 60 km
road the positions reach 6 to 7 digits in centimetres, so the converter's positions are rounded to 1 to 10
cm. rec() applies the same storage and printing to the live values, so that the network sees what the
dataset pipeline would have given it (first check of 2026-11-17: without it, differences of up to 1 cm).

Python 3.8 (the client's environment); numpy and the carla module of the client. The check against the
converter on the same recording is in datasets/dmir/live_check.py.
"""
from __future__ import annotations

import math

import numpy as np

DELTA_T = 0.1                    # the converter divides velocity differences of kept frames by 0.1 s
CURV_STEP = 0.1                  # metres between the waypoints of the curvature helpers
N_FEAT = 31


def rec(x: float) -> float:
    """A value as the recorder's text gives it: float32, printed with 6 significant digits."""
    return float(f"{float(np.float32(x)):g}")


def angle_difference(unit1: float, unit2: float) -> float:
    """generate_CDF_Carla_log_to_H5.get_angle_difference: degrees in (-180, 180]."""
    phi = abs(unit2 - unit1) % 360
    sign = 1
    if not ((0 <= unit1 - unit2 <= 180) or (-360 <= unit1 - unit2 <= -180)):
        sign = -1
    return (360 - phi if phi > 180 else phi) * sign


def calcu_c_1(wp, d: float) -> float:
    """The converter's curvature from the waypoints d before and after."""
    nxt, prv = wp.next(d)[0].transform, wp.previous(d)[0].transform
    x1, y1, yaw1 = nxt.location.x, nxt.location.y, nxt.rotation.yaw
    x2, y2, yaw2 = prv.location.x, prv.location.y, prv.rotation.yaw
    return 2 * math.sin(math.radians((yaw1 - yaw2) / 2)) / math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calcu_c_derivative(wp, d: float) -> float:
    """The converter's derivative of the curvature."""
    nxt, cen, prv = wp.next(d)[0].transform, wp.transform, wp.previous(d)[0].transform
    yaw1, yawc, yaw2 = nxt.rotation.yaw, cen.rotation.yaw, prv.rotation.yaw
    xc, yc, x2, y2 = cen.location.x, cen.location.y, prv.location.x, prv.location.y
    return math.radians(yaw1 - 2 * yawc + yaw2) / (math.sqrt((xc - x2) ** 2 + (yc - y2) ** 2)) ** 2


def base_vector(yaw_deg: float) -> np.ndarray:
    """The converter's ego_base_vector: row 0 longitudinal, row 1 lateral (toward the left positive)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, s], [s, -c]])


class LiveFeatures:
    """Call step(world) after every simulator step once the recorder runs; every second step returns the
    row of 31 raw features (the others return None), as the converter keeps every second frame."""

    def __init__(self, carla_module, keep_parity: int = 0):
        self.carla = carla_module
        self.n = 0                                          # steps since the recorder started
        self.keep_parity = keep_parity
        self.old_v = np.zeros(3)                            # the converter starts from a velocity of 0
        self.lane = {"width": 0.0, "dy0": 0.0, "dy1": 0.0, "yaw": 0.0, "cur": 0.0, "cdx": 0.0}
        self.map = None
        self.ego_id = None

    def _ego(self, actors):
        if self.ego_id is not None:
            a = actors.find(self.ego_id)
            if a is not None:
                return a
        for a in actors.filter("vehicle.*"):
            if a.attributes.get("role_name") == "hero":
                self.ego_id = a.id
                return a
        return None

    def _lane_lines(self, ego_pos: np.ndarray, ego_yaw: float) -> None:
        """generate_CDF_Carla_log_to_H5.save_lane_lines, the values the 31 features use; off the road the
        previous values stay, as in the converter."""
        carla = self.carla
        loc = carla.Location(float(ego_pos[0]), float(ego_pos[1]), float(ego_pos[2]))
        if self.map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving) is None:
            return
        wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        width = wp.lane_width
        left, right = wp.get_left_lane(), wp.get_right_lane()
        dy0, dy1 = self.lane["dy0"], self.lane["dy1"]
        if right is not None:                               # both branches of the converter give these two
            dy0 = math.sqrt((right.transform.location.x - ego_pos[0]) ** 2 + (right.transform.location.y - ego_pos[1]) ** 2) - right.lane_width / 2
            dy1 = width - dy0
        if left is not None:
            dy1 = math.sqrt((left.transform.location.x - ego_pos[0]) ** 2 + (left.transform.location.y - ego_pos[1]) ** 2) - left.lane_width / 2
            dy0 = width - dy1
        if right is None and left is None:
            yaw = wp.transform.rotation.yaw
            road = [[math.cos(math.radians(yaw)), math.sin(yaw)], [math.sin(math.radians(yaw)), -math.cos(math.radians(yaw))]]
            lat = np.matmul(road, ego_pos[:2] - np.array([wp.transform.location.x, wp.transform.location.y]))[1]
            dy0, dy1 = width + lat, width - lat
        ego_rot_360 = ego_yaw + 360
        if ego_rot_360 > 360:
            ego_rot_360 -= 360
        self.lane = {"width": width, "dy0": dy0, "dy1": dy1, "cur": calcu_c_1(wp, CURV_STEP),
                     "cdx": calcu_c_derivative(wp, CURV_STEP),
                     "yaw": math.radians(angle_difference(ego_rot_360, wp.transform.rotation.yaw))}

    def step(self, world):
        """One simulator step; returns (row of 31 raw features, extras) on the kept steps, else None."""
        self.n += 1
        if (self.n - 1) % 2 != self.keep_parity:
            return None
        if self.map is None:
            self.map = world.get_map()
        actors = world.get_actors()
        ego = self._ego(actors)
        if ego is None:
            return None
        vehicles = sorted(actors.filter("vehicle.*"), key=lambda a: a.id)       # creation order
        pos, vel, yaw, angz = [], [], [], []
        for a in vehicles:                                  # as the recording's text gives them (rec)
            t, v, w = a.get_transform(), a.get_velocity(), a.get_angular_velocity()
            pos.append(np.array([rec(t.location.x * 100) / 100, rec(t.location.y * 100) / 100, rec(t.location.z * 100) / 100]))
            vel.append(np.array([rec(v.x), rec(v.y), rec(v.z)]))
            yaw.append(rec(t.rotation.yaw))
            angz.append(rec(w.z))
        e = next(i for i, a in enumerate(vehicles) if a.id == ego.id)
        ego_pos, ego_vel, ego_yaw, ego_angz = pos[e], vel[e], yaw[e], angz[e]
        B = base_vector(ego_yaw)
        speed = float(np.linalg.norm(ego_vel))
        acc3 = (ego_vel - self.old_v) / DELTA_T
        self.old_v = ego_vel
        lat_acc, long_acc = float(np.matmul(B, acc3[:2])[1]), float(np.matmul(B, acc3[:2])[0])
        yaw_rate = -ego_angz * math.pi / 180
        ctrl = ego.get_control()
        throttle = float(np.int32(rec(ctrl.throttle) * 100))
        brake = float(np.int32(rec(ctrl.brake) * 100))
        steering = -rec(ctrl.steer) * math.pi * 5
        lights = ego.get_light_state()
        ls = self.carla.VehicleLightState
        indicator = 1 if int(lights) & int(ls.LeftBlinker) else 2 if int(lights) & int(ls.RightBlinker) else 0
        self._lane_lines(ego_pos, ego_yaw)
        ego_rot_360 = ego_yaw + 360
        if ego_rot_360 > 360:
            ego_rot_360 -= 360
        objs = []                                           # per slot: LatPos, LatVel, LongPos, LongVel, YawAngle, YawRate
        for i in range(len(vehicles)):
            dp, dv = B @ (pos[i] - ego_pos)[:2], B @ (vel[i] - ego_vel)[:2]
            r360 = yaw[i] + 360
            if r360 > 360:
                r360 -= 360
            objs.append([dp[1], dv[1], dp[0], dv[0], math.radians(angle_difference(ego_rot_360, r360)), ego_angz - angz[i]])
        car1 = [0.0] * 6
        traffic = objs[1:11]
        if traffic:
            j = int(np.argmin([math.hypot(o[0], o[2]) for o in traffic]))
            car1 = traffic[j]
        car2 = objs[11] if len(objs) > 11 else [0.0] * 6
        L = self.lane
        row = [lat_acc, long_acc, brake, steering, throttle, speed, yaw_rate, L["width"], L["yaw"], L["yaw"], L["cur"], L["cur"],
               L["cdx"], L["cdx"], L["dy0"], L["dy1"], *car1, *car2, float(indicator == 1), float(indicator == 2),
               float(len(objs) > 11)]
        extras = {"step": self.n, "sim_time": world.get_snapshot().timestamp.elapsed_seconds, "vehicles": len(vehicles),
                  "ego_slot": e, "car2_id": vehicles[11].id if len(vehicles) > 11 else -1}
        return np.array(row, np.float64), extras
