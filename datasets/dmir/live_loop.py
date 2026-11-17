"""Closed loop B: the board predicts the driver's lane change during a drive in the lab's simulator (2026-11-17).

install() puts it into the lab's public driving client (LC-Intention-Framework, data_collection/main.py)
without changing any of its files, from a launcher that runs the client in the same process: after every
simulator step LiveLoop.on_tick computes, on every second step since the recorder
started, the 31 raw features of the frame (live_features.LiveFeatures, the converter's formulas), sends them to
the STM32H7B3I-DK running the LCIR stream firmware (one frame out, int8 outputs back, as unas/hil_stream.py),
and raises the alert of the lab's application: the board's class is a left lane change while the scenario's
fast car (car2, the left lane) is present less than 15 m behind (car2LongPosition between -15 and 0 m). The
alert is a beep and an icon where a blind-spot warning would be, held for 1.5 s. Every kept frame is logged
(features, outputs, class, alert, timing) to a JSON-lines file next to the recording.

Design and the technical check: local notes 2026-11-17 14:55. Python 3.8, numpy, pyserial, pygame.
"""
from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path

import numpy as np

from live_features import N_FEAT, LiveFeatures

BAUD = 921600
LEFT = 2                                                     # classes: 0 lane keeping, 1 right, 2 left lane change
CAR2_LONG, CAR2_PRESENT = 24, 30                             # raw channels (classification layout)
ALERT_BEHIND_M, ALERT_HOLD_S = 15.0, 1.5


class BoardLink:
    """The stream firmware's protocol: 0xA5, flags (bit 0 = first frame), seq (uint16), N_FEAT float32; reply
    0x5A, ran, seq, preprocessing cycles, network cycles, the int8 outputs."""

    def __init__(self, n_out: int = 3, in_len: int = 50):
        import serial
        import serial.tools.list_ports as lp

        ports = [p for p in lp.comports() if p.vid == 0x0483]
        if len(ports) != 1:
            raise RuntimeError(f"expected one ST-LINK virtual COM port, found {len(ports)}")
        self.port = serial.Serial(ports[0].device, BAUD, timeout=0.5)
        self.port.reset_input_buffer()
        self.n_out, self.in_len, self.seq, self.n = n_out, in_len, 0, 0

    def send(self, row: np.ndarray, first: bool):
        if first:
            self.n = 0
        pkt = struct.pack("<BBH", 0xA5, 1 if first else 0, self.seq & 0xFFFF) + np.asarray(row, "<f4").tobytes()
        t0 = time.perf_counter()
        self.port.write(pkt)
        rep = self.port.read(12 + self.n_out)
        rtt = time.perf_counter() - t0
        if len(rep) != 12 + self.n_out or rep[0] != 0x5A or struct.unpack_from("<H", rep, 2)[0] != (self.seq & 0xFFFF):
            raise RuntimeError(f"bad reply at frame {self.seq}: {rep!r}")
        self.seq += 1
        self.n += 1
        c_pre, c_inf = struct.unpack_from("<II", rep, 4)
        outs = np.frombuffer(rep[12:], np.int8).copy() if rep[1] == 1 else None
        return outs, c_pre, c_inf, rtt

    def close(self):
        self.port.close()


class LiveLoop:
    def __init__(self, carla_module, log_dir: Path, board: bool = True, alert: bool = True, force_lc_s: float = 0.0):
        """force_lc_s > 0 (tests with the autopilot only): the Traffic Manager is asked for a lane change
        every force_lc_s seconds of simulation, alternating left and right, so that the check covers them."""
        self.carla = carla_module
        self.force_lc_s, self.next_lc, self.lc_left = force_lc_s, force_lc_s, True
        self.lf = LiveFeatures(carla_module)
        self.board = BoardLink() if board else None
        self.alert_on = alert
        self.active = False
        self.first = True
        self.alert_until = 0.0
        self.alert_was = False
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"live_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.log = open(self.log_path, "w")
        self.log.write(json.dumps({"start": time.strftime("%Y-%m-%d %H:%M:%S"), "board": board, "alert": alert,
                                   "rule": f"class {LEFT} and car2 present less than {ALERT_BEHIND_M} m behind"}) + "\n")

    def on_recorder_start(self):
        self.active, self.first = True, True
        self.lf.n = 0
        self.t0 = None

    def _t0(self, t):
        if self.t0 is None:
            self.t0 = t

    def on_tick(self, world):
        if not self.active:
            return
        r = self.lf.step(world)
        if r is None:
            return
        row, extras = r
        self._t0(extras["sim_time"])
        if self.force_lc_s > 0 and extras["sim_time"] - self.t0 >= self.next_lc:
            ego = world.get_actors().find(self.lf.ego_id)
            tm = self.carla.Client("127.0.0.1", 2000).get_trafficmanager(8000)
            tm.auto_lane_change(ego, True)
            tm.random_left_lanechange_percentage(ego, 100.0 if self.lc_left else 0.0)
            tm.random_right_lanechange_percentage(ego, 0.0 if self.lc_left else 100.0)
            tm.force_lane_change(ego, self.lc_left)
            self.lc_left, self.next_lc = not self.lc_left, self.next_lc + self.force_lc_s
        rec = {"step": extras["step"], "t": round(extras["sim_time"], 4), "row": [float(x) for x in row],
               "vehicles": extras["vehicles"], "ego_slot": extras["ego_slot"], "car2_id": extras["car2_id"]}
        cls = None
        if self.board is not None:
            outs, c_pre, c_inf, rtt = self.board.send(row, self.first)
            rec.update(rtt_ms=round(rtt * 1e3, 3), cycles_pre=c_pre)
            if outs is not None:
                cls = int(outs.astype(np.int16).argmax())
                rec.update(outs=[int(x) for x in outs], cls=cls, cycles_net=c_inf)
        self.first = False
        close = row[CAR2_PRESENT] > 0.5 and -ALERT_BEHIND_M < row[CAR2_LONG] < 0.0
        alert = self.alert_on and cls == LEFT and close
        rec.update(car2_close=bool(close), alert=bool(alert))
        if alert:
            self.alert_until = time.time() + ALERT_HOLD_S
            if not self.alert_was:
                threading.Thread(target=beep, daemon=True).start()
        self.alert_was = alert
        self.log.write(json.dumps(rec) + "\n")

    def draw(self, surface):
        """The alert icon where a blind-spot warning would be: the upper left of the screen."""
        if surface is None or time.time() > self.alert_until:
            return
        import pygame

        w, h = surface.get_size()
        x, y, s = int(w * 0.06), int(h * 0.12), int(min(w, h) * 0.09)
        pygame.draw.polygon(surface, (230, 120, 0), [(x + s // 2, y), (x + s, y + s), (x, y + s)])
        pygame.draw.polygon(surface, (20, 20, 20), [(x + s // 2, y), (x + s, y + s), (x, y + s)], 4)
        pygame.draw.rect(surface, (20, 20, 20), (x + s // 2 - s // 20, y + s // 4, s // 10, s // 2))
        pygame.draw.circle(surface, (20, 20, 20), (x + s // 2, y + int(s * 0.86)), s // 16)

    def close(self):
        self.log.close()
        if self.board is not None:
            self.board.close()


def beep():
    try:
        import winsound
        winsound.Beep(1000, 400)
    except Exception:
        pass


def install(carla_module, pygame_module, log_dir: Path, mode: str) -> LiveLoop:
    """mode: board|noboard[,noalert][,lanechangesN]. Patches carla.World.tick (the hook runs after every step),
    carla.Client.start_recorder (the frame count starts with the recording, as the converter's) and
    pygame.display.flip (the alert icon is drawn last); lanechangesN (autopilot tests only) asks the Traffic
    Manager for a lane change every N s of simulation."""
    parts = set(mode.split(","))
    force = next((float(x[len("lanechanges"):] or 20) for x in parts if x.startswith("lanechanges")), 0.0)
    loop = LiveLoop(carla_module, log_dir, board="board" in parts, alert="noalert" not in parts, force_lc_s=force)
    tick, start_rec, flip = carla_module.World.tick, carla_module.Client.start_recorder, pygame_module.display.flip

    def tick_hook(self, *a, **kw):
        frame = tick(self, *a, **kw)
        loop.on_tick(self)
        return frame

    def rec_hook(self, *a, **kw):
        res = start_rec(self, *a, **kw)
        loop.on_recorder_start()
        print(f"[live] loop started with the recorder; log {loop.log_path.name}", flush=True)
        return res

    def flip_hook(*a, **kw):
        loop.draw(pygame_module.display.get_surface())
        return flip(*a, **kw)

    carla_module.World.tick, carla_module.Client.start_recorder, pygame_module.display.flip = tick_hook, rec_hook, flip_hook
    return loop
