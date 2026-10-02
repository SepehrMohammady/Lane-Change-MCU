"""Render the exiD exit replay (scenario_replay.py): animated GIF for the slide and a still.

Everything drawn is real: the target vehicle and the traffic around it follow the
raw 25 Hz exiD tracks, the road is the location's Lanelet2 map projected to the
recording's local frame, and the two traces are the same hand-designed CNN trained
on exiD and trained on highD (seed 0), run over the scenario's 26 evaluation
windows. The view is rotated so the vehicle travels left to right; the camera
follows it. The test-set figures in the text panel are five-seed means from
results/transfer.jsonl; the board latency is the measured one of that network.

Run in .venv:  python datasets/exid/render_replay.py
Outputs: $EXID_REPLAY_OUT (default Materials/T4.5/exid)/{exid_exit.gif, exid_exit_still.png}
"""
from __future__ import annotations

import json
import os
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
S = json.loads((HERE / "data" / "replay" / "scenario.json").read_text())
OUT = Path(os.environ.get("EXID_REPLAY_OUT", ROOT / "Materials" / "T4.5" / "exid"))
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN = "#4A66AC", "#C27C2E", "#3C7A4E"
INK, INK2, GRID = "#242852", "#5A5F73", "#E1E0D9"
LANE, SHOULDER, RAMP, BG = "#E4E4DF", "#D8D8D2", "#CFE3CA", "#F7F7F4"
CLASS = {0: "lane keep", 1: "RIGHT lane change", 2: "LEFT lane change"}


def api_ms(ds, name, board="STM32H7B3I-DK"):
    path = ROOT / "datasets" / ds / "results" / "deploy" / "benchmarks_api.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["model"] == name and r["board"] == board and not r.get("error"):
            return r["duration_ms"]
    raise SystemExit(f"no measurement of {name} on {board}")


def kind_acc(method):
    """Five-seed mean window accuracy on the test exits (results/transfer.jsonl)."""
    v = [r["metrics"]["by_group"]["exit to off-ramp"]["acc"]
         for r in map(json.loads, (HERE / "results" / "transfer.jsonl").read_text(encoding="utf-8").splitlines())
         if r["method"] == method and r["task"] == "cls" and r["eval_on"] == "exiD" and r.get("fraction", 1.0) == 1.0]
    assert len(v) == 5, (method, len(v))
    return st.fmean(v)


LAT_F32 = api_ms("highd", "highd_dscnn_cls_f32.tflite")       # the same graph runs either set of weights
LAT_I8 = api_ms("highd", "highd_dscnn_cls_int8_io.tflite")
ACC_EXID, ACC_HIGHD = kind_acc("scratch"), kind_acc("zero-shot")
scores = json.loads((HERE / "data" / "replay" / "scores.json").read_text())
MEAN_R = scores["mean_robust_s"]

FPS = S["fps_raw"]
f0, fc, f1 = S["frame_first"], S["frame_cross"], S["frame_end"]
tv = S["tv"]
TV_INDEX = {f: i for i, f in enumerate(tv["frame"])}
LABEL = S["label"]
ttlc_axis = [-t for t in S["ttlc_s"]]                 # window end times, -5.2 .. -0.2 s
M_E, M_H = S["models"]["exid"], S["models"]["highd"]
p_true = {k: [p[LABEL] for p in m["probs"]] for k, m in (("exid", M_E), ("highd", M_H))}

# rotation: the vehicle's overall direction of travel becomes +x, origin at the crossing
xy = np.column_stack([tv["x"], tv["y"]])
d = xy[-1] - xy[0]
TH = np.arctan2(d[1], d[0])
C, SN = np.cos(-TH), np.sin(-TH)
ORIGIN = xy[TV_INDEX[fc]]


def rot(x, y):
    x, y = np.asarray(x) - ORIGIN[0], np.asarray(y) - ORIGIN[1]
    return C * x - SN * y, SN * x + C * y


ROT_PATH = rot(xy[:, 0], xy[:, 1])
Y_LO, Y_HI = float(ROT_PATH[1].min()) - 5.0, float(ROT_PATH[1].max()) + 9.0
# the lanelet the vehicle enters at the crossing (taken 0.5 s after it, clear of the boundary):
# shaded with the other off-ramp lanelets and labelled as the lane to the off-ramp
from matplotlib.path import Path as MPath
_after = min(fc + FPS // 2, tv["frame"][-1])
_cross_xy = (tv["x"][TV_INDEX[_after]], tv["y"][TV_INDEX[_after]])
TARGET_LANE = next((lid for lid, lan in S["lanelets"].items()
                    if MPath(np.array(S["ways"][str(lan["left"])]["pts"] + S["ways"][str(lan["right"])]["pts"][::-1]))
                    .contains_point(_cross_xy)), None)
LANE_POLYS, LINES = [], []
for lid, lan in S["lanelets"].items():
    L = S["ways"][str(lan["left"])]["pts"]
    R = S["ways"][str(lan["right"])]["pts"]
    poly = np.array(L + R[::-1])
    px, py = rot(poly[:, 0], poly[:, 1])
    fill = SHOULDER if lan["subtype"] == "emergency_lane" else (
        RAMP if lan["offramp"] or lid == TARGET_LANE else LANE)
    LANE_POLYS.append((np.column_stack([px, py]), fill))
for wid, w in S["ways"].items():
    if w["type"] == "virtual":
        continue
    p = np.array(w["pts"])
    px, py = rot(p[:, 0], p[:, 1])
    if w["type"] == "road_border":
        style = dict(color="#6E7380", lw=1.6, ls="-")
    elif w["subtype"] == "dashed":
        style = dict(color="#8A8F99", lw=1.0, ls=(0, (6, 6)))
    else:
        style = dict(color="#8A8F99", lw=1.0, ls="-")
    LINES.append((px, py, style))


def box(x, y, heading_deg, length, width):
    """Corners of a vehicle in the rotated frame."""
    cx, cy = rot(x, y)
    h = np.deg2rad(heading_deg) - TH
    u, v = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    c = np.array([cx, cy])
    return np.array([c + u * length / 2 + v * width / 2, c - u * length / 2 + v * width / 2,
                     c - u * length / 2 - v * width / 2, c + u * length / 2 - v * width / 2])


def others_at(frame):
    out = []
    for o in S["others"].values():
        if frame in o["frame"]:
            i = o["frame"].index(frame)
            out.append(box(o["x"][i], o["y"][i], o["heading"][i], o["l"][i], o["w"][i]))
    return out


t_rel = lambda f: (f - fc) / FPS


def says(m, k, t):
    if k is None:
        return "no complete 2-s window yet"
    if t > 0:
        return "crossing done"
    c = m["pred"][k]
    return f"{CLASS[c]}  (p = {m['probs'][k][c]:.2f})"


def draw(frame, ax_road, ax_tl, ax_txt):
    i = TV_INDEX[frame]
    t = t_rel(frame)
    cx, _ = rot(tv["x"][i], tv["y"][i])
    # ---- road
    ax_road.clear()
    ax_road.set_facecolor(BG)
    for poly, fill in LANE_POLYS:
        ax_road.add_patch(Polygon(poly, closed=True, fc=fill, ec="none", zorder=1))
    for px, py, style in LINES:
        ax_road.plot(px, py, zorder=2, **style)
    for corners in others_at(frame):
        ax_road.add_patch(Polygon(corners, closed=True, fc="#C9CDD6", ec="#7A7F8C", lw=0.8, zorder=3))
    ax_road.add_patch(Polygon(box(tv["x"][i], tv["y"][i], tv["heading"][i], tv["l"], tv["w"]), closed=True,
                              fc=BLUE, ec="#1F2D5C", lw=1.0, zorder=4))
    ax_road.set_xlim(cx - 55, cx + 55)
    ax_road.set_ylim(Y_LO, Y_HI)
    if TARGET_LANE is not None:
        lan = S["lanelets"][TARGET_LANE]
        Rr = np.array(S["ways"][str(lan["right"])]["pts"])
        rx, ry = rot(Rr[:, 0], Rr[:, 1])
        vis = (rx > cx - 50) & (rx < cx + 50) & (ry > Y_LO) & (ry < Y_HI)
        if vis.any():
            j = np.nonzero(vis)[0][np.argmax(rx[vis])]
            ax_road.text(rx[j] - 1, ry[j] + 0.6, "lane to the off-ramp", fontsize=9, color=GREEN,
                         ha="right", va="bottom", zorder=5)
    ax_road.set_xticks([]); ax_road.set_yticks([])
    for s in ax_road.spines.values():
        s.set_color(GRID)
    ax_road.set_title(f"exiD location {S['location_id']} ({S['city']}), recording {S['recording']}, vehicle {S['track_id']}: "
                      f"exit to the off-ramp at {S['speed_at_cross_mps'] * 3.6:.0f} km/h    "
                      f"t = {t:+.1f} s to lane crossing", loc="left", fontsize=11, color=INK, fontweight="bold")
    # ---- timeline
    ax_tl.clear()
    ax_tl.set_xlim(t_rel(f0), t_rel(f1)); ax_tl.set_ylim(-0.05, 1.12)
    ax_tl.axvspan(t_rel(f0), min(t, t_rel(f1)), color="#EEF1F8", lw=0)
    n_vis = sum(1 for te in ttlc_axis if te <= t)
    ax_tl.plot(ttlc_axis[:n_vis], p_true["exid"][:n_vis], color=GREEN, lw=2.2, label="CNN trained on exiD")
    ax_tl.plot(ttlc_axis[:n_vis], p_true["highd"][:n_vis], color=ORANGE, lw=2.0, ls=(0, (5, 3)),
               label="same CNN trained on highD, as is")
    ax_tl.axhline(0.5, color=GRID, lw=0.8)
    ax_tl.axvline(0, color=INK, lw=1)
    ax_tl.text(0.05, 1.04, "lane crossing", fontsize=8.5, color=INK)
    flag = -M_E["robust_s"] if M_E["robust_s"] else None
    if flag is not None and t >= flag:
        ax_tl.axvline(flag, color=GREEN, lw=1.4)
        ax_tl.text(flag + 0.05, 0.08, f"trained on exiD: exit\npredicted from {flag:+.1f} s", fontsize=8.5, color=GREEN)
    ax_tl.axvline(t, color=INK, lw=0.8, ls=":")
    ax_tl.set_xlabel("seconds relative to the lane crossing", fontsize=9, color=INK)
    ax_tl.set_ylabel("P(right lane change)", fontsize=9, color=INK)
    ax_tl.tick_params(labelsize=8, colors=INK, length=0)
    ax_tl.grid(True, axis="y", color=GRID, lw=0.6); ax_tl.set_axisbelow(True)
    for s in ("top", "right"):
        ax_tl.spines[s].set_visible(False)
    ax_tl.legend(loc="upper left", fontsize=8, frameon=False)
    # ---- status text
    ax_txt.clear(); ax_txt.axis("off")
    k = max(j for j, te in enumerate(ttlc_axis) if te <= t) if t >= ttlc_axis[0] else None
    r_h = M_H["robust_s"]
    wrong_tail = next((n for n in range(len(M_H["pred"])) if M_H["pred"][-1 - n] == LABEL), len(M_H["pred"]))
    lines = [
        ("Same network, two training sets", INK, 11, True),
        ("", INK, 4, False),
        ("hand-designed CNN (8.4k), trained on exiD", GREEN, 9.5, True),
        (f"   says: {says(M_E, k, t)}", INK, 9.5, False),
        (f"   exit predicted from {M_E['robust_s']:.1f} s before the crossing", INK2, 9, False),
        ("", INK, 4, False),
        ("the same CNN trained on highD, applied as is", ORANGE, 9.5, True),
        (f"   says: {says(M_H, k, t)}", INK, 9.5, False),
        (f"   exit predicted from {r_h:.1f} s" if r_h else
         f"   no lasting prediction: wrong again from {-0.2 * wrong_tail:+.1f} s", INK2, 9, False),
        ("", INK, 4, False),
        ("All 186 test exits", INK, 9.5, True),
        (f"   windows correct: {ACC_EXID * 100:.1f}% (exiD) vs {ACC_HIGHD * 100:.1f}% (highD), five seeds", INK, 9, False),
        (f"   prediction horizon, seed 0: {MEAN_R['exid']:.2f} s vs {MEAN_R['highd_as_is']:.2f} s", INK, 9, False),
        ("", INK, 4, False),
        ("Same graph, so the same board cost", INK, 9.5, True),
        (f"   STM32H7B3I-DK: {LAT_F32:.3f} ms float32, {LAT_I8:.3f} ms int8", INK, 9, False),
    ]
    yy = 0.98
    for text, col, size, bold in lines:
        ax_txt.text(0.0, yy, text, fontsize=size, color=col, fontweight="bold" if bold else "normal",
                    va="top", ha="left", transform=ax_txt.transAxes)
        yy -= 0.055 if size >= 9 else 0.025


def render_frame(frame):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#FAFAFA")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], width_ratios=[1.55, 1], hspace=0.22, wspace=0.08,
                          left=0.05, right=0.98, top=0.93, bottom=0.09)
    ax_road = fig.add_subplot(gs[0, :])
    ax_tl = fig.add_subplot(gs[1, 0]); ax_txt = fig.add_subplot(gs[1, 1])
    draw(frame, ax_road, ax_tl, ax_txt)
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    plt.close(fig)
    return img.convert("P", palette=Image.ADAPTIVE, colors=128)


frames = [f for f in range(f0, f1 + 1, 2) if f in TV_INDEX]
imgs = [render_frame(f) for f in frames]
hold = [imgs[-1]] * 12
imgs[0].save(OUT / "exid_exit.gif", save_all=True, append_images=imgs[1:] + hold, duration=80, loop=0, optimize=True)
key = min(frames, key=lambda f: abs(t_rel(f) + 0.4))     # just before the crossing: both answers visible
render_frame(key).convert("RGB").save(OUT / "exid_exit_still.png")
print(f"gif: {len(imgs)} frames -> {OUT / 'exid_exit.gif'} ({(OUT / 'exid_exit.gif').stat().st_size / 1024:.0f} KB)")
print(f"still at t={t_rel(key):+.1f} s -> {OUT / 'exid_exit_still.png'}")
