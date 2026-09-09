"""Render the DMIR "race": five cars on one real lane change, one per model.

Every row replays the same recorded manoeuvre of a held-out test driver (raw
positions from the H5 session, see scenario_race.py). What differs per row is
the model: the moment it first says "lane change" on the shared window stream
(its real output, exactly as the pipeline would compute it) and how much road
passes before that answer is available (board-farm latency x recorded speed).

The recorded scene is the dataset's own use case. The driver closes on a slower
car, lifts off, signals left, and a faster car is coming up in the target lane
(the CARLA scenario's dedicated "left car", object slot 11). The recorded driver
waited for that car and only steered after it had passed, so the recording
contains no conflict and none is claimed.

The what-if overlay adds one counterfactual driver, the same in every row: a
driver who does NOT check the mirror. He commits when the recorded indicator
comes on and moves across at the recording's own maximum lateral rate. On that
path he reaches the recorded car at T_CONFLICT. Each row then differs only in
when its warning reaches him: he stops if warning + TAU < T_CONFLICT, and
otherwise keeps going into the car. TAU is an assumption, printed on the
picture with the range Green (2000) reports for perception-reaction time
(0.70-0.75 s expected signal, 1.25 s unexpected but common, 1.5 s surprise).
The outcome is sensitive to it, which the picture says and the HTML slider
shows: at 0.75 s and at 1.0 s the Transformer-size row is the only one too
late, at 1.25 s only the searched float model still stops, at 1.5 s none do.

Also not measurements, and labelled: row 5 is a Transformer of the reference
size, since no classifier Transformer exists for DMIR, so its latency is the
DMIR regression reference Transformer on the same board and it is assumed to
flag with the reference CNN. The model outputs come from the recorded run; the
counterfactual changes what the driver does after the warning, not the inputs
the models saw.

Run in .venv:  python datasets/dmir/render_race.py
Outputs: Materials/T4.5/race/{race.gif, race_whatif.gif, race_still*.png, race.html}
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RACE = HERE / "data" / "race"
OUT = ROOT / "Materials" / "T4.5" / "race"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, NEUTRAL = "#4A66AC", "#C27C2E", "#9AA0A6"
INK, INK2, GRID, ROAD = "#242852", "#5A5F73", "#E1E0D9", "#F4F4F1"
GREEN = "#3C7A4E"
LANE_W = 3.75

S = json.loads((RACE / "scenario.json").read_text())
# start once the car in the target lane is inside the drawn view, so the first frame
# (the static preview of the GIF in a printed deck) already shows the situation
FR = [f for f in S["frames"] if f["t"] >= -4.6]
L, WID = S["ego"]["length"], S["ego"]["width"]
DIRECTION = S["direction"]
T_W = S["t_windows"]
CLASS = {0: "no intention", 1: "RIGHT lane change", 2: "LEFT lane change"}
HZ = S["hazard"]
STEER = HZ["steer_start"]

TAU = 1.0                   # assumed driver perception-reaction time, stated on the picture
BAND = (0.75, 1.25)         # Green (2000): expected signal .. unexpected but common signal

# board-farm latencies, STM32H7B3I-DK / NUCLEO-F401RE, ST Edge AI 4.0.1 balanced
# (datasets/dmir/docs/deployment.md). The QAT row is measured as two builds: the
# int8-I/O build on the H7B3 (1.435 ms, the build whose outputs are replayed here)
# and the float32-I/O build on the F401 (7.381 ms); same weights, same test accuracy.
ROWS = [
    ("ref",  "Reference CNN",              "441k params, float32",         33.52,  NEUTRAL, "does not fit"),
    ("best", "Searched, best accuracy",    "84k params, float32",          3.628,  BLUE,    "18.35 ms"),
    ("qat",  "Searched, QAT int8",         "84k params, int8 in and out",  1.435,  BLUE,    "7.381 ms*"),
    ("tiny", "Searched, smallest",         "8k params, float32",           0.7931, BLUE,    "4.376 ms"),
    ("tr",   "Transformer, reference size", "333k params, float32",        368.82, ORANGE,  "does not fit"),
]
FLAG = {k: S["flags"][k] for k in ("ref", "best", "qat", "tiny")}
FLAG["tr"] = FLAG["ref"]                  # assumption, stated on the picture
IND_ON = next(f["t"] for f in FR if f["ind"] != 0)


def at(t):
    return min(FR, key=lambda f: abs(f["t"] - t))


def output_at(key, t):
    """(class, p) of the row's model for the last window that ends at or before t."""
    if key == "tr":
        return None
    idx = [i for i, tw in enumerate(T_W) if tw <= t + 1e-6]
    if not idx:
        return None
    p = S["probs"][key][idx[-1]]
    c = max(range(3), key=lambda i: p[i])
    return c, p[c]


def left_car(f):
    return next((o for o in f["objs"] if o["slot"] == HZ["slot"]), None)


# ---------------------------------------------------------------- counterfactual driver
_ts = [f["t"] for f in FR]
_ys = [f["y"] for f in FR]
LAT_RATE = max(abs(_ys[i + 1] - _ys[i]) / (_ts[i + 1] - _ts[i]) for i in range(len(_ts) - 1))
T_COMMIT = IND_ON                                   # recorded indicator onset
Y_COMMIT = at(T_COMMIT)["y"]
Y_TARGET = 1.5 * LANE_W                             # centre of the target lane


def y_ghost(t):
    """Lateral position of the driver who does not check the mirror: he commits when
    the indicator goes on and crosses at the recording's own maximum lateral rate."""
    return min(Y_COMMIT + LAT_RATE * max(0.0, t - T_COMMIT), Y_TARGET)


def _conflict():
    for f in FR:
        o = left_car(f)
        if o is None:
            continue
        if abs(o["long"]) < (L + o["len"]) / 2 and abs(y_ghost(f["t"]) - (f["y"] + o["lat"])) < (WID + o["wid"]) / 2:
            return f["t"]
    return None


T_CONFLICT = _conflict()


def row_geometry(key, lat_ms):
    """Where the question is asked, where the answer arrives, and the warning margin:
    how long before the counterfactual conflict the model's answer is available."""
    tf = FLAG[key]
    f = at(tf)
    q = f["x"] + L / 2                        # front bumper at the flag window
    v = f["v"]
    t_ans = tf + lat_ms / 1000
    return {"t_flag": tf, "v": v, "q": q, "ans": q + v * lat_ms / 1000, "t_ans": t_ans,
            "dist": v * lat_ms / 1000, "margin": T_CONFLICT - t_ans}


GEO = {k: row_geometry(k, ms) for k, _, _, ms, _, _ in ROWS}


def fmt(m):
    return f"{m:.1f} m" if m >= 1 else (f"{m*100:.0f} cm" if m >= 0.01 else f"{m*1000:.0f} mm")


def burst(ax, x, y, r, color):
    pts = [(x + (r if i % 2 == 0 else r * 0.45) * math.cos(i * math.pi / 8),
            y + (r if i % 2 == 0 else r * 0.45) * math.sin(i * math.pi / 8)) for i in range(16)]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=7))


def draw_row(ax, row, t, whatif, tau=TAU, first=False):
    key, title, sub, lat_ms, col, f401 = row
    g = GEO[key]
    f = at(t)
    xe = f["x"]
    stops = whatif and g["margin"] > tau
    t_stop = g["t_ans"] + tau
    crashed = whatif and not stops and t >= T_CONFLICT
    if whatif:
        ye = y_ghost(min(t, t_stop) if stops else min(t, T_CONFLICT))
    else:
        ye = f["y"]
    ax.clear()
    ax.set_facecolor(ROAD)
    ax.set_xlim(xe - 45, xe + 45)
    ax.set_ylim(-0.5, 2 * LANE_W + 1.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    for yv, solid in ((0, True), (LANE_W, False), (2 * LANE_W, True)):
        ax.axhline(yv, color="#7A7F8C", lw=1.4 if solid else 0.9, ls="-" if solid else (0, (6, 6)))
    for o in f["objs"]:
        hz = whatif and o["slot"] == HZ["slot"]
        ax.add_patch(Rectangle((xe + o["long"] - o["len"] / 2, f["y"] + o["lat"] - o["wid"] / 2), o["len"], o["wid"],
                               fc="#C9CDD6", ec=ORANGE if hz else "#7A7F8C", lw=1.6 if hz else 0.7, zorder=4))
        if hz and first and xe - 44 < xe + o["long"] < xe + 44 and t < g["t_ans"]:
            ax.annotate("car coming up in the target lane (recorded)",
                        xy=(xe + o["long"], f["y"] + o["lat"] + o["wid"] / 2), xytext=(xe - 44, 2 * LANE_W + 1.05),
                        fontsize=7.5, color=ORANGE, va="center",
                        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.7, shrinkA=0, shrinkB=2))
    ax.add_patch(Rectangle((xe - L / 2, ye - WID / 2), L, WID, fc=col, ec=INK, lw=0.8,
                           hatch="////" if crashed else None, zorder=5))
    on = t >= g["t_ans"] and not crashed
    ax.add_patch(Circle((xe + L * 0.15, ye), 0.42, fc="#FFD34D" if on else "#DDDDDD", ec=INK, lw=0.5, zorder=6))
    if t >= g["t_flag"] and g["q"] > xe - 44:   # road covered while this model computes one answer
        ax.plot([g["q"], g["q"]], [0.05, 0.5], color=INK, lw=1.0)
        ax.add_patch(Rectangle((g["q"], 0.12), max(g["ans"] - g["q"], 0.2), 0.3, fc=col, ec="none", zorder=3))
        ax.text(min(g["ans"] + 0.8, xe + 43), 0.27, f"answer after {fmt(g['dist'])}", va="center", ha="left",
                fontsize=7.5, color=INK2, clip_on=True)
    if whatif and t >= T_COMMIT:
        note, ncol = None, ORANGE
        if crashed:
            burst(ax, xe + L / 2, ye + WID / 2 + 0.2, 1.6, ORANGE)
            note = "warned too late: this driver is already across"
        elif t >= t_stop and stops:
            note, ncol = f"warned {g['margin']:.2f} s before the conflict: stopped short", GREEN
        elif t >= g["t_ans"]:
            note = f"warning available, driver needs {tau:.2f} s"
        if note:
            ax.text(xe - 44, 2 * LANE_W + 1.05, note, va="center", fontsize=7.5, color=ncol,
                    fontweight="bold" if crashed else None, clip_on=True)


def status_lines(row, t, whatif=False, tau=TAU):
    key, title, sub, lat_ms, col, f401 = row
    g = GEO[key]
    lines = [(title, col if key != "ref" else INK, 10, True), (sub, INK2, 8, False),
             (f"H7B3I-DK {lat_ms:.5g} ms   F401RE {f401}", INK2, 8, False)]
    if key == "tr":
        if t < g["t_flag"]:
            lines.append(("no output yet", INK2, 8.5, False))
        else:
            lines.append((f"assumed to flag with the reference CNN at {g['t_flag']:+.1f} s", INK2, 8, False))
            lines.append((f"answer {lat_ms/1000:.3f} s later = {fmt(g['dist'])} of road", ORANGE, 8.5, True))
        if not whatif:
            lines.append(("latency: DMIR regression reference, same board;", INK2, 7, False))
            lines.append(("no classifier Transformer exists for this task", INK2, 7, False))
    else:
        o = output_at(key, t)
        if t >= 0:
            lines.append(("crossing done", INK2, 8.5, False))
        elif o is None:
            lines.append(("no complete 5-s window yet", INK2, 8.5, False))
        else:
            c, p = o
            lines.append((f"says: {CLASS[c]}  (p = {p:.2f})", col if c == DIRECTION else INK, 8.5, c == DIRECTION))
        if t >= g["t_flag"]:
            lines.append((f"flag from {g['t_flag']:+.1f} s, answer after {fmt(g['dist'])}", INK2, 8, False))
    if whatif and t >= g["t_ans"]:
        ok = g["margin"] > tau
        lines.append((f"margin {g['margin']:.2f} s vs driver {tau:.2f} s: {'stops in time' if ok else 'TOO LATE'}",
                      GREEN if ok else ORANGE, 8.5, True))
    return lines


def margin_bar(fig, y0, h, row, t, tau=TAU):
    """Warning margin of this row against the assumed reaction time and Green's band."""
    key, _, _, _, col, _ = row
    g = GEO[key]
    if t < g["t_ans"]:
        return
    ax = fig.add_axes([0.762, y0 + 0.014, 0.185, 0.036])
    ax.set_xlim(0, 1.8); ax.set_ylim(0, 1); ax.axis("off")
    ax.axvspan(BAND[0], BAND[1], ymin=0.30, ymax=0.78, color="#E8E8E4", lw=0)
    ax.barh(0.54, min(max(g["margin"], 0.02), 1.8), height=0.34, color=col if g["margin"] > tau else ORANGE)
    ax.plot([tau, tau], [0.24, 0.92], color=INK, lw=1.1)
    ax.plot([0, 1.8], [0.3, 0.3], color=INK2, lw=0.6)
    for xv in (0.5, 1.0, 1.5):
        ax.plot([xv, xv], [0.22, 0.3], color=INK2, lw=0.6)
    ax.text(0.0, 0.16, "0", fontsize=5.6, color=INK2, ha="center", va="top")
    ax.text(1.0, 0.16, "1 s", fontsize=5.6, color=INK2, ha="center", va="top")
    ax.text(1.8, 0.16, "margin to the conflict", fontsize=6.0, color=INK2, va="top", ha="right")


def frame_figure(t, whatif, tau=TAU):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    v_flag = GEO["best"]["v"] * 3.6
    if whatif:
        head = f"DMIR, test driver 2: the same recorded lane change   |   what-if: a driver who does not check the mirror"
        sub = (f"Recorded: he signalled at {IND_ON:+.1f} s, waited for the car in the target lane and only steered at "
               f"{STEER:+.1f} s, after it had passed. No conflict happened.   Below: a driver who does not wait, and what "
               "each model's warning buys him.")
    else:
        head = (f"DMIR, test driver 2: one real {S['label_name']}, five cars on the same inputs")
        sub = (f"Same recorded vehicle in every row ({v_flag:.0f} km/h at the moment the models flag, {S['speed_at_cross_mps']*3.6:.0f} km/h "
               "at the crossing); only the model reading the windows changes.")
    fig.text(0.012, 0.975, head, fontsize=12, color=INK, fontweight="bold", va="center")
    fig.text(0.985, 0.975, f"t = {t:+.1f} s to lane crossing", fontsize=11.5, color=INK, fontweight="bold",
             va="center", ha="right")
    fig.text(0.012, 0.945, sub, fontsize=7.8, color=INK2, va="center")
    top, h = 0.930, 0.157
    for i, row in enumerate(ROWS):
        y0 = top - (i + 1) * h
        ax = fig.add_axes([0.205, y0 + 0.016, 0.545, h - 0.026])
        draw_row(ax, row, t, whatif, tau, first=(i == 0))
        st = status_lines(row, t, whatif, tau)
        yy = y0 + h - 0.028
        for txt, colr, size, bold in st[:3]:
            fig.text(0.012, yy, txt, fontsize=size, color=colr, fontweight="bold" if bold else None, va="top")
            yy -= 0.028 if size >= 10 else 0.022
        yy = y0 + h - 0.028
        for txt, colr, size, bold in st[3:]:
            if whatif and yy < y0 + 0.074:
                break
            fig.text(0.762, yy, txt, fontsize=size, color=colr, fontweight="bold" if bold else None, va="top")
            yy -= 0.0225
        if whatif:
            margin_bar(fig, y0, h, row, t, tau)
    axt = fig.add_axes([0.205, 0.090, 0.545, 0.058])
    axt.set_xlim(FR[0]["t"], FR[-1]["t"]); axt.set_ylim(0, 1); axt.axis("off")
    axt.plot([FR[0]["t"], FR[-1]["t"]], [0.38, 0.38], color=GRID, lw=2)
    axt.plot([FR[0]["t"], t], [0.38, 0.38], color=INK, lw=2)
    f_lo, f_hi = min(FLAG.values()), max(FLAG.values())
    for tv in (IND_ON, f_lo, f_hi, 0) + ((T_CONFLICT,) if whatif else ()):
        axt.plot([tv, tv], [0.30, 0.46], color=INK2, lw=0.9)
    axt.text(IND_ON - 0.10, 0.92, "indicator on", fontsize=7, color=INK2, ha="right", va="center")
    axt.text((f_lo + f_hi) / 2, 0.62, f"models flag {f_lo:+.1f} to {f_hi:+.1f} s", fontsize=7, color=BLUE,
             ha="center", va="center")
    axt.text(0, 0.92, "lane crossing", fontsize=7, color=INK, ha="center", va="center")
    if whatif:
        axt.text(T_CONFLICT, 0.92, "conflict", fontsize=7, color=ORANGE, ha="center", va="center")
    axt.plot([t], [0.38], marker="o", color=INK, ms=5)
    for tv in range(int(FR[0]["t"]), int(FR[-1]["t"]) + 1):
        axt.text(tv, 0.12, f"{tv:+d}", fontsize=6.5, color=INK2, ha="center", va="center")
    if whatif:
        late = {tv: sum(1 for k, *_ in ROWS if GEO[k]["margin"] <= tv) for tv in (0.75, 1.25, 1.5)}
        foot = (f"Recorded: every vehicle, the timing, the driver's wait. Counterfactual: one driver, the same in all five rows, who commits "
                f"when the indicator goes on and crosses at the\nrecording's own top lateral rate ({LAT_RATE:.2f} m/s), reaching the recorded "
                f"car at {T_CONFLICT:+.1f} s; he stops if his warning arrives more than {tau:.2f} s before that. Model outputs are from the "
                f"recorded run.\nThe reaction time is an assumption and the outcome turns on it (Green 2000: {BAND[0]:.2f} s expected signal, "
                f"{BAND[1]:.2f} s unexpected, 1.5 s surprise): {late[0.75]} of 5 rows are too late at 0.75 s, {late[1.25]} at 1.25 s, "
                f"{late[1.5]} at 1.5 s.\nRow 5 latency: DMIR regression reference Transformer, assumed to flag with the reference CNN. "
                "*QAT F401RE figure is the float-I/O build of the same weights.")
    else:
        foot = ("Every row: same recorded vehicle, same windows, real model outputs; latencies measured on the board (ST Edge AI 4.0.1, "
                "STM32H7B3I-DK and NUCLEO-F401RE).\nRow 5 latency: DMIR regression reference Transformer, assumed to flag with the reference "
                "CNN, since no classifier Transformer exists for DMIR.\n*QAT F401RE figure is the float-I/O build of the same weights.")
    fig.text(0.012, 0.004, foot, fontsize=6.4, color=INK2, va="bottom", linespacing=1.4)
    return fig


def render_gif(path, whatif, fps=10):
    frames = []
    for f in FR:
        fig = frame_figure(f["t"], whatif)
        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
        plt.close(fig)
    dur = [100] * len(frames)
    dur[-1] = 2000
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
    print(f"gif: {len(frames)} frames, {sum(dur)/1000:.1f} s -> {path} ({path.stat().st_size // 1024} KB)")


def render_html(path):
    D = {"frames": FR, "L": L, "W": WID, "laneW": LANE_W, "direction": DIRECTION, "t_w": T_W,
         "probs": {k: S["probs"][k] for k in ("ref", "best", "qat", "tiny")},
         "rows": [{"key": k, "title": t, "sub": s, "lat": ms, "col": c, "f401": f, **GEO[k]} for k, t, s, ms, c, f in ROWS],
         "ind_on": IND_ON, "label": S["label_name"], "kmh": round(S["speed_at_cross_mps"] * 3.6),
         "class": CLASS, "hz_slot": HZ["slot"], "tau": TAU, "band": BAND, "steer": STEER,
         "commit": T_COMMIT, "rate": LAT_RATE, "yCommit": Y_COMMIT, "yTarget": Y_TARGET, "conflict": T_CONFLICT}
    html = r"""<!doctype html><html><head><meta charset="utf-8"><title>DMIR race: one lane change, five models</title>
<style>body{font-family:Raleway,Calibri,Arial,sans-serif;color:#242852;margin:16px;background:#fff}
h1{font-size:18px;margin:0 0 4px} p.sub{font-size:13px;color:#5A5F73;margin:0 0 8px}
.ctl{margin:8px 0;font-size:13px} .ctl label{margin-right:14px}
canvas{border:1px solid #E1E0D9} input[type=range]{vertical-align:middle} #sl{width:460px} #tau{width:150px} button{font-size:13px;padding:3px 10px}
p.note{font-size:12px;color:#5A5F73;max-width:1140px}</style></head><body>
<h1>DMIR, test driver 2: one real __LABEL__, five cars on the same inputs</h1>
<p class="sub">Recorded: the driver signalled at __IND__ s, waited for the car coming up in the target lane and only steered at __STEER__ s,
after it had passed. No conflict happened in the recording.</p>
<div class="ctl"><button id="play">&#9654; Play</button> <input type="range" id="sl" min="0" max="__NF__" value="0">
<span id="tl"></span> &nbsp; <label><input type="checkbox" id="whatif"> what-if: a driver who does not check the mirror</label>
driver reaction <input type="range" id="tau" min="0.5" max="2.0" step="0.05" value="__TAU__"> <span id="tauv"></span> s &nbsp;
<label><input type="checkbox" id="showTr" checked> Transformer row</label></div>
<canvas id="c" width="1140" height="700"></canvas>
<p class="note">Every row replays the same recorded vehicle on the same 5-s windows; the "says" line is the model's real output for the
last complete window. The bar on the road is latency (STM32H7B3I-DK, ST Edge AI 4.0.1) times the recorded speed. Row 5: no classifier
Transformer exists for DMIR, so its latency is the regression reference Transformer measured on the same board, and it is assumed to
flag at the same window as the reference CNN. The QAT row's F401RE figure is the float-I/O build of the same weights.
<br>What-if: one counterfactual driver, the same in every row, who commits when the indicator goes on and crosses at the recording's own
top lateral rate, reaching the recorded car at the conflict tick; each row stops him if its warning arrives more than the chosen
reaction time before that. Green (2000) reports 0.70-0.75 s for an expected signal, 1.25 s for an unexpected but common one and 1.5 s
for a surprise; move the slider to see the outcome turn on it. Model outputs are from the recorded run.</p>
<script>
const D = __DATA__;
const BLUE="#4A66AC",ORANGE="#C27C2E",NEUT="#9AA0A6",INK="#242852",INK2="#5A5F73",GRID="#E1E0D9",GREEN="#3C7A4E";
const cv=document.getElementById("c"),g=cv.getContext("2d"),sl=document.getElementById("sl"),tl=document.getElementById("tl"),tauEl=document.getElementById("tau"),tauv=document.getElementById("tauv");
const F=D.frames,L=D.L,WD=D.W,LW=D.laneW; let playing=false,timer=null;
const fmt=m=>m>=1?`${m.toFixed(1)} m`:(m>=0.01?`${(m*100).toFixed(0)} cm`:`${(m*1000).toFixed(0)} mm`);
const at=t=>F.reduce((a,b)=>Math.abs(b.t-t)<Math.abs(a.t-t)?b:a);
const yGhost=t=>Math.min(D.yCommit+D.rate*Math.max(0,t-D.commit),D.yTarget);
function outAt(key,t){let k=-1;for(let i=0;i<D.t_w.length;i++)if(D.t_w[i]<=t+1e-6)k=i;if(k<0)return null;const p=D.probs[key][k];let c=0;for(let i=1;i<3;i++)if(p[i]>p[c])c=i;return [c,p[c]];}
function draw(idx){
  const f=F[idx],t=f.t; g.clearRect(0,0,cv.width,cv.height);
  const whatif=document.getElementById("whatif").checked, showTr=document.getElementById("showTr").checked, tau=+tauEl.value; tauv.textContent=tau.toFixed(2);
  const rows=D.rows.filter(r=>showTr||r.key!=="tr"); const rowH=Math.floor(640/rows.length);
  g.fillStyle=INK;g.font="bold 15px Calibri";
  g.fillText(`t = ${t>=0?"+":""}${t.toFixed(1)} s to lane crossing`+(whatif?"   |   what-if: a driver who does not check the mirror":""),12,20);
  rows.forEach((r,i)=>{
    const y0=30+i*rowH, rp={x:230,y:y0+6,w:560,h:rowH-26}; const sx=rp.w/90, sy=rp.h/(2*LW+2);
    const stops=whatif&&r.margin>tau, tStop=r.t_ans+tau, crashed=whatif&&!stops&&t>=D.conflict;
    const ye=whatif?yGhost(stops?Math.min(t,tStop):Math.min(t,D.conflict)):f.y;
    const px=x=>rp.x+(x-(f.x-45))*sx, py=y=>rp.y+rp.h-(y+0.5)*sy;
    g.save();g.beginPath();g.rect(rp.x-1,rp.y-1,rp.w+2,rp.h+2);g.clip();
    g.fillStyle="#F4F4F1";g.fillRect(rp.x,rp.y,rp.w,rp.h);
    [[0,1],[LW,0],[2*LW,1]].forEach(([yv,solid])=>{g.strokeStyle="#7A7F8C";g.lineWidth=solid?1.4:0.9;g.setLineDash(solid?[]:[6,6]);g.beginPath();g.moveTo(rp.x,py(yv));g.lineTo(rp.x+rp.w,py(yv));g.stroke();g.setLineDash([]);});
    f.objs.forEach(o=>{const hz=whatif&&o.slot===D.hz_slot;g.fillStyle="#C9CDD6";g.strokeStyle=hz?ORANGE:"#7A7F8C";g.lineWidth=hz?1.6:0.7;const X=px(f.x+o.long-o.len/2),Y=py(f.y+o.lat+o.wid/2);g.fillRect(X,Y,o.len*sx,o.wid*sy);g.strokeRect(X,Y,o.len*sx,o.wid*sy);
      if(hz&&i===0&&t<r.t_ans){g.strokeStyle=ORANGE;g.lineWidth=0.7;g.beginPath();g.moveTo(px(f.x+o.long),Y);g.lineTo(rp.x+250,rp.y+12);g.stroke();g.fillStyle=ORANGE;g.font="11px Calibri";g.fillText("car coming up in the target lane (recorded)",rp.x+6,rp.y+14);}});
    g.fillStyle=r.col;g.strokeStyle=INK;g.lineWidth=0.8;const EX=px(f.x-L/2),EY=py(ye+WD/2);g.fillRect(EX,EY,L*sx,WD*sy);g.strokeRect(EX,EY,L*sx,WD*sy);
    if(crashed){g.strokeStyle="#fff";g.lineWidth=1;for(let k=0;k<6;k++){g.beginPath();g.moveTo(EX+k*L*sx/6,EY+WD*sy);g.lineTo(EX+(k+1)*L*sx/6,EY);g.stroke();}}
    const on=t>=r.t_ans&&!crashed; g.fillStyle=on?"#FFD34D":"#DDDDDD";g.beginPath();g.arc(px(f.x+L*0.15),py(ye),4,0,6.283);g.fill();g.strokeStyle=INK;g.lineWidth=0.5;g.stroke();
    g.font="11px Calibri";
    if(t>=r.t_flag){
      g.strokeStyle=INK;g.lineWidth=1;g.beginPath();g.moveTo(px(r.q),py(0.05));g.lineTo(px(r.q),py(0.5));g.stroke();
      g.fillStyle=r.col;g.fillRect(px(r.q),py(0.42),Math.max((r.ans-r.q)*sx,2),0.3*sy);
      g.fillStyle=INK2;g.fillText(`answer after ${fmt(r.dist)}`,Math.min(px(r.ans)+5,rp.x+rp.w-120),py(0.27));
    }
    if(whatif&&t>=D.commit){
      let note=null,nc=ORANGE;
      if(crashed){const bx=px(f.x+L/2),by=py(ye+WD/2+0.2);g.fillStyle=ORANGE;g.beginPath();for(let k=0;k<16;k++){const rr=(k%2?0.45:1)*1.6;const a=k*Math.PI/8;g.lineTo(bx+rr*sx*Math.cos(a),by-rr*sy*Math.sin(a));}g.closePath();g.fill();note="warned too late: this driver is already across";}
      else if(t>=tStop&&stops){note=`warned ${r.margin.toFixed(2)} s before the conflict: stopped short`;nc=GREEN;}
      else if(t>=r.t_ans){note=`warning available, driver needs ${tau.toFixed(2)} s`;}
      if(note){g.fillStyle=nc;g.font=(crashed?"bold ":"")+"11px Calibri";g.fillText(note,rp.x+6,rp.y+14);}
    }
    g.restore();
    let yy=y0+18; const line=(txt,col,bold,size)=>{g.fillStyle=col;g.font=`${bold?"bold ":""}${size||12}px Calibri`;g.fillText(txt,12,yy);yy+=(size||12)+4;};
    line(r.title,r.key==="ref"?INK:r.col,true,14);line(r.sub,INK2,false,11);line(`H7B3I-DK ${r.lat} ms   F401RE ${r.f401}`,INK2,false,11);
    yy=y0+18; const rline=(txt,col,bold,size)=>{g.fillStyle=col;g.font=`${bold?"bold ":""}${size||12}px Calibri`;g.fillText(txt,806,yy);yy+=(size||12)+4;};
    if(r.key==="tr"){
      if(t<r.t_flag)rline("no output yet",INK2,false,12);
      else{rline(`assumed to flag with the reference CNN at ${r.t_flag.toFixed(1)} s`,INK2,false,11);rline(`answer ${(r.lat/1000).toFixed(3)} s later = ${fmt(r.dist)} of road`,ORANGE,true,12);}
      if(!whatif){rline("latency: DMIR regression reference, same board;",INK2,false,10);rline("no classifier Transformer exists for this task",INK2,false,10);}
    } else {
      const o=outAt(r.key,t);
      if(t>=0)rline("crossing done",INK2,false,12);
      else if(!o)rline("no complete 5-s window yet",INK2,false,12);
      else rline(`says: ${D.class[o[0]]}  (p = ${o[1].toFixed(2)})`,o[0]===D.direction?r.col:INK,o[0]===D.direction,12);
      if(t>=r.t_flag)rline(`flag from ${r.t_flag.toFixed(1)} s, answer after ${fmt(r.dist)}`,INK2,false,11);
    }
    if(whatif&&t>=r.t_ans){
      rline(`warning ${r.margin.toFixed(2)} s before the conflict`,INK2,false,11);
      if(r.margin>tau)rline(`driver needing ${tau.toFixed(2)} s: stops in time`,GREEN,true,12);else rline(`driver needing ${tau.toFixed(2)} s: too late`,ORANGE,true,12);
      const bx=806,by=y0+rowH-26,bw=250,bh=9; const X=v=>bx+bw*Math.min(v,1.8)/1.8;
      g.fillStyle="#E8E8E4";g.fillRect(X(D.band[0]),by-bh,X(D.band[1])-X(D.band[0]),bh+5);
      g.fillStyle=r.margin>tau?r.col:ORANGE;g.fillRect(bx,by-bh+1,Math.max(X(Math.max(r.margin,0))-bx,2),bh-1);
      g.strokeStyle=INK;g.lineWidth=1;g.beginPath();g.moveTo(X(tau),by-bh-3);g.lineTo(X(tau),by+5);g.stroke();
      g.strokeStyle=INK2;g.lineWidth=0.6;g.beginPath();g.moveTo(bx,by+1);g.lineTo(bx+bw,by+1);g.stroke();
      g.fillStyle=INK2;g.font="9px Calibri";[0.5,1.0,1.5].forEach(v=>g.fillText(v.toFixed(1),X(v)-6,by+13));g.fillText("margin to conflict, s",bx,by-bh-6);
    }
  });
  tl.textContent=`t = ${t>=0?"+":""}${t.toFixed(1)} s`;
}
sl.oninput=()=>draw(+sl.value); ["whatif","showTr","tau"].forEach(id=>document.getElementById(id).oninput=()=>draw(+sl.value));
function step(){let i=+sl.value;if(i>=F.length-1){playing=false;document.getElementById("play").textContent="▶ Play";clearInterval(timer);return;}sl.value=i+1;draw(i+1);}
document.getElementById("play").onclick=()=>{if(playing){playing=false;clearInterval(timer);document.getElementById("play").textContent="▶ Play";}else{if(+sl.value>=F.length-1)sl.value=0;playing=true;document.getElementById("play").textContent="⏸ Pause";timer=setInterval(step,100);}};
draw(0);
</script></body></html>"""
    html = (html.replace("__DATA__", json.dumps(D)).replace("__NF__", str(len(FR) - 1))
            .replace("__LABEL__", S["label_name"]).replace("__TAU__", str(TAU))
            .replace("__IND__", f"{IND_ON:+.1f}").replace("__STEER__", f"{STEER:+.1f}"))
    path.write_text(html, encoding="utf-8")
    print("html ->", path)


if __name__ == "__main__":
    print(f"counterfactual driver: commits {T_COMMIT:+.1f} s, lateral rate {LAT_RATE:.2f} m/s, "
          f"reaches the recorded car at {T_CONFLICT:+.1f} s")
    for k, *_ in ROWS:
        g = GEO[k]
        print(f"  {k:5s} flag {g['t_flag']:+.1f} s  warning {g['t_ans']:+.3f} s  margin {g['margin']:.2f} s  "
              f"answer after {fmt(g['dist'])}  stops at tau=1.0: {g['margin'] > 1.0}")
    for tau in (0.75, 1.0, 1.25, 1.5):
        late = [k for k, *_ in ROWS if GEO[k]["margin"] <= tau]
        print(f"  tau {tau}: too late -> {late or 'none'}")
    for name, wi, tt in (("race_still.png", False, FLAG["ref"] + 0.3),
                         ("race_still_whatif.png", True, T_CONFLICT + 0.4)):
        fig = frame_figure(at(tt)["t"], wi)
        fig.savefig(OUT / name, dpi=100)
        plt.close(fig)
    print("stills ->", OUT)
    render_gif(OUT / "race.gif", False)
    render_gif(OUT / "race_whatif.gif", True)
    render_html(OUT / "race.html")
