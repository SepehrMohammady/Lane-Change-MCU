"""Render the scenario replay: animated GIF (for the slide), a still, and a
self-contained interactive HTML page (for questions).

Everything drawn is real: the target vehicle and its neighbours follow the raw
25 Hz highD track positions; lane markings come from the recording metadata;
the prediction flags come from running our trained models over the scenario's
26 evaluation windows (scenario_replay.py); latencies are board-farm
measurements. Two figures are *not* per-scenario and are labelled as such on
the picture: the published T-IV model's robust horizon (test-set average, the
model itself is not available) and the Transformer latency (the DMIR reference
Transformer measured on the same board; no highD Transformer exists).

Run in .venv:  python datasets/highd/render_replay.py
Outputs: Materials/T4.5/replay/{replay.gif, replay_still.png, replay.html}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPLAY = HERE / "data" / "replay"
OUT = ROOT / "Materials" / "T4.5" / "replay"
OUT.mkdir(parents=True, exist_ok=True)

# --- measured constants (see datasets/*/docs/*.md and benchmarks_api.jsonl) ---
LAT_SEARCHED_MS = 0.1063      # highD 5.3k classifier, int8 I/O, STM32H7B3I-DK
LAT_TRANSFORMER_MS = 368.8164 # DMIR reference Transformer LCR, fp32, same board
TIV_ROBUST_AVG_S = 3.96       # Mozaffari et al. T-IV 2022, Table III, tau_c
OURS_ROBUST_AVG_S = 4.53      # searched 5.3k model, our transcription of their metric

BLUE, ORANGE, NEUTRAL = "#4A66AC", "#C27C2E", "#9AA0A6"
INK, INK2, GRID, ROAD = "#242852", "#5A5F73", "#E1E0D9", "#F4F4F1"

scen = json.loads((REPLAY / "scenario.json").read_text())
p_hand = json.loads((REPLAY / "pred_hand_cnn.json").read_text())
p_srch = json.loads((REPLAY / "pred_searched.json").read_text())

FPS = scen["fps_raw"]
f0, fc, f1 = scen["frame_first"], scen["frame_cross"], scen["frame_end"]
v = scen["speed_at_cross_mps"]
d = scen["driving_direction"]
mirror = d == 1                      # make travel left-to-right for the viewer
label = scen["label_name"]
tv = scen["tv"]
marks = scen["lane_markings_y"]

def X(x, w=0.0):                     # road x -> display x (mirrored if needed)
    return -(x + w) if mirror else x

t_rel = lambda f: (f - fc) / FPS     # seconds relative to the crossing
flag_srch = -p_srch["robust_from_ttlc_s"] if p_srch["robust_from_ttlc_s"] else None
flag_hand = -p_hand["robust_from_ttlc_s"] if p_hand["robust_from_ttlc_s"] else None
first_srch = -p_srch["first_correct_ttlc_s"] if p_srch["first_correct_ttlc_s"] else None
dist_srch = v * LAT_SEARCHED_MS / 1000
dist_tr = v * LAT_TRANSFORMER_MS / 1000
ttlc_axis = [-t for t in p_srch["ttlc_s"]]           # window end times, -5.2 .. -0.2


def fmt_dist(m):
    return f"{m:.1f} m" if m >= 1 else (f"{m*100:.0f} cm" if m >= 0.01 else f"{m*1000:.0f} mm")

# neighbour lookup per frame
def nb_at(frame):
    out = []
    for nid, n in scen["neighbours"].items():
        if frame in n["frame"]:
            i = n["frame"].index(frame)
            out.append((n["x"][i], n["y"][i], n["w"], n["h"]))
    return out

frames = list(range(f0, f1 + 1, 2))                  # 12.5 fps
tv_index = {f: i for i, f in enumerate(tv["frame"])}


def draw(frame, ax_road, ax_band, ax_tl, ax_txt):
    i = tv_index[frame]
    x, y, w, h = tv["x"][i], tv["y"][i], tv["w"], tv["h"]
    t = t_rel(frame)
    cx = X(x, w) + w / 2
    # ---- road
    ax_road.clear()
    ax_road.set_facecolor(ROAD)
    y0, y1 = min(marks) - 1.0, max(marks) + 1.0
    ax_road.set_xlim(cx - 55, cx + 55)
    ax_road.set_ylim(y1, y0)                       # highD y grows downwards
    for k, m in enumerate(marks):
        ax_road.axhline(m, color="#7A7F8C", lw=1.6 if k in (0, len(marks) - 1) else 1.0,
                        ls="-" if k in (0, len(marks) - 1) else (0, (6, 6)))
    for nx, ny, nw, nh in nb_at(frame):
        ax_road.add_patch(Rectangle((X(nx, nw), ny), nw, nh, fc="#C9CDD6", ec="#7A7F8C", lw=0.8))
    ax_road.add_patch(Rectangle((X(x, w), y), w, h, fc=BLUE, ec="#1F2D5C", lw=1.0))
    ax_road.add_patch(FancyArrow(cx + w / 2 + 1, y + h / 2, 3, 0, width=0.25,
                                 head_width=0.9, head_length=1.4, fc=BLUE, ec="none"))
    ax_road.set_yticks([]); ax_road.set_xticks([])
    # ---- band under the road: road covered while ONE inference is computed,
    #      on the road's own x scale, starting at the car's front bumper
    ax_band.clear(); ax_band.set_xlim(cx - 55, cx + 55); ax_band.set_ylim(0, 1)
    ax_band.set_yticks([]); ax_band.set_xticks([])
    for sp in ax_band.spines.values():
        sp.set_visible(False)
    front = cx + w / 2
    if flag_srch is not None and t >= flag_srch:
        ax_band.add_patch(Rectangle((front, 0.58), dist_tr, 0.3, fc=NEUTRAL, ec="none"))
        ax_band.text(front + dist_tr + 0.8, 0.73, f"reference Transformer, one inference: {fmt_dist(dist_tr)} of road",
                     va="center", fontsize=9, color=INK2)
        ax_band.add_patch(Rectangle((front, 0.12), max(dist_srch, 0.12), 0.3, fc=BLUE, ec="none"))
        ax_band.text(front + max(dist_srch, 0.12) + 0.8, 0.27, f"searched model, one inference: {fmt_dist(dist_srch)}",
                     va="center", fontsize=9, color=INK2)
    else:
        ax_band.text(front, 0.5, "road covered during one inference appears here once the model first calls the lane change",
                     va="center", fontsize=8.5, color=INK2, style="italic")
    ax_band.axvline(front, color=INK, lw=0.8, ymin=0.05, ymax=0.95)
    for s in ax_road.spines.values():
        s.set_color(GRID)
    ax_road.set_title(f"highD recording {scen['recording']}, vehicle {scen['track_id']}: {label} at "
                      f"{v*3.6:.0f} km/h    t = {t:+.1f} s to lane crossing",
                      loc="left", fontsize=11, color=INK, fontweight="bold")
    # ---- timeline
    ax_tl.clear()
    ax_tl.set_xlim(t_rel(f0), t_rel(f1)); ax_tl.set_ylim(-0.05, 1.12)
    ax_tl.axvspan(t_rel(f0), min(t, t_rel(f1)), color="#EEF1F8", lw=0)
    # traces are revealed as the replay advances: a window's answer exists only once its last frame has passed
    n_vis = sum(1 for te in ttlc_axis if te <= t)
    ax_tl.plot(ttlc_axis[:n_vis], p_srch["p_lc"][:n_vis], color=BLUE, lw=2, label="searched CNN, 5.3k (this scenario)")
    ax_tl.plot(ttlc_axis[:n_vis], p_hand["p_lc"][:n_vis], color=BLUE, lw=1.4, ls=(0, (4, 3)), label="hand CNN, 8.4k (this scenario)")
    ax_tl.axhline(0.5, color=GRID, lw=0.8)
    ax_tl.axvline(0, color=INK, lw=1)
    ax_tl.text(0.05, 1.04, "lane crossing", fontsize=8.5, color=INK)
    if flag_srch is not None and t >= flag_srch:
        ax_tl.axvline(flag_srch, color=BLUE, lw=1.4)
        ax_tl.text(flag_srch + 0.05, 0.08, f"searched model: lane change\npredicted from {flag_srch:+.1f} s",
                   fontsize=8.5, color=BLUE)
    ax_tl.axvline(-TIV_ROBUST_AVG_S, color=ORANGE, lw=1.4)
    ax_tl.text(-TIV_ROBUST_AVG_S + 0.05, 0.62, f"published T-IV model:\n{-TIV_ROBUST_AVG_S:+.2f} s (test-set average)",
               fontsize=8.5, color=ORANGE)
    ax_tl.axvline(t, color=INK, lw=0.8, ls=":")
    ax_tl.set_xlabel("seconds relative to the lane crossing", fontsize=9, color=INK)
    ax_tl.set_ylabel("P(lane change)", fontsize=9, color=INK)
    ax_tl.tick_params(labelsize=8, colors=INK, length=0)
    ax_tl.grid(True, axis="y", color=GRID, lw=0.6); ax_tl.set_axisbelow(True)
    for s in ("top", "right"):
        ax_tl.spines[s].set_visible(False)
    ax_tl.legend(loc="lower right", fontsize=8, frameon=False)
    # ---- status text
    ax_txt.clear(); ax_txt.axis("off")
    k = max(j for j, te in enumerate(ttlc_axis) if te <= t) if t >= ttlc_axis[0] else None
    if k is None:
        srch_txt = hand_txt = "no complete 2-s window yet"
    elif t > 0:
        srch_txt = hand_txt = "crossing done"
    else:
        srch_txt = f"{'LANE CHANGE' if p_srch['pred'][k] else 'lane keep'}  (p = {p_srch['p_lc'][k]:.2f})"
        hand_txt = f"{'LANE CHANGE' if p_hand['pred'][k] else 'lane keep'}  (p = {p_hand['p_lc'][k]:.2f})"
    lines = [
        ("Same scenario, different methods", INK, 11, True),
        ("", INK, 4, False),
        (f"searched CNN, 5.3k, int8 on STM32H7B3I-DK", BLUE, 9.5, True),
        (f"   says: {srch_txt}", INK, 9.5, False),
        (f"   inference {LAT_SEARCHED_MS:.3f} ms  =  {fmt_dist(dist_srch)} of road", INK2, 9, False),
        ("", INK, 4, False),
        ("hand CNN, 8.4k (this scenario)", BLUE, 9.5, True),
        (f"   says: {hand_txt}", INK, 9.5, False),
        ("", INK, 4, False),
        ("published T-IV model (Mozaffari 2022)", ORANGE, 9.5, True),
        (f"   robust from {TIV_ROBUST_AVG_S:.2f} s before crossing, test-set average", INK, 9, False),
        ("   per-scenario output not available (model not public)", INK2, 8.5, False),
        ("", INK, 4, False),
        ("reference Transformer, same board", "#6B7280", 9.5, True),
        (f"   inference {LAT_TRANSFORMER_MS:.0f} ms  =  {fmt_dist(dist_tr)} of road at {v*3.6:.0f} km/h", INK, 9, False),
        ("   DMIR reference model, measured; no highD Transformer exists", INK2, 8.5, False),
    ]
    yy = 0.98
    for text, col, size, bold in lines:
        ax_txt.text(0.0, yy, text, fontsize=size, color=col, fontweight="bold" if bold else "normal",
                    va="top", ha="left", transform=ax_txt.transAxes)
        yy -= 0.055 if size >= 9 else 0.03


def render_frame(frame):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.22, 1.0], width_ratios=[1.55, 1], hspace=0.25, wspace=0.08,
                          left=0.05, right=0.98, top=0.93, bottom=0.09)
    ax_road = fig.add_subplot(gs[0, :]); ax_band = fig.add_subplot(gs[1, :])
    ax_tl = fig.add_subplot(gs[2, 0]); ax_txt = fig.add_subplot(gs[2, 1])
    draw(frame, ax_road, ax_band, ax_tl, ax_txt)
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    plt.close(fig)
    return img.convert("P", palette=Image.ADAPTIVE, colors=128)


imgs = [render_frame(f) for f in frames]
hold = [imgs[-1]] * 12                                  # pause on the last frame
imgs[0].save(OUT / "replay.gif", save_all=True, append_images=imgs[1:] + hold, duration=80, loop=0, optimize=True)
# a still at the searched model's prediction moment, for static use
key = min(frames, key=lambda f: abs(t_rel(f) - (flag_srch if flag_srch is not None else 0)))
render_frame(key).convert("RGB").save(OUT / "replay_still.png")
print(f"gif: {len(imgs)} frames -> {OUT/'replay.gif'} ({(OUT/'replay.gif').stat().st_size/1024:.0f} KB)")
print(f"still at t={t_rel(key):+.1f} s -> {OUT/'replay_still.png'}")

# ---------------------------------------------------------------- interactive HTML
data = {"scen": scen, "hand": p_hand, "srch": p_srch,
        "const": {"lat_srch_ms": LAT_SEARCHED_MS, "lat_tr_ms": LAT_TRANSFORMER_MS,
                  "tiv_robust": TIV_ROBUST_AVG_S, "ours_robust_avg": OURS_ROBUST_AVG_S}}
html = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Lane-change replay</title>
<style>
body{margin:0;font-family:Calibri,Segoe UI,sans-serif;background:#fff;color:#242852}
main{max-width:1180px;margin:0 auto;padding:16px 20px}
h1{font-size:20px;margin:0 0 6px}.sub{color:#5A5F73;font-size:13px;margin:0 0 12px}
canvas{display:block;width:100%;border:1px solid #E1E0D9;border-radius:8px}
.ctl{display:flex;gap:14px;align-items:center;margin:10px 0;font-size:14px;flex-wrap:wrap}
input[type=range]{flex:1;min-width:240px}
button{font:inherit;padding:5px 12px;border:1px solid #C9CDD6;border-radius:8px;background:#fff;cursor:pointer}
.legend span{display:inline-block;width:12px;height:12px;border-radius:2px;margin:0 6px 0 12px;vertical-align:middle}
.note{font-size:12.5px;color:#5A5F73;margin-top:10px;line-height:1.5}
</style></head><body><main>
<h1>One real lane change, several methods: when does the answer arrive?</h1>
<p class="sub" id="sub"></p>
<canvas id="c" width="1140" height="560"></canvas>
<div class="ctl"><button id="play">▶ Play</button><input type="range" id="t" min="0" max="1" step="1" value="0">
<span id="tlab" style="min-width:150px"></span>
<label><input type="checkbox" id="showTr" checked> Transformer latency</label>
<label><input type="checkbox" id="showTiv" checked> published T-IV horizon</label>
<label><input type="checkbox" id="showHand" checked> hand CNN</label></div>
<div class="legend" style="font-size:13px"><span style="background:#4A66AC"></span>searched CNN (5.3k, int8)
<span style="background:#4A66AC;border:1px dashed #fff"></span>hand CNN (8.4k)
<span style="background:#C27C2E"></span>published T-IV model (average)
<span style="background:#9AA0A6"></span>reference Transformer (latency)</div>
<p class="note">Positions are the raw 25 Hz highD track of this vehicle and its neighbours; the prediction traces are our trained models
run over the scenario's 26 evaluation windows; latencies are ST board-farm measurements on the STM32H7B3I-DK. The T-IV horizon is
that paper's test-set average (its model is not public), and the Transformer latency is the DMIR reference Transformer measured on
the same board (no highD Transformer exists). Display is mirrored when the vehicle drives right-to-left in the recording.</p>
</main>
<script>
const D = __DATA__;
const S=D.scen, TV=S.tv, FPS=S.fps_raw, fc=S.frame_cross, f0=S.frame_first, f1=S.frame_end;
const v=S.speed_at_cross_mps, mirror=S.driving_direction===1, marks=S.lane_markings_y;
const BLUE="#4A66AC",ORANGE="#C27C2E",NEUT="#9AA0A6",INK="#242852",INK2="#5A5F73",GRID="#E1E0D9";
const ttlc=D.srch.ttlc_s.map(t=>-t);
const robust=r=>r.robust_from_ttlc_s?-r.robust_from_ttlc_s:null;
const flagS=robust(D.srch), flagH=robust(D.hand);
const distS=v*D.const.lat_srch_ms/1000, distT=v*D.const.lat_tr_ms/1000;
const frames=[];for(let f=f0;f<=f1;f++)if(TV.frame.includes(f))frames.push(f);
const sl=document.getElementById("t");sl.max=frames.length-1;
const cv=document.getElementById("c"),g=cv.getContext("2d");
document.getElementById("sub").textContent=`highD recording ${S.recording}, vehicle ${S.track_id}: ${S.label_name} at ${(v*3.6).toFixed(0)} km/h. Drag the slider or press play.`;
const X=(x,w)=>mirror?-(x+w):x;
function nbAt(f){const o=[];for(const k in S.neighbours){const n=S.neighbours[k];const i=n.frame.indexOf(f);if(i>=0)o.push([n.x[i],n.y[i],n.w,n.h]);}return o;}
function draw(idx){
  const f=frames[idx], i=TV.frame.indexOf(f), t=(f-fc)/FPS;
  const x=TV.x[i],y=TV.y[i],w=TV.w,h=TV.h, cx=X(x,w)+w/2;
  const W=cv.width,H=cv.height; g.clearRect(0,0,W,H);
  // road panel
  const rp={x:20,y:34,w:W-40,h:232}; const y0=Math.min(...marks)-1, y1=Math.max(...marks)+1;
  const sx=rp.w/110, sy=rp.h/(y1-y0);
  const px=xx=>rp.x+(xx-(cx-55))*sx, py=yy=>rp.y+(yy-y0)*sy;
  g.fillStyle="#F4F4F1";g.fillRect(rp.x,rp.y,rp.w,rp.h);
  marks.forEach((m,k)=>{g.strokeStyle="#7A7F8C";g.lineWidth=(k===0||k===marks.length-1)?2:1;g.setLineDash((k===0||k===marks.length-1)?[]:[8,8]);g.beginPath();g.moveTo(rp.x,py(m));g.lineTo(rp.x+rp.w,py(m));g.stroke();});
  g.setLineDash([]);
  for(const [nx,ny,nw,nh] of nbAt(f)){g.fillStyle="#C9CDD6";g.strokeStyle="#7A7F8C";g.fillRect(px(X(nx,nw)),py(ny),nw*sx,nh*sy);g.strokeRect(px(X(nx,nw)),py(ny),nw*sx,nh*sy);}
  g.fillStyle=BLUE;g.fillRect(px(X(x,w)),py(y),w*sx,h*sy);
  const fd=m=>m>=1?`${m.toFixed(1)} m`:(m>=0.01?`${(m*100).toFixed(0)} cm`:`${(m*1000).toFixed(0)} mm`);
  const by=rp.y+rp.h+8; g.strokeStyle=INK;g.lineWidth=1;g.beginPath();g.moveTo(px(cx+w/2),by);g.lineTo(px(cx+w/2),by+26);g.stroke();
  g.font="12px Calibri";
  if(flagS!==null&&t>=flagS){
    if(document.getElementById("showTr").checked){g.fillStyle=NEUT;g.fillRect(px(cx+w/2),by+2,distT*sx,8);g.fillStyle=INK2;g.fillText(`reference Transformer, one inference: ${fd(distT)} of road`,px(cx+w/2)+distT*sx+6,by+10);}
    g.fillStyle=BLUE;g.fillRect(px(cx+w/2),by+16,Math.max(distS*sx,2),8);g.fillStyle=INK2;g.fillText(`searched model, one inference: ${fd(distS)}`,px(cx+w/2)+Math.max(distS*sx,2)+6,by+24);
  } else {g.fillStyle=INK2;g.font="italic 12px Calibri";g.fillText("road covered during one inference appears here once the model first calls the lane change",px(cx+w/2)+6,by+16);}
  g.fillStyle=INK;g.font="bold 15px Calibri";g.fillText(`t = ${t>=0?"+":""}${t.toFixed(1)} s to lane crossing`,rp.x,rp.y-10);
  // timeline
  const tp={x:60,y:340,w:W-420,h:190}; const t0=(f0-fc)/FPS,t1=(f1-fc)/FPS;
  const tx=tt=>tp.x+(tt-t0)/(t1-t0)*tp.w, ty=p=>tp.y+tp.h-(p*tp.h);
  g.fillStyle="#EEF1F8";g.fillRect(tp.x,tp.y,tx(Math.min(t,t1))-tp.x,tp.h);
  g.strokeStyle=GRID;g.lineWidth=1;[0,0.5,1].forEach(p=>{g.beginPath();g.moveTo(tp.x,ty(p));g.lineTo(tp.x+tp.w,ty(p));g.stroke();});
  g.fillStyle=INK2;g.font="11px Calibri";[0,0.5,1].forEach(p=>g.fillText(p.toFixed(1),tp.x-26,ty(p)+4));
  for(let s=Math.ceil(t0);s<=t1;s++){g.fillText((s>0?"+":"")+s+" s",tx(s)-8,tp.y+tp.h+16);}
  const trace=(arr,dash)=>{g.strokeStyle=BLUE;g.lineWidth=dash?1.5:2.2;g.setLineDash(dash?[5,4]:[]);g.beginPath();arr.forEach((p,k)=>{if(ttlc[k]>t)return;k?g.lineTo(tx(ttlc[k]),ty(p)):g.moveTo(tx(ttlc[k]),ty(p));});g.stroke();g.setLineDash([]);};
  trace(D.srch.p_lc,false); if(document.getElementById("showHand").checked)trace(D.hand.p_lc,true);
  g.strokeStyle=INK;g.lineWidth=1.2;g.beginPath();g.moveTo(tx(0),tp.y);g.lineTo(tx(0),tp.y+tp.h);g.stroke();g.fillStyle=INK;g.fillText("lane crossing",tx(0)+4,tp.y+12);
  if(flagS!==null&&t>=flagS){g.strokeStyle=BLUE;g.beginPath();g.moveTo(tx(flagS),tp.y);g.lineTo(tx(flagS),tp.y+tp.h);g.stroke();g.fillStyle=BLUE;g.fillText(`searched: predicted from ${flagS.toFixed(1)} s`,tx(flagS)+4,tp.y+tp.h-8);}
  if(document.getElementById("showTiv").checked){g.strokeStyle=ORANGE;g.beginPath();g.moveTo(tx(-D.const.tiv_robust),tp.y);g.lineTo(tx(-D.const.tiv_robust),tp.y+tp.h);g.stroke();g.fillStyle=ORANGE;g.fillText(`published T-IV: ${(-D.const.tiv_robust).toFixed(2)} s (avg)`,tx(-D.const.tiv_robust)+4,tp.y+tp.h/2);}
  g.strokeStyle=INK;g.setLineDash([2,3]);g.beginPath();g.moveTo(tx(t),tp.y);g.lineTo(tx(t),tp.y+tp.h);g.stroke();g.setLineDash([]);
  g.fillStyle=INK;g.font="12px Calibri";g.fillText("P(lane change) vs seconds to crossing",tp.x,tp.y-8);
  // status
  const k=t>=ttlc[0]?ttlc.map((te,j)=>te<=t?j:-1).reduce((a,b)=>Math.max(a,b),-1):-1;
  const say=r=>k<0?"no complete 2-s window yet":(t>0?"crossing done":`${r.pred[k]?"LANE CHANGE":"lane keep"} (p = ${r.p_lc[k].toFixed(2)})`);
  let yy=350;const sx2=W-330;const line=(txt,col,bold,size)=>{g.fillStyle=col;g.font=`${bold?"bold ":""}${size||13}px Calibri`;g.fillText(txt,sx2,yy);yy+=size?size+8:21;};
  line("Same scenario, different methods",INK,true,15);
  line("searched CNN, 5.3k, int8 on STM32H7B3I-DK",BLUE,true);line("   says: "+say(D.srch),INK,false);line(`   inference ${D.const.lat_srch_ms.toFixed(3)} ms = ${fd(distS)} of road`,INK2,false,12);
  if(document.getElementById("showHand").checked){line("hand CNN, 8.4k (this scenario)",BLUE,true);line("   says: "+say(D.hand),INK,false);}
  if(document.getElementById("showTiv").checked){line("published T-IV model (Mozaffari 2022)",ORANGE,true);line(`   robust from ${D.const.tiv_robust.toFixed(2)} s before crossing (test-set avg)`,INK,false,12);}
  if(document.getElementById("showTr").checked){line("reference Transformer, same board",INK2,true);line(`   inference ${D.const.lat_tr_ms.toFixed(0)} ms = ${fd(distT)} of road at ${(v*3.6).toFixed(0)} km/h`,INK,false,12);}
  document.getElementById("tlab").textContent=`t = ${t>=0?"+":""}${t.toFixed(2)} s`;
}
let playing=false,timer=null;
function step(){let i=+sl.value;if(i>=frames.length-1){playing=false;document.getElementById("play").textContent="▶ Play";clearInterval(timer);return;}sl.value=i+1;draw(i+1);}
document.getElementById("play").onclick=()=>{if(playing){playing=false;clearInterval(timer);document.getElementById("play").textContent="▶ Play";}else{if(+sl.value>=frames.length-1)sl.value=0;playing=true;document.getElementById("play").textContent="❚❚ Pause";timer=setInterval(step,1000/FPS);}};
sl.oninput=()=>draw(+sl.value);
["showTr","showTiv","showHand"].forEach(id=>document.getElementById(id).onchange=()=>draw(+sl.value));
draw(0);
</script></body></html>"""
(OUT / "replay.html").write_text(html.replace("__DATA__", json.dumps(data)), encoding="utf-8")
print("html ->", OUT / "replay.html")
