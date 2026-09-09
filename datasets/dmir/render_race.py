"""Render the DMIR "race": five cars on one real lane change, one per model.

Every row replays the same recorded manoeuvre of a held-out test driver (raw
positions from the H5 session, see scenario_race.py). What differs per row is
the model: the moment it first says "lane change" on the shared window stream
(its real output, exactly as the pipeline would compute it) and how much road
passes before that answer is available (board-farm latency x recorded speed).

Two things are not measurements and are labelled as such on the picture:
  * the fifth row is a Transformer of the reference size. No classifier
    Transformer exists for DMIR, so its latency is the DMIR regression reference
    Transformer measured on the same board (368.82 ms) and it is assumed to
    flag at the same window as the reference CNN;
  * the "what-if" overlay (off by default in the still, on in the second GIF)
    puts a hazard one car length ahead of the point where the question is asked
    and shows what a late answer means. It is an illustration of the measured
    latency, not an event in the data.

Run in .venv:  python datasets/dmir/render_race.py
Outputs: Materials/T4.5/race/{race.gif, race_whatif.gif, race_still.png, race.html}
"""
from __future__ import annotations

import json
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
LANE_W = 3.75

S = json.loads((RACE / "scenario.json").read_text())
FR = [f for f in S["frames"] if f["t"] >= -6.0]
L, WID = S["ego"]["length"], S["ego"]["width"]
DIRECTION = S["direction"]
T_W = S["t_windows"]
CLASS = {0: "no intention", 1: "RIGHT lane change", 2: "LEFT lane change"}

# board-farm latencies, STM32H7B3I-DK, ST Edge AI 4.0.1 balanced (datasets/dmir/docs/deployment.md)
ROWS = [
    ("ref",  "Reference CNN",              "441k params, float32",         33.52,  NEUTRAL, "does not fit"),
    ("best", "Searched, best accuracy",    "84k params, float32",          3.628,  BLUE,    "18.35 ms"),
    ("qat",  "Searched, QAT int8",         "84k params, int8 in and out",  1.435,  BLUE,    "7.381 ms"),
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


def row_geometry(key, lat_ms):
    """Where the question is asked, where the answer arrives, where the what-if hazard sits."""
    tf = FLAG[key]
    f = at(tf)
    q = f["x"] + L / 2                        # front bumper at the flag window
    v = f["v"]
    return {"t_flag": tf, "v": v, "q": q, "ans": q + v * lat_ms / 1000, "haz": q + L,
            "t_ans": tf + lat_ms / 1000, "t_haz": tf + L / v, "dist": v * lat_ms / 1000}


GEO = {k: row_geometry(k, ms) for k, _, _, ms, _, _ in ROWS}


def fmt(m):
    return f"{m:.1f} m" if m >= 1 else (f"{m*100:.0f} cm" if m >= 0.01 else f"{m*1000:.0f} mm")


def burst(ax, x, y, r, color):
    import math
    pts = [(x + (r if i % 2 == 0 else r * 0.45) * math.cos(i * math.pi / 8),
            y + (r if i % 2 == 0 else r * 0.45) * math.sin(i * math.pi / 8)) for i in range(16)]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=6))


def draw_row(ax, row, t, whatif):
    key, title, sub, lat_ms, col, f401 = row
    g = GEO[key]
    f = at(t)
    xe, ye = f["x"], f["y"]
    ax.clear()
    ax.set_facecolor(ROAD)
    ax.set_xlim(xe - 27, xe + 63)
    ax.set_ylim(-0.9, 2 * LANE_W + 0.6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    for yv, solid in ((0, True), (LANE_W, False), (2 * LANE_W, True)):
        ax.axhline(yv, color="#7A7F8C", lw=1.4 if solid else 0.9, ls="-" if solid else (0, (6, 6)))
    for o in f["objs"]:
        ax.add_patch(Rectangle((xe + o["long"] - o["len"] / 2, ye + o["lat"] - o["wid"] / 2), o["len"], o["wid"],
                               fc="#C9CDD6", ec="#7A7F8C", lw=0.7))
    crashed = whatif and key == "tr" and g["ans"] > g["haz"] and t >= g["t_haz"]
    ax.add_patch(Rectangle((xe - L / 2, ye - WID / 2), L, WID, fc=col, ec=INK, lw=0.8,
                           hatch="////" if crashed else None, zorder=5))
    # roof light: on once the answer is available
    on = t >= g["t_ans"] and not crashed
    ax.add_patch(Circle((xe + L * 0.15, ye), 0.42, fc="#FFD34D" if on else "#DDDDDD", ec=INK, lw=0.5, zorder=6))
    if t >= g["t_flag"]:
        # question tick + answer bar under the right edge line, on the road's own scale
        ax.plot([g["q"], g["q"]], [-0.15, -0.75], color=INK, lw=1.0)
        ax.add_patch(Rectangle((g["q"], -0.6), max(g["ans"] - g["q"], 0.15), 0.3, fc=col, ec="none"))
        ax.text(g["ans"] + 0.4, -0.45, f"answer after {fmt(g['dist'])}", va="center", fontsize=7.5, color=INK2, clip_on=True)
        if whatif:
            passed = t >= g["t_haz"]
            hz_col = ORANGE if (not passed or key == "tr") else "#B8BCC4"
            ax.add_patch(Rectangle((g["haz"] - 0.6, (ye // LANE_W) * LANE_W + 0.4), 1.2, LANE_W - 0.8,
                                   fc="none", ec=hz_col, lw=1.2, ls=(0, (3, 2)), zorder=4))
            if crashed:
                burst(ax, xe + L / 2, ye, 1.6, ORANGE)
                ax.text(xe + L / 2 + 2.4, ye, "what-if crash (illustration)", va="center", fontsize=7.5, color=ORANGE,
                        fontweight="bold", clip_on=True)
            elif not passed:
                ax.text(g["haz"] + 0.8, (ye // LANE_W) * LANE_W + LANE_W / 2, "what-if hazard, one car length ahead",
                        va="center", fontsize=7, color=ORANGE, clip_on=True)


def status_lines(row, t, whatif=False):
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
        lines.append(("latency: DMIR regression reference, same board;", INK2, 7, False))
        lines.append(("no classifier Transformer exists for this task", INK2, 7, False))
        if whatif and t >= g["t_haz"] and g["ans"] > g["haz"]:
            lines.append(("what-if: hazard reached before the answer", ORANGE, 8, True))
        return lines
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
    if whatif and t >= g["t_haz"] and g["ans"] <= g["haz"]:
        lines.append(("what-if: answered before the hazard", "#7A7F8C", 8, False))
    return lines


def frame_figure(t, whatif):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("white")
    fig.text(0.012, 0.965, f"DMIR, test driver 2: one real {S['label_name']} at {S['speed_at_cross_mps']*3.6:.0f} km/h, "
             f"five cars on the same inputs        t = {t:+.1f} s to lane crossing",
             fontsize=12.5, color=INK, fontweight="bold", va="center")
    top, h = 0.905, 0.163
    for i, row in enumerate(ROWS):
        y0 = top - (i + 1) * h
        ax = fig.add_axes([0.205, y0 + 0.012, 0.545, h - 0.02])
        draw_row(ax, row, t, whatif)
        yy = y0 + h - 0.025
        st = status_lines(row, t, whatif)
        for txt, colr, size, bold in st[:3]:
            fig.text(0.012, yy, txt, fontsize=size, color=colr, fontweight="bold" if bold else None, va="top")
            yy -= 0.028 if size >= 10 else 0.022
        yy = y0 + h - 0.025
        for txt, colr, size, bold in st[3:]:
            fig.text(0.762, yy, txt, fontsize=size, color=colr, fontweight="bold" if bold else None, va="top")
            yy -= 0.024
    # time bar
    axt = fig.add_axes([0.205, 0.035, 0.545, 0.04])
    axt.set_xlim(FR[0]["t"], FR[-1]["t"]); axt.set_ylim(0, 1); axt.axis("off")
    axt.plot([FR[0]["t"], FR[-1]["t"]], [0.5, 0.5], color=GRID, lw=2)
    axt.plot([FR[0]["t"], t], [0.5, 0.5], color=INK, lw=2)
    for tv, lab, c, ha in ((IND_ON, "indicator on ", INK2, "right"), (FLAG["best"], " models flag", BLUE, "left"),
                           (0, "lane crossing", INK, "center")):
        axt.plot([tv, tv], [0.25, 0.75], color=c, lw=1)
        axt.text(tv, 0.85, lab, fontsize=7, color=c, ha=ha)
    axt.plot([t], [0.5], marker="o", color=INK, ms=5)
    for tv in range(int(FR[0]["t"]), int(FR[-1]["t"]) + 1, 2):
        axt.text(tv, 0.05, f"{tv:+d} s", fontsize=6.5, color=INK2, ha="center")
    fig.text(0.012, 0.012, "Every row: same recorded vehicle, same windows, real model outputs; latencies measured on the board. "
             "Row 5 latency: DMIR regression reference Transformer. What-if hazard (when shown): illustration, not data.",
             fontsize=7, color=INK2)
    return fig


def render_gif(path, whatif, fps=12.5):
    frames = []
    for f in FR:
        fig = frame_figure(f["t"], whatif)
        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
        plt.close(fig)
    dur = [int(1000 / fps)] * len(frames)
    dur[-1] = 1500
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
    print(f"gif: {len(frames)} frames -> {path} ({path.stat().st_size // 1024} KB)")


def render_html(path):
    D = {"frames": FR, "L": L, "W": WID, "laneW": LANE_W, "direction": DIRECTION, "t_w": T_W,
         "probs": {k: S["probs"][k] for k in ("ref", "best", "qat", "tiny")},
         "rows": [{"key": k, "title": t, "sub": s, "lat": ms, "col": c, "f401": f, **GEO[k]} for k, t, s, ms, c, f in ROWS],
         "ind_on": IND_ON, "label": S["label_name"], "kmh": round(S["speed_at_cross_mps"] * 3.6),
         "class": CLASS}
    html = r"""<!doctype html><html><head><meta charset="utf-8"><title>DMIR race: one lane change, five models</title>
<style>body{font-family:Raleway,Calibri,Arial,sans-serif;color:#242852;margin:16px;background:#fff}
h1{font-size:18px;margin:0 0 6px} .ctl{margin:8px 0;font-size:13px} .ctl label{margin-right:14px}
canvas{border:1px solid #E1E0D9} input[type=range]{width:520px;vertical-align:middle} button{font-size:13px;padding:3px 10px}
p.note{font-size:12px;color:#5A5F73;max-width:1140px}</style></head><body>
<h1>DMIR, test driver 2: one real __LABEL__ at __KMH__ km/h, five cars on the same inputs</h1>
<div class="ctl"><button id="play">&#9654; Play</button> <input type="range" id="sl" min="0" max="__NF__" value="0">
<span id="tl"></span> &nbsp; <label><input type="checkbox" id="whatif"> what-if hazard (illustration)</label>
<label><input type="checkbox" id="showTr" checked> Transformer row</label></div>
<canvas id="c" width="1140" height="700"></canvas>
<p class="note">Every row replays the same recorded vehicle on the same 5-s windows; the "says" line is the model's real output for the
last complete window. The answer bar under each road is latency (STM32H7B3I-DK, ST Edge AI 4.0.1) times the recorded speed.
Row 5: no classifier Transformer exists for DMIR, so its latency is the regression reference Transformer measured on the same board,
and it is assumed to flag at the same window as the reference CNN. The what-if hazard is an illustration, not an event in the data.</p>
<script>
const D = __DATA__;
const BLUE="#4A66AC",ORANGE="#C27C2E",NEUT="#9AA0A6",INK="#242852",INK2="#5A5F73",GRID="#E1E0D9";
const cv=document.getElementById("c"),g=cv.getContext("2d"),sl=document.getElementById("sl"),tl=document.getElementById("tl");
const F=D.frames,L=D.L,WD=D.W,LW=D.laneW; let playing=false,timer=null;
const fmt=m=>m>=1?`${m.toFixed(1)} m`:(m>=0.01?`${(m*100).toFixed(0)} cm`:`${(m*1000).toFixed(0)} mm`);
function outAt(key,t){let k=-1;for(let i=0;i<D.t_w.length;i++)if(D.t_w[i]<=t+1e-6)k=i;if(k<0)return null;const p=D.probs[key][k];let c=0;for(let i=1;i<3;i++)if(p[i]>p[c])c=i;return [c,p[c]];}
function draw(idx){
  const f=F[idx],t=f.t; g.clearRect(0,0,cv.width,cv.height);
  const whatif=document.getElementById("whatif").checked, showTr=document.getElementById("showTr").checked;
  const rows=D.rows.filter(r=>showTr||r.key!=="tr"); const rowH=Math.floor(660/rows.length);
  g.fillStyle=INK;g.font="bold 15px Calibri";g.fillText(`t = ${t>=0?"+":""}${t.toFixed(1)} s to lane crossing`,12,20);
  rows.forEach((r,i)=>{
    const y0=32+i*rowH, rp={x:230,y:y0+6,w:560,h:rowH-26}; const sx=rp.w/90, sy=rp.h/(2*LW+1.5);
    const px=x=>rp.x+(x-(f.x-27))*sx, py=y=>rp.y+rp.h-(y+0.9)*sy;
    g.save();g.beginPath();g.rect(rp.x-1,rp.y-1,rp.w+2,rp.h+2);g.clip();
    g.fillStyle="#F4F4F1";g.fillRect(rp.x,rp.y,rp.w,rp.h);
    [[0,1],[LW,0],[2*LW,1]].forEach(([yv,solid])=>{g.strokeStyle="#7A7F8C";g.lineWidth=solid?1.4:0.9;g.setLineDash(solid?[]:[6,6]);g.beginPath();g.moveTo(rp.x,py(yv));g.lineTo(rp.x+rp.w,py(yv));g.stroke();g.setLineDash([]);});
    f.objs.forEach(o=>{g.fillStyle="#C9CDD6";g.strokeStyle="#7A7F8C";g.lineWidth=0.7;const X=px(f.x+o.long-o.len/2),Y=py(f.y+o.lat+o.wid/2);g.fillRect(X,Y,o.len*sx,o.wid*sy);g.strokeRect(X,Y,o.len*sx,o.wid*sy);});
    const crashed=whatif&&r.key==="tr"&&r.ans>r.haz&&t>=r.t_haz;
    g.fillStyle=r.col;g.strokeStyle=INK;g.lineWidth=0.8;const EX=px(f.x-L/2),EY=py(f.y+WD/2);g.fillRect(EX,EY,L*sx,WD*sy);g.strokeRect(EX,EY,L*sx,WD*sy);
    if(crashed){g.strokeStyle="#fff";g.lineWidth=1;for(let k=0;k<6;k++){g.beginPath();g.moveTo(EX+k*L*sx/6,EY+WD*sy);g.lineTo(EX+(k+1)*L*sx/6,EY);g.stroke();}}
    const on=t>=r.t_ans&&!crashed; g.fillStyle=on?"#FFD34D":"#DDDDDD";g.beginPath();g.arc(px(f.x+L*0.15),py(f.y),4,0,6.283);g.fill();g.strokeStyle=INK;g.lineWidth=0.5;g.stroke();
    if(t>=r.t_flag){
      g.strokeStyle=INK;g.lineWidth=1;g.beginPath();g.moveTo(px(r.q),py(-0.15));g.lineTo(px(r.q),py(-0.75));g.stroke();
      g.fillStyle=r.col;g.fillRect(px(r.q),py(-0.3),Math.max((r.ans-r.q)*sx,2),0.3*sy);
      g.fillStyle=INK2;g.font="11px Calibri";g.fillText(`answer after ${fmt(r.dist)}`,px(r.ans)+5,py(-0.45)+4);
      if(whatif){
        const passed=t>=r.t_haz, laneY=Math.floor(f.y/LW)*LW; const hc=(!passed||r.key==="tr")?ORANGE:"#B8BCC4";
        g.strokeStyle=hc;g.lineWidth=1.2;g.setLineDash([3,2]);g.strokeRect(px(r.haz-0.6),py(laneY+LW-0.4),1.2*sx,(LW-0.8)*sy);g.setLineDash([]);
        g.font="11px Calibri";
        if(crashed){g.fillStyle=ORANGE;const cx=px(f.x+L/2),cy=py(f.y);g.beginPath();for(let k=0;k<16;k++){const rr=(k%2?0.45:1)*1.6;const a=k*Math.PI/8;g.lineTo(cx+rr*sx*Math.cos(a),cy-rr*sy*Math.sin(a));}g.closePath();g.fill();
          g.font="bold 11px Calibri";g.fillText("what-if crash (illustration)",cx+2.2*sx,py(f.y+1.7));}
        else if(!passed){g.fillStyle=ORANGE;g.fillText("what-if hazard, one car length ahead",px(r.haz)+8,py(laneY+LW/2)+4);}
      }
    }
    g.restore();
    // left text
    let yy=y0+20; const line=(txt,col,bold,size)=>{g.fillStyle=col;g.font=`${bold?"bold ":""}${size||12}px Calibri`;g.fillText(txt,12,yy);yy+=(size||12)+4;};
    line(r.title,r.key==="ref"?INK:r.col,true,14);line(r.sub,INK2,false,11);line(`H7B3I-DK ${r.lat} ms   F401RE ${r.f401}`,INK2,false,11);
    // right text
    yy=y0+20; const rline=(txt,col,bold,size)=>{g.fillStyle=col;g.font=`${bold?"bold ":""}${size||12}px Calibri`;g.fillText(txt,806,yy);yy+=(size||12)+4;};
    if(r.key==="tr"){
      if(t<r.t_flag)rline("no output yet",INK2,false,12);
      else{rline(`assumed to flag with the reference CNN at ${r.t_flag.toFixed(1)} s`,INK2,false,11);rline(`answer ${(r.lat/1000).toFixed(3)} s later = ${fmt(r.dist)} of road`,ORANGE,true,12);}
      rline("latency: DMIR regression reference, same board;",INK2,false,10);rline("no classifier Transformer exists for this task",INK2,false,10);
      if(whatif&&t>=r.t_haz&&r.ans>r.haz)rline("what-if: hazard reached before the answer",ORANGE,true,11);
    } else {
      const o=outAt(r.key,t);
      if(t>=0)rline("crossing done",INK2,false,12);
      else if(!o)rline("no complete 5-s window yet",INK2,false,12);
      else rline(`says: ${D.class[o[0]]}  (p = ${o[1].toFixed(2)})`,o[0]===D.direction?r.col:INK,o[0]===D.direction,12);
      if(t>=r.t_flag)rline(`flag from ${r.t_flag.toFixed(1)} s, answer after ${fmt(r.dist)}`,INK2,false,11);
      if(whatif&&t>=r.t_haz&&r.ans<=r.haz)rline("what-if: answered before the hazard","#7A7F8C",false,11);
    }
  });
  tl.textContent=`t = ${t>=0?"+":""}${t.toFixed(1)} s`;
}
sl.oninput=()=>draw(+sl.value); document.getElementById("whatif").onchange=()=>draw(+sl.value); document.getElementById("showTr").onchange=()=>draw(+sl.value);
function step(){let i=+sl.value;if(i>=F.length-1){playing=false;document.getElementById("play").textContent="▶ Play";clearInterval(timer);return;}sl.value=i+1;draw(i+1);}
document.getElementById("play").onclick=()=>{if(playing){playing=false;clearInterval(timer);document.getElementById("play").textContent="▶ Play";}else{if(+sl.value>=F.length-1)sl.value=0;playing=true;document.getElementById("play").textContent="⏸ Pause";timer=setInterval(step,100);}};
draw(0);
</script></body></html>"""
    html = (html.replace("__DATA__", json.dumps(D)).replace("__NF__", str(len(FR) - 1))
            .replace("__LABEL__", S["label_name"]).replace("__KMH__", str(round(S["speed_at_cross_mps"] * 3.6))))
    path.write_text(html, encoding="utf-8")
    print("html ->", path)


if __name__ == "__main__":
    for k, *_ in ROWS:
        g = GEO[k]
        print(f"{k:5s} flag {g['t_flag']:+.1f} s  v {g['v']:.1f} m/s  answer after {fmt(g['dist'])}  hazard reached {g['t_haz']:+.2f} s")
    still_t = FLAG["ref"] + 0.3
    fig = frame_figure(at(still_t)["t"], False)
    fig.savefig(OUT / "race_still.png", dpi=100)
    plt.close(fig)
    fig = frame_figure(at(still_t)["t"], True)
    fig.savefig(OUT / "race_still_whatif.png", dpi=100)
    plt.close(fig)
    print("stills ->", OUT)
    render_gif(OUT / "race.gif", False)
    render_gif(OUT / "race_whatif.gif", True)
    render_html(OUT / "race.html")
