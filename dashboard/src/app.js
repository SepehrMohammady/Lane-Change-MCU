(() => {
  "use strict";
  const D = window.DATA;
  const SVG = "http://www.w3.org/2000/svg";
  const BOARDS = ["STM32H7B3I-DK", "NUCLEO-F401RE"];
  const BOARD_SHORT = { "STM32H7B3I-DK": "H7B3I-DK · Cortex-M7", "NUCLEO-F401RE": "F401RE · Cortex-M4" };

  // ------------------------------------------------------------ helpers
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style") el.style.cssText = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const store = {
    get(k, d) { try { const v = localStorage.getItem("lcnas:" + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem("lcnas:" + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
  };

  const f = {
    pct: v => (v * 100).toFixed(2) + "%",
    pct1: v => (v * 100).toFixed(1) + "%",
    f3: v => v.toFixed(3),
    s3: v => v.toFixed(3) + " s",
    s2: v => v.toFixed(2) + " s",
    int: v => Math.round(v).toLocaleString("en-US"),
    params(v) { return v >= 1e5 ? Math.round(v / 1e3) + " k" : v >= 1e4 ? Math.round(v / 1e3) + " k" : (v / 1e3).toFixed(1) + " k"; },
    bytes(b) { if (b == null) return "–"; const kb = b / 1024; return kb >= 1000 ? (kb / 1024).toFixed(2) + " MB" : kb >= 100 ? Math.round(kb) + " KB" : kb.toFixed(1) + " KB"; },
    ms(v) { if (v == null) return "–"; return v < 1 ? v.toFixed(3) + " ms" : v < 10 ? v.toFixed(2) + " ms" : v < 100 ? v.toFixed(1) + " ms" : Math.round(v) + " ms"; },
    dist(m) { return m >= 1 ? m.toFixed(1) + " m" : m >= 0.01 ? (m * 100).toFixed(0) + " cm" : (m * 1000).toFixed(1) + " mm"; },
  };
  const fmtMetric = (key, v) => (f[key] || f.f3)(v);

  // ------------------------------------------------------------ tooltip
  const tip = document.getElementById("tip");
  function showTip(evt, build) {
    tip.replaceChildren(...build());
    tip.hidden = false;
    const r = evt.target.getBoundingClientRect ? evt.target.getBoundingClientRect() : null;
    const x = evt.clientX != null && evt.clientX !== 0 ? evt.clientX : (r ? r.left + r.width / 2 : 0);
    const y = evt.clientY != null && evt.clientY !== 0 ? evt.clientY : (r ? r.top : 0);
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = x + 14, top = y - th - 12;
    if (left + tw > window.innerWidth - 8) left = x - tw - 14;
    if (top < 8) top = y + 16;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = top + "px";
  }
  const hideTip = () => { tip.hidden = true; };
  function tipRows(title, value, rows) {
    const out = [];
    if (value != null) out.push(h("div", { class: "tv" }, value));
    if (title) out.push(h("div", { class: "tl" }, title));
    for (const [k, v] of rows || []) out.push(h("div", { class: "tr" }, h("span", {}, k), h("b", {}, v)));
    return out;
  }
  function hover(el, build) {
    el.setAttribute("tabindex", "0");
    el.addEventListener("pointermove", e => showTip(e, build));
    el.addEventListener("pointerleave", hideTip);
    el.addEventListener("focus", e => showTip(e, build));
    el.addEventListener("blur", hideTip);
  }

  // ------------------------------------------------------------ scales
  function linScale(d0, d1, r0, r1) {
    const sc = v => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0);
    sc.ticks = n => {
      const span = d1 - d0, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(st => span / st <= n) || mag * 10;
      const out = [];
      for (let t = Math.ceil(d0 / step) * step; t <= d1 + 1e-12; t += step) out.push(+t.toFixed(10));
      return out;
    };
    return sc;
  }
  function logScale(d0, d1, r0, r1) {
    const l0 = Math.log10(d0), l1 = Math.log10(d1);
    const sc = v => r0 + (Math.log10(v) - l0) / (l1 - l0 || 1) * (r1 - r0);
    sc.ticks = (unit = 1) => {
      const out = [], mults = (l1 - l0) > 2.4 ? [1, 3] : [1, 2, 5];
      const u0 = Math.log10(d0 / unit), u1 = Math.log10(d1 / unit);
      for (let e = Math.floor(u0); e <= Math.ceil(u1); e++) for (const m of mults) {
        const v = Number((m * Math.pow(10, e)).toPrecision(6)) * unit;
        if (v >= d0 * 0.999 && v <= d1 * 1.001) out.push(v);
      }
      return out;
    };
    return sc;
  }
  const tickParams = v => v >= 1e6 ? v / 1e6 + " M" : v >= 1e3 ? v / 1e3 + " k" : String(v);
  const tickMs = v => v + " ms";

  function chartCard(title, caption, body, opts = {}) {
    const box = h("div", { class: "chart-box" }, body);
    const card = h("section", { class: "card" + (opts.span ? " span2" : "") },
      h("div", { class: "card-head" },
        h("div", {}, h("h2", {}, title), caption ? h("p", { class: "cap" }, caption) : null),
        opts.table ? h("button", {
          class: "linkbtn", type: "button", "aria-pressed": "false",
          onclick: e => {
            const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
            e.currentTarget.setAttribute("aria-pressed", on);
            e.currentTarget.textContent = on ? "Chart" : "Table";
            box.replaceChildren(on ? opts.table() : body);
          }
        }, "Table") : null),
      box, opts.after || null);
    return card;
  }
  function simpleTable(head, rows, numCols = []) {
    return h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
      h("thead", {}, h("tr", {}, head.map((c, i) => h("th", { class: numCols.includes(i) ? "num" : null }, c)))),
      h("tbody", {}, rows.map(r => h("tr", {}, r.map((c, i) => h("td", { class: numCols.includes(i) ? "num" : null }, c)))))));
  }

  // ------------------------------------------------------------ state + routing
  const ds = id => D.datasets.find(d => d.id === id);
  const state = {
    board: store.get("board", BOARDS[0]),
    speed: store.get("speed", 100),
    task: store.get("task", {}),
    recipe: store.get("recipe", "search"),
  };
  const ALL_VIEWS = [
    ["overview", "Overview"], ["search", "Search"], ["hardware", "Hardware"], ["quant", "Quantization"],
    ["transfer", "Transfer"], ["robust", "Seeds"], ["bench", "Benchmark"], ["scenario", "Scenario"],
  ];
  const DEFAULT_VIEWS = ["overview", "search", "hardware", "quant", "robust", "bench", "scenario"];
  const viewsOf = d => ALL_VIEWS.filter(([id]) => (d.views || DEFAULT_VIEWS).includes(id));
  function route() {
    const [a, b, c] = decodeURIComponent(location.hash.slice(1)).split("/");
    if (a === "compare" || (a === "status" && D.status)) return { page: a };
    const d = ds(a) ? a : "dmir";
    const planned = ds(d).status === "planned";
    const v = viewsOf(ds(d)).some(x => x[0] === b) && !planned ? b : "overview";
    return { page: "ds", ds: d, view: v, model: c };
  }
  const go = hash => { if (location.hash.slice(1) !== hash) location.hash = hash; else render(); };

  // ------------------------------------------------------------ header
  function renderHeader(r) {
    const nav = document.getElementById("ds-nav");
    const items = D.datasets.map(d => h("button", {
      type: "button", "aria-current": String(r.page === "ds" && r.ds === d.id),
      onclick: () => go(d.id + "/" + (r.page === "ds" && d.status !== "planned" ? r.view : "overview")),
    }, h("span", { class: "dot", style: `background:var(--s${d.accent})` }), d.name,
      d.status === "planned" ? h("span", { class: "soon" }, "next") : null));
    items.push(h("span", { class: "ds-sep", "aria-hidden": "true" }));
    items.push(h("button", { type: "button", "aria-current": String(r.page === "compare"), onclick: () => go("compare") }, "Compare"));
    if (D.status) items.push(h("button", { type: "button", "aria-current": String(r.page === "status"), onclick: () => go("status") }, "Status"));
    nav.replaceChildren(...items);

    const tabs = document.getElementById("tab-nav");
    if (r.page === "ds" && ds(r.ds).status !== "planned") {
      tabs.replaceChildren(...viewsOf(ds(r.ds)).map(([id, label]) => h("button", {
        type: "button", "aria-current": String(r.view === id), onclick: () => go(r.ds + "/" + id),
      }, label)));
    } else tabs.replaceChildren();
    document.body.dataset.accent = r.page === "ds" ? ds(r.ds).accent : 1;
  }

  // ------------------------------------------------------------ overview
  function viewOverview(d) {
    const hero = h("div", { class: "hero" },
      h("section", { class: "card hero-main" },
        h("div", { class: "eyebrow" }, d.name + " · " + (d.status === "planned" ? "planned" : "complete")),
        h("div", { class: "hero-title" }, d.title),
        h("div", { class: "hero-num" }, d.headline.value),
        h("div", { class: "hero-label" }, d.headline.label),
        h("p", { class: "hero-sub" }, d.headline.sub)),
      h("section", { class: "card" },
        h("div", { class: "card-head" }, h("h2", {}, "The data"),
          h("span", { class: "status-pill" + (d.status === "planned" ? " planned" : "") }, d.status === "planned" ? "planned" : "done")),
        h("dl", { class: "facts" }, d.facts.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]))));
    if (d.status === "planned") return [hero];

    const kpis = h("div", { class: "grid g4" }, d.kpis.map(k => h("section", { class: "card tile" },
      h("div", { class: "t-label" }, k.label), h("div", { class: "t-value" }, k.value), h("div", { class: "t-sub" }, k.sub),
      k.seed ? h("div", { class: "t-seed" }, k.seed) : null)));
    const pipe = h("section", { class: "card" },
      h("div", { class: "card-head" }, h("div", {}, h("h2", {}, d.pipeline_title || "From search to silicon"),
        h("p", { class: "cap" }, d.pipeline_caption || "Search under memory and compute limits, quantize, then measure on real boards."))),
      h("div", { class: "pipe" }, d.pipeline.map(([l, n]) => h("div", { class: "step" }, h("div", { class: "n" }, n), h("div", { class: "l" }, l)))));

    const shortcuts = h("div", { class: "grid g3" }, (d.shortcuts || [
      ["search", "Search", "Every model the search saved, and where the picks sit."],
      ["hardware", "Hardware", "Latency, flash and RAM measured on both boards."],
      ["robust", "Seeds", "The final models retrained with five seeds."],
    ]).map(([id, t, c]) => h("button", { class: "card", type: "button", style: "text-align:left;cursor:pointer", onclick: () => go(d.id + "/" + id) },
      h("h2", {}, t + " →"), h("p", { class: "cap" }, c))));
    return [hero, h("div", { class: "section-title" }, "Headline"), kpis, h("div", { style: "height:16px" }), pipe,
      h("div", { style: "height:16px" }), shortcuts];
  }

  // ------------------------------------------------------------ search: Pareto scatter
  function seedStat(d, key, metric, recipe) {
    const pool = recipe === "final" && d.seeds_final ? d.seeds_final : d.seeds;
    const g = pool && pool[key];
    return g && g.stats[metric] ? g.stats[metric] : null;
  }
  function scatter(d, task, showSeeds) {
    const m = task.metric;
    const W = 760, H = 380, M = { l: 58, r: task.refs.some(r => r.kind === "line") ? 170 : 40, t: 16, b: 44 };
    const all = task.fronts.flatMap(fr => fr.rows);
    const pts = [...all.map(r => ({ x: r.params, y: r.value })),
      ...task.refs.filter(r => r.kind === "point").map(r => ({ x: r.params, y: r.value }))];
    if (showSeeds) {
      for (const key of [...task.picks, ...task.refs].map(x => x.seed_key).filter(Boolean)) {
        for (const recipe of ["search", "final"]) {
          const st = seedStat(d, key, m.key, recipe);
          if (st) pts.push({ x: pts[0].x, y: st.mean - st.std }, { x: pts[0].x, y: st.mean + st.std });
        }
      }
    }
    let yLo = Math.min(...pts.map(p => p.y)), yHi = Math.max(...pts.map(p => p.y));
    const lines = task.refs.filter(r => r.kind === "line");
    const pad = (yHi - yLo) * 0.08 || 0.01;
    yLo -= pad; yHi += pad;
    const inRange = [], offScale = [];
    for (const ln of lines) {
      if (ln.value >= yLo - (yHi - yLo) * 0.6 && ln.value <= yHi + (yHi - yLo) * 0.6) { yLo = Math.min(yLo, ln.value - pad); yHi = Math.max(yHi, ln.value + pad); inRange.push(ln); }
      else offScale.push(ln);
    }
    const xLo = Math.min(...pts.map(p => p.x)) / 1.35, xHi = Math.max(...pts.map(p => p.x)) * 1.35;
    const X = logScale(xLo, xHi, M.l, W - M.r), Y = linScale(yLo, yHi, H - M.b, M.t);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": task.label + " search results" });

    for (const t of Y.ticks(5)) {
      svg.append(s("line", { class: "gridline", x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t) }),
        s("text", { class: "ax-tick", x: M.l - 8, y: Y(t) + 4, "text-anchor": "end" }, m.fmt === "pct" ? (t * 100).toFixed(t * 100 % 1 ? 1 : 0) + "%" : t.toFixed(2)));
    }
    for (const t of X.ticks()) svg.append(s("text", { class: "ax-tick", x: X(t), y: H - M.b + 18, "text-anchor": "middle" }, tickParams(t)));
    svg.append(s("line", { class: "baseline", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }),
      s("text", { class: "ax-title", x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle" }, "parameters (log scale)"),
      s("text", { class: "ax-title", x: 14, y: (M.t + H - M.b) / 2, transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})`, "text-anchor": "middle" },
        m.name + (m.better === "high" ? "  ↑ better" : "  ↓ better")));

    for (const ln of inRange) {
      svg.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(ln.value), y2: Y(ln.value), stroke: "var(--ref)", "stroke-width": 1.5 }),
        s("text", { class: "lbl-2", x: W - M.r + 8, y: Y(ln.value) + 4 }, ln.label),
        s("text", { class: "lbl-m", x: W - M.r + 8, y: Y(ln.value) + 17 }, fmtMetric(m.fmt, ln.value)));
    }
    offScale.forEach((ln, i) => {
      const yy = ln.value > yHi ? M.t + 6 + i * 26 : H - M.b - 20 - i * 26;
      svg.append(s("text", { class: "lbl-2", x: W - M.r + 8, y: yy }, (ln.value > yHi ? "↑ " : "↓ ") + ln.label),
        s("text", { class: "lbl-m", x: W - M.r + 8, y: yy + 13 }, fmtMetric(m.fmt, ln.value) + ", off the scale"));
    });

    const layer = s("g", {});
    svg.append(layer);
    task.fronts.forEach((fr, fi) => {
      for (const r of fr.rows) {
        if (task.picks.some(p => p.front === fi && p.model === r.model)) continue;
        const g = s("g", { class: "mk" },
          s("circle", { cx: X(r.params), cy: Y(r.value), r: 4, fill: fi === 0 ? "var(--front)" : "var(--front-2)", stroke: "var(--card)", "stroke-width": 2 }),
          s("circle", { class: "hit", cx: X(r.params), cy: Y(r.value), r: 12 }));
        hover(g, () => tipRows(fr.label + " · " + r.model, fmtMetric(m.fmt, r.value), [["parameters", f.int(r.params)], ["int8 size", r.kb.toFixed(1) + " KB"]]));
        layer.append(g);
      }
    });

    // label placement: try positions around the mark, avoid marks, whiskers and earlier labels
    const occupied = [];
    const overlaps = (r) => occupied.some(o => r.x0 < o.x1 && o.x0 < r.x1 && r.y0 < o.y1 && o.y0 < r.y1);
    const inside = (r) => r.x0 >= M.l - 40 && r.x1 <= W - 4 && r.y0 >= 0 && r.y1 <= H - M.b;
    const reserve = (x0, y0, x1, y1) => occupied.push({ x0, y0, x1, y1 });
    const markPts = [
      ...task.picks.map(p => { const row = (task.fronts[p.front] || { rows: [] }).rows.find(r => r.model === p.model); return row && { x: X(row.params), y: Y(row.value), key: p.seed_key, recipe: state.recipe }; }),
      ...task.refs.filter(r => r.kind === "point").map(r => ({ x: X(r.params), y: Y(r.value), key: r.seed_key, recipe: "search" })),
    ].filter(Boolean);
    for (const mp of markPts) {
      reserve(mp.x - 7, mp.y - 7, mp.x + 7, mp.y + 7);
      const st = showSeeds && mp.key ? seedStat(d, mp.key, m.key, mp.recipe) : null;
      if (st) reserve(mp.x + 5, Y(st.mean + st.std) - 2, mp.x + 15, Y(st.mean - st.std) + 2);
    }
    const placeLabel = (x, y, text, cls) => {
      const w = text.length * (cls === "lbl" ? 6.9 : 6.3), hgt = 13;
      const cands = [[14, 4, "start"], [14, -14, "start"], [14, 22, "start"], [-14, 4, "end"], [-14, -14, "end"], [-14, 22, "end"],
        [20, -32, "start"], [20, 40, "start"], [-20, -32, "end"], [-20, 40, "end"], [26, -50, "start"], [26, 58, "start"]];
      for (const [dx, dy, anchor] of cands) {
        const lx = x + dx, ly = y + dy;
        const r = anchor === "start" ? { x0: lx - 2, y0: ly - hgt + 3, x1: lx + w + 2, y1: ly + 4 } : { x0: lx - w - 2, y0: ly - hgt + 3, x1: lx + 2, y1: ly + 4 };
        if (inside(r) && !overlaps(r)) {
          occupied.push(r);
          const g = s("g", {});
          if (Math.abs(dy) > 10) g.append(s("line", { x1: x, y1: y + (dy < 0 ? -7 : 7), x2: lx + (anchor === "start" ? -2 : 2), y2: ly - 4, stroke: "var(--axis)", "stroke-width": 1 }));
          g.append(s("text", { class: cls, x: lx, y: ly, "text-anchor": anchor }, text));
          return g;
        }
      }
      return s("text", { class: cls, x: x + 14, y: y + 4 }, text);
    };
    const whisker = (x, st) => st && s("g", { class: "mk" },
      s("line", { x1: x + 10, x2: x + 10, y1: Y(st.mean - st.std), y2: Y(st.mean + st.std), stroke: "var(--ink-2)", "stroke-width": 2, "stroke-linecap": "round" }),
      s("circle", { cx: x + 10, cy: Y(st.mean), r: 4, fill: "var(--card)", stroke: "var(--ink-2)", "stroke-width": 2 }));

    const labels = [];
    for (const rf of task.refs.filter(r => r.kind === "point")) {
      const x = X(rf.params), y = Y(rf.value), st = showSeeds && rf.seed_key ? seedStat(d, rf.seed_key, m.key, "search") : null;
      const g = s("g", { class: "mk hl" },
        s("rect", { x: x - 6, y: y - 6, width: 12, height: 12, transform: `rotate(45 ${x} ${y})`, fill: "var(--ref)", stroke: "var(--card)", "stroke-width": 2 }),
        s("circle", { class: "hit", cx: x, cy: y, r: 14 }));
      hover(g, () => tipRows(rf.label, fmtMetric(m.fmt, rf.value), [["parameters", f.int(rf.params)],
        ...(st ? [["5-seed mean", fmtMetric(m.fmt, st.mean) + " ± " + (m.fmt === "pct" ? (st.std * 100).toFixed(2) + " pt" : st.std.toFixed(3))]] : [])]));
      layer.append(whisker(x, st) || s("g"), g);
      labels.push([x, y, rf.label, "lbl-2"]);
    }
    for (const p of task.picks) {
      const row = (task.fronts[p.front] || { rows: [] }).rows.find(r => r.model === p.model);
      if (!row) continue;
      const x = X(row.params), y = Y(row.value);
      const st = showSeeds ? seedStat(d, p.seed_key, m.key, state.recipe) : null;
      const g = s("g", { class: "mk hl", style: "cursor:pointer", onclick: () => openModel(d, p.id) },
        s("circle", { cx: x, cy: y, r: 6, fill: "var(--acc)", stroke: "var(--card)", "stroke-width": 2 }),
        s("circle", { class: "hit", cx: x, cy: y, r: 14 }));
      hover(g, () => tipRows(p.label, fmtMetric(m.fmt, row.value), [["parameters", f.int(row.params)], ["int8 size", row.kb.toFixed(1) + " KB"],
        ...(st ? [["5-seed mean", fmtMetric(m.fmt, st.mean) + " ± " + (m.fmt === "pct" ? (st.std * 100).toFixed(2) + " pt" : st.std.toFixed(3))]] : []),
        ["", "click for details"]]));
      layer.append(whisker(x, st) || s("g"), g);
      labels.push([x, y, p.label, "lbl"]);
    }
    for (const [x, y, text, cls] of labels.sort((a, b) => a[0] - b[0])) layer.append(placeLabel(x, y, text, cls));
    return svg;
  }

  function viewSearch(d) {
    const tid = state.task[d.id] && d.tasks.some(t => t.id === state.task[d.id]) ? state.task[d.id] : d.tasks[0].id;
    const task = d.tasks.find(t => t.id === tid);
    const hasSeeds = task.picks.some(p => d.seeds && d.seeds[p.seed_key]);
    const showSeeds = hasSeeds && store.get("showSeeds", false);
    const controls = h("div", { class: "controls" },
      h("div", {}, h("span", { class: "ctl-label" }, "Task"),
        h("div", { class: "seg" }, d.tasks.map(t => h("button", {
          type: "button", "aria-pressed": String(t.id === tid),
          onclick: () => { state.task[d.id] = t.id; store.set("task", state.task); render(); },
        }, t.label)))),
      hasSeeds ? h("label", { class: "range" }, h("input", {
        type: "checkbox", checked: showSeeds, onchange: e => { store.set("showSeeds", e.target.checked); render(); },
      }), "show five-seed spread") : null);

    const legend = h("div", { class: "legend" },
      h("span", {}, h("i", { class: "sw", style: "background:var(--acc)" }), "taken to the board"),
      task.fronts.map((fr, i) => h("span", {}, h("i", { class: "sw", style: `background:var(${i ? "--front-2" : "--front"})` }), fr.label)),
      task.refs.some(r => r.kind === "point") ? h("span", {}, h("i", { class: "sw sq", style: "background:var(--ref);transform:rotate(45deg)" }), "reference") : null,
      task.refs.some(r => r.kind === "line") ? h("span", {}, h("i", { class: "sw line", style: "background:var(--ref)" }), "published figure") : null,
      showSeeds ? h("span", {}, h("i", { class: "sw", style: "background:transparent;border:2px solid var(--ink-2)" }), "mean ± 1 std over seeds") : null);

    const table = () => simpleTable(["model", "search", "parameters", "int8 KB", task.metric.name],
      task.fronts.flatMap((fr, fi) => fr.rows.map(r => [r.model + (task.picks.some(p => p.front === fi && p.model === r.model) ? "  ★" : ""), fr.label, f.int(r.params), r.kb.toFixed(1), fmtMetric(task.metric.fmt, r.value)]))
        .sort((a, b) => +b[2].replace(/,/g, "") - +a[2].replace(/,/g, "")), [2, 3, 4]);
    return [controls, chartCard(task.long, "Each dot is one model the search saved, re-scored on the test set. Click a blue model for its measurements.",
      scatter(d, task, showSeeds), { table, after: legend })];
  }

  // ------------------------------------------------------------ hardware
  function variantRows(d) {
    const rows = [];
    for (const m of d.registry.models) for (const v of m.variants) {
      const b = v.boards[state.board];
      rows.push({ m, v, b });
    }
    return rows;
  }
  const metricText = v => {
    const k = Object.keys(v.metric)[0], val = v.metric[k];
    if (k === "acc") return f.pct(val);
    if (k === "rmse_reported") return "RMSE " + val.toFixed(2) + " s (reported)";
    return k.toUpperCase() + " " + val.toFixed(3) + " s";
  };
  const variantName = v => v.precision + (v.io === "int8" ? " · int8 I/O" : "");
  const roleColor = role => role === "searched" ? "var(--acc)" : "var(--ref)";

  function latencyPlot(d) {
    const rows = variantRows(d).filter(r => r.b && r.b.latency_ms != null).sort((a, b) => a.b.latency_ms - b.b.latency_ms);
    const missing = variantRows(d).filter(r => r.b && r.b.fits === false);
    if (!rows.length) return h("p", { class: "empty" }, "No measurements on this board.");
    const W = 760, rowH = 30, M = { l: 250, r: 130, t: 10, b: 36 }, H = M.t + M.b + rowH * (rows.length + missing.length);
    const lo = Math.min(...rows.map(r => r.b.latency_ms)) / 1.6, hi = Math.max(...rows.map(r => r.b.latency_ms)) * 1.6;
    const X = logScale(lo, hi, M.l, W - M.r);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Measured latency" });
    for (const t of X.ticks()) {
      svg.append(s("line", { class: "gridline", x1: X(t), x2: X(t), y1: M.t, y2: H - M.b }),
        s("text", { class: "ax-tick", x: X(t), y: H - M.b + 16, "text-anchor": "middle" }, tickMs(t)));
    }
    svg.append(s("text", { class: "ax-title", x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle" }, "one inference, milliseconds (log scale) · shorter is better"));
    rows.forEach((r, i) => {
      const y = M.t + rowH * i + rowH / 2, x = X(r.b.latency_ms), dist = state.speed / 3.6 * r.b.latency_ms / 1000;
      const shape = r.m.role === "transformer"
        ? s("path", { d: `M${x} ${y - 6} L${x + 6} ${y + 5} L${x - 6} ${y + 5} Z`, fill: "var(--ref)", stroke: "var(--card)", "stroke-width": 2 })
        : r.m.role === "reference"
          ? s("rect", { x: x - 5, y: y - 5, width: 10, height: 10, transform: `rotate(45 ${x} ${y})`, fill: "var(--ref)", stroke: "var(--card)", "stroke-width": 2 })
          : s("circle", { cx: x, cy: y, r: 6, fill: roleColor(r.m.role), stroke: "var(--card)", "stroke-width": 2 });
      const g = s("g", { class: "mk", style: "cursor:pointer", onclick: () => openModel(d, r.m.id) },
        s("line", { x1: M.l, x2: x, y1: y, y2: y, stroke: "var(--grid)", "stroke-width": 1 }),
        shape, s("rect", { class: "hit", x: 0, y: y - rowH / 2, width: W, height: rowH }));
      hover(g, () => tipRows(r.m.label + " · " + variantName(r.v), f.ms(r.b.latency_ms), [
        ["accuracy / error", metricText(r.v)], ["flash", f.bytes(r.b.flash_b)], ["RAM", f.bytes(r.b.ram_b)],
        ["road at " + state.speed + " km/h", f.dist(dist)], ["", "click for details"]]));
      svg.append(g,
        s("text", { class: "lbl", x: M.l - 12, y: y - 1, "text-anchor": "end" }, r.m.label),
        s("text", { class: "lbl-m", x: M.l - 12, y: y + 11, "text-anchor": "end" }, variantName(r.v) + " · " + f.params(r.m.params)),
        s("text", { class: "lbl-2", x: x + 12, y: y + 4 }, f.ms(r.b.latency_ms) + "  ·  " + f.dist(dist)));
    });
    missing.forEach((r, j) => {
      const y = M.t + rowH * (rows.length + j) + rowH / 2;
      const g = s("g", {}, s("text", { class: "lbl", x: M.l - 12, y: y - 1, "text-anchor": "end" }, r.m.label),
        s("text", { class: "lbl-m", x: M.l - 12, y: y + 11, "text-anchor": "end" }, variantName(r.v) + " · " + f.params(r.m.params)),
        s("text", { x: M.l, y: y + 4, fill: "var(--bad)", "font-size": 12, "font-weight": 600 }, "✕ does not run on this board"),
        s("rect", { class: "hit", x: 0, y: y - rowH / 2, width: W, height: rowH }));
      hover(g, () => tipRows(r.m.label + " · " + variantName(r.v), "does not run", [["why", r.b.why]]));
      svg.append(g);
    });
    return svg;
  }

  function fitMeters(d) {
    const cap = d.registry.boards[state.board];
    const seen = new Map();
    for (const m of d.registry.models) for (const v of m.variants) {
      const b = v.boards[state.board];
      if (!b) continue;
      const flash = b.flash_b != null ? b.flash_b : (b.fits === false && m.variants[0].boards["STM32H7B3I-DK"] ? m.variants[0].boards["STM32H7B3I-DK"].flash_b : null);
      if (flash == null) continue;
      const key = m.id + v.precision + v.io;
      if (!seen.has(key)) seen.set(key, { m, v, flash, ram: b.ram_b, fits: b.fits !== false });
    }
    const rows = [...seen.values()].sort((a, b) => b.flash - a.flash);
    const meter = (r, used, total, fits, sub) => {
      const over = used > total || !fits;
      const el = h("div", { class: "meter" + (over ? " over" : "") },
        h("div", { class: "m-name" }, h("div", { class: "m-model" }, r.m.label), h("div", { class: "m-build" }, variantName(r.v))),
        h("div", { class: "track" }, h("div", { class: "fill", style: `width:${Math.min(100, used / total * 100)}%` })),
        h("div", { class: "m-val" }, over ? h("span", { class: "no" }, "✕ " + f.bytes(used)) : f.bytes(used) + " · " + (used / total * 100).toFixed(used / total < 0.1 ? 1 : 0) + "%"));
      el.title = sub;
      return el;
    };
    return h("div", { class: "grid g2" },
      h("div", {}, h("div", { class: "section-title", style: "margin-top:0" }, "Flash, out of " + f.bytes(cap.flash_b)),
        rows.map(r => meter(r, r.flash, cap.flash_b, r.fits, r.v.source))),
      h("div", {}, h("div", { class: "section-title", style: "margin-top:0" }, "RAM, out of " + f.bytes(cap.ram_b)),
        rows.filter(r => r.ram != null).map(r => meter(r, r.ram, cap.ram_b, r.ram <= cap.ram_b, r.v.source))));
  }

  function hardwareTable(d) {
    const rows = [];
    for (const m of d.registry.models) for (const v of m.variants) {
      const b1 = v.boards["STM32H7B3I-DK"] || {}, b2 = v.boards["NUCLEO-F401RE"] || {};
      if (!Object.keys(v.boards).length) continue;
      rows.push({ m, v, b1, b2 });
    }
    const cell = (b, key) => b.fits === false ? h("span", { class: "no" }, "✕ no") : b[key] != null ? (key === "latency_ms" ? f.ms(b[key]) : f.bytes(b[key])) : h("span", { class: "dash" }, "–");
    let sortKey = store.get("hwSort", "lat1"), dir = 1;
    const val = (r, k) => k === "params" ? r.m.params : k === "lat1" ? (r.b1.latency_ms ?? 1e9) : k === "lat2" ? (r.b2.latency_ms ?? 1e9) : k === "flash" ? (r.b1.flash_b ?? r.b2.flash_b ?? 1e12) : 0;
    const wrap = h("div", { class: "tbl-wrap" });
    const draw = () => {
      const sorted = [...rows].sort((a, b) => dir * (val(a, sortKey) - val(b, sortKey)));
      const th = (label, key, num) => h("th", { class: num ? "num" : null, "data-sort": key || null, onclick: key ? () => { dir = sortKey === key ? -dir : 1; sortKey = key; store.set("hwSort", key); draw(); } : null },
        label + (key === sortKey ? (dir > 0 ? " ↑" : " ↓") : ""));
      wrap.replaceChildren(h("table", { class: "tbl" },
        h("thead", {}, h("tr", {}, th("Model"), th("Build"), th("Accuracy / error"), th("Params", "params", true), th("M7 latency", "lat1", true), th("M4 latency", "lat2", true), th("Flash", "flash", true), th("RAM", null, true))),
        h("tbody", {}, sorted.map(r => h("tr", { class: "click", onclick: () => openModel(d, r.m.id) },
          h("td", { class: "nowrap" }, h("span", { class: "role" }, h("i", { class: "sw", style: `background:${roleColor(r.m.role)}` }), r.m.label)),
          h("td", { class: "nowrap" }, variantName(r.v)), h("td", { class: "nowrap" }, metricText(r.v)), h("td", { class: "num" }, f.params(r.m.params)),
          h("td", { class: "num" }, cell(r.b1, "latency_ms")), h("td", { class: "num" }, cell(r.b2, "latency_ms")),
          h("td", { class: "num" }, r.b1.flash_b != null ? f.bytes(r.b1.flash_b) : r.b2.flash_b != null ? f.bytes(r.b2.flash_b) : h("span", { class: "dash" }, "–")),
          h("td", { class: "num" }, r.b1.ram_b != null ? f.bytes(r.b1.ram_b) : r.b2.ram_b != null ? f.bytes(r.b2.ram_b) : h("span", { class: "dash" }, "–")))))));
    };
    draw();
    return wrap;
  }

  function viewHardware(d) {
    const controls = h("div", { class: "controls" },
      h("div", {}, h("span", { class: "ctl-label" }, "Board"),
        h("div", { class: "seg" }, BOARDS.map(b => h("button", {
          type: "button", "aria-pressed": String(b === state.board),
          onclick: () => { state.board = b; store.set("board", b); render(); },
        }, BOARD_SHORT[b])))),
      h("label", { class: "range" }, "Speed",
        h("input", { type: "range", min: 30, max: 150, step: 10, value: state.speed,
          oninput: e => { state.speed = +e.target.value; e.target.nextSibling.textContent = state.speed + " km/h"; },
          onchange: e => { store.set("speed", state.speed); render(); } }),
        h("span", {}, state.speed + " km/h")));
    const legend = h("div", { class: "legend" },
      h("span", {}, h("i", { class: "sw", style: "background:var(--acc)" }), "searched"),
      h("span", {}, h("i", { class: "sw sq", style: "background:var(--ref);transform:rotate(45deg)" }), "reference model"),
      d.registry.models.some(m => m.role === "transformer") ? h("span", {}, h("i", { class: "sw", style: "background:var(--ref);clip-path:polygon(50% 0,100% 100%,0 100%);border-radius:0" }), "reference Transformer") : null);
    return [controls,
      chartCard("Latency on " + BOARD_SHORT[state.board],
        "Measured with ST Edge AI Core 4.0.1. The distance is how far the car travels while one answer is computed.",
        latencyPlot(d), { after: legend }),
      h("div", { style: "height:16px" }),
      chartCard("Does it fit on " + BOARD_SHORT[state.board] + "?", "Footprint reported by the toolchain, against the board's memory.", fitMeters(d)),
      h("div", { style: "height:16px" }),
      chartCard("Every measured build", "Both boards side by side. Click a row for the architecture; click a column to sort.", hardwareTable(d))];
  }

  // ------------------------------------------------------------ quantization
  const STEP_STYLE = {
    "fp32": { label: "float32", mark: (x, y) => s("circle", { cx: x, cy: y, r: 6, fill: "var(--card)", stroke: "var(--ink-2)", "stroke-width": 2 }) },
    "int8 PTQ": { label: "int8, post-training", mark: (x, y) => s("rect", { x: x - 5, y: y - 5, width: 10, height: 10, rx: 2, fill: "var(--ref)", stroke: "var(--card)", "stroke-width": 2 }) },
    "int8 QAT": { label: "int8, quantization-aware", mark: (x, y) => s("circle", { cx: x, cy: y, r: 6, fill: "var(--acc)", stroke: "var(--card)", "stroke-width": 2 }) },
  };
  function dumbbell(items, title) {
    const W = 520, rowH = 44, M = { l: 180, r: 30, t: 10, b: 38 }, H = M.t + M.b + rowH * items.length;
    const vals = items.flatMap(it => it.steps.map(st => st[1]));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.12 || 0.01;
    lo -= pad; hi += pad;
    const X = linScale(lo, hi, M.l, W - M.r);
    const it0 = items[0];
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title });
    for (const t of X.ticks(5)) svg.append(s("line", { class: "gridline", x1: X(t), x2: X(t), y1: M.t, y2: H - M.b }),
      s("text", { class: "ax-tick", x: X(t), y: H - M.b + 16, "text-anchor": "middle" }, it0.fmt === "pct" ? (t * 100).toFixed(0) + "%" : t.toFixed(2)));
    svg.append(s("text", { class: "ax-title", x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle" }, title + (it0.better === "high" ? " · right is better" : " · left is better")));
    items.forEach((it, i) => {
      const y = M.t + rowH * i + rowH / 2, xs = it.steps.map(st => X(st[1]));
      svg.append(s("line", { x1: Math.min(...xs), x2: Math.max(...xs), y1: y, y2: y, stroke: "var(--axis)", "stroke-width": 2, "stroke-linecap": "round" }),
        s("text", { class: "lbl", x: M.l - 12, y: y - 1, "text-anchor": "end" }, it.model),
        s("text", { class: "lbl-m", x: M.l - 12, y: y + 12, "text-anchor": "end" }, f.params(it.params) + " params"));
      for (const [name, v] of it.steps) {
        const g = s("g", { class: "mk" }, STEP_STYLE[name].mark(X(v), y), s("circle", { class: "hit", cx: X(v), cy: y, r: 14 }));
        const base = it.steps[0][1], delta = v - base;
        hover(g, () => tipRows(it.model + " · " + STEP_STYLE[name].label, fmtMetric(it.fmt, v), name === "fp32" ? [] :
          [["change vs float32", it.fmt === "pct" ? (delta >= 0 ? "+" : "") + (delta * 100).toFixed(2) + " pt" : (delta >= 0 ? "+" : "") + (delta / base * 100).toFixed(0) + "%"]]));
        svg.append(g);
      }
    });
    return svg;
  }
  function viewQuant(d) {
    const cls = d.quant.filter(q => q.fmt === "pct"), reg = d.quant.filter(q => q.fmt !== "pct");
    const legend = h("div", { class: "legend" }, Object.values(STEP_STYLE).map(st => {
      const sv = s("svg", { width: 14, height: 14, viewBox: "0 0 14 14" }, st.mark(7, 7));
      return h("span", {}, sv, st.label);
    }));
    const table = items => () => simpleTable(["model", "float32", "int8 PTQ", "int8 QAT"],
      items.map(q => [q.model, ...["fp32", "int8 PTQ", "int8 QAT"].map(k => { const st = q.steps.find(x => x[0] === k); return st ? fmtMetric(q.fmt, st[1]) : "–"; })]), [1, 2, 3]);
    return [h("div", { class: "grid g2" },
      cls.length ? chartCard("Classifiers", "Accuracy before and after conversion to 8-bit integers.", dumbbell(cls, "test accuracy"), { table: table(cls), after: legend }) : null,
      reg.length ? chartCard("Time-to-change regressors", "Mean absolute error in seconds before and after conversion.", dumbbell(reg, "MAE (s)"), { table: table(reg), after: legend }) : null)];
  }

  // ------------------------------------------------------------ seeds (five retrainings per model)
  function stripPlot(d, rows, metric, fmt, better) {
    const W = 760, rowH = 46, M = { l: 250, r: 40, t: 10, b: 38 }, H = M.t + M.b + rowH * rows.length;
    const vals = rows.flatMap(r => [...r.runs.map(x => x[metric]), r.orig ?? r.runs[0][metric]]).filter(v => v != null);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.12 || 0.01;
    lo -= pad; hi += pad;
    const X = linScale(lo, hi, M.l, W - M.r);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Seed spread" });
    for (const t of X.ticks(6)) svg.append(s("line", { class: "gridline", x1: X(t), x2: X(t), y1: M.t, y2: H - M.b }),
      s("text", { class: "ax-tick", x: X(t), y: H - M.b + 16, "text-anchor": "middle" }, fmt === "pct" ? (t * 100).toFixed(1) + "%" : t.toFixed(3)));
    svg.append(s("text", { class: "ax-title", x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle" },
      (fmt === "pct" ? "test accuracy" : "test RMSE (s)") + (better === "high" ? " · right is better" : " · left is better")));
    rows.forEach((r, i) => {
      const y = M.t + rowH * i + rowH / 2, st = r.stats;
      const col = r.color ? `var(--${r.color})` : r.role === "searched" ? "var(--acc)" : "var(--ref)";
      svg.append(s("text", { class: "lbl", x: M.l - 12, y: y - 2, "text-anchor": "end" }, r.label),
        s("text", { class: "lbl-m", x: M.l - 12, y: y + 11, "text-anchor": "end" }, r.recipe + " · " + st.n + " seeds"));
      svg.append(s("rect", { x: X(st.mean - st.std), width: Math.max(2, X(st.mean + st.std) - X(st.mean - st.std)), y: y - 9, height: 18, rx: 4,
        fill: col, opacity: 0.14 }),
        s("line", { x1: X(st.mean), x2: X(st.mean), y1: y - 11, y2: y + 11, stroke: col, "stroke-width": 2 }));
      r.runs.forEach((run, k) => {
        const v = run[metric];
        if (v == null) return;
        const g = s("g", { class: "mk" }, s("circle", { cx: X(v), cy: y + ((k % 2) ? 4 : -4), r: 4.5, fill: col, stroke: "var(--card)", "stroke-width": 2 }),
          s("circle", { class: "hit", cx: X(v), cy: y, r: 12 }));
        hover(g, () => tipRows(r.label + " · seed " + run.seed, fmtMetric(fmt, v), run.epochs ? [["epochs trained", run.epochs]] : []));
        svg.append(g);
      });
      if (r.orig != null) {
        const x = X(r.orig);
        const g = s("g", { class: "mk" }, s("rect", { x: x - 5, y: y - 5, width: 10, height: 10, transform: `rotate(45 ${x} ${y})`, fill: "var(--card)", stroke: "var(--ink)", "stroke-width": 1.5 }),
          s("circle", { class: "hit", cx: x, cy: y, r: 12 }));
        hover(g, () => tipRows(r.label + " · the run reported so far", fmtMetric(fmt, r.orig), [["5-seed mean", fmtMetric(fmt, st.mean) + " ± " + (fmt === "pct" ? (st.std * 100).toFixed(2) + " pt" : st.std.toFixed(3))]]));
        svg.append(g);
      }
    });
    return svg;
  }
  function viewRobust(d) {
    const groups = [];
    const push = (pool, key, label, role, recipeLabel, metricKey) => {
      const g = pool && pool[key];
      if (!g || !g.stats[metricKey]) return null;
      return { label, role, recipe: recipeLabel, runs: g.runs, stats: g.stats[metricKey], orig: g.original && g.original[metricKey] };
    };
    const acc = [], rm = [];
    const regTasks = d.tasks.filter(t => t.metric.key !== "acc");
    for (const t of d.tasks) {
      const target = t.metric.key === "acc" ? acc : rm;
      const name = lbl => target === rm && regTasks.length > 1 ? t.label + " · " + lbl : lbl;
      const recipeOf = (pool, key) => (pool && pool[key] && pool[key].recipe_short) || "search recipe";
      const finalLabel = d.final_label || "hand-designed recipe";
      for (const p of t.picks) {
        target.push(push(d.seeds, p.seed_key, name(p.label), "searched", recipeOf(d.seeds, p.seed_key), t.metric.key));
        if (d.seeds_final) target.push(push(d.seeds_final, p.seed_key, name(p.label), "searched", finalLabel, t.metric.key));
      }
      for (const rf of t.refs.filter(r => r.seed_key)) {
        const row = push(d.seeds, rf.seed_key, name(rf.label), "reference", recipeOf(d.seeds, rf.seed_key), t.metric.key);
        if (row && row.orig == null) row.orig = rf.value;
        target.push(row);
        if (d.seeds_final && d.seeds_final[rf.seed_key]) target.push(push(d.seeds_final, rf.seed_key, name(rf.label), "reference", finalLabel, t.metric.key));
      }
    }
    const clean = arr => arr.filter(Boolean);
    const A = clean(acc), R = clean(rm);
    if (!A.length && !R.length) return [h("section", { class: "card" }, h("p", { class: "empty" }, "Seed runs for this dataset have not finished yet. Rebuild the page when they do."))];
    const legend = () => h("div", { class: "legend" },
      h("span", {}, h("i", { class: "sw", style: "background:var(--acc)" }), "searched, one dot per seed"),
      h("span", {}, h("i", { class: "sw", style: "background:var(--ref)" }), "reference / hand-designed"),
      h("span", {}, h("i", { class: "sw sq", style: "background:var(--acc);opacity:.25" }), "mean ± 1 std"),
      d.tasks.some(t => t.refs.some(r => r.value != null)) || Object.values(d.seeds || {}).some(g => g.original && Object.keys(g.original).length)
        ? h("span", {}, h("i", { class: "sw sq", style: "border:1.5px solid var(--ink);transform:rotate(45deg);width:9px;height:9px" }), "the single run reported so far") : null);
    const table = (rows, pct) => () => simpleTable(["model", "recipe", "seeds", "mean", "std", "min", "max", "reported run"],
      rows.map(r => {
        const v = x => pct ? f.pct(x) : x.toFixed(3);
        return [r.label, r.recipe, r.stats.n, v(r.stats.mean), pct ? (r.stats.std * 100).toFixed(2) + " pt" : r.stats.std.toFixed(3),
          v(r.stats.min), v(r.stats.max), r.orig != null ? v(r.orig) : "–"];
      }), [2, 3, 4, 5, 6, 7]);
    const out = [];
    const anyOrig = rows => rows.some(r => r.orig != null);
    if (A.length) out.push(chartCard("Classifiers across seeds", "Each architecture retrained from scratch five times." + (anyOrig(A) ? " Where the diamond sits outside the band, the reported number was a favourable run." : ""),
      stripPlot(d, A, "acc", "pct", "high"), { table: table(A, true), after: legend() }));
    if (R.length) { out.push(h("div", { style: "height:16px" })); out.push(chartCard("Regressors across seeds", (regTasks.length === 1 ? regTasks[0].label + ": test" : "Test") + " RMSE in seconds.", stripPlot(d, R, "rmse", "s3", "low"), { table: table(R, false), after: legend() })); }
    const RR = d.rerank;
    if (RR) {
      const rows = RR.rows.map(g => ({ label: f.int(g.params) + " params", role: "searched", recipe: g.search, runs: g.runs, stats: g.stats, orig: g.orig }));
      const ref = d.rerank_ref && d.seeds[d.rerank_ref];
      const refPoint = d.tasks.flatMap(t => t.refs).find(r => r.seed_key === d.rerank_ref);
      if (ref && ref.stats.acc) rows.unshift({ label: "hand-designed CNN", role: "reference", recipe: ref.recipe_short, runs: ref.runs, stats: ref.stats.acc, orig: refPoint ? refPoint.value : null });
      out.push(h("div", { style: "height:16px" }));
      out.push(chartCard("Every saved classifier, retrained",
        RR.n + " models from both searches, five seeds each, smallest at the top. The single run did not predict the average: rank correlation " +
        RR.rho.toFixed(2) + ", and the best single run ranks " + RR.best_single_rank + " of " + RR.n + " by average.",
        stripPlot(d, rows, "acc", "pct", "high"), { table: table(rows, true), after: legend() }));
    }
    return out;
  }

  // ------------------------------------------------------------ transfer (exiD)
  const METHOD_KEY = { "scratch": "trained on exiD", "zero-shot": "highD model as is", "fine-tune": "highD, fine-tuned" };
  function fractionChart(t) {
    const W = 480, H = 300, M = { l: 58, r: 20, t: 18, b: 46 };
    const xs = [0.1, 0.25, 1.0], X = logScale(0.08, 1.25, M.l, W - M.r);
    const pts = [];
    for (const [mid, list] of Object.entries(t.fractions)) for (const p of list) if (p.stats) pts.push(p.stats.mean - p.stats.std, p.stats.mean + p.stats.std);
    const zero = t.methods.find(m => m.id === "zero-shot").stats;
    if (zero) pts.push(zero.mean);
    if (!pts.length) return h("p", { class: "empty" }, "Runs still in progress.");
    let lo = Math.min(...pts), hi = Math.max(...pts);
    const pad = (hi - lo) * 0.1 || 0.01; lo -= pad; hi += pad;
    const Y = linScale(lo, hi, H - M.b, M.t);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Accuracy against share of exiD training data" });
    for (const v of Y.ticks(5)) svg.append(s("line", { class: "gridline", x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v) }),
      s("text", { class: "ax-tick", x: M.l - 8, y: Y(v) + 4, "text-anchor": "end" }, t.fmt === "pct" ? (v * 100).toFixed(0) + "%" : v.toFixed(2)));
    for (const x of xs) svg.append(s("text", { class: "ax-tick", x: X(x), y: H - M.b + 18, "text-anchor": "middle" }, (x * 100) + "%"));
    svg.append(s("line", { class: "baseline", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }),
      s("text", { class: "ax-title", x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle" }, "share of the exiD training scenarios (log scale)"),
      s("text", { class: "ax-title", x: 14, y: (M.t + H - M.b) / 2, transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})`, "text-anchor": "middle" },
        (t.fmt === "pct" ? "exiD test accuracy" : "exiD test RMSE (s)") + (t.better === "high" ? "  ↑ better" : "  ↓ better")));
    if (zero) {
      const up = t.better === "high" ? -6 : 16;
      svg.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(zero.mean), y2: Y(zero.mean), stroke: "var(--s2)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }),
        s("text", { class: "lbl-m", x: W - M.r, y: Y(zero.mean) + up, "text-anchor": "end" }, "highD model as is, no exiD data: " + fmtMetric(t.fmt, zero.mean)));
    }
    const firsts = {};
    for (const [mid, color] of [["scratch", "s3"], ["fine-tune", "s1"]]) {
      const list = (t.fractions[mid] || []).filter(p => p.stats);
      if (!list.length) continue;
      svg.append(s("polyline", { points: list.map(p => `${X(p.fraction)},${Y(p.stats.mean)}`).join(" "), fill: "none", stroke: `var(--${color})`, "stroke-width": 2 }));
      for (const p of list) {
        const x = X(p.fraction), y = Y(p.stats.mean);
        const g = s("g", { class: "mk" },
          s("line", { x1: x, x2: x, y1: Y(p.stats.mean - p.stats.std), y2: Y(p.stats.mean + p.stats.std), stroke: `var(--${color})`, "stroke-width": 2 }),
          s("circle", { cx: x, cy: y, r: 5, fill: `var(--${color})`, stroke: "var(--card)", "stroke-width": 2 }),
          s("circle", { class: "hit", cx: x, cy: y, r: 13 }));
        hover(g, () => tipRows(METHOD_KEY[mid] + " · " + (p.fraction * 100) + "% of exiD", fmtMetric(t.fmt, p.stats.mean) + " ± " + (t.fmt === "pct" ? (p.stats.std * 100).toFixed(2) + " pt" : p.stats.std.toFixed(3)), [["seeds", p.stats.n]]));
        svg.append(g);
      }
      firsts[mid] = list[0];
    }
    if (firsts.scratch && firsts["fine-tune"]) {             // label both lines at 10%, the one above goes up
      const hi = Y(firsts.scratch.stats.mean) < Y(firsts["fine-tune"].stats.mean) ? "scratch" : "fine-tune";
      for (const mid of ["scratch", "fine-tune"]) {
        const p = firsts[mid];
        svg.append(s("text", { class: "lbl", x: X(p.fraction) + 8, y: Y(p.stats.mean) + (mid === hi ? -12 : 20) }, METHOD_KEY[mid]));
      }
    }
    return svg;
  }
  function kindChart(t, kinds) {
    const order = kinds.filter(k => Object.values(t.by_kind).some(m => m[k]));
    const methods = [["scratch", "s3"], ["zero-shot", "s2"], ["fine-tune", "s1"]].filter(([m]) => t.by_kind[m] && Object.keys(t.by_kind[m]).length);
    if (!order.length || !methods.length) return h("p", { class: "empty" }, "Runs still in progress.");
    const W = 480, rowH = 40, M = { l: 142, r: 20, t: 10, b: 40 }, H = M.t + M.b + rowH * order.length;
    const vals = methods.flatMap(([m]) => order.map(k => t.by_kind[m][k]).filter(Boolean).flatMap(st => [st.mean - st.std, st.mean + st.std]));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.08 || 0.01; lo -= pad; hi += pad;
    const X = linScale(lo, hi, M.l, W - M.r);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Result by lane-change kind" });
    for (const v of X.ticks(4)) svg.append(s("line", { class: "gridline", x1: X(v), x2: X(v), y1: M.t, y2: H - M.b }),
      s("text", { class: "ax-tick", x: X(v), y: H - M.b + 16, "text-anchor": "middle" }, t.fmt === "pct" ? (v * 100).toFixed(0) + "%" : v.toFixed(2)));
    svg.append(s("text", { class: "ax-title", x: (M.l + W - M.r) / 2, y: H - 4, "text-anchor": "middle" },
      (t.fmt === "pct" ? "exiD test accuracy" : "exiD test RMSE (s)") + (t.better === "high" ? " · right is better" : " · left is better")));
    order.forEach((k, i) => {
      const y0 = M.t + rowH * i + rowH / 2;
      svg.append(s("text", { class: "lbl", x: M.l - 12, y: y0 + 4, "text-anchor": "end" }, k));
      methods.forEach(([m, color], j) => {
        const st = t.by_kind[m][k];
        if (!st) return;
        const y = y0 + (j - (methods.length - 1) / 2) * 9, x = X(st.mean);
        const g = s("g", { class: "mk" },
          s("line", { x1: X(st.mean - st.std), x2: X(st.mean + st.std), y1: y, y2: y, stroke: `var(--${color})`, "stroke-width": 2 }),
          s("circle", { cx: x, cy: y, r: 4.5, fill: `var(--${color})`, stroke: "var(--card)", "stroke-width": 1.5 }),
          s("circle", { class: "hit", cx: x, cy: y, r: 11 }));
        hover(g, () => tipRows(k + " · " + METHOD_KEY[m], fmtMetric(t.fmt, st.mean) + " ± " + (t.fmt === "pct" ? (st.std * 100).toFixed(2) + " pt" : st.std.toFixed(3)), [["seeds", st.n]]));
        svg.append(g);
      });
    });
    return svg;
  }
  function viewTransfer(d) {
    const T = d.transfer, out = [];
    const legend = () => h("div", { class: "legend" },
      h("span", {}, h("i", { class: "sw", style: "background:var(--s3)" }), "trained on exiD"),
      h("span", {}, h("i", { class: "sw", style: "background:var(--s2)" }), "highD model, applied as is"),
      h("span", {}, h("i", { class: "sw", style: "background:var(--s1)" }), "highD model, fine-tuned on exiD"));
    for (const t of T.tasks) {
      const rows = t.methods.filter(m => m.stats).map(m => ({ label: m.label, recipe: m.id === "fine-tune" ? "same recipe, highD start" : m.id === "zero-shot" ? "no exiD training" : "hand-designed recipe",
        runs: m.runs, stats: m.stats, color: m.color, role: "reference" }));
      if (!rows.length) continue;
      const tbl = () => simpleTable(["model", "tested on", "seeds", "mean", "std"],
        t.methods.flatMap(m => [["exiD", m.stats], ["highD", t.on_highd[m.id]]].filter(([, st]) => st).map(([on, st]) =>
          [m.label, on, st.n, fmtMetric(t.fmt, st.mean), t.fmt === "pct" ? (st.std * 100).toFixed(2) + " pt" : st.std.toFixed(3)])), [2, 3, 4]);
      out.push(chartCard(t.label + ": does a highD model work on exiD?",
        "Hand-designed CNN (8.4 k parameters), five seeds each, exiD test set. The fine-tuned model starts from the highD weights and uses the same recipe.",
        stripPlot(d, rows, t.metric, t.fmt, t.better), { table: tbl, after: legend() }));
      out.push(h("div", { style: "height:16px" }));
      out.push(h("div", { class: "grid g2" },
        chartCard("With less exiD data", "Training on 10%, 25% and all exiD training scenarios; the same random subset for both.",
          fractionChart(t), { table: () => simpleTable(["model", "share of exiD", "seeds", "mean", "std"],
            Object.entries(t.fractions).flatMap(([mid, list]) => list.filter(p => p.stats).map(p => [METHOD_KEY[mid], (p.fraction * 100) + "%", p.stats.n,
              fmtMetric(t.fmt, p.stats.mean), t.fmt === "pct" ? (p.stats.std * 100).toFixed(2) + " pt" : p.stats.std.toFixed(3)])), [2, 3, 4]) }),
        chartCard("By kind of lane change", "exiD test scenarios grouped by what the vehicle does; mean ± std over five seeds.",
          kindChart(t, T.kinds), { after: legend(),  table: () => simpleTable(["kind", "model", "mean", "std"],
            T.kinds.flatMap(k => Object.entries(t.by_kind).filter(([, bk]) => bk[k]).map(([m, bk]) => [k, METHOD_KEY[m], fmtMetric(t.fmt, bk[k].mean),
              t.fmt === "pct" ? (bk[k].std * 100).toFixed(2) + " pt" : bk[k].std.toFixed(3)])), [2, 3]) })));
      out.push(h("div", { style: "height:16px" }));
    }
    if (T.searched_zero && T.searched_zero.length) {
      out.push(h("section", { class: "card" }, h("h2", {}, "Searched highD models, applied to exiD as deployed"),
        h("p", { class: "cap" }, "The deployed weights of the three models found by the highD searches, one run each."),
        simpleTable(["model", "parameters", "tested on", "result"], T.searched_zero.map(r => [r.model.replace("highd_", ""), f.int(r.params), r.eval_on,
          r.task === "cls" ? f.pct(r.metrics.acc) : "RMSE " + f.s3(r.metrics.rmse)]), [1])));
    }
    return out.length ? out : [h("section", { class: "card" }, h("p", { class: "empty" }, "Runs still in progress."))];
  }

  // ------------------------------------------------------------ benchmark
  function barsH(rows, fmt, better, W = 520) {
    const narrow = W < 400;
    const rowH = 34, M = { l: narrow ? 92 : 200, r: narrow ? 64 : 70, t: 6, b: 8 }, H = M.t + M.b + rowH * rows.length;
    const hi = Math.max(...rows.map(r => r[1])) * 1.05;
    const X = linScale(0, hi, M.l, W - M.r);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img" });
    rows.forEach(([label, v, role], i) => {
      const y = M.t + rowH * i, bh = Math.min(20, rowH - 12), w = Math.max(2, X(v) - M.l);
      const col = role === "ours" ? "var(--acc)" : role === "ours2" ? "color-mix(in srgb, var(--acc) 50%, var(--card))" : "var(--ref)";
      const g = s("g", { class: "mk" },
        s("path", { d: `M${M.l} ${y + (rowH - bh) / 2} h${w - 4} q4 0 4 4 v${bh - 8} q0 4 -4 4 h${-(w - 4)} z`, fill: col }),
        s("rect", { class: "hit", x: 0, y, width: W, height: rowH }));
      hover(g, () => tipRows(label, fmtMetric(fmt, v), []));
      svg.append(g, s("text", { class: "lbl-2", x: M.l - 10, y: y + rowH / 2 + 4, "text-anchor": "end" }, label),
        s("text", { class: "lbl", x: M.l + w + 8, y: y + rowH / 2 + 4 }, fmtMetric(fmt, v)));
    });
    svg.append(s("line", { class: "baseline", x1: M.l, x2: M.l, y1: M.t, y2: H - M.b }));
    return svg;
  }
  function viewBench(d) {
    const B = d.bench, out = [];
    if (d.id === "dmir") {
      out.push(h("div", { class: "grid g2" },
        B.regression.map(g => chartCard("Time to " + g.task.toLowerCase() + " lane change", "Test RMSE in seconds · shorter is better · ours: five-seed means",
          barsH(g.rows.map(r => [r[0], r[1], r[2] === "ours" || r[2] === "ours2" ? r[2] : "ref"]), "s3", "low"),
          { table: () => simpleTable(["model", "RMSE (s)"], g.rows.map(r => [r[0], r[1].toFixed(3)]), [1]) }))));
      out.push(h("div", { style: "height:16px" }));
      out.push(h("div", { class: "grid g2" },
        chartCard("Does it just read the turn signal?", "Test accuracy of the three-class task with different inputs · five-seed means",
          barsH(B.ablation.map((r, i) => [r[0], r[1], i === 0 ? "ref" : i === 1 ? "ours2" : "ours"]), "pct", "high"),
          { table: () => simpleTable(["input", "accuracy"], B.ablation.map(r => [r[0], f.pct(r[1])]), [1]) }),
        h("section", { class: "card" }, h("h2", {}, "Read before comparing"), B.notes.map(n => h("p", { class: "note" }, "• " + n)))));
    } else {
      out.push(h("div", { class: "grid g3" }, B.metrics.map(m => chartCard(m.name, m.better === "high" ? "higher is better" : "lower is better",
        barsH(m.rows.map(r => [r[0], r[1], r[0] === "searched" ? "ours" : r[0] === "hand-designed" ? "ours2" : "ref"]), m.fmt, m.better, 340)))));
      out.push(h("div", { style: "height:16px" }));
      const sp = B.splits;
      out.push(h("div", { class: "grid g2" },
        h("section", { class: "card" }, h("h2", {}, "Protocol check"), h("p", { class: "cap" }, "Scenario counts after our preprocessing, against the paper"),
          simpleTable(["split", "ours", "paper", ""], ["train", "val", "test"].map(k => [k, f.int(sp[k][0]), f.int(sp[k][1]),
            sp[k][0] === sp[k][1] ? h("span", { class: "yes" }, "✓ exact") : h("span", { class: "dash" }, (sp[k][0] - sp[k][1]) + " (" + ((sp[k][0] - sp[k][1]) / sp[k][1] * 100).toFixed(1) + "%)")]), [1, 2])),
        h("section", { class: "card" }, h("h2", {}, "Read before comparing"), B.notes.map(n => h("p", { class: "note" }, "• " + n)))));
    }
    return out;
  }

  // ------------------------------------------------------------ scenario
  function viewScenario(d) {
    if (!d.media || !d.media.length) return [h("section", { class: "card" }, h("p", { class: "empty" }, "No scenario replay for this dataset yet."))];
    const key = "media:" + d.id;
    const cur = d.media.find(m => m.id === store.get(key)) || d.media[0];
    const controls = d.media.length > 1 ? h("div", { class: "controls" }, h("div", { class: "seg" }, d.media.map(m => h("button", {
      type: "button", "aria-pressed": String(m.id === cur.id), onclick: () => { store.set(key, m.id); render(); },
    }, m.label)))) : null;
    return [controls, h("section", { class: "card media" },
      h("div", { class: "card-head" }, h("div", {}, h("h2", {}, cur.label), h("p", { class: "cap" }, cur.caption || ""))),
      cur.withheld ? h("div", { class: "withheld" }, cur.withheld) : h("img", { src: cur.src, alt: cur.label + " (animated replay)" }))];
  }

  // ------------------------------------------------------------ model drawer
  const drawer = document.getElementById("drawer"), scrim = document.getElementById("scrim");
  function closeDrawer() { drawer.hidden = true; scrim.hidden = true; }
  scrim.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeDrawer(); });
  function openModel(d, id) {
    const m = d.registry.models.find(x => x.id === id);
    if (!m) return;
    hideTip();
    const rows = m.variants.map(v => {
      const b1 = v.boards["STM32H7B3I-DK"] || {}, b2 = v.boards["NUCLEO-F401RE"] || {};
      const lat = b => b.fits === false ? h("span", { class: "no" }, "✕") : b.latency_ms != null ? f.ms(b.latency_ms) : "–";
      return [variantName(v), metricText(v), lat(b1), lat(b2), f.bytes(b1.flash_b ?? b2.flash_b), f.bytes(b1.ram_b ?? b2.ram_b)];
    });
    const why = m.variants.flatMap(v => Object.entries(v.boards).filter(([, b]) => b.fits === false).map(([board, b]) => h("p", { class: "note" }, "✕ " + BOARD_SHORT[board] + ": " + b.why)));
    const arch = m.arch_svg_markup ? (() => { const box = h("div", { class: "arch" }); box.innerHTML = m.arch_svg_markup; return box; })() : null;
    drawer.replaceChildren(...[
      h("button", { class: "icon-btn close", type: "button", "aria-label": "Close", onclick: closeDrawer }, "✕"),
      h("div", { class: "eyebrow" }, d.name + " · " + ({ searched: "found by the search", reference: "reference model", transformer: "reference Transformer", baseline: "hand-designed baseline" }[m.role] || m.role)),
      h("h3", {}, m.label),
      h("p", { class: "cap" }, f.int(m.params) + " parameters"),
      m.note ? h("p", { class: "note" }, m.note) : null,
      h("div", { class: "section-title" }, "Measured builds"),
      simpleTable(["build", "accuracy / error", "M7", "M4", "flash", "RAM"], rows, [2, 3, 4, 5]),
      why,
      h("div", { class: "section-title" }, "Sources"),
      m.variants.map(v => h("p", { class: "note" }, variantName(v) + ": " + v.source)),
      arch ? [h("div", { class: "section-title" }, "Architecture"), arch] : null].flat(Infinity).filter(Boolean));
    drawer.hidden = false; scrim.hidden = false;
    drawer.focus && drawer.setAttribute("tabindex", "-1");
    drawer.focus();
  }

  // ------------------------------------------------------------ compare
  function viewCompare() {
    const live = D.datasets.filter(d => d.status !== "planned");
    const W = 760, H = 400, M = { l: 64, r: 30, t: 16, b: 44 };
    const pts = [];
    for (const d of live) for (const m of (d.registry || { models: [] }).models) for (const v of m.variants) {
      const b = v.boards["STM32H7B3I-DK"];
      if (b && b.latency_ms != null && b.flash_b != null) pts.push({ d, m, v, b });
    }
    const X = logScale(Math.min(...pts.map(p => p.b.latency_ms)) / 1.8, Math.max(...pts.map(p => p.b.latency_ms)) * 1.8, M.l, W - M.r);
    const Y = logScale(Math.min(...pts.map(p => p.b.flash_b)) / 1.8, Math.max(...pts.map(p => p.b.flash_b)) * 1.8, H - M.b, M.t);
    const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Every measured build" });
    for (const t of Y.ticks(1024)) svg.append(s("line", { class: "gridline", x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t) }),
      s("text", { class: "ax-tick", x: M.l - 8, y: Y(t) + 4, "text-anchor": "end" }, f.bytes(t)));
    for (const t of X.ticks()) svg.append(s("text", { class: "ax-tick", x: X(t), y: H - M.b + 18, "text-anchor": "middle" }, tickMs(t)));
    svg.append(s("line", { class: "baseline", x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b }),
      s("text", { class: "ax-title", x: (M.l + W - M.r) / 2, y: H - 6, "text-anchor": "middle" }, "latency on the Cortex-M7 (log) · left is faster"),
      s("text", { class: "ax-title", x: 14, y: (M.t + H - M.b) / 2, transform: `rotate(-90 14 ${(M.t + H - M.b) / 2})`, "text-anchor": "middle" }, "flash (log) · lower is smaller"));
    const f401 = D.datasets[0].registry.boards["NUCLEO-F401RE"].flash_b;
    svg.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(f401), y2: Y(f401), stroke: "var(--bad)", "stroke-width": 1.5 }),
      s("text", { x: W - M.r - 4, y: Y(f401) - 6, "text-anchor": "end", fill: "var(--bad)", "font-size": 11.5, "font-weight": 600 }, "F401RE flash: 512 KB"));
    for (const p of pts) {
      const x = X(p.b.latency_ms), y = Y(p.b.flash_b), col = p.m.role === "searched" ? `var(--s${p.d.accent})` : "var(--ref)";
      const shape = p.m.role === "searched" ? s("circle", { cx: x, cy: y, r: 6, fill: col, stroke: "var(--card)", "stroke-width": 2 })
        : p.m.role === "transformer" ? s("path", { d: `M${x} ${y - 7} L${x + 7} ${y + 5} L${x - 7} ${y + 5} Z`, fill: col, stroke: "var(--card)", "stroke-width": 2 })
          : s("rect", { x: x - 5, y: y - 5, width: 10, height: 10, transform: `rotate(45 ${x} ${y})`, fill: col, stroke: "var(--card)", "stroke-width": 2 });
      const g = s("g", { class: "mk", style: "cursor:pointer", onclick: () => openModel(p.d, p.m.id) }, shape, s("circle", { class: "hit", cx: x, cy: y, r: 13 }));
      hover(g, () => tipRows(p.d.name + " · " + p.m.label + " · " + variantName(p.v), f.ms(p.b.latency_ms), [["flash", f.bytes(p.b.flash_b)], ["accuracy / error", metricText(p.v)]]));
      svg.append(g);
    }
    const legend = h("div", { class: "legend" },
      live.filter(d => d.registry).map(d => h("span", {}, h("i", { class: "sw", style: `background:var(--s${d.accent})` }), d.name + ", searched")),
      h("span", {}, h("i", { class: "sw sq", style: "background:var(--ref);transform:rotate(45deg)" }), "reference model"),
      h("span", {}, h("i", { class: "sw", style: "background:var(--ref);clip-path:polygon(50% 0,100% 100%,0 100%);border-radius:0" }), "reference Transformer"));
    const cmpRows = [
      ["Whose behaviour", ...D.datasets.map(d => d.facts[0][1])],
      ["Input", ...D.datasets.map(d => (d.facts.find(x => x[0] === "Input") || ["", "–"])[1])],
      ["Headline", ...D.datasets.map(d => d.headline.value + " " + d.headline.label)],
      ["Status", ...D.datasets.map(d => d.status)],
    ];
    return [h("section", { class: "card" }, h("h2", {}, "Three datasets, one pipeline"), h("p", { class: "cap" }, "Different prediction problems, one training and measurement chain."),
      h("div", { style: "height:10px" }), simpleTable(["", ...D.datasets.map(d => d.name)], cmpRows)),
    h("div", { style: "height:16px" }),
    chartCard("Everything measured on the boards", "Each mark is one build on the Cortex-M7. The red line is the Cortex-M4 board's total flash; a build just under it can still fail there, because the runtime needs room too. exiD models run on the highD builds (same graphs, other weights), so they add no marks.", svg, { after: legend })];
  }

  // ------------------------------------------------------------ status (local build only)
  function viewStatus() {
    const S = D.status;
    const col = (title, items, icon) => h("section", { class: "card" },
      h("div", { class: "col-h" }, icon, title, h("span", { class: "cnt" }, items.length)),
      items.map(it => h("div", { class: "item" + (it.priority ? " p-" + it.priority : "") },
        h("div", { class: "it" }, it.title), it.detail ? h("div", { class: "id" }, it.detail) : null, it.who ? h("span", { class: "who" }, it.who) : null)));
    return [h("section", { class: "card", style: "margin-bottom:16px" }, h("h2", {}, S.title), h("p", { class: "cap" }, S.summary)),
      h("div", { class: "board-cols" }, col("Done", S.done, "✓ "), col("Open on our side", S.ours, "◐ "), col("Needs someone else", S.others, "→ "))];
  }

  // ------------------------------------------------------------ render
  const app = document.getElementById("app");
  function render() {
    const r = route();
    renderHeader(r);
    hideTip();
    let nodes;
    if (r.page === "compare") nodes = viewCompare();
    else if (r.page === "status") nodes = viewStatus();
    else {
      const d = ds(r.ds);
      nodes = ({ overview: viewOverview, search: viewSearch, hardware: viewHardware, quant: viewQuant, transfer: viewTransfer, robust: viewRobust, bench: viewBench, scenario: viewScenario })[r.view](d);
    }
    app.replaceChildren(...[].concat(nodes).filter(Boolean));
    if (r.page === "ds" && r.model) openModel(ds(r.ds), r.model);
    document.title = (r.page === "ds" ? ds(r.ds).name + " · " : r.page === "compare" ? "Compare · " : "Status · ") + "Lane-Change-MCU results";
  }

  // theme toggle
  const themeBtn = document.getElementById("theme-btn");
  const urlTheme = new URLSearchParams(location.search).get("theme");
  const savedTheme = urlTheme === "light" || urlTheme === "dark" ? urlTheme : store.get("theme", null);
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  themeBtn.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    store.set("theme", document.documentElement.dataset.theme);
  });
  document.getElementById("foot").textContent = "Built " + D.built + " from the project's result files. " +
    (D.share ? "Shared copy." : "Local copy.") + " Board numbers: ST Edge AI Core 4.0.1, balanced optimization.";
  window.addEventListener("hashchange", render);
  render();
})();
