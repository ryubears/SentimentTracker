"""
Generate a self-contained HTML dashboard from the tracker db:
  - per-account weight trajectories (top 5 by current weight highlighted),
  - cumulative and per-period returns from following the aggregate score,
  - rolling Pearson/Spearman between score and realized return,
  - Sortino ratio and headline metrics.

Usage:
  python dashboard.py                 # reads config.yaml, writes dashboard.html
  python dashboard.py --out path.html
"""

from __future__ import annotations
from datetime import datetime, timezone
import argparse
import json
import sys
import numpy as np
import yaml

sys.path.insert(0, "src")
from sentiment_tracker import db
from sentiment_tracker.evaluate import metrics, rolling_corr, sortino, strategy_returns

PERIODS_PER_YEAR = {"1d": 365.0, "1h": 24 * 365.0}
ROLL_WINDOW = 14
TOP_N = 5

def build_payload(con, cfg: dict) -> dict:
    rows = con.execute("SELECT period_ts, agg_score, agg_uniform, realized_return, resolved "
                       "FROM periods ORDER BY period_ts").fetchall()
    resolved = [r for r in rows if r[4]]
    ts = [r[0] for r in resolved]
    agg = np.array([r[1] for r in resolved], dtype=float)
    unif = np.array([r[2] for r in resolved], dtype=float)
    ret = np.array([r[3] for r in resolved], dtype=float)

    deadband = cfg.get("signal", {}).get("deadband", 0.0)
    strat = strategy_returns(agg, ret, deadband)
    strat_u = strategy_returns(unif, ret, deadband)
    cum = np.cumprod(1 + strat) - 1
    cum_u = np.cumprod(1 + strat_u) - 1
    ppy = PERIODS_PER_YEAR[cfg["horizon"]]
    overall = metrics(agg, ret) if len(agg) > 1 else {}

    snaps = con.execute("SELECT period_ts, weights_json FROM weight_snapshots "
                        "ORDER BY period_ts").fetchall()
    wdates = [s[0] for s in snaps]
    parsed = [json.loads(s[1]) for s in snaps]
    accounts = sorted({a for w in parsed for a in w})
    series = {a: [w.get(a) for w in parsed] for a in accounts}
    last = {a: next((v for v in reversed(series[a]) if v is not None), 0.0) for a in accounts}
    top = sorted(accounts, key=lambda a: -last[a])[:TOP_N]

    return {
        "meta": {
            "symbol": cfg["symbol"], "horizon": cfg["horizon"],
            "n_resolved": len(resolved), "n_accounts": len(accounts),
            "span": [ts[0][:10], ts[-1][:10]] if ts else ["", ""],
            "window": ROLL_WINDOW, "deadband": deadband,
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        },
        "tiles": {
            "cum": float(cum[-1]) if len(cum) else None,
            "cum_uniform": float(cum_u[-1]) if len(cum_u) else None,
            "sortino": sortino(strat, ppy),
            "sortino_uniform": sortino(strat_u, ppy),
            "pearson": overall.get("pearson"),
            "spearman": overall.get("spearman"),
            "hit_rate": overall.get("hit_rate"),
        },
        "weights": {"dates": wdates, "series": series, "top": top,
                    "uniform": 1.0 / len(accounts) if accounts else None},
        "returns": {"dates": ts, "strategy": strat.tolist(),
                    "cum": cum.tolist(), "cum_uniform": cum_u.tolist()},
        "rolling": rolling_corr(agg, ret, ROLL_WINDOW) | {"dates": ts},
        "table": [[r[0], r[1], r[2], r[3], float(s)] for r, s in zip(resolved, strat)],
    }

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="dashboard.html")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    payload = build_payload(db.connect(cfg["db_path"]), cfg)
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    open(args.out, "w").write(html)
    print(f"wrote {args.out}: {payload['meta']['n_resolved']} resolved periods, "
          f"{payload['meta']['n_accounts']} accounts")


TEMPLATE = r"""<title>Sentiment vs BTC</title>
<style>
:root {
  color-scheme: light;
  --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
  --pos: #2a78d6; --neg: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
    --pos: #3987e5; --neg: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  --pos: #3987e5; --neg: #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1060px; margin: 0 auto; padding: 28px 20px 48px; }
header h1 { font-size: 22px; font-weight: 650; margin: 0 0 4px; }
header p { margin: 0; color: var(--ink2); font-size: 13.5px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
         gap: 12px; margin: 22px 0 18px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
        padding: 14px 16px 12px; }
.tile .k { font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
           color: var(--muted); margin-bottom: 6px; }
.tile .v { font-size: 26px; font-weight: 650; line-height: 1.1; }
.tile .sub { font-size: 12.5px; color: var(--ink2); margin-top: 5px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
        padding: 16px 18px 12px; margin-bottom: 16px; position: relative; }
.card h2 { font-size: 14.5px; font-weight: 650; margin: 0 0 2px; }
.card .sub { font-size: 12.5px; color: var(--ink2); margin: 0 0 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px;
          color: var(--ink2); margin: 2px 0 6px; }
.legend .chip { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
                margin-right: 5px; vertical-align: -1px; }
.chart { width: 100%; }
.chart svg { display: block; width: 100%; }
.tip { position: absolute; pointer-events: none; background: var(--surface);
       border: 1px solid var(--border); border-radius: 6px; padding: 7px 10px;
       font-size: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.12); display: none;
       z-index: 2; min-width: 130px; }
.tip .d { color: var(--muted); margin-bottom: 3px; }
.tip .row { display: flex; justify-content: space-between; gap: 12px; }
.tip .row span:last-child { font-variant-numeric: tabular-nums; }
details { margin-top: 20px; }
summary { cursor: pointer; font-size: 13.5px; color: var(--ink2); }
summary:focus-visible { outline: 2px solid var(--s1); outline-offset: 2px; }
.tablebox { overflow-x: auto; margin-top: 10px; }
table { border-collapse: collapse; font-size: 12.5px; width: 100%; }
th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--grid);
         font-variant-numeric: tabular-nums; white-space: nowrap; }
th { color: var(--muted); font-weight: 550; }
th:first-child, td:first-child { text-align: left; }
footer { margin-top: 26px; font-size: 12px; color: var(--muted); }
</style>
<div class="wrap">
  <header>
    <h1>Sentiment vs BTC</h1>
    <p id="subtitle"></p>
  </header>
  <div class="tiles" id="tiles"></div>
  <div class="card">
    <h2>Account weights</h2>
    <p class="sub">Adaptive softmax weight snapshotted at each period; the top 5 accounts by
      current weight are highlighted, the rest are muted.</p>
    <div class="legend" id="wlegend"></div>
    <div class="chart" id="weights"></div>
  </div>
  <div class="card">
    <h2>Cumulative strategy return</h2>
    <p class="sub" id="cumsub">Compounded return from going long when the aggregate score is
      bullish and short when bearish, adaptive vs uniform weights.</p>
    <div class="legend" id="clegend"></div>
    <div class="chart" id="cumret"></div>
  </div>
  <div class="card">
    <h2>Per-period strategy return</h2>
    <p class="sub">Realized return of the adaptive score's direction, one bar per resolved period.</p>
    <div class="chart" id="perret"></div>
  </div>
  <div class="card">
    <h2 id="rolltitle">Rolling correlation</h2>
    <p class="sub">Score vs next-period return over a trailing window; above zero means the
      score has been pointing the right way.</p>
    <div class="legend" id="rlegend"></div>
    <div class="chart" id="rolling"></div>
  </div>
  <details>
    <summary>Data table &mdash; resolved periods</summary>
    <div class="tablebox"><table id="datatable"></table></div>
  </details>
  <footer id="foot"></footer>
</div>
<script>
const D = __PAYLOAD__;
const PALETTE = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
const fmtPct = (v, dp=1) => v == null ? "\u2013" : (v*100).toFixed(dp) + "%";
const fmtNum = (v, dp=2) => v == null ? "\u2013" : v.toFixed(dp);
const fmtDate = iso => new Date(iso).toLocaleDateString(undefined, {month:"short", day:"numeric"});

function niceTicks(lo, hi, n=4) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m*mag).find(s => span/s <= n+1) || 10*mag;
  const t = []; for (let v = Math.ceil(lo/step)*step; v <= hi + 1e-12; v += step) t.push(v);
  return t;
}

// Shared frame: scales, gridlines, axes, crosshair + tooltip plumbing.
function frame(el, dates, yLo, yHi, yFmt) {
  const W = Math.max(el.clientWidth, 320), H = 280, L = 56, R = 14, T = 12, B = 26;
  const x = i => L + (dates.length < 2 ? 0.5 : i/(dates.length-1)) * (W-L-R);
  const y = v => T + (1 - (v-yLo)/(yHi-yLo)) * (H-T-B);
  const ticks = niceTicks(yLo, yHi);
  let g = "";
  for (const t of ticks) g += `<line x1="${L}" x2="${W-R}" y1="${y(t)}" y2="${y(t)}" style="stroke:var(--grid)"/>`
    + `<text x="${L-8}" y="${y(t)+4}" text-anchor="end" style="fill:var(--muted);font-size:11px">${yFmt(t)}</text>`;
  const every = Math.max(1, Math.ceil(dates.length/6));
  dates.forEach((d, i) => { if (i % every === 0)
    g += `<text x="${x(i)}" y="${H-6}" text-anchor="middle" style="fill:var(--muted);font-size:11px">${fmtDate(d)}</text>`; });
  g += `<line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" style="stroke:var(--axis)"/>`;
  return { W, H, L, R, T, B, x, y, g };
}

function pathFor(values, x, y) {
  let d = "", pen = false;
  values.forEach((v, i) => {
    if (v == null) { pen = false; return; }
    d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); pen = true;
  });
  return d;
}

function attachHover(el, svg, f, dates, rows) {
  const tip = document.createElement("div"); tip.className = "tip"; el.parentNode.appendChild(tip);
  const cross = document.createElementNS("http://www.w3.org/2000/svg", "line");
  cross.setAttribute("style", "stroke:var(--axis);display:none");
  cross.setAttribute("y1", f.T); cross.setAttribute("y2", f.H - f.B); svg.appendChild(cross);
  svg.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (f.W / r.width);
    const i = Math.max(0, Math.min(dates.length-1,
      Math.round((px - f.L) / (f.W-f.L-f.R) * (dates.length-1))));
    cross.setAttribute("x1", f.x(i)); cross.setAttribute("x2", f.x(i));
    cross.style.display = "block";
    const body = rows(i, (ev.clientY - r.top) * (f.H / r.height));
    if (!body) { tip.style.display = "none"; return; }
    tip.innerHTML = `<div class="d">${fmtDate(dates[i])}</div>` + body;
    tip.style.display = "block";
    const cardR = el.parentNode.getBoundingClientRect();
    const left = ev.clientX - cardR.left + 14, flip = left + tip.offsetWidth > cardR.width - 8;
    tip.style.left = (flip ? ev.clientX - cardR.left - tip.offsetWidth - 14 : left) + "px";
    tip.style.top = (ev.clientY - cardR.top - 10) + "px";
  });
  svg.addEventListener("mouseleave", () => { tip.style.display = "none"; cross.style.display = "none"; });
}

function endLabels(f, series) {
  // Direct labels at each colored line's end, nudged apart to avoid collisions.
  const pts = series.filter(s => s.label).map(s => {
    const li = s.values.map((v,i)=>[v,i]).filter(p=>p[0]!=null).pop();
    return li ? { name: s.name, color: s.color, y: f.y(li[0]) } : null;
  }).filter(Boolean).sort((a,b) => a.y - b.y);
  for (let i = 1; i < pts.length; i++) pts[i].y = Math.max(pts[i].y, pts[i-1].y + 13);
  return pts.map(p =>
    `<text x="${f.W-f.R-2}" y="${p.y+4}" text-anchor="end" style="fill:${p.color};font-size:11px;font-weight:600;paint-order:stroke;stroke:var(--surface);stroke-width:3px">${p.name}</text>`).join("");
}

function lineChart(el, dates, series, { yFmt = v => fmtNum(v), ref = null, tooltip = "all" } = {}) {
  const vals = series.flatMap(s => s.values).filter(v => v != null).concat(ref == null ? [] : [ref]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo || 1) * 0.08; lo -= pad; hi += pad;
  const f = frame(el, dates, lo, hi, yFmt);
  let marks = "";
  if (ref != null) marks += `<line x1="${f.L}" x2="${f.W-f.R}" y1="${f.y(ref)}" y2="${f.y(ref)}" style="stroke:var(--axis);stroke-dasharray:5 4"/>`;
  for (const s of series)
    marks += `<path d="${pathFor(s.values, f.x, f.y)}" fill="none" style="stroke:${s.color};stroke-width:${s.width||2};${s.dash?`stroke-dasharray:${s.dash};`:""}stroke-opacity:${s.opacity??1};stroke-linejoin:round"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${f.W} ${f.H}" role="img">${f.g}${marks}${endLabels(f, series)}</svg>`;
  attachHover(el, el.querySelector("svg"), f, dates, (i, py) => {
    let shown = series.filter(s => s.label && s.values[i] != null);
    if (tooltip === "nearest") {
      const near = series.filter(s => s.values[i] != null)
        .map(s => ({ s, d: Math.abs(f.y(s.values[i]) - py) })).sort((a,b) => a.d - b.d)[0];
      if (near && !shown.some(s => s === near.s)) shown = shown.concat(near.s);
    }
    return shown.map(s => `<div class="row"><span><span class="chip" style="background:${s.color}"></span>${s.name}</span><span>${yFmt(s.values[i])}</span></div>`).join("");
  });
}

function barChart(el, dates, values, { yFmt = v => fmtPct(v), name = "return" } = {}) {
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  const pad = (hi - lo || 1) * 0.08; lo -= pad; hi += pad;
  const f = frame(el, dates, lo, hi, yFmt);
  const bw = Math.max(2, (f.W-f.L-f.R)/Math.max(values.length,1) - 2);
  let marks = "";
  values.forEach((v, i) => {
    const y0 = f.y(0), y1 = f.y(v), top = Math.min(y0, y1), h = Math.max(Math.abs(y0-y1), 1);
    const rr = Math.min(4, bw/2, h);
    marks += `<path d="M${f.x(i)-bw/2} ${v >= 0 ? top+h : top}
      ${v >= 0 ? `V${top+rr} q0 ${-rr} ${rr} ${-rr} h${bw-2*rr} q${rr} 0 ${rr} ${rr} V${top+h}`
               : `V${top+h-rr} q0 ${rr} ${rr} ${rr} h${bw-2*rr} q${rr} 0 ${rr} ${-rr} V${top}`} Z"
      style="fill:var(${v >= 0 ? "--pos" : "--neg"})"/>`;
  });
  marks += `<line x1="${f.L}" x2="${f.W-f.R}" y1="${f.y(0)}" y2="${f.y(0)}" style="stroke:var(--axis)"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${f.W} ${f.H}" role="img">${f.g}${marks}</svg>`;
  attachHover(el, el.querySelector("svg"), f, dates, i =>
    `<div class="row"><span><span class="chip" style="background:var(${values[i] >= 0 ? "--pos" : "--neg"})"></span>${name}</span><span>${yFmt(values[i])}</span></div>`);
}

function legend(el, items) {
  el.innerHTML = items.map(([name, color, dash]) =>
    `<span><span class="chip" style="background:${dash ? "transparent" : color};${dash ? `border:1.5px dashed ${color};width:9px;height:0;margin-bottom:4px;` : ""}"></span>${name}</span>`).join("");
}

function render() {
  const m = D.meta, t = D.tiles;
  document.getElementById("subtitle").textContent =
    `${m.symbol} \u00b7 ${m.horizon} horizon \u00b7 ${m.n_resolved} resolved periods, ` +
    `${m.span[0]} \u2192 ${m.span[1]} \u00b7 ${m.n_accounts} accounts`;
  document.getElementById("tiles").innerHTML = [
    ["Cumulative return", fmtPct(t.cum), `uniform weights ${fmtPct(t.cum_uniform)}`],
    ["Sortino (annualized)", fmtNum(t.sortino), `uniform weights ${fmtNum(t.sortino_uniform)}`],
    ["Pearson", fmtNum(t.pearson), "score vs next-period return"],
    ["Spearman", fmtNum(t.spearman), `hit rate ${fmtPct(t.hit_rate, 0)}`],
  ].map(([k, v, sub]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="sub">${sub}</div></div>`).join("");

  const W = D.weights, others = Object.keys(W.series).filter(a => !W.top.includes(a));
  lineChart(document.getElementById("weights"), W.dates,
    others.map(a => ({ name: a, values: W.series[a], color: "var(--muted)", width: 1, opacity: 0.45 }))
      .concat(W.top.map((a, i) => ({ name: a, values: W.series[a], color: PALETTE[i], label: true }))),
    { yFmt: v => fmtPct(v), ref: W.uniform, tooltip: "nearest" });
  legend(document.getElementById("wlegend"),
    W.top.map((a, i) => [a, PALETTE[i]]).concat([["other accounts", "var(--muted)"], ["uniform 1/N", "var(--axis)", true]]));

  if (m.deadband > 0) document.getElementById("cumsub").textContent =
    `Compounded return from going long when the aggregate score is bullish and short when `
    + `bearish, flat when |score| \u2264 ${m.deadband} \u2014 adaptive vs uniform weights.`;
  const R = D.returns;
  lineChart(document.getElementById("cumret"), R.dates, [
    { name: "adaptive", values: R.cum, color: PALETTE[0], label: true },
    { name: "uniform", values: R.cum_uniform, color: PALETTE[1], label: true },
  ], { yFmt: v => fmtPct(v), ref: 0 });
  legend(document.getElementById("clegend"), [["adaptive weights", PALETTE[0]], ["uniform weights", PALETTE[1]]]);

  barChart(document.getElementById("perret"), R.dates, R.strategy);

  document.getElementById("rolltitle").textContent = `Rolling ${m.window}-period correlation`;
  lineChart(document.getElementById("rolling"), D.rolling.dates, [
    { name: "Pearson", values: D.rolling.pearson, color: PALETTE[0], label: true },
    { name: "Spearman", values: D.rolling.spearman, color: PALETTE[1], label: true },
  ], { yFmt: v => fmtNum(v), ref: 0 });
  legend(document.getElementById("rlegend"), [["Pearson", PALETTE[0]], ["Spearman", PALETTE[1]]]);

  document.getElementById("datatable").innerHTML =
    "<tr><th>period</th><th>score</th><th>uniform</th><th>return</th><th>strategy</th></tr>" +
    D.table.map(r => `<tr><td>${r[0].slice(0, 16).replace("T", " ")}</td><td>${fmtNum(r[1], 3)}</td>` +
      `<td>${fmtNum(r[2], 3)}</td><td>${fmtPct(r[3], 2)}</td><td>${fmtPct(r[4], 2)}</td></tr>`).join("");
  document.getElementById("foot").textContent =
    `Generated ${m.generated} by dashboard.py from the local tracker db.`;
}
render();
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(render, 150); });
</script>
"""

if __name__ == "__main__":
    main()
