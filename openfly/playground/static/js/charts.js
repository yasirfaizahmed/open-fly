// Small canvas charts: rolling population firing rates and behaviour meters.

export class RateChart {
  constructor(canvas, colors, maxPoints = 240) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.colors = colors; // {name: "#hex"}
    this.max = maxPoints;
    this.series = {}; // name -> array
    this.t = [];
  }
  push(t, rates) {
    this.t.push(t);
    for (const [k, v] of Object.entries(rates)) {
      (this.series[k] ||= []).push(v);
    }
    for (const k of Object.keys(this.series)) {
      const s = this.series[k];
      while (s.length < this.t.length) s.unshift(0);
      while (s.length > this.max) s.shift();
    }
    while (this.t.length > this.max) this.t.shift();
  }
  clear() { this.series = {}; this.t = []; }
  draw() {
    const c = this.c, ctx = this.ctx, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth * dpr, h = c.clientHeight * dpr;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    ctx.clearRect(0, 0, w, h);
    const n = this.t.length;
    if (n < 2) return;
    // rank series by recent activity, show the top 6
    const ranked = Object.entries(this.series).map(([k, s]) => [k, s.slice(-30).reduce((a, b) => a + b, 0)]).sort((a, b) => b[1] - a[1]).slice(0, 6);
    let ymax = 1;
    for (const [k] of ranked) for (const v of this.series[k]) ymax = Math.max(ymax, v);
    ymax = Math.pow(10, Math.ceil(Math.log10(ymax * 1.1)) ) / (ymax < 3 ? 3 : 1);
    const y = (v) => h - 6 * dpr - (Math.log1p(v) / Math.log1p(ymax)) * (h - 18 * dpr);
    ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
    for (const g of [1, 10, 100]) { if (g > ymax) break; ctx.beginPath(); ctx.moveTo(0, y(g)); ctx.lineTo(w, y(g)); ctx.stroke(); ctx.fillStyle = "rgba(255,255,255,0.3)"; ctx.font = `${9 * dpr}px monospace`; ctx.fillText(`${g} Hz`, 4, y(g) - 2); }
    ranked.forEach(([k], j) => {
      const s = this.series[k];
      ctx.strokeStyle = this.colors[k] || "#ccc";
      ctx.lineWidth = 1.6 * dpr;
      ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = 6 * dpr;
      ctx.beginPath();
      for (let i = 0; i < s.length; i++) {
        const x = (i / (this.max - 1)) * w;
        i ? ctx.lineTo(x, y(s[i])) : ctx.moveTo(x, y(s[i]));
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = ctx.strokeStyle; ctx.font = `${10 * dpr}px sans-serif`;
      ctx.fillText(k.replace("_", " "), w - 110 * dpr, 13 * dpr + j * 12 * dpr);
    });
  }
}

const METER_COLORS = { feeding: "#ffd166", walk_forward: "#5dffa1", walk_backward: "#c77dff", turn: "#6ae4ff", escape: "#ff5d73",
  receptivity: "#ff7ab6", grooming: "#f4a261" };
const METER_LABELS = { feeding: "Feeding (MN9)", walk_forward: "Walk forward", walk_backward: "Walk backward", turn: "Turning",
  escape: "Escape (giant fiber)", receptivity: "Receptivity (pC1)", grooming: "Grooming" };

export class Meters {
  constructor(root, channels) {
    this.root = root;
    this.els = {};
    root.innerHTML = "";
    for (const ch of channels) {
      const m = document.createElement("div");
      m.className = "meter" + (ch.mode === "lr" ? " lr" : "");
      m.innerHTML = `<div class="top"><span>${METER_LABELS[ch.name] || ch.name}</span><span class="v">0.00</span></div>
        <div class="bar"><div class="fill" style="background:${METER_COLORS[ch.name] || "#6ae4ff"};box-shadow:0 0 12px ${METER_COLORS[ch.name] || "#6ae4ff"}"></div></div>
        <div class="desc">${ch.description}</div>`;
      root.appendChild(m);
      this.els[ch.name] = { m, fill: m.querySelector(".fill"), v: m.querySelector(".v"), mode: ch.mode };
    }
  }
  update(drive) {
    for (const [k, e] of Object.entries(this.els)) {
      const d = drive[k] ?? 0;
      if (e.mode === "lr") {
        const w = Math.min(50, Math.abs(d) * 50);
        e.fill.style.width = w + "%";
        e.fill.style.left = d >= 0 ? "50%" : (50 - w) + "%";
        e.v.textContent = Math.abs(d) < 0.03 ? "straight" : (d > 0 ? "right " : "left ") + Math.abs(d).toFixed(2);
      } else {
        e.fill.style.width = Math.min(100, d * 100) + "%";
        e.v.textContent = d.toFixed(2);
      }
      e.m.classList.toggle("hot", Math.abs(d) > 0.4);
    }
  }
}
