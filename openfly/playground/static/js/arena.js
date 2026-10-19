// ArenaView: top-down view of the virtual fly's world.

export class ArenaView {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext("2d");
    this.arena = null;
    this.field = null;
    this.trail = [];
    this.fly = null;
    this.flash = 0;
    this.tool = "move";
    canvas.addEventListener("click", (e) => {
      if (!this.arena || !this.onClick) return;
      const r = canvas.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * this.arena.width;
      const y = (1 - (e.clientY - r.top) / r.height) * this.arena.height;
      this.onClick(this.tool, x, y);
    });
  }
  setArena(arena, field) {
    this.arena = arena;
    this.trail = [];
    if (field) {
      const bin = atob(field.data), buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      this.field = { n: field.n, max: field.max || 1, data: new Float32Array(buf.buffer) };
      this._fieldImg = null;
    }
  }
  update(fly) {
    if (!fly) return;
    this.fly = fly;
    this.trail.push([fly.x, fly.y, fly.state]);
    if (this.trail.length > 900) this.trail.shift();
    if (fly.state === "escaping") this.flash = 1;
    if (fly.items) this.items = fly.items;
  }
  _renderField(w, h) {
    const f = this.field, off = document.createElement("canvas");
    off.width = f.n; off.height = f.n;
    const ctx = off.getContext("2d"), img = ctx.createImageData(f.n, f.n);
    for (let j = 0; j < f.n; j++) for (let i = 0; i < f.n; i++) {
      const v = Math.pow(f.data[j * f.n + i] / f.max, 0.6), k = 4 * ((f.n - 1 - j) * f.n + i);
      img.data[k] = 40 + 200 * v; img.data[k + 1] = 30 + 120 * v; img.data[k + 2] = 90 + 40 * v; img.data[k + 3] = 30 + 150 * v;
    }
    ctx.putImageData(img, 0, 0);
    this._fieldImg = off;
  }
  draw() {
    const c = this.c, ctx = this.ctx, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth * dpr, h = c.clientHeight * dpr;
    if (!w || !h) return;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    ctx.fillStyle = "#070b16"; ctx.fillRect(0, 0, w, h);
    if (!this.arena) return;
    const A = this.arena, sx = w / A.width, sy = h / A.height;
    const X = (x) => x * sx, Y = (y) => h - y * sy;
    if (this.field) {
      if (!this._fieldImg) this._renderField(w, h);
      ctx.imageSmoothingEnabled = true; ctx.globalAlpha = 0.9; ctx.drawImage(this._fieldImg, 0, 0, w, h); ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    for (let g = 20; g < A.width; g += 20) { ctx.beginPath(); ctx.moveTo(X(g), 0); ctx.lineTo(X(g), h); ctx.stroke(); ctx.beginPath(); ctx.moveTo(0, Y(g)); ctx.lineTo(w, Y(g)); ctx.stroke(); }
    if (A.wind_from_deg !== null && A.wind_from_deg !== undefined && A.wind_speed > 0) {
      ctx.fillStyle = "rgba(160,200,255,0.6)"; ctx.font = `${11 * dpr}px sans-serif`;
      ctx.fillText(`wind from ${A.wind_from_deg} deg`, 8 * dpr, h - 26 * dpr);
    }
    for (const it of A.items) {
      if (it.kind === "food") {
        ctx.beginPath(); ctx.arc(X(it.x), Y(it.y), Math.max(4, it.radius_mm * sx), 0, 2 * Math.PI);
        ctx.fillStyle = it.color || "#f5d742"; ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 18 * dpr; ctx.fill(); ctx.shadowBlur = 0;
        ctx.fillStyle = "#fff"; ctx.font = `${11 * dpr}px sans-serif`; ctx.fillText(it.name, X(it.x) + 8 * dpr, Y(it.y) - 8 * dpr);
      } else if (it.kind === "heat") {
        const g = ctx.createRadialGradient(X(it.x), Y(it.y), 0, X(it.x), Y(it.y), it.radius_mm * sx * 1.6);
        g.addColorStop(0, "rgba(255,90,40,0.55)"); g.addColorStop(1, "rgba(255,90,40,0)");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(X(it.x), Y(it.y), it.radius_mm * sx * 1.6, 0, 2 * Math.PI); ctx.fill();
      }
    }
    // trail
    ctx.lineWidth = 1.5 * dpr;
    for (let i = 1; i < this.trail.length; i++) {
      const [x0, y0] = this.trail[i - 1], [x1, y1, st] = this.trail[i];
      if (Math.hypot(x1 - x0, y1 - y0) > 8) continue;
      const a = i / this.trail.length;
      ctx.strokeStyle = st === "feeding" ? `rgba(255,209,102,${a})` : st === "backing up" ? `rgba(199,125,255,${a})` : `rgba(106,228,255,${a * 0.8})`;
      ctx.beginPath(); ctx.moveTo(X(x0), Y(y0)); ctx.lineTo(X(x1), Y(y1)); ctx.stroke();
    }
    const f = this.fly;
    if (f) {
      const hd = (f.heading_deg * Math.PI) / 180, s = 9 * dpr;
      ctx.save(); ctx.translate(X(f.x), Y(f.y)); ctx.rotate(-hd);
      ctx.fillStyle = f.state === "feeding" ? "#ffd166" : f.state === "escaping" ? "#ff5d73" : "#e8f6ff";
      ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 14 * dpr;
      ctx.beginPath(); ctx.moveTo(s, 0); ctx.lineTo(-s * 0.7, s * 0.55); ctx.lineTo(-s * 0.4, 0); ctx.lineTo(-s * 0.7, -s * 0.55); ctx.closePath(); ctx.fill();
      ctx.shadowBlur = 0; ctx.strokeStyle = "rgba(255,255,255,0.6)"; ctx.lineWidth = 1 * dpr;
      ctx.beginPath(); ctx.moveTo(s * 0.8, 0); ctx.lineTo(s * 1.4, s * 0.35); ctx.moveTo(s * 0.8, 0); ctx.lineTo(s * 1.4, -s * 0.35); ctx.stroke();
      ctx.restore();
    }
    if (this.flash > 0.01) { ctx.fillStyle = `rgba(255,60,90,${this.flash * 0.35})`; ctx.fillRect(0, 0, w, h); this.flash *= 0.9; }
  }
}
