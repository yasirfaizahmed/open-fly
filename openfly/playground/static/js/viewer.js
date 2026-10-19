// BrainViewer: a dependency-free WebGL2 renderer for whole-brain activity.
//
// Every neuron is a glowing point. When a neuron spikes we store the spike
// time; the shader turns "time since last spike" into a flash that decays
// over `decayMs`, so activity looks like light rippling through the brain.
// Points are drawn with additive blending (order-independent, no depth sort).

const VS = `#version 300 es
in vec3 aPos; in vec3 aColor; in float aSpike; in float aFlag;
uniform mat4 uMVP; uniform float uTime; uniform float uDecay; uniform float uSize;
uniform float uBase; uniform float uDpr; uniform float uActivityOnly;
out vec4 vColor; out float vGlow;
void main() {
  vec4 p = uMVP * vec4(aPos, 1.0);
  gl_Position = p;
  float dt = uTime - aSpike;
  float a = (dt >= 0.0) ? exp(-dt / uDecay) : 0.0;
  float hl = (aFlag > 0.5 && aFlag < 1.5) ? 1.0 : 0.0;
  float sil = aFlag > 1.5 ? 1.0 : 0.0;
  float alpha = max(uBase * (1.0 - 0.9 * uActivityOnly), a);
  vec3 col = mix(aColor, vec3(1.0, 0.97, 0.92), a * 0.6);
  if (hl > 0.5) { col = mix(col, vec3(0.45, 1.0, 1.0), 0.55); alpha = max(alpha, 0.5); }
  if (sil > 0.5) { col = vec3(1.0, 0.18, 0.28); alpha = max(uBase * 1.5, 0.12); a = 0.0; }
  vColor = vec4(col, alpha);
  vGlow = a;
  float size = uSize * (1.0 + 2.8 * a + 0.9 * hl);
  gl_PointSize = clamp(size * uDpr * (2.4 / max(p.w, 0.05)), 1.0, 72.0);
}`;

const FS = `#version 300 es
precision mediump float;
in vec4 vColor; in float vGlow; out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  float core = exp(-r2 * 3.5);
  float halo = exp(-r2 * 1.1) * vGlow;
  float a = vColor.a * (core + 0.7 * halo);
  o = vec4(vColor.rgb * a, a);
}`;

const PATH_VS = `#version 300 es
in vec3 aPos; in float aS;
uniform mat4 uMVP; uniform float uTime; uniform float uDpr; uniform float uLen;
out float vA;
void main() {
  vec4 p = uMVP * vec4(aPos, 1.0);
  gl_Position = p;
  float wave = fract(uTime * 0.00045 - aS / max(uLen, 1e-3));
  vA = 0.25 + 0.75 * smoothstep(0.85, 1.0, wave);
  gl_PointSize = clamp((2.5 + 5.0 * vA) * uDpr * (2.4 / max(p.w, 0.05)), 1.0, 40.0);
}`;
const PATH_FS = `#version 300 es
precision mediump float;
in float vA; out vec4 o;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0; float r2 = dot(d, d); if (r2 > 1.0) discard;
  float a = exp(-r2 * 2.5) * vA;
  vec3 c = mix(vec3(0.35, 0.9, 1.0), vec3(1.0, 1.0, 1.0), vA * 0.6);
  o = vec4(c * a, a);
}`;

function compile(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

// ---- tiny matrix helpers (column-major, like OpenGL)
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(e, c, up) {
  let zx = e[0] - c[0], zy = e[1] - c[1], zz = e[2] - c[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz); xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
    -(xx * e[0] + xy * e[1] + xz * e[2]), -(yx * e[0] + yy * e[1] + yz * e[2]), -(zx * e[0] + zy * e[1] + zz * e[2]), 1]);
}
function mul(a, b) {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
    o[i * 4 + j] = s;
  }
  return o;
}
export function hexToRgb(h) {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export class BrainViewer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.prog = compile(gl, VS, FS);
    this.pathProg = compile(gl, PATH_VS, PATH_FS);
    this.N = 0;
    this.cam = { yaw: 0, pitch: 0.12, dist: 2.6, target: [0, 0, 0], fov: 0.75 };
    this.params = { size: 2.0, decay: 140, base: 0.08, activityOnly: 0, autoRotate: false };
    this.time = 0;
    this.pathN = 0;
    this._bindInput();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.dpr = dpr;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  setGeometry(positions, codes, palettes) {
    const gl = this.gl;
    this.N = positions.length / 3;
    this.positions = positions;
    this.codes = codes; // {cls, nt, side}: Uint8Array
    this.palettes = palettes; // {cls: [[r,g,b]], nt: [...], side: [...]}
    this.colors = new Float32Array(this.N * 3);
    this.spikes = new Float32Array(this.N).fill(-1e9);
    this.flags = new Float32Array(this.N);
    // idle neurons: brighter and bigger for small brains, fainter for dense ones
    this.params.base = Math.min(0.32, Math.max(0.08, 9 / Math.sqrt(this.N)));
    this.params.size = this.N > 50000 ? 1.8 : 3.0;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const mk = (data, loc, n, usage) => {
      const b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, usage);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0);
      return b;
    };
    const P = this.prog;
    this.bPos = mk(positions, gl.getAttribLocation(P, "aPos"), 3, gl.STATIC_DRAW);
    this.bCol = mk(this.colors, gl.getAttribLocation(P, "aColor"), 3, gl.DYNAMIC_DRAW);
    this.bSpk = mk(this.spikes, gl.getAttribLocation(P, "aSpike"), 1, gl.DYNAMIC_DRAW);
    this.bFlag = mk(this.flags, gl.getAttribLocation(P, "aFlag"), 1, gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    this.setColorMode("class");
  }

  setColorMode(mode, values) {
    this.colorMode = mode;
    const c = this.colors, N = this.N;
    if (mode === "influence" && values) {
      for (let i = 0; i < N; i++) {
        const v = Math.max(-1, Math.min(1, values[i] * 6));
        const a = Math.min(1, Math.abs(v) * 1.0 + 0.04);
        if (v >= 0) { c[3 * i] = 1.0 * a; c[3 * i + 1] = 0.45 * a; c[3 * i + 2] = 0.2 * a; }
        else { c[3 * i] = 0.25 * a; c[3 * i + 1] = 0.55 * a; c[3 * i + 2] = 1.0 * a; }
      }
    } else {
      const key = mode === "nt" ? "nt" : mode === "side" ? "side" : "cls";
      const codes = this.codes[key], pal = this.palettes[key];
      for (let i = 0; i < N; i++) {
        const col = pal[codes[i]] || [0.5, 0.5, 0.6];
        c[3 * i] = col[0]; c[3 * i + 1] = col[1]; c[3 * i + 2] = col[2];
      }
    }
    this.params.activityOnly = mode === "activity" ? 1 : 0;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bCol);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, c);
  }

  spike(indices, tMs) {
    const s = this.spikes;
    for (let k = 0; k < indices.length; k++) s[indices[k]] = tMs;
    this._spikesDirty = true;
  }

  clearSpikes() { this.spikes.fill(-1e9); this._spikesDirty = true; }

  setFlags(highlight, silenced) {
    const f = this.flags;
    f.fill(0);
    if (highlight) for (let k = 0; k < highlight.length; k++) f[highlight[k]] = 1;
    if (silenced) for (let k = 0; k < silenced.length; k++) f[silenced[k]] = 2;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bFlag);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, f);
  }

  setPath(indices) {
    const gl = this.gl;
    if (!indices || indices.length < 2) { this.pathN = 0; return; }
    const pts = [], ss = [];
    let s = 0;
    const per = 60;
    for (let k = 0; k < indices.length - 1; k++) {
      const a = indices[k], b = indices[k + 1];
      const A = [this.positions[3 * a], this.positions[3 * a + 1], this.positions[3 * a + 2]];
      const B = [this.positions[3 * b], this.positions[3 * b + 1], this.positions[3 * b + 2]];
      const len = Math.hypot(B[0] - A[0], B[1] - A[1], B[2] - A[2]);
      const lift = 0.15 * len;
      for (let i = 0; i < per; i++) {
        const t = i / per, arc = Math.sin(Math.PI * t) * lift;
        pts.push(A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t + arc, A[2] + (B[2] - A[2]) * t);
        ss.push(s + len * t);
      }
      s += len;
    }
    this.pathLen = s;
    this.pathN = pts.length / 3;
    if (!this.pathVao) {
      this.pathVao = gl.createVertexArray();
      gl.bindVertexArray(this.pathVao);
      this.bPathPos = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bPathPos);
      const lp = gl.getAttribLocation(this.pathProg, "aPos");
      gl.enableVertexAttribArray(lp); gl.vertexAttribPointer(lp, 3, gl.FLOAT, false, 0, 0);
      this.bPathS = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bPathS);
      const ls = gl.getAttribLocation(this.pathProg, "aS");
      gl.enableVertexAttribArray(ls); gl.vertexAttribPointer(ls, 1, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bPathPos); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bPathS); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(ss), gl.STATIC_DRAW);
  }

  view(name) {
    const v = { back: [0, 0.12], front: [Math.PI, 0.12], top: [0, 1.45], left: [-Math.PI / 2, 0.1], right: [Math.PI / 2, 0.1] }[name];
    if (v) { this.cam.yaw = v[0]; this.cam.pitch = v[1]; this.cam.dist = 2.6; this.cam.target = [0, 0, 0]; }
  }

  matrices() {
    const c = this.cam, cp = Math.cos(c.pitch);
    const eye = [c.target[0] + c.dist * cp * Math.sin(c.yaw), c.target[1] + c.dist * Math.sin(c.pitch), c.target[2] + c.dist * cp * Math.cos(c.yaw)];
    const V = lookAt(eye, c.target, [0, 1, 0]);
    const P = perspective(c.fov, this.canvas.width / this.canvas.height, 0.02, 50);
    return mul(P, V);
  }

  render(nowMs, dtMs) {
    const gl = this.gl;
    if (this.params.autoRotate && !this._dragging) this.cam.yaw += 0.00012 * (dtMs || 16);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.016, 0.024, 0.047, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.N) return;
    if (this._spikesDirty) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bSpk);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.spikes);
      this._spikesDirty = false;
    }
    this.mvp = this.matrices();
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.prog);
    const u = (n) => gl.getUniformLocation(this.prog, n);
    gl.uniformMatrix4fv(u("uMVP"), false, this.mvp);
    gl.uniform1f(u("uTime"), this.time);
    gl.uniform1f(u("uDecay"), this.params.decay);
    gl.uniform1f(u("uSize"), this.params.size);
    gl.uniform1f(u("uBase"), this.params.base);
    gl.uniform1f(u("uDpr"), this.dpr);
    gl.uniform1f(u("uActivityOnly"), this.params.activityOnly);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.POINTS, 0, this.N);
    if (this.pathN) {
      gl.useProgram(this.pathProg);
      const up = (n) => gl.getUniformLocation(this.pathProg, n);
      gl.uniformMatrix4fv(up("uMVP"), false, this.mvp);
      gl.uniform1f(up("uTime"), nowMs);
      gl.uniform1f(up("uDpr"), this.dpr);
      gl.uniform1f(up("uLen"), this.pathLen);
      gl.bindVertexArray(this.pathVao);
      gl.drawArrays(gl.POINTS, 0, this.pathN);
    }
    gl.bindVertexArray(null);
  }

  pick(clientX, clientY) {
    if (!this.mvp || !this.N) return -1;
    const r = this.canvas.getBoundingClientRect();
    const px = ((clientX - r.left) / r.width) * 2 - 1, py = -(((clientY - r.top) / r.height) * 2 - 1);
    const m = this.mvp, P = this.positions;
    const tol = (14 / r.width) * 2;
    let best = -1, bestScore = Infinity;
    for (let i = 0; i < this.N; i++) {
      const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (w <= 0) continue;
      const sx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / w, sy = (m[1] * x + m[5] * y + m[9] * z + m[13]) / w;
      const d = Math.hypot(sx - px, (sy - py) * (r.height / r.width));
      if (d < tol) {
        const recent = this.time - this.spikes[i] < 300 ? -0.5 : 0;
        const score = d / tol + w * 0.05 + recent + (this.flags[i] === 1 ? -0.3 : 0);
        if (score < bestScore) { bestScore = score; best = i; }
      }
    }
    return best;
  }

  _bindInput() {
    const c = this.canvas;
    let last = null, button = 0, moved = 0;
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => { last = [e.clientX, e.clientY]; button = e.button; moved = 0; this._dragging = true; c.setPointerCapture(e.pointerId); });
    c.addEventListener("pointermove", (e) => {
      if (!last) return;
      const dx = e.clientX - last[0], dy = e.clientY - last[1];
      moved += Math.abs(dx) + Math.abs(dy);
      last = [e.clientX, e.clientY];
      if (button === 2 || e.shiftKey) {
        const s = this.cam.dist * 0.0012, cy = Math.cos(this.cam.yaw), sy = Math.sin(this.cam.yaw);
        this.cam.target[0] -= (dx * cy) * s; this.cam.target[2] += (dx * sy) * s; this.cam.target[1] += dy * s;
      } else {
        this.cam.yaw -= dx * 0.006;
        this.cam.pitch = Math.max(-1.5, Math.min(1.5, this.cam.pitch + dy * 0.006));
      }
    });
    c.addEventListener("pointerup", (e) => {
      this._dragging = false;
      if (moved < 5 && this.onPick) this.onPick(this.pick(e.clientX, e.clientY));
      last = null;
    });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.cam.dist = Math.max(0.25, Math.min(8, this.cam.dist * Math.exp(e.deltaY * 0.001))); }, { passive: false });
    // pinch zoom
    const touches = new Map();
    c.addEventListener("touchstart", (e) => { for (const t of e.changedTouches) touches.set(t.identifier, [t.clientX, t.clientY]); });
    c.addEventListener("touchmove", (e) => {
      if (e.touches.length === 2) {
        const [a, b] = e.touches;
        const pa = touches.get(a.identifier), pb = touches.get(b.identifier);
        if (pa && pb) {
          const d0 = Math.hypot(pa[0] - pb[0], pa[1] - pb[1]), d1 = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
          this.cam.dist = Math.max(0.25, Math.min(8, this.cam.dist * d0 / Math.max(d1, 1)));
        }
        for (const t of e.touches) touches.set(t.identifier, [t.clientX, t.clientY]);
      }
    });
    c.addEventListener("touchend", (e) => { for (const t of e.changedTouches) touches.delete(t.identifier); });
  }
}
