// Sonification: hear the brain think.
//
// Each super class gets a note of a pentatonic scale. Every frame, classes
// that spiked play a soft pluck whose loudness grows with the log of their
// spike count, so a quiet brain ticks and a busy one shimmers. Behaviour
// events (escape, feeding) get their own sounds.

const SCALE = [261.6, 293.7, 329.6, 392.0, 440.0, 523.3, 587.3, 659.3, 784.0, 880.0, 1046.5];

export class Sonifier {
  constructor() { this.ctx = null; this.on = false; this.vol = 0.5; this.prev = {}; }
  enable() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.master = this.ctx.createGain();
      this.master.gain.value = this.vol * 0.25;
      const comp = this.ctx.createDynamicsCompressor();
      this.master.connect(comp); comp.connect(this.ctx.destination);
    }
    this.ctx.resume();
    this.on = true;
  }
  disable() { this.on = false; }
  setVolume(v) { this.vol = v; if (this.master) this.master.gain.value = v * 0.25; }
  pluck(freq, gain, when, type = "sine", dur = 0.18) {
    const ctx = this.ctx, o = ctx.createOscillator(), g = ctx.createGain();
    o.type = type; o.frequency.value = freq * (1 + (Math.random() - 0.5) * 0.004);
    g.gain.setValueAtTime(0.0001, when);
    g.gain.exponentialRampToValueAtTime(Math.max(gain, 0.0002), when + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, when + dur);
    o.connect(g); g.connect(this.master); o.start(when); o.stop(when + dur + 0.02);
  }
  frame(classCounts, classIndex, drive) {
    if (!this.on || !this.ctx) return;
    const t = this.ctx.currentTime + 0.01;
    for (const [name, n] of Object.entries(classCounts)) {
      if (n <= 0) continue;
      const k = classIndex[name] ?? 0;
      const g = Math.min(0.5, 0.06 * Math.log1p(n));
      this.pluck(SCALE[k % SCALE.length], g, t + Math.random() * 0.015, "triangle");
    }
    if ((drive.escape ?? 0) > 0.5 && (this.prev.escape ?? 0) <= 0.5) this.whoosh(t);
    if ((drive.feeding ?? 0) > 0.4 && (this.prev.feeding ?? 0) <= 0.4) [0, 0.09, 0.18].forEach((d, i) => this.pluck([523.3, 659.3, 784][i], 0.3, t + d, "sine", 0.5));
    this.prev = { ...drive };
  }
  whoosh(t) {
    const ctx = this.ctx, len = 0.5, buf = ctx.createBuffer(1, ctx.sampleRate * len, ctx.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / d.length);
    const src = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
    src.buffer = buf; f.type = "bandpass"; f.frequency.setValueAtTime(300, t); f.frequency.exponentialRampToValueAtTime(3000, t + len);
    g.gain.value = 0.6; src.connect(f); f.connect(g); g.connect(this.master); src.start(t);
  }
}
