"""Brain movies: one self-contained HTML file that replays a simulation in 3D.

    res = sim.run(500)
    res.movie("sugar.html")          # open in any browser, share by email...

The file embeds the neuron positions, the spikes (frame by frame) and the
same WebGL renderer the playground uses. No server, no internet needed.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np

_VIEWER = Path(__file__).resolve().parent.parent / "playground" / "static" / "js" / "viewer.js"

_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#04060c;--panel:rgba(13,18,32,.88);--line:rgba(120,150,255,.16);--text:#dfe6ff;--muted:#8592b8;--accent:#6ae4ff}
html,body{margin:0;height:100%;background:var(--bg);color:var(--text);font:13px Inter,system-ui,sans-serif;overflow:hidden}
canvas{position:absolute;inset:0;width:100%;height:100%}
.bar{position:absolute;left:16px;right:16px;bottom:16px;display:flex;gap:10px;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 12px}
.bar input[type=range]{flex:1;accent-color:var(--accent)}
button{background:rgba(255,255,255,.06);border:1px solid var(--line);color:var(--text);border-radius:8px;padding:5px 12px;cursor:pointer}
.t{font-family:monospace;color:var(--accent);min-width:110px}
.head{position:absolute;left:16px;top:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px 14px;max-width:60%}
.head b{font-size:15px}.head div{color:var(--muted);margin-top:4px}
.legend{position:absolute;right:16px;top:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:8px 12px;font-size:11.5px}
.legend span{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
</style></head><body>
<canvas id="gl"></canvas>
<div class="head"><b>__TITLE__</b><div>__SUBTITLE__</div></div>
<div class="legend" id="legend"></div>
<div class="bar"><button id="play">Pause</button><input type="range" id="scrub" min="0" max="1" value="0"><span class="t" id="t"></span>
<select id="spd"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select></div>
<script type="module">
__VIEWER__
const DATA = __DATA__;
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const pos = dec(DATA.positions, Float32Array), cls = dec(DATA.classes, Uint8Array);
const spikes = dec(DATA.spikes, Uint32Array), offs = dec(DATA.offsets, Uint32Array);
const v = new BrainViewer(document.getElementById("gl"));
v.setGeometry(pos, { cls, nt: cls, side: cls }, { cls: DATA.colors.map(hexToRgb), nt: DATA.colors.map(hexToRgb), side: DATA.colors.map(hexToRgb) });
v.params.autoRotate = true;
document.getElementById("legend").innerHTML = DATA.names.map((n, k) => DATA.counts[k] ? `<div><span style="background:${DATA.colors[k]}"></span>${n}</div>` : "").join("");
const nF = offs.length - 1, scrub = document.getElementById("scrub"); scrub.max = nF - 1;
let f = 0, playing = true, acc = 0, last = performance.now();
const show = (k) => { for (let i = offs[k]; i < offs[k + 1]; i++) v.spikes[spikes[i]] = k * DATA.frame_ms; v._spikesDirty = true; v.time = k * DATA.frame_ms;
  document.getElementById("t").textContent = `t = ${(k * DATA.frame_ms).toFixed(0)} ms`; scrub.value = k; };
document.getElementById("play").onclick = (e) => { playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; };
scrub.oninput = () => { v.clearSpikes(); f = +scrub.value; show(f); };
function loop(now) {
  const dt = now - last; last = now;
  if (playing) { acc += dt * +document.getElementById("spd").value; while (acc >= DATA.frame_ms) { acc -= DATA.frame_ms; f = (f + 1) % nF; if (f === 0) v.clearSpikes(); show(f); } }
  v.render(now, dt); requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
</script></body></html>"""


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def export_movie(result, path, trial: int = 0, frame_ms: float = 10.0, title: str | None = None, subtitle: str | None = None):
    """Write ``path`` (HTML) replaying ``trial`` of a :class:`~openfly.SimResult` at real time."""
    path = Path(path)
    path.write_text(movie_html(result, trial, frame_ms, title, subtitle), encoding="utf-8")
    return path


def show(result, height: int = 520, **kw):
    """Display the 3D replay inline in a Jupyter notebook (an iframe holding the movie page)."""
    from IPython.display import HTML

    page = movie_html(result, **kw)
    return HTML(f'<iframe srcdoc="{html.escape(page)}" style="width:100%;height:{height}px;border:0;border-radius:12px"></iframe>')


def movie_html(result, trial: int = 0, frame_ms: float = 10.0, title: str | None = None, subtitle: str | None = None) -> str:
    """The self-contained replay page as a string."""
    from ..playground.session import PALETTE_CLASS, _codes

    b = result.brain
    P = b.positions.astype(np.float64)
    P = P - np.nanmedian(P, axis=0)
    P = (P / (np.nanmax(np.abs(P)) or 1.0)).astype(np.float32)
    P[:, 1] *= -1
    P = np.nan_to_num(P)
    codes, names, colors = _codes(b.neurons["super_class"].to_numpy(), PALETTE_CLASS)
    m = result.trials == trial
    t, n = result.times_ms[m], result.neurons[m]
    n_frames = max(1, int(np.ceil(result.duration_ms / frame_ms)))
    frame = np.minimum((t / frame_ms).astype(np.int64), n_frames - 1)
    order = np.lexsort((n, frame))
    frame, n = frame[order], n[order]
    offsets = np.searchsorted(frame, np.arange(n_frames + 1)).astype(np.uint32)
    data = {
        "positions": _b64(P), "classes": _b64(codes), "spikes": _b64(n.astype(np.uint32)), "offsets": _b64(offsets),
        "colors": colors, "names": [x or "unannotated" for x in names],
        "counts": np.bincount(codes, minlength=len(names)).tolist(), "frame_ms": frame_ms,
    }
    viewer = _VIEWER.read_text().replace("export class", "class").replace("export function", "function")
    return (_TEMPLATE.replace("__VIEWER__", viewer).replace("__DATA__", json.dumps(data))
            .replace("__TITLE__", html.escape(title or result.label or "openfly simulation"))
            .replace("__SUBTITLE__", html.escape(subtitle or f"{b.meta.get('title', b.name)}: {b.N:,} neurons, "
                                                 f"{result.duration_ms:g} ms, {int(m.sum()):,} spikes")))
