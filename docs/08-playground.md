# 8. The playground

Code: `openfly/playground/` (`server.py`, `session.py`, `lessons.py`, `static/`)

```bash
openfly playground                                         # toy brain, opens your browser
openfly playground -d flywire_783 --transmitters curated   # the real brain (first run downloads it)
openfly playground --host 0.0.0.0 --port 8765              # share on your local network
```

or from Python / Jupyter:

```python
from openfly.playground import start
pg = start("toy")            # runs in background threads
print(pg.url)
pg.stop()
```

![playground](images/playground_flywire_looming.png)

## Tour

**Top bar.** Switch between **Brain** mode (the brain alone) and **Virtual
fly** mode (the brain drives a body in an arena); pause/play (space bar);
speed (relative to real time; big brains run slower than real time);
reset; **Sound** (sonification, below); the simulation clock and live
stats (neurons firing, neurons "awake", compute time per 20 ms step).

**3D brain (centre).** Every dot is a neuron at its real position. Spiking
neurons flash white-hot and fade over ~140 ms ("glow" slider).

* drag = rotate, right-drag (or shift-drag) = pan, wheel / pinch = zoom
* views: Back (fly's left on screen-left), Front, Top, Side; keys 1-4
* colours: **Class**, **Transmitter**, **Side**, or **Activity** (only firing neurons visible)
* **click a neuron** to inspect it: cell type, class, side, transmitter, synapse
  counts, strongest inputs/outputs, buttons to activate/silence/show its type
* click a legend entry to highlight that whole category
* chips (top right) list active stimuli and silencing; x removes them

**Left panel tabs.**

* **Senses**: one-click scenarios (sugar, bitter, banana, predator, song...),
  then fine control: odour + strength + left/right balance, tastants, vision
  (looming left/right, flash, fruit ahead, passing fly, optic flow), hearing,
  wind, touch, heat, cold. "replace" vs "add" decides whether a new stimulus
  replaces the current ones.
* **Neurons**: search any alias / cell type / class, then **Activate** (at the
  chosen rate and side), **Silence** or **Show**.
* **Learn**: guided lessons (each step can set up the brain for you with
  "Show me") and challenges (below).
* **Lab**:
  * **Pathway finder**: strongest synaptic chain between two groups, drawn as
    an animated pulse travelling through the 3D brain;
  * **Ripple map**: colour every neuron by the net excitation (red) or inhibition
    (blue) it receives from a source within 3 synapses, with no simulation needed;
  * **Activation screen**: which cell types drive a target (rate model);
  * **Olfactory conditioning**: train the fly (see [10](10-training-the-brain.md)).

**Right panel.** Behaviour meters (feeding, walking forward/backward,
turning gauge, escape, receptivity), a rolling chart of firing rate per
super class, the neuron inspector and the brain's citation.

**Virtual fly mode** adds an arena (bottom right): odour field (heat map),
food, heat zones, the fly with its trail (cyan = walking, yellow = feeding,
purple = backing up, red flash = escape). Pick a preset arena, click to move
the fly, drop a banana or a bitter bait, add heat, or launch a **Predator!**
attack.

## Sonification

"Sound on" turns spikes into music. Each super class has a note of a
pentatonic scale; every 20 ms, classes that spiked play a soft pluck whose
loudness grows with the log of their spike count. A quiet brain ticks; a
busy one shimmers. Escapes whoosh and feeding chimes. You can *hear*
the optic lobes erupt when a predator looms.

## Lessons and challenges

Lessons (`openfly/playground/lessons.py`):

1. Meet the brain, 2. From taste to action, 3. A smell's journey,
4. Danger from above, 5. Lesion studio, 6. Teaching a fly, 7. Closing the loop.

Challenges (checked live by the server):

| challenge | goal |
|---|---|
| Puppet master | feeding > 0.5 without any taste stimulus |
| Moonwalker | walk backward > 0.4 |
| Escape artist | escape > 0.5 without any visual stimulus |
| Bitter beats sweet | sugar on, feeding < 0.1 |
| Neural sharpshooter | feeding > 0.3 while driving 10 neurons or fewer |

Adding a lesson is adding a dict to `LESSONS`; step `action`s are the same
JSON commands the UI sends (below).

## How it works

```
 browser                                   python
 ───────                                   ──────
 index.html + js/app.js   ── GET /api/info, /api/geometry ──>  Session.info(), geometry()
 js/viewer.js (WebGL2)    <── SSE /api/stream (frames) ─────   Session thread: advance 20 ms,
 js/charts.js, audio.js                                         read motor, build frame, broadcast
 js/arena.js              ── POST /api/command {cmd...} ──>    Session.command() (under a lock)
                          ── GET /api/job/<id> ──────────>     long experiments in threads
```

* **Frames** (JSON, ~20 per second of brain time): simulation time,
  indices of neurons that spiked (base64 `uint32`), firing rate per super
  class, behavioural drives, active stimuli, fly state.
* **Geometry**: positions as `float32` (normalised, dorsal up) + `uint8`
  class/transmitter/side codes, 15 bytes per neuron (2 MB for FlyWire).
* **Renderer**: a single WebGL2 point cloud with additive blending. The
  shader turns "time since last spike" into glow and size. The pathway pulse
  is a second particle system. No libraries.
* **Server**: `ThreadingHTTPServer` from the standard library.

### Command reference (`POST /api/command`)

| cmd | parameters |
|---|---|
| `scenario` | `name`, `replace` |
| `smell` | `odor`, `concentration`, `left`, `right` |
| `taste` | `tastant`, `intensity`, `side` |
| `vision` | `kind` (`looming`/`flash`/`spot`/`grating`/`object`), `azimuth`, ... |
| `hear`, `wind`, `touch`, `temperature` | `sound`; `direction`; `where`; `kind`, `side` |
| `stimulate` | `name`, `rate`, `side` |
| `remove` / `clear` | `id` / - |
| `silence` / `unsilence` | `name` / - |
| `play`, `pause`, `reset`, `speed` | `value` |
| `mode` | `mode` (`brain`/`fly`), `arena` |
| `fly_place`, `arena_add` | `x`, `y`, (`kind`, `odor`, `taste`) |
| `highlight`, `path`, `influence`, `neuron`, `search` | `name`; `source`, `target`; `source`; `index`; `q` |
| `challenge` | `id` |
| `job` | `kind` (`conditioning`/`screen`), options |

Anything that can talk HTTP can drive the brain, e.g. a robot, another
program or a notebook.

## Brain movies

```python
res = sim.run(500)
res.movie("sugar.html")
```

writes one self-contained HTML file (positions, spikes and the same
WebGL renderer inlined) that replays the simulation with play/pause,
scrubbing and auto-rotation. It opens in any browser, offline, and can be emailed.

## Performance tips

* Toy brain: faster than real time.
* FlyWire: install `numba`. Taste scenarios run at ~0.5x real time; large
  visual stimuli (looming, gratings) activate tens of thousands of optic-lobe
  neurons and run at ~0.05-0.2x real time. The clock shows simulated time,
  so what you see stays correct, only slower.
* Use `--transmitters curated` with FlyWire for meaningful smell responses.
