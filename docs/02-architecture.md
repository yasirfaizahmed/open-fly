# 2. Architecture

openfly is a stack of small layers. Each layer only depends on the ones
below it, so you can use as much or as little as you like: just the
data, the data plus the simulator, or the whole interactive playground.

```
 +--------------------------------------------------------------------------+
 |  playground/  web app: WebGL brain, lessons, challenges, lab, fly view   |  <- you, in a browser
 |  cli.py       `openfly ...` commands                                      |  <- you, in a terminal
 +--------------------------------------------------------------------------+
 |  agent/      VirtualFly + Arena (closed loop: sense -> brain -> act)      |
 |  learning/   mushroom-body plasticity, evolution strategies, screens     |
 |  viz/        matplotlib figures, standalone 3D HTML movies                |
 |  analysis/   paths, hops, information flow, influence, hubs, diffs        |
 +--------------------------------------------------------------------------+
 |  senses/     smell, taste, vision, hearing/touch, temperature, scenarios |
 |  motor.py    behaviour readout from descending / motor neurons            |
 +--------------------------------------------------------------------------+
 |  sim/        LIFEngine (NumPy) / FastLIFEngine (numba), Simulator,        |
 |              SimResult, RateModel                                         |
 +--------------------------------------------------------------------------+
 |  brain.py    Brain (neurons table + signed sparse W), NeuronSet           |
 |  knowledge.py  curated names ("MN9", "giant_fiber"...)                    |
 |  transmitters.py  sign policies (published vs curated)                    |
 +--------------------------------------------------------------------------+
 |  data/       registry, downloader, .flybrain format, adapters             |
 |              (FlyWire, Codex, neuPrint), BrainBuilder, toy brain          |
 +--------------------------------------------------------------------------+
```

## The one data structure that matters

Everything is built on `Brain`:

```
brain.neurons   pandas.DataFrame, one row per neuron (row number = model index)
                columns: root_id, x, y, z, flow, super_class, cell_class,
                         cell_sub_class, cell_type, side, nt, neuropil, ...
brain.W         scipy.sparse.csr_matrix (N x N), W[pre, post] = signed synapse count
brain.meta      dict: name, title, citation, licence, sources, ...
```

Because every connectome (FlyWire, BANC, MANC, hemibrain, your own
circuit) can be written in this form, every tool works on every brain.
New datasets only need an *adapter* that produces these two tables (see
[12-adding-new-brains.md](12-adding-new-brains.md)).

## Data flow of a simulation

```
 senses.smell(brain, "banana")  ──>  Stimulus(targets=ORN indices, rates=Hz, window)
                                             │
 Simulator.add(...) / .silence(...)          ▼
          ┌───────────────────────── Simulator.advance(ms) ─────────────────────────┐
          │  every update: sum active stimuli -> engine.replace_drive(idx, rates)    │
          │  engine.run(n_steps):  integrate -> threshold -> deliver -> reset       │
          └────────────────────────────────────┬──────────────────────────────────┘
                                               ▼
                         spikes (step, neuron)  ──>  SimResult  ──>  rates, top(), psth(),
                                                                     behavior(), movie()
```

## Key design decisions

**1. Our own engine, step by step.** The published model runs in Brian2,
which is excellent but compiles a whole network per run and is hard to
pause mid-way. openfly re-implements the exact same equations as an
explicit step function. That gives:

* *interactivity*: the playground and virtual fly advance the brain 20 ms
  at a time and change stimuli between chunks;
* *plasticity hooks*: learning rules can run after every time step;
* *speed*: an **awake set** means neurons at rest cost nothing (the
  validation run is ~20x faster than Brian2 on the same machine);
* *no compiler needed*: pure NumPy by default, numba optional.

The engine reproduces Brian2's results (per-neuron firing-rate correlation
r = 0.9996 with the published reference simulation; see
[13-validation.md](13-validation.md)).

**2. Names, not numbers.** Neuroscientists talk about "MN9" or "sugar
neurons", not row 39,871. `brain["MN9"]` resolves curated aliases, cell
types, classes and IDs, and suggests close matches when you mistype.

**3. Real data by default, cartoon for learning.** `openfly.load()` gives
you the real FlyWire brain (downloaded once, then cached as a 52 MB
`.flybrain` file that loads in about a second). `openfly.load("toy")` gives a
2,300-neuron hand-wired cartoon with the same names, which is instant and offline.
That is ideal for lessons, tests and figuring things out.

**4. Honest knobs.** Wherever the model needs a simplification (sign
corrections, the virtual fly's exploration drive, estimated retinotopy)
it is a documented, switchable parameter, with the published behaviour
available.

**5. Zero-dependency front end.** The playground is plain HTML/JS with a
hand-written WebGL2 renderer, served by Python's standard library. No
Node, no build step, no CDN. It works offline and on a laptop.

## Module map

| module | role | doc |
|---|---|---|
| `openfly.data` | datasets, downloads, `.flybrain`, adapters, BrainBuilder, toy | [03](03-data-and-brain-files.md) |
| `openfly.brain`, `openfly.neurons` | Brain and NeuronSet | [04](04-brain-api.md) |
| `openfly.knowledge` | curated aliases | [04](04-brain-api.md) |
| `openfly.sim` | engines, Simulator, SimResult, RateModel, params | [05](05-simulation.md) |
| `openfly.transmitters` | sign policies | [05](05-simulation.md) |
| `openfly.senses` | stimuli for every sense, scenarios | [06](06-senses-and-stimuli.md) |
| `openfly.motor` | behaviour readout | [07](07-behavior-readout.md) |
| `openfly.playground` | web app, session, lessons | [08](08-playground.md) |
| `openfly.agent` | virtual fly, arena | [09](09-virtual-fly.md) |
| `openfly.learning` | plasticity, ES training, screens | [10](10-training-the-brain.md) |
| `openfly.analysis` | graph tools, comparisons | [11](11-analysis-toolkit.md) |
| `openfly.viz` | plots, HTML movies | [08](08-playground.md#brain-movies) |
| `openfly.cli` | command line | [README](../README.md#command-line) |

## Dependencies

* required: `numpy`, `scipy`, `pandas`, `pyarrow`
* optional: `numba` (~3x faster engine), `matplotlib` (figures),
  `brian2` (to re-run the original reference model), `pytest`
