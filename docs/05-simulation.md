# 5. Simulation

Code: `openfly/sim/` (`params.py`, `engine.py`, `fast.py`, `simulator.py`,
`results.py`, `rate.py`) and `openfly/transmitters.py`

## The model in one picture

Every neuron is a **leaky integrate-and-fire** (LIF) unit, the model of
Shiu et al. (2024):

```
   presynaptic spike ──(1.8 ms delay)──>  g += w_syn x (signed synapse count)
                                           │
              dv/dt = (v_rest - v + g) / tau_m       (membrane, tau_m = 20 ms)
              dg/dt = -g / tau_syn                   (synaptic drive, tau_syn = 5 ms)
                                           │
              v > v_thresh  ->  SPIKE, v = v_reset, g = 0, refractory 2.2 ms
```

| parameter | value | source |
|---|---|---|
| `v_rest`, `v_reset` | -52 mV | Kakaria & de Bivort 2017 |
| `v_thresh` | -45 mV | Kakaria & de Bivort 2017 |
| `tau_m` | 20 ms | Kakaria & de Bivort 2017 |
| `tau_syn` | 5 ms | Jürgensen et al. 2021 |
| `t_refrac_ms` | 2.2 ms | Lazar et al. 2021 |
| `delay_ms` | 1.8 ms | Paul et al. 2015 |
| `w_syn` | 0.275 mV per synapse | the model's one free parameter, fitted to behaviour |
| `poisson_scale` | 250 | Poisson input kicks v by 68.75 mV, forcing a spike |
| `dt_ms` | 0.1 ms | Brian2 default |

Change any of them with `LIFParams`:

```python
from openfly import SHIU_2024
p = SHIU_2024.with_(w_syn=0.3, tau_m=15.0)
sim = brain.simulator(params=p)
```

Sign convention: a connection is excitatory unless the presynaptic
neuron's predicted transmitter is GABA or glutamate (glutamate is
inhibitory at most fly central synapses via GluCl). Dopamine,
serotonin and octopamine are treated as excitatory, as in the paper.

### A subtle detail we had to reproduce

In the original Brian2 code the conductance `g` is flagged
"(unless refractory)". Testing it directly against Brian2 2.10 showed that
**synaptic input arriving while a neuron is refractory is lost**. Without
reproducing that, strongly driven neurons fire 20-30% faster than in the
paper. openfly reproduces it by default
(`LIFParams.drop_input_when_refractory=True`); set it to `False` to explore
the alternative. See [13-validation.md](13-validation.md).

## Engines

`LIFEngine` (pure NumPy) advances the network one 0.1 ms step at a time
using **exact integration** of the linear ODEs:

```
v(t+dt) = v_inf + (v - v_inf) e^(-dt/tau_m) + g * tau_s/(tau_s - tau_m) * (e^(-dt/tau_s) - e^(-dt/tau_m))
g(t+dt) = g e^(-dt/tau_s)
```

Within each step the order matches Brian2's schedule: state update, then
threshold, then synaptic delivery (and Poisson input), then reset.

**The awake set.** Almost every neuron sits exactly at rest (`v = v_rest`,
`g = 0`), where the update does nothing. The engine only integrates the
neurons that received input recently and puts them back to sleep once
they relax to within 1e-4 mV of rest. On the full brain that is typically
a few thousand to tens of thousands of neurons instead of 139,000.

`FastLIFEngine` (`pip install numba`) runs the same algorithm as compiled
loops with a dense delay ring buffer. It is picked automatically when
numba is installed (`backend="auto"`), or chosen with `backend="numpy"` / `"numba"`.

Speed on the full FlyWire brain (4-core laptop-class CPU, seconds of
compute per simulated second):

| workload | NumPy | numba |
|---|---|---|
| sugar taste (~500 neurons firing) | 5.7 s | 1.7 s |
| banana smell + sight (~1.4k firing, ~50k awake) | 14 s | 5 s |
| Brian2 (original code, C++ codegen), same sugar run | ~120 s | |

The toy brain runs faster than real time with either engine.

## Simulator

```python
sim = brain.simulator(seed=0)                 # or openfly.Simulator(brain, params=..., backend=...)
sim.stimulate("sugar_GRN", rate_hz=100)       # Poisson drive (optogenetic-like)
sim.stimulate("MN9", 5.0, kind="current")     # +5 mV steady depolarisation instead
sim.add(openfly.senses.smell(brain, "banana"))  # stimuli from the senses module
sim.silence("CB0553")                         # remove a cell type's output
res = sim.run(duration_ms=1000, n_trials=10, n_jobs=4)
```

* Stimuli have time windows (`start_ms`, `stop_ms`) and can be time-varying
  (`modulation`). Static ones are applied once; dynamic ones every
  `update_ms` (2 ms).
* `run()` resets the brain before each trial; trial seeds are derived from
  `seed`, so results are reproducible (also with `n_jobs > 1`).
* `advance(ms)` continues the current simulation without resetting. The
  playground and the virtual fly use this. `stream(chunk_ms)` is a generator version.
* `on_step(fn)` runs `fn(engine, spikes)` after every time step. Learning rules hook in here.
* `openfly.simulate(brain, stimuli, silence=..., duration_ms=...)` is a one-liner.

## Results

```python
res.rates()                   # Hz for every neuron (mean over trials)
res.rate("MN9")               # mean rate of a group
res.top(15, exclude_driven=True)            # most active neurons, without the stimulated ones
res.top(10, by="cell_type")
res.by("super_class")         # how many neurons fired in each class
res.psth("giant_fiber", 5)    # rate over time
res.first_spike_ms("giant_fiber")
res.compare(other_result)     # biggest rate changes between two experiments
res.behavior()                # feeding / walking / turning / escape drives (see 07)
res.spikes("LC4")             # DataFrame of spikes
res.save("run.parquet"); SimResult.load("run.parquet", brain)
res.raster(); res.movie("run.html")
```

## The rate model: fast approximate answers

`RateModel` replaces spikes with firing rates and relaxes the network to
a fixed point:

```
rate = r_max * tanh(k * max(D - theta, 0) / r_max),   D = sum_pre rate_pre * signed_synapses
k = 0.0045, theta = 1000, r_max = 150 Hz      (fitted to the LIF input-output curve)
```

Driven neurons are clamped to their stimulus rate. On FlyWire it takes
about 1 s and agrees with the spiking model to r = 0.96-0.98 across neurons
for static stimuli. It ignores timing, so it can't represent looming. Screens and
evolution strategies use it to evaluate thousands of variants.

```python
from openfly.sim.rate import RateModel
rm = RateModel(brain)
rates = rm.run(openfly.senses.taste(brain, "sugar"))
```

## Transmitter policies: when predictions go wrong

Predicted neurotransmitters are very good overall, but exploring the model
with openfly turned up a case where errors dominate behaviour:

> In the published sign assignment, about 130 **antennal-lobe local neurons**
> are predicted to release serotonin, dopamine or acetylcholine (so they
> excite), and ~650 **olfactory receptor neurons** are predicted
> serotonergic. In real flies most of these LNs are GABAergic (Wilson &
> Laurent 2005; Chou et al. 2010) and ORNs are cholinergic. Because the LNs
> are densely interconnected, **any odour sets off a runaway loop**: ~8,000
> neurons fire, LNs reach 260 Hz, the APL neuron 300 Hz.

`openfly.transmitters` offers two policies:

| policy | what |
|---|---|
| `"shiu2024"` (default) | exactly the published signs, for comparability with the paper |
| `"curated"` | sensory neurons with monoamine predictions -> acetylcholine; antennal-lobe LNs with ACh/monoamine predictions -> GABA |

With `"curated"`, banana activates ~1,300 neurons and **~3.5% of Kenyon
cells**, close to the 5-10% measured in real flies, and the mushroom body
behaves (see [10](10-training-the-brain.md)). Taste and looming results
are essentially unchanged.

```python
brain = openfly.load("flywire_783", transmitters="curated")
openfly.transmitters.report(raw_brain)      # what would change, and why
custom = [openfly.transmitters.Override({"cell_type": "CB0553"}, "gaba", "what if it inhibits?")]
what_if = openfly.transmitters.apply(raw_brain, custom)
```

Every override records its reason in `brain.neurons.nt_note`.
