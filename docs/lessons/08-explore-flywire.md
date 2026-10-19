# Lesson 8: Exploring the real FlyWire brain

**Goal:** go from the cartoon to the 139,000-neuron connectome.

```python
import openfly as of
brain = of.load("flywire_783", transmitters="curated")    # first time: ~140 MB download + ~30 s build
print(brain.summary())
```

## Census

```python
print(brain.neurons.super_class.value_counts())
print(brain["KC"].counts("cell_type"))        # Kenyon cell subtypes
print(brain["descending"].counts().head(20))
print(of.analysis.hubs(brain, 10))            # the most connected neurons (CT1 tops the list)
```

## Reproduce the Shiu et al. feeding result

```python
for rate in [25, 50, 100, 200]:
    sim = brain.simulator(seed=0)
    sim.stimulate("sugar_GRN", rate)
    print(rate, "Hz ->", sim.run(500).rate("MN9"), "Hz in MN9")
```

## Everything at once

```python
sim = brain.simulator()
sim.add(of.senses.scenario(brain, "ripe_banana"))
res = sim.run(500)
print(res.by("super_class"))
res.movie("banana_flywire.html")            # share a 3D replay
```

## Wiring questions

```python
from openfly import analysis
analysis.strongest_path(brain, "sugar_GRN", "MN9")
analysis.hops(brain, "photoreceptors").value_counts().sort_index()
analysis.summarize(brain, analysis.influence(brain, "bitter_GRN"), "cell_type", top=10)
```

## Other connectomes

```python
manc = of.load("manc_1.2.1")                 # the male ventral nerve cord
hemi = of.load("hemibrain_1.2")              # the Janelia hemibrain
of.analysis.compare_brains(brain, hemi)
```

Tip: install `numba` for a ~3x faster simulator.

**Next:** [Training with evolution](09-train-with-evolution.md)
