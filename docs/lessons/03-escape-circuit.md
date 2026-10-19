# Lesson 3: The escape circuit

**Goal:** make a shadow loom, follow it through the looming detectors to
the giant fiber, and measure the reaction time.

```python
import openfly as of
from openfly import senses

brain = of.load("toy")            # also try of.load("flywire_783", transmitters="curated")
loom = senses.looming(brain, azimuth=50, duration_ms=400)
for s in loom:
    print(s)                       # retina + LC4 + LPLC2 parts
```

The disk's angular size follows `theta(t) = 2 atan(r/v / (t_collision - t))`:
slow at first, then explosive.

```python
sim = brain.simulator()
sim.add(loom)
res = sim.run(500)
print("giant fiber first spike at", res.first_spike_ms("giant_fiber"), "ms")
print(res.psth("LC4", 20).round(1).tolist())
print(res.behavior()["escape"])
```

## Is it really looming-specific?

```python
for name, st in [("flash", senses.flash(brain)), ("grating", senses.grating(brain, stop_ms=500))]:
    sim = brain.simulator(); sim.add(st)
    print(name, sim.run(500).behavior()["escape"]["drive"])
```

In the toy brain a wide-field inhibitory neuron (`CT1`) suppresses the
looming detectors for full-field changes. Silence it and see what
happens to the flash response:

```python
sim = brain.simulator(); sim.add(senses.flash(brain)); sim.silence("CT1")
print(sim.run(300).behavior()["escape"]["drive"])
```

## On the real brain

```python
fw = of.load("flywire_783", transmitters="curated")
sim = fw.simulator(); sim.add(senses.looming(fw, azimuth=50))
res = sim.run(500)
print(res.rate(fw["giant_fiber"].left), res.rate(fw["giant_fiber"].right))   # right side dominates
print(of.analysis.strongest_path(fw, "LC4", "giant_fiber"))
```

**Next:** [Virtual lesions](04-virtual-lesions.md)
