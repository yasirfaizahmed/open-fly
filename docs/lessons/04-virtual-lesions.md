# Lesson 4: Virtual lesions

**Goal:** answer "which neurons are *sufficient* to cause a behaviour, and
which are *necessary*?", the two questions behind most neuroscience
experiments.

## Sufficiency: activate one group at a time

```python
import openfly as of
from openfly import senses
from openfly.learning import activation_screen, silencing_screen

brain = of.load("toy")
print(activation_screen(brain, "MN9", top=5))
```

`G2N-1` comes out on top: activating it alone drives MN9 harder than the
sugar neurons do.

## Necessity: silence one group at a time

```python
print(silencing_screen(brain, senses.taste(brain, "sugar"), "MN9", model="rate"))
```

Silencing `G2N-1` takes MN9 to 0% of baseline. In this circuit it is the
bottleneck.

## By hand

```python
sim = brain.simulator()
sim.add(senses.taste(brain, "sugar"))
sim.silence("G2N-1")
res = sim.run(500)
print(res.rate("sugar_GRN"), res.rate("MN9"))    # sensing still works; feeding doesn't
```

## On FlyWire: a real hypothesis

```python
fw = of.load("flywire_783", transmitters="curated")
print(activation_screen(fw, "MN9", top=10, progress=True))      # ~2 min, rate model
sim = fw.simulator(); sim.add(senses.taste(fw, "sugar")); sim.silence("CB0553")
print(sim.run(500).rate("MN9"))      # ~4 Hz instead of ~80 Hz
```

The model predicts CB0553 is required for sugar-evoked proboscis
extension. A lab could test that by silencing CB0553 genetically.

**Next:** [Teach a fly](05-teach-a-fly.md)
