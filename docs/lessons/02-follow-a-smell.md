# Lesson 2: Follow a smell

**Goal:** trace an odour from the antenna to the learning centre and see
why the fly turns toward food.

```python
import openfly as of
from openfly import senses

brain = of.load("toy")
odor = senses.ODORS["banana"]
print(odor.description, odor.glomeruli)        # which receptor types banana activates
```

## Smell it

```python
sim = brain.simulator()
sim.add(senses.smell(brain, "banana", concentration=1.0))
res = sim.run(500)
for group in ["ORN", "ALPN", "LHN", "KC", "MBON"]:
    print(f"{group:5s} {res.rate(group):6.1f} Hz")
```

## Sparse coding in the mushroom body

```python
kc = brain["KC"].idx
print("fraction of Kenyon cells active:", (res.rates()[kc] > 0).mean())
```

Only ~10% of Kenyon cells respond (~3.5% on FlyWire with curated
transmitters). Each odour gets its own small set. Try `"acetic_acid"`
and compare *which* KCs fire:

```python
sim2 = brain.simulator()
sim2.add(senses.smell(brain, "acetic_acid"))
r2 = sim2.run(500).rates()
a, b = res.rates()[kc] > 0, r2[kc] > 0
print("overlap:", (a & b).sum() / (a | b).sum())
```

## Which way to turn?

```python
for left, right in [(1.0, 0.0), (0.0, 1.0)]:
    sim = brain.simulator()
    sim.add(senses.smell(brain, "banana", left=left, right=right))
    print(left, right, sim.run(500).behavior()["turn"])
```

Smell on the left -> negative turn (left). Smell an aversive odour
(`"geosmin"`, `"co2"`) and the sign flips, plus `walk_backward` rises.

## Trace the wiring

```python
from openfly import analysis
print(analysis.strongest_path(brain, "ORN_DM2", "DNa02"))
```

**Next:** [The escape circuit](03-escape-circuit.md)
