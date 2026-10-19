# Lesson 7: Build your own brain

**Goal:** design a circuit from scratch, simulate it and explore it in 3D:
the sandbox.

We'll build a two-sided "Braitenberg fly" whose sugar sensors cross over
to motor neurons, so it turns toward food.

```python
import openfly as of
from openfly import BrainBuilder

b = BrainBuilder("braitenberg", seed=0)
L_sense = b.add("LB3c", 10, side="left",  super_class="sensory", cell_class="gustatory",
                cell_sub_class="sugar", region="GNG", center=(480, 330, 120))
R_sense = b.add("LB3c", 10, side="right", super_class="sensory", cell_class="gustatory",
                cell_sub_class="sugar", region="GNG", center=(576, 330, 120))
L_dn = b.add("DNa02", 1, side="left",  super_class="descending", center=(470, 200, 100))
R_dn = b.add("DNa02", 1, side="right", super_class="descending", center=(586, 200, 100))
inh  = b.add("LAL_inh", 4, nt="gaba", center=(528, 210, 100))

b.connect(L_sense, L_dn, synapses=18)        # same-side excitation
b.connect(R_sense, R_dn, synapses=18)
b.connect(L_sense, inh, synapses=6)
b.connect(inh, R_dn, synapses=40)            # cross inhibition sharpens the contrast
brain = b.build()
print(brain.summary())
```

## Simulate

```python
sim = brain.simulator()
sim.add(of.senses.taste(brain, "sugar", side="left"))
res = sim.run(500)
print(res.behavior()["turn"])               # turns left
```

## Calibrate

The LIF model needs a summed drive (rate x synapses) of ~3,000 Hz*syn to
fire and ~12,000 for ~50 Hz (table in [03](../03-data-and-brain-files.md#calibrating-a-hand-made-brain)).
Here 10 neurons x 150 Hz x 18 synapses = 27,000, which is plenty.

## Save, share, view

```python
brain.save("braitenberg.flybrain")
```

```bash
openfly playground -d braitenberg.flybrain
```

Everything works on your brain: senses (they find neurons by FlyWire
names), behaviour readouts, analysis, learning (add `KC`/`MBON`/`DAN` cell
classes), the virtual fly.

**Next:** [Exploring the real FlyWire brain](08-explore-flywire.md)
