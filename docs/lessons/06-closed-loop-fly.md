# Lesson 6: A fly in a world

**Goal:** put the brain in a body and let it forage.

```python
import openfly as of
from openfly.agent import VirtualFly, Arena

brain = of.load("toy")
fly = VirtualFly(brain, "banana", seed=0)
traj = fly.run(25, progress=True)
print(fly.events)
fly.plot()                          # needs matplotlib
```

Each 20 ms the fly smells with two antennae, tastes what it stands on,
sees the banana, thinks (whole-brain simulation) and moves according to
its descending neurons.

## Experiments

```python
for arena in Arena.PRESETS:
    fly = VirtualFly(brain, arena, seed=1)
    fly.run(20)
    print(arena, round(fly.energy, 2), [e["event"] for e in fly.events][:3])
```

* `bitter_trap`: the fly walks straight to the vinegar but never feeds.
* `predator`: a looming attack at 4 s triggers the giant fiber and an escape jump.
* `choice`: it avoids the mouldy fruit.

## Brain surgery in closed loop

```python
fly = VirtualFly(brain, "banana", seed=0)
fly.sim.silence("LHAV1a3")       # the toy's attractive lateral-horn neurons
fly.run(25)
print(fly.energy, fly.distance_to())   # without innate attraction the fly rarely finds the banana
```

## Purely brain-driven

`VirtualFly(..., explore=False, klinokinesis=0)` removes the body-level
exploration and run-and-turn rule. Then only the connectome moves the fly.

In the playground: **Virtual fly**, pick an arena, click to drop food,
press **Predator!**.

**Next:** [Build your own brain](07-build-your-own-brain.md)
