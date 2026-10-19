# Lesson 9: Training with evolution

**Goal:** tune a circuit for a task without touching its wiring.

## The task

Make the fly **reject sugar that's been spiked with a little bitter**,
while still eating plain sugar.

```python
import openfly as of
from openfly import senses
from openfly.learning import GainTask

brain = of.load("toy")
conditions = {
    "sugar": senses.taste(brain, "sugar"),
    "laced": senses.taste(brain, "sugar") + senses.taste(brain, "bitter", 0.25),
}
task = GainTask(
    brain,
    groups=["G2N-1", "GABA_bitter_IN", "LB3c", "LB1a"],   # trainable: one output gain each
    conditions=conditions,
    readouts={"MN9": "MN9"},
    objective=lambda r: r["sugar"]["MN9"] / 100 - r["laced"]["MN9"] / 100,
)
print(task.evaluate([1, 1, 1, 1]))       # before training: laced sugar still drives MN9 (~52 Hz)
res = task.train(generations=25, progress=True)
print(res.summary())
```

The optimiser discovers it should **amplify the bitter pathway**
(`GABA_bitter_IN` and `LB1a` gains go up), not silence sugar.

## How it works

Evolution strategies: try a population of random gain perturbations,
score each by simulating every condition, move the gains toward the better
ones (Adam step), repeat. No gradients are needed, so the same code trains
spiking models:

```python
task_lif = GainTask(brain, ["G2N-1", "GABA_bitter_IN"], conditions, {"MN9": "MN9"},
                    task.objective, model="lif", duration_ms=300)
print(task_lif.train(generations=8, popsize=8).summary())
```

## Impossible tasks

Ask for something the wiring can't do, e.g. "feed on water but not on sugar"
(both go through the same relay in the toy brain). The optimiser plateaus
and adds a note: *"no improvement: the wiring may not allow this
behaviour with these groups"*. That is an answer about the circuit.

## Your turn

* Make the fly turn **away** from banana (groups along ORN -> LH -> LAL -> DNa02).
* Reduce the escape response to looming (habituation).
* On FlyWire, use `analysis.strongest_path` or `activation_screen` to pick
  the groups, and `model="rate"` for speed.
