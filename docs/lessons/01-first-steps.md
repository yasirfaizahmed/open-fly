# Lesson 1: First steps

**Goal:** load a brain, find famous neurons, make the fly taste sugar and
watch it decide to eat. About 10 minutes.

## 1. Load a brain

```python
import openfly as of

brain = of.load("toy")        # instant, offline. Later: of.load("flywire_783")
print(brain)                  # <Brain 'toy': 2,284 neurons, 16,745 connections>
print(brain.summary())
```

`brain.neurons` is a table with one row per neuron; `brain.W` holds the
synapses (rows send, columns receive, negative = inhibitory).

## 2. Find neurons by name

```python
mn9 = brain["MN9"]            # the motor neuron that extends the proboscis
print(mn9)                    # <NeuronSet 'MN9': 2 neurons (CB0701 x2)>
print(mn9.df[["cell_type", "side", "nt"]])

sugar = brain["sugar_GRN"]    # sugar-sensing taste neurons
print(len(sugar), sugar.left, sugar.right)
print(brain.search("giant"))  # don't know the name? search
```

## 3. Who talks to MN9?

```python
print(brain.inputs("MN9"))
```

You'll see `G2N-1` (excitatory) and `GABA_bitter_IN` (negative `signed`
column, so inhibitory). Feeding is a tug of war between them.

## 4. Taste sugar

```python
sim = brain.simulator()
sim.add(of.senses.taste(brain, "sugar"))
res = sim.run(500)                       # 500 ms
print(res)
print(res.top(8, exclude_driven=True))   # who responded (not counting the sugar neurons themselves)
print(of.motor.describe(res.behavior()))
```

```
feeding       ########   0.78  (69 Hz)
walk_forward             0.00  (0 Hz)
...
```

## 5. Now add bitter

```python
sim = brain.simulator()
sim.add(of.senses.taste(brain, "sugar") + of.senses.taste(brain, "bitter"))
print(of.motor.describe(sim.run(500).behavior()))
```

Feeding drops and `walk_backward` rises: the fly backs away.

## 6. See it

```bash
openfly playground
```

Click **A drop of sugar water**, watch the gustatory neurons light up and
the feeding meter rise. Then try **Sugar spiked with quinine**.

**Next:** [Follow a smell](02-follow-a-smell.md)
