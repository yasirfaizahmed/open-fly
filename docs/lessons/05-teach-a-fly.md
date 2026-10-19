# Lesson 5: Teach a fly

**Goal:** give the fly a memory: banana means reward.

```python
import openfly as of
from openfly.learning import OlfactoryConditioning

brain = of.load("toy")
exp = OlfactoryConditioning(brain, cs_plus="banana", cs_minus="acetic_acid", reinforcer="sugar")
res = exp.run(n_trials=3, odor_ms=600, test_ms=400, progress=True)
print(res.summary())
```

```
Olfactory conditioning: banana (CS+) paired with sugar; acetic_acid (CS-) unpaired
  MBON valence  pre:  CS+ -0.071   CS- +0.000
  MBON valence  post: CS+ +0.920   CS- +0.000
  learning index: +0.991 (learned approach)
  MBON03 (right, glutamate) KC input now 75% of original
  ...
```

What happened: during pairing, the Kenyon cells that banana activates
fired *while* sugar excited PAM dopamine neurons. Their synapses onto the
glutamatergic (avoidance) MBONs in the PAM compartments weakened. Banana
now drives avoidance less, so it means "approach". Acetic acid uses
different Kenyon cells and is untouched.

## Punishment

```python
res = OlfactoryConditioning(brain, reinforcer="punishment").run(n_trials=3, odor_ms=600, test_ms=400)
print(res.learning_index)        # negative: learned avoidance
```

## Look inside

```python
print(res.weights.sort_values("weight_ratio").head())   # which MBONs lost KC input
print(res.curve)                                          # valence after each training trial
from openfly import viz; viz.conditioning_plot(res)
```

## On FlyWire

```python
fw = of.load("flywire_783", transmitters="curated")
res = OlfactoryConditioning(fw, reinforcer="reward").run(n_trials=3, odor_ms=800, test_ms=500, progress=True)
print(res.summary())        # learning index ~ +0.27, CS- unchanged
```

Try odours that share Kenyon cells (banana vs apple cider vinegar, 33%
overlap) and watch the memory **generalise**.

**Next:** [A fly in a world](06-closed-loop-fly.md)
