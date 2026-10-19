# 10. Training the brain

Code: `openfly/learning/` (`mushroom_body.py`, `evolve.py`, `screens.py`)

> *"Can we train the brain to do certain tasks? I'm not sure if this is possible."*

**Yes, in three different senses**, and the difference matters:

| | what changes | how | biological? | openfly |
|---|---|---|---|---|
| **1. Learning the way flies learn** | KC -> MBON synapse strengths | dopamine-gated plasticity during experience | yes: this is how flies form memories | `MushroomBodyPlasticity`, `OlfactoryConditioning` |
| **2. Optimising the circuit for a task** | gains of chosen cell types (never the wiring) | evolution strategies on any objective | as a model of neuromodulation or development, not of experience | `GainTask`, `EvolutionStrategy` |
| **3. Finding the controls** | nothing; you search | activate / silence each candidate group | it's what biologists do with genetics | `activation_screen`, `silencing_screen` |

What you *cannot* do is make the fly do something its wiring has no route
for: training never adds connections. When a task is impossible, the
optimiser plateaus, which is itself a finding about the circuit.

## 1. Learning the way flies learn: the mushroom body

The biology, in four steps:

1. An odour activates a small, odour-specific set of the ~5,000 **Kenyon
   cells** (KCs), about 5%.
2. KCs synapse onto ~35 types of **MBONs** (output neurons). Glutamatergic
   MBONs push the fly to *avoid*; cholinergic and GABAergic ones to *approach*
   (Aso et al. 2014).
3. **Dopaminergic neurons** carry teaching signals: PAM DANs for reward
   (sugar), PPL1 DANs for punishment. Each DAN type innervates one
   *compartment*, reaching specific MBONs.
4. When KCs fire **while** dopamine arrives, their synapses onto that
   compartment's MBON weaken (Hige et al. 2015).

After reward, the odour's KCs drive the avoidance MBONs less, so the fly approaches.

openfly implements step 4 as a **three-factor rule** on the real connectome:

```
dw/dt = -lr x kc_trace[pre] x max(dopamine[post] - threshold, 0) x w0,   w in [w_min x w0, w0]
```

* `kc_trace`, `da_trace`: exponentially filtered spike counts (tau = 60 ms);
* `dopamine[m]` = synapse-weighted mean trace of the DANs that contact MBON m.
  **Compartments come from the wiring, not from a lookup table.** In FlyWire
  the DAN -> MBON synapses reproduce the textbook map: PPL1-gamma1 ->
  MBON11 (gamma1pedc), PAM clusters -> MBON01-09 (mostly glutamatergic),
  PPL1 -> MBON11-19 (mostly cholinergic).

```python
from openfly.learning import OlfactoryConditioning
brain = openfly.load("flywire_783", transmitters="curated")
exp = OlfactoryConditioning(brain, cs_plus="banana", cs_minus="acetic_acid", reinforcer="reward")
res = exp.run(n_trials=3, odor_ms=800, test_ms=500)
print(res.summary())
```

The protocol is differential conditioning (Tully & Quinn 1985 in silico):
pre-test both odours; pair CS+ with the reinforcer and present CS- alone;
post-test. Reinforcers: `"reward"` / `"punishment"` drive PAM / PPL1
directly (as in optogenetic memory-implantation experiments, Claridge-Chang
et al. 2009; Aso & Rubin 2016), or `"sugar"` / `"bitter"` go through taste.

### Results

**FlyWire v783 (curated transmitters), banana (CS+) vs acetic acid (CS-),
3 trials.** The two odours share only 3% of their Kenyon cells:

| reinforcer | CS+ valence change | CS- valence change | learning index |
|---|---|---|---|
| reward (PAM) | **+0.27** | -0.00 | **+0.27 (approach)** |
| punishment (PPL1) | **-0.28** | -0.07 | **-0.21 (avoidance)** |

With reward, banana-evoked firing of PAM-compartment MBONs (MBON02, MBON07)
dropped to 0 while acetic acid was unaffected: an **odour-specific memory**.
With overlapping odours (banana vs apple cider vinegar, 33% KC overlap) the
memory *generalises* to the CS-, as it does in real flies.

**Toy brain:** learning index +0.99 (sugar), -0.87 (punishment).

Two lessons from building this on the real brain (both documented options):
in the whole-brain model, odours alone excite some DANs a little, so a
dopamine threshold keeps those from writing memories
(`da_threshold`), and learning is switched off during CS- presentations
(`learn_during_cs_minus=False`).

Sugar taste does **not** reach PAM neurons in the FlyWire LIF model, so
`"sugar"` as a reinforcer only works on the toy brain. That is a prediction
(or a model limitation) worth knowing.

## 2. Optimising a circuit for a task: evolution strategies

`GainTask` learns one **output gain** per chosen neuron group (gain 1 = as
wired) so that a set of *conditions* produces the *readouts* your
*objective* rewards. It uses OpenAI-style evolution strategies (antithetic
sampling, rank shaping, Adam), which need only forward simulations, so they
work with the spiking simulator (`model="lif"`) or the fast rate model (`model="rate"`).

Example: *make the fly refuse sugar spiked with a little bitter* (as
starved flies become more tolerant and sated ones less):

```python
from openfly.learning import GainTask
conds = {"sugar": senses.taste(brain, "sugar"),
         "laced": senses.taste(brain, "sugar") + senses.taste(brain, "bitter", 0.25)}
task = GainTask(brain, groups=["G2N-1", "GABA_bitter_IN", "LB3c", "LB1a"], conditions=conds,
                readouts={"MN9": "MN9"},
                objective=lambda r: r["sugar"]["MN9"] / 100 - r["laced"]["MN9"] / 100)
res = task.train(generations=25)
print(res.summary())
```

On the toy brain (1 s with the rate model) the objective goes from 0.13 to 1.37: the
optimiser **turns up the bitter pathway** (`GABA_bitter_IN` gain 4.0, `LB1a`
3.9), so laced sugar no longer triggers feeding (52 -> 0 Hz) while plain sugar
still does. With `model="lif"` it finds the same direction (objective 0.08 -> 0.43
in 8 generations, ~25 s).

This is the idea behind connectome-constrained models (Lappalainen et al.
2024 trained a connectome-constrained visual system to compute motion).
Ideas to try: train turning toward an odour on the left (groups along
ORN -> LH -> LAL -> DNa02), habituate the escape response, or wrap
`VirtualFly` in an objective and evolve foraging.

## 3. Finding the controls: virtual genetic screens

```python
from openfly.learning import activation_screen, silencing_screen
activation_screen(brain, "MN9", top=10)                         # sufficiency
silencing_screen(brain, senses.taste(brain, "sugar"), "MN9")    # necessity
```

On FlyWire, the activation screen (200 upstream cell types, rate model,
~2 min) ranks **CB0553** as the strongest driver of MN9. The silencing
screen says it is also **necessary**: with CB0553 silenced, sugar-evoked
MN9 firing drops from ~80 Hz to ~4 Hz in the spiking model. That is a
concrete, testable circuit hypothesis produced in minutes, the kind of
result Shiu et al. confirmed in real flies for other neurons.

## What would it take to go further?

* **Reinforcement learning in closed loop**: let dopamine come from the
  virtual fly's experience (food found = PAM burst) and watch it learn to
  prefer an odour while foraging. All the pieces exist: `VirtualFly`,
  `MushroomBodyPlasticity`, an `on_step` hook.
* **Gradient-based training** of a differentiable rate model (e.g. in
  PyTorch) for many parameters at once.
* **Data-constrained objectives**: fit gains so simulated responses match
  recorded calcium imaging.
