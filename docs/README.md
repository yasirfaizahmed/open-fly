# openfly documentation

Welcome! These pages explain **what** every part of openfly does, **why**
it was built that way, and **how** to use it. They are written so that
an engineer with no neuroscience background can follow along. Each one
starts from intuition and ends with code.

## Suggested reading order

| # | Page | What you'll learn |
|---|------|-------------------|
| 1 | [The fly brain and its connectome](01-the-fly-brain-and-connectome.md) | How the brain "file" was made (slicing, imaging, AI tracing), fly brain anatomy, glossary |
| 2 | [Architecture](02-architecture.md) | How the SDK is layered, data flow, design decisions |
| 3 | [Data and brain files](03-data-and-brain-files.md) | Datasets, downloads, the `.flybrain` format, the toy brain, BrainBuilder |
| 4 | [Brain API](04-brain-api.md) | Selecting neurons by name, set algebra, wiring queries |
| 5 | [Simulation](05-simulation.md) | The spiking model, its equations, engines, results, transmitter policies |
| 6 | [Senses and stimuli](06-senses-and-stimuli.md) | Smell, taste, vision, hearing, touch, temperature, scenarios |
| 7 | [Behaviour readout](07-behavior-readout.md) | Turning motor-neuron spikes into "feeding", "walking", "escape" |
| 8 | [The playground](08-playground.md) | The interactive 3D web app: tour, lessons, challenges, API |
| 9 | [The virtual fly](09-virtual-fly.md) | Closed loop: brain + body + world |
| 10 | [Training the brain](10-training-the-brain.md) | Can a connectome learn? Dopamine plasticity, optimisation, screens |
| 11 | [Analysis toolkit](11-analysis-toolkit.md) | Pathways, distances, influence, hubs, comparisons |
| 12 | [Adding new brains](12-adding-new-brains.md) | Loading future releases and other connectomes |
| 13 | [Validation](13-validation.md) | How we know the simulator is right |
| 14 | [Findings and limitations](14-findings-and-limitations.md) | What exploring FlyWire with openfly turned up, and what the model can't do |
| 15 | [Build journal](15-build-journal.md) | How openfly was built: decisions, bugs found, dead ends |
| 16 | [FAQ](16-faq.md) | Quick answers |

## Hands-on lessons

Short, runnable tutorials in [`lessons/`](lessons/):

1. [First steps](lessons/01-first-steps.md): load a brain, find neurons, taste sugar
2. [Follow a smell](lessons/02-follow-a-smell.md): from odour molecules to the mushroom body
3. [The escape circuit](lessons/03-escape-circuit.md): looming, LC4 and the giant fiber
4. [Virtual lesions](lessons/04-virtual-lesions.md): necessity and sufficiency screens
5. [Teach a fly](lessons/05-teach-a-fly.md): olfactory conditioning with dopamine
6. [A fly in a world](lessons/06-closed-loop-fly.md): the virtual fly forages
7. [Build your own brain](lessons/07-build-your-own-brain.md): BrainBuilder from scratch
8. [Exploring the real FlyWire brain](lessons/08-explore-flywire.md)
9. [Training with evolution](lessons/09-train-with-evolution.md)

Runnable scripts live in [`examples/`](../examples/).

## Conventions used everywhere

* **Model index**: neurons are numbered `0..N-1` (row of `brain.neurons`).
  **IDs** (`root_id`) are the dataset's own identifiers (e.g. FlyWire root IDs).
* **Weights**: `brain.W[pre, post]` = signed synapse count (negative = inhibitory).
* **Units**: positions in micrometres, time in milliseconds, rates in Hz, voltages in mV.
* **Sides**: `left` / `right` are the *fly's* left and right.
