# 16. FAQ

**Is this the real fly brain?**
`openfly.load("flywire_783")` is the real FlyWire connectome: every
reconstructed neuron and connection, with published annotations. The
*simulation* is a model of it: simple spiking neurons wired exactly as
the data says. `openfly.load("toy")` is a hand-made cartoon for learning.

**How big is it? Will it run on my laptop?**
~140 MB download once, 52 MB cached brain file, ~1 GB RAM while
simulating. Any recent laptop works; `pip install numba` makes it ~3x faster.

**Why is it slower than real time?**
139,000 neurons x 10,000 time steps per second. Taste experiments run
at ~0.5x real time with numba, large visual stimuli (tens of thousands of
optic-lobe neurons active) at ~0.05-0.2x. The toy brain is faster than real time.

**GPU?**
Not yet. The engine is event-driven (only active neurons are updated),
which suits CPUs. A GPU backend is a good contribution.

**How accurate is it?**
It reproduces the published model (r = 0.9996, [13](13-validation.md)).
How well that model matches real flies is a research question. It predicted
real feeding and grooming circuits in Shiu et al. 2024, but see
[14](14-findings-and-limitations.md) for its limits.

**Can I use other connectomes?**
Yes: BANC, MANC, male CNS, male optic lobe and hemibrain are registered;
any Codex release loads with `codex:<dataset>/<version>`; anything else
via two tables ([12](12-adding-new-brains.md)).

**Can it learn?**
Yes, in four ways: biological mushroom-body plasticity, optimisation of
circuit gains for a task, virtual genetic screens to find the controls,
and trained population decoders that read intentions ([10](10-training-the-brain.md)).

**Why do my smells look weird on FlyWire?**
Use `transmitters="curated"`. The published signs make the antennal lobe
explode ([05](05-simulation.md#transmitter-policies-when-predictions-go-wrong)).

**Where are files stored? Can I work offline?**
`~/.openfly` (or `$OPENFLY_HOME`). After the first download everything is
offline. `OPENFLY_DATA_DIR` points openfly at raw files you already have.

**Licence?**
Code: MIT. Data: the publishers' terms. Please cite them ([CITATION.md](../CITATION.md)).
