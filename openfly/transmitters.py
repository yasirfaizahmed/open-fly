"""Neurotransmitter signs, and how to correct them.

In the Shiu et al. (2024) model, a connection is excitatory or
inhibitory depending on the presynaptic neuron's *predicted*
neurotransmitter (Eckstein et al. 2024): GABA and glutamate inhibit,
everything else excites. The predictions are very good overall but not
perfect, and a few errors matter a lot.

**The antennal-lobe problem.** Most antennal-lobe local neurons (LNs)
are GABAergic in real flies (Wilson & Laurent 2005; Chou et al. 2010),
but the predictor calls ~130 of them serotonergic, dopaminergic or
cholinergic. Those LNs are densely interconnected, so in the published
model any olfactory input sets off a runaway loop: thousands of neurons
fire at hundreds of Hz. Many olfactory receptor neurons are also
predicted serotonergic, although ORNs are cholinergic.

``openfly`` keeps the published signs by default (``"shiu2024"``), so
results stay comparable with the paper, and offers a literature-based
``"curated"`` policy that fixes these cases. Every override is listed
with its reason, and you can write your own to test a hypothesis, e.g.
"what if this neuron were inhibitory?". See ``docs/05-simulation.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INHIBITORY = {"gaba", "glutamate"}


@dataclass(frozen=True)
class Override:
    selector: dict
    nt: str
    reason: str


CURATED = [
    Override(
        {"super_class": ["sensory"], "nt": ["serotonin", "dopamine", "octopamine"]},
        "acetylcholine",
        "Sensory neurons (ORNs, GRNs, mechanoreceptors) are cholinergic; monoamine predictions there are "
        "likely errors (Eckstein et al. 2024 report lower accuracy for sensory neurons).",
    ),
    Override(
        {"cell_class": "ALLN", "nt": ["serotonin", "dopamine", "octopamine", "acetylcholine"]},
        "gaba",
        "Most antennal-lobe local neurons are GABAergic (Wilson & Laurent 2005; Chou et al. 2010). "
        "Treating them as excitatory makes the antennal lobe explode in the LIF model.",
    ),
]

POLICIES = {"shiu2024": [], "curated": CURATED}


def apply(brain, policy="curated", name: str | None = None):
    """Return a copy of ``brain`` with transmitter overrides applied.

    ``policy`` is ``"shiu2024"`` (no change), ``"curated"``, or a list of
    :class:`Override`. Connection weights of every overridden neuron are
    re-signed; ``neurons["nt"]`` is updated and ``neurons["nt_note"]``
    records why.
    """
    from .brain import Brain

    rules = POLICIES[policy] if isinstance(policy, str) else list(policy)
    neurons = brain.neurons.copy()
    if "nt_note" not in neurons:
        neurons["nt_note"] = ""
    W = brain.W.copy()
    changed = np.zeros(brain.N, dtype=bool)
    for rule in rules:
        m = brain._mask(rule.selector) & ~changed
        idx = np.flatnonzero(m)
        if idx.size == 0:
            continue
        neurons.loc[idx, "nt"] = rule.nt
        neurons.loc[idx, "nt_note"] = rule.reason[:120]
        changed[idx] = True
    if changed.any():
        target = np.where(neurons["nt"].isin(INHIBITORY).to_numpy(), -1.0, 1.0).astype(np.float32)
        rows = np.repeat(np.arange(brain.N), np.diff(W.indptr))
        flip = changed[rows]
        W.data[flip] = np.abs(W.data[flip]) * target[rows[flip]]
    label = policy if isinstance(policy, str) else "custom"
    meta = dict(brain.meta, name=name or brain.name, transmitter_policy=label,
                transmitter_overrides=int(changed.sum()))
    return Brain(neurons, W, meta)


def report(brain, policy="curated"):
    """Table of the neurons a policy would change, grouped by class and predicted transmitter."""
    import pandas as pd

    rules = POLICIES[policy] if isinstance(policy, str) else list(policy)
    rows = []
    done = np.zeros(brain.N, dtype=bool)
    for rule in rules:
        m = brain._mask(rule.selector) & ~done
        done |= m
        sub = brain.neurons[m]
        for (cls, nt), n in sub.groupby(["cell_class", "nt"]).size().items():
            rows.append({"cell_class": cls, "predicted_nt": nt, "new_nt": rule.nt, "neurons": int(n), "reason": rule.reason})
    return pd.DataFrame(rows)
