# 11. Analysis toolkit

Code: `openfly/analysis/` (`graph.py`, `compare.py`)

Questions about the wiring itself, no simulation needed. Weights are
**input fractions**: a connection i->j counts as `|w_ij| / (total input
synapses of j)`. Ten synapses matter a lot to a small neuron and very
little to a giant one.

## How does a signal get from A to B?

```python
from openfly import analysis
analysis.strongest_path(brain, "sugar_GRN", "MN9")
```

Dijkstra on costs `-log(input fraction)`, which maximises the product of
input fractions along the path. Connections below `min_fraction` (1%) are
ignored. On FlyWire:

```
LB3c (sugar GRN) -> CB0499 (35 syn, 1.5% of its input) -> CB0465 (GABA, 26 syn) -> CB0701 = MN9 (387 syn, 6.6%)
LC4 -> DNp04 -> PVLP010 -> DNp01 (giant fiber)
```

The playground's **Pathway finder** draws this as a pulse moving through the brain.

## How far is everything from the eyes?

```python
h = analysis.hops(brain, "photoreceptors", min_synapses=5)
h.value_counts().sort_index()
```

On FlyWire: 10,582 photoreceptors (0 hops), 6,783 neurons 1 hop away,
34,216 at 2, 51,542 at 3, 25,487 at 4, 4,606 at 5, then a thin tail out to 8.

## Information flow depth

`analysis.traversal(brain, source)` is the probabilistic traversal model
of Schlegel et al. (2021): each step, an inactive neuron becomes active with
probability `min(1, input fraction from active neurons / 0.3)`. The mean
activation step is a smooth "depth" in the sensory hierarchy.

## Who feels it, and with which sign?

```python
infl = analysis.influence(brain, "bitter_GRN", n_hops=3)    # signed, summed over 1..3-synapse paths
analysis.summarize(brain, infl, by="cell_type", top=10)
```

Effective connectivity: `v_{k+1} = F^T v_k` with the signed input-fraction
matrix. Positive = net excitation, negative = net inhibition. In the
playground this is the **Ripple map**.

## Hubs, matrices, reciprocity

```python
analysis.hubs(brain, top=10, by="out")    # FlyWire's biggest: CT1 (~100k output synapses each), Am1, cM15
analysis.type_matrix(brain, by="super_class", normalize="row")
analysis.reciprocity(brain, "KC")
```

## Comparing brains

For a new release vs the old one, female vs male, or your edit vs the original:

```python
analysis.compare_brains(flywire, banc, by="super_class")    # counts and synapses side by side
analysis.shared_types(flywire, mcns)                        # cell types in both / only one
analysis.type_wiring_similarity(flywire, mcns)              # per type: cosine similarity of output profiles
```
