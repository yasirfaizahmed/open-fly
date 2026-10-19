# 12. Adding new brains

openfly is built to outlive any single dataset. Everything runs on the
same two tables (neurons + signed edges, [02](02-architecture.md)), so a new
connectome needs only an *adapter*.

## Case 1: a new release on FlyWire Codex: nothing to do

Codex serves every connectome in a shared layout (`neurons.csv.gz` with
`Root ID, Predicted NT type, Super Class, Class, ..., Primary Cell Type`,
and `connections_princeton.csv.gz` with
`pre_root_id, post_root_id, neuropil, syn_count, nt_type`).

```python
brain = openfly.load("codex:banc/999")     # any <dataset>/<version>
```

The adapter (`openfly/data/codex.py`) sums synapses over neuropils, signs
each row by its transmitter (falling back to the neuron's prediction),
assigns each neuron its dominant neuropil and, as these tables have no
coordinates, places neurons with the **schematic layout**: region centre
(from FlyWire anchor medians, plus schematic nerve-cord positions) plus
spread. The playground shows a "schematic layout" note.

Verified on real downloads: **MANC 1.2.1** (23,665 neurons, 5.3 M
connections) loads in ~20 s, and the giant fiber is found by name (`brain["giant_fiber"]`).

## Case 2: a neuPrint export

```python
from openfly.data.neuprint import build_neuprint, lookups_from_flywire
nt, classes = lookups_from_flywire("Supplemental_file1_neuron_annotations.tsv")
brain = build_neuprint("exported-traced-adjacencies.tar.gz", nt_lookup=nt, class_lookup=classes)
```

neuPrint's compact exports have no transmitters or classes, so openfly
**borrows them from FlyWire** through matching cell types. For the
hemibrain this signs 15,703 of 21,739 neurons and gives them super classes.
The same looming code then drives the hemibrain's giant fiber to 162 Hz.

## Case 3: anything else: two tables

```python
import pandas as pd, openfly
neurons = pd.DataFrame({"root_id": [...], "cell_type": [...], "side": [...], "nt": [...],
                        "x": [...], "y": [...], "z": [...]})          # positions optional
edges = pd.DataFrame({"pre": [...], "post": [...], "weight": [...]})   # signed synapse counts, by ID
brain = openfly.Brain.from_edges(neurons, edges, meta={"name": "my_connectome", "citation": "..."})
brain.save("my_connectome.flybrain")
```

Unsigned weights? Sign them first, e.g.
`edges.weight *= np.where(edges.pre.map(nt_of).isin(["gaba", "glutamate"]), -1, 1)`.

## Registering a dataset (so `openfly.load("name")` works)

```python
from openfly.data import Dataset, register

def build(paths):
    return my_adapter(paths["neurons"], paths["edges"])

register(Dataset(
    name="larva_2023", title="Drosophila larva connectome",
    description="...", build=build,
    files={"neurons": ["https://example.org/neurons.csv"], "edges": ["https://example.org/edges.csv"]},
    citation="Winding et al. 2023 Science", license="CC BY 4.0", download_mb=5,
))
brain = openfly.load("larva_2023")      # downloads, builds, caches as .flybrain
```

## Checklist for a good adapter

* `root_id` unique; `side` in `left`/`right`/`center`; `nt` lower-case
  (`acetylcholine`, `gaba`, `glutamate`, ...).
* Use FlyWire-style names where they exist (`cell_class`, `cell_type`) so aliases,
  senses and motor channels work. Add dataset-specific names as alias alternatives
  ([04](04-brain-api.md#curated-aliases-openflyknowledge)).
* Fill `neuropil` if you know it: it feeds the schematic layout and filtering.
* Put `citation`, `license` and source URLs in `meta`.
* Test with `brain.summary()`, `openfly playground -d your.flybrain`, and a
  known reflex (does the looming or taste circuit behave?).
