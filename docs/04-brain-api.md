# 4. The Brain API

Code: `openfly/brain.py`, `openfly/neurons.py`, `openfly/knowledge.py`

## Brain

```python
brain = openfly.load("flywire_783")
brain.N, brain.n_connections, brain.n_synapses
print(brain.summary())
brain.neurons            # DataFrame, row = model index
brain.W                  # sparse N x N, W[pre, post] = signed synapse count
brain.positions          # (N, 3) micrometres
brain.meta               # name, citation, sources...
```

## Finding neurons

`brain[...]` accepts almost anything and returns a `NeuronSet`:

```python
brain["MN9"]                      # curated alias -> cell type CB0701 (2 neurons)
brain["giant_fiber"]              # alias -> DNp01
brain["DNa02"]                    # a cell type
brain["Kenyon_Cell"]              # a cell class
brain["descending"]               # a super class
brain["re:^PAM"]                  # regular expression on cell type
brain[720575940660219265]         # a FlyWire root ID
brain[["MN9", "giant_fiber"]]     # union of several
brain["MN99"]                     # KeyError: ... Did you mean: MN9?
```

Lookup order: alias, `cell_type`, `hemibrain_type`, `cell_sub_class`,
`cell_class`, `super_class`, `re:` regex, numeric ID.

Precise selection by any column:

```python
brain.select(cell_type="ORN_DM1", side="right")
brain.select(super_class=["descending", "motor"])
brain.select(cell_type="re:^LC1[01]")
brain.select(nt="gaba", x=lambda s: s > 500)
brain.query("super_class == 'descending' and nt == 'gaba'")
brain.search("giant")             # aliases and cell types matching a word
```

## NeuronSet

An immutable, sorted set of model indices bound to its brain.

```python
s = brain["sugar_GRN"]
len(s), s.ids, s.df, s.positions
s.left, s.right, s.side("left")
s.where(cell_type="LB3c")
s.counts("cell_type")
s.sample(5)
both = brain["sugar_GRN"] | brain["bitter_GRN"]       # union
brain.all() - brain["optic"]                           # difference
brain["KC"] & brain.select(side="left")                # intersection
```

## Wiring queries

```python
brain.inputs("MN9")                   # who talks to MN9, grouped by cell type
brain.inputs("MN9", by=None, top=10)  # per neuron
brain.outputs("giant_fiber", by="super_class")
brain.connections("LC4", "giant_fiber")   # every connection between two groups
brain.describe(brain["MN9"].left)     # annotations + strongest partners of one neuron
sub = brain.subbrain(brain["KC"] | brain["MBON"] | brain["DAN"], name="mushroom body")
```

`inputs`/`outputs` return `synapses` (unsigned), `signed` (excitatory minus
inhibitory), `neurons` and `fraction` (share of the target's total input).

## Curated aliases (`openfly.knowledge`)

A curated vocabulary of famous neurons, with a reference for each.
`openfly.knowledge.all_aliases()` lists them.

| group | aliases |
|---|---|
| taste | `sugar_GRN`, `bitter_GRN`, `water_GRN`, `salt_high_GRN`, `glutamate_GRN`, `taste_peg_GRN`, `pharyngeal_GRN` |
| smell | `ORN`, `ALPN`, `uPN`, `LHN`, `CO2_ORN`, `geosmin_ORN`, `cVA_ORN`, `vinegar_ORN` |
| learning | `KC`, `MBON`, `DAN`, `PAM`, `PPL1`, `APL`, `DPM` |
| vision | `photoreceptors`, `R1-6`, `R7`, `R8`, `ocellar_PR`, `T4`, `T5`, `LC4`, `LPLC2`, `LC10`, `LC11`, `HS`, `VS` |
| hearing/touch | `JO_auditory`, `JO_wind_gravity`, `head_bristle`, `eye_bristle` |
| temperature | `hot_TRN`, `cold_TRN`, `dry_HRN`, `moist_HRN` |
| central | `EPG`, `PFL3`, `pC1` |
| motor | `MN9`, `proboscis_MN`, `ingestion_MN`, `antennal_MN`, `neck_MN`, `descending`, `giant_fiber`/`GF`, `MDN`, `DNa02`, `DNa01`, `P9`/`DNp09`, `oviDN` |

Every alias has a `confidence` (`established`, `likely`). For example
`P9 = DNp09` is marked *likely*.

Add your own:

```python
openfly.knowledge.register_alias("my_cells", {"cell_type": ["CB0553", "DNge080"]},
                                 "feeding hubs found by a screen", reference="my lab notebook")
brain["my_cells"]
```

A selector can be a list of alternatives for datasets with different
naming. `giant_fiber` is `[{"cell_type": "DNp01"}, {"cell_type": "Giant Fiber"}]`
so it works on FlyWire, MANC and the hemibrain alike.
