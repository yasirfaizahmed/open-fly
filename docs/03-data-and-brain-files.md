# 3. Data and brain files

Code: `openfly/data/` (`registry.py`, `download.py`, `brainfile.py`,
`flywire.py`, `codex.py`, `neuprint.py`, `builder.py`, `toy.py`, `layout.py`)

## Getting a brain

```python
import openfly as of

brain = of.load("flywire_783")          # the real adult fly brain (downloads ~140 MB once)
brain = of.load("flywire_783", transmitters="curated")   # with literature sign fixes (see 05)
toy   = of.load("toy")                  # 2,300-neuron teaching brain, offline, instant
other = of.load("manc_1.2.1")           # any registered dataset
new   = of.load("codex:banc/999")       # any future FlyWire Codex release
mine  = of.load("my_brain.flybrain")    # a brain file on disk
```

From the terminal: `openfly datasets`, `openfly download flywire_783`, `openfly info flywire_783`.

### Built-in datasets

| name | what | neurons | download |
|---|---|---|---|
| `toy` | synthetic teaching brain (not real data) | ~2.3k | none |
| `flywire_783` | **FlyWire FAFB v783**, adult female whole brain (default) | 138,639 | ~140 MB |
| `flywire_630` | FlyWire v630, the version used in Shiu et al. 2024 | 127,400 | ~125 MB |
| `banc_888` | BANC: brain **and** ventral nerve cord | 158,262 | ~32 MB |
| `manc_1.2.1` | MANC: male adult ventral nerve cord | 23,665 | ~28 MB |
| `mcns_1.0` | male central nervous system | 166,700 | ~48 MB |
| `maol_1.1` | male optic lobe | 52,445 | ~32 MB |
| `hemibrain_1.2` | Janelia hemibrain (central brain) | 21,739 | ~46 MB |

### Where the FlyWire data comes from

`flywire_783` is assembled from three public sources, pinned to exact
commits so results are reproducible:

1. **Connectivity** (`Connectivity_783.parquet`, `Completeness_783.csv`) from
   the repository of the Shiu et al. (2024) whole-brain model
   (`github.com/philshiu/Drosophila_brain_model`). It holds one row per connected
   neuron pair with its synapse count and excitatory/inhibitory sign. Using
   *exactly* these files keeps openfly comparable to the published model.
2. **Annotations** (`Supplemental_file1_neuron_annotations.tsv`) from
   `github.com/flyconnectome/flywire_annotations` (Schlegel et al. 2024 and
   later updates): super class, class, cell type, side, predicted transmitter
   and position of every neuron. Positions are stored in voxels of 4x4x40 nm;
   openfly converts them to micrometres.
3. **Neuropils** (optional, `neuropil_synapse_table.csv.gz`) from FlyWire Codex:
   synapse counts per brain region, used to give each neuron its main
   neuropil (`brain.neurons.neuropil`).

### Cache and offline use

```
~/.openfly/                 (or $OPENFLY_HOME)
    raw/flywire_783/        original downloads, untouched
    brains/flywire_783.flybrain   compiled brain (52 MB, loads in ~1.5 s)
```

* `OPENFLY_HOME=/data/openfly`: put everything somewhere else.
* `OPENFLY_DATA_DIR=/path/with/files`: if you already have the raw files
  (e.g. from a `git clone`), openfly finds them there by file name and
  skips downloading.
* `of.load(name, rebuild=True)`: rebuild the `.flybrain` from the raw files.

The downloader uses only the Python standard library, retries, tries
mirror URLs, writes to `*.part` and renames on success, and honours
`HTTPS_PROXY`.

## The `.flybrain` format

One portable file per connectome. It is a plain ZIP archive, so `unzip -l brain.flybrain` works:

| member | content |
|---|---|
| `meta.json` | name, title, species, version, citation, licence, source URLs, units, sign convention, `format: "flybrain"`, `format_version: 1` |
| `neurons.parquet` | one row per neuron; row order = model index; schema below plus any extra columns |
| `edges.parquet` | `pre`, `post` (int32 model indices), `weight` (int32 **signed** synapse count) |

Neuron schema (missing columns are filled in automatically):

| column | type | meaning |
|---|---|---|
| `root_id` | int64 | dataset ID (FlyWire root ID, MANC body ID...) |
| `x, y, z` | float32 | position in micrometres (FlyWire: an anchor point inside the neuron) |
| `flow` | str | `afferent` / `intrinsic` / `efferent` |
| `super_class`, `cell_class`, `cell_sub_class`, `cell_type` | str | hierarchical identity |
| `side` | str | `left` / `right` / `center` |
| `nt` | str | predicted neurotransmitter |
| `neuropil` | str | main brain region |

Parquet was chosen because it is compact, fast and readable from Python,
R, Julia, JavaScript and Spark, so brain files are useful outside openfly too.

```python
brain.save("my_edit.flybrain")
b = of.load("my_edit.flybrain")
of.data.read_meta("my_edit.flybrain")   # just the metadata
```

## BrainBuilder: make your own brain

```python
from openfly import BrainBuilder

b = BrainBuilder("reflex", seed=1)
sugar = b.add("LB3c", 10, super_class="sensory", cell_class="gustatory", cell_sub_class="sugar",
              region="GNG", nt="acetylcholine")
inter = b.add("G2N-1", 3, region="GNG")
mn9   = b.add("CB0701", 1, super_class="motor", region="GNG")
b.connect(sugar, inter, synapses=16)          # all-to-all
b.connect(inter, mn9, synapses=(40, 50))      # random counts in a range
brain = b.build()
brain["MN9"]                                   # aliases work on your brain too
```

Connection helpers: `p=` (probability), `n_inputs=` (each target gets k
random inputs), `one_to_one=True`, `connect_by_distance(radius=..., key=("azimuth", "elevation"))`
for map-like (retinotopic) wiring, and `group(*pops)` to treat several
populations as one. Signs follow the presynaptic transmitter (GABA and
glutamate inhibit).

### Calibrating a hand-made brain

In the Shiu LIF model a neuron's output depends on its summed drive
`D = sum over inputs of (rate x synapses)`. Measured with the openfly engine:

| D (Hz x synapses) | output rate |
|---|---|
| < ~2,500 | silent |
| 4,500 | ~11 Hz |
| 7,500 | ~30 Hz |
| 10,000 | ~44 Hz |
| 15,000 | ~63 Hz |
| 23,000 | ~81 Hz |

So 8 sensory neurons at 120 Hz need ~16 synapses each onto a relay to
drive it at ~60 Hz. The toy brain was wired with this table.

## The toy fly

`openfly.load("toy")` builds (in ~0.2 s, deterministic) a cartoon brain
with real cell-type names and textbook pathways:

* **smell**: 10 glomeruli (`ORN_DM1` ... `ORN_V`) with bilateral projections ->
  uniglomerular PNs -> lateral horn (attractive vs aversive) and 600 Kenyon cells
  -> 4 MBON types; PAM (reward) and PPL1 (punishment) DANs define compartments;
  APL feedback inhibition keeps KC activity sparse (~10%).
* **steering**: lateral horn -> LAL -> DNa02 (ipsilateral for attractive, contralateral
  for aversive odours), with bilateral mutual inhibition (a winner-take-all
  "flip-flop", as in insect LAL circuits).
* **taste**: sugar/water GRNs -> G2N-1 -> MN9 (`CB0701`); bitter/high-salt -> GABAergic
  interneurons that inhibit MN9; sugar -> PAM, bitter -> PPL1.
* **vision**: 80 R1-6 per eye on a retinotopic grid -> L2 -> LC4/LPLC2 -> giant fiber
  (`DNp01`); LC10 small-object detectors -> steering; a wide-field GABAergic CT1
  neuron suppresses full-field flashes; T4/T5 subtypes -> HS/VS.
* **hearing / wind / touch / heat / humidity** pathways, and 400 background neurons.

It is **not real data**. Use it to learn the tools and to sanity-check ideas,
then run the same code on FlyWire.

## Licences and citations

openfly's code is MIT-licensed. **Connectome data is not part of openfly**:
it is downloaded from its publishers and remains under their terms.
`brain.meta["citation"]` and `brain.meta["license"]` say what to cite and
where the terms are. See [CITATION.md](../CITATION.md).
