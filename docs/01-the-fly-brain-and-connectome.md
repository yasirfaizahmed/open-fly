# 1. The fly brain and its connectome

## The "brain file" you heard about

In October 2024 the **FlyWire** consortium published the first complete
wiring diagram (*connectome*) of an adult animal brain with a complex
behavioural repertoire: the brain of a female fruit fly, *Drosophila
melanogaster* (Dorkenwald et al. 2024; Schlegel et al. 2024, both in
*Nature*). It is what you heard about, and it is what openfly loads by
default (`openfly.load("flywire_783")`).

How it was made, step by step:

1. **Slicing.** A fly brain (about 0.6 mm wide, roughly the size of a
   poppy seed) was embedded in resin and cut into about **7,000 slices,
   each ~40 nanometres thick**. That is about a thousandth of a human hair.
   This is the FAFB volume ("Full Adult Fly Brain"; Zheng et al. 2018, *Cell*).
2. **Imaging.** Every slice was photographed with a transmission
   electron microscope at a resolution of about **4 nm per pixel**, which
   gives tens of millions of images and well over 100 terabytes of data.
   At this resolution you can see individual synaptic vesicles.
3. **Alignment.** Software stitched the images of each slice and aligned
   consecutive slices into one 3D volume.
4. **AI tracing (segmentation).** Convolutional neural networks labelled
   every voxel with the cell it belongs to, splitting the volume into
   millions of fragments, then merged fragments into neurons
   (Dorkenwald et al. 2022, *Nature Methods*).
5. **Human proofreading.** The automatic result makes mistakes (merges
   and splits). A worldwide community of scientists and citizen
   proofreaders corrected them in a shared web app, FlyWire. That took
   on the order of 30 person-years of work.
6. **Synapses.** Another neural network found the synapses (Buhmann et al.
   2021) and assigned each one to its pre- and postsynaptic neuron.
7. **Neurotransmitters.** A third network predicted, from the look of
   each synapse, which neurotransmitter it releases (acetylcholine, GABA,
   glutamate, dopamine, serotonin or octopamine; Eckstein et al. 2024, *Cell*).
8. **Annotation.** Every neuron was given a hierarchical identity
   (flow, super class, class, cell type, hemilineage, side) by comparison
   with the literature and with another connectome, the Janelia hemibrain
   (Schlegel et al. 2024). There are more than 8,000 cell types.

The result, in the version openfly uses (snapshot "783"):

| | |
|---|---|
| neurons (with connectivity) | **138,639** |
| connected neuron pairs | **15,091,983** |
| synapses | **~54.5 million** |
| neuron positions | anchor point inside each neuron + soma location |
| annotations | super class, class, cell type, side, transmitter, nerve, ... |

You can run `openfly info flywire_783` to print these numbers from the data itself.

## What a connectome tells you, and what it doesn't

A connectome is a **map of who talks to whom, and how much** (synapse
counts). With predicted transmitters you also know *how*: exciting or
inhibiting. It does **not** directly contain:

* synaptic *strengths* beyond counts (real synapses differ in size and
  efficacy),
* electrical synapses (gap junctions), which EM can't see reliably,
* neuromodulation (dopamine, serotonin and neuropeptides act in slow, diffuse ways),
* each neuron's electrical personality (ion channels, graded potentials),
* plasticity: how the connections change with experience,
* the body: muscles, sensors and the ventral nerve cord (spinal cord equivalent).

That is why the model built from it is a *hypothesis generator*. Shiu
et al. (2024, *Nature*) showed that even a very simple spiking model of
the whole connectome predicts real circuits for feeding and grooming, which
they then confirmed in living flies. openfly is built around that model.

## A tour of the fly brain

```
               dorsal
        _________________________
       /   MB      CX      MB    \        MB  = mushroom bodies (learning & memory)
  OL  |  LH    central brain   LH |  OL   LH  = lateral horn (innate smell responses)
      |      AL    SEZ    AL      |       CX  = central complex (navigation, compass)
       \_________________________/        AL  = antennal lobes (smell)
               ventral                   SEZ = subesophageal zone (taste, feeding)
                                          OL  = optic lobes (vision, >50% of all neurons)
```

**Super classes** used in FlyWire (and as colours in the playground):

| super class | what | count |
|---|---|---|
| `optic` | neurons inside the optic lobes | ~77k |
| `central` | central brain neurons | ~32k |
| `sensory` | sensory neurons entering the brain (eyes, antennae, mouth...) | ~16k |
| `visual_projection` | carry visual features from optic lobes to the central brain | ~8k |
| `ascending` / `descending` | to/from the ventral nerve cord (body) | ~1.7k / ~1.3k |
| `motor` | motor neurons whose axons leave the brain (proboscis, neck, antennae) | ~110 |
| `endocrine` | hormone-releasing neurons | ~80 |

### The sensory systems in openfly

* **Smell:** ~2,300 olfactory receptor neurons (ORNs) of ~50 types, each
  type expressing one receptor and wiring into one *glomerulus* of the
  antennal lobe. Projection neurons (PNs) relay to the **mushroom body**
  (learned meaning) and **lateral horn** (innate meaning).
* **Taste:** gustatory receptor neurons on the labellum (mouth), legs and
  pharynx, typed by modality: sugar, bitter, water, salt...
* **Vision:** ~11,000 photoreceptors (R1-6 for luminance/motion, R7/R8 for
  colour) feed a deep, columnar optic lobe. Motion detectors (T4/T5) and
  feature detectors (LC4 = looming, LC10 = small objects...) project to the
  central brain.
* **Mechanosensation:** Johnston's organ in the antenna hears sound (the
  courtship song) and senses wind and gravity; bristles feel touch.
* **Temperature and humidity:** hot/cold and dry/moist receptors in the arista.

### Outputs

The brain controls the body mostly through **descending neurons**. A few
are famous "command-like" neurons:

* **Giant fiber (DNp01)**: escape take-off
* **MDN** ("moonwalker"): backward walking
* **DNa02**: steering
* **MN9**: a motor neuron that extends the proboscis (feeding)

## Glossary

| term | meaning |
|---|---|
| connectome | complete wiring diagram of a nervous system |
| synapse | a contact where one neuron passes a signal to another |
| spike / action potential | the all-or-none electrical pulse neurons send |
| firing rate | spikes per second (Hz) |
| excitatory / inhibitory | makes the target more / less likely to spike |
| neurotransmitter | the chemical a synapse releases (ACh, GABA, glutamate...) |
| neuropil | a dense region of synapses (e.g. AL, MB, LH) |
| glomerulus | a ball of neuropil in the antennal lobe; one per receptor type |
| cell type | neurons with the same shape and connectivity (often a left/right pair, sometimes hundreds) |
| ipsilateral / contralateral | same side / opposite side |
| LIF | leaky integrate-and-fire: the simplest spiking neuron model |
| DAN / MBON / KC | dopaminergic neuron / mushroom body output neuron / Kenyon cell |

## Key references

* Zheng et al. 2018, *Cell* 174:730. The FAFB electron-microscopy volume.
* Dorkenwald et al. 2024, *Nature* 634:124. FlyWire: the whole-brain connectome.
* Schlegel et al. 2024, *Nature* 634:139. Annotation and cell typing.
* Shiu et al. 2024, *Nature* 634:210. The whole-brain LIF model used here.
* Eckstein et al. 2024, *Cell* 187:2574. Neurotransmitter prediction.
* Matsliah et al. 2024, *Nature* 634:166. The optic lobe parts list.
* Scheffer et al. 2020, *eLife* 9:e57443. The Janelia hemibrain.
