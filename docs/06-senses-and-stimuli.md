# 6. Senses and stimuli

Code: `openfly/senses/` (`base.py`, `olfaction.py`, `taste.py`, `vision.py`,
`mechano.py`, `thermo.py`, `scenarios.py`)

Every sense function takes the brain first and returns a `Stimulus` (or a
`StimulusSet`) saying **which neurons**, **how hard** (Hz) and **when**:

```python
from openfly import senses
st = senses.smell(brain, "banana", 0.8)
sim.add(st, senses.taste(brain, "sugar"), senses.looming(brain, azimuth=40))
```

`Stimulus(targets, rates, start_ms, stop_ms, name, kind="poisson" | "current", modulation=f)`
* `"poisson"`: Poisson input that forces spikes at ~`rates` Hz (the Shiu
  model's sensory/optogenetic drive).
* `"current"`: steady depolarisation of `rates` mV (graded, only fires if it crosses threshold).
* `modulation(t_ms)`: multiplies the rates over time (moving stimuli, songs...).
* `a + b` makes a `StimulusSet`; `.between(start, stop)`, `.scaled(k)`.

## Smell

ORNs are typed by glomerulus in FlyWire (`ORN_DM1`, `ORN_DA2`...).
An odour is a profile of glomerulus activations (0..1). ORN rate = 150 Hz x
weight x saturation(concentration).

| odour | main glomeruli | valence | reference |
|---|---|---|---|
| `apple_cider_vinegar` | DM1, VA2, DM4, DP1m (+DM5 when strong) | attractive | Semmelhack & Wang 2009 |
| `banana` | DM2, DM3, VM2, DM1, VA2, DM4 | attractive | Hallem & Carlson 2006 |
| `ethyl_acetate`, `isoamyl_acetate` | DM1/DM2/DM3/VM2 | attractive | Hallem & Carlson 2006 |
| `acetic_acid` | DP1l, DC4, DP1m | attractive | Silbering et al. 2011 |
| `ammonia` | VM1 | attractive | Min et al. 2013 |
| `citrus`, `farnesol` | DC1, DC3 | attractive | Dweck et al. 2013; Ronderos et al. 2014 |
| `green_leaves`, `methyl_salicylate` | DL5, DL1 | mixed | Hallem & Carlson 2006 |
| `geosmin` | DA2 | aversive | Stensmyr et al. 2012 |
| `co2` | V | aversive | Suh et al. 2004 |
| `cva`, `fly_odor` | DA1; VA1v/VA1d | pheromone | Kurtovic et al. 2007; Dweck et al. 2015 |
| `moldy_banana` | banana + DA2 | conflict | |

This is a **simplified, educational** table. For research, define profiles
from the DoOR database with `senses.custom_odor(name, {"DM1": 0.8, ...})`.

**Laterality:** ORNs are annotated with the antenna their nerve comes
from, so `smell(brain, "banana", left=1.0, right=0.3)` gives each antenna
its own concentration.

## Taste

`taste(brain, tastant, intensity, side="both")`, using FlyWire's GRN
modality typing (Tastekin et al. 2026):

| tastant | neurons |
|---|---|
| `sugar` | `sugar_GRN` (LB3c "sugar" + LB3b "sugar/low salt") |
| `bitter` | `bitter_GRN` (LB1a-d) |
| `water` | `water_GRN` (LB3a) |
| `low_salt`, `high_salt`, `glutamate` | the matching GRN classes |
| `fruit_juice`, `rotten_fruit` | mixtures |

## Vision

### Where does each photoreceptor look?

FlyWire does not annotate retinotopy, so `VisualField` **estimates** it.
The eye's map is preserved in the lamina and medulla. Within each eye the
photoreceptor terminals form a curved sheet whose main axis is
dorsal-ventral (-> elevation) and whose second axis runs front-back (->
azimuth; flipped for R7/R8 because of the optic chiasm). Values are
rank-normalised to -15..165 deg azimuth per eye (0 = straight ahead,
positive = right) and -60..60 deg elevation. That is good enough for left vs
right, up vs down, front vs back. Brains with explicit `azimuth`/`elevation`
columns (the toy) use those.

### Retina-level stimuli (light on photoreceptors)

```python
senses.flash(brain)                                  # 50 ms full field
senses.spot(brain, azimuth=30, elevation=0, radius_deg=12, color="yellow")
senses.image(brain, picture)                         # any picture as a panorama (H x W, 0..1)
senses.looming(brain, azimuth=45, duration_ms=400)   # expanding disk: theta(t) = 2 atan(r/v / (t_c - t))
senses.moving_object(brain, -60, 60)                 # a small object sweeping across
senses.grating(brain, "front_to_back")               # drifting stripes
```

### Feature-level stimuli (why both levels?)

Real early vision computes motion and looming with graded potentials and
precise delays, which a point-neuron LIF model reproduces only crudely.
So `looming`, `moving_object` and `grating` also drive the known
*feature detectors* directly (`features=True`):

* looming -> **LC4** (rate follows expansion speed) and **LPLC2** (follows size), on the stimulated side
  (von Reyn et al. 2017; Klapoetke et al. 2017);
* small moving objects -> **LC10**/**LC11** on the side the object is on;
* gratings -> the **T4/T5** subtype preferring that direction
  (a = front-to-back, b = back-to-front, c = up, d = down; Maisak et al. 2013).

Use `retina=False` or `features=False` to isolate either level.

## Hearing, wind and touch

```python
senses.hear(brain, "pulse_song", ipi_ms=35)   # D. melanogaster courtship pulses, ~35 ms apart
senses.hear(brain, "sine_song")
senses.hear(brain, "tone", freq_hz=200)       # tuned around 100-300 Hz
senses.wind(brain, direction_deg=-90)         # from the left: left JO-C/E neurons fire more
senses.touch(brain, "eye")                    # interommatidial bristles -> grooming
```

## Temperature and humidity

```python
senses.temperature(brain, "hot", side="left")   # TRN_VP2 hot cells
senses.temperature(brain, "cold")
senses.humidity(brain, "dry")
```

## Scenarios: stories made of senses

`senses.scenario(brain, name)` returns the stimuli; `senses.SCENARIOS[name]`
holds the story, the neurons worth watching and what a real fly does.

| scenario | stimuli | a real fly... |
|---|---|---|
| `sugar_drop` | sugar | extends its proboscis |
| `bitter_leaf` | bitter | refuses to feed |
| `sugar_laced_with_bitter` | sugar + bitter | feeding suppressed |
| `thirsty_finds_water` | water | drinks |
| `ripe_banana` | banana smell + yellow spot ahead | approaches, then feeds |
| `banana_on_the_left` | banana, left antenna stronger | turns left |
| `vinegar_trap` | apple cider vinegar | strong attraction |
| `mouldy_fruit` | banana esters + geosmin | avoids |
| `co2_alarm` | CO2 | walks away |
| `predator_from_right` | looming at +50 deg | escape take-off |
| `passing_fly` | small object left -> right | tracks it |
| `optic_flow_spin` | front-to-back grating | optomotor response |
| `courtship_song` | pulse song | slows, may accept a male |
| `gust_from_left` | wind | orients |
| `dusty_eyes` | eye bristles | grooms |
| `hot_plate_left` | heat on the left | turns away |
| `flash` | full-field flash | startles |

## Write your own sense

Any function returning a `Stimulus` works. For example, an
"optogenetics" stimulus that flickers a cell type at 10 Hz:

```python
import numpy as np
from openfly.senses import Stimulus

def flicker(brain, target, hz=10, rate=150):
    ns = brain[target]
    return Stimulus(ns.idx, rate, name=f"flicker {target}",
                    modulation=lambda t: float((t * hz / 1000) % 1 < 0.5))

sim.add(flicker(brain, "PAM"))
```
