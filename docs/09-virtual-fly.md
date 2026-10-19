# 9. The virtual fly: closing the loop

Code: `openfly/agent/` (`arena.py`, `fly.py`)

A brain in a dish only reacts. A brain in a body acts, and its actions
change what it senses next. `VirtualFly` closes that loop:

```
          ┌──────────── world (Arena) ────────────┐
          │ odour plumes, food, heat, predators    │
          └──────┬─────────────────────────▲───────┘
         sense   │                         │  act
   left/right    ▼                         │  walk, turn, feed, jump
   antennae, labellum,          ┌──────────┴──────────┐
   eyes, arista  ──stimuli──>   │  whole-brain LIF    │ ──spikes──> motor readout
                                │  (20 ms per step)   │             (DNs, MN9)
                                └─────────────────────┘
```

```python
from openfly.agent import VirtualFly
fly = VirtualFly(brain, "banana", seed=0)
traj = fly.run(seconds=25, progress=True)   # DataFrame: t_s, x, y, heading_deg, state, energy, drive_*
fly.events                                   # [{'t_s': 12.7, 'event': 'feeding on ripe banana'}, ...]
fly.plot()                                   # trajectory over the odour field
```

## Senses

* **Smell.** Two antennae, 2 mm either side of the head (exaggerated for a
  2D world). Each samples the arena's odour field. ORNs on each side get
  that antenna's concentration through a **Weber-law adaptation** model:
  response = 0.5 + gain x ln(c / c_adapted), so small left/right and
  moment-to-moment differences stay visible at any absolute level.
* **Taste.** Standing on food activates the GRNs for its taste (sugar, bitter...).
* **Vision.** Food items appear as gentle bright spots at their bearing,
  capped at 20 deg so a static fruit never looks like a looming threat.
  Predators trigger a real looming stimulus from their bearing.
* **Heat.** Each antenna's arista reads the heat field.

## Motor

From `MotorMonitor` (smoothed DN/MN drives):

* speed = `max_speed x walk_forward - 0.6 x max_speed x walk_backward` (+ exploration, below)
* turning rate = `-max_turn x turn` (turn > 0 = right); the fly slows while turning sharply
* **feeding**: on food with `feeding > 0.25`, the fly stops and eats (`energy` rises)
* **escape**: `escape > 0.5`, the fly jumps 15-25 mm in a random direction (1 s cooldown)

## Navigation: two real strategies

1. **Bilateral comparison** (tropotaxis): stronger smell on the left drives
   left-side circuits harder and the steering readout turns left.
2. **Temporal comparison** (klinokinesis): the agent watches its brain's
   lateral horn (`LHN`). While activity rises it walks straighter and faster;
   when it falls it turns more. The *signal* comes from simulated neurons;
   the run-and-turn *policy* is a simple body-level rule (`klinokinesis=0` disables it).

## Honest compromises (all switchable)

* `explore=True`: a small baseline walk and random turning. Real flies have
  central pattern generators and internal drive the brain-only model lacks.
  `explore=False` is purely brain-driven.
* `turn_highpass_s`: removes slow constant left/right biases of the steering
  readout. The FlyWire model's DNa02 readout has an intrinsic left bias under
  olfactory input; `turn_highpass_s=3` compensates (the playground sets it for big brains).
* The body is a point with a heading, like a fly on a treadmill ball.
  Biomechanics are for NeuroMechFly (Lobato-Rios et al. 2022; Wang-Chen et
  al. 2024); this agent's motor output could drive such a body.

## Arenas

`Arena.preset(name)`:

| preset | contents |
|---|---|
| `banana` | one ripe banana (sugar) |
| `choice` | banana vs mouldy fruit (geosmin + bitter) |
| `bitter_trap` | vinegar that tastes bitter (smells great, tastes bad) |
| `predator` | banana + a predator attack at 4 s |
| `windy_vinegar` | vinegar plume stretched by wind |
| `hot_corner` | heat in one corner, banana in another |

Build your own:

```python
from openfly.agent import Arena, FoodSource, HeatZone, Predator
world = Arena(items=[FoodSource(80, 80, "apple_cider_vinegar", "sugar", name="cider"),
                     HeatZone(40, 90, radius_mm=20), Predator(t_s=6, bearing_deg=-60)],
              wind_from_deg=0, wind_speed=1)
```

## Results on the toy brain

**Default start, 6 seeds, 25 s.** The fly starts at (25, 25) facing 45 deg,
which points within ~8 deg of the banana at (90, 75). That makes these
runs the easy case:

| arena | outcome |
|---|---|
| `banana` | found and ate the banana 6/6 (first meal at 10-21 s) |
| `choice` | never went near the mouldy fruit (closest 25-67 mm); ate the banana 2/6 |
| `bitter_trap` | reached the vinegar every time (min distance 0.3-10 mm) but **never fed**: smell attracts, bitter taste vetoes |
| `hot_corner` | avoided the heat, **fed 6/6** |
| `predator` | **escaped every attack (6/6)** and still fed 5/6 |

**Random starts with sensory controls (the fair test).** A good start
heading could make the brain look smarter than it is, so the benchmark
uses 10 random starts (positions in 10-50 mm, any heading, `rng(42)`),
30 s each, in the `banana` arena, under three conditions:

| condition | fed | first meal (s) |
|---|---|---|
| intact brain | **10 / 10** | 3.9 - 20.2 (median 14.7) |
| anosmic: all ORNs silenced | 3 / 10 | 18.9, 28.3, 29.9 |
| blind + anosmic: ORNs and photoreceptors silenced | 2 / 10 | 10.9, 28.4 |

Silencing the smell neurons removes most of the foraging: the remaining
successes are the exploration drive bumping into food. Odour guidance is
coming from the simulated circuit, not from the body rule. Reproduce it with:

```python
import numpy as np, openfly as of
from openfly.agent import VirtualFly
brain = of.load("toy")
rng = np.random.default_rng(42)
starts = zip(rng.uniform(10, 50, 10), rng.uniform(10, 50, 10), rng.uniform(0, 360, 10))
for k, (x, y, h) in enumerate(starts):
    fly = VirtualFly(brain, "banana", x=x, y=y, heading_deg=h, seed=k)
    fly.sim.silence("ORN")              # comment out for the intact brain
    fly.run(30)
    print(k, any("feeding" in e["event"] for e in fly.events))
```

These behaviours come from the toy wiring (a cartoon), so they show
that the *loop* works, not that real flies use these exact circuits.

## On the real FlyWire brain

The same code runs on FlyWire (curated transmitters). It is expensive,
about 25-35 s of compute per simulated second with numba, so each
condition below was run **once**. Treat the numbers as anecdotes that point
in a direction, not as statistics.

### Facing away: does smell turn the fly around?

The fly starts 34 mm from the banana, facing directly away (start
(60, 60), heading 225 deg, seed 3, 15 s). Bearing = angle of the banana
relative to the heading. A *negative* correlation between bearing and
turn signal means correct steering: food on the left (bearing > 0)
gives a left turn (turn < 0).

| readout | smell | klinokinesis | start -> closest | final bearing | corr(bearing, turn) |
|---|---|---|---|---|---|
| DNa02, `turn_highpass_s=3` | intact | on | 33.7 -> **19.5 mm** | 44 deg | -0.17 |
| DNa02, `turn_highpass_s=3` | intact | off | 33.7 -> **20.7 mm** | 3 deg | -0.14 |
| `SteeringDecoder` | intact | on | 33.6 -> **17.8 mm** | 31 deg | 0.01 |
| `SteeringDecoder` | intact | off | 33.6 -> **17.5 mm** | 5 deg | -0.11 |
| DNa02, `turn_highpass_s=3` | ORNs silenced | on | 33.7 -> 33.7 mm (walked off to 84 mm) | 43 deg | n/a |
| `SteeringDecoder` | ORNs silenced | on | 33.6 -> 33.2 mm | 166 deg | n/a |

What this says:

* **Smell steers the FlyWire fly.** With intact ORNs it turned around and
  cut its distance to the banana by 40-50% in every condition. With its ORNs
  silenced it never got closer. The signal travels through the real
  connectome, from the antennae to premotor and descending neurons.
* It does so **without the body's run-and-turn rule** (klinokinesis off):
  the brain's left/right steering signal alone is enough.
* It is **weak and slow**: correlations of about -0.1 to -0.2, and no fly
  reached the food (4 mm) within 15 s. The toy fly, by comparison,
  eats within ~15 s.
* The two readouts are not clearly different at n = 1.

### From the default start

From (25, 25), 82 mm away, where the odour is weak:

| readout | duration | start -> closest | fed |
|---|---|---|---|
| DNa02, `turn_highpass_s=3` | 12 s | 82 mm -> 79.5 mm (end) | no |
| `SteeringDecoder`, seed 0 | 20 s | 82 mm -> 36.9 mm | no |
| `SteeringDecoder`, seed 1 | 20 s | 82 mm -> 36.2 mm | no |

The default start faces the banana, so much of this approach is
walking straight ahead; the bearing-turn correlation on seed 0 was ~0.

### What works without caveats

The **escape**. In the `predator` arena the attack came at 4.00 s and the
giant fiber fired the escape jump at **4.02 s**, a 20 ms reaction time
produced entirely by the real connectome (LC4/LPLC2 -> giant fiber).
Taste-evoked feeding works too: MN9 fires when the fly stands on sugar.

## A population decoder for real brains

`SteeringDecoder` reads the turn from ~170 premotor neurons instead of one
cell type, the way a brain-computer interface reads a reach direction
from motor cortex (how it is trained:
[10](10-training-the-brain.md#4-training-the-interface-decoding-intentions)).

```python
from openfly.agent import SteeringDecoder, VirtualFly
dec = SteeringDecoder.fit(brain, odor="banana")      # ~6 min, once
dec.save("banana_decoder.npz")                       # reuse: SteeringDecoder.load(...)
fly = VirtualFly(brain, "banana", steering=dec)
```

Standing still, it gets the odour side right on held-out conditions
(-0.19 left, +0.21 right, -0.06 symmetric). In closed loop it steered about
as well as the high-passed DNa02 readout (tables above), not better. A
likely reason: it was calibrated on 1 s steady presentations while the
fly stood still, but in closed loop the odour changes every 20 ms, so
the decoder reads the brain in states it never saw.

**Open problems, good projects:**

* Calibrate on closed-loop data: record rates while the fly walks
  (`fly.sim.on_step`) and fit to the true bearing.
* Make steering stronger without touching the decoder: tune readout or
  circuit gains with `GainTask` on a foraging objective.
* Run many seeds on a bigger machine and turn these anecdotes into statistics.
* Add the VNC (BANC) so the motor side is also connectome-driven.
