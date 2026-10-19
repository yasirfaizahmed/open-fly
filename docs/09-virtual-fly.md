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

## Results on the toy brain (6 random seeds each, 25 s)

| arena | outcome |
|---|---|
| `banana` | **found and ate the banana 6/6** (first meal at 10-21 s) |
| `choice` | never went near the mouldy fruit (closest 25-67 mm); ate the banana 2/6 |
| `bitter_trap` | reached the vinegar every time (min distance 0.3-10 mm) but **never fed**: smell attracts, bitter taste vetoes |
| `hot_corner` | avoided the heat, **fed 6/6** |
| `predator` | **escaped every attack (6/6)** and still fed 5/6 |

These behaviours come from the toy wiring (a cartoon).

## On the real FlyWire brain

The same code runs on FlyWire (`turn_highpass_s=3`, curated transmitters),
but **it does not forage yet**. In a 12 s run in the `banana` arena the fly
ended 79.5 mm from the banana (it started 82 mm away) and never fed. That
run took ~6 minutes of compute. The reason is the readout, not the body: in
the whole-brain LIF model, banana odour barely recruits the descending
neurons used for walking (DNp09) and steering (DNa02 carries a left bias), so
movement is dominated by the exploration drive. Taste-evoked feeding and
looming-evoked escape do work on FlyWire, so expect the virtual fly to
**eat when it lands on sugar** and **jump when attacked**.

Making odour-guided navigation emerge from the real connectome is an open
problem and a good project. Candidate approaches: population decoding across
many DNs instead of a single type, the central-complex steering pathway
(PFL3), training readout gains with `GainTask`, or a VNC connectome (BANC)
for the motor side.
