"""Scenarios: little stories made of several senses at once.

"The fly sees a ripe banana and smells it." "A shadow swoops in from
the right." Each :class:`Scenario` bundles the stimuli, a short story,
the neurons worth watching and what a real fly would do, so you can
compare the model with biology.

    from openfly.senses import scenarios
    stim = scenarios.get("ripe_banana").build(brain)
    sim.add(stim)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import mechano, olfaction, taste, thermo, vision
from .base import StimulusSet, as_stimuli


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    story: str
    make: Callable
    duration_ms: float = 1000.0
    watch: tuple = ()
    real_fly: str = ""
    tags: tuple = field(default_factory=tuple)

    def build(self, brain, start_ms: float = 0.0) -> StimulusSet:
        """Stimuli for this scenario, shifted to begin at ``start_ms``."""
        out = StimulusSet()
        for s in as_stimuli(self.make(brain)):
            stop = None if s.stop_ms is None else s.stop_ms + start_ms
            out.append(s.between(s.start_ms + start_ms, stop))
        return out


SCENARIOS: dict[str, Scenario] = {}


def _s(sc: Scenario):
    SCENARIOS[sc.name] = sc


def get(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(SCENARIOS)}")
    return SCENARIOS[name]


def list_scenarios() -> list[str]:
    return list(SCENARIOS)


_s(Scenario(
    "sugar_drop", "A drop of sugar water",
    "The fly's proboscis touches sugar. Sugar GRNs fire and, a few synapses later, MN9 extends the proboscis.",
    lambda b: taste.taste(b, "sugar", 1.0),
    watch=("sugar_GRN", "MN9"), real_fly="Extends its proboscis and drinks (proboscis extension reflex).",
    tags=("taste", "feeding")))
_s(Scenario(
    "bitter_leaf", "A bitter leaf",
    "Bitter compounds (e.g. caffeine, quinine) hit bitter GRNs, which inhibit the feeding circuit.",
    lambda b: taste.taste(b, "bitter", 1.0),
    watch=("bitter_GRN", "MN9"), real_fly="Refuses to feed and may walk away.", tags=("taste",)))
_s(Scenario(
    "sugar_laced_with_bitter", "Sugar spiked with quinine",
    "Sugar and bitter at the same time. Does the excitation or the inhibition win at MN9?",
    lambda b: taste.taste(b, "sugar", 1.0) + taste.taste(b, "bitter", 1.0),
    watch=("sugar_GRN", "bitter_GRN", "MN9"), real_fly="Feeding is strongly suppressed.", tags=("taste", "conflict")))
_s(Scenario(
    "thirsty_finds_water", "Water for a thirsty fly",
    "Water GRNs respond to low osmolarity; for a thirsty fly water is rewarding.",
    lambda b: taste.taste(b, "water", 1.0),
    watch=("water_GRN", "MN9"), real_fly="A thirsty fly drinks.", tags=("taste",)))
_s(Scenario(
    "ripe_banana", "A ripe banana, seen and smelled",
    "A yellow banana sits straight ahead. Its fruity esters reach both antennae while its shape lights up the "
    "frontal visual field.",
    lambda b: olfaction.smell(b, "banana", 0.8) + vision.spot(b, 0.0, -5.0, 18.0, color="yellow"),
    watch=("ORN", "ALPN", "LHN", "KC", "DNa02", "DNp09"),
    real_fly="Walks toward the banana and, on contact, tastes and feeds.", tags=("smell", "vision", "food")))
_s(Scenario(
    "banana_on_the_left", "Banana smell from the left",
    "The odour plume hits the left antenna harder than the right. Bilateral comparison lets flies turn toward it.",
    lambda b: olfaction.smell(b, "banana", left=1.0, right=0.35),
    watch=("ORN", "ALPN", "DNa02"), real_fly="Turns left, toward the odour (Gaudry et al. 2013).",
    tags=("smell", "navigation")))
_s(Scenario(
    "vinegar_trap", "The vinegar trap",
    "Apple cider vinegar: irresistible to flies, which is why vinegar traps work.",
    lambda b: olfaction.smell(b, "apple_cider_vinegar", 0.8),
    watch=("vinegar_ORN", "ALPN", "LHN"), real_fly="Strong attraction; walks/flies upwind.", tags=("smell",)))
_s(Scenario(
    "mouldy_fruit", "Mould on the fruit",
    "Fruity smell plus geosmin, the warning signal of toxic moulds. Geosmin has its own labelled line (DA2).",
    lambda b: olfaction.smell(b, "moldy_banana", 1.0),
    watch=("geosmin_ORN", "LHN"), real_fly="Avoids the fruit and won't lay eggs there (Stensmyr et al. 2012).",
    tags=("smell", "conflict")))
_s(Scenario(
    "co2_alarm", "Stress smell",
    "CO2 is released by stressed flies, a 'danger here' signal.",
    lambda b: olfaction.smell(b, "co2", 1.0),
    watch=("CO2_ORN", "LHN"), real_fly="Walks away from CO2 (Suh et al. 2004).", tags=("smell", "danger")))
_s(Scenario(
    "predator_from_right", "A shadow swoops in from the right",
    "A dark object expands rapidly on the right eye, the visual signature of a collision course.",
    lambda b: vision.looming(b, azimuth=50.0, duration_ms=400.0),
    duration_ms=600.0, watch=("LC4", "LPLC2", "giant_fiber"),
    real_fly="Escape take-off driven by the giant fiber, within ~10 ms of the final expansion (von Reyn et al. 2014).",
    tags=("vision", "danger", "escape")))
_s(Scenario(
    "passing_fly", "Another fly walks past",
    "A small dark object sweeps from left to right across the visual field.",
    lambda b: vision.moving_object(b, -80.0, 80.0, size_deg=8.0, duration_ms=1000.0),
    watch=("LC10", "LC11", "DNa02"), real_fly="Tracks the object; males chase it during courtship (LC10).",
    tags=("vision", "social")))
_s(Scenario(
    "optic_flow_spin", "The world spins",
    "Front-to-back motion on both eyes, as if the fly were walking forward fast.",
    lambda b: vision.grating(b, "front_to_back"),
    watch=("T4", "T5", "HS", "VS"), real_fly="Optomotor responses stabilise its course.", tags=("vision", "motion")))
_s(Scenario(
    "courtship_song", "A male sings",
    "A male vibrates one wing, producing pulses ~35 ms apart. The female's Johnston's organ picks them up.",
    lambda b: mechano.hear(b, "pulse_song"),
    watch=("JO_auditory", "pC1"), real_fly="A receptive female slows down and may accept mating.",
    tags=("hearing", "social")))
_s(Scenario(
    "gust_from_left", "A gust of wind from the left",
    "Wind deflects the aristae; Johnston's organ C/E neurons report the direction.",
    lambda b: mechano.wind(b, -90.0, 1.0),
    watch=("JO_wind_gravity",), real_fly="Orients relative to the wind; stops walking in strong gusts.",
    tags=("wind",)))
_s(Scenario(
    "dusty_eyes", "Dust on the eyes",
    "Dust tickles the bristles between the ommatidia.",
    lambda b: mechano.touch(b, "eye"),
    watch=("eye_bristle",), real_fly="Grooms its eyes with its front legs (Hampel et al. 2017).", tags=("touch",)))
_s(Scenario(
    "hot_plate_left", "Heat on the left",
    "A hot surface radiates from the left side.",
    lambda b: thermo.temperature(b, "hot", 1.0, side="left"),
    watch=("hot_TRN",), real_fly="Turns away from the heat.", tags=("temperature",)))
_s(Scenario(
    "flash", "Lights flash",
    "A bright full-field flash on both eyes.",
    lambda b: vision.flash(b, 1.0, stop_ms=60.0),
    duration_ms=300.0, watch=("photoreceptors",), real_fly="A startle; brief pause in walking.", tags=("vision",)))
