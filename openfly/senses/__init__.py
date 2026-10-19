"""Senses: turn things a fly experiences into neuron drive.

Every function takes the brain first and returns a :class:`Stimulus`
(or a :class:`StimulusSet`) ready for ``sim.add(...)``::

    from openfly import senses
    senses.smell(brain, "banana", concentration=0.8)
    senses.taste(brain, "sugar")
    senses.looming(brain, azimuth=45)
    senses.hear(brain, "pulse_song")
    senses.scenario(brain, "ripe_banana")

See ``docs/06-senses-and-stimuli.md``.
"""

from . import mechano, olfaction, scenarios, taste as _taste, thermo, vision
from .base import Stimulus, StimulusSet, as_stimuli
from .mechano import hear, touch, wind
from .olfaction import ODORS, Odor, custom_odor, list_odors, smell
from .scenarios import SCENARIOS, Scenario, list_scenarios
from .taste import TASTANTS, list_tastants, taste
from .thermo import humidity, temperature
from .vision import VisualField, flash, grating, image, looming, moving_object, spot


def scenario(brain, name: str, start_ms: float = 0.0) -> StimulusSet:
    """Stimuli of a named scenario (see :mod:`openfly.senses.scenarios`)."""
    return scenarios.get(name).build(brain, start_ms)


__all__ = [
    "ODORS", "Odor", "SCENARIOS", "Scenario", "Stimulus", "StimulusSet", "TASTANTS", "VisualField",
    "as_stimuli", "custom_odor", "flash", "grating", "hear", "humidity", "image", "list_odors",
    "list_scenarios", "list_tastants", "looming", "mechano", "moving_object", "olfaction", "scenario",
    "scenarios", "smell", "spot", "taste", "temperature", "thermo", "touch", "vision", "wind",
]
