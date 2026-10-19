"""Learning: teach the brain new things.

* :mod:`~openfly.learning.mushroom_body`: biological learning (dopamine-gated KC->MBON plasticity, conditioning)
* :mod:`~openfly.learning.evolve`: optimise parameters for a task (evolution strategies)
* :mod:`~openfly.learning.screens`: activation / silencing screens to find which neurons drive a behaviour
"""

from .evolve import EvolutionStrategy, GainTask, TrainingResult
from .mushroom_body import ConditioningResult, MushroomBodyPlasticity, OlfactoryConditioning, valence_index
from .screens import activation_screen, silencing_screen, upstream_groups

__all__ = [
    "ConditioningResult", "EvolutionStrategy", "GainTask", "MushroomBodyPlasticity", "OlfactoryConditioning",
    "TrainingResult", "activation_screen", "silencing_screen", "upstream_groups", "valence_index",
]
