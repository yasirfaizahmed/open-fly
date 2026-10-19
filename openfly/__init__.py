"""openfly: explore, simulate, visualise and train the fruit fly brain.

Quick start::

    import openfly as of

    brain = of.load("toy")            # offline teaching brain; "flywire_783" = the real one
    sim = brain.simulator()
    sim.add(of.senses.taste(brain, "sugar"))
    res = sim.run(500)
    print(of.motor.describe(res.behavior()))

Start here: ``docs/README.md``. Live playground: ``openfly playground``.
"""

from ._version import __version__
from . import knowledge, motor
from .brain import Brain
from .data import BrainBuilder, datasets, download, load
from .neurons import NeuronSet
from .sim import LIFParams, SHIU_2024, SimResult, Simulator, simulate
from . import senses  # noqa: E402  (needs Brain/sim)

__all__ = [
    "Brain",
    "BrainBuilder",
    "LIFParams",
    "NeuronSet",
    "SHIU_2024",
    "SimResult",
    "Simulator",
    "__version__",
    "datasets",
    "download",
    "knowledge",
    "load",
    "motor",
    "senses",
    "simulate",
]
