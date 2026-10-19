"""Quickstart: load a brain, taste sugar, read the behaviour.

    python examples/01_quickstart.py            # toy brain (offline)
    python examples/01_quickstart.py flywire_783
"""

import sys

import openfly as of

name = sys.argv[1] if len(sys.argv) > 1 else "toy"
brain = of.load(name, transmitters="curated" if name.startswith("flywire") else None)
print(brain.summary(), "\n")

for label, stimuli in [
    ("sugar", of.senses.taste(brain, "sugar")),
    ("sugar + bitter", of.senses.taste(brain, "sugar") + of.senses.taste(brain, "bitter")),
]:
    sim = brain.simulator(seed=0)
    sim.add(stimuli)
    res = sim.run(500)
    print(f"--- {label}: {res}")
    print(res.top(5, exclude_driven=True)[["cell_type", "side", "rate_hz"]].to_string())
    print(of.motor.describe(res.behavior()), "\n")
