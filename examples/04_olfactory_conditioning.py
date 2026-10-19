"""Teach the fly that banana means reward (dopamine-gated mushroom body plasticity)."""

import sys

import openfly as of
from openfly.learning import OlfactoryConditioning

name = sys.argv[1] if len(sys.argv) > 1 else "toy"
brain = of.load(name, transmitters="curated" if name.startswith("flywire") else None)
for reinforcer in ["reward", "punishment"]:
    exp = OlfactoryConditioning(brain, cs_plus="banana", cs_minus="acetic_acid", reinforcer=reinforcer)
    res = exp.run(n_trials=3, odor_ms=800, test_ms=500, progress=True)
    print(res.summary(), "\n")
