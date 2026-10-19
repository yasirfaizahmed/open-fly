"""Shiu et al. 2024 in miniature: how hard must sugar neurons fire to drive MN9?

    python examples/02_dose_response.py flywire_783
"""

import sys

import openfly as of

name = sys.argv[1] if len(sys.argv) > 1 else "toy"
brain = of.load(name, transmitters="curated" if name.startswith("flywire") else None)
print(f"{'sugar GRN rate':>15} {'MN9 rate':>10} {'neurons active':>15}")
for rate in [10, 25, 50, 100, 150, 200]:
    sim = brain.simulator(seed=0)
    sim.stimulate("sugar_GRN", rate)
    res = sim.run(500)
    print(f"{rate:>12} Hz {res.rate('MN9'):>8.1f} Hz {int((res.rates() > 0).sum()):>15,}")
