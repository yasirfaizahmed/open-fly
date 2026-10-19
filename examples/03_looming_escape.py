"""A predator looms: LC4/LPLC2 -> giant fiber -> escape. Writes a 3D HTML replay."""

import sys

import openfly as of
from openfly import analysis, senses

name = sys.argv[1] if len(sys.argv) > 1 else "toy"
brain = of.load(name, transmitters="curated" if name.startswith("flywire") else None)

sim = brain.simulator(seed=0)
sim.add(senses.looming(brain, azimuth=50, duration_ms=400))
res = sim.run(500)
gf = brain["giant_fiber"]
print("giant fiber rate  left %.0f Hz / right %.0f Hz" % (res.rate(gf.left), res.rate(gf.right)))
print("first giant fiber spike at %.1f ms" % res.first_spike_ms(gf))
print("escape drive:", res.behavior()["escape"]["drive"])
print(analysis.strongest_path(brain, "LC4", "giant_fiber")[["cell_type", "side", "synapses"]])
print("replay:", res.movie("looming.html", title="A shadow looms from the right"))
