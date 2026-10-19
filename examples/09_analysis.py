"""Wiring analysis without simulation: paths, hops, influence, hubs."""

import sys

import openfly as of
from openfly import analysis

name = sys.argv[1] if len(sys.argv) > 1 else "toy"
brain = of.load(name)
print(analysis.strongest_path(brain, "sugar_GRN", "MN9")[["step", "cell_type", "side", "nt", "synapses"]])
print("\nhops from photoreceptors:\n", analysis.hops(brain, "photoreceptors").value_counts().sort_index())
print("\nbitter influence by cell type:\n", analysis.summarize(brain, analysis.influence(brain, "bitter_GRN"), "cell_type", top=8))
print("\nhubs:\n", analysis.hubs(brain, 5)[["cell_type", "out_partners", "out_synapses"]])
