"""A Braitenberg fly built from scratch: it turns toward the side that tastes sugar."""

import openfly as of
from openfly import BrainBuilder

b = BrainBuilder("braitenberg", seed=0)
kw = dict(super_class="sensory", cell_class="gustatory", cell_sub_class="sugar", region="GNG")
L = b.add("LB3c", 10, side="left", center=(480, 330, 120), **kw)
R = b.add("LB3c", 10, side="right", center=(576, 330, 120), **kw)
dL = b.add("DNa02", 1, side="left", super_class="descending", center=(470, 200, 100))
dR = b.add("DNa02", 1, side="right", super_class="descending", center=(586, 200, 100))
inh_l = b.add("LAL_inh", 2, side="left", nt="gaba", center=(510, 210, 100))
inh_r = b.add("LAL_inh", 2, side="right", nt="gaba", center=(546, 210, 100))
b.connect(L, dL, synapses=18)
b.connect(R, dR, synapses=18)
b.connect(L, inh_l, synapses=8)
b.connect(R, inh_r, synapses=8)
b.connect(inh_l, dR, synapses=60)
b.connect(inh_r, dL, synapses=60)
brain = b.build()
print(brain.summary())

for side in ["left", "right", "both"]:
    sim = brain.simulator()
    sim.add(of.senses.taste(brain, "sugar", side=side))
    print(f"sugar on {side:<5} -> turn drive {sim.run(400).behavior()['turn']['drive']:+.2f}")

print("saved:", brain.save("braitenberg.flybrain"), "(try: openfly playground -d braitenberg.flybrain)")
