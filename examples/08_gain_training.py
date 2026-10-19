"""Connectome-constrained training: tune cell-type gains so laced sugar is rejected."""

import openfly as of
from openfly import senses
from openfly.learning import GainTask

brain = of.load("toy")
conditions = {"sugar": senses.taste(brain, "sugar"),
              "laced": senses.taste(brain, "sugar") + senses.taste(brain, "bitter", 0.25)}
task = GainTask(brain, ["G2N-1", "GABA_bitter_IN", "LB3c", "LB1a"], conditions, {"MN9": "MN9"},
                objective=lambda r: r["sugar"]["MN9"] / 100 - r["laced"]["MN9"] / 100)
res = task.train(generations=25, progress=True)
print(res.summary())
