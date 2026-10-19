"""Virtual genetic screens: who drives MN9 (sufficiency) and who is needed (necessity)?"""

import sys

import openfly as of
from openfly import senses
from openfly.learning import activation_screen, silencing_screen

name = sys.argv[1] if len(sys.argv) > 1 else "toy"
brain = of.load(name, transmitters="curated" if name.startswith("flywire") else None)
print("Activation screen (rate model):")
print(activation_screen(brain, "MN9", top=10, progress=True).to_string())
print("\nSilencing screen while tasting sugar (rate model):")
print(silencing_screen(brain, senses.taste(brain, "sugar"), "MN9", model="rate", top=10).to_string())
