"""The brain drives a virtual fly through several worlds."""

import sys

import openfly as of
from openfly.agent import Arena, VirtualFly

name = sys.argv[1] if len(sys.argv) > 1 else "toy"
brain = of.load(name, transmitters="curated" if name.startswith("flywire") else None)
for arena in Arena.PRESETS:
    fly = VirtualFly(brain, arena, seed=0, turn_highpass_s=3.0 if brain.N > 20000 else None)
    fly.run(20)
    events = ", ".join(f"{e['t_s']:.1f}s {e['event']}" for e in fly.events[:4]) or "none"
    print(f"{arena:<14} energy {fly.energy:5.2f}  distance to food {fly.distance_to():5.1f} mm  events: {events}")
    try:
        import matplotlib

        matplotlib.use("Agg")
        fly.plot().figure.savefig(f"fly_{arena}.png", dpi=100, bbox_inches="tight")
    except ImportError:
        pass
