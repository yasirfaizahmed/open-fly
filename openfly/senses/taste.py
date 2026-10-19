"""Taste: tastants -> gustatory receptor neurons (GRNs).

Flies taste with their labellum (the tip of the proboscis), legs, wings
and pharynx. FlyWire's gustatory neurons are typed by likely modality
(Tastekin et al. 2026): sugar, bitter, water, high salt, glutamate...
Tasting sugar activates sugar GRNs, which, through a few interneurons,
drive MN9 and proboscis extension. That chain is the flagship result of
Shiu et al. (2024), and you can watch it happen.
"""

from __future__ import annotations

import numpy as np

from .base import Stimulus

#: tastant -> alias names (see knowledge.py) whose neurons it activates, with weights
TASTANTS: dict[str, dict[str, float]] = {
    "sugar": {"sugar_GRN": 1.0},
    "bitter": {"bitter_GRN": 1.0},
    "water": {"water_GRN": 1.0},
    "low_salt": {"sugar_GRN": 0.4},
    "high_salt": {"salt_high_GRN": 1.0, "bitter_GRN": 0.3},
    "glutamate": {"glutamate_GRN": 1.0},
    "fruit_juice": {"sugar_GRN": 0.8, "water_GRN": 0.6, "taste_peg_GRN": 0.3},
    "rotten_fruit": {"sugar_GRN": 0.5, "bitter_GRN": 0.6},
}


def list_tastants() -> list[str]:
    return sorted(TASTANTS)


def taste(
    brain,
    tastant: str = "sugar",
    intensity: float = 1.0,
    *,
    side: str = "both",
    max_rate_hz: float = 150.0,
    start_ms: float = 0.0,
    stop_ms: float | None = None,
) -> Stimulus:
    """Stimulus for touching ``tastant`` with the mouthparts.

    ``side`` = ``"left"``, ``"right"`` or ``"both"``; ``intensity`` 0..1.
    """
    if tastant not in TASTANTS:
        raise KeyError(f"unknown tastant {tastant!r}; known: {', '.join(list_tastants())}")
    targets, rates = [], []
    for alias, w in TASTANTS[tastant].items():
        try:
            ns = brain[alias]
        except KeyError:
            continue
        if side in ("left", "right"):
            ns = ns.where(side=side)
        if len(ns):
            targets.append(ns.idx)
            rates.append(np.full(len(ns), max_rate_hz * w * float(np.clip(intensity, 0, 1.5))))
    if not targets:
        return Stimulus(np.zeros(0, np.int64), 0.0, start_ms, stop_ms, f"taste:{tastant}")
    idx = np.concatenate(targets)
    r = np.concatenate(rates)
    u, inv = np.unique(idx, return_inverse=True)
    return Stimulus(u, np.bincount(inv, weights=r), start_ms, stop_ms, f"taste:{tastant}",
                    info={"tastant": tastant, "intensity": intensity, "side": side})
