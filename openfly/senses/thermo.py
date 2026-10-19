"""Temperature and humidity: thermo- and hygrosensory neurons of the arista.

Flies have hot cells and cold cells in the arista and sacculus, and
"dry" and "moist" hygroreceptors (Frank et al. 2015; Knecht et al. 2017;
Marin et al. 2020). They project to the posterior antennal lobe (VP
glomeruli).
"""

from __future__ import annotations

import numpy as np

from .base import Stimulus

_ALIASES = {"hot": "hot_TRN", "cold": "cold_TRN", "dry": "dry_HRN", "moist": "moist_HRN"}


def _stim(brain, kind, intensity, side, max_rate_hz, start_ms, stop_ms, label):
    try:
        ns = brain[_ALIASES[kind]]
    except KeyError:
        return Stimulus(np.zeros(0, np.int64), 0.0, start_ms, stop_ms, label)
    if side in ("left", "right"):
        ns = ns.where(side=side)
    return Stimulus(ns.idx, max_rate_hz * intensity, start_ms, stop_ms, label)


def temperature(brain, kind: str = "hot", intensity: float = 1.0, side: str = "both", max_rate_hz: float = 150.0,
                start_ms: float = 0.0, stop_ms: float | None = None) -> Stimulus:
    """``kind="hot"`` or ``"cold"``. Use ``side`` for a heat source on one side."""
    if kind not in ("hot", "cold"):
        raise ValueError("kind must be 'hot' or 'cold'")
    return _stim(brain, kind, intensity, side, max_rate_hz, start_ms, stop_ms, f"temperature:{kind}")


def humidity(brain, kind: str = "moist", intensity: float = 1.0, side: str = "both", max_rate_hz: float = 150.0,
             start_ms: float = 0.0, stop_ms: float | None = None) -> Stimulus:
    """``kind="moist"`` or ``"dry"`` air."""
    if kind not in ("moist", "dry"):
        raise ValueError("kind must be 'moist' or 'dry'")
    return _stim(brain, kind, intensity, side, max_rate_hz, start_ms, stop_ms, f"humidity:{kind}")
