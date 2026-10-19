"""Hearing, wind and touch: mechanosensory neurons.

* **Johnston's organ (JO)** in the antenna: subgroups A/B respond to
  sound (near-field particle velocity, e.g. the courtship song a male
  plays with his wing); C/E respond to sustained deflections from wind
  and gravity (Kamikouchi et al. 2009; Yorozu et al. 2009).
* **Bristles** on the head and eyes: touch triggers grooming
  (Hampel et al. 2015, 2017).

The courtship song of *D. melanogaster* is a pulse train with an
inter-pulse interval (IPI) of ~35 ms; :func:`hear` with
``sound="pulse_song"`` reproduces that rhythm.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Stimulus


def _set(brain, alias, side="both"):
    try:
        ns = brain[alias]
    except KeyError:
        return None
    if side in ("left", "right"):
        ns = ns.where(side=side)
    return ns if len(ns) else None


def hear(brain, sound: str = "pulse_song", intensity: float = 1.0, ipi_ms: float = 35.0, pulse_ms: float = 8.0,
         freq_hz: float = 200.0, side: str = "both", max_rate_hz: float = 150.0,
         start_ms: float = 0.0, stop_ms: float | None = None) -> Stimulus:
    """Sound at the antennae.

    ``sound``: ``"pulse_song"`` (courtship pulses), ``"sine_song"`` (the
    continuous ~160 Hz hum), or ``"tone"`` (steady drive scaled by ``freq_hz``,
    strongest around 100-300 Hz).
    """
    ns = _set(brain, "JO_auditory", side)
    if ns is None:
        return Stimulus(np.zeros(0, np.int64), 0.0, start_ms, stop_ms, f"hear:{sound}")
    if sound == "pulse_song":
        def mod(t):
            return 1.0 if (t % ipi_ms) < pulse_ms else 0.05
    elif sound == "sine_song":
        def mod(t):
            return 0.7
    elif sound == "tone":
        tuning = math.exp(-((math.log10(max(freq_hz, 1.0)) - math.log10(200.0)) ** 2) / 0.18)

        def mod(t, tuning=tuning):
            return tuning
    else:
        raise ValueError("sound must be 'pulse_song', 'sine_song' or 'tone'")
    return Stimulus(ns.idx, max_rate_hz * intensity, start_ms, stop_ms, f"hear:{sound}", modulation=mod,
                    info={"sound": sound, "ipi_ms": ipi_ms})


def wind(brain, direction_deg: float = 0.0, speed: float = 1.0, max_rate_hz: float = 120.0,
         start_ms: float = 0.0, stop_ms: float | None = None) -> Stimulus:
    """Wind blowing *from* ``direction_deg`` (0 = from the front, 90 = from the right).

    Wind from one side deflects both aristae, but the antenna facing the
    wind more strongly, so JO-C/E neurons on that side fire more.
    """
    ns = _set(brain, "JO_wind_gravity")
    if ns is None:
        return Stimulus(np.zeros(0, np.int64), 0.0, start_ms, stop_ms, "wind")
    side = brain.neurons["side"].to_numpy()[ns.idx]
    lateral = math.sin(math.radians(direction_deg))
    gain = np.where(side == "right", 0.6 + 0.4 * lateral, np.where(side == "left", 0.6 - 0.4 * lateral, 0.6))
    return Stimulus(ns.idx, max_rate_hz * min(speed, 1.5) * gain, start_ms, stop_ms, f"wind({direction_deg:.0f})")


def touch(brain, where: str = "eye", intensity: float = 1.0, side: str = "both", max_rate_hz: float = 150.0,
          start_ms: float = 0.0, stop_ms: float | None = None) -> Stimulus:
    """Touch bristles (``where`` = ``"eye"`` or ``"head"``), e.g. dust that needs grooming."""
    alias = {"eye": "eye_bristle", "head": "head_bristle"}[where]
    ns = _set(brain, alias, side)
    if ns is None:
        return Stimulus(np.zeros(0, np.int64), 0.0, start_ms, stop_ms, f"touch:{where}")
    return Stimulus(ns.idx, max_rate_hz * intensity, start_ms, stop_ms, f"touch:{where}")
