"""``Stimulus``: what the simulator feeds into the brain.

Every sense in openfly (smell, taste, vision, hearing...) produces
``Stimulus`` objects. A Stimulus says *which neurons* to drive, *how
hard* (firing rate in Hz) and *when*. Senses differ only in how they
choose the neurons and rates.

There are two kinds of drive:

* ``"poisson"`` (default): each target receives Poisson input events that
  force spikes, so it fires at about the requested rate. This is the
  Shiu et al. model of sensory/optogenetic activation.
* ``"current"``: a constant depolarisation in millivolts. Gentler; the
  neuron only fires if the push takes it over threshold.

Stimuli compose with ``+`` and can be time-varying (``modulation``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np


@dataclass
class Stimulus:
    """Drive ``targets`` (model indices) at ``rates`` between ``start_ms`` and ``stop_ms``."""

    targets: np.ndarray
    rates: np.ndarray | float = 150.0
    start_ms: float = 0.0
    stop_ms: float | None = None
    name: str = "stimulus"
    kind: str = "poisson"
    #: optional f(t_ms) -> scalar or per-target array multiplying ``rates``
    modulation: Callable[[float], np.ndarray | float] | None = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.targets = np.asarray(self.targets, dtype=np.int64).ravel()
        r = np.asarray(self.rates, dtype=np.float64)
        self.rates = np.broadcast_to(r, self.targets.shape).copy() if r.ndim == 0 else r.ravel().copy()
        if self.rates.shape != self.targets.shape:
            raise ValueError("rates must be a scalar or match targets")
        if self.kind not in ("poisson", "current"):
            raise ValueError("kind must be 'poisson' or 'current'")

    # ----------------------------------------------------------- timing
    @property
    def dynamic(self) -> bool:
        return self.modulation is not None

    def active(self, t_ms: float) -> bool:
        return t_ms >= self.start_ms and (self.stop_ms is None or t_ms < self.stop_ms)

    def rates_at(self, t_ms: float) -> np.ndarray:
        if not self.active(t_ms):
            return np.zeros_like(self.rates)
        if self.modulation is None:
            return self.rates
        m = np.asarray(self.modulation(t_ms - self.start_ms), dtype=np.float64)
        return np.clip(self.rates * m, 0.0, None) if self.kind == "poisson" else self.rates * m

    # ------------------------------------------------------- transforms
    def between(self, start_ms: float, stop_ms: float | None) -> "Stimulus":
        """Same stimulus, active only from ``start_ms`` to ``stop_ms``."""
        return replace(self, start_ms=float(start_ms), stop_ms=None if stop_ms is None else float(stop_ms))

    def scaled(self, factor: float) -> "Stimulus":
        return replace(self, rates=self.rates * factor)

    def named(self, name: str) -> "Stimulus":
        return replace(self, name=name)

    def __add__(self, other) -> "StimulusSet":
        return StimulusSet([self]) + other

    def __len__(self) -> int:
        return int(self.targets.size)

    def __repr__(self) -> str:
        when = f"{self.start_ms:g}-{'end' if self.stop_ms is None else f'{self.stop_ms:g}'} ms"
        mean = float(self.rates.mean()) if self.rates.size else 0.0
        rate = f"{mean:.0f} Hz" if self.kind == "poisson" else f"{mean:.1f} mV"
        dyn = ", dynamic" if self.dynamic else ""
        return f"<Stimulus {self.name!r}: {len(self)} neurons, {rate}, {when}{dyn}>"


class StimulusSet(list):
    """A list of stimuli that behaves like one (``a + b + c``)."""

    def __add__(self, other):
        out = StimulusSet(self)
        if isinstance(other, Stimulus):
            out.append(other)
        else:
            out.extend(other)
        return out

    def between(self, start_ms, stop_ms):
        return StimulusSet(s.between(start_ms, stop_ms) for s in self)

    def scaled(self, factor):
        return StimulusSet(s.scaled(factor) for s in self)

    @property
    def name(self) -> str:
        return " + ".join(s.name for s in self)


def as_stimuli(obj) -> list[Stimulus]:
    if obj is None:
        return []
    if isinstance(obj, Stimulus):
        return [obj]
    out = []
    for s in obj:
        out.extend(as_stimuli(s))
    return out
