"""A fast firing-rate approximation of the LIF model.

Spiking simulations are faithful but slow (seconds per trial on the full
brain). Many questions ("roughly what lights up if I activate X?",
"which of these 200 cell types drives MN9 most?", "which gains make
the fly do Y?") need thousands of evaluations. :class:`RateModel` gives
answers in milliseconds.

Each neuron's rate is a threshold-linear function of its summed
synaptic drive ``D = sum_pre rate_pre * signed_synapses``, fitted to the
LIF model's input-output curve (``docs/05-simulation.md``)::

    rate = r_max * tanh(gain * k * max(D - theta, 0) / r_max)

with k ~ 0.0045 Hz per (Hz*synapse), theta ~ 1000 Hz*synapse and
r_max ~ 150 Hz for the Shiu parameters (the tanh reproduces the LIF
model's saturation; e.g. D = 10k -> 40 Hz, 23k -> 87 Hz vs 44/81 Hz
in the spiking model). The network is relaxed to a fixed point by damped
iteration. Stimulated neurons are clamped to their stimulus rate.

It ignores spike timing, refractoriness and synchrony, so treat it as a
quick map. Check important conclusions with the spiking :class:`Simulator`.
"""

from __future__ import annotations

import numpy as np

from ..senses.base import as_stimuli


class RateModel:
    def __init__(self, brain, k: float = 0.0045, theta: float = 1000.0, r_max: float = 150.0,
                 damping: float = 0.5, gains: np.ndarray | None = None):
        self.brain = brain
        self.WT = brain.W.T.tocsr().astype(np.float32)  # rows = post
        self.k, self.theta, self.r_max, self.damping = k, theta, r_max, damping
        self.gains = np.ones(brain.N) if gains is None else np.asarray(gains, float)
        self.silenced = np.zeros(brain.N, dtype=bool)

    def silence(self, target) -> "RateModel":
        self.silenced[self.brain[target].idx] = True
        return self

    def run(self, stimuli=None, clamp=None, n_iter: int = 40, tol: float = 0.1, out_gain: np.ndarray | None = None) -> np.ndarray:
        """Fixed-point rates (Hz) of every neuron.

        ``stimuli``: Stimulus objects (their rates at t=0 are used), or
        ``clamp``: dict ``{target: rate_hz}``.
        """
        N = self.brain.N
        fixed = np.zeros(N, dtype=bool)
        fixed_rate = np.zeros(N)
        for s in as_stimuli(stimuli):
            if s.kind == "poisson":
                fixed[s.targets] = True
                fixed_rate[s.targets] += s.rates_at(s.start_ms)
        for target, rate in (clamp or {}).items():
            idx = self.brain[target].idx
            fixed[idx] = True
            fixed_rate[idx] = rate
        r = np.where(fixed, fixed_rate, 0.0)
        g = self.gains
        og = np.ones(N) if out_gain is None else out_gain
        og = np.where(self.silenced, 0.0, og)
        for _ in range(n_iter):
            D = self.WT @ (r * og).astype(np.float32)
            new = self.r_max * np.tanh(g * self.k * np.maximum(D - self.theta, 0.0) / self.r_max)
            new = np.where(fixed, fixed_rate, new)
            new[self.silenced & ~fixed] = 0.0
            nxt = (1 - self.damping) * r + self.damping * new
            if np.max(np.abs(nxt - r)) < tol:
                r = nxt
                break
            r = nxt
        return r

    def response(self, target, **kw) -> float:
        """Mean fixed-point rate of a group under ``run(**kw)``."""
        r = self.run(**kw)
        return float(r[self.brain[target].idx].mean())
