"""Optimise a brain for a task: connectome-constrained training.

Can you "train" a connectome? The wiring diagram fixes **who connects to
whom** but not exactly **how strong** each connection is, how excitable
each cell type is, or how neuromodulators reshape them. Those are
legitimate free parameters. Training means searching them, *without
adding wires*, until the circuit does what you want. This is the idea
behind connectome-constrained models of the visual system
(Lappalainen et al. 2024 Nature), at the scale of a laptop.

This module uses **evolution strategies** (Salimans et al. 2017), which
only need to *run* the model, not differentiate it, so they work with
the spiking simulator and the fast rate model alike:

1. sample ``popsize`` random perturbations of the parameters,
2. score each with your objective,
3. move the parameters toward the better-scoring perturbations,
4. repeat.

:class:`GainTask` packages the common case: learn one output gain per
neuron group so that a set of experimental *conditions* produces the
behaviour your *objective* rewards. When the wiring makes a task
impossible, training will plateau, which tells you something about the
circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


class EvolutionStrategy:
    """OpenAI-style ES with antithetic sampling and rank-based fitness shaping."""

    def __init__(self, x0, sigma: float = 0.1, lr: float = 0.05, popsize: int = 16, seed: int | None = 0,
                 bounds: tuple[float, float] | None = None):
        self.x = np.asarray(x0, dtype=float).copy()
        self.sigma, self.lr = sigma, lr
        self.popsize = popsize + (popsize % 2)
        self.rng = np.random.default_rng(seed)
        self.bounds = bounds
        self._eps = None
        self.m = np.zeros_like(self.x)
        self.v = np.zeros_like(self.x)
        self.t = 0

    def ask(self) -> np.ndarray:
        half = self.rng.standard_normal((self.popsize // 2, self.x.size))
        self._eps = np.concatenate([half, -half])
        cand = self.x + self.sigma * self._eps
        return np.clip(cand, *self.bounds) if self.bounds else cand

    def tell(self, fitness) -> None:
        f = np.asarray(fitness, dtype=float)
        ranks = np.empty_like(f)
        ranks[np.argsort(f)] = np.arange(f.size)
        shaped = ranks / (f.size - 1) - 0.5
        grad = (shaped[:, None] * self._eps).mean(axis=0) / self.sigma
        # Adam step (ascent)
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad**2
        mh = self.m / (1 - 0.9**self.t)
        vh = self.v / (1 - 0.999**self.t)
        self.x = self.x + self.lr * mh / (np.sqrt(vh) + 1e-8)
        if self.bounds:
            self.x = np.clip(self.x, *self.bounds)


@dataclass
class TrainingResult:
    groups: list
    gains: np.ndarray
    history: pd.DataFrame
    before: dict
    after: dict
    objective_before: float
    objective_after: float
    notes: list = field(default_factory=list)

    def gain_table(self) -> pd.DataFrame:
        return pd.DataFrame({"group": self.groups, "gain": self.gains}).sort_values("gain", key=lambda s: (s - 1).abs(), ascending=False)

    def summary(self) -> str:
        lines = [f"objective: {self.objective_before:.3f} -> {self.objective_after:.3f}"]
        for _, r in self.gain_table().head(8).iterrows():
            arrow = "up" if r.gain > 1.05 else "down" if r.gain < 0.95 else "="
            lines.append(f"  {r.group:<24} gain {r.gain:5.2f}  ({arrow})")
        for cond in self.before:
            b = {k: round(v, 2) for k, v in self.before[cond].items()}
            a = {k: round(v, 2) for k, v in self.after[cond].items()}
            lines.append(f"  {cond}: {b} -> {a}")
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


class GainTask:
    """Learn per-group output gains so that conditions produce the desired readouts.

    Parameters
    ----------
    brain:
        any Brain.
    groups:
        names / NeuronSets whose output gain is trainable (gain 1 = as wired).
    conditions:
        ``{name: stimuli}``: the experiments run for every candidate.
    readouts:
        ``{name: target}``: neuron groups whose mean rate is reported.
    objective:
        ``f(results) -> float`` to maximise, where
        ``results[condition][readout]`` is a rate in Hz.
    model:
        ``"rate"`` (fast :class:`~openfly.sim.rate.RateModel`) or ``"lif"``
        (spiking :class:`~openfly.Simulator`, slower but faithful).
    """

    def __init__(self, brain, groups, conditions: dict, readouts: dict, objective: Callable[[dict], float],
                 model: str = "rate", duration_ms: float = 300.0, bounds=(0.0, 4.0), seed: int = 0):
        self.brain = brain
        self.groups = [brain[g] for g in groups]
        self.names = [g if isinstance(g, str) else (g.label or f"group{k}") for k, g in enumerate(groups)]
        self.conditions, self.readouts, self.objective = conditions, readouts, objective
        self.model, self.duration_ms, self.bounds, self.seed = model, duration_ms, bounds, seed
        self._readout_idx = {k: brain[v].idx for k, v in readouts.items()}
        if model == "rate":
            from ..sim.rate import RateModel

            self._rm = RateModel(brain)
        elif model == "lif":
            from ..sim.simulator import Simulator

            self._sim = Simulator(brain, seed=seed)
        else:
            raise ValueError("model must be 'rate' or 'lif'")

    def _gain_vector(self, x) -> np.ndarray:
        g = np.ones(self.brain.N, dtype=np.float32)
        for ns, val in zip(self.groups, x):
            g[ns.idx] = val
        return g

    def evaluate(self, x) -> dict:
        g = self._gain_vector(x)
        out = {}
        for cname, stim in self.conditions.items():
            if self.model == "rate":
                r = self._rm.run(stim, out_gain=g)
            else:
                sim = self._sim
                sim.stimuli = []
                sim.add(stim)
                sim.engine.set_out_gain(g)
                sim.reset(self.seed)
                _, nr = sim.advance(self.duration_ms)
                r = np.bincount(nr, minlength=self.brain.N) / (self.duration_ms / 1000)
            out[cname] = {k: float(r[idx].mean()) for k, idx in self._readout_idx.items()}
        return out

    def score(self, x) -> float:
        return float(self.objective(self.evaluate(x)))

    def train(self, generations: int = 30, popsize: int = 12, sigma: float = 0.25, lr: float = 0.15,
              progress: bool = False, callback=None) -> TrainingResult:
        x0 = np.ones(len(self.groups))
        es = EvolutionStrategy(x0, sigma=sigma, lr=lr, popsize=popsize, seed=self.seed, bounds=self.bounds)
        before = self.evaluate(x0)
        f0 = float(self.objective(before))
        hist = []
        best_x, best_f = x0.copy(), f0
        for gen in range(generations):
            cands = es.ask()
            fits = [self.score(c) for c in cands]
            es.tell(fits)
            fx = self.score(es.x)
            if fx > best_f:
                best_x, best_f = es.x.copy(), fx
            hist.append({"generation": gen + 1, "objective": fx, "best": best_f, "population_mean": float(np.mean(fits))})
            if progress:
                print(f"  gen {gen + 1:3d}: objective {fx:.3f} (best {best_f:.3f})", flush=True)
            if callback:
                callback(gen, es.x, fx)
        after = self.evaluate(best_x)
        notes = []
        if best_f <= f0 + 1e-6:
            notes.append("no improvement: the wiring may not allow this behaviour with these groups")
        return TrainingResult(self.names, best_x, pd.DataFrame(hist), before, after, f0, best_f, notes)
