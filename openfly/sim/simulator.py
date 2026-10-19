"""High-level simulator: "stimulate these, silence those, run, look at results".

    sim = brain.simulator()
    sim.stimulate("sugar_GRN", rate_hz=100)
    sim.silence("MBON")
    result = sim.run(duration_ms=1000, n_trials=5)
    result.top(10)

The Simulator turns names into neurons, schedules stimuli in time, runs
repeated trials (in parallel if you like) and returns a
:class:`~openfly.sim.results.SimResult`. It can also be advanced a few
milliseconds at a time (:meth:`advance`), which is what the live
playground and the virtual fly do.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from ..senses.base import Stimulus, as_stimuli
from .engine import LIFEngine
from .params import SHIU_2024, LIFParams
from .results import SimResult

_FORK_SIM = None


def _trial_worker(args):
    duration_ms, seed = args
    return _FORK_SIM._run_trial(duration_ms, seed)


class Simulator:
    """Whole-brain LIF simulation of one :class:`~openfly.Brain`.

    Parameters
    ----------
    brain:
        the connectome to simulate.
    params:
        :class:`~openfly.sim.params.LIFParams` (default: Shiu et al. 2024).
    seed:
        base random seed; trial ``k`` uses an independent stream derived from it.
    update_ms:
        how often time-varying stimuli are re-evaluated.
    """

    def __init__(self, brain, params: LIFParams = SHIU_2024, seed: int | None = 0, update_ms: float = 2.0):
        self.brain = brain
        self.params = params
        self.seed = seed
        self.update_ms = update_ms
        self.stimuli: list[Stimulus] = []
        self.silenced = np.zeros(0, dtype=np.int64)
        self.engine = LIFEngine(brain.W, params, seed)
        self._step_hooks: list = []

    # ------------------------------------------------------------ setup
    def stimulate(
        self,
        target,
        rate_hz: float | np.ndarray = 150.0,
        start_ms: float = 0.0,
        stop_ms: float | None = None,
        kind: str = "poisson",
        name: str | None = None,
    ) -> Stimulus | list[Stimulus]:
        """Drive neurons. ``target`` is a name, NeuronSet, list of names, or Stimulus objects.

        ``kind="poisson"`` forces spikes at ``rate_hz`` (optogenetics-like);
        ``kind="current"`` adds ``rate_hz`` *millivolts* of steady depolarisation.
        """
        if isinstance(target, Stimulus) or (isinstance(target, (list, tuple)) and target and isinstance(target[0], Stimulus)):
            stims = as_stimuli(target)
            self.stimuli.extend(stims)
            return stims
        ns = self.brain[target]
        stim = Stimulus(ns.idx, rate_hz, start_ms, stop_ms, name or ns.label or "stimulus", kind)
        self.stimuli.append(stim)
        return stim

    def add(self, *stimuli) -> "Simulator":
        """Add stimuli produced by :mod:`openfly.senses`."""
        for s in stimuli:
            self.stimuli.extend(as_stimuli(s))
        return self

    def silence(self, target) -> "Simulator":
        """Silence neurons for all following runs (like genetic silencing with Kir2.1)."""
        ns = self.brain[target]
        self.silenced = np.union1d(self.silenced, ns.idx)
        self.engine.silence(ns.idx, True)
        return self

    def unsilence(self, target=None) -> "Simulator":
        idx = self.silenced if target is None else self.brain[target].idx
        self.engine.silence(idx, False)
        self.silenced = np.setdiff1d(self.silenced, idx)
        return self

    def clear(self) -> "Simulator":
        """Remove all stimuli and silencing."""
        self.stimuli = []
        self.unsilence()
        return self

    def driven_neurons(self) -> np.ndarray:
        """Model indices of every neuron targeted by a stimulus."""
        if not self.stimuli:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate([s.targets for s in self.stimuli]))

    def on_step(self, fn) -> None:
        """Register ``fn(engine, spikes)`` to run after every time step (plasticity, recording...)."""
        self._step_hooks.append(fn)

    # ------------------------------------------------------------ drive
    def _drive_at(self, t_ms: float) -> None:
        poi_idx, poi_r, cur_idx, cur_v = [], [], [], []
        for s in self.stimuli:
            if not s.active(t_ms):
                continue
            r = s.rates_at(t_ms)
            if s.kind == "poisson":
                poi_idx.append(s.targets)
                poi_r.append(r)
            else:
                cur_idx.append(s.targets)
                cur_v.append(r)
        eng = self.engine
        if poi_idx:
            idx = np.concatenate(poi_idx)
            r = np.concatenate(poi_r)
            u, inv = np.unique(idx, return_inverse=True)
            eng.replace_drive(u, np.bincount(inv, weights=r))
        elif eng._drive_idx.size:
            eng.replace_drive(np.zeros(0, np.int64), np.zeros(0))
        if cur_idx or eng._current_idx.size:
            eng.current[eng._current_idx] = 0
            if cur_idx:
                idx = np.concatenate(cur_idx)
                v = np.concatenate(cur_v)
                u, inv = np.unique(idx, return_inverse=True)
                eng.set_current(u, np.bincount(inv, weights=v))
            else:
                eng._current_idx = np.zeros(0, np.int64)

    def _boundaries(self, t0: float, t1: float) -> list[float]:
        pts = {t0, t1}
        for s in self.stimuli:
            for b in (s.start_ms, s.stop_ms):
                if b is not None and t0 < b < t1:
                    pts.add(float(b))
        return sorted(pts)

    # ---------------------------------------------------------- running
    def advance(self, duration_ms: float) -> tuple[np.ndarray, np.ndarray]:
        """Continue the current simulation for ``duration_ms``.

        Returns ``(spike_times_ms, neuron_indices)`` with absolute times.
        Stimulus windows are interpreted in absolute engine time.
        """
        eng = self.engine
        dt = self.params.dt_ms
        t0 = eng.time_ms
        t_end = t0 + duration_ms
        dynamic = any(s.dynamic for s in self.stimuli)
        hook = self._make_hook()
        times, neurons = [], []
        edges = self._boundaries(t0, t_end)
        for a, b in zip(edges[:-1], edges[1:]):
            t = a
            while t < b - 1e-9:
                seg_end = min(b, t + self.update_ms) if dynamic else b
                self._drive_at(t)
                n = int(round((seg_end - eng.time_ms) / dt))
                if n > 0:
                    st, nr = eng.run(n, callback=hook)
                    if st.size:
                        times.append(st * dt)
                        neurons.append(nr)
                t = eng.time_ms
        if not times:
            return np.zeros(0, np.float32), np.zeros(0, np.int64)
        return np.concatenate(times).astype(np.float32), np.concatenate(neurons)

    def _make_hook(self):
        if not self._step_hooks:
            return None
        hooks = list(self._step_hooks)

        def hook(engine, spikes):
            for h in hooks:
                h(engine, spikes)

        return hook

    def reset(self, seed: int | None = None) -> None:
        """Back to rest (stimuli and silencing are kept)."""
        self.engine.reset()
        if seed is not None:
            self.engine.rng = np.random.default_rng(seed)

    def _run_trial(self, duration_ms: float, seed) -> tuple[np.ndarray, np.ndarray]:
        self.reset(seed)
        return self.advance(duration_ms)

    def _trial_seeds(self, n_trials: int, seed) -> list:
        base = self.seed if seed is None else seed
        ss = np.random.SeedSequence(base)
        return [np.random.default_rng(s).integers(2**63) for s in ss.spawn(n_trials)]

    def run(
        self,
        duration_ms: float = 1000.0,
        n_trials: int = 1,
        seed: int | None = None,
        n_jobs: int = 1,
        progress: bool = False,
        label: str | None = None,
    ) -> SimResult:
        """Run ``n_trials`` independent trials of ``duration_ms`` each.

        ``n_jobs > 1`` runs trials in parallel processes (Linux/macOS).
        """
        seeds = self._trial_seeds(n_trials, seed)
        t0 = time.time()
        outs = None
        if n_jobs != 1 and n_trials > 1 and "fork" in mp.get_all_start_methods():
            global _FORK_SIM
            _FORK_SIM = self
            procs = mp.cpu_count() if n_jobs in (-1, 0) else min(n_jobs, n_trials)
            try:
                with mp.get_context("fork").Pool(procs) as pool:
                    outs = pool.map(_trial_worker, [(duration_ms, s) for s in seeds])
            finally:
                _FORK_SIM = None
        if outs is None:
            outs = []
            for k, s in enumerate(seeds):
                outs.append(self._run_trial(duration_ms, s))
                if progress:
                    print(f"  trial {k + 1}/{n_trials} done ({time.time() - t0:.1f} s)", flush=True)
        times = np.concatenate([o[0] for o in outs]) if outs else np.zeros(0, np.float32)
        neurons = np.concatenate([o[1] for o in outs]) if outs else np.zeros(0, np.int64)
        trials = np.concatenate([np.full(o[0].size, k, np.int32) for k, o in enumerate(outs)]) if outs else np.zeros(0, np.int32)
        return SimResult(
            brain=self.brain,
            times_ms=times,
            neurons=neurons,
            trials=trials,
            duration_ms=float(duration_ms),
            n_trials=n_trials,
            label=label or " + ".join(s.name for s in self.stimuli) or "spontaneous",
            stimuli=[repr(s) for s in self.stimuli],
            silenced=self.silenced.copy(),
            driven=self.driven_neurons(),
            wall_time_s=time.time() - t0,
        )

    def stream(self, chunk_ms: float = 20.0, duration_ms: float | None = None):
        """Generator yielding ``(t_ms, spike_times_ms, neuron_indices)`` chunk by chunk.

        Keeps going forever when ``duration_ms`` is None, which suits live displays.
        """
        done = 0.0
        while duration_ms is None or done < duration_ms:
            st, nr = self.advance(chunk_ms)
            done += chunk_ms
            yield self.engine.time_ms, st, nr


def simulate(brain, stimuli=None, silence=None, duration_ms: float = 1000.0, n_trials: int = 1, **kwargs) -> SimResult:
    """One-liner: ``openfly.simulate(brain, openfly.senses.taste(brain, "sugar"))``."""
    run_kw = {k: kwargs.pop(k) for k in ("seed", "n_jobs", "progress", "label") if k in kwargs}
    sim = Simulator(brain, **kwargs)
    if stimuli is not None:
        sim.add(stimuli)
    if silence is not None:
        sim.silence(silence)
    return sim.run(duration_ms, n_trials, **run_kw)
