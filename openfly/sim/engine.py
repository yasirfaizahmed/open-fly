"""Step-wise, event-driven leaky integrate-and-fire engine (pure NumPy).

This is the heart of the simulator. It runs the Shiu et al. (2024)
whole-brain model on ~140k neurons and ~15M connections on a laptop CPU,
and it can be advanced a few steps at a time. Being able to step it is
what makes the live playground, the virtual fly and the learning rules
possible.

Why it is fast: almost all neurons in the model sit exactly at rest
(``v == v_rest``, ``g == 0``), and for them the update equations do
nothing. The engine therefore keeps an *awake set*, the neurons that are
not at rest, and only integrates those. A neuron wakes up when it gets
synaptic input, Poisson drive or current, and goes back to sleep once it
has relaxed back to rest.

The order of operations inside one time step mirrors Brian2's default
schedule (state update -> threshold -> synapses -> reset), so the results
match the original Brian2 implementation statistically (see
``docs/13-validation.md``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from .params import SHIU_2024, LIFParams

_SLEEP_EPS = 1e-4  # mV; below this distance from rest a neuron is put to sleep
_COMPACT_EVERY = 32  # steps between awake-set compactions
_EMPTY = np.zeros(0, dtype=np.int64)


class LIFEngine:
    """Low-level LIF network that works on model indices ``0..N-1``.

    Parameters
    ----------
    weights:
        Sparse matrix of shape (N, N), **rows = presynaptic,
        columns = postsynaptic**, holding signed synapse counts
        (positive = excitatory, negative = inhibitory).
    params:
        :class:`~openfly.sim.params.LIFParams`. Defaults to Shiu et al. 2024.
    seed:
        Seed for the Poisson drive.

    Most users should use :class:`openfly.Simulator` instead, which accepts
    neuron names and sets and returns tidy results.
    """

    def __init__(self, weights: sparse.spmatrix, params: LIFParams = SHIU_2024, seed: int | None = None):
        W = sparse.csr_matrix(weights, dtype=np.float32)
        W.sum_duplicates()
        if W.shape[0] != W.shape[1]:
            raise ValueError(f"weights must be square, got {W.shape}")
        self.params = params
        self.N = W.shape[0]
        self._indptr = W.indptr.astype(np.int64)
        self._indices = W.indices.astype(np.int32)
        # Per-synapse weights in mV. Public on purpose: plasticity rules edit it.
        self.syn_weight = (W.data * params.w_syn).astype(np.float32)
        self.syn_count = W.data.astype(np.float32)
        # Multiplier for everything a neuron sends. 0 = silenced output.
        self.out_gain = np.ones(self.N, dtype=np.float32)
        self._gain_is_one = True

        self.rng = np.random.default_rng(seed)
        self._dt = params.dt_ms
        self._em = math.exp(-self._dt / params.tau_m)
        self._eg = math.exp(-self._dt / params.tau_syn)
        if abs(params.tau_syn - params.tau_m) < 1e-9:
            # Degenerate case tau_syn == tau_m: limit of the formula below.
            self._ag = (self._dt / params.tau_m) * self._em
        else:
            self._ag = params.tau_syn / (params.tau_syn - params.tau_m) * (self._eg - self._em)
        self._D = params.delay_steps
        self.refrac_steps = np.full(self.N, params.refrac_steps, dtype=np.int32)
        # Configuration that survives reset(): silenced neurons.
        self.clamped = np.zeros(self.N, dtype=bool)
        self._any_clamped = False
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Put every neuron back to rest and clear pending spikes, drives and currents.

        Silencing and output gains are configuration and are kept.
        """
        p = self.params
        self.step_count = 0
        self.v = np.full(self.N, p.v_rest, dtype=np.float32)
        self.g = np.zeros(self.N, dtype=np.float32)
        self.current = np.zeros(self.N, dtype=np.float32)  # mV of constant drive
        self._refrac_until = np.zeros(self.N, dtype=np.int64)
        self._pending: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(self._D)]
        self._awake = np.zeros(self.N, dtype=bool)
        self._awake_idx = np.zeros(0, dtype=np.int64)
        self._drive_idx = np.zeros(0, dtype=np.int64)
        self._drive_p = np.zeros(0, dtype=np.float64)
        self._current_idx = np.zeros(0, dtype=np.int64)
        self.refrac_steps[:] = self.params.refrac_steps
        self.last_spike_count = 0

    @property
    def time_ms(self) -> float:
        return self.step_count * self._dt

    @property
    def n_awake(self) -> int:
        return int(self._awake_idx.size)

    # ------------------------------------------------------------- controls
    def set_drive(self, idx, rates_hz) -> None:
        """Poisson-drive neurons ``idx`` at ``rates_hz`` (scalar or array).

        This is the model's version of optogenetic activation: each Poisson
        event kicks the membrane far above threshold, so the neuron fires
        at roughly the requested rate. Calling it again replaces the rates
        of those neurons; a rate of 0 removes the drive.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        rates = np.broadcast_to(np.asarray(rates_hz, dtype=np.float64), idx.shape)
        full = np.zeros(self.N, dtype=np.float64)
        if self._drive_idx.size:
            full[self._drive_idx] = self._drive_p
        full[idx] = np.clip(rates * self._dt / 1000.0, 0.0, 1.0)
        self._drive_idx = np.flatnonzero(full)
        self._drive_p = full[self._drive_idx]
        # Shiu et al.: driven neurons have no refractory period.
        self.refrac_steps[:] = self.params.refrac_steps
        self.refrac_steps[self._drive_idx] = 0

    def replace_drive(self, idx, rates_hz) -> None:
        """Replace the *whole* drive with ``idx`` (unique indices) at ``rates_hz``.

        Cheaper than :meth:`set_drive` when the drive changes often
        (moving visual stimuli, the virtual fly), because it never touches
        all N neurons.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        p = np.clip(np.broadcast_to(np.asarray(rates_hz, dtype=np.float64), idx.shape) * self._dt / 1000.0, 0.0, 1.0)
        keep = p > 0
        self.refrac_steps[self._drive_idx] = self.params.refrac_steps
        self._drive_idx = idx[keep]
        self._drive_p = p[keep]
        self.refrac_steps[self._drive_idx] = 0

    def clear_drive(self) -> None:
        self._drive_idx = np.zeros(0, dtype=np.int64)
        self._drive_p = np.zeros(0, dtype=np.float64)
        self.refrac_steps[:] = self.params.refrac_steps

    @property
    def drive(self) -> tuple[np.ndarray, np.ndarray]:
        """(indices, rates in Hz) of all currently driven neurons."""
        return self._drive_idx.copy(), self._drive_p * 1000.0 / self._dt

    def set_current(self, idx, mv) -> None:
        """Add a constant depolarising (or, if negative, hyperpolarising) drive.

        Unlike :meth:`set_drive`, this does not force spikes. It shifts the
        neuron's resting point by ``mv`` millivolts, which is a gentler,
        graded kind of stimulation.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        self.current[idx] = np.asarray(mv, dtype=np.float32)
        self._current_idx = np.flatnonzero(self.current)
        self._wake(self._current_idx)

    def silence(self, idx, silent: bool = True) -> None:
        """Silence neurons: they can no longer spike or affect anyone.

        Matches the Shiu model (all outgoing synapses set to zero) and also
        clamps the neuron itself, so it looks dark in visualisations.
        """
        idx = np.asarray(idx, dtype=np.int64).ravel()
        self.clamped[idx] = silent
        self._any_clamped = bool(self.clamped.any())
        self.out_gain[idx] = 0.0 if silent else 1.0
        self._gain_is_one = bool(np.all(self.out_gain == 1.0))

    def set_out_gain(self, gain) -> None:
        """Scale everything each neuron sends (array of length N)."""
        self.out_gain = np.asarray(gain, dtype=np.float32).copy()
        self.out_gain[self.clamped] = 0.0
        self._gain_is_one = bool(np.all(self.out_gain == 1.0))

    def outgoing(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(postsynaptic indices, CSR positions) of neuron ``i``'s synapses."""
        a, b = self._indptr[i], self._indptr[i + 1]
        return self._indices[a:b], np.arange(a, b)

    # ------------------------------------------------------------ internals
    def _wake(self, idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        new = idx[~self._awake[idx]]
        if new.size:
            new = np.unique(new)
            self._awake[new] = True
            self._awake_idx = np.concatenate([self._awake_idx, new])

    def _compact(self) -> None:
        p = self.params
        a = self._awake_idx
        if a.size == 0:
            return
        resting = (np.abs(self.v[a] - p.v_rest) < _SLEEP_EPS) & (np.abs(self.g[a]) < _SLEEP_EPS)
        resting &= self.current[a] == 0
        resting &= self._refrac_until[a] <= self.step_count
        if resting.any():
            s = a[resting]
            self.v[s] = p.v_rest
            self.g[s] = 0.0
            self._awake[s] = False
            self._awake_idx = a[~resting]

    def _propagate(self, spk: np.ndarray, slot: int) -> None:
        """Queue the synaptic output of spiking neurons ``spk`` into ``slot``."""
        indptr = self._indptr
        starts = indptr[spk]
        lens = indptr[spk + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return
        if spk.size == 1:
            s = int(starts[0])
            pos = np.arange(s, s + total)
        else:
            offsets = np.cumsum(lens) - lens
            pos = np.arange(total) - np.repeat(offsets - starts, lens)
        w = self.syn_weight[pos]
        if not self._gain_is_one:
            w = w * np.repeat(self.out_gain[spk], lens)
        self._pending[slot].append((self._indices[pos], w))

    # ----------------------------------------------------------------- step
    def step(self) -> np.ndarray:
        """Advance one time step. Returns the indices of neurons that spiked."""
        p = self.params
        k = self.step_count
        a = self._awake_idx

        # 1) state update (exact integration) for awake, non-refractory neurons
        # 2) threshold. Refractory neurons are frozen below threshold, so
        #    only the neurons that were just integrated can spike.
        spk = _EMPTY
        if a.size:
            refractory = self._refrac_until[a] > k
            free = a[~refractory] if refractory.any() else a
            if free.size:
                v = self.v[free]
                g = self.g[free]
                vinf = p.v_rest + self.current[free] if self._current_idx.size else p.v_rest
                v = vinf + (v - vinf) * self._em + g * self._ag
                self.v[free] = v
                self.g[free] = g * self._eg
                spk = free[v > p.v_thresh]
                if spk.size and self._any_clamped:
                    spk = spk[~self.clamped[spk]]

        # 3) synapses: deliver spikes emitted D steps ago, then Poisson drive
        slot = k % self._D
        bucket = self._pending[slot]
        if bucket:
            if len(bucket) == 1:
                tg, w = bucket[0]
            else:
                tg = np.concatenate([b[0] for b in bucket])
                w = np.concatenate([b[1] for b in bucket])
            if p.drop_input_when_refractory:
                ok = self._refrac_until[tg] <= k
                if not ok.all():
                    tg, w = tg[ok], w[ok]
            np.add.at(self.g, tg, w)
            self._wake(tg.astype(np.int64))
            self._pending[slot] = []
        if self._drive_idx.size:
            hit = self._drive_idx[self.rng.random(self._drive_idx.size) < self._drive_p]
            if hit.size:
                if self._any_clamped:
                    hit = hit[~self.clamped[hit]]
                self.v[hit] += p.poisson_kick_mv
                self._wake(hit)

        # 4) reset spiking neurons and queue their output
        if spk.size:
            self.v[spk] = p.v_reset
            self.g[spk] = 0.0
            self._refrac_until[spk] = k + self.refrac_steps[spk]
            self._propagate(spk, slot)  # slot == (k + D) % D, delivered at step k + D

        self.step_count = k + 1
        if self.step_count % _COMPACT_EVERY == 0:
            self._compact()
        self.last_spike_count = int(spk.size)
        return spk

    def run(self, n_steps: int, callback=None) -> tuple[np.ndarray, np.ndarray]:
        """Run ``n_steps`` steps.

        Returns ``(step_indices, neuron_indices)`` of every spike. If
        ``callback`` is given it is called as ``callback(engine, spikes)``
        after every step (useful for plasticity rules or live displays).
        """
        steps: list[np.ndarray] = []
        neurons: list[np.ndarray] = []
        for _ in range(int(n_steps)):
            k = self.step_count
            spk = self.step()
            if spk.size:
                steps.append(np.full(spk.size, k, dtype=np.int64))
                neurons.append(spk)
            if callback is not None:
                callback(self, spk)
        if not steps:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate(steps), np.concatenate(neurons)
