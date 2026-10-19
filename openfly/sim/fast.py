"""Optional compiled engine (``pip install numba``): same model, ~5-20x faster.

:class:`FastLIFEngine` implements exactly the update order of
:class:`~openfly.sim.engine.LIFEngine` (state update -> threshold ->
synaptic delivery -> Poisson drive -> reset), but the inner loops run
as machine code via numba. Synaptic delays use a dense ring buffer
(``delay_steps x N`` floats) plus per-slot "touched" lists, so a step
costs time proportional to the active part of the brain only.

Random numbers come from numba's generator, seeded from the engine's
NumPy generator at every call, so runs are reproducible, though not
spike-for-spike identical to the NumPy engine; statistics match (see
``tests/test_engine.py``).
"""

from __future__ import annotations

import numpy as np

from .engine import _COMPACT_EVERY, _SLEEP_EPS, LIFEngine

try:  # pragma: no cover - availability depends on the environment
    import numba

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False


if HAVE_NUMBA:

    @numba.njit(cache=True, nogil=True)
    def _kernel(n_steps, k, seed, v, g, current, has_current, refrac_until, refrac_steps, clamped, out_gain,
                indptr, indices, syn_weight, buf, touched, n_touched, tflag, awake_flag, awake_list, n_awake_arr,
                drive_idx, drive_p, v_rest, v_reset, v_th, em, eg, ag, kick, D, drop_refrac, sleep_eps,
                compact_every, spk, out_steps, out_neurons):
        np.random.seed(seed)
        n_awake = n_awake_arr[0]
        n_out = 0
        cap = out_steps.size
        done = 0
        for _ in range(n_steps):
            if n_out + n_awake + drive_idx.size > cap:
                break
            # 1-2) integrate awake, non-refractory neurons and detect spikes
            ns = 0
            for a in range(n_awake):
                i = awake_list[a]
                if refrac_until[i] > k:
                    continue
                vinf = v_rest + current[i] if has_current else v_rest
                vi = vinf + (v[i] - vinf) * em + g[i] * ag
                v[i] = vi
                g[i] = g[i] * eg
                if vi > v_th and clamped[i] == 0:
                    spk[ns] = i
                    ns += 1
            # 3) deliver what was sent D steps ago
            slot = k % D
            for t in range(n_touched[slot]):
                j = touched[slot, t]
                tflag[slot, j] = 0
                w = buf[slot, j]
                buf[slot, j] = 0.0
                if drop_refrac and refrac_until[j] > k:
                    continue
                g[j] += w
                if awake_flag[j] == 0:
                    awake_flag[j] = 1
                    awake_list[n_awake] = j
                    n_awake += 1
            n_touched[slot] = 0
            # Poisson drive
            for d in range(drive_idx.size):
                if np.random.random() < drive_p[d]:
                    j = drive_idx[d]
                    if clamped[j]:
                        continue
                    v[j] += kick
                    if awake_flag[j] == 0:
                        awake_flag[j] = 1
                        awake_list[n_awake] = j
                        n_awake += 1
            # 4) reset spikers and queue their output into this slot (arrives at k + D)
            for s in range(ns):
                i = spk[s]
                v[i] = v_reset
                g[i] = 0.0
                refrac_until[i] = k + refrac_steps[i]
                gain = out_gain[i]
                if gain != 0.0:
                    for p in range(indptr[i], indptr[i + 1]):
                        j = indices[p]
                        buf[slot, j] += syn_weight[p] * gain
                        if tflag[slot, j] == 0:
                            tflag[slot, j] = 1
                            touched[slot, n_touched[slot]] = j
                            n_touched[slot] += 1
                out_steps[n_out] = k
                out_neurons[n_out] = i
                n_out += 1
            k += 1
            done += 1
            if k % compact_every == 0:
                w_ = 0
                for a in range(n_awake):
                    i = awake_list[a]
                    if (abs(v[i] - v_rest) < sleep_eps and abs(g[i]) < sleep_eps and current[i] == 0.0
                            and refrac_until[i] <= k):
                        v[i] = v_rest
                        g[i] = 0.0
                        awake_flag[i] = 0
                    else:
                        awake_list[w_] = i
                        w_ += 1
                n_awake = w_
        n_awake_arr[0] = n_awake
        return done, n_out


class FastLIFEngine(LIFEngine):
    """Drop-in replacement for :class:`LIFEngine` backed by numba."""

    backend = "numba"

    def __init__(self, *args, **kwargs):
        if not HAVE_NUMBA:
            raise ImportError("FastLIFEngine needs numba: pip install numba")
        super().__init__(*args, **kwargs)
        self._cap = 1 << 20
        self._out_steps = np.empty(self._cap, np.int64)
        self._out_neurons = np.empty(self._cap, np.int64)
        self._spk = np.empty(self.N, np.int64)

    def reset(self) -> None:
        super().reset()
        D, N = self._D, self.N
        self._buf = np.zeros((D, N), np.float32)
        self._touched = np.zeros((D, N), np.int32)
        self._n_touched = np.zeros(D, np.int64)
        self._tflag = np.zeros((D, N), np.uint8)
        self._awake_flag = np.zeros(N, np.uint8)
        self._awake_list = np.zeros(N, np.int64)
        self._n_awake = np.zeros(1, np.int64)
        self._clamped_u8 = self.clamped.astype(np.uint8)

    @property
    def n_awake(self) -> int:
        return int(self._n_awake[0])

    def _wake(self, idx: np.ndarray) -> None:
        idx = np.unique(np.asarray(idx, np.int64))
        new = idx[self._awake_flag[idx] == 0]
        if new.size:
            n = int(self._n_awake[0])
            self._awake_flag[new] = 1
            self._awake_list[n:n + new.size] = new
            self._n_awake[0] = n + new.size

    def silence(self, idx, silent: bool = True) -> None:
        super().silence(idx, silent)
        self._clamped_u8 = self.clamped.astype(np.uint8)

    def _run_kernel(self, n: int):
        p = self.params
        steps_out, neurons_out = [], []
        remaining = int(n)
        while remaining > 0:
            seed = int(self.rng.integers(2**31 - 1))
            done, n_out = _kernel(
                remaining, self.step_count, seed, self.v, self.g, self.current, bool(self._current_idx.size),
                self._refrac_until, self.refrac_steps, self._clamped_u8, self.out_gain,
                self._indptr, self._indices, self.syn_weight, self._buf, self._touched, self._n_touched, self._tflag,
                self._awake_flag, self._awake_list, self._n_awake, self._drive_idx, self._drive_p,
                np.float32(p.v_rest), np.float32(p.v_reset), np.float32(p.v_thresh), np.float32(self._em),
                np.float32(self._eg), np.float32(self._ag), np.float32(p.poisson_kick_mv), self._D,
                bool(p.drop_input_when_refractory), _SLEEP_EPS, _COMPACT_EVERY, self._spk,
                self._out_steps, self._out_neurons,
            )
            self.step_count += done
            remaining -= done
            if n_out:
                steps_out.append(self._out_steps[:n_out].copy())
                neurons_out.append(self._out_neurons[:n_out].copy())
            if done == 0:  # output buffer too small for one step: grow it
                self._cap *= 2
                self._out_steps = np.empty(self._cap, np.int64)
                self._out_neurons = np.empty(self._cap, np.int64)
        if not steps_out:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(steps_out), np.concatenate(neurons_out)

    def step(self) -> np.ndarray:
        _, neurons = self._run_kernel(1)
        self.last_spike_count = int(neurons.size)
        return neurons

    def run(self, n_steps: int, callback=None):
        if callback is None:
            return self._run_kernel(n_steps)
        steps, neurons = [], []
        for _ in range(int(n_steps)):
            k = self.step_count
            spk = self.step()
            if spk.size:
                steps.append(np.full(spk.size, k, np.int64))
                neurons.append(spk)
            callback(self, spk)
        if not steps:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(steps), np.concatenate(neurons)


def make_engine(weights, params, seed=None, backend: str = "auto") -> LIFEngine:
    """``backend``: ``"auto"`` (numba if installed), ``"numba"`` or ``"numpy"``."""
    if backend == "numpy" or (backend == "auto" and not HAVE_NUMBA):
        return LIFEngine(weights, params, seed)
    return FastLIFEngine(weights, params, seed)
