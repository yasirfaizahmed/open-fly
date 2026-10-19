"""The LIF engine: single-neuron physics, Brian2-compatible details, backends."""

import numpy as np
import pytest
from scipy import sparse

from openfly.sim import HAVE_NUMBA, SHIU_2024, LIFEngine, make_engine
from openfly.sim.fast import FastLIFEngine

BACKENDS = ["numpy"] + (["numba"] if HAVE_NUMBA else [])


def two_neurons(nsyn):
    return sparse.csr_matrix(([float(nsyn)], ([0], [1])), shape=(2, 2))


@pytest.mark.parametrize("backend", BACKENDS)
def test_poisson_drive_sets_rate(backend):
    eng = make_engine(two_neurons(0), SHIU_2024, seed=1, backend=backend)
    eng.set_drive([0], 100.0)
    _, nr = eng.run(50_000)  # 5 s
    rate = np.sum(nr == 0) / 5.0
    assert 85 < rate < 105  # ~100 Hz minus events lost during reset


@pytest.mark.parametrize("backend", BACKENDS)
def test_more_synapses_more_output(backend):
    out = []
    for n in (10, 30, 100):
        eng = make_engine(two_neurons(n), SHIU_2024, seed=2, backend=backend)
        eng.set_drive([0], 200.0)
        _, nr = eng.run(50_000)
        out.append(np.sum(nr == 1) / 5.0)
    assert out[0] < out[1] < out[2]
    assert out[0] == 0  # 10 synapses (2.75 mV) can't reach threshold alone


def test_exact_integration_matches_closed_form():
    eng = LIFEngine(two_neurons(1), SHIU_2024)
    eng.g[1] = 2.0
    eng._wake(np.array([1]))
    for _ in range(100):
        eng.step()
    p = SHIU_2024
    t = 10.0
    A = 2.0 * p.tau_syn / (p.tau_syn - p.tau_m)
    expected_u = A * (np.exp(-t / p.tau_syn) - np.exp(-t / p.tau_m))
    assert eng.v[1] - p.v_rest == pytest.approx(expected_u, rel=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_input_during_refractory_is_dropped(backend):
    """Brian2 behaviour of the Shiu model (verified against Brian2 2.10)."""
    W = two_neurons(300)
    eng = make_engine(W, SHIU_2024, seed=0, backend=backend)
    spikes1 = []
    for k in range(120):
        if k in (0, 30):  # neuron 0 fires at steps 0 and 30
            eng.v[0] = 0.0
            eng._wake(np.array([0]))
        s = eng.step()
        if 1 in s:
            spikes1.append(k)
    # first input (arrives at step ~18) -> spike ~23 steps later; second input arrives while refractory
    assert len(spikes1) == 1
    assert 38 <= spikes1[0] <= 44


def test_keeping_refractory_input_increases_firing():
    p = SHIU_2024.with_(drop_input_when_refractory=False)
    eng_keep = LIFEngine(two_neurons(300), p, seed=0)
    eng_drop = LIFEngine(two_neurons(300), SHIU_2024, seed=0)
    for e in (eng_keep, eng_drop):
        e.set_drive([0], 200.0)
    _, a = eng_keep.run(20_000)
    _, b = eng_drop.run(20_000)
    assert np.sum(a == 1) > np.sum(b == 1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_silenced_neuron_is_silent_and_harmless(backend):
    eng = make_engine(two_neurons(300), SHIU_2024, seed=0, backend=backend)
    eng.set_drive([0], 200.0)
    eng.silence([0])
    _, nr = eng.run(10_000)
    assert nr.size == 0


def test_seed_reproducible():
    runs = []
    for _ in range(2):
        eng = LIFEngine(two_neurons(100), SHIU_2024, seed=42)
        eng.set_drive([0], 150.0)
        runs.append(eng.run(5000))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


def test_current_injection_graded():
    eng = LIFEngine(two_neurons(0), SHIU_2024)
    eng.set_current([1], 5.0)  # below the 7 mV threshold distance
    eng.run(5000)
    assert eng.v[1] - SHIU_2024.v_rest == pytest.approx(5.0, abs=0.05)
    _, nr = eng.run(1)
    assert nr.size == 0
    eng.set_current([1], 10.0)
    _, nr = eng.run(5000)
    assert np.sum(nr == 1) > 10


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_numba_matches_numpy_statistically(toy):
    from openfly import senses

    rates = {}
    for be in ("numpy", "numba"):
        sim = toy.simulator(seed=5, backend=be)
        sim.add(senses.scenario(toy, "ripe_banana"))
        rates[be] = sim.run(400, n_trials=2).rates()
    a, b = rates["numpy"], rates["numba"]
    m = (a > 0) | (b > 0)
    assert np.corrcoef(a[m], b[m])[0, 1] > 0.95
    assert isinstance(toy.simulator(backend="numba").engine, FastLIFEngine)
