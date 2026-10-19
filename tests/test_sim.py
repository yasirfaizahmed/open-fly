"""Simulator, results, behaviour readout, rate model on the toy brain."""

import numpy as np

import openfly
from openfly import motor, senses
from openfly.sim.rate import RateModel


def run(brain, stims, ms=400, **kw):
    sim = brain.simulator(seed=1, **kw)
    sim.add(stims)
    return sim.run(ms)


def test_sugar_drives_feeding_and_bitter_suppresses(toy):
    sugar = run(toy, senses.taste(toy, "sugar"))
    both = run(toy, senses.taste(toy, "sugar") + senses.taste(toy, "bitter"))
    f_sugar = sugar.behavior()["feeding"]["drive"]
    f_both = both.behavior()["feeding"]["drive"]
    assert f_sugar > 0.6
    assert f_both < f_sugar - 0.3


def test_looming_triggers_escape_and_flash_does_not(toy):
    loom = run(toy, senses.looming(toy, azimuth=50, duration_ms=400), 500)
    flash = run(toy, senses.flash(toy), 300)
    assert loom.behavior()["escape"]["drive"] > 0.4
    assert loom.first_spike_ms("giant_fiber") is not None
    assert flash.behavior()["escape"]["drive"] < 0.1


def test_odor_side_steers(toy):
    left = run(toy, senses.smell(toy, "banana", left=1.0, right=0.0)).behavior()["turn"]["drive"]
    right = run(toy, senses.smell(toy, "banana", left=0.0, right=1.0)).behavior()["turn"]["drive"]
    assert left < -0.2 and right > 0.2


def test_silencing_the_relay_blocks_feeding(toy):
    sim = toy.simulator(seed=1)
    sim.add(senses.taste(toy, "sugar"))
    sim.silence("G2N-1")
    res = sim.run(300)
    assert res.rate("MN9") == 0
    assert res.rate("sugar_GRN") > 100  # the sense still works


def test_result_api(toy, tmp_path):
    res = run(toy, senses.scenario(toy, "ripe_banana"), 300)
    assert res.n_spikes > 0
    top = res.top(5, exclude_driven=True)
    assert len(top) == 5 and not set(top.index) & set(res.driven.tolist())
    assert "neurons" in res.by("super_class").columns
    ps = res.psth("ORN", 20)
    assert len(ps) == 15 and ps.max() > 0
    assert len(res.active(1.0)) > 50
    other = run(toy, senses.taste(toy, "sugar"), 300)
    assert len(res.compare(other, 10)) == 10
    p = res.save(tmp_path / "res.parquet")
    back = openfly.SimResult.load(p, toy)
    assert back.n_spikes == res.n_spikes
    html = res.movie(tmp_path / "m.html")
    assert html.stat().st_size > 10_000 and "BrainViewer" in html.read_text()


def test_trials_and_parallel(toy):
    sim = toy.simulator(seed=3)
    sim.add(senses.taste(toy, "sugar"))
    a = sim.run(200, n_trials=3)
    b = sim.run(200, n_trials=3, n_jobs=2)
    assert a.n_trials == 3 and set(np.unique(a.trials)) == {0, 1, 2}
    assert np.array_equal(a.rates(), b.rates())  # same seeds -> same result in parallel
    assert a.rates_by_trial().shape == (3, toy.N)


def test_stream_and_advance(toy):
    sim = toy.simulator()
    sim.stimulate("sugar_GRN", 120)
    chunks = [c for _, c in zip(range(3), sim.stream(10))]
    assert chunks[-1][0] == 30.0
    assert sum(len(c[2]) for c in chunks) > 0


def test_motor_monitor(toy):
    mon = motor.MotorMonitor(toy)
    sim = toy.simulator()
    sim.add(senses.taste(toy, "sugar"))
    for _ in range(10):
        _, nr = sim.advance(20)
        d = mon.update(nr, 20)
    assert d["feeding"] > 0.5
    assert "feeding" in motor.describe(run(toy, senses.taste(toy, "sugar")).behavior())


def test_rate_model_tracks_lif(toy):
    st = senses.scenario(toy, "ripe_banana")
    lif = run(toy, st, 400).rates()
    rm = RateModel(toy).run(st)
    m = (lif > 0) | (rm > 0)
    assert np.corrcoef(lif[m], rm[m])[0, 1] > 0.85


def test_simulate_one_liner(toy):
    res = openfly.simulate(toy, senses.taste(toy, "sugar"), duration_ms=200, seed=0)
    assert res.rate("MN9") > 20
