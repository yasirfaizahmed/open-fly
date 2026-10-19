"""Checks on the real FlyWire connectome. Slow and needs the ~140 MB download.

Run with:  OPENFLY_TEST_FLYWIRE=1 pytest tests/test_flywire.py
"""

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.flywire,
    pytest.mark.skipif(os.environ.get("OPENFLY_TEST_FLYWIRE") != "1", reason="set OPENFLY_TEST_FLYWIRE=1"),
]


@pytest.fixture(scope="module")
def fw():
    import openfly

    return openfly.load("flywire_783", transmitters="curated")


def test_size_and_annotations(fw):
    assert fw.N == 138_639
    assert fw.n_connections == 15_091_983
    assert len(fw["MN9"]) == 2
    assert len(fw["giant_fiber"]) == 2
    assert 50 < len(fw["sugar_GRN"]) < 70


def test_sugar_extends_proboscis(fw):
    from openfly import senses

    sim = fw.simulator(seed=0)
    sim.add(senses.taste(fw, "sugar"))
    res = sim.run(500)
    assert res.rate("MN9") > 20


def test_looming_drives_giant_fiber(fw):
    from openfly import senses

    sim = fw.simulator(seed=0)
    sim.add(senses.looming(fw, azimuth=50, duration_ms=400))
    res = sim.run(500)
    assert res.rate(fw["giant_fiber"].right) > 20


def test_curated_policy_prevents_antennal_lobe_runaway(fw):
    from openfly import senses

    sim = fw.simulator(seed=0)
    sim.add(senses.smell(fw, "banana"))
    res = sim.run(300)
    assert res.rate(fw.select(cell_class="ALLN")) < 50
    kc_active = (res.rates()[fw["KC"].idx] > 0).mean()
    assert 0.005 < kc_active < 0.2  # sparse Kenyon cell code


def test_strongest_path(fw):
    from openfly.analysis import strongest_path

    p = strongest_path(fw, "sugar_GRN", "MN9")
    assert p.iloc[-1]["cell_type"] == "CB0701"
    assert np.isfinite(p.input_fraction.iloc[1:]).all()
