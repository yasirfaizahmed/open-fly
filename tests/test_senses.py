"""Senses produce the right neurons, rates and timing."""

import numpy as np
import pytest

from openfly import senses
from openfly.senses import Stimulus, StimulusSet


def test_smell_targets_orns_with_laterality(toy):
    st = senses.smell(toy, "banana", left=1.0, right=0.2)
    types = set(toy.neurons.cell_type.iloc[st.targets])
    assert types <= {f"ORN_{g}" for g in ("DM1", "DM2", "DM3", "DM4", "VA2", "VM2")}
    side = toy.neurons.side.to_numpy()[st.targets]
    assert st.rates[side == "left"].mean() > st.rates[side == "right"].mean()
    with pytest.raises(KeyError):
        senses.smell(toy, "durian")


def test_high_concentration_recruits_more(toy):
    lo = senses.smell(toy, "apple_cider_vinegar", 0.5)
    hi = senses.smell(toy, "apple_cider_vinegar", 2.0)
    assert hi.rates.sum() > lo.rates.sum()


def test_custom_odor(toy):
    senses.custom_odor("test_odor", {"DA2": 1.0})
    st = senses.smell(toy, "test_odor")
    assert set(toy.neurons.cell_type.iloc[st.targets]) == {"ORN_DA2"}


def test_taste(toy):
    st = senses.taste(toy, "sugar")
    assert len(st) == 16 and np.allclose(st.rates, 150)
    left = senses.taste(toy, "bitter", side="left")
    assert set(toy.neurons.side.iloc[left.targets]) == {"left"}


def test_visual_field_and_spot(toy):
    vf = senses.VisualField.of(toy)
    side = vf.side
    assert (vf.azimuth[side == "right"] > -20).all() and (vf.azimuth[side == "left"] < 20).all()
    st = senses.spot(toy, azimuth=60, elevation=0, radius_deg=20)
    assert len(st) and set(toy.neurons.side.iloc[st.targets]) == {"right"}


def test_looming_is_dynamic_and_accelerates(toy):
    loom = senses.looming(toy, azimuth=50, duration_ms=400)
    assert isinstance(loom, StimulusSet) and all(s.stop_ms == 400 for s in loom)
    retina = [s for s in loom if "retina" in s.name][0]
    early = (retina.rates_at(100) > 0).sum()
    late = (retina.rates_at(390) > 0).sum()
    assert late > early
    lc4 = [s for s in loom if "LC4" in s.name][0]
    # LC4 follows expansion speed: strongest just before the object fills the view
    assert lc4.rates_at(360).mean() > lc4.rates_at(100).mean()


def test_image_stimulus(toy):
    img = np.zeros((90, 180))
    img[:, 110:] = 1.0  # bright from 40 deg right onwards (beyond the binocular overlap)
    st = senses.image(toy, img)
    assert set(toy.neurons.side.iloc[st.targets]) == {"right"}


def test_pulse_song_rhythm(toy):
    st = senses.hear(toy, "pulse_song", ipi_ms=35, pulse_ms=8)
    assert st.rates_at(2).mean() > st.rates_at(20).mean()
    assert st.rates_at(37).mean() == pytest.approx(st.rates_at(2).mean())


def test_all_scenarios_build(toy):
    for name in senses.list_scenarios():
        stims = senses.scenario(toy, name, start_ms=100)
        assert len(stims) >= 1, name
        assert all(s.start_ms >= 100 for s in stims)


def test_stimulus_algebra():
    a = Stimulus(np.array([1, 2]), 100.0, name="a")
    b = Stimulus(np.array([3]), 50.0, name="b")
    s = a + b
    assert isinstance(s, StimulusSet) and len(s) == 2
    assert s.between(10, 20)[0].start_ms == 10
    assert a.scaled(0.5).rates[0] == 50
    assert not a.between(10, 20).active(5)
