"""Brain, NeuronSet, BrainBuilder, .flybrain files, transmitter policies."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import openfly
from openfly import Brain, BrainBuilder, transmitters
from openfly.data import read_brain, write_brain


def test_toy_brain_basics(toy):
    assert 2000 < toy.N < 3000
    assert toy.n_connections > 10_000
    assert toy.W.shape == (toy.N, toy.N)
    assert set(["root_id", "x", "y", "z", "cell_type", "super_class", "side", "nt"]) <= set(toy.neurons.columns)
    assert "toy" in repr(toy)
    assert "neurons" in toy.summary()


def test_aliases_and_lookup(toy):
    mn9 = toy["MN9"]
    assert len(mn9) == 2 and set(mn9.df.cell_type) == {"CB0701"}
    assert len(toy["giant_fiber"]) == 2
    assert len(toy["sugar_GRN"]) == 16
    assert len(toy["KC"].left) == 300
    assert len(toy["re:^PAM"]) == 6
    assert len(toy[int(mn9.ids[0])]) == 1
    with pytest.raises(KeyError, match="Did you mean"):
        toy["MN99"]


def test_set_algebra(toy):
    a, b = toy["sugar_GRN"], toy["bitter_GRN"]
    both = a | b
    assert len(both) == len(a) + len(b)
    assert len(both & a) == len(a)
    assert len(both - a) == len(b)
    assert (a | b).left == (a.left | b.left)
    with pytest.raises(ValueError):
        a | openfly.load("toy")["MN9"]


def test_select_and_query(toy):
    s = toy.select(super_class="descending", side="left")
    assert len(s) and all(s.df.side == "left")
    q = toy.query("super_class == 'motor'")
    assert len(q) == 2
    assert len(toy.select(cell_type=["CB0701", "DNp01"])) == 4


def test_inputs_outputs_connections(toy):
    inp = toy.inputs("MN9")
    assert "G2N-1" in inp.index and "GABA_bitter_IN" in inp.index
    assert inp.loc["GABA_bitter_IN", "signed"] < 0
    out = toy.outputs("sugar_GRN")
    assert out["fraction"].sum() == pytest.approx(1.0)
    con = toy.connections("G2N-1", "MN9")
    assert len(con) == 12 and (con.weight > 0).all()
    d = toy.describe(toy["MN9"].left)
    assert d["cell_type"] == "CB0701" and d["top_inputs"]


def test_flybrain_roundtrip(toy, tmp_path):
    p = write_brain(toy, tmp_path / "toy.flybrain")
    b2 = read_brain(p)
    assert b2.N == toy.N
    assert (b2.W != toy.W).nnz == 0
    pd.testing.assert_series_equal(b2.neurons["cell_type"], toy.neurons["cell_type"])
    assert openfly.data.read_meta(p)["format"] == "flybrain"
    b3 = openfly.load(p)
    assert b3.N == toy.N


def test_subbrain(toy):
    sub = toy.subbrain(toy["sugar_GRN"] | toy["G2N-1"] | toy["MN9"], name="feeding")
    assert sub.N == 24
    assert sub.n_connections > 0
    assert sub.name == "feeding"


def test_from_edges_and_schematic_layout():
    neurons = pd.DataFrame({"root_id": [10, 20, 30], "cell_type": ["a", "b", "c"], "neuropil": ["AL_L", "MB_CA_L", "GNG"]})
    edges = pd.DataFrame({"pre": [10, 20, 10], "post": [20, 30, 20], "weight": [5, -3, 2]})
    b = Brain.from_edges(neurons, edges)
    assert b.W[0, 1] == 7 and b.W[1, 2] == -3
    assert b.meta.get("layout") == "schematic"
    assert np.isfinite(b.positions).all()
    # AL is anterior of the MB calyx in the schematic too
    assert b.positions[0, 2] < b.positions[1, 2]


def test_builder():
    bb = BrainBuilder("reflex", seed=1)
    sugar = bb.add("sugar", 5, super_class="sensory")
    inter = bb.add("inter", 2, nt="gaba")
    mn = bb.add("CB0701", 1, super_class="motor")
    assert bb.connect(sugar, mn, synapses=(3, 5)) == 5
    bb.connect(inter, mn, synapses=10)
    brain = bb.build()
    assert brain.N == 8
    assert (brain.W[5:7, 7].toarray() < 0).all()  # GABA -> inhibitory
    assert len(brain["MN9"]) == 1  # aliases work on hand-made brains too


def test_transmitter_policy_flips_signs():
    bb = BrainBuilder("t")
    ln = bb.add("lLN_x", 3, cell_class="ALLN", nt="serotonin")
    pn = bb.add("PN", 2, cell_class="ALPN")
    bb.connect(ln, pn, synapses=4)
    b = bb.build()
    assert (b.W.data > 0).all()
    fixed = transmitters.apply(b, "curated")
    assert (fixed.W[0:3].data < 0).all()
    assert set(fixed.neurons.nt[:3]) == {"gaba"}
    assert fixed.meta["transmitter_overrides"] == 3
    assert len(transmitters.report(b)) == 1
    unchanged = transmitters.apply(b, "shiu2024")
    assert (unchanged.W != b.W).nnz == 0
