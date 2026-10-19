"""Learning rules, optimisation, screens, graph analysis and the virtual fly."""

import numpy as np
import pytest

from openfly import analysis, senses
from openfly.agent import Arena, FoodSource, Predator, VirtualFly
from openfly.learning import GainTask, OlfactoryConditioning, activation_screen, silencing_screen


@pytest.mark.parametrize("reinforcer,sign", [("reward", 1), ("punishment", -1)])
def test_conditioning_changes_valence(toy, reinforcer, sign):
    res = OlfactoryConditioning(toy, reinforcer=reinforcer).run(n_trials=1, odor_ms=500, test_ms=300)
    assert sign * res.learning_index > 0.3
    assert res.weights["weight_ratio"].min() < 0.95
    assert "learning index" in res.summary()


def test_gain_task_learns_bitter_sensitivity(toy):
    conds = {"sugar": senses.taste(toy, "sugar"),
             "laced": senses.taste(toy, "sugar") + senses.taste(toy, "bitter", 0.25)}
    task = GainTask(toy, ["G2N-1", "GABA_bitter_IN"], conds, {"MN9": "MN9"},
                    objective=lambda r: (r["sugar"]["MN9"] - r["laced"]["MN9"]) / 100)
    res = task.train(generations=10, popsize=8, lr=0.3)
    assert res.objective_after > res.objective_before + 0.2
    assert res.gain_table().set_index("group").loc["GABA_bitter_IN", "gain"] > 1.0


def test_screens_find_the_relay(toy):
    act = activation_screen(toy, "MN9", top=5)
    assert act.iloc[0]["group"] == "G2N-1"
    sil = silencing_screen(toy, senses.taste(toy, "sugar"), "MN9", model="rate", top=3)
    assert sil.iloc[0]["group"] == "G2N-1" and sil.iloc[0]["percent_of_baseline"] < 5


def test_graph_tools(toy):
    path = analysis.strongest_path(toy, "sugar_GRN", "MN9")
    assert list(path.cell_type) == ["LB3c", "G2N-1", "CB0701"]
    h = analysis.hops(toy, "photoreceptors")
    assert h[toy["LC4"].idx].max() == 2
    infl = analysis.influence(toy, "bitter_GRN")
    assert infl[toy["MN9"].idx].mean() < 0  # bitter inhibits feeding
    tr = analysis.traversal(toy, "ORN", n_iter=2)
    assert tr[toy["KC"].idx].mean() > tr[toy["ALPN"].idx].mean()
    assert len(analysis.hubs(toy, 5)) == 5
    M = analysis.type_matrix(toy, "super_class")
    assert M.loc["sensory"].sum() > 0
    assert 0 <= analysis.reciprocity(toy, "KC") <= 1


def test_compare_brains(toy):
    sub = toy.subbrain(toy["KC"] | toy["MBON"], name="mb")
    df = analysis.compare_brains(toy, sub)
    assert "neurons_change" in df.columns
    assert analysis.shared_types(toy, sub).status.isin(["shared", "only toy"]).all()


def test_virtual_fly_feeds_on_sugar_not_bitter(toy):
    sweet = Arena(items=[FoodSource(50, 50, "banana", "sugar", name="sweet")])
    fly = VirtualFly(toy, sweet, x=50, y=50, explore=False)
    fly.run(1.5)
    assert any(e["event"].startswith("feeding") for e in fly.events)
    bitter = Arena(items=[FoodSource(50, 50, "banana", "bitter", name="bitter")])
    fly = VirtualFly(toy, bitter, x=50, y=50, explore=False)
    fly.run(1.5)
    assert not any(e["event"].startswith("feeding") for e in fly.events)


def test_virtual_fly_escapes_predator(toy):
    arena = Arena(items=[Predator(t_s=0.2, bearing_deg=60)])
    fly = VirtualFly(toy, arena, x=60, y=60)
    tr = fly.run(1.2)
    assert any("escape" in e["event"] for e in fly.events)
    assert {"x", "y", "state", "drive_escape"} <= set(tr.columns)


def test_arena_fields():
    a = Arena.preset("windy_vinegar")
    src = a.items[0]
    up = a.odors_at(src.x - 20, src.y)["apple_cider_vinegar"] if a.odors_at(src.x - 20, src.y) else 0
    down = a.odors_at(src.x + 20, src.y)["apple_cider_vinegar"]
    # wind blows from 0 deg (from +x) so the plume extends toward -x... check asymmetry either way
    assert up != pytest.approx(down)
    for name in Arena.PRESETS:
        assert Arena.preset(name).items
