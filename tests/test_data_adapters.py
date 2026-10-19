"""Dataset adapters on small synthetic files in the real formats (no network)."""

import gzip
import io
import os
import tarfile

import numpy as np
import pandas as pd
import pytest

from openfly.data import codex, flywire, neuprint, registry


def test_flywire_builder(tmp_path):
    ids = [720575940000000001, 720575940000000002, 720575940000000003]
    pd.DataFrame({"Completed": [True] * 3}, index=pd.Index(ids)).to_csv(tmp_path / "comp.csv")
    pd.DataFrame({
        "Presynaptic_ID": [ids[0], ids[1]], "Postsynaptic_ID": [ids[1], ids[2]],
        "Presynaptic_Index": [0, 1], "Postsynaptic_Index": [1, 2],
        "Connectivity": [12, 7], "Excitatory": [1, -1], "Excitatory x Connectivity": [12, -7],
    }).to_parquet(tmp_path / "con.parquet")
    ann = pd.DataFrame({
        "root_id": ids[:2], "pos_x": [100000, 120000], "pos_y": [50000, 60000], "pos_z": [3000, 4000],
        "soma_x": [np.nan] * 2, "soma_y": [np.nan] * 2, "soma_z": [np.nan] * 2,
        "super_class": ["sensory", "central"], "cell_class": ["gustatory", ""], "cell_sub_class": ["sugar", ""],
        "cell_type": ["LB3c", "X"], "side": ["left", "left"], "top_nt": ["acetylcholine", "gaba"], "flow": ["afferent", "intrinsic"],
    })
    ann.to_csv(tmp_path / "ann.tsv", sep="\t", index=False)
    b = flywire.build_flywire(tmp_path / "con.parquet", tmp_path / "comp.csv", tmp_path / "ann.tsv")
    assert b.N == 3
    assert b.W[0, 1] == 12 and b.W[1, 2] == -7
    assert b.neurons.x[0] == pytest.approx(400.0)  # 100000 voxels * 4 nm = 400 um
    assert np.isfinite(b.positions[2]).all()  # unannotated neuron placed near its partner
    assert len(b["sugar_GRN"]) == 1


def test_codex_adapter(tmp_path):
    n = pd.DataFrame({
        "Root ID": [1, 2, 3], "Predicted NT type": ["ACH", "GABA", "GLUT"], "Flow": ["afferent", "intrinsic", "efferent"],
        "Super Class": ["sensory", "central", "motor"], "Class": ["", "", ""], "Sub Class": ["", "", ""],
        "Soma side": ["left", "right", "left"], "Primary Cell Type": ["S", "I", "M"],
    })
    with gzip.open(tmp_path / "neurons.csv.gz", "wt") as f:
        n.to_csv(f, index=False)
    c = pd.DataFrame({"pre_root_id": [1, 1, 2, 4], "post_root_id": [2, 2, 3, 3], "neuropil": ["AL_L", "LH_L", "GNG", "GNG"],
                      "syn_count": [5, 3, 9, 2], "nt_type": ["ACH", "ACH", "", "GABA"]})
    with gzip.open(tmp_path / "connections.csv.gz", "wt") as f:
        c.to_csv(f, index=False)
    b = codex.build_codex(tmp_path / "neurons.csv.gz", tmp_path / "connections.csv.gz")
    assert b.N == 4  # neuron 4 only appears in the connection table
    assert b.W[0, 1] == 8  # summed over neuropils
    assert b.W[1, 2] == -9  # empty row nt -> neuron prediction (GABA)
    assert b.neurons.neuropil[0] == "AL_L"
    assert b.neurons.nt[2] == "glutamate"
    assert b.meta.get("layout") == "schematic"


def test_neuprint_adapter_borrows_transmitters(tmp_path):
    files = {
        "export/traced-neurons.csv": "bodyId,type,instance\n11,PN_a,PN_a_R\n12,LN_b,LN_b_R\n13,KC,KC_L\n",
        "export/traced-total-connections.csv": "bodyId_pre,bodyId_post,weight\n11,13,20\n12,11,6\n",
        "export/traced-roi-connections.csv": "bodyId_pre,bodyId_post,roi,weight\n11,13,CA(R),20\n12,11,AL(R),6\n",
    }
    arc = tmp_path / "export.tar.gz"
    with tarfile.open(arc, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    b = neuprint.build_neuprint(arc, nt_lookup={"LN_b": "gaba", "PN_a": "acetylcholine"})
    assert b.W[0, 2] == 20 and b.W[1, 0] == -6
    assert b.neurons.side.tolist() == ["right", "right", "left"]
    assert b.neurons.neuropil[0] in ("MB_CA_R", "AL_R")
    assert b.meta["nt_matched_neurons"] == 2


def test_registry():
    ds = registry.get_dataset("codex:banc/999")
    assert "banc/999/neurons.csv.gz" in ds.files["neurons"][0]
    with pytest.raises(KeyError):
        registry.get_dataset("not_a_brain")
    table = registry.datasets()
    assert {"toy", "flywire_783", "banc_888", "hemibrain_1.2"} <= set(table.index)


def test_local_override(tmp_path, monkeypatch):
    (tmp_path / "Completeness_783.csv").write_text("x")
    monkeypatch.setenv("OPENFLY_DATA_DIR", str(tmp_path))
    assert registry._local_override("Completeness_783.csv") == tmp_path / "Completeness_783.csv"
