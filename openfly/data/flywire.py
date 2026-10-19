"""Build a Brain from the public FlyWire (FAFB) release.

Ingredients (all downloaded automatically by ``openfly.load("flywire_783")``):

``Connectivity_783.parquet`` and ``Completeness_783.csv``
    From the whole-brain LIF model of Shiu et al. (2024)
    (github.com/philshiu/Drosophila_brain_model, MIT licence). One row per
    connected neuron pair with synapse count and an excitatory/inhibitory
    sign derived from predicted neurotransmitters (Eckstein et al. 2024).
    Using exactly these files keeps openfly's simulations comparable
    with the published model.
``Supplemental_file1_neuron_annotations.tsv``
    The FlyWire cell typing of Schlegel et al. (2024), extended by Matsliah
    et al. (2024), Berg et al. (2026) and Tastekin et al. (2026)
    (github.com/flyconnectome/flywire_annotations): super class, class,
    cell type, side, neurotransmitter, positions...
``neuropil_synapse_table.csv.gz`` (optional)
    Per-neuron synapse counts per brain region, from FlyWire Codex, used
    to give every neuron its main neuropil.

FlyWire coordinates are voxels of 4 x 4 x 40 nm; we store micrometres.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

VOXEL_NM = np.array([4.0, 4.0, 40.0])

_ANNOTATION_COLUMNS = {
    "flow": "flow",
    "super_class": "super_class",
    "cell_class": "cell_class",
    "cell_sub_class": "cell_sub_class",
    "cell_type": "cell_type",
    "hemibrain_type": "hemibrain_type",
    "side": "side",
    "top_nt": "nt",
    "top_nt_conf": "nt_conf",
    "nerve": "nerve",
    "ito_lee_hemilineage": "hemilineage",
    "synonyms": "synonyms",
}


def read_annotations(path: str | Path) -> pd.DataFrame:
    a = pd.read_csv(path, sep="\t", low_memory=False)
    out = pd.DataFrame({"root_id": a["root_id"].astype(np.int64)})
    pos = a[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_NM / 1000.0
    soma = a[["soma_x", "soma_y", "soma_z"]].to_numpy(float) * VOXEL_NM / 1000.0
    out["x"], out["y"], out["z"] = pos.T
    out["soma_x"], out["soma_y"], out["soma_z"] = soma.T
    for src, dst in _ANNOTATION_COLUMNS.items():
        if src in a:
            out[dst] = a[src]
    return out


def dominant_neuropil(table_path: str | Path) -> pd.DataFrame:
    """Main brain region of each neuron from Codex's neuropil synapse table."""
    t = pd.read_csv(table_path)
    inc = [c for c in t.columns if c.startswith("input synapses in ")]
    names = [c.removeprefix("input synapses in ") for c in inc]
    outc = [f"output synapses in {n}" for n in names]
    total = t[inc].to_numpy() + t[[c for c in outc if c in t]].to_numpy()
    best = np.asarray(names, dtype=object)[total.argmax(axis=1)]
    best[total.sum(axis=1) == 0] = ""
    return pd.DataFrame({"root_id": t["root_id"].astype(np.int64), "neuropil": best})


def fill_missing_positions(neurons: pd.DataFrame, W: sparse.csr_matrix) -> pd.DataFrame:
    """Place neurons without coordinates at the synapse-weighted mean of their partners."""
    xyz = neurons[["x", "y", "z"]].to_numpy(float)
    bad = np.flatnonzero(np.isnan(xyz).any(axis=1))
    if bad.size == 0:
        return neurons
    A = abs(W) + abs(W).T
    good = ~np.isnan(xyz).any(axis=1)
    centre = np.nanmedian(xyz, axis=0)
    for i in bad:
        row = A.getrow(i)
        nb, w = row.indices, row.data
        keep = good[nb]
        if keep.any():
            xyz[i] = np.average(xyz[nb[keep]], axis=0, weights=w[keep])
        else:
            xyz[i] = centre
    neurons = neurons.copy()
    neurons[["x", "y", "z"]] = xyz.astype(np.float32)
    return neurons


def build_flywire(
    connectivity: str | Path,
    completeness: str | Path,
    annotations: str | Path | None = None,
    neuropil_table: str | Path | None = None,
    meta: dict | None = None,
):
    """Assemble the FlyWire Brain from local files (see module docstring)."""
    from ..brain import Brain

    comp = pd.read_csv(completeness, index_col=0)
    ids = comp.index.to_numpy(np.int64)
    n = ids.size
    con = pd.read_parquet(
        connectivity, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    )
    W = sparse.csr_matrix(
        (
            con["Excitatory x Connectivity"].to_numpy(np.float32),
            (con["Presynaptic_Index"].to_numpy(), con["Postsynaptic_Index"].to_numpy()),
        ),
        shape=(n, n),
    )
    del con
    neurons = pd.DataFrame({"root_id": ids})
    if annotations is not None:
        ann = read_annotations(annotations)
        neurons = neurons.merge(ann, on="root_id", how="left")
    if neuropil_table is not None:
        neurons = neurons.merge(dominant_neuropil(neuropil_table), on="root_id", how="left")
    for col in ("x", "y", "z"):
        if col not in neurons:
            neurons[col] = np.nan
    if not neurons[["x", "y", "z"]].isna().all(axis=None):
        neurons = fill_missing_positions(neurons, W)
    return Brain(neurons, W, meta or {"name": "flywire"})
