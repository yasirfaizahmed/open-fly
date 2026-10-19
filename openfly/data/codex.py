"""Load any connectome published through FlyWire Codex.

Codex (codex.flywire.ai) serves several connectomes in one shared table
layout:

``neurons.csv.gz``
    ``Root ID, Predicted NT type, Flow, Super Class, Class, Sub Class,
    Soma side, Primary Cell Type, ...``
``connections_princeton.csv.gz``
    ``pre_root_id, post_root_id, neuropil, syn_count, nt_type``
    (one row per neuron pair *per brain region*).

At the time of writing these include ``fafb/783`` (FlyWire),
``banc/888`` (brain and nerve cord), ``manc/1.2.1`` (male ventral nerve
cord), ``mcns/1.0`` (male central nervous system) and ``maol/1.1`` (male
optic lobe). Because the layout is shared, **new releases load with no
code changes**: ``openfly.load("codex:banc/999")``.

These tables carry no neuron coordinates, so positions come from
:func:`openfly.data.layout.schematic_layout` (region centre + spread).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

BASE_URL = "https://storage.googleapis.com/flywire-data/codex/data"

NT_NAMES = {
    "ACH": "acetylcholine",
    "GABA": "gaba",
    "GLUT": "glutamate",
    "DA": "dopamine",
    "SER": "serotonin",
    "OCT": "octopamine",
    "HIST": "histamine",
    "TYR": "tyramine",
}
INHIBITORY = {"GABA", "GLUT", "gaba", "glutamate"}

_UNIFIED = {
    "Root ID": "root_id",
    "Predicted NT type": "nt",
    "Predicted NT confidence": "nt_conf",
    "Flow": "flow",
    "Super Class": "super_class",
    "Class": "cell_class",
    "Sub Class": "cell_sub_class",
    "Primary Cell Type": "cell_type",
    "Soma side": "side",
    "Nerve": "nerve",
    "Hemilineage": "hemilineage",
    "Body Part": "body_part",
    "Function": "function",
    "Top in/out region": "top_region",
}


def urls(dataset: str, version: str) -> dict[str, list[str]]:
    root = f"{BASE_URL}/{dataset}/{version}"
    return {
        "neurons": [f"{root}/neurons.csv.gz"],
        "connections": [f"{root}/connections_princeton.csv.gz", f"{root}/connections.csv.gz"],
    }


def read_neurons(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "Root ID" in df.columns:
        df = df.rename(columns=_UNIFIED)[[v for k, v in _UNIFIED.items() if k in df.columns]]
    else:  # FAFB-style neurons.csv (root_id, group, nt_type, ...)
        df = df.rename(columns={"nt_type": "nt", "nt_type_score": "nt_conf", "group": "top_region"})
    df["root_id"] = pd.to_numeric(df["root_id"], errors="coerce")
    df = df.dropna(subset=["root_id"]).drop_duplicates("root_id")
    df["root_id"] = df["root_id"].astype(np.int64)
    if "nt" in df:
        df["nt"] = df["nt"].map(lambda v: NT_NAMES.get(str(v).upper(), "" if pd.isna(v) else str(v).lower()))
    if "side" in df:
        df["side"] = df["side"].fillna("").astype(str).str.lower()
    return df


def build_codex(neurons_path: str | Path, connections_path: str | Path, meta: dict | None = None):
    """Build a Brain from a Codex ``neurons`` + ``connections`` table pair."""
    from ..brain import Brain

    neurons = read_neurons(neurons_path)
    con = pd.read_csv(
        connections_path,
        usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"],
        dtype={"neuropil": "category", "nt_type": "category"},
    )
    # neurons that appear only in the connection table
    seen = pd.unique(np.concatenate([con["pre_root_id"].to_numpy(), con["post_root_id"].to_numpy()]))
    missing = np.setdiff1d(seen, neurons["root_id"].to_numpy())
    if missing.size:
        neurons = pd.concat([neurons, pd.DataFrame({"root_id": missing.astype(np.int64)})], ignore_index=True)
    index = pd.Index(neurons["root_id"].to_numpy())

    # sign: per-row transmitter if known, else the neuron's prediction
    neuron_nt = pd.Series(neurons.get("nt", pd.Series([""] * len(neurons))).fillna("").to_numpy(), index=index)
    row_nt = con["nt_type"].astype(object)
    missing = (row_nt.isna() | row_nt.isin(["", "nan"])).to_numpy()
    fallback = neuron_nt.reindex(con["pre_root_id"].to_numpy()).fillna("").to_numpy()
    nt = np.where(missing, fallback, row_nt.fillna("").astype(str).to_numpy())
    sign = np.where(pd.Series(nt).isin(INHIBITORY).to_numpy(), -1.0, 1.0)

    i = index.get_indexer(con["pre_root_id"].to_numpy())
    j = index.get_indexer(con["post_root_id"].to_numpy())
    w = con["syn_count"].to_numpy(np.float32)
    n = len(neurons)
    W = sparse.csr_matrix(((w * sign).astype(np.float32), (i, j)), shape=(n, n))

    # main neuropil of each neuron = region with most synapses in + out
    np_codes = con["neuropil"].cat.codes.to_numpy()
    names = np.asarray(con["neuropil"].cat.categories, dtype=object)
    per = pd.DataFrame(
        {"neuron": np.concatenate([i, j]), "np": np.concatenate([np_codes, np_codes]), "w": np.concatenate([w, w])}
    )
    per = per[per["np"] >= 0].groupby(["neuron", "np"], sort=False)["w"].sum().reset_index()
    best = per.sort_values("w", ascending=False).drop_duplicates("neuron")
    neuropil = np.full(n, "", dtype=object)
    neuropil[best["neuron"].to_numpy()] = names[best["np"].to_numpy()]
    neurons["neuropil"] = neuropil
    return Brain(neurons, W, meta or {"name": "codex"})
