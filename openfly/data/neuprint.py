"""Load neuPrint CSV exports, e.g. the Janelia *hemibrain* v1.2.

The hemibrain "compact connection matrix" export
(``exported-traced-adjacencies-v1.2.tar.gz``) contains:

* ``traced-neurons.csv``: ``bodyId, type, instance``
* ``traced-total-connections.csv``: ``bodyId_pre, bodyId_post, weight``
* ``traced-roi-connections.csv``: ``bodyId_pre, bodyId_post, roi, weight``

It has no neurotransmitter information, so signs are borrowed. If a
FlyWire annotation table is available, each hemibrain type inherits the
transmitter FlyWire predicts for the matching ``hemibrain_type``. That is
cross-dataset knowledge transfer: one connectome annotates another.
Anything left unknown is treated as excitatory, and the brain records how
many neurons were signed by matching.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HEMIBRAIN_URL = "https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"

_ROI_ALIASES = {"CA": "MB_CA", "PED": "MB_PED", "aL": "MB_VL", "a'L": "MB_VL", "bL": "MB_ML", "b'L": "MB_ML",
                "gL": "MB_ML", "MB": "MB_CA", "AB": "FB", "NO1": "NO", "NO2": "NO", "NO3": "NO"}


def _side_from_instance(instance: str) -> str:
    s = str(instance)
    if s.endswith("_R") or "_R_" in s or s.endswith("(R)"):
        return "right"
    if s.endswith("_L") or "_L_" in s or s.endswith("(L)"):
        return "left"
    return ""


def _roi_name(roi: str) -> str:
    r = str(roi)
    side = ""
    if r.endswith("(R)"):
        r, side = r[:-3], "_R"
    elif r.endswith("(L)"):
        r, side = r[:-3], "_L"
    r = _ROI_ALIASES.get(r, r)
    return r + side


def _member(tar: tarfile.TarFile, suffix: str):
    for m in tar.getmembers():
        if m.name.endswith(suffix):
            return tar.extractfile(m)
    raise FileNotFoundError(suffix)


def build_neuprint(archive_or_dir: str | Path, nt_lookup: dict[str, str] | None = None, meta: dict | None = None):
    """Build a Brain from a neuPrint adjacency export (``.tar.gz`` or folder)."""
    from ..brain import Brain

    src = Path(archive_or_dir)
    if src.is_dir():
        read = lambda name: open(next(src.rglob(name)), "rb")  # noqa: E731
        tar = None
    else:
        tar = tarfile.open(src, "r:gz")
        read = lambda name: _member(tar, name)  # noqa: E731
    try:
        neurons = pd.read_csv(read("traced-neurons.csv"))
        total = pd.read_csv(read("traced-total-connections.csv"))
        try:
            roi = pd.read_csv(read("traced-roi-connections.csv"))
        except (FileNotFoundError, StopIteration):
            roi = None
    finally:
        if tar is not None:
            tar.close()

    df = pd.DataFrame(
        {
            "root_id": neurons["bodyId"].astype(np.int64),
            "cell_type": neurons["type"].fillna("").astype(str),
            "instance": neurons["instance"].fillna("").astype(str),
        }
    )
    df["hemibrain_type"] = df["cell_type"]
    df["side"] = df["instance"].map(_side_from_instance)
    matched = 0
    if nt_lookup:
        df["nt"] = df["cell_type"].map(nt_lookup).fillna("")
        matched = int((df["nt"] != "").sum())
    index = pd.Index(df["root_id"].to_numpy())
    pre_nt = df["nt"].to_numpy() if "nt" in df else np.full(len(df), "")
    i = index.get_indexer(total["bodyId_pre"].to_numpy())
    j = index.get_indexer(total["bodyId_post"].to_numpy())
    ok = (i >= 0) & (j >= 0)
    i, j = i[ok], j[ok]
    w = total["weight"].to_numpy(np.float32)[ok]
    sign = np.where(np.isin(pre_nt[i], ["gaba", "glutamate"]), -1.0, 1.0).astype(np.float32)
    n = len(df)
    W = sparse.csr_matrix((w * sign, (i, j)), shape=(n, n))

    if roi is not None:
        r = roi.assign(roi=roi["roi"].map(_roi_name))
        both = pd.concat(
            [r[["bodyId_pre", "roi", "weight"]].rename(columns={"bodyId_pre": "id"}),
             r[["bodyId_post", "roi", "weight"]].rename(columns={"bodyId_post": "id"})]
        )
        best = both.groupby(["id", "roi"])["weight"].sum().reset_index().sort_values("weight", ascending=False)
        best = best.drop_duplicates("id").set_index("id")["roi"]
        df["neuropil"] = df["root_id"].map(best).fillna("")
    info = dict(meta or {"name": "neuprint"})
    info["nt_matched_neurons"] = matched
    return Brain(df, W, info)


def nt_lookup_from_flywire(annotations_path: str | Path) -> dict[str, str]:
    """hemibrain type -> most common FlyWire predicted transmitter."""
    a = pd.read_csv(annotations_path, sep="\t", low_memory=False, usecols=["hemibrain_type", "top_nt"])
    a = a.dropna()
    a = a.assign(hemibrain_type=a["hemibrain_type"].str.split(",")).explode("hemibrain_type")
    return a.groupby("hemibrain_type")["top_nt"].agg(lambda s: s.value_counts().index[0]).to_dict()
