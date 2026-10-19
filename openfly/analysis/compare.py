"""Compare two brains, e.g. a new connectome release against the old one,
FlyWire (female) against the male CNS, or a brain before and after your edits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare_brains(a, b, by: str = "super_class") -> pd.DataFrame:
    """Neuron counts and synapse totals per category in two brains, side by side."""

    def stats(brain):
        keys = brain.neurons[by].replace("", "?")
        df = pd.DataFrame({"key": keys, "out": brain.out_synapses, "in": brain.in_synapses})
        return df.groupby("key").agg(neurons=("out", "size"), out_synapses=("out", "sum"))

    sa, sb = stats(a), stats(b)
    out = sa.join(sb, how="outer", lsuffix=f"_{a.name}", rsuffix=f"_{b.name}").fillna(0)
    out[f"neurons_change"] = out[f"neurons_{b.name}"] - out[f"neurons_{a.name}"]
    out.index.name = by
    return out.sort_values(f"neurons_{a.name}", ascending=False)


def shared_types(a, b, column: str = "cell_type") -> pd.DataFrame:
    """Cell types present in both brains with their counts, plus types unique to each."""
    ca = a.neurons[column].replace("", np.nan).dropna().value_counts()
    cb = b.neurons[column].replace("", np.nan).dropna().value_counts()
    df = pd.concat([ca.rename(a.name), cb.rename(b.name)], axis=1).fillna(0).astype(int)
    df["status"] = np.where((df[a.name] > 0) & (df[b.name] > 0), "shared", np.where(df[a.name] > 0, f"only {a.name}", f"only {b.name}"))
    df.index.name = column
    return df.sort_values([a.name, b.name], ascending=False)


def type_wiring_similarity(a, b, column: str = "cell_type", min_neurons: int = 1) -> pd.DataFrame:
    """For each shared cell type, cosine similarity of its output profile (by target type) in both brains.

    A high value means that cell type talks to the same partners in both
    datasets: a quick check of stereotypy across individuals or sexes.
    """
    from .graph import type_matrix

    shared = shared_types(a, b, column)
    shared = shared[(shared["status"] == "shared")]
    shared = shared[(shared[a.name] >= min_neurons) & (shared[b.name] >= min_neurons)].index
    Ma = type_matrix(a, column)
    Mb = type_matrix(b, column)
    cols = Ma.columns.intersection(Mb.columns)
    rows = []
    for t in shared:
        if t in Ma.index and t in Mb.index:
            x = Ma.loc[t, cols].to_numpy(float)
            y = Mb.loc[t, cols].to_numpy(float)
            den = np.linalg.norm(x) * np.linalg.norm(y)
            rows.append({column: t, "cosine": float(x @ y / den) if den else np.nan})
    return pd.DataFrame(rows).set_index(column).sort_values("cosine")
