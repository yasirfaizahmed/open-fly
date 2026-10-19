"""Graph analysis: pathways, distances, hubs, influence.

These tools answer questions about the wiring itself, no simulation
needed:

* :func:`strongest_path` asks "how does a sugar signal get to MN9?" and
  returns the chain of neurons with the strongest connections.
* :func:`hops` asks "how many synapses away from the eyes is every neuron?"
* :func:`traversal` gives a probabilistic "information flow" depth
  (Schlegel et al. 2021, eLife).
* :func:`influence` asks "if these neurons are active, who feels it, and
  with which sign?" using effective connectivity over multi-synapse paths.
* :func:`hubs` and :func:`type_matrix` are degree and summary statistics.

Weights are normalised as **input fractions**: a connection i->j counts
for ``|w_ij| / total input synapses of j``. A 10-synapse input matters a
lot to a small neuron and very little to a giant one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph


def input_fraction(brain, signed: bool = False) -> sparse.csr_matrix:
    """W normalised by each postsynaptic neuron's total input (cached per brain)."""
    key = "_input_fraction_signed" if signed else "_input_fraction"
    cached = brain.__dict__.get(key)
    if cached is not None:
        return cached
    W = brain.W if signed else abs(brain.W)
    tot = brain.in_synapses.astype(np.float64)
    inv = np.divide(1.0, tot, out=np.zeros_like(tot), where=tot > 0)
    F = sparse.csr_matrix(W @ sparse.diags(inv.astype(np.float32)))
    brain.__dict__[key] = F
    return F


def strongest_path(brain, source, target, min_fraction: float = 0.01, max_paths: int = 1) -> pd.DataFrame:
    """Chain of neurons from ``source`` to ``target`` with the strongest connections.

    Each hop costs ``-log(input fraction)``, so the path maximises the
    product of input fractions. Connections weaker than ``min_fraction``
    are ignored. Returns one row per neuron on the path.
    """
    src, tgt = brain[source], brain[target]
    F = input_fraction(brain).copy()
    F.data[F.data < min_fraction] = 0
    F.eliminate_zeros()
    C = F.copy()
    C.data = -np.log(C.data)
    dist, pred, origin = csgraph.dijkstra(C, directed=True, indices=src.idx, return_predecessors=True, min_only=True)
    order = tgt.idx[np.argsort(dist[tgt.idx])]
    rows = []
    for rank, t in enumerate(order[:max_paths]):
        if not np.isfinite(dist[t]):
            break
        path = [int(t)]
        while pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        for step, i in enumerate(path):
            n = brain.neurons.iloc[i]
            frac = float(F[path[step - 1], i]) if step else np.nan
            rows.append({"path": rank, "step": step, "index": i, "root_id": int(n.root_id), "cell_type": n.cell_type,
                         "super_class": n.super_class, "side": n.side, "nt": n.nt, "input_fraction": frac,
                         "synapses": float(abs(brain.W[path[step - 1], i])) if step else np.nan})
    if not rows:
        raise ValueError(f"no path from {src.label} to {tgt.label} with connections >= {min_fraction:.0%}")
    return pd.DataFrame(rows)


def hops(brain, source, min_synapses: int = 5, max_hops: int = 10) -> pd.Series:
    """Synaptic distance (number of hops) of every neuron from ``source``.

    Only connections with at least ``min_synapses`` synapses count.
    Unreachable neurons get ``NaN``.
    """
    src = brain[source]
    A = abs(brain.W).copy()
    A.data = (A.data >= min_synapses).astype(np.float32)
    A.eliminate_zeros()
    A = A.T.tocsr()  # rows = post: frontier propagation via A @ frontier
    d = np.full(brain.N, np.nan)
    d[src.idx] = 0
    frontier = np.zeros(brain.N, dtype=np.float32)
    frontier[src.idx] = 1
    seen = frontier > 0
    for h in range(1, max_hops + 1):
        nxt = (A @ frontier) > 0
        nxt &= ~seen
        if not nxt.any():
            break
        d[nxt] = h
        seen |= nxt
        frontier = nxt.astype(np.float32)
    return pd.Series(d, name=f"hops_from_{src.label}")


def traversal(brain, source, threshold: float = 0.3, n_iter: int = 10, max_steps: int = 20, seed: int = 0) -> pd.Series:
    """Probabilistic traversal depth (Schlegel et al. 2021 eLife 'information flow').

    Starting from ``source``, at every step each inactive neuron becomes
    active with probability ``min(1, input_fraction_from_active / threshold)``.
    The returned value is the mean step at which each neuron was reached
    (``NaN`` if never), a smooth measure of how deep in the hierarchy it sits.
    """
    src = brain[source]
    F = input_fraction(brain).T.tocsr()  # rows = post
    rng = np.random.default_rng(seed)
    total = np.zeros(brain.N)
    count = np.zeros(brain.N)
    for _ in range(n_iter):
        active = np.zeros(brain.N, dtype=bool)
        active[src.idx] = True
        layer = np.full(brain.N, np.nan)
        layer[src.idx] = 0
        for step in range(1, max_steps + 1):
            frac = F @ active.astype(np.float32)
            p = np.clip(frac / threshold, 0, 1)
            new = (~active) & (rng.random(brain.N) < p)
            if not new.any():
                break
            layer[new] = step
            active |= new
        ok = ~np.isnan(layer)
        total[ok] += layer[ok]
        count[ok] += 1
    out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return pd.Series(out, name=f"traversal_from_{src.label}")


def influence(brain, source, n_hops: int = 4, signed: bool = True, decay: float = 1.0) -> pd.Series:
    """Effective connectivity: summed influence of ``source`` on every neuron over 1..n_hops synapses.

    ``v_{k+1} = F^T v_k`` with F the (signed) input-fraction matrix; the result
    is ``sum_k decay^k v_k``. Positive = net excitation, negative = net inhibition.
    Fast linear proxy for "what would light up" before running a simulation.
    """
    src = brain[source]
    F = input_fraction(brain, signed=signed).T.tocsr()
    v = np.zeros(brain.N, dtype=np.float64)
    v[src.idx] = 1.0
    acc = np.zeros(brain.N)
    for k in range(1, n_hops + 1):
        v = F @ v
        acc += (decay**k) * v
    acc[src.idx] = np.nan
    return pd.Series(acc, name=f"influence_of_{src.label}")


def summarize(brain, values: pd.Series, by: str = "super_class", top: int | None = None) -> pd.DataFrame:
    """Aggregate a per-neuron Series (e.g. from :func:`influence`) by an annotation column."""
    df = pd.DataFrame({"key": brain.neurons[by].replace("", "?").to_numpy(), "v": values.to_numpy()})
    g = df.groupby("key")["v"].agg(["count", "mean", "sum", "max"]).sort_values("sum", key=np.abs, ascending=False)
    g.index.name = by
    return g.head(top) if top else g


def hubs(brain, top: int = 20, by: str = "out") -> pd.DataFrame:
    """Neurons with the most synaptic partners (``by`` = ``"out"``, ``"in"`` or ``"total"``)."""
    A = abs(brain.W)
    out_deg = np.diff(A.indptr)
    in_deg = np.diff(A.tocsc().indptr)
    df = brain.neurons[["root_id", "cell_type", "super_class", "side", "nt"]].copy()
    df["out_partners"], df["in_partners"] = out_deg, in_deg
    df["out_synapses"], df["in_synapses"] = brain.out_synapses, brain.in_synapses
    key = {"out": "out_partners", "in": "in_partners", "total": None}[by]
    if key is None:
        df["total_partners"] = out_deg + in_deg
        key = "total_partners"
    df.index.name = "index"
    return df.sort_values(key, ascending=False).head(top)


def type_matrix(brain, by: str = "super_class", value: str = "synapses", normalize: str | None = None) -> pd.DataFrame:
    """Connectivity aggregated between categories (rows = pre, columns = post).

    ``value``: ``"synapses"`` (unsigned), ``"signed"`` or ``"connections"``.
    ``normalize``: ``None``, ``"row"`` (fraction of output) or ``"col"`` (fraction of input).
    """
    coo = brain.W.tocoo()
    keys = brain.neurons[by].replace("", "?").to_numpy()
    w = {"synapses": np.abs(coo.data), "signed": coo.data, "connections": np.ones_like(coo.data)}[value]
    df = pd.DataFrame({"pre": keys[coo.row], "post": keys[coo.col], "w": w})
    M = df.pivot_table(index="pre", columns="post", values="w", aggfunc="sum", fill_value=0.0)
    if normalize == "row":
        M = M.div(M.abs().sum(axis=1), axis=0)
    elif normalize == "col":
        M = M.div(M.abs().sum(axis=0), axis=1)
    return M


def reciprocity(brain, target) -> float:
    """Fraction of ``target``'s connections (within the set) that are reciprocated."""
    ns = brain[target]
    S = abs(brain.W[ns.idx][:, ns.idx]) > 0
    n = S.nnz
    return float((S.multiply(S.T)).nnz / n) if n else 0.0
