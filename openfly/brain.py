"""The :class:`Brain`: one connectome held in memory.

A Brain is two things:

* ``brain.neurons``: a table (pandas DataFrame) with one row per neuron,
  with its ID, 3D position (micrometres), cell type, neurotransmitter and
  other annotations. The row number is the neuron's *model index*.
* ``brain.W``: a sparse N x N matrix, **rows = presynaptic,
  columns = postsynaptic**, holding *signed synapse counts*
  (+ excitatory, - inhibitory).

That is all a simulator needs, and every connectome (FlyWire, BANC,
MANC, hemibrain, your own toy circuit...) can be expressed this way.
See ``docs/04-brain-api.md``.
"""

from __future__ import annotations

import difflib
import re
from functools import cached_property
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from . import knowledge
from .neurons import NeuronSet

#: Columns every Brain's neuron table has (missing ones are filled in).
SCHEMA = {
    "root_id": "int64",
    "x": "float32",
    "y": "float32",
    "z": "float32",
    "flow": "str",
    "super_class": "str",
    "cell_class": "str",
    "cell_sub_class": "str",
    "cell_type": "str",
    "side": "str",
    "nt": "str",
    "neuropil": "str",
}

TEXT_COLUMNS = [c for c, t in SCHEMA.items() if t == "str"]


def normalize_neurons(neurons: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``neurons`` that follows :data:`SCHEMA`."""
    df = neurons.reset_index(drop=True).copy()
    if "root_id" not in df:
        df["root_id"] = np.arange(len(df), dtype=np.int64)
    df["root_id"] = df["root_id"].astype(np.int64)
    for col, kind in SCHEMA.items():
        if col not in df:
            df[col] = np.nan if kind == "float32" else ""
        if kind == "str":
            df[col] = df[col].astype(object).where(df[col].notna(), "").astype(str)
            df[col] = df[col].replace({"nan": "", "None": ""})
        elif kind == "float32":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    return df


class Brain:
    """A connectome: neuron table + signed synapse-count matrix.

    Build one with :func:`openfly.load` (downloads real data), with
    :class:`openfly.BrainBuilder` (design your own circuit), or with
    :meth:`Brain.from_edges` (any table of connections).
    """

    def __init__(self, neurons: pd.DataFrame, W: sparse.spmatrix, meta: dict | None = None):
        self.neurons = normalize_neurons(neurons)
        W = sparse.csr_matrix(W, dtype=np.float32)
        W.sum_duplicates()
        W.eliminate_zeros()
        n = len(self.neurons)
        if W.shape != (n, n):
            raise ValueError(f"W has shape {W.shape}, expected ({n}, {n})")
        self.W = W
        self.meta = dict(meta or {})
        self.meta.setdefault("name", "brain")
        self._id_index = pd.Index(self.neurons["root_id"].to_numpy())
        if not self._id_index.is_unique:
            raise ValueError("root_id values must be unique")
        if self.neurons[["x", "y", "z"]].isna().all(axis=None):
            from .data.layout import schematic_layout

            xyz = schematic_layout(self.neurons, self.W)
            self.neurons[["x", "y", "z"]] = xyz
            self.meta["layout"] = "schematic"

    # ------------------------------------------------------------ factory
    @classmethod
    def from_edges(
        cls,
        neurons: pd.DataFrame,
        edges: pd.DataFrame,
        pre: str = "pre",
        post: str = "post",
        weight: str = "weight",
        by: str = "id",
        meta: dict | None = None,
    ) -> "Brain":
        """Build a Brain from a neuron table and an edge list.

        ``edges[pre]``/``edges[post]`` hold neuron IDs (``by="id"``, matched
        against ``neurons["root_id"]``) or model indices (``by="index"``).
        ``edges[weight]`` must already be signed (negative = inhibitory).
        Duplicate pairs are summed.
        """
        neurons = normalize_neurons(neurons)
        n = len(neurons)
        if by == "id":
            idx = pd.Index(neurons["root_id"].to_numpy())
            i = idx.get_indexer(edges[pre].to_numpy())
            j = idx.get_indexer(edges[post].to_numpy())
            ok = (i >= 0) & (j >= 0)
            i, j = i[ok], j[ok]
            w = edges[weight].to_numpy()[ok]
        elif by == "index":
            i, j, w = edges[pre].to_numpy(), edges[post].to_numpy(), edges[weight].to_numpy()
        else:
            raise ValueError("by must be 'id' or 'index'")
        W = sparse.csr_matrix((w.astype(np.float32), (i, j)), shape=(n, n))
        return cls(neurons, W, meta)

    @classmethod
    def load(cls, path: str | Path) -> "Brain":
        """Load a ``.flybrain`` file."""
        from .data.brainfile import read_brain

        return read_brain(path)

    def save(self, path: str | Path) -> Path:
        """Save as a portable ``.flybrain`` file."""
        from .data.brainfile import write_brain

        return write_brain(self, path)

    # --------------------------------------------------------- basic info
    @property
    def N(self) -> int:
        return len(self.neurons)

    @property
    def name(self) -> str:
        return self.meta.get("name", "brain")

    @property
    def n_connections(self) -> int:
        return int(self.W.nnz)

    @property
    def n_synapses(self) -> int:
        return int(np.abs(self.W.data).sum())

    @cached_property
    def positions(self) -> np.ndarray:
        """(N, 3) neuron positions in micrometres."""
        return self.neurons[["x", "y", "z"]].to_numpy(np.float32)

    @cached_property
    def W_csc(self) -> sparse.csc_matrix:
        """Column-major copy of W, for fast 'who talks to these neurons' queries."""
        return self.W.tocsc()

    @cached_property
    def in_synapses(self) -> np.ndarray:
        """Total input synapses (unsigned) of every neuron."""
        return np.asarray(abs(self.W).sum(axis=0)).ravel()

    @cached_property
    def out_synapses(self) -> np.ndarray:
        return np.asarray(abs(self.W).sum(axis=1)).ravel()

    def summary(self) -> str:
        n = self.neurons
        lines = [
            f"{self.meta.get('title', self.name)}",
            f"  neurons:     {self.N:,}",
            f"  connections: {self.n_connections:,} (neuron pairs)",
            f"  synapses:    {self.n_synapses:,}",
        ]
        sc = n["super_class"].replace("", "?").value_counts()
        if len(sc):
            lines.append("  super classes: " + ", ".join(f"{k} {v:,}" for k, v in sc.head(8).items()))
        nt = n["nt"].replace("", "?").value_counts()
        if len(nt):
            lines.append("  transmitters:  " + ", ".join(f"{k} {v:,}" for k, v in nt.head(7).items()))
        if self.meta.get("citation"):
            lines.append(f"  cite: {self.meta['citation']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<Brain {self.name!r}: {self.N:,} neurons, {self.n_connections:,} connections>"

    # ---------------------------------------------------------- selection
    def all(self) -> NeuronSet:
        return NeuronSet(self, np.arange(self.N), "all")

    def at(self, indices: Iterable[int], label: str | None = None) -> NeuronSet:
        """NeuronSet from model indices (row numbers)."""
        return NeuronSet(self, list(indices), label)

    def index_of(self, ids, missing: str = "raise") -> np.ndarray:
        """Model indices for dataset IDs (e.g. FlyWire root IDs)."""
        ids = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        idx = self._id_index.get_indexer(ids)
        if (idx < 0).any():
            if missing == "raise":
                bad = ids[idx < 0][:5]
                raise KeyError(f"IDs not in {self.name}: {bad.tolist()}")
            idx = idx[idx >= 0]
        return idx.astype(np.int64)

    def from_ids(self, ids, label: str | None = None, missing: str = "raise") -> NeuronSet:
        return NeuronSet(self, self.index_of(ids, missing), label)

    def _column_mask(self, col: str, value) -> np.ndarray:
        if col in ("index", "idx"):
            m = np.zeros(self.N, dtype=bool)
            m[np.asarray(value, dtype=np.int64)] = True
            return m
        if col not in self.neurons:
            raise KeyError(f"unknown column {col!r}; columns: {list(self.neurons.columns)}")
        s = self.neurons[col]
        if callable(value):
            return np.asarray(value(s), dtype=bool)
        if isinstance(value, str) and value.startswith("re:"):
            return s.astype(str).str.contains(value[3:], regex=True, na=False).to_numpy()
        if isinstance(value, (list, tuple, set, frozenset, np.ndarray, pd.Index)):
            return s.isin(list(value)).to_numpy()
        return (s == value).to_numpy()

    def _mask(self, filters: dict) -> np.ndarray:
        m = np.ones(self.N, dtype=bool)
        for col, value in filters.items():
            m &= self._column_mask(col, value)
        return m

    def select(self, label: str | None = None, **filters) -> NeuronSet:
        """Select neurons by annotation columns.

        Examples::

            brain.select(cell_type="ORN_DM1", side="right")
            brain.select(super_class=["descending", "motor"])
            brain.select(cell_type="re:^PAM")      # regular expression
            brain.select(nt="gaba", x=lambda s: s > 500)
        """
        if label is None:
            label = ",".join(f"{k}={v}" for k, v in filters.items())
        return NeuronSet(self, np.flatnonzero(self._mask(filters)), label)

    def query(self, expr: str) -> NeuronSet:
        """Select with a pandas query string, e.g. ``"super_class == 'descending' and x > 500"``."""
        df = self.neurons.query(expr)
        return NeuronSet(self, df.index.to_numpy(), expr)

    _LOOKUP_COLUMNS = ("cell_type", "hemibrain_type", "cell_sub_class", "cell_class", "super_class")

    def resolve(self, name: str) -> NeuronSet:
        """Turn a human name into neurons.

        Tried in order: curated alias (``knowledge.py``), cell type,
        hemibrain type, cell sub-class, cell class, super class,
        ``re:`` regular expression on cell type, numeric ID.
        """
        key = str(name).strip()
        alias = knowledge.get_alias(key)
        if alias is not None:
            ns = self.select(label=alias.name, **alias.selector)
            if len(ns):
                return ns
        for col in self._LOOKUP_COLUMNS:
            if col in self.neurons:
                m = (self.neurons[col] == key).to_numpy()
                if m.any():
                    return NeuronSet(self, np.flatnonzero(m), key)
        if key.startswith("re:"):
            return self.select(label=key, cell_type=key)
        if re.fullmatch(r"\d+", key):
            return self.from_ids([int(key)], label=key)
        raise KeyError(self._suggest(key))

    def _suggest(self, key: str) -> str:
        pool = set(knowledge.ALIASES) | set(self.neurons["cell_type"].unique())
        close = difflib.get_close_matches(key, [p for p in pool if p], n=6, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return f"No neurons called {key!r} in {self.name}.{hint}"

    def __getitem__(self, key) -> NeuronSet:
        if isinstance(key, NeuronSet):
            return key
        if isinstance(key, (int, np.integer)):
            return self.from_ids([int(key)], label=str(key))
        if isinstance(key, str):
            return self.resolve(key)
        if isinstance(key, dict):
            return self.select(**key)
        if isinstance(key, (list, tuple, set)):
            out = None
            for k in key:
                ns = self[k]
                out = ns if out is None else out | ns
            return out if out is not None else NeuronSet(self, [], "empty")
        raise TypeError(f"cannot select neurons with {type(key).__name__}")

    def search(self, text: str, limit: int = 25) -> pd.DataFrame:
        """Find aliases and cell types whose name contains ``text``."""
        t = text.lower()
        rows = []
        for a in knowledge.ALIASES.values():
            if t in a.name.lower() or t in a.description.lower():
                n = len(self.select(**a.selector))
                if n:
                    rows.append({"name": a.name, "kind": "alias", "n": n, "description": a.description})
        ct = self.neurons["cell_type"]
        counts = ct[ct.str.lower().str.contains(re.escape(t), na=False)].value_counts()
        for name, n in counts.items():
            if name:
                rows.append({"name": name, "kind": "cell_type", "n": int(n), "description": ""})
        df = pd.DataFrame(rows, columns=["name", "kind", "n", "description"])
        return df.head(limit)

    # ---------------------------------------------------------- wiring
    def _partners(self, target: NeuronSet, direction: str, by: str | None, top: int | None) -> pd.DataFrame:
        target = self[target]
        if direction == "in":
            sub = self.W_csc[:, target.idx].tocoo()
            other = sub.row
        else:
            sub = self.W[target.idx, :].tocoo()
            other = sub.col
        w = sub.data
        total = np.abs(w).sum()
        df = pd.DataFrame({"partner": other, "signed": w, "synapses": np.abs(w)})
        if by is None or by == "neuron":
            g = df.groupby("partner").agg(synapses=("synapses", "sum"), signed=("signed", "sum"))
            meta = self.neurons.iloc[g.index][["root_id", "cell_type", "super_class", "side", "nt"]]
            g = pd.concat([meta.set_index(g.index), g], axis=1)
            g.index.name = "index"
        else:
            df["key"] = self.neurons[by].to_numpy()[other]
            g = df.groupby("key").agg(
                synapses=("synapses", "sum"), signed=("signed", "sum"), neurons=("partner", "nunique")
            )
            g.index.name = by
        g["fraction"] = g["synapses"] / total if total else 0.0
        g = g.sort_values("synapses", ascending=False)
        return g.head(top) if top else g

    def inputs(self, target, by: str | None = "cell_type", top: int | None = 20) -> pd.DataFrame:
        """Who sends synapses to ``target``? Grouped by ``by`` (``None`` = per neuron)."""
        return self._partners(self[target], "in", by, top)

    def outputs(self, target, by: str | None = "cell_type", top: int | None = 20) -> pd.DataFrame:
        """Whom does ``target`` send synapses to?"""
        return self._partners(self[target], "out", by, top)

    def connections(self, source, target) -> pd.DataFrame:
        """Every connection from ``source`` neurons onto ``target`` neurons."""
        s, t = self[source], self[target]
        sub = self.W[s.idx, :][:, t.idx].tocoo()
        pre, post = s.idx[sub.row], t.idx[sub.col]
        ct = self.neurons["cell_type"].to_numpy()
        return pd.DataFrame(
            {"pre": pre, "post": post, "pre_type": ct[pre], "post_type": ct[post], "weight": sub.data}
        ).sort_values("weight", key=np.abs, ascending=False, ignore_index=True)

    def describe(self, neuron) -> dict:
        """Everything about one neuron: annotations plus top partners."""
        ns = self[neuron]
        if len(ns) != 1:
            raise ValueError(f"describe() needs exactly one neuron, got {len(ns)}")
        i = int(ns.idx[0])
        row = self.neurons.iloc[i].to_dict()
        row["index"] = i
        row["n_inputs"] = int(self.in_synapses[i])
        row["n_outputs"] = int(self.out_synapses[i])
        row["top_inputs"] = self.inputs(ns, by=None, top=8).reset_index().to_dict("records")
        row["top_outputs"] = self.outputs(ns, by=None, top=8).reset_index().to_dict("records")
        return row

    def subbrain(self, neurons, name: str | None = None) -> "Brain":
        """A new Brain containing only ``neurons`` and the wiring among them."""
        ns = self[neurons]
        W = self.W[ns.idx, :][:, ns.idx]
        meta = dict(self.meta, name=name or f"{self.name}[{ns.label or 'subset'}]", parent=self.name)
        return Brain(self.neurons.iloc[ns.idx], W, meta)

    # ------------------------------------------------------------ shortcuts
    def simulator(self, **kwargs):
        """Create a :class:`openfly.Simulator` for this brain."""
        from .sim.simulator import Simulator

        return Simulator(self, **kwargs)
