"""``BrainBuilder``: design your own brain, neuron population by population.

This is the "sandbox" half of openfly. You can sketch a circuit from a
paper, wire up a hypothesis, or make a tiny brain to understand an idea,
and every openfly tool (simulator, senses, viewer, learning rules) works
on it exactly as it does on the real FlyWire connectome::

    from openfly import BrainBuilder

    b = BrainBuilder("reflex")
    sugar = b.add("sugar_GRN", n=10, super_class="sensory", cell_class="gustatory",
                  cell_sub_class="sugar", region="GNG", nt="acetylcholine")
    mn9 = b.add("CB0701", n=1, super_class="motor", region="GNG", nt="acetylcholine")
    b.connect(sugar, mn9, synapses=8)
    brain = b.build()

Signs come from the presynaptic neurotransmitter: GABA and glutamate
are inhibitory, everything else excitatory (the Shiu et al. convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

INHIBITORY_NT = {"gaba", "glutamate", "glut", "glu"}


def nt_sign(nt: str) -> int:
    return -1 if str(nt).lower() in INHIBITORY_NT else 1


@dataclass
class Population:
    """A block of neurons created by :meth:`BrainBuilder.add`."""

    name: str
    start: int
    stop: int
    attrs: dict = field(default_factory=dict)

    @property
    def idx(self) -> np.ndarray:
        return np.arange(self.start, self.stop)

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, key) -> "Population":
        """Slice a population: ``pop[:5]`` -> first five neurons."""
        r = range(self.start, self.stop)[key]
        if isinstance(r, int):
            return Population(self.name, r, r + 1, self.attrs)
        if r.step != 1:
            raise ValueError("only contiguous slices are supported")
        return Population(self.name, r.start, r.stop, self.attrs)


@dataclass
class Group:
    """Several populations treated as one (see :func:`group`)."""

    name: str
    idx: np.ndarray

    def __len__(self) -> int:
        return int(self.idx.size)


def group(*pops, name: str = "group") -> Group:
    """Merge populations (which need not be contiguous) into one connectable group."""
    return Group(name, np.concatenate([np.asarray(p.idx) for p in pops]))


class BrainBuilder:
    """Incrementally assemble neurons and connections, then :meth:`build`."""

    def __init__(self, name: str = "custom", seed: int = 0, **meta):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.meta = dict(meta)
        self._rows: list[dict] = []
        self._pre: list[np.ndarray] = []
        self._post: list[np.ndarray] = []
        self._w: list[np.ndarray] = []
        self.populations: dict[str, list[Population]] = {}
        self._next_id = 1

    @property
    def n(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------ neurons
    def add(
        self,
        cell_type: str,
        n: int = 1,
        *,
        nt: str = "acetylcholine",
        super_class: str = "central",
        cell_class: str = "",
        cell_sub_class: str = "",
        flow: str | None = None,
        side: str = "",
        region: str = "",
        center=(528.0, 220.0, 140.0),
        spread: float = 8.0,
        positions: np.ndarray | None = None,
        **extra,
    ) -> Population:
        """Add ``n`` neurons of one cell type.

        Positions are drawn around ``center`` (micrometres) with standard
        deviation ``spread``, unless explicit ``positions`` (n x 3) are given.
        Any extra keyword becomes an annotation column (e.g. ``azimuth=...``;
        per-neuron arrays of length ``n`` are allowed).
        """
        if flow is None:
            flow = {"sensory": "afferent", "motor": "efferent", "descending": "efferent"}.get(super_class, "intrinsic")
        if positions is None:
            positions = np.asarray(center, dtype=float) + self.rng.normal(0, spread, (n, 3))
        positions = np.asarray(positions, dtype=float).reshape(n, 3)
        start = self.n
        for k in range(n):
            row = {
                "root_id": self._next_id,
                "x": positions[k, 0],
                "y": positions[k, 1],
                "z": positions[k, 2],
                "flow": flow,
                "super_class": super_class,
                "cell_class": cell_class,
                "cell_sub_class": cell_sub_class,
                "cell_type": cell_type,
                "side": side,
                "nt": nt,
                "neuropil": region,
            }
            for key, val in extra.items():
                row[key] = val[k] if isinstance(val, (list, tuple, np.ndarray)) and len(val) == n else val
            self._rows.append(row)
            self._next_id += 1
        pop = Population(cell_type, start, self.n, {"nt": nt, "side": side, "region": region})
        self.populations.setdefault(cell_type, []).append(pop)
        return pop

    # -------------------------------------------------------- connections
    def _syn(self, synapses, size: int) -> np.ndarray:
        if callable(synapses):
            return np.asarray(synapses(self.rng, size), dtype=float)
        if isinstance(synapses, (tuple, list)):
            lo, hi = synapses
            return self.rng.integers(int(lo), int(hi) + 1, size).astype(float)
        return np.full(size, float(synapses))

    def _signs(self, pre_idx: np.ndarray, sign: int | None) -> np.ndarray:
        if sign is not None:
            return np.full(pre_idx.size, float(np.sign(sign) or 1))
        nts = np.array([self._rows[i]["nt"] for i in pre_idx])
        return np.array([nt_sign(t) for t in nts], dtype=float)

    def connect(
        self,
        pre: Population,
        post: Population,
        *,
        p: float = 1.0,
        n_inputs: int | None = None,
        synapses=5,
        sign: int | None = None,
        one_to_one: bool = False,
        allow_self: bool = False,
    ) -> int:
        """Wire ``pre`` onto ``post``. Returns the number of connections made.

        * ``p``: probability of each possible pair being connected, or
        * ``n_inputs``: every post neuron gets exactly this many random pre partners,
        * ``one_to_one``: neuron k of pre -> neuron k of post (same sizes).

        ``synapses`` is an int, a ``(low, high)`` range or ``f(rng, size)``.
        """
        a, b = pre.idx, post.idx
        if one_to_one:
            if len(a) != len(b):
                raise ValueError("one_to_one needs equal population sizes")
            i, j = a, b
        elif n_inputs is not None:
            k = min(n_inputs, len(a))
            i = np.concatenate([self.rng.choice(a, k, replace=False) for _ in b]) if len(b) else np.array([], int)
            j = np.repeat(b, k)
        else:
            mask = self.rng.random((len(a), len(b))) < p
            ii, jj = np.nonzero(mask)
            i, j = a[ii], b[jj]
        if not allow_self:
            keep = i != j
            i, j = i[keep], j[keep]
        w = self._syn(synapses, i.size) * self._signs(i, sign)
        self._pre.append(i)
        self._post.append(j)
        self._w.append(w)
        return int(i.size)

    def connect_by_distance(
        self, pre: Population, post: Population, *, radius: float, synapses=5, sign: int | None = None, key=("x", "y", "z")
    ) -> int:
        """Connect pairs closer than ``radius`` in the given coordinate columns.

        Handy for retinotopic (map-like) wiring, e.g. ``key=("azimuth", "elevation")``.
        """
        A = np.array([[self._rows[i][k] for k in key] for i in pre.idx], dtype=float)
        B = np.array([[self._rows[j][k] for k in key] for j in post.idx], dtype=float)
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
        ii, jj = np.nonzero(d <= radius)
        i, j = pre.idx[ii], post.idx[jj]
        w = self._syn(synapses, i.size) * self._signs(i, sign)
        self._pre.append(i)
        self._post.append(j)
        self._w.append(w)
        return int(i.size)

    # -------------------------------------------------------------- build
    def build(self, **meta):
        from ..brain import Brain

        neurons = pd.DataFrame(self._rows)
        n = len(neurons)
        if self._pre:
            i, j, w = (np.concatenate(x) for x in (self._pre, self._post, self._w))
        else:
            i = j = np.zeros(0, int)
            w = np.zeros(0)
        W = sparse.csr_matrix((w.astype(np.float32), (i, j)), shape=(n, n))
        info = {"name": self.name, "title": self.name, "kind": "built", **self.meta, **meta}
        return Brain(neurons, W, info)
