"""``NeuronSet``: a named group of neurons inside a :class:`~openfly.Brain`.

A NeuronSet is just an array of model indices plus a reference to the
brain it came from. It supports set algebra, so you can describe
experiments the way a biologist would say them::

    sugar = brain["sugar_GRN"]
    right_sugar = sugar.right
    not_kc = brain.all() - brain["KC"]
    both = brain["DNa02"] | brain["DNa01"]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .brain import Brain


class NeuronSet:
    """An immutable, sorted set of neurons belonging to one brain."""

    __slots__ = ("brain", "idx", "label")

    def __init__(self, brain: "Brain", idx, label: str | None = None):
        self.brain = brain
        self.idx = np.unique(np.asarray(idx, dtype=np.int64).ravel())
        if self.idx.size and (self.idx[0] < 0 or self.idx[-1] >= brain.N):
            raise IndexError("neuron index out of range for this brain")
        self.label = label

    # ------------------------------------------------------------ basics
    def __len__(self) -> int:
        return int(self.idx.size)

    def __iter__(self) -> Iterator[int]:
        return iter(self.idx.tolist())

    def __contains__(self, i) -> bool:
        return bool(np.isin(i, self.idx))

    def __bool__(self) -> bool:
        return self.idx.size > 0

    def _check(self, other: "NeuronSet") -> None:
        if not isinstance(other, NeuronSet):
            raise TypeError("can only combine NeuronSets")
        if other.brain is not self.brain:
            raise ValueError("NeuronSets come from different brains")

    def __or__(self, other: "NeuronSet") -> "NeuronSet":
        self._check(other)
        return NeuronSet(self.brain, np.union1d(self.idx, other.idx), _join(self.label, "|", other.label))

    def __and__(self, other: "NeuronSet") -> "NeuronSet":
        self._check(other)
        return NeuronSet(self.brain, np.intersect1d(self.idx, other.idx), _join(self.label, "&", other.label))

    def __sub__(self, other: "NeuronSet") -> "NeuronSet":
        self._check(other)
        return NeuronSet(self.brain, np.setdiff1d(self.idx, other.idx), _join(self.label, "-", other.label))

    def __xor__(self, other: "NeuronSet") -> "NeuronSet":
        self._check(other)
        return NeuronSet(self.brain, np.setxor1d(self.idx, other.idx), _join(self.label, "^", other.label))

    def __eq__(self, other) -> bool:
        return isinstance(other, NeuronSet) and other.brain is self.brain and np.array_equal(self.idx, other.idx)

    def __hash__(self) -> int:
        return hash((id(self.brain), self.idx.tobytes()))

    def named(self, label: str) -> "NeuronSet":
        """Same neurons, new label."""
        return NeuronSet(self.brain, self.idx, label)

    # --------------------------------------------------------- properties
    @property
    def ids(self) -> np.ndarray:
        """Dataset IDs (e.g. FlyWire root IDs) of these neurons."""
        return self.brain.neurons["root_id"].to_numpy()[self.idx]

    @property
    def df(self) -> pd.DataFrame:
        """Annotation table rows for these neurons."""
        return self.brain.neurons.iloc[self.idx]

    @property
    def positions(self) -> np.ndarray:
        return self.brain.positions[self.idx]

    @property
    def left(self) -> "NeuronSet":
        return self.where(side="left").named(_join(self.label, "", "(left)"))

    @property
    def right(self) -> "NeuronSet":
        return self.where(side="right").named(_join(self.label, "", "(right)"))

    def side(self, side: str) -> "NeuronSet":
        return self.where(side=side)

    def where(self, **filters) -> "NeuronSet":
        """Keep only neurons matching column filters (see :meth:`Brain.select`)."""
        mask = self.brain._mask(filters)
        return NeuronSet(self.brain, self.idx[mask[self.idx]], self.label)

    def sample(self, n: int, seed: int | None = 0) -> "NeuronSet":
        rng = np.random.default_rng(seed)
        n = min(n, len(self))
        return NeuronSet(self.brain, rng.choice(self.idx, n, replace=False), self.label)

    def counts(self, by: str = "cell_type") -> pd.Series:
        """How many neurons of each kind (default: per cell type)."""
        return self.df[by].fillna("?").value_counts()

    # --------------------------------------------------------- wiring
    def inputs(self, by: str = "cell_type", top: int | None = 20) -> pd.DataFrame:
        return self.brain.inputs(self, by=by, top=top)

    def outputs(self, by: str = "cell_type", top: int | None = 20) -> pd.DataFrame:
        return self.brain.outputs(self, by=by, top=top)

    # --------------------------------------------------------- display
    def __repr__(self) -> str:
        name = f" {self.label!r}" if self.label else ""
        n = len(self)
        if n == 0:
            return f"<NeuronSet{name}: empty>"
        c = self.counts("cell_type")
        parts = [f"{t} x{k}" for t, k in c.head(4).items()]
        more = f", +{len(c) - 4} types" if len(c) > 4 else ""
        return f"<NeuronSet{name}: {n} neuron{'s' if n != 1 else ''} ({', '.join(parts)}{more})>"


def _join(a: str | None, op: str, b: str | None) -> str | None:
    if a is None and b is None:
        return None
    if op == "":
        return f"{a or ''}{b or ''}"
    return f"{a or '?'} {op} {b or '?'}"
