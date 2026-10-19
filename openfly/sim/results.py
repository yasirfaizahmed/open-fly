"""``SimResult``: every spike of a simulation, plus ways to look at them.

    res = sim.run(1000, n_trials=5)
    res.top(15)                    # most active neurons
    res.rate("MN9")                # mean rate (Hz) of a neuron group
    res.by("super_class")          # activity summary per class
    res.psth("giant_fiber", 10)    # firing rate over time
    res.behavior()                 # motor readout (feeding, walking, escape...)
    res.compare(other_result)      # what changed between two experiments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    brain: object
    times_ms: np.ndarray
    neurons: np.ndarray
    trials: np.ndarray
    duration_ms: float
    n_trials: int = 1
    label: str = ""
    stimuli: list = field(default_factory=list)
    silenced: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    driven: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    wall_time_s: float = 0.0

    # ------------------------------------------------------------ basics
    @property
    def n_spikes(self) -> int:
        return int(self.neurons.size)

    def __repr__(self) -> str:
        n_active = int((self.counts() > 0).sum())
        return (
            f"<SimResult {self.label!r}: {self.n_trials} trial(s) x {self.duration_ms:g} ms, "
            f"{self.n_spikes:,} spikes, {n_active:,} active neurons>"
        )

    def counts(self) -> np.ndarray:
        """Total spikes per neuron (summed over trials)."""
        return np.bincount(self.neurons, minlength=self.brain.N)

    def rates(self) -> np.ndarray:
        """Mean firing rate of every neuron in Hz (averaged over trials)."""
        return self.counts() / (self.n_trials * self.duration_ms / 1000.0)

    def rates_by_trial(self) -> np.ndarray:
        """(n_trials, N) firing rates."""
        out = np.zeros((self.n_trials, self.brain.N))
        np.add.at(out, (self.trials, self.neurons), 1.0)
        return out / (self.duration_ms / 1000.0)

    def rate(self, target, window_ms: tuple[float, float] | None = None) -> float:
        """Mean rate (Hz per neuron) of a group, optionally inside a time window."""
        ns = self.brain[target]
        if len(ns) == 0:
            return 0.0
        m = np.isin(self.neurons, ns.idx)
        dur = self.duration_ms
        if window_ms is not None:
            a, b = window_ms
            t = self.times_ms
            m &= (t >= a) & (t < b)
            dur = b - a
        return float(m.sum() / (len(ns) * self.n_trials * dur / 1000.0))

    def active(self, min_rate_hz: float = 1.0):
        """NeuronSet of neurons firing at least ``min_rate_hz``."""
        from ..neurons import NeuronSet

        return NeuronSet(self.brain, np.flatnonzero(self.rates() >= min_rate_hz), f"active>={min_rate_hz:g}Hz")

    # ----------------------------------------------------------- tables
    def top(self, n: int = 20, exclude_driven: bool = False, by: str | None = None) -> pd.DataFrame:
        """The ``n`` most active neurons (or groups, with ``by="cell_type"``).

        ``exclude_driven=True`` hides the neurons you stimulated directly,
        so you see only what the brain did in response.
        """
        r = self.rates()
        df = self.brain.neurons[["root_id", "cell_type", "super_class", "cell_class", "side", "nt"]].copy()
        df["rate_hz"] = r
        if exclude_driven and self.driven.size:
            df = df.drop(index=self.driven)
        df = df[df["rate_hz"] > 0]
        if by:
            g = df.groupby(by).agg(neurons=("rate_hz", "size"), mean_rate_hz=("rate_hz", "mean"))
            return g.sort_values("mean_rate_hz", ascending=False).head(n)
        df.index.name = "index"
        return df.sort_values("rate_hz", ascending=False).head(n)

    def by(self, column: str = "super_class") -> pd.DataFrame:
        """Activity per category: how many neurons fired, and how fast."""
        r = self.rates()
        keys = self.brain.neurons[column].replace("", "?")
        df = pd.DataFrame({"key": keys, "rate": r, "active": r > 0})
        g = df.groupby("key").agg(neurons=("rate", "size"), active=("active", "sum"), mean_rate_hz=("rate", "mean"))
        g["active_rate_hz"] = df[df.active].groupby("key")["rate"].mean()
        g.index.name = column
        return g.fillna(0.0).sort_values("active", ascending=False)

    def psth(self, target, bin_ms: float = 10.0) -> pd.Series:
        """Peri-stimulus time histogram: population rate (Hz/neuron) over time."""
        ns = self.brain[target]
        m = np.isin(self.neurons, ns.idx)
        bins = np.arange(0, self.duration_ms + bin_ms, bin_ms)
        h, _ = np.histogram(self.times_ms[m], bins)
        rate = h / (max(len(ns), 1) * self.n_trials * bin_ms / 1000.0)
        return pd.Series(rate, index=pd.Index(bins[:-1], name="t_ms"), name=f"{ns.label or 'group'} (Hz)")

    def first_spike_ms(self, target) -> float | None:
        """Latency: time of the first spike of a group (mean over trials that spiked)."""
        ns = self.brain[target]
        m = np.isin(self.neurons, ns.idx)
        if not m.any():
            return None
        t = pd.Series(self.times_ms[m]).groupby(self.trials[m]).min()
        return float(t.mean())

    def spikes(self, target=None) -> pd.DataFrame:
        """Spike table (time, trial, neuron) optionally limited to a group."""
        m = slice(None) if target is None else np.isin(self.neurons, self.brain[target].idx)
        idx = self.neurons[m]
        return pd.DataFrame(
            {
                "t_ms": self.times_ms[m],
                "trial": self.trials[m],
                "index": idx,
                "root_id": self.brain.neurons["root_id"].to_numpy()[idx],
                "cell_type": self.brain.neurons["cell_type"].to_numpy()[idx],
            }
        )

    def compare(self, other: "SimResult", n: int = 20) -> pd.DataFrame:
        """Neurons whose rate changed most between ``self`` (A) and ``other`` (B)."""
        a, b = self.rates(), other.rates()
        df = self.brain.neurons[["root_id", "cell_type", "super_class", "side"]].copy()
        df["rate_A"], df["rate_B"] = a, b
        df["change"] = b - a
        df = df[(a > 0) | (b > 0)]
        df.index.name = "index"
        return df.reindex(df["change"].abs().sort_values(ascending=False).index).head(n)

    # ------------------------------------------------------ behaviour
    def behavior(self, channels=None) -> dict:
        """Read out behaviour from motor/descending neurons (see :mod:`openfly.motor`)."""
        from ..motor import read_behavior

        return read_behavior(self.brain, self.rates(), channels)

    # ------------------------------------------------------- storage
    def to_frame(self) -> pd.DataFrame:
        return self.spikes()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        df = pd.DataFrame({"t_ms": self.times_ms, "trial": self.trials, "index": self.neurons})
        df.attrs = {}
        df.to_parquet(path, index=False)
        meta = path.with_suffix(".json")
        import json

        meta.write_text(
            json.dumps(
                {"brain": self.brain.name, "duration_ms": self.duration_ms, "n_trials": self.n_trials,
                 "label": self.label, "stimuli": self.stimuli, "silenced": self.silenced.tolist()},
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path, brain) -> "SimResult":
        import json

        path = Path(path)
        df = pd.read_parquet(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(
            brain=brain,
            times_ms=df["t_ms"].to_numpy(np.float32),
            neurons=df["index"].to_numpy(np.int64),
            trials=df["trial"].to_numpy(np.int32),
            duration_ms=meta["duration_ms"],
            n_trials=meta["n_trials"],
            label=meta.get("label", ""),
            stimuli=meta.get("stimuli", []),
            silenced=np.asarray(meta.get("silenced", []), np.int64),
        )

    # ---------------------------------------------------------- viz
    def raster(self, target=None, **kw):
        from ..viz.plots import raster

        return raster(self, target, **kw)

    def movie(self, path: str | Path, **kw) -> Path:
        """Write a self-contained HTML file that replays this simulation in 3D."""
        from ..viz.export import export_movie

        return export_movie(self, path, **kw)

    def show(self, height: int = 520, **kw):
        """Replay this simulation in 3D inline in a Jupyter notebook."""
        from ..viz.export import show

        return show(self, height=height, **kw)
