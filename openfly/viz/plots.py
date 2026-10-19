"""Static figures with matplotlib (optional dependency: ``pip install matplotlib``).

* :func:`raster`: spike raster, neurons grouped and coloured by class
* :func:`rates_bar`: the most active cell types
* :func:`psth`: firing rate over time for a few groups
* :func:`brain_map`: 2D projection of the brain coloured by any per-neuron value
* :func:`plot_trajectory`: the virtual fly's path through its arena
* :func:`conditioning_plot`: before/after valence of a conditioning experiment
"""

from __future__ import annotations

import numpy as np

CLASS_COLORS = {
    "optic": "#3fa7ff", "central": "#ff7ab6", "sensory": "#ffd166", "visual_projection": "#06d6a0",
    "ascending": "#c77dff", "descending": "#ff5d5d", "sensory_ascending": "#f4a261", "visual_centrifugal": "#4cc9f0",
    "motor": "#ff9f1c", "endocrine": "#b5e48c",
}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install matplotlib") from exc
    return plt


def _ax(ax, figsize=(8, 4)):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def raster(result, target=None, trial: int = 0, max_neurons: int = 300, by: str = "super_class", ax=None):
    """Spike raster of one trial. Rows = the most active neurons (of ``target``), grouped by ``by``."""
    ax = _ax(ax, (9, 5))
    b = result.brain
    m = result.trials == trial
    t, n = result.times_ms[m], result.neurons[m]
    if target is not None:
        keep = np.isin(n, b[target].idx)
        t, n = t[keep], n[keep]
    if n.size == 0:
        ax.set_title("no spikes")
        return ax
    counts = np.bincount(n, minlength=b.N)
    top = np.argsort(-counts)[: min(max_neurons, int((counts > 0).sum()))]
    groups = b.neurons[by].to_numpy()[top]
    order = top[np.lexsort((-counts[top], groups))]
    row = np.full(b.N, -1)
    row[order] = np.arange(order.size)
    keep = row[n] >= 0
    cols = [CLASS_COLORS.get(g, "#888888") for g in b.neurons[by].to_numpy()[n[keep]]]
    ax.scatter(t[keep], row[n[keep]], s=2, c=cols, marker="|", linewidths=0.8)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel(f"neurons (top {order.size}, grouped by {by})")
    ax.set_xlim(0, result.duration_ms)
    ax.set_title(result.label)
    return ax


def rates_bar(result, n: int = 15, by: str = "cell_type", exclude_driven: bool = True, ax=None):
    """Horizontal bars of the most active groups."""
    ax = _ax(ax, (6, 0.32 * n + 1))
    df = result.top(n, exclude_driven=exclude_driven, by=by)
    df = df.iloc[::-1]
    ax.barh([str(i) for i in df.index], df["mean_rate_hz"], color="#6ae4ff")
    ax.set_xlabel("mean rate (Hz)")
    ax.set_title(f"most active {by.replace('_', ' ')}s" + (" (excluding stimulated)" if exclude_driven else ""))
    return ax


def psth(result, targets, bin_ms: float = 10.0, ax=None):
    """Population rate over time for one or more groups (names or NeuronSets)."""
    ax = _ax(ax)
    for t in targets if isinstance(targets, (list, tuple)) else [targets]:
        s = result.psth(t, bin_ms)
        ax.plot(s.index + bin_ms / 2, s.to_numpy(), label=str(t) if isinstance(t, str) else (t.label or "group"))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("rate (Hz per neuron)")
    ax.legend(frameon=False)
    return ax


def brain_map(brain, values=None, view: str = "front", s: float = 0.3, cmap: str = "magma", highlight=None, ax=None,
              title: str | None = None):
    """Scatter of all neurons projected on a plane, coloured by ``values`` (e.g. ``result.rates()``).

    ``view``: ``"front"`` (x-y), ``"top"`` (x-z) or ``"side"`` (z-y).
    """
    ax = _ax(ax, (8, 5))
    P = brain.positions
    i, j = {"front": (0, 1), "top": (0, 2), "side": (2, 1)}[view]
    x, y = P[:, i], -P[:, j] if j == 1 else P[:, j]
    ax.set_facecolor("#05070d")
    if values is None:
        cols = [CLASS_COLORS.get(c, "#7f8fa6") for c in brain.neurons["super_class"]]
        ax.scatter(x, y, s=s, c=cols, alpha=0.5, linewidths=0)
    else:
        v = np.asarray(values, float)
        order = np.argsort(v)
        sc = ax.scatter(x[order], y[order], s=np.where(v[order] > 0, s * 6, s), c=v[order], cmap=cmap, linewidths=0)
        _plt().colorbar(sc, ax=ax, shrink=0.7)
    if highlight is not None:
        h = brain[highlight].idx
        ax.scatter(x[h], y[h], s=s * 20, facecolors="none", edgecolors="#6ae4ff", linewidths=0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or f"{brain.name} ({view} view)")
    return ax


def plot_trajectory(fly, ax=None):
    """The virtual fly's path, coloured by behavioural state, over its arena."""
    plt = _plt()
    ax = _ax(ax, (6, 6))
    a = fly.arena
    tr = fly.trajectory()
    n = 80
    xs, ys = np.linspace(0, a.width, n), np.linspace(0, a.height, n)
    field = np.array([[sum(a.odors_at(x, y).values()) for x in xs] for y in ys])
    ax.imshow(field, origin="lower", extent=(0, a.width, 0, a.height), cmap="magma", alpha=0.8)
    colors = {"walking": "#6ae4ff", "feeding": "#ffd166", "backing up": "#c77dff", "escaping": "#ff5d73", "still": "#aaaaaa"}
    if len(tr):
        for state, g in tr.groupby("state"):
            ax.scatter(g.x, g.y, s=3, c=colors.get(state, "#ffffff"), label=state)
        ax.plot(tr.x.iloc[0], tr.y.iloc[0], "o", color="#5dffa1", label="start")
    for it in a.items:
        if it.kind == "food":
            ax.add_patch(plt.Circle((it.x, it.y), it.radius_mm, color=it.color))
            ax.text(it.x + 3, it.y + 3, it.name, color="white", fontsize=8)
        elif it.kind == "heat":
            ax.add_patch(plt.Circle((it.x, it.y), it.radius_mm, color="#ff5a28", alpha=0.3))
    ax.set_xlim(0, a.width)
    ax.set_ylim(0, a.height)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor="white")
    ax.set_title(f"virtual fly in '{a.name}' ({tr.t_s.iloc[-1] if len(tr) else 0:.0f} s)")
    return ax


def conditioning_plot(result, ax=None):
    """Valence of CS+ and CS- before and after training."""
    ax = _ax(ax, (5, 4))
    v = result.valence
    x = np.arange(2)
    ax.bar(x - 0.18, [v.loc["pre", "CS+"], v.loc["post", "CS+"]], 0.36, label=f"CS+ ({result.cs_plus})", color="#ffd166")
    ax.bar(x + 0.18, [v.loc["pre", "CS-"], v.loc["post", "CS-"]], 0.36, label=f"CS- ({result.cs_minus})", color="#6ae4ff")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x, ["before", "after"])
    ax.set_ylabel("MBON valence (+ approach / - avoid)")
    ax.set_title(f"paired with {result.reinforcer}: learning index {result.learning_index:+.2f}")
    ax.legend(frameon=False)
    return ax
