"""Virtual genetic screens: which neurons drive a behaviour, and which are needed?

Biologists find behaviour circuits by activating or silencing thousands of
genetically defined neuron types one at a time. Shiu et al. (2024) showed
the whole-brain model can do this in silico. These helpers run such
screens for you:

* :func:`activation_screen`: "activate each candidate group: who drives the target?"
  (sufficiency).
* :func:`silencing_screen`: "with this stimulus on, silence each candidate:
  who is needed?" (necessity).

Candidates default to cell types upstream of the target within a few
synapses. ``model="rate"`` makes screens of hundreds of groups take
seconds; confirm hits with ``model="lif"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..analysis.graph import hops


def upstream_groups(brain, target, n_hops: int = 2, min_synapses: int = 5, by: str = "cell_type",
                    max_groups: int = 200, exclude_target: bool = True) -> list[str]:
    """Cell types (or other groups) within ``n_hops`` upstream of ``target``."""
    from ..brain import Brain

    rev = Brain.__new__(Brain)  # lightweight reversed view for hop counting
    rev.__dict__.update(brain.__dict__)
    rev.W = brain.W.T.tocsr()
    rev.__dict__.pop("W_csc", None)
    d = hops(rev, target, min_synapses=min_synapses, max_hops=n_hops)
    sel = brain.neurons.loc[d.notna().to_numpy() & (d.to_numpy() > 0)]
    if exclude_target:
        sel = sel.drop(index=brain[target].idx, errors="ignore")
    counts = sel[by].replace("", np.nan).dropna().value_counts()
    return counts.index[:max_groups].tolist()


def _rates(brain, model, stimuli, duration_ms, seed, silence=None, rm=None):
    if model == "rate":
        from ..sim.rate import RateModel

        rm = rm or RateModel(brain)
        rm.silenced[:] = False
        if silence is not None:
            rm.silenced[silence] = True
        return rm.run(stimuli)
    from ..sim.simulator import Simulator

    sim = Simulator(brain, seed=seed)
    sim.add(stimuli)
    if silence is not None:
        sim.engine.silence(silence)
    return sim.run(duration_ms).rates()


def activation_screen(brain, target, candidates=None, by: str = "cell_type", rate_hz: float = 150.0,
                      model: str = "rate", duration_ms: float = 300.0, n_hops: int = 2, top: int = 20,
                      seed: int = 0, progress: bool = False) -> pd.DataFrame:
    """Activate each candidate group alone and measure the target's rate."""
    from ..senses.base import Stimulus
    from ..sim.rate import RateModel

    cands = candidates or upstream_groups(brain, target, n_hops=n_hops, by=by)
    tgt = brain[target].idx
    rm = RateModel(brain) if model == "rate" else None
    rows = []
    for k, g in enumerate(cands):
        ns = brain.select(**{by: g}) if isinstance(g, str) and by != "name" else brain[g]
        if not len(ns):
            continue
        r = _rates(brain, model, Stimulus(ns.idx, rate_hz, name=str(g)), duration_ms, seed, rm=rm)
        rows.append({"group": g, "neurons": len(ns), "target_rate_hz": float(r[tgt].mean()),
                     "brain_active": int((r > 1).sum())})
        if progress and (k + 1) % 20 == 0:
            print(f"  screened {k + 1}/{len(cands)}", flush=True)
    df = pd.DataFrame(rows)
    return df.sort_values("target_rate_hz", ascending=False).head(top).reset_index(drop=True) if len(df) else df


def silencing_screen(brain, stimuli, target, candidates=None, by: str = "cell_type", model: str = "lif",
                     duration_ms: float = 500.0, top: int = 20, max_groups: int = 60, seed: int = 0,
                     progress: bool = False) -> pd.DataFrame:
    """With ``stimuli`` on, silence each candidate group and measure the target.

    Candidates default to the most active cell types in an unperturbed run.
    """
    from ..sim.rate import RateModel

    tgt = brain[target].idx
    rm = RateModel(brain) if model == "rate" else None
    base = _rates(brain, model, stimuli, duration_ms, seed, rm=rm)
    base_rate = float(base[tgt].mean())
    if candidates is None:
        active = brain.neurons.loc[base > 1.0]
        active = active.drop(index=tgt, errors="ignore")
        candidates = active[by].replace("", np.nan).dropna().value_counts().index[:max_groups].tolist()
    rows = []
    driven = set()
    from ..senses.base import as_stimuli

    for s in as_stimuli(stimuli):
        driven.update(s.targets.tolist())
    for k, g in enumerate(candidates):
        ns = brain.select(**{by: g})
        idx = np.array([i for i in ns.idx if i not in driven], dtype=np.int64)
        if not idx.size:
            continue
        r = _rates(brain, model, stimuli, duration_ms, seed, silence=idx, rm=rm)
        rows.append({"group": g, "neurons": int(idx.size), "target_rate_hz": float(r[tgt].mean()),
                     "change_hz": float(r[tgt].mean()) - base_rate})
        if progress and (k + 1) % 10 == 0:
            print(f"  silenced {k + 1}/{len(candidates)}", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        df["percent_of_baseline"] = 100 * df["target_rate_hz"] / base_rate if base_rate else np.nan
        df = df.sort_values("change_hz").head(top).reset_index(drop=True)
    df.attrs["baseline_rate_hz"] = base_rate
    return df
