"""Smell: odours -> olfactory receptor neurons (ORNs).

A fly has ~50 kinds of ORN. Each kind expresses one receptor and sends
its axons to one *glomerulus* of the antennal lobe (DM1, DA2, V, ...),
so an odour can be described by how strongly it activates each
glomerulus. In FlyWire, ORNs are typed by glomerulus (``ORN_DM1``...).

The odour table below is a **simplified, educational** summary of
published receptor responses (Hallem & Carlson 2006; the DoOR database,
Münch & Galizia 2016; plus the papers cited per odour). Real tuning is
graded and concentration-dependent. For research, plug in DoOR response
profiles via :func:`custom_odor`.

Laterality: ORNs are annotated with the antenna (side) their nerve enters
from, so you can give each antenna its own concentration (``left=``,
``right=``). The virtual fly uses this to steer up odour gradients, as real
flies do (Gaudry et al. 2013).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import Stimulus


@dataclass(frozen=True)
class Odor:
    name: str
    glomeruli: dict
    valence: str
    description: str
    reference: str = ""
    #: glomeruli recruited only at high concentration (often aversive)
    high_concentration: dict = field(default_factory=dict)


ODORS: dict[str, Odor] = {}


def _o(name, glomeruli, valence, description, reference="", high=None):
    ODORS[name] = Odor(name, dict(glomeruli), valence, description, reference, dict(high or {}))


_o("apple_cider_vinegar", {"DM1": 1.0, "VA2": 0.8, "DM4": 0.6, "DP1m": 0.4, "DM2": 0.3}, "attractive",
   "Fermenting fruit. Attractive at low concentration; recruits aversive DM5 when very strong.",
   "Semmelhack & Wang 2009 Nature", high={"DM5": 0.9})
_o("banana", {"DM2": 1.0, "DM3": 0.8, "VM2": 0.6, "DM1": 0.5, "VA2": 0.3, "DM4": 0.3}, "attractive",
   "Ripe banana: a bouquet of fruity esters (isoamyl acetate, ethyl butyrate...).",
   "Hallem & Carlson 2006 Cell; Stensmyr et al. 2003")
_o("ethyl_acetate", {"DM1": 1.0, "DM2": 0.8, "DM3": 0.5, "VM2": 0.4}, "attractive",
   "Fruity solvent smell typical of fermentation.", "Hallem & Carlson 2006 Cell")
_o("isoamyl_acetate", {"DM3": 0.9, "DM2": 0.7, "VM2": 0.5, "DL3": 0.2}, "attractive",
   "The main 'banana' ester.", "Hallem & Carlson 2006 Cell")
_o("acetic_acid", {"DP1l": 1.0, "DC4": 0.5, "DP1m": 0.3}, "attractive",
   "Vinegar acid, sensed by ionotropic receptor IR75a.", "Silbering et al. 2011 J Neurosci")
_o("ammonia", {"VM1": 1.0, "DP1l": 0.2}, "attractive", "Amines from decaying food (IR92a).", "Min et al. 2013 PNAS")
_o("citrus", {"DC1": 1.0, "DC3": 0.4}, "attractive", "Limonene/valencene from citrus peel (Or19a).",
   "Dweck et al. 2013 Curr Biol")
_o("farnesol", {"DC3": 1.0}, "attractive", "Citrus-peel terpene (Or83c).", "Ronderos et al. 2014 Front Cell Neurosci")
_o("green_leaves", {"DL5": 1.0, "VA5": 0.3}, "mixed", "E2-hexenal, the smell of cut leaves (Or7a).", "Hallem & Carlson 2006 Cell")
_o("methyl_salicylate", {"DL1": 1.0}, "mixed", "Wintergreen / plant stress volatile (Or10a).", "Hallem & Carlson 2006 Cell")
_o("geosmin", {"DA2": 1.0}, "aversive", "Earthy smell of toxic moulds and bacteria. One dedicated channel (Or56a).",
   "Stensmyr et al. 2012 Cell")
_o("co2", {"V": 1.0}, "aversive", "Carbon dioxide, released by stressed flies (Gr21a/Gr63a).", "Suh et al. 2004 Nature")
_o("cva", {"DA1": 1.0, "VA6": 0.2}, "pheromone", "11-cis-vaccenyl acetate, a male pheromone (Or67d).",
   "Kurtovic et al. 2007 Nature")
_o("fly_odor", {"VA1v": 0.8, "VA1d": 0.8, "DA1": 0.3}, "pheromone", "Body odour of other flies (Or47b/Or88a).",
   "Dweck et al. 2015 Cell Rep")
_o("moldy_banana", {"DM2": 0.9, "DM3": 0.7, "VM2": 0.5, "DM1": 0.4, "DA2": 1.0}, "conflict",
   "A ripe banana spoiled by mould: attractive esters plus geosmin. Which wins?", "Stensmyr et al. 2012 Cell")


def list_odors() -> list[str]:
    return sorted(ODORS)


def custom_odor(name: str, glomeruli: dict, valence: str = "unknown", description: str = "") -> Odor:
    """Define and register a new odour by its glomerulus activation profile (0..1)."""
    o = Odor(name, dict(glomeruli), valence, description or "user-defined odour")
    ODORS[name] = o
    return o


def _saturate(c: float) -> float:
    """Concentration (0 = none, 1 = typical) -> activation 0..1 (Hill-like)."""
    c = max(float(c), 0.0)
    return min(1.0, 1.3 * c / (c + 0.3))


def orn_profile(brain, odor) -> dict:
    """{glomerulus: (NeuronSet of ORNs, weight)} for glomeruli present in this brain."""
    o = ODORS[odor] if isinstance(odor, str) else odor
    gl = o.glomeruli if isinstance(o, Odor) else dict(o)
    out = {}
    for g, w in gl.items():
        ns = brain.select(cell_type=f"ORN_{g}")
        if len(ns):
            out[g] = (ns, float(w))
    return out


def smell(
    brain,
    odor="banana",
    concentration: float = 1.0,
    *,
    left: float | None = None,
    right: float | None = None,
    max_rate_hz: float = 150.0,
    start_ms: float = 0.0,
    stop_ms: float | None = None,
) -> Stimulus:
    """Stimulus for an odour at a given concentration.

    ``odor`` is a name from :data:`ODORS`, an :class:`Odor`, or a dict
    ``{"DM1": 1.0, ...}``. ``left``/``right`` override the concentration
    at each antenna. Above ``concentration=1`` the odour's
    high-concentration glomeruli are recruited too.
    """
    o = ODORS[odor] if isinstance(odor, str) else odor
    if isinstance(odor, str) and odor not in ODORS:
        raise KeyError(f"unknown odour {odor!r}; known: {', '.join(list_odors())}")
    profile = dict(o.glomeruli) if isinstance(o, Odor) else dict(o)
    cl = concentration if left is None else left
    cr = concentration if right is None else right
    if isinstance(o, Odor) and max(cl, cr) > 1.0:
        for g, w in o.high_concentration.items():
            profile[g] = max(profile.get(g, 0.0), w * min(1.0, max(cl, cr) - 1.0))
    targets, rates = [], []
    side = brain.neurons["side"].to_numpy()
    for g, (ns, w) in orn_profile(brain, profile).items():
        s = side[ns.idx]
        c = np.where(s == "left", cl, np.where(s == "right", cr, (cl + cr) / 2))
        r = max_rate_hz * w * np.vectorize(_saturate)(c)
        targets.append(ns.idx)
        rates.append(r)
    name = o.name if isinstance(o, Odor) else "custom odour"
    if not targets:
        return Stimulus(np.zeros(0, np.int64), 0.0, start_ms, stop_ms, f"smell:{name}")
    return Stimulus(np.concatenate(targets), np.concatenate(rates), start_ms, stop_ms, f"smell:{name}",
                    info={"odor": name, "concentration": concentration})
