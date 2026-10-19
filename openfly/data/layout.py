"""Fallback 3D layout for connectomes that come without neuron coordinates.

Some releases (e.g. the Codex tables of BANC, MANC or the male CNS)
ship connectivity but no positions. To still draw them, every neuron is
placed near the centre of the brain region (neuropil) it uses most,
with a little random spread. The centres below are medians of FlyWire
v783 neuron anchor positions (micrometres, FAFB space), symmetrised
left/right. VNC neuropils are given schematic positions below the
brain.

The result is a *schematic*, not anatomy. ``brain.meta["layout"]`` is
set to ``"schematic"`` so viewers can say so.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

_MID_X = 528.0

# Brain neuropils: (|dx| from midline, y, z). Bilateral ones get _L/_R.
_BILATERAL = {
    "AL": (56, 255, 30), "AME": (186, 260, 122), "AMMC": (87, 292, 95), "AOTU": (192, 186, 161),
    "ATL": (103, 171, 177), "AVLP": (146, 225, 107), "BU": (131, 174, 75), "CAN": (8, 335, 198),
    "CRE": (81, 174, 73), "EPA": (55, 209, 146), "FLA": (6, 283, 131), "GA": (80, 179, 88),
    "GOR": (56, 270, 110), "IB": (62, 185, 191), "ICL": (86, 181, 137), "IPS": (50, 254, 181),
    "LAL": (75, 197, 100), "LA": (335, 314, 131), "LH": (155, 171, 136), "LOP": (199, 285, 235),
    "LO": (210, 279, 195), "MB_CA": (95, 155, 186), "MB_ML": (88, 137, 180), "MB_PED": (58, 158, 111),
    "MB_VL": (92, 135, 176), "ME": (254, 283, 173), "PLP": (140, 216, 173), "PVLP": (151, 216, 141),
    "SCL": (118, 165, 151), "SIP": (96, 140, 130), "SLP": (153, 157, 135), "SMP": (68, 142, 143),
    "SPS": (45, 200, 182), "VES": (45, 263, 135), "WED": (89, 265, 118),
}
_MIDLINE = {
    "EB": (522, 182, 94), "FB": (529, 158, 165), "GNG": (528, 336, 144), "NO": (535, 159, 174),
    "OCG": (533, 95, 155), "PB": (529, 182, 194), "PRW": (525, 322, 72), "SAD": (503, 305, 114),
}
# Ventral nerve cord (schematic): (|dx|, y, z) below the brain.
_VNC_BILATERAL = {
    "LegNpT1": (55, 520, 150), "LegNpT2": (60, 640, 150), "LegNpT3": (55, 760, 150),
    "WTct": (35, 600, 110), "HTct": (30, 680, 110), "NTct": (25, 470, 120), "IntTct": (20, 560, 130),
    "LTct": (20, 520, 130), "mVAC": (25, 650, 170), "AMNp": (15, 840, 140), "Ov": (40, 560, 175),
}
_VNC_MIDLINE = {"ANm": (528, 880, 150), "CV": (528, 450, 140), "CvC": (528, 450, 140), "ANm_vnc": (528, 880, 150)}


def region_centers() -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for table in (_BILATERAL, _VNC_BILATERAL):
        for name, (dx, y, z) in table.items():
            out[f"{name}_L"] = (_MID_X - dx, y, z)
            out[f"{name}_R"] = (_MID_X + dx, y, z)
            out[name] = (_MID_X, y, z)
    for table in (_MIDLINE, _VNC_MIDLINE):
        out.update({k: tuple(map(float, v)) for k, v in table.items()})
    return out


def _hash_point(name: str) -> np.ndarray:
    """Deterministic pseudo-random point for an unknown region name."""
    h = hashlib.sha1(name.encode()).digest()
    u = np.frombuffer(h[:12], dtype=np.uint32) / 2**32
    theta, phi, r = 2 * np.pi * u[0], np.arccos(2 * u[1] - 1), 0.6 + 0.4 * u[2]
    return np.array(
        [_MID_X + 260 * r * np.sin(phi) * np.cos(theta), 230 + 130 * r * np.cos(phi), 140 + 90 * r * np.sin(phi) * np.sin(theta)]
    )


def _normalise_region(name: str, side: str) -> str:
    n = str(name).strip()
    if not n:
        return ""
    for suffix, s in (("_L", "left"), ("_R", "right"), ("(L)", "left"), ("(R)", "right")):
        if n.endswith(suffix):
            return n[: -len(suffix)].rstrip("_") + ("_L" if s == "left" else "_R")
    return n + ("_L" if side == "left" else "_R" if side == "right" else "")


def schematic_layout(neurons: pd.DataFrame, W=None, seed: int = 0) -> np.ndarray:
    """(N, 3) float32 positions: region centre + spread, as described above."""
    rng = np.random.default_rng(seed)
    centers = region_centers()
    lowered = {k.lower(): v for k, v in centers.items()}
    region_col = neurons.get("neuropil", pd.Series([""] * len(neurons)))
    sides = neurons.get("side", pd.Series([""] * len(neurons))).astype(str).to_numpy()
    fallback = neurons.get("super_class", pd.Series([""] * len(neurons))).astype(str).to_numpy()
    regions = region_col.astype(str).to_numpy()

    xyz = np.zeros((len(neurons), 3), dtype=np.float32)
    keys = np.array([_normalise_region(r, s) if r else f"class:{f}:{s}" for r, s, f in zip(regions, sides, fallback)])
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    for k, (key, n) in enumerate(zip(uniq, counts)):
        c = lowered.get(key.lower())
        if c is None and key.endswith(("_L", "_R")):
            # A midline region (e.g. GNG) reported with a side: nudge it.
            base = lowered.get(key[:-2].lower())
            if base is not None:
                c = (base[0] + (15.0 if key.endswith("_R") else -15.0), base[1], base[2])
        if c is None:
            c = tuple(_hash_point(key))
            if key.endswith(":left"):
                c = (c[0] - 120, c[1], c[2])
            elif key.endswith(":right"):
                c = (c[0] + 120, c[1], c[2])
        spread = 6.0 + 2.2 * float(n) ** (1 / 3)
        m = inv == k
        xyz[m] = np.asarray(c, dtype=np.float32) + rng.normal(0, spread, (int(m.sum()), 3)).astype(np.float32)
    return xyz
