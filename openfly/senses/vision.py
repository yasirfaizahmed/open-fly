"""Vision: what the fly sees -> photoreceptors and visual feature detectors.

Each compound eye has ~750 ommatidia, each looking in its own direction
and holding 8 photoreceptors: R1-6 (luminance/motion, end in the lamina)
and R7/R8 (colour, end in the medulla). FlyWire reconstructs ~11,000 of
them.

Retinotopy, i.e. which direction each photoreceptor looks in, is not
annotated, so :class:`VisualField` *estimates* it from anatomy. The eye's
map is preserved in the lamina and medulla, so within each eye the
dorsal-ventral axis of the photoreceptor terminals (the main principal
component, aligned with y) gives elevation, and the second in-sheet axis
gives azimuth (front -> back; inverted for R7/R8 because of the optic
chiasm between lamina and medulla). Coordinates are rank-normalised to
cover -15..165 deg azimuth per eye and -60..60 deg elevation. This is an
approximation, good enough for "left vs right, up vs down, front vs
back". Brains that carry explicit ``azimuth``/``elevation`` columns
(like the toy fly) use those directly.

Two levels of stimulation:

* **retina level**: :func:`flash`, :func:`spot`, :func:`image`,
  :func:`looming`, :func:`moving_object`. Photoreceptors are driven by
  light and the connectome does the rest.
* **feature level**: a point-neuron model cannot compute motion from
  photoreceptors the way real graded-potential circuits do, so
  :func:`looming`, :func:`moving_object` and :func:`grating` can also
  drive the known feature detectors directly: LC4/LPLC2 (looming), LC10/LC11
  (small objects), T4/T5 subtypes (direction of motion; Maisak et al. 2013).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .base import Stimulus, StimulusSet

PHOTORECEPTOR_TYPES = ("R1-6", "R7", "R8")
COLOR_CHANNELS = {"white": ("R1-6", "R7", "R8"), "uv": ("R7",), "blue": ("R8",), "green": ("R1-6", "R8"),
                  "yellow": ("R1-6", "R8"), "luminance": ("R1-6",)}
#: T4/T5 subtype -> preferred motion direction (Maisak et al. 2013 Nature)
MOTION_SUBTYPES = {"front_to_back": "a", "back_to_front": "b", "up": "c", "down": "d"}


class VisualField:
    """Azimuth/elevation (degrees) of every photoreceptor in a brain.

    Azimuth: 0 = straight ahead, positive = to the fly's right.
    Elevation: positive = up.
    """

    def __init__(self, brain):
        self.brain = brain
        n = brain.neurons
        is_pr = n["cell_type"].isin(PHOTORECEPTOR_TYPES).to_numpy()
        self.idx = np.flatnonzero(is_pr)
        self.cell_type = n["cell_type"].to_numpy()[self.idx]
        self.side = n["side"].to_numpy()[self.idx]
        if {"azimuth", "elevation"} <= set(n.columns) and n["azimuth"].iloc[self.idx].notna().all():
            self.azimuth = n["azimuth"].to_numpy(float)[self.idx]
            self.elevation = n["elevation"].to_numpy(float)[self.idx]
            self.estimated = False
        else:
            self.azimuth, self.elevation = self._estimate(brain.positions[self.idx])
            self.estimated = True

    @classmethod
    def of(cls, brain) -> "VisualField":
        cache = brain.__dict__.setdefault("_visual_field", None)
        if cache is None:
            cache = brain.__dict__["_visual_field"] = cls(brain)
        return cache

    def _estimate(self, P: np.ndarray):
        az = np.zeros(len(P))
        el = np.zeros(len(P))
        for side in ("left", "right"):
            for group in (("R1-6",), ("R7", "R8")):
                m = (self.side == side) & np.isin(self.cell_type, group)
                if m.sum() < 10:
                    continue
                X = P[m].astype(float)
                X = X - X.mean(axis=0)
                _, _, vt = np.linalg.svd(X, full_matrices=False)
                k_el = int(np.argmax(np.abs(vt[:2, 1])))  # the in-sheet axis closest to y
                e_axis, a_axis = vt[k_el], vt[1 - k_el]
                pe = X @ e_axis * -np.sign(e_axis[1])  # dorsal (small y) -> up
                pa = X @ a_axis * np.sign(a_axis[2] or 1.0)  # posterior (larger z) -> backward
                if group != ("R1-6",):
                    pa = -pa  # optic chiasm flips front/back between lamina and medulla
                ra = pd.Series(pa).rank(pct=True).to_numpy()
                re = pd.Series(pe).rank(pct=True).to_numpy()
                sgn = 1.0 if side == "right" else -1.0
                az[m] = sgn * (-15.0 + 180.0 * ra)
                el[m] = -60.0 + 120.0 * re
        return az, el

    def select(self, azimuth: float, elevation: float, radius_deg: float, colors=("R1-6", "R7", "R8")):
        """Photoreceptors (indices, angular distance) within ``radius_deg`` of a direction."""
        d = angular_distance(self.azimuth, self.elevation, azimuth, elevation)
        m = (d <= radius_deg) & np.isin(self.cell_type, colors)
        return self.idx[m], d[m]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"index": self.idx, "cell_type": self.cell_type, "side": self.side,
                             "azimuth": self.azimuth, "elevation": self.elevation})


def angular_distance(az1, el1, az2, el2):
    """Great-circle distance in degrees."""
    a1, e1, a2, e2 = map(np.radians, (az1, el1, az2, el2))
    c = np.sin(e1) * np.sin(e2) + np.cos(e1) * np.cos(e2) * np.cos(a1 - a2)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# ----------------------------------------------------------- retina level
def flash(brain, intensity: float = 1.0, eye: str = "both", color: str = "white", max_rate_hz: float = 150.0,
          start_ms: float = 0.0, stop_ms: float | None = 50.0) -> Stimulus:
    """A full-field light flash (default 50 ms)."""
    vf = VisualField.of(brain)
    m = np.isin(vf.cell_type, COLOR_CHANNELS[color])
    if eye in ("left", "right"):
        m &= vf.side == eye
    return Stimulus(vf.idx[m], max_rate_hz * intensity, start_ms, stop_ms, f"vision:flash({eye})")


def spot(brain, azimuth: float = 0.0, elevation: float = 0.0, radius_deg: float = 12.0, intensity: float = 1.0,
         color: str = "white", max_rate_hz: float = 150.0, start_ms: float = 0.0, stop_ms: float | None = None) -> Stimulus:
    """A bright patch (e.g. a piece of fruit) at a direction in the visual field."""
    vf = VisualField.of(brain)
    idx, d = vf.select(azimuth, elevation, radius_deg, COLOR_CHANNELS[color])
    rates = max_rate_hz * intensity * (1.0 - 0.5 * (d / max(radius_deg, 1e-6)) ** 2)
    return Stimulus(idx, rates, start_ms, stop_ms, f"vision:spot({azimuth:.0f},{elevation:.0f})",
                    info={"azimuth": azimuth, "elevation": elevation, "radius": radius_deg})


def image(brain, picture, max_rate_hz: float = 150.0, start_ms: float = 0.0, stop_ms: float | None = None,
          color: str = "white") -> Stimulus:
    """Show the fly a picture.

    ``picture`` is a 2-D array (H x W, values 0..1) or an RGB array,
    interpreted as a panorama: columns span azimuth -180..180 deg (left to
    right), rows span elevation +90 (top) .. -90 (bottom). Load any photo
    with Pillow and pass ``np.asarray(img) / 255``.
    """
    arr = np.asarray(picture, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    if arr.max() > 1.0:
        arr = arr / 255.0
    H, W = arr.shape
    vf = VisualField.of(brain)
    col = np.clip(((vf.azimuth + 180.0) / 360.0 * W).astype(int), 0, W - 1)
    row = np.clip(((90.0 - vf.elevation) / 180.0 * H).astype(int), 0, H - 1)
    vals = arr[row, col]
    m = np.isin(vf.cell_type, COLOR_CHANNELS[color]) & (vals > 0.01)
    return Stimulus(vf.idx[m], max_rate_hz * vals[m], start_ms, stop_ms, "vision:image")


def _feature_cells(brain, cell_type: str, side: str | None, azimuth: float | None = None, radius: float = 60.0):
    try:
        ns = brain[cell_type]
    except KeyError:
        return np.zeros(0, np.int64), np.zeros(0)
    if side in ("left", "right"):
        ns = ns.where(side=side)
    idx = ns.idx
    w = np.ones(idx.size)
    n = brain.neurons
    if azimuth is not None and "azimuth" in n and n["azimuth"].iloc[idx].notna().all() and idx.size:
        d = angular_distance(n["azimuth"].to_numpy(float)[idx], n["elevation"].fillna(0).to_numpy(float)[idx], azimuth, 0.0)
        w = np.clip(1.0 - d / radius, 0.0, 1.0)
        keep = w > 0
        idx, w = idx[keep], w[keep]
    return idx, w


def looming(
    brain,
    azimuth: float = 45.0,
    elevation: float = 0.0,
    start_ms: float = 0.0,
    duration_ms: float = 400.0,
    r_over_v_ms: float = 40.0,
    max_size_deg: float = 120.0,
    features: bool = True,
    retina: bool = True,
    max_rate_hz: float = 150.0,
) -> StimulusSet:
    """An object on a collision course (a predator or a swatter).

    The disk's angular size follows the classic ``theta(t) = 2 atan(r/v / (t_c - t))``
    (von Reyn et al. 2017), so it expands slowly and then explodes. Photoreceptors
    inside the disk are driven (``retina``); LC4 (tuned to expansion speed)
    and LPLC2 (tuned to size) on that side are driven too (``features``).
    """
    t_c = duration_ms

    def theta(t):
        tt = min(t, t_c - 1.0)
        return min(max_size_deg, math.degrees(2 * math.atan(r_over_v_ms / max(t_c - tt, 1.0))))

    out = StimulusSet()
    stop = start_ms + duration_ms
    side = "right" if azimuth > 0 else "left"
    if retina:
        vf = VisualField.of(brain)
        d_all = angular_distance(vf.azimuth, vf.elevation, azimuth, elevation)
        cand = d_all <= max_size_deg / 2
        idx, d = vf.idx[cand], d_all[cand]

        def mod(t, d=d):
            return (d <= theta(t) / 2).astype(float)

        out.append(Stimulus(idx, max_rate_hz, start_ms, stop, "vision:looming(retina)", modulation=mod))
    if features:
        lc4, w4 = _feature_cells(brain, "LC4", side, azimuth)
        lplc2, w2 = _feature_cells(brain, "LPLC2", side, azimuth)

        def speed(t):
            th0, th1 = theta(t), theta(t + 5.0)
            return min(1.0, (th1 - th0) / 5.0 / 0.6)  # deg/ms, saturating ~600 deg/s

        if lc4.size:
            out.append(Stimulus(lc4, max_rate_hz * w4, start_ms, stop, "vision:looming(LC4)",
                                modulation=lambda t: speed(t)))
        if lplc2.size:
            out.append(Stimulus(lplc2, max_rate_hz * w2, start_ms, stop, "vision:looming(LPLC2)",
                                modulation=lambda t: min(1.0, theta(t) / 60.0)))
    return out


def moving_object(
    brain,
    start_azimuth: float = -60.0,
    end_azimuth: float = 60.0,
    elevation: float = 0.0,
    size_deg: float = 10.0,
    duration_ms: float = 1000.0,
    start_ms: float = 0.0,
    features: bool = True,
    max_rate_hz: float = 150.0,
) -> StimulusSet:
    """A small object (another fly, a fruit fly larva...) sweeping across the view."""
    vf = VisualField.of(brain)

    def az_at(t):
        f = np.clip(t / duration_ms, 0, 1)
        return start_azimuth + f * (end_azimuth - start_azimuth)

    cand = np.abs(vf.elevation - elevation) <= size_deg * 2
    idx = vf.idx[cand]
    az, el = vf.azimuth[cand], vf.elevation[cand]

    def mod(t):
        return (angular_distance(az, el, az_at(t), elevation) <= size_deg / 2).astype(float)

    stop = start_ms + duration_ms
    out = StimulusSet([Stimulus(idx, max_rate_hz, start_ms, stop, "vision:object(retina)", modulation=mod)])
    if features:
        for ct in ("LC10", "LC11"):
            for side, sgn in (("left", -1), ("right", 1)):
                cells, w = _feature_cells(brain, ct, side)
                if not cells.size:
                    continue
                out.append(Stimulus(cells, max_rate_hz * 0.8 * w, start_ms, stop, f"vision:object({ct},{side})",
                                    modulation=lambda t, sgn=sgn: float(np.sign(az_at(t)) == sgn or abs(az_at(t)) < 15)))
    return out


def grating(
    brain,
    direction: str = "front_to_back",
    eye: str = "both",
    temporal_hz: float = 2.0,
    contrast: float = 1.0,
    start_ms: float = 0.0,
    stop_ms: float | None = None,
    features: bool = True,
    retina: bool = True,
    max_rate_hz: float = 120.0,
) -> StimulusSet:
    """A drifting stripe pattern filling the view (optic flow).

    ``direction``: ``front_to_back`` / ``back_to_front`` / ``up`` / ``down``.
    Retina: photoreceptors flicker at ``temporal_hz`` with a phase that
    travels across the eye. Features: the T4/T5 subtype preferring that
    direction is driven (Maisak et al. 2013).
    """
    if direction not in MOTION_SUBTYPES:
        raise ValueError(f"direction must be one of {list(MOTION_SUBTYPES)}")
    out = StimulusSet()
    eyes = ("left", "right") if eye == "both" else (eye,)
    if retina:
        vf = VisualField.of(brain)
        m = np.isin(vf.side, eyes)
        idx = vf.idx[m]
        coord = np.abs(vf.azimuth[m]) if direction in ("front_to_back", "back_to_front") else vf.elevation[m]
        sign = 1.0 if direction in ("front_to_back", "up") else -1.0
        phase = np.radians(coord) * 4.0  # 4 stripes per radian of visual angle

        def mod(t, phase=phase, sign=sign):
            return 0.5 + 0.5 * contrast * np.sin(2 * np.pi * temporal_hz * t / 1000.0 - sign * phase)

        out.append(Stimulus(idx, max_rate_hz, start_ms, stop_ms, f"vision:grating({direction})", modulation=mod))
    if features:
        sub = MOTION_SUBTYPES[direction]
        for kind in ("T4", "T5"):
            for side in eyes:
                cells, w = _feature_cells(brain, f"{kind}{sub}", side)
                if cells.size:
                    out.append(Stimulus(cells, max_rate_hz * contrast * w, start_ms, stop_ms,
                                        f"vision:{kind}{sub}({side})"))
    return out
