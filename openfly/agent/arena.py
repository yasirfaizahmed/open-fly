"""A 2D world for the virtual fly: food, odour plumes, objects, heat and predators.

Units are millimetres and seconds. A fruit fly walks at roughly 10-20 mm/s
and the default arena is a 120 x 120 mm box (a large Petri dish).

Odours spread from each source as a Gaussian cloud. With wind, the cloud
is stretched downwind into a plume (a crude steady-state advection
model): narrow, long and absent upwind of the source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class FoodSource:
    x: float
    y: float
    odor: str | None = "banana"
    taste: str | None = "sugar"
    radius_mm: float = 4.0
    odor_strength: float = 1.0
    odor_sigma_mm: float = 45.0
    name: str = "food"
    color: str = "#f5d742"
    kind: str = "food"


@dataclass
class HeatZone:
    x: float
    y: float
    radius_mm: float = 15.0
    intensity: float = 1.0
    name: str = "heat"
    kind: str = "heat"


@dataclass
class Predator:
    """Something that swoops at the fly. Appears at ``t_s`` from ``bearing_deg`` (relative to the fly)."""

    t_s: float = 5.0
    bearing_deg: float = 60.0
    duration_ms: float = 400.0
    name: str = "predator"
    kind: str = "predator"
    fired: bool = False


@dataclass
class Arena:
    width: float = 120.0
    height: float = 120.0
    items: list = field(default_factory=list)
    wind_from_deg: float | None = None  # direction the wind blows *from* (0 = +x)
    wind_speed: float = 0.0
    name: str = "arena"

    # ------------------------------------------------------------ fields
    def odors_at(self, x: float, y: float) -> dict[str, float]:
        """Concentration (0..~1) of each odour at a point."""
        out: dict[str, float] = {}
        for it in self.items:
            if getattr(it, "kind", "") != "food" or not it.odor:
                continue
            dx, dy = x - it.x, y - it.y
            s = it.odor_sigma_mm
            if self.wind_from_deg is not None and self.wind_speed > 0:
                # plume axis points downwind (away from where the wind comes from)
                a = math.radians(self.wind_from_deg + 180.0)
                along = dx * math.cos(a) + dy * math.sin(a)
                across = -dx * math.sin(a) + dy * math.cos(a)
                if along < -s * 0.2:
                    c = it.odor_strength * math.exp(-(along**2) / (2 * (0.2 * s) ** 2) - across**2 / (2 * (0.35 * s) ** 2))
                else:
                    width = 0.35 * s + 0.15 * max(along, 0.0)
                    c = it.odor_strength * math.exp(-across**2 / (2 * width**2) - max(along, 0) / (4 * s))
            else:
                c = it.odor_strength * math.exp(-(dx * dx + dy * dy) / (2 * s * s))
            out[it.odor] = out.get(it.odor, 0.0) + c
        return out

    def food_at(self, x: float, y: float, reach_mm: float = 0.0):
        """The food item the fly is touching (within its radius + ``reach_mm`` of proboscis/legs)."""
        for it in self.items:
            if getattr(it, "kind", "") == "food" and it.taste and math.hypot(x - it.x, y - it.y) <= it.radius_mm + reach_mm:
                return it
        return None

    def heat_at(self, x: float, y: float) -> float:
        h = 0.0
        for it in self.items:
            if getattr(it, "kind", "") == "heat":
                d = math.hypot(x - it.x, y - it.y)
                h += it.intensity * math.exp(-((d / it.radius_mm) ** 2))
        return h

    def predators(self):
        return [it for it in self.items if getattr(it, "kind", "") == "predator"]

    def visible_objects(self):
        return [it for it in self.items if getattr(it, "kind", "") == "food"]

    def to_dict(self) -> dict:
        def item(it):
            d = {k: v for k, v in it.__dict__.items()}
            return d

        return {"name": self.name, "width": self.width, "height": self.height,
                "wind_from_deg": self.wind_from_deg, "wind_speed": self.wind_speed,
                "items": [item(it) for it in self.items]}

    # ----------------------------------------------------------- presets
    @classmethod
    def preset(cls, name: str = "banana") -> "Arena":
        """Ready-made worlds: ``banana``, ``choice``, ``bitter_trap``, ``predator``, ``windy_vinegar``, ``hot_corner``."""
        P = {
            "banana": lambda: cls(items=[FoodSource(90, 75, "banana", "sugar", name="ripe banana")], name="banana"),
            "choice": lambda: cls(items=[
                FoodSource(95, 95, "banana", "sugar", name="banana"),
                FoodSource(25, 95, "geosmin", "bitter", name="mouldy fruit", color="#6b8e23", odor_sigma_mm=30.0),
            ], name="choice"),
            "bitter_trap": lambda: cls(items=[FoodSource(80, 70, "apple_cider_vinegar", "bitter", name="vinegar spiked with quinine",
                                                         color="#c97b2a")], name="bitter_trap"),
            "predator": lambda: cls(items=[FoodSource(90, 80, "banana", "sugar", name="banana"),
                                           Predator(t_s=4.0, bearing_deg=70.0)], name="predator"),
            "windy_vinegar": lambda: cls(items=[FoodSource(95, 60, "apple_cider_vinegar", "sugar", name="vinegar",
                                                           odor_sigma_mm=40, color="#c97b2a")],
                                         wind_from_deg=0.0, wind_speed=1.0, name="windy_vinegar"),
            "hot_corner": lambda: cls(items=[HeatZone(90, 90, 25, 1.0), FoodSource(30, 90, "banana", "sugar")], name="hot_corner"),
        }
        if name not in P:
            raise KeyError(f"unknown arena {name!r}; presets: {', '.join(P)}")
        return P[name]()

    PRESETS = ("banana", "choice", "bitter_trap", "predator", "windy_vinegar", "hot_corner")
