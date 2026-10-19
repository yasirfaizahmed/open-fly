"""Embodiment: put a brain in a virtual body inside a 2D world (closed loop)."""

from .arena import Arena, FoodSource, HeatZone, Predator
from .fly import VirtualFly

__all__ = ["Arena", "FoodSource", "HeatZone", "Predator", "VirtualFly"]
