"""Getting brains: downloads, the dataset catalogue, the ``.flybrain`` format,
adapters for FlyWire / Codex / neuPrint, the BrainBuilder and the toy fly."""

from .brainfile import read_brain, read_meta, write_brain
from .builder import BrainBuilder, Population, group
from .registry import REGISTRY, Dataset, datasets, download, get_dataset, load, register

__all__ = [
    "BrainBuilder",
    "Dataset",
    "Population",
    "REGISTRY",
    "datasets",
    "download",
    "get_dataset",
    "group",
    "load",
    "read_brain",
    "read_meta",
    "register",
    "write_brain",
]
