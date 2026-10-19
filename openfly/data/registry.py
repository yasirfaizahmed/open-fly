"""Catalogue of brains openfly knows how to fetch and build.

``openfly.load(name)`` looks the name up here, downloads the original
files (once), builds a :class:`~openfly.Brain`, caches it as a
``.flybrain`` file and returns it. The next load reads the cache in
seconds.

Built-in entries:

=================  ==========================================================
``toy``            synthetic ~2,500-neuron teaching brain (no download)
``flywire_783``    FlyWire adult female brain, v783, ~139k neurons (default)
``flywire_630``    FlyWire v630, the version used by Shiu et al. 2024
``banc_888``       BANC brain *and* nerve cord (Codex)
``manc_1.2.1``     male adult nerve cord (Codex)
``mcns_1.0``       male central nervous system (Codex)
``maol_1.1``       male optic lobe (Codex)
``hemibrain_1.2``  Janelia hemibrain v1.2 (neuPrint export)
``codex:<d>/<v>``  any other Codex release, e.g. ``codex:banc/999``
``*.flybrain``     a brain file on disk
=================  ==========================================================

Register your own with :func:`register`. See ``docs/12-adding-new-brains.md``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .. import config
from .download import fetch

SHIU_REF = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOTATIONS_REF = "a83b2776d60d5764cef36b927f5f9679c16c47a2"


def _gh(owner_repo: str, ref: str, path: str) -> list[str]:
    """Pinned raw-GitHub URL first, then the moving ``main`` branch as a fallback."""
    return [
        f"https://raw.githubusercontent.com/{owner_repo}/{ref}/{path}",
        f"https://raw.githubusercontent.com/{owner_repo}/main/{path}",
    ]


@dataclass
class Dataset:
    """Everything needed to obtain and assemble one brain."""

    name: str
    title: str
    description: str
    build: Callable[[dict], object]
    files: dict[str, list[str]] = field(default_factory=dict)
    optional_files: dict[str, list[str]] = field(default_factory=dict)
    citation: str = ""
    license: str = ""
    homepage: str = ""
    download_mb: float = 0.0
    species: str = "Drosophila melanogaster"
    tags: tuple = ()
    cache: bool = True

    def meta(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "citation": self.citation,
            "license": self.license,
            "homepage": self.homepage,
            "species": self.species,
            "sources": {k: v[0] for k, v in {**self.files, **self.optional_files}.items()},
        }


REGISTRY: dict[str, Dataset] = {}


def register(dataset: Dataset) -> Dataset:
    REGISTRY[dataset.name] = dataset
    return dataset


def datasets() -> pd.DataFrame:
    """Table of registered datasets."""
    rows = [
        {"name": d.name, "title": d.title, "download_mb": d.download_mb, "cached": _cache_path(d).exists()}
        for d in REGISTRY.values()
    ]
    return pd.DataFrame(rows).set_index("name")


def get_dataset(name: str) -> Dataset:
    if name in REGISTRY:
        return REGISTRY[name]
    m = re.fullmatch(r"codex:([\w-]+)/([\w.\-]+)", name)
    if m:
        return _codex_dataset(m.group(1), m.group(2), name=name)
    raise KeyError(f"Unknown dataset {name!r}. Known: {', '.join(REGISTRY)} or 'codex:<dataset>/<version>'.")


def _cache_path(ds: Dataset) -> Path:
    safe = re.sub(r"[^\w.\-]+", "_", ds.name)
    return config.brains_dir() / f"{safe}.flybrain"


def _local_override(filename: str) -> Path | None:
    d = os.environ.get("OPENFLY_DATA_DIR")
    if not d:
        return None
    for p in Path(d).expanduser().rglob(filename):
        return p
    return None


def download(name: str, include_optional: bool = True, progress: bool = True) -> dict[str, Path | None]:
    """Fetch the raw files of a dataset (no-op for files already present)."""
    ds = get_dataset(name)
    out: dict[str, Path | None] = {}
    raw = config.raw_dir(ds.name)
    for key, urls in ds.files.items():
        fname = urls[0].rsplit("/", 1)[-1]
        out[key] = _local_override(fname) or fetch(urls, raw / fname, progress=progress)
    if include_optional:
        for key, urls in ds.optional_files.items():
            fname = urls[0].rsplit("/", 1)[-1]
            try:
                out[key] = _local_override(fname) or fetch(urls, raw / fname, progress=progress)
            except Exception:  # optional extras must never block loading
                out[key] = None
    return out


def load(name: str | Path = "flywire_783", rebuild: bool = False, progress: bool = True):
    """Load a brain by dataset name or ``.flybrain`` path. Downloads on first use."""
    from .brainfile import read_brain, write_brain

    p = Path(str(name)).expanduser()
    if p.suffix == ".flybrain":
        return read_brain(p)
    ds = get_dataset(str(name))
    cache = _cache_path(ds)
    if ds.cache and cache.exists() and not rebuild:
        return read_brain(cache)
    paths = download(ds.name, progress=progress) if (ds.files or ds.optional_files) else {}
    brain = ds.build(paths)
    brain.meta.update(ds.meta())
    if ds.cache:
        write_brain(brain, cache)
        brain.meta["file"] = str(cache)
    return brain


# ======================================================================
# Built-in datasets
# ======================================================================

FLYWIRE_CITATION = (
    "Dorkenwald et al. 2024 Nature 634:124 (FlyWire connectome); Schlegel et al. 2024 Nature 634:139 "
    "(annotations); Shiu et al. 2024 Nature 634:210 (LIF model and connectivity table); "
    "Eckstein et al. 2024 Cell 187:2574 (neurotransmitter predictions)"
)
FLYWIRE_TERMS = (
    "FlyWire data: see FlyWire/Codex terms of use and citation guidelines (https://codex.flywire.ai). "
    "Connectivity table: MIT (Shiu & Spiller)."
)


def _build_toy(_paths):
    from .toy import make_toy_brain

    return make_toy_brain()


def _build_flywire(version: str):
    def build(paths):
        from .flywire import build_flywire

        return build_flywire(
            paths["connectivity"],
            paths["completeness"],
            annotations=paths.get("annotations"),
            neuropil_table=paths.get("neuropils") if version == "783" else None,
        )

    return build


register(
    Dataset(
        name="toy",
        title="Toy fly (synthetic teaching brain)",
        description="~2,500 hand-wired neurons with real cell-type names. Builds offline in a second.",
        build=_build_toy,
        citation="openfly toy brain (synthetic, not real data)",
        license="MIT",
        cache=False,
        tags=("offline", "teaching"),
    )
)

register(
    Dataset(
        name="flywire_783",
        title="FlyWire FAFB v783: adult female Drosophila brain",
        description=(
            "Whole adult fly brain reconstructed from electron microscopy (7,050 sections of 40 nm). "
            "~139k neurons, ~15M connections, ~50M synapses, with cell types, sides and transmitters."
        ),
        build=_build_flywire("783"),
        files={
            "connectivity": _gh("philshiu/Drosophila_brain_model", SHIU_REF, "Connectivity_783.parquet"),
            "completeness": _gh("philshiu/Drosophila_brain_model", SHIU_REF, "Completeness_783.csv"),
            "annotations": _gh(
                "flyconnectome/flywire_annotations",
                ANNOTATIONS_REF,
                "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
            ),
        },
        optional_files={
            "neuropils": ["https://storage.googleapis.com/flywire-data/codex/data/fafb/783/neuropil_synapse_table.csv.gz"]
        },
        citation=FLYWIRE_CITATION,
        license=FLYWIRE_TERMS,
        homepage="https://flywire.ai",
        download_mb=139,
        tags=("default", "whole-brain", "female"),
    )
)

register(
    Dataset(
        name="flywire_630",
        title="FlyWire FAFB v630 (version used in Shiu et al. 2024)",
        description="Earlier FlyWire snapshot, for exact replication of the published model. Few annotations.",
        build=_build_flywire("630"),
        files={
            "connectivity": _gh(
                "philshiu/Drosophila_brain_model", SHIU_REF, "2023_03_23_connectivity_630_final.parquet"
            ),
            "completeness": _gh("philshiu/Drosophila_brain_model", SHIU_REF, "2023_03_23_completeness_630_final.csv"),
        },
        optional_files={
            "annotations": _gh(
                "flyconnectome/flywire_annotations",
                ANNOTATIONS_REF,
                "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
            )
        },
        citation=FLYWIRE_CITATION,
        license=FLYWIRE_TERMS,
        homepage="https://flywire.ai",
        download_mb=125,
        tags=("replication",),
    )
)


def _codex_dataset(dataset: str, version: str, name: str | None = None, **info) -> Dataset:
    from . import codex

    def build(paths):
        return codex.build_codex(paths["neurons"], paths["connections"])

    return Dataset(
        name=name or f"codex:{dataset}/{version}",
        title=info.get("title", f"Codex {dataset} v{version}"),
        description=info.get("description", f"Connectome '{dataset}' version {version} from FlyWire Codex."),
        build=build,
        files=codex.urls(dataset, version),
        citation=info.get("citation", "See https://codex.flywire.ai for the papers to cite."),
        license=info.get("license", "See the dataset's terms on https://codex.flywire.ai"),
        homepage="https://codex.flywire.ai",
        download_mb=info.get("download_mb", 0.0),
        tags=("codex",),
    )


for _name, _d, _v, _title, _mb, _cite in [
    ("banc_888", "banc", "888", "BANC: female brain and nerve cord connectome", 32,
     "Bates et al. 2025 bioRxiv (Brain And Nerve Cord connectome)"),
    ("manc_1.2.1", "manc", "1.2.1", "MANC: male adult ventral nerve cord", 28,
     "Takemura et al. 2024 eLife; Marin et al. 2024 eLife; Cheong et al. 2024 eLife"),
    ("mcns_1.0", "mcns", "1.0", "Male CNS: complete male central nervous system", 48,
     "Berg et al. 2026 Cell (male CNS connectome)"),
    ("maol_1.1", "maol", "1.1", "Male optic lobe connectome", 32, "Nern et al. 2025 Nature (optic lobe)"),
]:
    register(_codex_dataset(_d, _v, name=_name, title=_title, download_mb=_mb, citation=_cite))


def _build_hemibrain(paths):
    from . import neuprint

    lookup = None
    ann = paths.get("flywire_annotations")
    if ann is not None:
        lookup = neuprint.nt_lookup_from_flywire(ann)
    return neuprint.build_neuprint(paths["archive"], nt_lookup=lookup)


register(
    Dataset(
        name="hemibrain_1.2",
        title="Janelia hemibrain v1.2 (central brain, one side mostly)",
        description="~25k traced neurons of the central brain. Transmitters borrowed from FlyWire type matches.",
        build=_build_hemibrain,
        files={"archive": ["https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"]},
        optional_files={
            "flywire_annotations": _gh(
                "flyconnectome/flywire_annotations",
                ANNOTATIONS_REF,
                "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
            )
        },
        citation="Scheffer et al. 2020 eLife 9:e57443",
        license="CC BY 4.0 (Janelia; check source)",
        homepage="https://neuprint.janelia.org",
        download_mb=46,
        tags=("janelia",),
    )
)
