"""The ``.flybrain`` file format: one portable file per connectome.

A ``.flybrain`` file is an ordinary ZIP archive (open it with any unzip
tool) holding:

``meta.json``
    name, title, species, version, citation, licence, source URLs, units,
    sign convention and the format version.
``neurons.parquet``
    one row per neuron; row order defines the model index. Columns follow
    :data:`openfly.brain.SCHEMA` plus any extra annotations.
``edges.parquet``
    ``pre``, ``post`` (int32 model indices) and ``weight`` (int32 *signed*
    synapse count: + excitatory, - inhibitory).

Parquet is used because it is compact, fast and readable from Python, R,
Julia, JavaScript and Spark, so the files are useful outside openfly too.
See ``docs/03-data-and-brain-files.md``.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

FORMAT = "flybrain"
FORMAT_VERSION = 1


def write_brain(brain, path: str | Path) -> Path:
    from .. import __version__

    path = Path(path)
    if path.suffix != ".flybrain":
        path = path.with_suffix(".flybrain")
    path.parent.mkdir(parents=True, exist_ok=True)
    coo = brain.W.tocoo()
    edges = pd.DataFrame(
        {
            "pre": coo.row.astype(np.int32),
            "post": coo.col.astype(np.int32),
            "weight": np.rint(coo.data).astype(np.int32),
        }
    )
    meta = dict(brain.meta)
    meta.update(
        {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "n_neurons": brain.N,
            "n_connections": int(brain.W.nnz),
            "units": meta.get("units", "micrometre"),
            "sign_convention": meta.get(
                "sign_convention", "weight = synapse count, negative for inhibitory (GABA, glutamate)"
            ),
            "written_by": f"openfly {__version__}",
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("meta.json", json.dumps(_jsonable(meta), indent=2))
        z.writestr("neurons.parquet", _to_parquet(brain.neurons))
        z.writestr("edges.parquet", _to_parquet(edges))
    tmp.replace(path)
    return path


def read_meta(path: str | Path) -> dict:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("meta.json"))


def read_brain(path: str | Path):
    from ..brain import Brain

    path = Path(path)
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a .flybrain file")
        if int(meta.get("format_version", 0)) > FORMAT_VERSION:
            raise ValueError(f"{path} needs a newer openfly (format v{meta['format_version']})")
        neurons = pd.read_parquet(io.BytesIO(z.read("neurons.parquet")))
        edges = pd.read_parquet(io.BytesIO(z.read("edges.parquet")))
    n = len(neurons)
    W = sparse.csr_matrix(
        (edges["weight"].to_numpy(np.float32), (edges["pre"].to_numpy(), edges["post"].to_numpy())), shape=(n, n)
    )
    meta["file"] = str(path)
    return Brain(neurons, W, meta)


def _to_parquet(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
