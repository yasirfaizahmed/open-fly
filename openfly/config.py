"""Where openfly keeps its files.

Everything openfly downloads or builds lives under one folder:

* ``$OPENFLY_HOME`` if that environment variable is set, otherwise
* ``~/.openfly``

Layout::

    ~/.openfly/
        raw/<dataset>/...        original files exactly as downloaded
        brains/<dataset>.flybrain the compiled, ready-to-load brain file
        assets/...               meshes and other optional extras
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    """Root folder for all openfly data (created on first use)."""
    root = Path(os.environ.get("OPENFLY_HOME", Path.home() / ".openfly")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def subdir(*parts: str) -> Path:
    """``home()/parts...``, created if needed."""
    p = home().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir(dataset: str) -> Path:
    return subdir("raw", dataset.replace("/", "_").replace(":", "_"))


def brains_dir() -> Path:
    return subdir("brains")


def assets_dir() -> Path:
    return subdir("assets")
