"""Small, dependency-free downloader with progress, mirrors and checksums.

Uses only the standard library (``urllib``), which honours the usual
``HTTPS_PROXY`` / ``SSL_CERT_FILE`` environment variables, so it works
behind corporate proxies too.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


def _fmt_mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(
    urls: str | list[str],
    dest: str | Path,
    sha256: str | None = None,
    progress: bool = True,
    retries: int = 3,
) -> Path:
    """Download the first URL that works into ``dest``.

    The file is written to ``dest.part`` and renamed only when complete,
    so an interrupted download never leaves a corrupt file behind. If
    ``dest`` already exists (and matches ``sha256`` when given) nothing is
    downloaded.
    """
    dest = Path(dest)
    if isinstance(urls, str):
        urls = [urls]
    if dest.exists() and (sha256 is None or sha256sum(dest) == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        for attempt in range(retries):
            try:
                _download_one(url, dest, progress)
                if sha256 is not None and sha256sum(dest) != sha256:
                    dest.unlink(missing_ok=True)
                    raise DownloadError(f"checksum mismatch for {url}")
                return dest
            except (urllib.error.URLError, OSError, DownloadError) as exc:
                errors.append(f"{url}: {exc}")
                time.sleep(min(2**attempt, 8))
    raise DownloadError(
        "Could not download any of:\n  " + "\n  ".join(urls) + "\nErrors:\n  " + "\n  ".join(errors[-len(urls) :])
    )


def _download_one(url: str, dest: Path, progress: bool) -> None:
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "openfly"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        t0 = time.time()
        show = progress and sys.stderr.isatty() and os.environ.get("OPENFLY_QUIET") is None
        label = dest.name
        while True:
            block = resp.read(_CHUNK)
            if not block:
                break
            out.write(block)
            done += len(block)
            if show:
                rate = done / max(time.time() - t0, 1e-6)
                if total:
                    pct = 100 * done / total
                    bar = "#" * int(pct // 4)
                    sys.stderr.write(f"\r  {label}: [{bar:<25}] {pct:5.1f}% {_fmt_mb(done)} ({_fmt_mb(rate)}/s)")
                else:
                    sys.stderr.write(f"\r  {label}: {_fmt_mb(done)} ({_fmt_mb(rate)}/s)")
                sys.stderr.flush()
        if show:
            sys.stderr.write("\n")
    shutil.move(tmp, dest)
