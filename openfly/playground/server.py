"""The playground web server (standard library only).

    openfly playground                     # toy brain, opens your browser
    openfly playground --dataset flywire_783 --transmitters curated

or from Python::

    from openfly.playground import serve
    serve("toy", port=8765)

Endpoints (all JSON unless noted):

=====================  ======================================================
``GET  /``              the single-page app (``static/index.html``)
``GET  /api/info``      brain metadata, palettes, scenarios, lessons...
``GET  /api/geometry``  binary: float32 positions (N x 3) + uint8 codes (N x 3)
``GET  /api/stream``    Server-Sent Events: one JSON frame per simulation chunk
``POST /api/command``   ``{"cmd": ..., ...}`` (see ``Session._cmd_*``)
``GET  /api/job/<id>``  status / result of a long-running experiment
=====================  ======================================================
"""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .session import Session

STATIC = Path(__file__).parent / "static"


def make_handler(session: Session):
    class Handler(BaseHTTPRequestHandler):
        server_version = "openfly"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json", cache: bool = False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, default=_default).encode())

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path.startswith("/static/"):
                    f = (STATIC / path[len("/static/"):]).resolve()
                    if STATIC.resolve() not in f.parents or not f.is_file():
                        return self._send(404, b"not found", "text/plain")
                    ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                    if f.suffix == ".js":
                        ctype = "text/javascript"
                    return self._send(200, f.read_bytes(), ctype)
                if path == "/api/info":
                    return self._json(session.info())
                if path == "/api/geometry":
                    return self._send(200, session.geometry(), "application/octet-stream")
                if path.startswith("/api/job/"):
                    job = session.jobs.get(int(path.rsplit("/", 1)[-1]))
                    return self._json(job or {"status": "error", "result": {"error": "no such job"}})
                if path == "/api/stream":
                    return self._stream()
                return self._send(404, b"not found", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/command":
                return self._send(404, b"not found", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                cmd = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json({"ok": False, "error": "invalid JSON"}, 400)
            return self._json(session.command(cmd))

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = session.subscribe()
            try:
                while True:
                    try:
                        frame = q.get(timeout=10)
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(b"data: " + json.dumps(frame, separators=(",", ":"), default=_default).encode() + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                session.unsubscribe(q)

    return Handler


def _default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Playground:
    """A running playground (server + live session). Use :func:`serve` or :func:`start`."""

    def __init__(self, brain, host: str = "127.0.0.1", port: int = 8765, chunk_ms: float = 20.0):
        self.session = Session(brain, chunk_ms=chunk_ms).start()
        self.httpd = ThreadingHTTPServer((host, port), make_handler(self.session))
        self.httpd.daemon_threads = True
        self.url = f"http://{host if host not in ('0.0.0.0', '') else 'localhost'}:{self.httpd.server_address[1]}/"
        self._thread = None

    def start(self) -> "Playground":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.session.stop()
        self.httpd.shutdown()
        self.httpd.server_close()


def _brain(brain_or_name, transmitters):
    if isinstance(brain_or_name, str):
        from ..data import load

        return load(brain_or_name, transmitters=transmitters)
    if transmitters:
        from .. import transmitters as tx

        return tx.apply(brain_or_name, transmitters)
    return brain_or_name


def start(brain="toy", host: str = "127.0.0.1", port: int = 8765, transmitters=None, chunk_ms: float = 20.0) -> Playground:
    """Start the playground in background threads and return it (handy in notebooks)."""
    return Playground(_brain(brain, transmitters), host, port, chunk_ms).start()


def serve(brain="toy", host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, transmitters=None,
          chunk_ms: float = 20.0) -> None:
    """Run the playground until Ctrl+C."""
    pg = Playground(_brain(brain, transmitters), host, port, chunk_ms)
    print(f"openfly playground: {pg.session.brain} -> {pg.url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(pg.url)).start()
    try:
        pg.httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        pg.stop()
