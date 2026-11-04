"""Health state and a tiny stdlib HTTP server (no extra deps).

Endpoints
  /healthz  liveness  - 200 while the loop has made progress recently
                        (successful cycle or closed-market heartbeat within
                        ``stale_after`` seconds), 503 once it is stuck.
  /readyz   readiness - 200 only after the first successful cycle AND the
                        database answers; 503 otherwise.
  /metrics  Prometheus text format.

The loop calls ``state.beat()`` on every successful iteration. The server
runs in a daemon thread and is read-only.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from observability import metrics


class HealthState:
    def __init__(self, stale_after: float, db_check: Optional[Callable[[], bool]] = None,
                 clock: Callable[[], float] = time.time):
        self.stale_after = stale_after
        self._db_check = db_check
        self._clock = clock
        self._started = clock()
        self._last_beat: Optional[float] = None
        self._lock = threading.Lock()
        self.shutting_down = False

    def beat(self) -> None:
        with self._lock:
            self._last_beat = self._clock()

    def liveness(self) -> tuple[bool, dict]:
        now = self._clock()
        with self._lock:
            last = self._last_beat
        # Before the first beat, allow one staleness window from process start.
        ref = last if last is not None else self._started
        age = now - ref
        ok = age <= self.stale_after
        return ok, {"status": "ok" if ok else "stale", "seconds_since_progress": round(age, 1),
                    "stale_after": self.stale_after, "shutting_down": self.shutting_down}

    def readiness(self) -> tuple[bool, dict]:
        with self._lock:
            started = self._last_beat is not None
        db_ok = True
        if self._db_check is not None:
            try:
                db_ok = bool(self._db_check())
            except Exception:
                db_ok = False
        ok = started and db_ok and not self.shutting_down
        return ok, {"status": "ready" if ok else "not_ready", "first_cycle_done": started,
                    "database": "ok" if db_ok else "down", "shutting_down": self.shutting_down}


def make_handler(state: HealthState):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/metrics":
                body, ctype = metrics.render()
                self._send(200, body, ctype)
            elif path in ("/healthz", "/readyz"):
                ok, detail = state.liveness() if path == "/healthz" else state.readiness()
                self._send(200 if ok else 503, json.dumps(detail).encode(), "application/json")
            else:
                self._send(404, b'{"error":"not found"}', "application/json")

        def log_message(self, *args):  # silence per-request stderr noise
            pass

    return Handler


def start_server(state: HealthState, port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="obs-http", daemon=True).start()
    return server
