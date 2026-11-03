"""Graceful shutdown on SIGTERM/SIGINT.

docker stop sends SIGTERM; without a handler Python dies mid-cycle. Here the
handler sets a flag. If the loop is mid-cycle it finishes the cycle (so a
fill is never half-recorded) and exits at the next check; if it is asleep,
the handler raises KeyboardInterrupt to wake it immediately.
"""

from __future__ import annotations

import contextlib
import signal
import threading


class GracefulShutdown:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._sleeping = False
        self._previous: dict = {}

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def request(self) -> None:
        self._event.set()

    def _handle(self, signum, frame) -> None:
        self._event.set()
        if self._sleeping:
            raise KeyboardInterrupt

    @contextlib.contextmanager
    def interruptible(self):
        """Wrap a blocking sleep so a signal wakes it at once."""
        if self.requested:
            raise KeyboardInterrupt
        self._sleeping = True
        try:
            yield
        finally:
            self._sleeping = False

    def install(self) -> "GracefulShutdown":
        if threading.current_thread() is not threading.main_thread():
            return self  # signals can only be installed from the main thread
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, self._handle)
        return self

    def uninstall(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for sig, prev in self._previous.items():
            signal.signal(sig, prev)
        self._previous.clear()
