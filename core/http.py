"""Default network timeout for alpaca-py clients.

alpaca-py builds a plain `requests.Session` and never passes `timeout=` to
`session.request`, so a connection that goes quiet (half-open TCP, a stalled
proxy, a dropped NAT entry) blocks the calling thread FOREVER. In StockBot
that thread is the single live-trading loop, which would simply freeze:
no exits, no heartbeat, nothing in the logs. The loop's retry/backoff logic
(main.run_live_loop) only helps if the call actually fails.
"""

from __future__ import annotations

from typing import Any

# (connect, read) seconds. Alpaca's REST API normally answers well under 1s;
# 20s read leaves generous headroom without letting a stall hang the bot.
DEFAULT_TIMEOUT = (5.0, 20.0)


def apply_default_timeout(client: Any, timeout=DEFAULT_TIMEOUT) -> Any:
    """Make every request made through `client` carry a timeout unless the
    caller already supplied one. Returns `client`. No-op for objects without
    an alpaca-py style `_session` (e.g. test fakes)."""
    session = getattr(client, "_session", None)
    if session is None or getattr(session, "_stockbot_timeout_applied", False):
        return client
    original = session.request

    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original(method, url, **kwargs)

    session.request = request_with_timeout
    session._stockbot_timeout_applied = True
    return client
