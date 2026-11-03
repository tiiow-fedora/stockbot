"""
Market calendar: is the market open right now, and when does it open
next?

Kept as its own small interface, separate from MarketDataProvider,
because "is the market open" is a fundamentally different question from
"what's the price" - a classroom platform might trade on its own
schedule that has nothing to do with NYSE hours, or might have no
concept of market hours at all. `AlwaysOpenCalendar` is the default for
exactly that reason: synthetic/CSV data (and, until we know otherwise,
a future classroom platform) doesn't need this at all.

Why this exists: the local paper simulator has no concept of trading
hours on its own - if the live loop asked it to fill an order at 2am, it
would happily do so instantly, at a stale price, which is not what a
real trading day would produce. The live loop uses a MarketCalendar to
decide whether to run the strategy/order pipeline at all; the executor
itself stays unaware of time-of-day, same as it stays unaware of
everything else about where prices came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Optional

from core.http import apply_default_timeout
from core.logging import get_logger
from data.base import MarketDataError

logger = get_logger("data.market_calendar")


class MarketCalendar(ABC):
    @abstractmethod
    def is_open(self, timestamp: datetime) -> bool:
        """Is the market open at `timestamp` (must be timezone-aware)?"""

    @abstractmethod
    def next_open(self, timestamp: datetime) -> datetime:
        """The next time the market opens at or after `timestamp`. If the
        market is already open at `timestamp`, implementations may return
        `timestamp` itself."""


class AlwaysOpenCalendar(MarketCalendar):
    """The default. Correct for synthetic/CSV data (no real trading
    session to respect) and for any future platform with no concept of
    market hours."""

    def is_open(self, timestamp: datetime) -> bool:
        return True

    def next_open(self, timestamp: datetime) -> datetime:
        return timestamp


class USEquityMarketCalendar(MarketCalendar):
    """A deterministic, offline approximation of NYSE/Nasdaq hours:
    09:30-16:00 America/New_York, Monday-Friday.

    Does NOT account for market holidays (Thanksgiving, etc.) - there's
    no simple offline source for that which won't silently drift out of
    date. Use AlpacaMarketCalendar instead when you have Alpaca
    credentials and want exact accuracy including holidays. This class
    exists for when you don't (e.g. developing against `synthetic` data
    but still wanting realistic session gating)."""

    OPEN = dt_time(9, 30)
    CLOSE = dt_time(16, 0)

    def __init__(self, tz_name: str = "America/New_York"):
        from zoneinfo import ZoneInfo

        self._tz = ZoneInfo(tz_name)

    def is_open(self, timestamp: datetime) -> bool:
        local = timestamp.astimezone(self._tz)
        if local.weekday() >= 5:  # Saturday/Sunday
            return False
        return self.OPEN <= local.time() < self.CLOSE

    def next_open(self, timestamp: datetime) -> datetime:
        local = timestamp.astimezone(self._tz)

        if local.weekday() < 5 and local.time() < self.OPEN:
            candidate = local.replace(hour=self.OPEN.hour, minute=self.OPEN.minute, second=0, microsecond=0)
        else:
            candidate = (local + timedelta(days=1)).replace(
                hour=self.OPEN.hour, minute=self.OPEN.minute, second=0, microsecond=0
            )
            while candidate.weekday() >= 5:
                candidate += timedelta(days=1)

        return candidate.astimezone(timezone.utc)


class AlpacaMarketCalendar(MarketCalendar):
    """Exact market hours, including real holidays, via Alpaca's own
    clock endpoint. Only usable with ALPACA_API_KEY/ALPACA_SECRET_KEY.

    Alpaca's clock reflects the CURRENT moment server-side - it has no
    way to answer "was the market open at some arbitrary past/future
    timestamp." That's fine here because the live loop only ever asks
    "is it open right now," but it does mean the `timestamp` argument is
    ignored; this is documented rather than silently wrong."""

    def __init__(self):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover
            raise MarketDataError(
                "alpaca-py is not installed. Run `pip install alpaca-py` "
                "or set data.market_calendar: us_equity or always_open in config.yaml."
            ) from exc

        api_key, secret_key = _get_alpaca_credentials()
        self._client = apply_default_timeout(TradingClient(api_key, secret_key, paper=True))

    def is_open(self, timestamp: datetime) -> bool:
        return bool(self._client.get_clock().is_open)

    def next_open(self, timestamp: datetime) -> datetime:
        return self._client.get_clock().next_open


def _get_alpaca_credentials() -> tuple:
    import os

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise MarketDataError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set - required for "
            "data.market_calendar: alpaca. Set them in .env, or use "
            "us_equity/always_open instead."
        )
    return api_key, secret_key


def build_market_calendar(name: str) -> MarketCalendar:
    if name == "always_open":
        return AlwaysOpenCalendar()
    if name == "us_equity":
        return USEquityMarketCalendar()
    if name == "alpaca":
        return AlpacaMarketCalendar()
    raise ValueError(f"unknown market calendar: {name!r}")
