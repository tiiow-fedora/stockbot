"""
Abstract data interfaces.

Every concrete provider (Alpaca, CSV, synthetic, or a future classroom
platform adapter) implements one of these two interfaces. Nothing above
this layer (analysis, strategy, portfolio, execution) is allowed to know
which concrete provider is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from core.models import Candle, MarketSnapshot, NewsItem


class MarketDataError(Exception):
    """Raised when a provider cannot supply data (outage, bad symbol, etc.)."""


class MarketDataProvider(ABC):
    """Source of live/latest quotes and historical OHLCV candles."""

    @abstractmethod
    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        """Return the latest known price for `symbol`."""

    @abstractmethod
    def get_historical_candles(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> List[Candle]:
        """Return candles for `symbol` in [start, end], oldest first."""

    def get_snapshots(self, symbols: List[str]) -> List[MarketSnapshot]:
        """Default implementation just loops; providers with a batch API
        (e.g. Alpaca) should override this for efficiency."""
        return [self.get_snapshot(s) for s in symbols]


class NewsDataProvider(ABC):
    """Source of news headlines, optionally with sentiment already scored."""

    @abstractmethod
    def get_news(self, symbol: str, start: datetime, end: datetime) -> List[NewsItem]:
        """Return news items for `symbol` in [start, end], oldest first."""


class NullNewsProvider(NewsDataProvider):
    """Used whenever no news source is configured (the default). The
    strategy must work perfectly with this provider - it always returns
    an empty list, so sentiment scoring is simply skipped."""

    def get_news(self, symbol: str, start: datetime, end: datetime) -> List[NewsItem]:
        return []
