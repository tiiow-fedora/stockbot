"""
MarketDataService: the thin orchestration layer between a raw
MarketDataProvider and the database.

Strategies never talk to this directly during a backtest (the backtester
hands them plain lists of Candle objects), but the live `main.py run` loop
uses it to fetch the latest snapshot + warm-up history and persist
everything it sees, so every decision is reproducible later.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from core.logging import get_logger
from core.models import Candle, MarketSnapshot
from data.base import MarketDataError, MarketDataProvider
from database.database import DatabaseManager

logger = get_logger("data.market_data")


class MarketDataService:
    def __init__(self, provider: MarketDataProvider, db: Optional[DatabaseManager] = None):
        self._provider = provider
        self._db = db

    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        snapshot = self._provider.get_snapshot(symbol)
        if self._db:
            self._db.save_snapshot(snapshot)
        return snapshot

    def get_history(self, symbol: str, lookback_days: int, timeframe: str = "1Day") -> List[Candle]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        candles = self._provider.get_historical_candles(symbol, start, end, timeframe)
        if self._db and candles:
            self._db.save_candles(candles)
        return candles

    def get_history_range(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> List[Candle]:
        candles = self._provider.get_historical_candles(symbol, start, end, timeframe)
        if self._db and candles:
            self._db.save_candles(candles)
        return candles


def completed_bars(candles: List[Candle], as_of: datetime, timeframe: str = "1Day") -> List[Candle]:
    """Candles that are fully formed as of `as_of`.

    For daily bars, today's still-forming bar is timestamped at today's
    session start, i.e. *before* `as_of`, so a plain `timestamp < as_of`
    filter lets it through - duplicating the snapshot (which already
    represents today) and putting a partial-day volume into the history
    that full-day volumes get averaged against. Compare calendar dates
    for daily bars instead."""
    if timeframe == "1Day":
        return [c for c in candles if c.timestamp.date() < as_of.date()]
    return [c for c in candles if c.timestamp < as_of]
