"""
Deterministic synthetic market data provider.

This exists so the whole pipeline (data -> analysis -> strategy -> risk ->
execution -> portfolio -> database) can be developed, tested, and run in
Docker with zero external dependencies and zero API keys. It is NOT meant
to resemble real market microstructure - it is a seeded random walk with
mild trend/mean-reversion so indicators have something meaningful to
compute on.

Because it is seeded per-symbol, repeated runs with the same seed produce
identical price series, which makes tests and backtests reproducible.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from core.models import Candle, MarketSnapshot
from data.base import MarketDataProvider


class SyntheticMarketDataProvider(MarketDataProvider):
    def __init__(self, symbols: List[str], seed: int = 42, start_price: float = 150.0):
        self._symbols = list(symbols)
        self._seed = seed
        self._start_price = start_price
        self._cache: Dict[str, List[Candle]] = {}

    def _generate(self, symbol: str, start: datetime, end: datetime, timeframe: str) -> List[Candle]:
        rng = random.Random(f"{self._seed}:{symbol}")
        step = _timeframe_to_timedelta(timeframe)

        candles: List[Candle] = []
        price = self._start_price + (abs(hash(symbol)) % 100)
        trend = rng.uniform(-0.0003, 0.0006)
        ts = start
        while ts <= end:
            if _is_tradeable(ts):
                daily_vol = rng.uniform(0.006, 0.02)
                open_px = price
                drift = trend + rng.gauss(0, daily_vol)
                close_px = max(0.5, open_px * (1 + drift))
                high_px = max(open_px, close_px) * (1 + abs(rng.gauss(0, daily_vol / 2)))
                low_px = min(open_px, close_px) * (1 - abs(rng.gauss(0, daily_vol / 2)))
                volume = int(rng.uniform(5e5, 5e6))
                candles.append(
                    Candle(
                        symbol=symbol,
                        timestamp=ts,
                        open=round(open_px, 2),
                        high=round(high_px, 2),
                        low=round(low_px, 2),
                        close=round(close_px, 2),
                        volume=volume,
                        timeframe=timeframe,
                    )
                )
                price = close_px
            ts += step
        return candles

    def get_historical_candles(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> List[Candle]:
        cache_key = f"{symbol}:{timeframe}"
        if cache_key not in self._cache:
            # Generate a wide buffer so repeated calls with different windows
            # are consistent with each other (same underlying path).
            buffer_start = start - timedelta(days=400)
            self._cache[cache_key] = self._generate(symbol, buffer_start, end + timedelta(days=1), timeframe)
        return [c for c in self._cache[cache_key] if start <= c.timestamp <= end]

    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        now = datetime.now(timezone.utc)
        candles = self.get_historical_candles(symbol, now - timedelta(days=5), now, "1Day")
        if not candles:
            # Weekend/holiday with no bar yet today - synthesize one on the fly.
            candles = self.get_historical_candles(symbol, now - timedelta(days=10), now, "1Day")
        last = candles[-1]
        return MarketSnapshot(
            symbol=symbol,
            timestamp=now,
            price=last.close,
            volume=last.volume,
            open=last.open,
            high=last.high,
            low=last.low,
            close=last.close,
        )


def _is_tradeable(ts: datetime) -> bool:
    return ts.weekday() < 5  # skip weekends; good enough for a paper simulator


def _timeframe_to_timedelta(timeframe: str) -> timedelta:
    mapping = {
        "1Min": timedelta(minutes=1),
        "5Min": timedelta(minutes=5),
        "15Min": timedelta(minutes=15),
        "1Hour": timedelta(hours=1),
        "1Day": timedelta(days=1),
    }
    if timeframe not in mapping:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return mapping[timeframe]
