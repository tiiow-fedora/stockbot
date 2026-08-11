from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from core.models import Candle


def make_candles(
    symbol: str = "TEST",
    n: int = 60,
    start_price: float = 100.0,
    daily_return: float = 0.0,
    start: datetime = None,
) -> List[Candle]:
    """Deterministic helper for building a clean OHLCV series for tests.
    `daily_return` applied compounding lets tests build steady uptrends/
    downtrends without randomness."""
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles = []
    price = start_price
    ts = start
    for i in range(n):
        open_px = price
        close_px = price * (1 + daily_return)
        high_px = max(open_px, close_px) * 1.001
        low_px = min(open_px, close_px) * 0.999
        candles.append(
            Candle(
                symbol=symbol,
                timestamp=ts,
                open=round(open_px, 4),
                high=round(high_px, 4),
                low=round(low_px, 4),
                close=round(close_px, 4),
                volume=1_000_000 + i * 1000,
            )
        )
        price = close_px
        ts += timedelta(days=1)
    return candles


@pytest.fixture
def uptrend_candles() -> List[Candle]:
    return make_candles(daily_return=0.01, n=80)


@pytest.fixture
def downtrend_candles() -> List[Candle]:
    return make_candles(daily_return=-0.01, n=80)


@pytest.fixture
def flat_candles() -> List[Candle]:
    return make_candles(daily_return=0.0, n=80)
