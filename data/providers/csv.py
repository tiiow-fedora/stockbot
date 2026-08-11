"""
CSV market data provider.

Reads historical OHLCV data from `{csv_dir}/{SYMBOL}.csv` with columns:

    timestamp,open,high,low,close,volume

`timestamp` must be ISO-8601 and either timezone-aware or naive UTC (naive
values are assumed to be UTC). This provider is read-only and has no
concept of "latest" beyond the last row in the file, which makes it a
good fit for backtesting against real historical data you've downloaded
in advance, without touching any network API.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from core.models import Candle, MarketSnapshot
from data.base import MarketDataError, MarketDataProvider


class CSVMarketDataProvider(MarketDataProvider):
    def __init__(self, csv_dir: str):
        self._dir = Path(csv_dir)
        self._cache: Dict[str, List[Candle]] = {}

    def _load(self, symbol: str, timeframe: str) -> List[Candle]:
        if symbol in self._cache:
            return self._cache[symbol]

        path = self._dir / f"{symbol}.csv"
        if not path.exists():
            raise MarketDataError(f"no CSV file found for {symbol} at {path}")

        candles: List[Candle] = []
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required.issubset(reader.fieldnames or []):
                raise MarketDataError(f"{path} missing required columns {required}")
            for row in reader:
                ts = _parse_ts(row["timestamp"])
                candles.append(
                    Candle(
                        symbol=symbol,
                        timestamp=ts,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(float(row["volume"])),
                        timeframe=timeframe,
                    )
                )
        candles.sort(key=lambda c: c.timestamp)
        self._cache[symbol] = candles
        return candles

    def get_historical_candles(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> List[Candle]:
        candles = self._load(symbol, timeframe)
        return [c for c in candles if start <= c.timestamp <= end]

    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        candles = self._load(symbol, "1Day")
        if not candles:
            raise MarketDataError(f"no data available for {symbol}")
        last = candles[-1]
        return MarketSnapshot(
            symbol=symbol,
            timestamp=last.timestamp,
            price=last.close,
            volume=last.volume,
            open=last.open,
            high=last.high,
            low=last.low,
            close=last.close,
        )


def _parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts
