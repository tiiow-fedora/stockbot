"""
Classroom-simulator market data adapter (Milestone 6).

The classroom platform's API is unknown, so this adapter deliberately
binds to a *file contract* rather than a vendor API: the platform's
quotes are exported (by hand, a script, or a future bridge) to a small
JSON file, and history comes from per-symbol CSVs (same format as
`CSVMarketDataProvider`). Nothing here touches the network or any
credentials.

Quotes file (`data.class_simulator_quotes_path`):

    {
      "as_of": "2026-11-03T15:30:00Z",      # optional default timestamp
      "quotes": {
        "AAPL": {"price": 187.42, "volume": 31250000, "timestamp": "..."},
        "MSFT": {"price": 411.10}
      }
    }

UNITS (the contract - this is where unit bugs hide, so it is explicit):
  - `price`  : dollars per share (> 0).
  - `volume` : SHARES traded so far for the current day, the same unit as
               the history CSV `volume` column. Omit it if the platform
               does not show volume; a snapshot with volume 0 is treated
               by the strategies as "volume unavailable", not "volume
               collapsed".
A quote without a timestamp uses `as_of`, else the file's mtime (UTC).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from core.models import Candle, MarketSnapshot
from data.base import MarketDataError, MarketDataProvider
from data.providers.csv import CSVMarketDataProvider, _parse_ts


class ClassSimulatorMarketDataProvider(MarketDataProvider):
    def __init__(self, quotes_path: str, history_dir: str):
        self._quotes_path = Path(quotes_path)
        self._history = CSVMarketDataProvider(csv_dir=history_dir)

    def get_historical_candles(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1Day"
    ) -> List[Candle]:
        return self._history.get_historical_candles(symbol, start, end, timeframe)

    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        if not self._quotes_path.exists():
            raise MarketDataError(f"class simulator quotes file not found: {self._quotes_path}")
        try:
            raw = json.loads(self._quotes_path.read_text(encoding="utf-8"))
            quotes = raw["quotes"]
        except (ValueError, KeyError, TypeError) as exc:
            raise MarketDataError(f"malformed quotes file {self._quotes_path}: {exc}") from exc

        quote = quotes.get(symbol)
        if quote is None:
            raise MarketDataError(f"no quote for {symbol} in {self._quotes_path}")

        try:
            price = float(quote["price"])
            volume = int(float(quote.get("volume", 0) or 0))
            ts_raw = quote.get("timestamp") or raw.get("as_of")
            ts = (
                _parse_ts(ts_raw)
                if ts_raw
                else datetime.fromtimestamp(self._quotes_path.stat().st_mtime, tz=timezone.utc)
            )
            return MarketSnapshot(
                symbol=symbol, timestamp=ts, price=price, volume=volume, close=price
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise MarketDataError(f"invalid quote for {symbol}: {exc}") from exc
