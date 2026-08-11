from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from core.models import Candle, MarketSnapshot
from data.base import MarketDataError
from data.providers.csv import CSVMarketDataProvider
from data.providers.synthetic import SyntheticMarketDataProvider
from database.database import DatabaseManager

UTC_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
NAIVE_NOW = datetime(2024, 1, 1)


# -- model validation (invalid / malformed data) -----------------------------

def test_market_snapshot_rejects_naive_datetime():
    with pytest.raises(ValueError):
        MarketSnapshot(symbol="AAPL", timestamp=NAIVE_NOW, price=100.0)


def test_market_snapshot_rejects_non_positive_price():
    with pytest.raises(ValueError):
        MarketSnapshot(symbol="AAPL", timestamp=UTC_NOW, price=0.0)
    with pytest.raises(ValueError):
        MarketSnapshot(symbol="AAPL", timestamp=UTC_NOW, price=-5.0)


def test_market_snapshot_rejects_negative_volume():
    with pytest.raises(ValueError):
        MarketSnapshot(symbol="AAPL", timestamp=UTC_NOW, price=100.0, volume=-1)


def test_candle_rejects_high_below_low():
    with pytest.raises(ValueError):
        Candle(symbol="AAPL", timestamp=UTC_NOW, open=100, high=90, low=95, close=98, volume=100)


def test_candle_rejects_non_positive_ohlc():
    with pytest.raises(ValueError):
        Candle(symbol="AAPL", timestamp=UTC_NOW, open=100, high=110, low=0, close=105, volume=100)


# -- duplicate data handling --------------------------------------------------

def test_database_ignores_duplicate_candles(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    candle = Candle(symbol="AAPL", timestamp=UTC_NOW, open=100, high=101, low=99, close=100.5, volume=1000)

    db.save_candles([candle])
    db.save_candles([candle])  # duplicate insert attempt

    stored = db.get_candles("AAPL", UTC_NOW - timedelta(days=1), UTC_NOW + timedelta(days=1))
    assert len(stored) == 1


def test_database_roundtrips_candle_fields(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    candle = Candle(symbol="AAPL", timestamp=UTC_NOW, open=100, high=101, low=99, close=100.5, volume=1000)
    db.save_candles([candle])

    stored = db.get_candles("AAPL", UTC_NOW - timedelta(days=1), UTC_NOW + timedelta(days=1))
    assert len(stored) == 1
    assert stored[0].open == pytest.approx(100)
    assert stored[0].close == pytest.approx(100.5)
    assert stored[0].volume == 1000


# -- missing data ---------------------------------------------------------------

def test_csv_provider_raises_clear_error_for_missing_file(tmp_path):
    provider = CSVMarketDataProvider(csv_dir=str(tmp_path))
    with pytest.raises(MarketDataError):
        provider.get_snapshot("NOSUCHSYMBOL")


def test_csv_provider_rejects_missing_columns(tmp_path):
    bad_csv = tmp_path / "AAPL.csv"
    bad_csv.write_text("timestamp,open,high,low,close\n2024-01-01T00:00:00Z,1,2,0.5,1.5\n")
    provider = CSVMarketDataProvider(csv_dir=str(tmp_path))
    with pytest.raises(MarketDataError):
        provider.get_snapshot("AAPL")


# -- synthetic provider determinism (needed for reproducible backtests) ---------

def test_synthetic_provider_is_deterministic_for_same_seed():
    p1 = SyntheticMarketDataProvider(symbols=["AAPL"], seed=7)
    p2 = SyntheticMarketDataProvider(symbols=["AAPL"], seed=7)
    start = UTC_NOW - timedelta(days=30)
    c1 = p1.get_historical_candles("AAPL", start, UTC_NOW, "1Day")
    c2 = p2.get_historical_candles("AAPL", start, UTC_NOW, "1Day")
    assert [c.close for c in c1] == [c.close for c in c2]


def test_synthetic_provider_differs_for_different_seeds():
    p1 = SyntheticMarketDataProvider(symbols=["AAPL"], seed=1)
    p2 = SyntheticMarketDataProvider(symbols=["AAPL"], seed=2)
    start = UTC_NOW - timedelta(days=30)
    c1 = p1.get_historical_candles("AAPL", start, UTC_NOW, "1Day")
    c2 = p2.get_historical_candles("AAPL", start, UTC_NOW, "1Day")
    assert [c.close for c in c1] != [c.close for c in c2]


def test_synthetic_provider_produces_no_weekend_bars():
    provider = SyntheticMarketDataProvider(symbols=["AAPL"], seed=3)
    start = UTC_NOW - timedelta(days=14)
    candles = provider.get_historical_candles("AAPL", start, UTC_NOW, "1Day")
    assert all(c.timestamp.weekday() < 5 for c in candles)


# -- market data outage resilience -------------------------------------------

class _FlakyProvider:
    """Raises for one symbol, works for the rest - simulates a partial outage."""

    def __init__(self, good_symbols: List[str], bad_symbol: str):
        self._good = SyntheticMarketDataProvider(symbols=good_symbols, seed=9)
        self._bad_symbol = bad_symbol

    def get_snapshot(self, symbol: str) -> MarketSnapshot:
        if symbol == self._bad_symbol:
            raise MarketDataError("simulated outage")
        return self._good.get_snapshot(symbol)

    def get_historical_candles(self, symbol, start, end, timeframe="1Day"):
        return self._good.get_historical_candles(symbol, start, end, timeframe)


def test_orchestrator_skips_symbol_on_outage_without_crashing():
    """Mirrors the try/except in main._run_cycle: one bad symbol must not
    prevent the others from being processed."""
    provider = _FlakyProvider(good_symbols=["AAPL", "MSFT"], bad_symbol="MSFT")
    prices = {}
    errors = []
    for symbol in ["AAPL", "MSFT"]:
        try:
            snapshot = provider.get_snapshot(symbol)
            prices[symbol] = snapshot.price
        except MarketDataError as exc:
            errors.append((symbol, str(exc)))

    assert "AAPL" in prices
    assert "MSFT" not in prices
    assert errors == [("MSFT", "simulated outage")]
