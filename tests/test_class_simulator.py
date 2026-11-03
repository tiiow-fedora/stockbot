"""Milestone 6: classroom-simulator adapter (file contract, no network)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import pytest

from core.config import AppConfig
from core.models import Order, OrderSide
from data.base import MarketDataError
from data.providers.class_simulator import ClassSimulatorMarketDataProvider
from execution.base import ExecutionError
from execution.class_simulator import ClassSimulatorExecutor
from main import build_executor, build_market_data_provider

TS = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)


def _provider(tmp_path, payload):
    q = tmp_path / "q.json"
    q.write_text(json.dumps(payload))
    return ClassSimulatorMarketDataProvider(str(q), str(tmp_path))


def test_snapshot_reads_price_volume_and_as_of(tmp_path):
    p = _provider(tmp_path, {"as_of": "2026-11-03T15:00:00Z",
                             "quotes": {"AAPL": {"price": 187.42, "volume": 31_250_000}}})
    s = p.get_snapshot("AAPL")
    assert (s.price, s.volume, s.timestamp) == (187.42, 31_250_000, TS)


def test_missing_volume_is_zero_unavailable_and_quote_ts_wins(tmp_path):
    p = _provider(tmp_path, {"as_of": "2026-01-01T00:00:00Z",
                             "quotes": {"MSFT": {"price": 411.1, "timestamp": "2026-11-03T15:00:00Z"}}})
    s = p.get_snapshot("MSFT")
    assert s.volume == 0 and s.timestamp == TS


def test_missing_symbol_file_and_bad_quote_raise_market_data_error(tmp_path):
    p = _provider(tmp_path, {"quotes": {"AAPL": {"price": -1}}})
    with pytest.raises(MarketDataError):
        p.get_snapshot("AAPL")
    with pytest.raises(MarketDataError):
        p.get_snapshot("NOPE")
    with pytest.raises(MarketDataError):
        ClassSimulatorMarketDataProvider(str(tmp_path / "missing.json"), str(tmp_path)).get_snapshot("AAPL")


def test_history_comes_from_csv(tmp_path):
    (tmp_path / "AAPL.csv").write_text(
        "timestamp,open,high,low,close,volume\n2026-11-02T00:00:00Z,10,11,9,10,1000\n")
    p = _provider(tmp_path, {"quotes": {}})
    c = p.get_historical_candles("AAPL", datetime(2026, 1, 1, tzinfo=timezone.utc), TS)
    assert len(c) == 1 and c[0].volume == 1000


def test_executor_writes_whole_share_ticket_and_does_not_fill(tmp_path):
    ex = ClassSimulatorExecutor(str(tmp_path / "out" / "outbox.csv"))
    o = Order("AAPL", OrderSide.BUY, 12, timestamp=TS, reason="r", confidence=0.5)
    assert ex.execute(o, 187.42, TS) is None
    rows = list(csv.DictReader(open(tmp_path / "out" / "outbox.csv")))
    assert len(rows) == 1
    assert (rows[0]["side"], rows[0]["symbol"], rows[0]["quantity"], rows[0]["reference_price"]) == (
        "BUY", "AAPL", "12", "187.4200")


def test_executor_is_idempotent_across_restarts(tmp_path):
    path = str(tmp_path / "o.csv")
    o = Order("AAPL", OrderSide.SELL, 5, timestamp=TS)
    ClassSimulatorExecutor(path).execute(o, 100.0, TS)
    ClassSimulatorExecutor(path).execute(o, 100.0, TS)
    assert len(list(csv.DictReader(open(path)))) == 1


def test_executor_rejects_fractional_shares_and_bad_price(tmp_path):
    ex = ClassSimulatorExecutor(str(tmp_path / "o.csv"))
    with pytest.raises(ExecutionError):
        ex.execute(Order("AAPL", OrderSide.BUY, 1.5, timestamp=TS), 100.0, TS)
    with pytest.raises(ExecutionError):
        ex.execute(Order("AAPL", OrderSide.BUY, 1, timestamp=TS), 0.0, TS)


def test_factories_wire_class_simulator(tmp_path):
    cfg = AppConfig(data={"provider": "class_simulator", "class_simulator_quotes_path": str(tmp_path / "q.json")},
                    execution={"mode": "class_simulator", "class_simulator_outbox": str(tmp_path / "o.csv")})
    assert isinstance(build_market_data_provider(cfg), ClassSimulatorMarketDataProvider)
    assert isinstance(build_executor(cfg), ClassSimulatorExecutor)
