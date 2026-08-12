from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import main
from core.config import AppConfig, DatabaseConfig, DataConfig
from core.models import MarketSnapshot
from data.market_calendar import AlwaysOpenCalendar
from database.database import DatabaseManager
from portfolio.portfolio import Portfolio

NOW = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_build_market_calendar_auto_picks_alpaca_when_provider_is_alpaca(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    from data.market_calendar import AlpacaMarketCalendar

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeClient)

    config = AppConfig(data=DataConfig(provider="alpaca", market_calendar="auto"))
    cal = main.build_market_calendar(config)
    assert isinstance(cal, AlpacaMarketCalendar)


def test_build_market_calendar_auto_picks_always_open_for_synthetic():
    config = AppConfig(data=DataConfig(provider="synthetic", market_calendar="auto"))
    cal = main.build_market_calendar(config)
    assert isinstance(cal, AlwaysOpenCalendar)


def test_build_market_calendar_explicit_override_bypasses_auto():
    config = AppConfig(data=DataConfig(provider="alpaca", market_calendar="always_open"))
    cal = main.build_market_calendar(config)
    assert isinstance(cal, AlwaysOpenCalendar)


class _FakeMarketData:
    def __init__(self, prices):
        self._prices = prices

    def get_snapshot(self, symbol):
        return MarketSnapshot(symbol=symbol, timestamp=NOW, price=self._prices[symbol], volume=1000)


class _FarFutureCalendar:
    def next_open(self, now):
        return now + timedelta(hours=10)  # longer than max_closed_market_heartbeat_seconds


class _SoonCalendar:
    def next_open(self, now):
        return now + timedelta(minutes=5)  # shorter than the cap


def test_heartbeat_while_closed_marks_portfolio_without_trading(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    portfolio = Portfolio(100_000.0)
    config = AppConfig(symbols=["AAPL"])

    sleep_seconds = main._heartbeat_while_closed(
        config, db, _FakeMarketData({"AAPL": 150.0}), portfolio, AlwaysOpenCalendar(), NOW
    )

    history = db.get_portfolio_history(limit=10)
    assert len(history) == 1
    assert history[0]["total_value"] == pytest.approx(100_000.0)  # marked, but nothing traded
    assert sleep_seconds > 0


def test_heartbeat_caps_sleep_duration_at_the_configured_maximum(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    portfolio = Portfolio(100_000.0)
    config = AppConfig(symbols=["AAPL"])

    sleep_seconds = main._heartbeat_while_closed(
        config, db, _FakeMarketData({"AAPL": 150.0}), portfolio, _FarFutureCalendar(), NOW
    )

    assert sleep_seconds == config.max_closed_market_heartbeat_seconds


def test_heartbeat_does_not_cap_a_short_wait(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    portfolio = Portfolio(100_000.0)
    config = AppConfig(symbols=["AAPL"])

    sleep_seconds = main._heartbeat_while_closed(
        config, db, _FakeMarketData({"AAPL": 150.0}), portfolio, _SoonCalendar(), NOW
    )

    assert sleep_seconds == pytest.approx(300, abs=1)  # 5 minutes, well under the cap


def test_heartbeat_survives_a_data_outage(tmp_path):
    """If the data provider is unreachable while the market's closed, the
    heartbeat must not crash the live loop - it should just skip marking
    that symbol and keep going."""

    class _BrokenMarketData:
        def get_snapshot(self, symbol):
            raise RuntimeError("simulated outage")

    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    portfolio = Portfolio(100_000.0)
    config = AppConfig(symbols=["AAPL"])

    sleep_seconds = main._heartbeat_while_closed(
        config, db, _BrokenMarketData(), portfolio, AlwaysOpenCalendar(), NOW
    )
    assert sleep_seconds > 0  # did not raise
