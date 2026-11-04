"""Shared fixtures for the chaos suite. Everything here is offline and
paper-only: sockets are blocked so a test can never reach a real broker."""

from __future__ import annotations

import socket
from datetime import datetime, timezone

import pytest

from core.config import AppConfig, DatabaseConfig, DataConfig, RiskConfig
from database.database import DatabaseManager
from tests.chaos.fake_alpaca import FakeAlpacaBroker, install
from tests.conftest import make_candles


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    real_connect = socket.socket.connect

    def guarded(self, address, *a, **k):
        # AF_UNIX / loopback only; anything else is a test bug.
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"chaos test attempted a real network connection to {address!r}")
        return real_connect(self, address, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture
def broker(monkeypatch) -> FakeAlpacaBroker:
    b = FakeAlpacaBroker()
    install(monkeypatch, b)
    yield b
    assert b.live_attempts == 0, "something tried to build a non-paper trading client"


@pytest.fixture
def db(tmp_path) -> DatabaseManager:
    d = DatabaseManager(f"sqlite:///{tmp_path}/chaos.db")
    d.init_db()
    return d


@pytest.fixture
def db_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path}/chaos.db"


def make_config(db_url: str, symbols=("AAPL",), **risk) -> AppConfig:
    return AppConfig(
        symbols=list(symbols),
        log_file=None,
        data=DataConfig(provider="alpaca", market_calendar="alpaca", history_lookback_days=180),
        database=DatabaseConfig(url=db_url),
        risk=RiskConfig(**risk),
        strategy={"name": "mean_reversion"},
    )


def dip_candles(symbol="AAPL", n=80, end=None, crash=0.12):
    """A flat-then-sharp-drop series: reliably oversold, so mean-reversion
    emits a BUY. Used as the 'liveness' fixture - if a unit/scale bug
    silently suppresses BUYs, tests built on this series fail."""
    from datetime import timedelta

    end = end or datetime(2024, 3, 11, tzinfo=timezone.utc)
    start = end - timedelta(days=n - 1)
    candles = make_candles(symbol=symbol, n=n, start_price=100.0, daily_return=0.0, start=start)
    # last 6 bars fall `crash` total
    from core.models import Candle

    out = candles[: n - 6]
    price = 100.0
    step = (1 - crash) ** (1 / 6)
    for i in range(6):
        o, c = price, price * step
        out.append(Candle(symbol, candles[n - 6 + i].timestamp, round(o, 4), round(o * 1.001, 4),
                          round(c * 0.999, 4), round(c, 4), 1_500_000))
        price = c
    return out
