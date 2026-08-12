from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from core.config import AppConfig, DatabaseConfig
from core.models import (
    Action,
    Decision,
    Fill,
    Order,
    OrderSide,
    OrderType,
    PortfolioSnapshot,
    ScoreBreakdown,
)
from dashboard.app import create_app
from database.database import DatabaseManager

NOW = datetime.now(timezone.utc)


@pytest.fixture
def config(tmp_path):
    return AppConfig(
        symbols=["AAPL", "MSFT"],
        starting_balance=100_000.0,
        polling_interval_seconds=60,
        database=DatabaseConfig(url=f"sqlite:///{tmp_path}/dash_test.db"),
    )


@pytest.fixture
def client(config):
    app = create_app(config)
    return TestClient(app)


@pytest.fixture
def db(config):
    manager = DatabaseManager(config.database.url)
    manager.init_db()
    return manager


def make_decision(symbol: str, action: Action, ts: datetime) -> Decision:
    return Decision(
        timestamp=ts,
        symbol=symbol,
        price=150.0,
        action=action,
        confidence=0.65,
        scores=ScoreBreakdown(technical=0.5, momentum=0.5, volume=0.5, risk=0.0, sentiment=None, composite=0.5),
        indicators={"rsi": 55.0},
        current_position_qty=0.0,
        cash=100_000.0,
        portfolio_value=100_000.0,
        reason=f"{action.value}: composite=0.500",
    )


def test_status_not_started_when_db_is_empty(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_started"
    assert body["symbols"] == ["AAPL", "MSFT"]


def test_status_running_after_recent_snapshot(client, db):
    db.save_portfolio_snapshot(PortfolioSnapshot(NOW, cash=90_000, positions_value=15_000, total_value=105_000, positions={}))
    body = client.get("/api/status").json()
    assert body["status"] == "running"


def test_status_stale_after_old_snapshot(client, db, config):
    old = NOW - timedelta(seconds=config.max_closed_market_heartbeat_seconds * 2)
    db.save_portfolio_snapshot(PortfolioSnapshot(old, cash=90_000, positions_value=15_000, total_value=105_000, positions={}))
    body = client.get("/api/status").json()
    assert body["status"] == "stale"


def test_status_running_during_closed_market_heartbeat_gap(client, db, config):
    """A snapshot from 20 minutes ago is well past 3x the normal 60s
    polling interval, but must still read "running" - that's exactly the
    gap between heartbeats while the market is closed (up to
    max_closed_market_heartbeat_seconds), not a sign the bot died."""
    assert config.polling_interval_seconds * 3 < 20 * 60  # sanity: this age would fail the naive check
    recent_ish = NOW - timedelta(minutes=20)
    db.save_portfolio_snapshot(
        PortfolioSnapshot(recent_ish, cash=90_000, positions_value=15_000, total_value=105_000, positions={})
    )
    body = client.get("/api/status").json()
    assert body["status"] == "running"


def test_portfolio_defaults_to_starting_balance_when_no_snapshot(client):
    body = client.get("/api/portfolio").json()
    assert body["total_value"] == pytest.approx(100_000.0)
    assert body["return_pct"] == pytest.approx(0.0)


def test_portfolio_computes_return_pct_from_latest_snapshot(client, db):
    db.save_portfolio_snapshot(
        PortfolioSnapshot(NOW, cash=90_000, positions_value=20_000, total_value=110_000, positions={"AAPL": {"quantity": 100}})
    )
    body = client.get("/api/portfolio").json()
    assert body["total_value"] == pytest.approx(110_000.0)
    assert body["return_pct"] == pytest.approx(0.10)
    assert body["positions"]["AAPL"]["quantity"] == 100


def test_portfolio_history_is_chronological(client, db):
    for i in range(3):
        db.save_portfolio_snapshot(
            PortfolioSnapshot(NOW + timedelta(days=i), cash=100_000 + i, positions_value=0, total_value=100_000 + i, positions={})
        )
    body = client.get("/api/portfolio/history").json()
    assert len(body) == 3
    assert [p["total_value"] for p in body] == [100_000, 100_001, 100_002]


def test_signals_returns_latest_decision_per_symbol(client, db):
    db.save_decision(make_decision("AAPL", Action.BUY, NOW - timedelta(days=1)))
    db.save_decision(make_decision("AAPL", Action.HOLD, NOW))  # newer - should win
    db.save_decision(make_decision("MSFT", Action.SELL, NOW))

    body = client.get("/api/signals").json()
    assert body["AAPL"]["action"] == "HOLD"
    assert body["MSFT"]["action"] == "SELL"


def test_trades_endpoint_returns_fills_joined_with_orders(client, db):
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET, timestamp=NOW, reason="test")
    order_id = db.save_order(order)
    fill = Fill(order=order, fill_price=150.5, quantity=10, timestamp=NOW, commission=1.0, slippage=0.5)
    db.save_fill(fill, order_id)

    body = client.get("/api/trades").json()
    assert len(body) == 1
    assert body[0]["symbol"] == "AAPL"
    assert body[0]["side"] == "BUY"
    assert body[0]["quantity"] == 10
    assert body[0]["price"] == pytest.approx(150.5)


def test_decisions_endpoint_returns_recent_decisions_desc(client, db):
    db.save_decision(make_decision("AAPL", Action.BUY, NOW - timedelta(minutes=5)))
    db.save_decision(make_decision("AAPL", Action.SELL, NOW))
    body = client.get("/api/decisions").json()
    assert body[0]["action"] == "SELL"  # most recent first
    assert body[1]["action"] == "BUY"


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "StockBot" in resp.text
