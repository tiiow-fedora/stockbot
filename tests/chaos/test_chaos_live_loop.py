"""The real run_live_loop against the fake broker: outages, market
open/close flaps, and a liveness guard that BUY signals still fire."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import main
from database.database import DatabaseManager
from tests.chaos.conftest import dip_candles, make_config
from tests.chaos.fake_alpaca import ConnectionDrop, HTTPError, Timeout
from tests.chaos.loop_driver import run_loop

TODAY_BAR = datetime(2024, 3, 12, 4, 0, tzinfo=timezone.utc)


def mirrored_config(db_url, symbols=("AAPL",), **risk):
    cfg = make_config(db_url, symbols=symbols, **risk)
    cfg.execution.mirror_to_alpaca = True
    cfg.polling_interval_seconds = 60
    return cfg


def fills(db_url):
    return DatabaseManager(db_url).get_recent_trades(limit=500)


def test_liveness_a_clearly_oversold_stock_is_bought_end_to_end(broker, db_url, monkeypatch):
    """Guards the class of bug that previously suppressed every BUY (a unit
    mismatch between data and indicators): a textbook oversold series MUST
    produce a BUY that reaches the broker, through the real loop."""
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    out = run_loop(monkeypatch, broker, mirrored_config(db_url), iterations=2)
    assert out["portfolio"].position_qty("AAPL") > 0
    assert broker.positions["AAPL"]["qty"] == out["portfolio"].position_qty("AAPL")
    assert any(t["side"] == "BUY" for t in fills(db_url))


def test_flat_market_produces_no_trades(broker, db_url, monkeypatch):
    from tests.conftest import make_candles

    broker.set_bars_from_candles("AAPL", make_candles("AAPL", n=80, daily_return=0.0,
                                                      start=datetime(2023, 12, 20, tzinfo=timezone.utc)))
    run_loop(monkeypatch, broker, mirrored_config(db_url), iterations=3)
    assert fills(db_url) == [] and not broker.orders


@pytest.mark.parametrize("fault", [Timeout(), ConnectionDrop(), HTTPError(503), HTTPError(500)])
def test_calendar_outage_backs_off_then_recovers_and_trades(broker, db_url, monkeypatch, fault):
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    broker.fail_next("get_clock", fault, times=3)
    out = run_loop(monkeypatch, broker, mirrored_config(db_url), iterations=5)
    base = main.RETRY_BACKOFF_BASE_SECONDS
    assert out["sleeps"][:3] == [base, base * 2, base * 4]
    assert out["portfolio"].position_qty("AAPL") > 0


def test_market_closed_means_no_orders_even_with_a_screaming_buy(broker, db_url, monkeypatch):
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    broker.market_open = False
    run_loop(monkeypatch, broker, mirrored_config(db_url), iterations=4)
    assert not broker.orders and fills(db_url) == []
    assert DatabaseManager(db_url).get_portfolio_history()      # heartbeat still ran


def test_market_flaps_open_closed_without_double_trading(broker, db_url, monkeypatch):
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))

    def flap(n, _s):
        broker.market_open = n % 2 == 0

    out = run_loop(monkeypatch, broker, mirrored_config(db_url, max_trades_per_symbol_per_day=1), iterations=10,
                   on_sleep=flap)
    assert len([t for t in fills(db_url) if t["side"] == "BUY"]) == 1
    assert broker.positions["AAPL"]["qty"] == out["portfolio"].position_qty("AAPL")


def test_data_outage_mid_session_does_not_trade_blind_or_crash(broker, db_url, monkeypatch):
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    broker.fail_next("get_stock_bars", Timeout(), times=6)   # every symbol call for a few cycles
    out = run_loop(monkeypatch, broker, mirrored_config(db_url), iterations=6)
    assert out["sleeps"]                      # loop survived
    assert not broker.orders                  # and made no orders on missing data


def test_every_alpaca_client_the_loop_builds_is_paper(broker, db_url, monkeypatch):
    broker.set_bars_from_candles("AAPL", dip_candles(end=TODAY_BAR))
    run_loop(monkeypatch, broker, mirrored_config(db_url), iterations=2)
    assert broker.live_attempts == 0
