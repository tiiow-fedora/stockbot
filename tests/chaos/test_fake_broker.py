"""Trust-but-verify the simulator itself, so a chaos failure means a
StockBot bug rather than a harness bug."""

from __future__ import annotations

import pytest
import requests
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from tests.chaos.fake_alpaca import (
    FakeAlpacaBroker, HTTPError, LiveTradingAttempted, NoFills, PartialFills, Timeout,
)


def req(symbol="AAPL", qty=10, side=OrderSide.BUY, **kw):
    return MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY, **kw)


def TradingClient(*a, **k):
    # resolved lazily, exactly like the bot's own `from alpaca.trading.client import ...`
    from alpaca.trading.client import TradingClient as _TC

    return _TC(*a, **k)


@pytest.fixture
def client(broker):
    broker.prices["AAPL"] = 100.0
    return TradingClient("k", "s", paper=True)


def test_full_fill_moves_cash_and_positions(broker, client):
    client.submit_order(req(qty=10))
    assert broker.cash == pytest.approx(100_000 - 1000)
    assert broker.positions["AAPL"]["qty"] == 10


def test_partial_fill_then_settle_completes(broker, client):
    broker.fill_mode = PartialFills(0.5)
    o = client.submit_order(req(qty=10))
    assert o.status == "partially_filled" and float(o.filled_qty) == 5
    broker.settle()
    assert broker.orders[o.id].status == "filled"
    assert broker.positions["AAPL"]["qty"] == 10


def test_market_closed_order_waits_then_fills(broker, client):
    broker.market_open = False
    o = client.submit_order(req())
    assert o.status == "new" and not broker.positions
    broker.market_open = True
    broker.settle()
    assert broker.positions["AAPL"]["qty"] == 10


def test_no_fills_mode_leaves_order_open(broker, client):
    broker.fill_mode = NoFills()
    assert client.submit_order(req()).status == "new"


def test_rejection_raises_api_error_and_records_nothing(broker, client):
    broker.reject_next("insufficient buying power")
    with pytest.raises(APIError):
        client.submit_order(req())
    assert not broker.positions and broker.cash == 100_000


def test_oversell_is_rejected(broker, client):
    o = client.submit_order(req(side=OrderSide.SELL))
    assert o.status == "rejected"


def test_timeout_not_applied_vs_applied(broker, client):
    broker.fail_next("submit_order", Timeout())
    with pytest.raises(requests.exceptions.ReadTimeout):
        client.submit_order(req())
    assert not broker.positions

    broker.fail_next("submit_order", Timeout(applied=True))
    with pytest.raises(requests.exceptions.ReadTimeout):
        client.submit_order(req())
    assert broker.positions["AAPL"]["qty"] == 10  # ambiguous timeout: it DID happen


def test_http_5xx_is_an_api_error(broker, client):
    broker.fail_next("get_clock", HTTPError(503))
    with pytest.raises(APIError):
        client.get_clock()


def test_duplicate_client_order_id_rejected(broker, client):
    client.submit_order(req(client_order_id="abc"))
    with pytest.raises(APIError):
        client.submit_order(req(client_order_id="abc"))
    assert broker.positions["AAPL"]["qty"] == 10


def test_live_trading_client_is_refused(broker):
    with pytest.raises(LiveTradingAttempted):
        TradingClient("k", "s", paper=False)
    broker.live_attempts = 0  # expected here; fixture teardown must not trip
