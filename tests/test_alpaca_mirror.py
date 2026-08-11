from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.models import Fill, Order, OrderSide, OrderType
from execution.base import ExecutionError

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


class _FakeInnerExecutor:
    """Stands in for PaperExecutor - the authoritative local fill."""

    def __init__(self):
        self.calls = []

    def execute(self, order, market_price, timestamp):
        self.calls.append(order)
        return Fill(order=order, fill_price=market_price, quantity=order.quantity, timestamp=timestamp)


class _FakeAlpacaOrder:
    id = "fake-order-id"


class _FakeTradingClient:
    """Stands in for alpaca.trading.client.TradingClient."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.submitted = []

    def submit_order(self, request):
        self.submitted.append(request)
        return _FakeAlpacaOrder()


class _FailingTradingClient(_FakeTradingClient):
    def submit_order(self, request):
        raise RuntimeError("simulated Alpaca outage")


def make_order(side=OrderSide.BUY, qty=10) -> Order:
    return Order(symbol="AAPL", side=side, quantity=qty, order_type=OrderType.MARKET, timestamp=NOW)


def test_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    from execution.alpaca_mirror import AlpacaMirrorExecutor

    with pytest.raises(ExecutionError, match="ALPACA_API_KEY"):
        AlpacaMirrorExecutor(_FakeInnerExecutor())


def test_hard_codes_paper_true(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeTradingClient)
    from execution.alpaca_mirror import AlpacaMirrorExecutor

    executor = AlpacaMirrorExecutor(_FakeInnerExecutor())
    assert executor._client.init_kwargs.get("paper") is True


def test_execute_returns_the_local_fill_and_mirrors_the_order(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeTradingClient)
    from execution.alpaca_mirror import AlpacaMirrorExecutor

    inner = _FakeInnerExecutor()
    executor = AlpacaMirrorExecutor(inner)
    order = make_order(qty=7)

    fill = executor.execute(order, market_price=150.0, timestamp=NOW)

    assert fill is not None
    assert fill.quantity == 7
    assert inner.calls == [order]  # local ledger got the order
    assert len(executor._client.submitted) == 1  # and it was mirrored
    assert executor._client.submitted[0].qty == 7


def test_mirror_failure_never_affects_the_local_fill(monkeypatch, caplog):
    """The whole point of this class: if Alpaca is unreachable, the
    authoritative local simulation must proceed exactly as if mirroring
    didn't exist - no exception, no missing fill."""
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FailingTradingClient)
    from execution.alpaca_mirror import AlpacaMirrorExecutor

    inner = _FakeInnerExecutor()
    executor = AlpacaMirrorExecutor(inner)
    order = make_order()

    fill = executor.execute(order, market_price=150.0, timestamp=NOW)

    assert fill is not None  # local fill still happened
    assert inner.calls == [order]


def test_sell_order_is_mirrored_with_sell_side(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr("alpaca.trading.client.TradingClient", _FakeTradingClient)
    from alpaca.trading.enums import OrderSide as AlpacaOrderSide

    from execution.alpaca_mirror import AlpacaMirrorExecutor

    executor = AlpacaMirrorExecutor(_FakeInnerExecutor())
    executor.execute(make_order(side=OrderSide.SELL, qty=3), market_price=150.0, timestamp=NOW)

    assert executor._client.submitted[0].side == AlpacaOrderSide.SELL
