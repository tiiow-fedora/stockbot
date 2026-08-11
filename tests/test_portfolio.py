from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.models import Fill, Order, OrderSide, OrderType
from portfolio.portfolio import InsufficientFundsError, InsufficientPositionError, Portfolio

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def buy(symbol, qty, price, commission=0.0):
    order = Order(symbol=symbol, side=OrderSide.BUY, quantity=qty, order_type=OrderType.MARKET, timestamp=NOW)
    return Fill(order=order, fill_price=price, quantity=qty, timestamp=NOW, commission=commission)


def sell(symbol, qty, price, commission=0.0):
    order = Order(symbol=symbol, side=OrderSide.SELL, quantity=qty, order_type=OrderType.MARKET, timestamp=NOW)
    return Fill(order=order, fill_price=price, quantity=qty, timestamp=NOW, commission=commission)


def test_buy_reduces_cash_and_opens_position():
    p = Portfolio(10_000)
    p.apply_fill(buy("AAPL", 10, 100.0))
    assert p.cash == pytest.approx(9_000.0)
    assert p.position_qty("AAPL") == 10
    assert p.positions["AAPL"].avg_entry_price == pytest.approx(100.0)


def test_second_buy_updates_weighted_average_cost():
    p = Portfolio(10_000)
    p.apply_fill(buy("AAPL", 10, 100.0))
    p.apply_fill(buy("AAPL", 10, 120.0))
    assert p.position_qty("AAPL") == 20
    assert p.positions["AAPL"].avg_entry_price == pytest.approx(110.0)


def test_buy_rejects_when_insufficient_cash():
    p = Portfolio(1_000)
    with pytest.raises(InsufficientFundsError):
        p.apply_fill(buy("AAPL", 100, 100.0))


def test_sell_full_position_closes_it_and_realizes_pnl():
    p = Portfolio(10_000)
    p.apply_fill(buy("AAPL", 10, 100.0))
    p.apply_fill(sell("AAPL", 10, 120.0))
    assert "AAPL" not in p.positions
    assert p.realized_pnl == pytest.approx(200.0)
    assert p.cash == pytest.approx(10_000 - 1000 + 1200)


def test_sell_more_than_held_raises():
    p = Portfolio(10_000)
    p.apply_fill(buy("AAPL", 5, 100.0))
    with pytest.raises(InsufficientPositionError):
        p.apply_fill(sell("AAPL", 10, 100.0))


def test_commission_reduces_cash_on_buy_and_proceeds_on_sell():
    p = Portfolio(10_000)
    p.apply_fill(buy("AAPL", 10, 100.0, commission=5.0))
    assert p.cash == pytest.approx(10_000 - 1000 - 5)
    p.apply_fill(sell("AAPL", 10, 100.0, commission=5.0))
    assert p.cash == pytest.approx(10_000 - 5 - 5)


def test_total_value_uses_current_prices_not_entry_price():
    p = Portfolio(10_000)
    p.apply_fill(buy("AAPL", 10, 100.0))
    assert p.total_value({"AAPL": 150.0}) == pytest.approx(9_000 + 1_500)


def test_total_value_falls_back_to_entry_price_if_no_current_price():
    p = Portfolio(10_000)
    p.apply_fill(buy("AAPL", 10, 100.0))
    assert p.total_value({}) == pytest.approx(9_000 + 1_000)


def test_snapshot_reflects_unrealized_pnl():
    p = Portfolio(10_000)
    p.apply_fill(buy("AAPL", 10, 100.0))
    snap = p.snapshot(NOW, {"AAPL": 130.0})
    assert snap.positions["AAPL"]["unrealized_pnl"] == pytest.approx(300.0)
    assert snap.total_value == pytest.approx(9_000 + 1_300)
