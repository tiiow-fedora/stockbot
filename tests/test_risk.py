from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.config import RiskConfig
from core.models import Fill, Order, OrderSide, OrderType
from database.database import DatabaseManager
from portfolio.portfolio import Portfolio
from portfolio.risk import RiskManager

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_risk_manager(**overrides) -> RiskManager:
    return RiskManager(RiskConfig(**overrides))


def apply_buy(portfolio: Portfolio, symbol: str, qty: float, price: float, ts=NOW):
    order = Order(symbol=symbol, side=OrderSide.BUY, quantity=qty, order_type=OrderType.MARKET, timestamp=ts)
    fill = Fill(order=order, fill_price=price, quantity=qty, timestamp=ts)
    portfolio.apply_fill(fill)


def test_size_position_respects_max_position_size_pct():
    rm = make_risk_manager(max_position_size_pct=0.10, min_cash_reserve_pct=0.0)
    p = Portfolio(10_000)
    qty = rm.size_position("AAPL", 100.0, p, {"AAPL": 100.0}, confidence=1.0)
    # Max position value = 1000 -> at most 10 shares at $100, scaled by confidence (<=1.0)
    assert qty <= 10


def test_size_position_leaves_cash_reserve_untouched():
    rm = make_risk_manager(max_position_size_pct=1.0, min_cash_reserve_pct=0.20)
    p = Portfolio(10_000)
    qty = rm.size_position("AAPL", 100.0, p, {"AAPL": 100.0}, confidence=1.0)
    cost = qty * 100.0
    assert p.cash - cost >= 10_000 * 0.20 - 1e-6


def test_evaluate_order_rejects_over_max_position_size():
    rm = make_risk_manager(max_position_size_pct=0.10)
    p = Portfolio(10_000)
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=50, order_type=OrderType.MARKET, timestamp=NOW)
    decision = rm.evaluate_order(order, p, {"AAPL": 100.0}, now=NOW)
    assert decision.approved is False
    assert decision.reason == "max_position_size"


def test_evaluate_order_rejects_over_max_open_positions():
    rm = make_risk_manager(max_open_positions=1, max_position_size_pct=1.0)
    p = Portfolio(10_000)
    apply_buy(p, "AAPL", 10, 100.0)
    order = Order(symbol="MSFT", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=NOW)
    decision = rm.evaluate_order(order, p, {"AAPL": 100.0, "MSFT": 100.0}, now=NOW)
    assert decision.approved is False
    assert decision.reason == "max_open_positions"


def test_evaluate_order_rejects_over_max_portfolio_exposure():
    rm = make_risk_manager(max_portfolio_exposure_pct=0.5, max_position_size_pct=1.0, max_open_positions=10)
    p = Portfolio(10_000)
    apply_buy(p, "AAPL", 40, 100.0)  # already 40% exposed
    order = Order(symbol="MSFT", side=OrderSide.BUY, quantity=20, order_type=OrderType.MARKET, timestamp=NOW)
    decision = rm.evaluate_order(order, p, {"AAPL": 100.0, "MSFT": 100.0}, now=NOW)
    assert decision.approved is False
    assert decision.reason == "max_portfolio_exposure"


def test_evaluate_order_enforces_max_trade_frequency():
    rm = make_risk_manager(max_trades_per_symbol_per_day=1, max_position_size_pct=1.0)
    p = Portfolio(10_000)
    order1 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=NOW)
    decision1 = rm.evaluate_order(order1, p, {"AAPL": 100.0}, now=NOW)
    assert decision1.approved is True

    order2 = Order(symbol="AAPL", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=NOW)
    decision2 = rm.evaluate_order(order2, p, {"AAPL": 100.0}, now=NOW)
    assert decision2.approved is False
    assert decision2.reason == "max_trade_frequency"

    # A trade the next day should be allowed again.
    order3 = Order(
        symbol="AAPL", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=NOW + timedelta(days=1)
    )
    decision3 = rm.evaluate_order(order3, p, {"AAPL": 100.0}, now=NOW + timedelta(days=1))
    assert decision3.approved is True


def test_evaluate_sell_rejects_when_no_position_held():
    rm = make_risk_manager()
    p = Portfolio(10_000)
    order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=10, order_type=OrderType.MARKET, timestamp=NOW)
    decision = rm.evaluate_order(order, p, {"AAPL": 100.0}, now=NOW)
    assert decision.approved is False
    assert decision.reason == "no_position_to_sell"


def test_evaluate_sell_clamps_quantity_to_held_amount():
    rm = make_risk_manager()
    p = Portfolio(10_000)
    apply_buy(p, "AAPL", 5, 100.0)
    order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=100, order_type=OrderType.MARKET, timestamp=NOW)
    decision = rm.evaluate_order(order, p, {"AAPL": 100.0}, now=NOW)
    assert decision.approved is True
    assert decision.quantity == 5


def test_stop_loss_generates_sell_order_when_position_down():
    rm = make_risk_manager(stop_loss_pct=0.05)
    p = Portfolio(10_000)
    apply_buy(p, "AAPL", 10, 100.0)
    orders = rm.check_stop_loss_take_profit(p, {"AAPL": 90.0}, now=NOW)  # down 10%
    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL
    assert orders[0].quantity == 10
    assert "stop_loss" in orders[0].reason


def test_take_profit_generates_sell_order_when_position_up():
    rm = make_risk_manager(take_profit_pct=0.10)
    p = Portfolio(10_000)
    apply_buy(p, "AAPL", 10, 100.0)
    orders = rm.check_stop_loss_take_profit(p, {"AAPL": 120.0}, now=NOW)  # up 20%
    assert len(orders) == 1
    assert "take_profit" in orders[0].reason


def test_no_exit_order_within_normal_range():
    rm = make_risk_manager(stop_loss_pct=0.10, take_profit_pct=0.10)
    p = Portfolio(10_000)
    apply_buy(p, "AAPL", 10, 100.0)
    orders = rm.check_stop_loss_take_profit(p, {"AAPL": 103.0}, now=NOW)
    assert orders == []


def test_db_backed_frequency_limit_survives_a_simulated_restart(tmp_path):
    """Real bug this guards against: the in-memory trade counter used to
    reset to zero on every process restart, silently letting a symbol get
    bought more times in a day than max_trades_per_symbol_per_day allows
    (confirmed live: INTC got bought 6 times in one day against a limit
    of 3, exactly across several redeploys). A fresh RiskManager backed
    by the same database must still see - and enforce - prior fills."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    db = DatabaseManager(db_url)
    db.init_db()

    config = RiskConfig(max_trades_per_symbol_per_day=3, max_position_size_pct=1.0)
    portfolio = Portfolio(100_000)

    # First "process": place 3 approved buys, as a real live loop would.
    rm1 = RiskManager(config, db=db)
    for _ in range(3):
        order = Order(symbol="INTC", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=NOW)
        decision = rm1.evaluate_order(order, portfolio, {"INTC": 100.0}, now=NOW)
        assert decision.approved
        fill = Fill(order=order, fill_price=100.0, quantity=1, timestamp=NOW)
        portfolio.apply_fill(fill)
        order_id = db.save_order(order)
        db.save_fill(fill, order_id)

    # Simulate a restart: brand-new RiskManager and DatabaseManager, same DB.
    rm2 = RiskManager(config, db=DatabaseManager(db_url))
    order = Order(symbol="INTC", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=NOW)
    decision = rm2.evaluate_order(order, portfolio, {"INTC": 100.0}, now=NOW)

    assert decision.approved is False
    assert decision.reason == "max_trade_frequency"


def test_without_db_frequency_limit_is_in_memory_only_and_does_not_persist():
    """Backtester relies on this: no db means each RiskManager instance
    starts with a clean slate, which is correct for a self-contained
    backtest run - this is the deliberate default, not an oversight."""
    config = RiskConfig(max_trades_per_symbol_per_day=1, max_position_size_pct=1.0)
    portfolio = Portfolio(100_000)

    rm1 = RiskManager(config)  # no db
    order1 = Order(symbol="INTC", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=NOW)
    assert rm1.evaluate_order(order1, portfolio, {"INTC": 100.0}, now=NOW).approved is True

    rm2 = RiskManager(config)  # a fresh instance, still no db - starts over
    order2 = Order(symbol="INTC", side=OrderSide.BUY, quantity=1, order_type=OrderType.MARKET, timestamp=NOW)
    assert rm2.evaluate_order(order2, portfolio, {"INTC": 100.0}, now=NOW).approved is True
