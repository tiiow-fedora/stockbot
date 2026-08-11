from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.config import ExecutionConfig
from core.models import Order, OrderSide, OrderType
from execution.manual import ManualExecutor
from execution.paper import PaperExecutor

NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)


def test_paper_executor_buy_pays_slippage_above_market_price():
    executor = PaperExecutor(ExecutionConfig(slippage_bps=100.0, commission_per_trade=0.0, commission_pct=0.0))
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET, timestamp=NOW)
    fill = executor.execute(order, market_price=100.0, timestamp=NOW)
    assert fill.fill_price == pytest.approx(101.0)  # 100 bps = 1%


def test_paper_executor_sell_receives_below_market_price():
    executor = PaperExecutor(ExecutionConfig(slippage_bps=100.0, commission_per_trade=0.0, commission_pct=0.0))
    order = Order(symbol="AAPL", side=OrderSide.SELL, quantity=10, order_type=OrderType.MARKET, timestamp=NOW)
    fill = executor.execute(order, market_price=100.0, timestamp=NOW)
    assert fill.fill_price == pytest.approx(99.0)


def test_paper_executor_applies_flat_and_percentage_commission():
    executor = PaperExecutor(ExecutionConfig(slippage_bps=0.0, commission_per_trade=1.5, commission_pct=0.01))
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET, timestamp=NOW)
    fill = executor.execute(order, market_price=100.0, timestamp=NOW)
    # flat 1.5 + 1% of (100*10=1000) = 1.5 + 10 = 11.5
    assert fill.commission == pytest.approx(11.5)


def test_paper_executor_zero_cost_config_is_a_frictionless_fill():
    executor = PaperExecutor(ExecutionConfig(slippage_bps=0.0, commission_per_trade=0.0, commission_pct=0.0))
    order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=10, order_type=OrderType.MARKET, timestamp=NOW)
    fill = executor.execute(order, market_price=100.0, timestamp=NOW)
    assert fill.fill_price == pytest.approx(100.0)
    assert fill.commission == pytest.approx(0.0)


def test_manual_executor_never_returns_a_fill():
    executor = ManualExecutor()
    order = Order(
        symbol="NVDA",
        side=OrderSide.BUY,
        quantity=14,
        order_type=OrderType.MARKET,
        timestamp=NOW,
        reason="composite=0.81",
        confidence=0.81,
    )
    fill = executor.execute(order, market_price=182.34, timestamp=NOW)
    assert fill is None


def test_manual_executor_records_pending_suggestion():
    executor = ManualExecutor()
    order = Order(symbol="NVDA", side=OrderSide.BUY, quantity=14, order_type=OrderType.MARKET, timestamp=NOW, confidence=0.81)
    executor.execute(order, market_price=182.34, timestamp=NOW)
    assert len(executor.pending_suggestions) == 1
    suggestion = executor.pending_suggestions[0]
    assert suggestion["symbol"] == "NVDA"
    assert suggestion["quantity"] == 14
    assert suggestion["confidence"] == pytest.approx(0.81)
