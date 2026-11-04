"""
Shared order construction/routing logic used by BOTH the live loop
(main.py) and the backtester (backtesting/engine.py).

Keeping this in one place means the live trading path and the backtest
path can never quietly diverge in how a Decision becomes an Order, or how
an Order gets risk-checked and executed - which matters a lot for a
system whose whole point is "does the backtest predict live behavior".
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from core.logging import get_logger
from core.models import Action, Decision, Fill, Order, OrderSide, OrderType
from database.database import DatabaseManager
from execution.base import OrderExecutor
from observability import metrics
from portfolio.portfolio import Portfolio
from portfolio.risk import RiskManager

logger = get_logger("execution.order_builder")


def decision_to_order(
    decision: Decision, risk_manager: RiskManager, portfolio: Portfolio, prices: Dict[str, float]
) -> Optional[Order]:
    """Strategies signal direction + confidence, never share count - sizing
    is always the risk manager's job."""
    if decision.action == Action.HOLD:
        return None

    if decision.action == Action.BUY:
        qty = risk_manager.size_position(decision.symbol, decision.price, portfolio, prices, decision.confidence)
        if qty <= 0:
            return None
        return Order(
            symbol=decision.symbol,
            side=OrderSide.BUY,
            quantity=qty,
            order_type=OrderType.MARKET,
            timestamp=decision.timestamp,
            reason=decision.reason,
            confidence=decision.confidence,
        )

    # SELL: liquidate the full existing position. The strategy signals
    # direction only; there is nothing to partially trim in this design.
    qty = portfolio.position_qty(decision.symbol)
    if qty <= 0:
        return None
    return Order(
        symbol=decision.symbol,
        side=OrderSide.SELL,
        quantity=qty,
        order_type=OrderType.MARKET,
        timestamp=decision.timestamp,
        reason=decision.reason,
        confidence=decision.confidence,
    )


def route_order(
    order: Order,
    risk_manager: RiskManager,
    portfolio: Portfolio,
    executor: OrderExecutor,
    prices: Dict[str, float],
    fill_price: float,
    timestamp: datetime,
    db: Optional[DatabaseManager] = None,
) -> Optional[Fill]:
    """Risk-check `order`, execute it if approved, and apply the resulting
    fill to `portfolio`. Returns the Fill, or None if rejected/unfilled."""
    risk_decision = risk_manager.evaluate_order(order, portfolio, prices, now=order.timestamp or timestamp)

    if not risk_decision.approved or risk_decision.quantity <= 0:
        logger.info("%s %s rejected by risk manager: %s", order.side.value, order.symbol, risk_decision.reason)
        metrics.record_order(order.side.value, "rejected", risk_decision.reason)
        if db:
            db.save_risk_rejection(
                order.symbol, order.side.value, order.quantity, order.timestamp or timestamp, risk_decision.reason
            )
        return None

    order.quantity = risk_decision.quantity
    logger.info("Risk approved %s %s quantity=%s", order.side.value, order.symbol, order.quantity)

    fill = executor.execute(order, fill_price, timestamp)
    if fill is not None:
        metrics.record_order(order.side.value, "filled")
        portfolio.apply_fill(fill)
        if db:
            order_id = db.save_order(order)
            db.save_fill(fill, order_id)
    return fill
