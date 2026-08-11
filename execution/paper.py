"""
Paper (simulated) order executor.

Fills every order immediately at the reference market price, adjusted for
configurable slippage and commission. This NEVER sends anything to a real
brokerage - it only produces a Fill object; the caller is responsible for
applying that Fill to a Portfolio.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from core.config import ExecutionConfig
from core.logging import get_logger
from core.models import Fill, Order, OrderSide
from execution.base import OrderExecutor

logger = get_logger("execution.paper")


class PaperExecutor(OrderExecutor):
    def __init__(self, config: ExecutionConfig):
        self.config = config

    def execute(self, order: Order, market_price: float, timestamp: datetime) -> Optional[Fill]:
        slippage_frac = self.config.slippage_bps / 10_000.0
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1 + slippage_frac)
        else:
            fill_price = market_price * (1 - slippage_frac)

        slippage_amount = abs(fill_price - market_price) * order.quantity
        commission = self.config.commission_per_trade + (fill_price * order.quantity * self.config.commission_pct)

        fill = Fill(
            order=order,
            fill_price=round(fill_price, 4),
            quantity=order.quantity,
            timestamp=timestamp,
            commission=round(commission, 4),
            slippage=round(slippage_amount, 4),
        )
        logger.info(
            "Paper order executed %s %s qty=%s price=%.2f commission=%.2f slippage=%.2f",
            order.side.value,
            order.symbol,
            order.quantity,
            fill.fill_price,
            fill.commission,
            fill.slippage,
        )
        return fill
