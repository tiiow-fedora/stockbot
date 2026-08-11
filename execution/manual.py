"""
Manual order executor.

For platforms that don't allow (or that you don't yet trust with)
automated execution: this executor never touches the portfolio. It just
logs a clear, human-actionable suggestion and keeps a queue of pending
suggestions so a dashboard or CLI can display "here's what the bot wants
to do" for a person to execute by hand.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from core.logging import get_logger
from core.models import Fill, Order
from execution.base import OrderExecutor

logger = get_logger("execution.manual")


class ManualExecutor(OrderExecutor):
    def __init__(self):
        self.pending_suggestions: List[dict] = []

    def execute(self, order: Order, market_price: float, timestamp: datetime) -> Optional[Fill]:
        suggestion = {
            "timestamp": timestamp,
            "action": order.side.value,
            "symbol": order.symbol,
            "quantity": order.quantity,
            "price": market_price,
            "confidence": order.confidence,
            "reason": order.reason,
        }
        self.pending_suggestions.append(suggestion)
        confidence_str = f"{order.confidence:.0%}" if order.confidence is not None else "n/a"
        logger.info(
            "%s %s\nQuantity: %s\nCurrent price: $%.2f\nConfidence: %s\nReason: %s",
            order.side.value,
            order.symbol,
            order.quantity,
            market_price,
            confidence_str,
            order.reason,
        )
        return None  # no fill - a human must act; portfolio stays unchanged until they confirm
