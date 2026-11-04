"""
Classroom-simulator order executor (Milestone 6).

Like `ManualExecutor`, this never fills anything itself: the local
portfolio only changes once a human (or a future bridge) has actually
placed the trade on the classroom platform. Instead of just logging, it
appends each risk-approved order to a CSV "ticket outbox" that can be
copied into the platform or consumed by a future adapter.

Outbox columns / UNITS:
  ticket_id       deterministic id (symbol+side+timestamp), so re-writing the
                  same order never creates a duplicate ticket
  timestamp       ISO-8601 UTC
  side            BUY | SELL
  symbol
  quantity        whole SHARES (never dollars, never a fraction of a share)
  reference_price dollars per share at decision time (informational only;
                  the platform fills at its own price)
  confidence      0..1 or blank
  reason

Paper-only: this module has no network access and no credentials.
"""

from __future__ import annotations

import csv
import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.logging import get_logger
from core.models import Fill, Order
from execution.base import ExecutionError, OrderExecutor

logger = get_logger("execution.class_simulator")

FIELDS = ["ticket_id", "timestamp", "side", "symbol", "quantity", "reference_price", "confidence", "reason"]


class ClassSimulatorExecutor(OrderExecutor):
    def __init__(self, outbox_path: str):
        self._path = Path(outbox_path)
        self._seen = self._load_existing_ids()

    def _load_existing_ids(self) -> set:
        if not self._path.exists():
            return set()
        with open(self._path, "r", encoding="utf-8", newline="") as fh:
            return {row["ticket_id"] for row in csv.DictReader(fh)}

    @staticmethod
    def ticket_id(order: Order, timestamp: datetime) -> str:
        key = f"{order.symbol}|{order.side.value}|{timestamp.isoformat()}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    def execute(self, order: Order, market_price: float, timestamp: datetime) -> Optional[Fill]:
        if market_price <= 0:
            raise ExecutionError(f"invalid reference price {market_price!r} for {order.symbol}")
        if order.quantity != math.floor(order.quantity):
            raise ExecutionError(
                f"class simulator tickets are whole shares; got quantity={order.quantity!r} for {order.symbol}"
            )

        ticket_id = self.ticket_id(order, timestamp)
        if ticket_id in self._seen:
            logger.info("Ticket %s already in outbox; not duplicating.", ticket_id)
            return None

        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.exists() or self._path.stat().st_size == 0
        with open(self._path, "a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "ticket_id": ticket_id,
                    "timestamp": timestamp.isoformat(),
                    "side": order.side.value,
                    "symbol": order.symbol,
                    "quantity": int(order.quantity),
                    "reference_price": f"{market_price:.4f}",
                    "confidence": "" if order.confidence is None else f"{order.confidence:.4f}",
                    "reason": order.reason,
                }
            )
        self._seen.add(ticket_id)
        logger.info("Class simulator ticket %s: %s %d %s @ ~$%.2f", ticket_id, order.side.value,
                    int(order.quantity), order.symbol, market_price)
        return None  # no fill until the platform confirms
