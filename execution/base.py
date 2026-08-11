"""
Abstract order executor interface.

An OrderExecutor takes a risk-approved Order and a reference market price
and returns a Fill (or None if nothing was actually filled, e.g. a
ManualExecutor waiting on a human). It NEVER decides whether an order
should happen - that's the risk manager's job - and it never talks to
the strategy. This is the seam where a future ClassSimulatorExecutor
plugs in without touching anything upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from core.models import Fill, Order


class ExecutionError(Exception):
    """Raised for executor-level setup problems (e.g. missing credentials
    for a mirroring executor) - never raised mid-trade to abort a fill
    that's already been risk-approved."""


class OrderExecutor(ABC):
    @abstractmethod
    def execute(self, order: Order, market_price: float, timestamp: datetime) -> Optional[Fill]:
        """Simulate or submit `order` at/around `market_price`. Return the
        resulting Fill, or None if the order was not (yet) filled."""
