"""
Abstract strategy interface.

A Strategy receives ONLY standardized types (MarketSnapshot, Candle,
NewsItem) plus the caller's current position/cash/portfolio value - never
a provider, never a database session, never an executor. This is what
makes it possible to swap Alpaca for a classroom platform, or backtest
against a CSV, without changing a single line of strategy code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.models import Candle, Decision, MarketSnapshot, NewsItem


class Strategy(ABC):
    @abstractmethod
    def generate_decision(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        history: List[Candle],
        news: List[NewsItem],
        current_position_qty: float,
        cash: float,
        portfolio_value: float,
    ) -> Decision:
        """Produce a Decision using only data available up to `snapshot.timestamp`.

        `history` must contain no bar at or after `snapshot.timestamp` -
        callers (the live loop and the backtester) are responsible for
        that invariant; strategies should not need to re-check it, but
        may assume it holds.
        """
