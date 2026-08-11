"""
Simulated (paper) portfolio: cash + positions accounting.

This class owns the ONLY mutable ledger of what the bot "owns". Nothing
else should track cash or shares independently - the executor produces
Fills, and every Fill must go through `Portfolio.apply_fill` so that
cash, average cost basis, and realized P&L stay consistent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

from core.models import Fill, OrderSide, PortfolioSnapshot, Position


class InsufficientFundsError(Exception):
    pass


class InsufficientPositionError(Exception):
    pass


class Portfolio:
    def __init__(self, starting_cash: float):
        if starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, Position] = {}
        self.realized_pnl = 0.0
        self.trade_log: list = []  # list of dicts, most-recent-last, for win/loss stats

    def position_qty(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0

    def apply_fill(self, fill: Fill) -> None:
        symbol = fill.order.symbol
        cost_or_proceeds_commission = fill.commission

        if fill.order.side == OrderSide.BUY:
            total_cost = fill.fill_price * fill.quantity + cost_or_proceeds_commission
            if total_cost > self.cash + 1e-6:
                raise InsufficientFundsError(
                    f"cannot afford BUY {symbol} qty={fill.quantity} at {fill.fill_price}: "
                    f"needs {total_cost:.2f}, have {self.cash:.2f}"
                )
            self.cash -= total_cost
            existing = self.positions.get(symbol)
            if existing is None:
                self.positions[symbol] = Position(symbol, fill.quantity, fill.fill_price)
            else:
                new_qty = existing.quantity + fill.quantity
                existing.avg_entry_price = (
                    existing.avg_entry_price * existing.quantity + fill.fill_price * fill.quantity
                ) / new_qty
                existing.quantity = new_qty
            self.trade_log.append(
                {
                    "timestamp": fill.timestamp,
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": fill.quantity,
                    "price": fill.fill_price,
                    "realized_pnl": None,
                }
            )
        else:  # SELL
            existing = self.positions.get(symbol)
            if existing is None or existing.quantity < fill.quantity - 1e-6:
                have = existing.quantity if existing else 0.0
                raise InsufficientPositionError(
                    f"cannot SELL {symbol} qty={fill.quantity}: only hold {have}"
                )
            proceeds = fill.fill_price * fill.quantity - cost_or_proceeds_commission
            self.cash += proceeds
            trade_pnl = (fill.fill_price - existing.avg_entry_price) * fill.quantity - cost_or_proceeds_commission
            self.realized_pnl += trade_pnl
            existing.quantity -= fill.quantity
            if existing.quantity <= 1e-6:
                del self.positions[symbol]
            self.trade_log.append(
                {
                    "timestamp": fill.timestamp,
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": fill.quantity,
                    "price": fill.fill_price,
                    "realized_pnl": trade_pnl,
                }
            )

    def positions_value(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.avg_entry_price)
            total += pos.market_value(price)
        return total

    def total_value(self, prices: Dict[str, float]) -> float:
        return self.cash + self.positions_value(prices)

    def total_return_pct(self, prices: Dict[str, float]) -> float:
        return (self.total_value(prices) - self.starting_cash) / self.starting_cash

    def snapshot(self, timestamp: datetime, prices: Dict[str, float]) -> PortfolioSnapshot:
        positions_dict = {}
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos.avg_entry_price)
            positions_dict[symbol] = {
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
                "market_value": pos.market_value(price),
                "unrealized_pnl": pos.unrealized_pnl(price),
            }
        pv = self.positions_value(prices)
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=pv,
            total_value=self.cash + pv,
            positions=positions_dict,
        )
