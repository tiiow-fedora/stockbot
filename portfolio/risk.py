"""
Risk manager.

Every order the strategy wants to place passes through here before it
reaches the executor - the strategy has NO way to bypass this. The risk
manager is also responsible for sizing new positions (the strategy only
signals direction/confidence, not share count) and for generating
protective exit orders (stop-loss / take-profit) that the strategy itself
never sees coming.

All limits are configurable via RiskConfig (config.yaml `risk:` section).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

from core.config import RiskConfig
from core.logging import get_logger
from core.models import Order, OrderSide, OrderType
from portfolio.portfolio import Portfolio

if TYPE_CHECKING:
    from database.database import DatabaseManager

logger = get_logger("portfolio.risk")


@dataclass
class RiskDecision:
    approved: bool
    quantity: float
    reason: str


class RiskManager:
    def __init__(self, config: RiskConfig, db: Optional["DatabaseManager"] = None):
        """`db` is optional and only meaningful for the live loop. When
        provided, max_trades_per_symbol_per_day is enforced by counting
        actual fills in the database instead of an in-memory list, so a
        process restart (redeploying a fix, a crash, `docker compose
        restart`) can't silently reset the count and let a symbol get
        bought more times in a day than configured - which is exactly
        what happened before this fix existed. Backtester intentionally
        never passes db here: each backtest run is its own self-contained
        simulation and should count from a clean in-memory slate, not
        whatever happens to be in the database from other runs."""
        self.config = config
        self.db = db
        self._trade_timestamps: Dict[str, List[datetime]] = defaultdict(list)

    # -- sizing -----------------------------------------------------------

    def size_position(
        self, symbol: str, price: float, portfolio: Portfolio, prices: Dict[str, float], confidence: float
    ) -> float:
        """Return the number of whole shares a new BUY order should request,
        respecting max_position_size_pct and leaving the configured cash
        reserve untouched. Returns 0 if no meaningful position can be sized."""
        if price <= 0:
            return 0.0

        total_value = portfolio.total_value(prices)
        max_position_value = total_value * self.config.max_position_size_pct
        current_value = portfolio.position_qty(symbol) * price
        room = max(0.0, max_position_value - current_value)

        confidence_scale = min(1.0, max(0.5, 0.5 + confidence))
        target_value = room * confidence_scale

        min_cash_reserve = total_value * self.config.min_cash_reserve_pct
        available_cash = max(0.0, portfolio.cash - min_cash_reserve)

        target_value = min(target_value, available_cash)
        quantity = math.floor(target_value / price)
        return float(max(0, quantity))

    # -- pre-trade checks ---------------------------------------------------

    def evaluate_order(
        self, order: Order, portfolio: Portfolio, prices: Dict[str, float], now: datetime = None
    ) -> RiskDecision:
        now = now or order.timestamp or datetime.now(timezone.utc)
        price = prices.get(order.symbol)
        if price is None:
            return RiskDecision(False, 0.0, "no current price available for symbol")

        if order.side == OrderSide.BUY:
            decision = self._evaluate_buy(order, portfolio, prices, price, now)
        else:
            decision = self._evaluate_sell(order, portfolio, now)

        if decision.approved and decision.quantity > 0:
            self._trade_timestamps[order.symbol].append(now)

        return decision

    def _evaluate_buy(
        self, order: Order, portfolio: Portfolio, prices: Dict[str, float], price: float, now: datetime
    ) -> RiskDecision:
        cfg = self.config

        if self._trades_today(order.symbol, now) >= cfg.max_trades_per_symbol_per_day:
            return RiskDecision(False, 0.0, "max_trade_frequency")

        is_new_symbol = order.symbol not in portfolio.positions
        if is_new_symbol and len(portfolio.positions) >= cfg.max_open_positions:
            return RiskDecision(False, 0.0, "max_open_positions")

        total_value = portfolio.total_value(prices)
        existing_value = portfolio.position_qty(order.symbol) * price
        proposed_value = existing_value + order.quantity * price
        if total_value > 0 and proposed_value / total_value > cfg.max_position_size_pct + 1e-9:
            return RiskDecision(False, 0.0, "max_position_size")

        exposure_after = portfolio.positions_value(prices) + order.quantity * price
        if total_value > 0 and exposure_after / total_value > cfg.max_portfolio_exposure_pct + 1e-9:
            return RiskDecision(False, 0.0, "max_portfolio_exposure")

        min_cash_reserve = total_value * cfg.min_cash_reserve_pct
        order_cost = order.quantity * price
        if portfolio.cash - order_cost < min_cash_reserve - 1e-9:
            return RiskDecision(False, 0.0, "min_cash_reserve")

        if order_cost > portfolio.cash + 1e-9:
            return RiskDecision(False, 0.0, "insufficient_cash")

        return RiskDecision(True, order.quantity, "approved")

    def _evaluate_sell(self, order: Order, portfolio: Portfolio, now: datetime) -> RiskDecision:
        held = portfolio.position_qty(order.symbol)
        if held <= 0:
            return RiskDecision(False, 0.0, "no_position_to_sell")
        quantity = min(order.quantity, held)
        return RiskDecision(True, quantity, "approved")

    def _trades_today(self, symbol: str, now: datetime) -> int:
        if self.db is not None:
            return self.db.count_trades_today(symbol, now)
        return sum(1 for ts in self._trade_timestamps[symbol] if ts.date() == now.date())

    # -- protective exits -----------------------------------------------------

    def check_stop_loss_take_profit(
        self, portfolio: Portfolio, prices: Dict[str, float], now: datetime = None
    ) -> List[Order]:
        now = now or datetime.now(timezone.utc)
        cfg = self.config
        orders: List[Order] = []
        for symbol, pos in list(portfolio.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            pnl_pct = pos.unrealized_pnl_pct(price)
            if pnl_pct <= -cfg.stop_loss_pct:
                orders.append(
                    Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        order_type=OrderType.MARKET,
                        timestamp=now,
                        reason=f"stop_loss triggered at {pnl_pct:.2%}",
                    )
                )
            elif pnl_pct >= cfg.take_profit_pct:
                orders.append(
                    Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        order_type=OrderType.MARKET,
                        timestamp=now,
                        reason=f"take_profit triggered at {pnl_pct:.2%}",
                    )
                )
        return orders
