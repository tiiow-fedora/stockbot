"""
Backtesting engine.

Simulates the full DATA -> STRATEGY -> RISK -> EXECUTOR -> PORTFOLIO
pipeline against historical candles, with NO access to future data:

  * At step i, the strategy only ever sees `candles[:i-1]` as history plus
    the CLOSE of `candles[i-1]` as its "current price" - i.e. everything
    it uses was known by the end of that bar.
  * Orders decided from bar i-1's close are filled at bar i's OPEN (the
    next tradeable price), not at the same bar's close. Filling at the
    same close the decision was based on would be look-ahead bias.
  * A buy-and-hold benchmark is run through the exact same executor (so
    it pays the same slippage/commission) for a fair comparison.

This engine is single-symbol per run; call it once per symbol if you want
a multi-symbol comparison (see main.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from core.logging import get_logger
from core.models import Candle, MarketSnapshot, Order, OrderSide, OrderType
from database.database import DatabaseManager
from execution.order_builder import decision_to_order, route_order
from execution.paper import PaperExecutor
from portfolio.portfolio import Portfolio
from portfolio.risk import RiskManager
from strategy.base import Strategy

logger = get_logger("backtesting.engine")


@dataclass
class BacktestResult:
    symbol: str
    starting_balance: float
    ending_balance: float
    total_return_pct: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: Optional[float]
    equity_curve: List[Tuple[datetime, float]]
    benchmark_curve: List[Tuple[datetime, float]]
    benchmark_return_pct: float
    trade_log: List[dict] = field(default_factory=list)
    rejected_orders: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.symbol}: strategy {self.total_return_pct:+.2%} "
            f"vs buy-and-hold {self.benchmark_return_pct:+.2%} | "
            f"trades={self.num_trades} win_rate={self.win_rate:.1%} "
            f"max_drawdown={self.max_drawdown_pct:.2%} "
            f"sharpe={'n/a' if self.sharpe_ratio is None else f'{self.sharpe_ratio:.2f}'}"
        )


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        risk_config: RiskConfig,
        execution_config: ExecutionConfig,
        starting_balance: float = 100_000.0,
        warmup_bars: int = 40,
        db: Optional[DatabaseManager] = None,
    ):
        self.strategy = strategy
        self.risk_manager = RiskManager(risk_config)
        self.execution_config = execution_config
        self.executor = PaperExecutor(execution_config)
        self.starting_balance = starting_balance
        self.warmup_bars = warmup_bars
        self.db = db

    def run(self, symbol: str, candles: List[Candle]) -> BacktestResult:
        if len(candles) < self.warmup_bars + 2:
            raise ValueError(
                f"not enough candles for {symbol}: have {len(candles)}, "
                f"need at least warmup_bars+2={self.warmup_bars + 2}"
            )

        portfolio = Portfolio(self.starting_balance)
        benchmark_portfolio = Portfolio(self.starting_balance)
        benchmark_entered = False

        equity_curve: List[Tuple[datetime, float]] = []
        benchmark_curve: List[Tuple[datetime, float]] = []
        rejected_orders: List[dict] = []

        for i in range(self.warmup_bars, len(candles)):
            decision_bar = candles[i - 1]
            execution_bar = candles[i]
            history = candles[: i - 1]
            prices_at_decision = {symbol: decision_bar.close}

            # 1) Protective exits are checked first, using data known as of
            #    the decision bar's close, filled at the next bar's open.
            for order in self.risk_manager.check_stop_loss_take_profit(
                portfolio, prices_at_decision, now=decision_bar.timestamp
            ):
                fill = route_order(
                    order,
                    self.risk_manager,
                    portfolio,
                    self.executor,
                    prices_at_decision,
                    execution_bar.open,
                    execution_bar.timestamp,
                    self.db,
                )
                if fill is None:
                    rejected_orders.append(
                        {
                            "timestamp": order.timestamp,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "quantity": order.quantity,
                            "reason": order.reason,
                        }
                    )

            # 2) Strategy decision.
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=decision_bar.timestamp,
                price=decision_bar.close,
                volume=decision_bar.volume,
                open=decision_bar.open,
                high=decision_bar.high,
                low=decision_bar.low,
                close=decision_bar.close,
            )
            decision = self.strategy.generate_decision(
                symbol=symbol,
                snapshot=snapshot,
                history=history,
                news=[],
                current_position_qty=portfolio.position_qty(symbol),
                cash=portfolio.cash,
                portfolio_value=portfolio.total_value(prices_at_decision),
            )
            if self.db:
                self.db.save_decision(decision)

            order = decision_to_order(decision, self.risk_manager, portfolio, prices_at_decision)
            if order is not None:
                fill = route_order(
                    order,
                    self.risk_manager,
                    portfolio,
                    self.executor,
                    prices_at_decision,
                    execution_bar.open,
                    execution_bar.timestamp,
                    self.db,
                )
                if fill is None:
                    rejected_orders.append(
                        {
                            "timestamp": order.timestamp,
                            "symbol": order.symbol,
                            "side": order.side.value,
                            "quantity": order.quantity,
                            "reason": order.reason,
                        }
                    )

            # 3) Mark to market at the execution bar's close for the equity curve.
            mark_prices = {symbol: execution_bar.close}
            equity_curve.append((execution_bar.timestamp, portfolio.total_value(mark_prices)))
            if self.db:
                self.db.save_portfolio_snapshot(portfolio.snapshot(execution_bar.timestamp, mark_prices))

            # 4) Buy-and-hold benchmark, using the same executor/costs.
            if not benchmark_entered:
                bench_order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=self._max_affordable_quantity(
                        execution_bar.open, benchmark_portfolio.cash, self.execution_config
                    ),
                    order_type=OrderType.MARKET,
                    timestamp=execution_bar.timestamp,
                    reason="buy_and_hold_benchmark",
                )
                if bench_order.quantity > 0:
                    fill = self.executor.execute(bench_order, execution_bar.open, execution_bar.timestamp)
                    benchmark_portfolio.apply_fill(fill)
                benchmark_entered = True
            benchmark_curve.append((execution_bar.timestamp, benchmark_portfolio.total_value(mark_prices)))

        return self._build_result(symbol, portfolio, equity_curve, benchmark_curve, rejected_orders)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _max_affordable_quantity(price: float, cash: float, execution_config: ExecutionConfig) -> int:
        """Leave enough headroom that the resulting BUY fill - after
        slippage and commission are applied by PaperExecutor - still fits
        within `cash`, instead of sizing against the raw quote price."""
        if price <= 0:
            return 0
        import math

        effective_price = price * (1 + execution_config.slippage_bps / 10_000.0) * (1 + execution_config.commission_pct)
        usable_cash = cash - execution_config.commission_per_trade
        if usable_cash <= 0 or effective_price <= 0:
            return 0
        return max(0, math.floor(usable_cash / effective_price))

    def _build_result(
        self,
        symbol: str,
        portfolio: Portfolio,
        equity_curve: List[Tuple[datetime, float]],
        benchmark_curve: List[Tuple[datetime, float]],
        rejected_orders: List[dict],
    ) -> BacktestResult:
        ending_balance = equity_curve[-1][1] if equity_curve else portfolio.starting_cash
        total_return_pct = (ending_balance - portfolio.starting_cash) / portfolio.starting_cash

        benchmark_ending = benchmark_curve[-1][1] if benchmark_curve else portfolio.starting_cash
        benchmark_return_pct = (benchmark_ending - portfolio.starting_cash) / portfolio.starting_cash

        completed_trades = [t for t in portfolio.trade_log if t["side"] == "SELL"]
        winning = sum(1 for t in completed_trades if (t["realized_pnl"] or 0) > 0)
        losing = sum(1 for t in completed_trades if (t["realized_pnl"] or 0) <= 0)
        num_trades = len(completed_trades)
        win_rate = winning / num_trades if num_trades > 0 else 0.0

        max_dd = _max_drawdown([v for _, v in equity_curve])
        sharpe = _sharpe_ratio([v for _, v in equity_curve])

        return BacktestResult(
            symbol=symbol,
            starting_balance=portfolio.starting_cash,
            ending_balance=ending_balance,
            total_return_pct=total_return_pct,
            num_trades=num_trades,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            benchmark_return_pct=benchmark_return_pct,
            trade_log=portfolio.trade_log,
            rejected_orders=rejected_orders,
        )


def _max_drawdown(values: List[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)
    return max_dd


def _sharpe_ratio(values: List[float], periods_per_year: int = 252) -> Optional[float]:
    if len(values) < 2:
        return None
    arr = np.array(values, dtype=float)
    returns = np.diff(arr) / arr[:-1]
    std = float(np.std(returns))
    if std == 0:
        return None
    mean = float(np.mean(returns))
    return (mean / std) * (periods_per_year ** 0.5)
