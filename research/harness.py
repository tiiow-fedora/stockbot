"""
Glue between research strategies and the engine's ``Backtester``.

``run_backtest`` runs one strategy over one candle series through the REAL
engine (``backtesting.engine.Backtester``: next-bar-open fills, slippage,
RiskManager sizing, stop-loss/take-profit) and returns the engine's
``BacktestResult`` together with a ``research.metrics.PerformanceReport`` that
includes exposure (reconstructed from the trade log) - something the stock
``BacktestResult`` does not provide.

Everything is paper trading: ``PaperExecutor`` only simulates fills.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from backtesting.engine import Backtester, BacktestResult
from core.config import ExecutionConfig, RiskConfig
from core.models import Candle
from research import metrics
from strategy.base import Strategy

#: Research risk settings: one symbol, so let the strategy deploy most of the account
#: (the production default of 20% per name would make every strategy look ~80% cash).
#: Stop-loss / take-profit and the per-day trade cap are kept loose so the *strategy's*
#: signals, not the risk overlay, drive results. Use ``production_risk()`` to see the
#: engine's real defaults.
RESEARCH_RISK = RiskConfig(
    max_position_size_pct=0.95,
    max_portfolio_exposure_pct=0.95,
    max_open_positions=8,
    max_trades_per_symbol_per_day=3,
    stop_loss_pct=0.50,
    take_profit_pct=10.0,
    min_cash_reserve_pct=0.02,
)

#: Mildly pessimistic costs: 5 bps slippage per fill (engine default), no commission.
DEFAULT_EXECUTION = ExecutionConfig(mode="paper", slippage_bps=5.0, commission_per_trade=0.0, commission_pct=0.0)


@contextlib.contextmanager
def quiet_engine(level: int = logging.WARNING):
    """The engine logs every simulated fill at INFO; across thousands of
    backtests that drowns everything. Silence the ``stockbot`` logger tree."""
    lg = logging.getLogger("stockbot")
    old = lg.level
    lg.setLevel(level)
    try:
        yield
    finally:
        lg.setLevel(old)


def production_risk() -> RiskConfig:
    """The engine's shipped RiskConfig defaults (20% max position, 8% stop, 20% target)."""
    return RiskConfig()


@dataclass
class RunResult:
    symbol: str
    result: BacktestResult
    report: metrics.PerformanceReport
    equity: np.ndarray
    position_value: np.ndarray
    benchmark_report: metrics.PerformanceReport

    @property
    def sharpe(self) -> float:
        return self.report.sharpe


def position_value_series(result: BacktestResult, candles: Sequence[Candle]) -> np.ndarray:
    """Dollar value of the open position at each equity-curve timestamp,
    reconstructed by replaying the trade log against bar closes."""
    close_at = {c.timestamp: c.close for c in candles}
    fills = sorted(result.trade_log, key=lambda t: t["timestamp"])
    qty, k, out = 0.0, 0, []
    for ts, _ in result.equity_curve:
        while k < len(fills) and fills[k]["timestamp"] <= ts:
            f = fills[k]
            qty += f["quantity"] if f["side"] == "BUY" else -f["quantity"]
            k += 1
        out.append(max(qty, 0.0) * close_at.get(ts, 0.0))
    return np.array(out, dtype=float)


def trade_pnls(result: BacktestResult) -> List[float]:
    """Realised P&L of each closed round trip (one per SELL fill)."""
    return [t["realized_pnl"] for t in result.trade_log if t["side"] == "SELL" and t["realized_pnl"] is not None]


def run_backtest(
    strategy: Strategy,
    candles: Sequence[Candle],
    *,
    risk: Optional[RiskConfig] = None,
    execution: Optional[ExecutionConfig] = None,
    starting_balance: float = 100_000.0,
    warmup_bars: int = 40,
    periods_per_year: int = 252,
) -> RunResult:
    """Backtest ``strategy`` on ``candles`` through the real engine and attach metrics."""
    if not candles:
        raise ValueError("no candles")
    symbol = candles[0].symbol
    bt = Backtester(
        strategy,
        risk or RESEARCH_RISK,
        execution or DEFAULT_EXECUTION,
        starting_balance=starting_balance,
        warmup_bars=warmup_bars,
    )
    with quiet_engine():
        res = bt.run(symbol, list(candles))
    equity = np.array([v for _, v in res.equity_curve], dtype=float)
    pv = position_value_series(res, candles)
    report = metrics.compute_report(equity, pv, trade_pnls(res), periods_per_year)
    bench = np.array([v for _, v in res.benchmark_curve], dtype=float)
    bench_report = metrics.compute_report(bench, bench, None, periods_per_year)
    return RunResult(symbol, res, report, equity, pv, bench_report)


def compare(
    strategies: Dict[str, Strategy],
    datasets: Dict[str, Sequence[Candle]],
    **kwargs,
) -> Dict[str, Dict[str, RunResult]]:
    """Run every strategy over every dataset. Returns ``{strategy: {dataset: RunResult}}``."""
    return {
        sname: {dname: run_backtest(strat, candles, **kwargs) for dname, candles in datasets.items()}
        for sname, strat in strategies.items()
    }
