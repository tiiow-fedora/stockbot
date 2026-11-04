"""
Backtest harness (additive layer over `backtesting.engine.Backtester`).

Replays stored (CSV) or synthetic scenario bars through the existing
`Strategy` interface and the existing no-lookahead simulation engine, then
reports P&L, drawdown, win rate and a few extra trade statistics.

Paper simulation only: nothing here imports a broker, reads credentials,
or places orders anywhere but the in-memory `Portfolio`.
"""

from backtest.metrics import TradeStats, compute_trade_stats, max_drawdown_detail
from backtest.report import RunReport, load_report, save_report
from backtest.runner import BacktestHarness, HarnessResult
from backtest.scenarios import SCENARIOS, generate_scenario

__all__ = [
    "BacktestHarness",
    "HarnessResult",
    "RunReport",
    "SCENARIOS",
    "TradeStats",
    "compute_trade_stats",
    "generate_scenario",
    "load_report",
    "max_drawdown_detail",
    "save_report",
]
