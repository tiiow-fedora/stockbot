"""Harness: run one strategy over many symbols/scenarios and aggregate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from backtest.metrics import combine_equity_curves, compute_trade_stats, max_drawdown_detail
from backtest.report import RunReport, now_iso
from backtesting.engine import BacktestResult, Backtester
from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from core.models import Candle
from strategy.base import Strategy


@dataclass
class HarnessResult:
    report: RunReport
    results: Dict[str, BacktestResult]


class BacktestHarness:
    """Each series is simulated independently with the full starting balance
    (the engine is single-symbol); the aggregate treats them as equal-capital
    sleeves, so aggregate P&L is the sum of the sleeves' P&L."""

    def __init__(
        self,
        strategy: Strategy,
        strategy_name: str = "custom",
        risk_config: Optional[RiskConfig] = None,
        execution_config: Optional[ExecutionConfig] = None,
        starting_balance: float = 100_000.0,
        warmup_bars: int = 40,
    ):
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.risk_config = risk_config or RiskConfig()
        self.execution_config = execution_config or ExecutionConfig()
        self.starting_balance = starting_balance
        self.warmup_bars = warmup_bars

    def run(self, series: Dict[str, List[Candle]], name: str = "backtest") -> HarnessResult:
        if not series:
            raise ValueError("no series to backtest")
        results: Dict[str, BacktestResult] = {}
        for label, candles in series.items():
            # fresh Backtester (and RiskManager state) per series
            bt = Backtester(
                strategy=self.strategy,
                risk_config=self.risk_config,
                execution_config=self.execution_config,
                starting_balance=self.starting_balance,
                warmup_bars=self.warmup_bars,
            )
            symbol = candles[0].symbol if candles else label
            results[label] = bt.run(symbol, candles)
        return HarnessResult(report=self._aggregate(name, results), results=results)

    def _aggregate(self, name: str, results: Dict[str, BacktestResult]) -> RunReport:
        n = len(results)
        total_start = self.starting_balance * n
        total_end = sum(r.ending_balance for r in results.values())
        all_trades: List[dict] = []
        per_symbol = []
        for label, r in results.items():
            stats = compute_trade_stats(r.trade_log)
            dd = max_drawdown_detail(r.equity_curve)
            per_symbol.append(
                {
                    "label": label,
                    "symbol": r.symbol,
                    "pnl": r.ending_balance - r.starting_balance,
                    "return_pct": r.total_return_pct,
                    "benchmark_return_pct": r.benchmark_return_pct,
                    "max_drawdown_pct": dd["max_drawdown_pct"],
                    "win_rate": stats.win_rate,
                    "num_trades": stats.num_round_trips,
                    "profit_factor": stats.profit_factor,
                    "sharpe_ratio": r.sharpe_ratio,
                }
            )
            for t in r.trade_log:
                all_trades.append({**t, "label": label, "timestamp": t["timestamp"].isoformat()})
        all_trades.sort(key=lambda t: t["timestamp"])

        agg_stats = compute_trade_stats(all_trades)
        agg_curve = combine_equity_curves([r.equity_curve for r in results.values()])
        agg_dd = max_drawdown_detail(agg_curve)
        bench_end = sum(
            r.starting_balance * (1 + r.benchmark_return_pct) for r in results.values()
        )
        sharpes = [r.sharpe_ratio for r in results.values() if r.sharpe_ratio is not None]
        return RunReport(
            name=name,
            strategy=self.strategy_name,
            created_at=now_iso(),
            starting_balance=total_start,
            ending_balance=total_end,
            total_pnl=total_end - total_start,
            total_return_pct=(total_end - total_start) / total_start,
            benchmark_return_pct=(bench_end - total_start) / total_start,
            max_drawdown_pct=agg_dd["max_drawdown_pct"],
            max_drawdown_abs=agg_dd["max_drawdown_abs"],
            win_rate=agg_stats.win_rate,
            num_trades=agg_stats.num_round_trips,
            profit_factor=agg_stats.profit_factor,
            sharpe_ratio=(sum(sharpes) / len(sharpes)) if sharpes else None,
            per_symbol=per_symbol,
            equity_curve=[[t.isoformat(), v] for t, v in agg_curve],
            trades=all_trades,
        )
