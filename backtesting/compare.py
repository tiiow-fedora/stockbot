"""
Strategy comparison: run every registered strategy against the exact
same historical candles and report results side by side.

The point is the same one Backtester.summary() makes for a single
strategy vs. buy-and-hold: don't assume a strategy adds value - measure
it, on the same data, under the same risk/execution assumptions, and let
the numbers say so (or not).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from backtesting.engine import BacktestResult, Backtester
from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from database.database import DatabaseManager
from strategy.registry import STRATEGY_BUILDERS


def compare_strategies(
    symbol: str,
    candles: list,
    strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    execution_config: ExecutionConfig,
    starting_balance: float = 100_000.0,
    warmup_bars: int = 40,
    db: Optional[DatabaseManager] = None,
    strategy_names: Optional[List[str]] = None,
) -> Dict[str, BacktestResult]:
    """Every strategy runs with the same `strategy_config` (indicator
    periods, thresholds, weights) - only the decision logic differs. Each
    gets a completely fresh Portfolio/RiskManager/PaperExecutor internally
    (Backtester.run() already guarantees that), so results are directly
    comparable and one strategy's trades can't affect another's."""
    names = strategy_names or sorted(STRATEGY_BUILDERS)
    results: Dict[str, BacktestResult] = {}
    for name in names:
        strategy = STRATEGY_BUILDERS[name](strategy_config)
        backtester = Backtester(
            strategy=strategy,
            risk_config=risk_config,
            execution_config=execution_config,
            starting_balance=starting_balance,
            warmup_bars=warmup_bars,
            db=db,
        )
        results[name] = backtester.run(symbol, candles)
    return results


def format_comparison_table(results: Dict[str, BacktestResult]) -> str:
    if not results:
        return "(no results)"

    headers = ["strategy", "return", "vs benchmark", "trades", "win rate", "max dd", "sharpe"]
    rows = [headers]
    benchmark_pct = next(iter(results.values())).benchmark_return_pct
    for name, r in results.items():
        rows.append(
            [
                name,
                f"{r.total_return_pct:+.2%}",
                f"{r.total_return_pct - r.benchmark_return_pct:+.2%}",
                str(r.num_trades),
                f"{r.win_rate:.1%}",
                f"{r.max_drawdown_pct:.2%}",
                "n/a" if r.sharpe_ratio is None else f"{r.sharpe_ratio:.2f}",
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for i, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row))
        lines.append(line)
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    lines.append(f"\nbuy-and-hold benchmark: {benchmark_pct:+.2%}")
    return "\n".join(lines)
