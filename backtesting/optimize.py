"""
Parameter grid search: sweep a handful of StrategyConfig fields against
one symbol's historical data and rank the resulting backtests.

This is a research tool, not a promise. Every combination here is fit to
the exact historical window it was tested on - the classic overfitting
trap. A parameter set that wins this sweep is a hypothesis to validate
on a DIFFERENT time period (walk-forward / out-of-sample testing), not a
result to trade on directly. `run_grid_search` prints that caveat every
time for exactly this reason.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backtesting.engine import BacktestResult, Backtester
from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from strategy.registry import STRATEGY_BUILDERS

# Only top-level scalar StrategyConfig fields are swept (not the nested
# `weights`) - enough to demonstrate parameter sensitivity without a
# combinatorial explosion or a much more complex config-diffing mechanism.
DEFAULT_PARAM_GRID: Dict[str, Sequence[Any]] = {
    "buy_threshold": [0.15, 0.25, 0.35],
    "sell_threshold": [-0.15, -0.25, -0.35],
    "sma_short_period": [5, 10, 15],
    "sma_long_period": [20, 30, 40],
}


@dataclass
class GridSearchResult:
    params: Dict[str, Any]
    result: BacktestResult


def grid_search(
    symbol: str,
    candles: list,
    base_strategy_config: StrategyConfig,
    risk_config: RiskConfig,
    execution_config: ExecutionConfig,
    param_grid: Optional[Dict[str, Sequence[Any]]] = None,
    strategy_name: str = "technical",
    starting_balance: float = 100_000.0,
    warmup_bars: int = 40,
    objective: str = "total_return_pct",
) -> List[GridSearchResult]:
    if strategy_name not in STRATEGY_BUILDERS:
        raise ValueError(f"unknown strategy: {strategy_name!r}")
    if objective not in ("total_return_pct", "sharpe_ratio"):
        raise ValueError(f"unknown objective: {objective!r}")

    grid = param_grid or DEFAULT_PARAM_GRID
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]

    results: List[GridSearchResult] = []
    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        if params.get("sma_short_period", 0) >= params.get("sma_long_period", float("inf")):
            continue  # a short SMA that isn't shorter than the long one is not a meaningful combo

        strategy_config = base_strategy_config.model_copy(update=params)
        strategy = STRATEGY_BUILDERS[strategy_name](strategy_config)
        backtester = Backtester(
            strategy=strategy,
            risk_config=risk_config,
            execution_config=execution_config,
            starting_balance=starting_balance,
            warmup_bars=warmup_bars,
        )
        try:
            result = backtester.run(symbol, candles)
        except ValueError:
            continue  # not enough candles for this warmup - shouldn't happen if the caller sized candles right

        results.append(GridSearchResult(params=params, result=result))

    def sort_key(gr: GridSearchResult) -> float:
        value = getattr(gr.result, objective)
        return value if value is not None else float("-inf")

    results.sort(key=sort_key, reverse=True)
    return results


def format_grid_search_table(results: List[GridSearchResult], top: int = 5, objective: str = "total_return_pct") -> str:
    if not results:
        return "(no valid parameter combinations produced a result)"

    top_results = results[:top]
    param_keys = list(top_results[0].params.keys())
    headers = param_keys + ["return", "benchmark", "trades", "win rate", "max dd", "sharpe"]
    rows = [headers]
    for gr in top_results:
        r = gr.result
        rows.append(
            [str(gr.params[k]) for k in param_keys]
            + [
                f"{r.total_return_pct:+.2%}",
                f"{r.benchmark_return_pct:+.2%}",
                str(r.num_trades),
                f"{r.win_rate:.1%}",
                f"{r.max_drawdown_pct:.2%}",
                "n/a" if r.sharpe_ratio is None else f"{r.sharpe_ratio:.2f}",
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = []
    for i, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))

    lines.append(f"\n{len(results)} valid combinations tested, ranked by {objective}.")
    lines.append(
        "WARNING: these parameters are fit to this exact historical window. "
        "Before trusting them, re-test on a DIFFERENT time period (walk-forward / "
        "out-of-sample validation) - a top grid-search result is a hypothesis, not a result."
    )
    return "\n".join(lines)
