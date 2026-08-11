from __future__ import annotations

import pytest

from backtesting.optimize import format_grid_search_table, grid_search
from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from tests.conftest import make_candles

SMALL_GRID = {
    "buy_threshold": [0.2, 0.3],
    "sell_threshold": [-0.2, -0.3],
    "sma_short_period": [5],
    "sma_long_period": [15, 20],
}


def test_grid_search_returns_one_result_per_valid_combination():
    candles = make_candles(n=80, daily_return=0.01)
    results = grid_search(
        symbol="AAPL",
        candles=candles,
        base_strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        param_grid=SMALL_GRID,
        warmup_bars=30,
    )
    # 2 * 2 * 1 * 2 = 8 combinations, all valid (short=5 < long in {15,20}).
    assert len(results) == 8


def test_grid_search_skips_invalid_short_ge_long_combos():
    candles = make_candles(n=80, daily_return=0.01)
    grid = {
        "buy_threshold": [0.25],
        "sell_threshold": [-0.25],
        "sma_short_period": [20],
        "sma_long_period": [10],  # short >= long everywhere - nothing should run
    }
    results = grid_search(
        symbol="AAPL",
        candles=candles,
        base_strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        param_grid=grid,
        warmup_bars=30,
    )
    assert results == []


def test_grid_search_sorted_descending_by_objective():
    candles = make_candles(n=80, daily_return=0.01)
    results = grid_search(
        symbol="AAPL",
        candles=candles,
        base_strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        param_grid=SMALL_GRID,
        warmup_bars=30,
        objective="total_return_pct",
    )
    returns = [gr.result.total_return_pct for gr in results]
    assert returns == sorted(returns, reverse=True)


def test_grid_search_each_combo_uses_the_requested_params():
    candles = make_candles(n=80, daily_return=0.01)
    results = grid_search(
        symbol="AAPL",
        candles=candles,
        base_strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        param_grid=SMALL_GRID,
        warmup_bars=30,
    )
    seen = {tuple(sorted(gr.params.items())) for gr in results}
    assert len(seen) == 8  # every combination is distinct and was actually applied


def test_grid_search_rejects_unknown_strategy_name():
    candles = make_candles(n=80, daily_return=0.01)
    with pytest.raises(ValueError):
        grid_search(
            symbol="AAPL",
            candles=candles,
            base_strategy_config=StrategyConfig(),
            risk_config=RiskConfig(),
            execution_config=ExecutionConfig(),
            param_grid=SMALL_GRID,
            strategy_name="not_a_real_strategy",
        )


def test_grid_search_rejects_unknown_objective():
    candles = make_candles(n=80, daily_return=0.01)
    with pytest.raises(ValueError):
        grid_search(
            symbol="AAPL",
            candles=candles,
            base_strategy_config=StrategyConfig(),
            risk_config=RiskConfig(),
            execution_config=ExecutionConfig(),
            param_grid=SMALL_GRID,
            objective="not_a_real_objective",
        )


def test_format_grid_search_table_includes_overfitting_warning():
    candles = make_candles(n=80, daily_return=0.01)
    results = grid_search(
        symbol="AAPL",
        candles=candles,
        base_strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        param_grid=SMALL_GRID,
        warmup_bars=30,
    )
    table = format_grid_search_table(results, top=3)
    assert "WARNING" in table
    assert "out-of-sample" in table


def test_format_grid_search_table_handles_no_results():
    assert "no valid parameter combinations" in format_grid_search_table([])
