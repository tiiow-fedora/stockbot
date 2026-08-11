from __future__ import annotations

import pytest

from backtesting.compare import compare_strategies, format_comparison_table
from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from tests.conftest import make_candles


def test_compare_strategies_runs_every_registered_strategy():
    candles = make_candles(n=100, daily_return=0.01)
    results = compare_strategies(
        symbol="AAPL",
        candles=candles,
        strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
    )
    assert set(results.keys()) == {"technical", "mean_reversion"}
    for result in results.values():
        assert result.starting_balance == 100_000


def test_compare_strategies_uses_independent_portfolios():
    """A trade in one strategy's backtest must never leak into another's -
    each should start from the same clean starting_balance."""
    candles = make_candles(n=100, daily_return=0.015)
    results = compare_strategies(
        symbol="AAPL",
        candles=candles,
        strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=50_000,
        warmup_bars=40,
    )
    for result in results.values():
        assert result.equity_curve[0][1] != pytest.approx(0.0)
        assert result.starting_balance == 50_000


def test_compare_strategies_respects_strategy_names_filter():
    candles = make_candles(n=100, daily_return=0.01)
    results = compare_strategies(
        symbol="AAPL",
        candles=candles,
        strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
        strategy_names=["technical"],
    )
    assert set(results.keys()) == {"technical"}


def test_format_comparison_table_includes_all_strategies_and_benchmark():
    candles = make_candles(n=100, daily_return=0.01)
    results = compare_strategies(
        symbol="AAPL",
        candles=candles,
        strategy_config=StrategyConfig(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
    )
    table = format_comparison_table(results)
    assert "technical" in table
    assert "mean_reversion" in table
    assert "buy-and-hold benchmark" in table


def test_format_comparison_table_handles_empty_results():
    assert format_comparison_table({}) == "(no results)"
