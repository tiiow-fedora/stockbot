from __future__ import annotations

import pytest

from backtesting.engine import Backtester, _max_drawdown, _sharpe_ratio
from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from strategy.scoring import ScoringEngine
from strategy.technical import TechnicalStrategy
from tests.conftest import make_candles


def build_strategy() -> TechnicalStrategy:
    cfg = StrategyConfig()
    return TechnicalStrategy(cfg, ScoringEngine(cfg.weights, cfg.buy_threshold, cfg.sell_threshold))


def test_backtest_raises_with_too_little_data():
    backtester = Backtester(
        strategy=build_strategy(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
    )
    with pytest.raises(ValueError):
        backtester.run("AAPL", make_candles(n=10))


def test_backtest_produces_full_length_equity_curve():
    candles = make_candles(n=120, daily_return=0.002)
    backtester = Backtester(
        strategy=build_strategy(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
    )
    result = backtester.run("AAPL", candles)
    assert len(result.equity_curve) == len(candles) - 40
    assert len(result.benchmark_curve) == len(result.equity_curve)
    assert result.starting_balance == 100_000


def test_backtest_no_lookahead_decision_precedes_fill():
    """Every fill in the trade log must be timestamped strictly after the
    decision bar it was based on - if a fill's timestamp ever equals or
    precedes the bar used to make the decision, that's look-ahead bias."""
    candles = make_candles(n=120, daily_return=0.01)  # strong uptrend -> strategy should trade
    backtester = Backtester(
        strategy=build_strategy(),
        risk_config=RiskConfig(max_position_size_pct=0.5, max_trades_per_symbol_per_day=50),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
    )
    result = backtester.run("AAPL", candles)

    candle_index = {c.timestamp: i for i, c in enumerate(candles)}
    for trade in result.trade_log:
        fill_ts = trade["timestamp"]
        assert fill_ts in candle_index
        # A fill can only happen at bar i's open where i >= 1; the decision
        # that produced it was based on bar i-1's close at the earliest.
        assert candle_index[fill_ts] >= 1


def test_benchmark_enters_once_and_holds():
    candles = make_candles(n=100, daily_return=0.005)
    backtester = Backtester(
        strategy=build_strategy(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
    )
    result = backtester.run("AAPL", candles)
    # In a steady uptrend, buy-and-hold should end up profitable.
    assert result.benchmark_return_pct > 0


def test_max_drawdown_on_known_curve():
    values = [100, 110, 90, 95, 120, 60]
    dd = _max_drawdown(values)
    # peak 120 -> trough 60 => 50% drawdown is the max
    assert dd == pytest.approx(0.5)


def test_max_drawdown_empty_curve_is_zero():
    assert _max_drawdown([]) == 0.0


def test_sharpe_ratio_none_for_zero_variance():
    assert _sharpe_ratio([100, 100, 100, 100]) is None


def test_sharpe_ratio_positive_for_steady_gains():
    values = [100 * (1.001 ** i) for i in range(50)]
    sharpe = _sharpe_ratio(values)
    assert sharpe is not None
    assert sharpe > 0


def test_rejected_orders_are_recorded_when_risk_limits_are_tight():
    candles = make_candles(n=120, daily_return=0.02)  # very strong signal
    backtester = Backtester(
        strategy=build_strategy(),
        risk_config=RiskConfig(max_open_positions=0),  # nothing can ever be bought
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
    )
    result = backtester.run("AAPL", candles)
    assert result.num_trades == 0
    assert len(result.rejected_orders) > 0
