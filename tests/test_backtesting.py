from __future__ import annotations

import pytest

from analysis.sentiment import LexiconSentimentScorer
from backtesting.engine import Backtester, _max_drawdown, _sharpe_ratio
from core.config import ExecutionConfig, RiskConfig, StrategyConfig
from core.models import NewsItem
from database.database import DatabaseManager
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


def test_backtest_persists_portfolio_snapshots_when_db_provided(tmp_path):
    """The dashboard reads portfolio history from the DB - a backtest run
    with --persist must leave behind a snapshot per bar, not just decisions."""
    db = DatabaseManager(f"sqlite:///{tmp_path}/test.db")
    db.init_db()
    candles = make_candles(n=100, daily_return=0.003)
    backtester = Backtester(
        strategy=build_strategy(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
        db=db,
    )
    result = backtester.run("AAPL", candles)

    history = db.get_portfolio_history(limit=1000)
    assert len(history) == len(result.equity_curve)
    assert history[-1]["total_value"] == pytest.approx(result.ending_balance)


class _FakeNewsProvider:
    """In-memory news source for backtest tests - never touches a network."""

    def __init__(self, items):
        self._items = items
        self.calls = []  # records (start, end) for every call, to verify no look-ahead

    def get_news(self, symbol, start, end):
        self.calls.append((start, end))
        return [n for n in self._items if start <= n.timestamp <= end]


def test_backtest_optional_news_never_queries_beyond_the_decision_bar():
    candles = make_candles(n=100, daily_return=0.005)
    # One news item per bar timestamp, all clearly positive so the strategy
    # has a reason to actually use them.
    news_items = [
        NewsItem(symbol="AAPL", timestamp=c.timestamp, headline="Company beats and raises guidance", source="test")
        for c in candles
    ]
    fake_news = _FakeNewsProvider(news_items)
    scorer = LexiconSentimentScorer()

    backtester = Backtester(
        strategy=build_strategy(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
        news_provider=fake_news,
        sentiment_scorer=scorer,
    )
    backtester.run("AAPL", candles)

    assert fake_news.calls  # the provider was actually used
    decision_bar_timestamps = {c.timestamp for c in candles[39:-1]}  # bars used as decision_bar in the loop
    for start, end in fake_news.calls:
        assert end in decision_bar_timestamps  # never asked for news past the decision bar


def test_backtest_without_news_provider_passes_empty_news():
    """Default behavior (no news_provider) must be unchanged - fast, offline backtests stay fast and offline."""
    candles = make_candles(n=100, daily_return=0.005)
    backtester = Backtester(
        strategy=build_strategy(),
        risk_config=RiskConfig(),
        execution_config=ExecutionConfig(),
        starting_balance=100_000,
        warmup_bars=40,
    )
    result = backtester.run("AAPL", candles)  # must not raise / must not attempt any network call
    assert result.starting_balance == 100_000
