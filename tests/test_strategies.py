from __future__ import annotations

import pytest

from core.config import StrategyConfig
from core.models import Action, MarketSnapshot
from strategy.mean_reversion import MeanReversionStrategy
from strategy.registry import STRATEGY_BUILDERS, build_strategy_by_name
from strategy.scoring import ScoringEngine
from strategy.technical import TechnicalStrategy
from tests.conftest import make_candles


def build(strategy_cls, cfg: StrategyConfig):
    engine = ScoringEngine(cfg.weights, cfg.buy_threshold, cfg.sell_threshold)
    return strategy_cls(cfg, engine)


def test_technical_and_mean_reversion_react_oppositely_to_the_same_sma_divergence():
    """The whole point of a second strategy: given the same short-SMA-
    below-long-SMA / price-below-long-SMA situation, trend-following
    (technical) reads it as bearish and mean-reversion (contrarian) reads
    it as an oversold buy signal. RSI is excluded (set to None) here
    because it's identical reversal-style logic in both strategies - the
    real differentiator is the SMA-based component, isolated below."""
    cfg = StrategyConfig()
    technical = build(TechnicalStrategy, cfg)
    mean_reversion = build(MeanReversionStrategy, cfg)

    # price and sma_short both sitting well below sma_long: a "death
    # cross" style downtrend from the technical side.
    ind = {"price": 80.0, "sma_short": 78.0, "sma_long": 100.0, "rsi": None, "macd_histogram": None}

    tech_score = technical._technical_score(ind)
    mr_score = mean_reversion._contrarian_technical_score(ind)

    assert tech_score < 0  # trend-following reads the decline as bearish
    assert mr_score > 0  # mean-reversion reads the same decline as an oversold buy signal


def test_mean_reversion_holds_with_insufficient_history():
    cfg = StrategyConfig()
    strategy = build(MeanReversionStrategy, cfg)
    candles = make_candles(n=5)
    snapshot = MarketSnapshot(
        symbol="AAPL", timestamp=candles[-1].timestamp, price=candles[-1].close, volume=candles[-1].volume
    )
    decision = strategy.generate_decision(
        symbol="AAPL",
        snapshot=snapshot,
        history=candles[:-1],
        news=[],
        current_position_qty=0.0,
        cash=100_000.0,
        portfolio_value=100_000.0,
    )
    assert decision.action == Action.HOLD
    assert "insufficient history" in decision.reason


def test_mean_reversion_decision_is_reproducible():
    cfg = StrategyConfig()
    strategy = build(MeanReversionStrategy, cfg)
    candles = make_candles(n=80, daily_return=0.01)
    snapshot = MarketSnapshot(
        symbol="AAPL", timestamp=candles[-1].timestamp, price=candles[-1].close, volume=candles[-1].volume
    )
    kwargs = dict(
        symbol="AAPL",
        snapshot=snapshot,
        history=candles[:-1],
        news=[],
        current_position_qty=0.0,
        cash=100_000.0,
        portfolio_value=100_000.0,
    )
    d1 = strategy.generate_decision(**kwargs)
    d2 = strategy.generate_decision(**kwargs)
    assert d1.action == d2.action
    assert d1.scores.composite == pytest.approx(d2.scores.composite)


def test_registry_contains_both_strategies():
    assert "technical" in STRATEGY_BUILDERS
    assert "mean_reversion" in STRATEGY_BUILDERS


def test_registry_builds_technical_strategy():
    strategy = build_strategy_by_name("technical", StrategyConfig())
    assert isinstance(strategy, TechnicalStrategy)


def test_registry_builds_mean_reversion_strategy():
    strategy = build_strategy_by_name("mean_reversion", StrategyConfig())
    assert isinstance(strategy, MeanReversionStrategy)


def test_registry_rejects_unknown_strategy_name():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy_by_name("not_a_real_strategy", StrategyConfig())
