from __future__ import annotations

import pytest

from core.config import StrategyWeights
from core.models import Action
from strategy.scoring import ScoringEngine, clip


def test_clip_bounds():
    assert clip(5.0) == 1.0
    assert clip(-5.0) == -1.0
    assert clip(0.3) == pytest.approx(0.3)


def test_score_all_bullish_components_gives_positive_composite():
    engine = ScoringEngine(weights=StrategyWeights())
    breakdown = engine.score(technical=1.0, momentum=1.0, volume=1.0, risk=0.0, sentiment=None)
    assert breakdown.composite > 0
    assert engine.decide(breakdown.composite) == Action.BUY


def test_score_all_bearish_components_gives_negative_composite():
    engine = ScoringEngine(weights=StrategyWeights())
    breakdown = engine.score(technical=-1.0, momentum=-1.0, volume=-1.0, risk=-1.0, sentiment=None)
    assert breakdown.composite < 0
    assert engine.decide(breakdown.composite) == Action.SELL


def test_score_neutral_components_gives_hold():
    engine = ScoringEngine(weights=StrategyWeights())
    breakdown = engine.score(technical=0.0, momentum=0.0, volume=0.0, risk=0.0, sentiment=None)
    assert breakdown.composite == pytest.approx(0.0)
    assert engine.decide(breakdown.composite) == Action.HOLD


def test_missing_sentiment_redistributes_weight_not_zeroes_it():
    weights = StrategyWeights(technical=0.25, momentum=0.25, volume=0.25, risk=0.15, sentiment=0.10)
    engine = ScoringEngine(weights=weights)

    with_sentiment = engine.score(technical=1.0, momentum=0.0, volume=0.0, risk=0.0, sentiment=1.0)
    without_sentiment = engine.score(technical=1.0, momentum=0.0, volume=0.0, risk=0.0, sentiment=None)

    # Without news, technical's effective weight should be higher (weight
    # redistributed), not the same absolute contribution as when sentiment
    # is present and also bullish.
    assert without_sentiment.weights["technical"] > with_sentiment.weights["technical"]
    assert without_sentiment.sentiment is None


def test_risk_score_only_ever_pulls_composite_down():
    engine = ScoringEngine(weights=StrategyWeights())
    with_risk = engine.score(technical=0.5, momentum=0.5, volume=0.5, risk=-1.0, sentiment=None)
    without_risk = engine.score(technical=0.5, momentum=0.5, volume=0.5, risk=0.0, sentiment=None)
    assert with_risk.composite < without_risk.composite


def test_buy_sell_thresholds_are_configurable():
    engine = ScoringEngine(weights=StrategyWeights(), buy_threshold=0.9, sell_threshold=-0.9)
    breakdown = engine.score(technical=0.5, momentum=0.5, volume=0.5, risk=0.0, sentiment=None)
    # Would have been a BUY with default thresholds, but not with a strict 0.9 bar.
    assert engine.decide(breakdown.composite) == Action.HOLD
