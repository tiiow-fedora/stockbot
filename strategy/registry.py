"""
Single source of truth for "which strategies exist".

Used by main.py's build_strategy() factory, and by
backtesting/compare.py and backtesting/optimize.py to enumerate every
registered strategy for comparison/parameter sweeps. Adding a new
strategy means adding one entry here - nothing else needs to change.
"""

from __future__ import annotations

from typing import Callable, Dict

from core.config import StrategyConfig
from strategy.base import Strategy
from strategy.mean_reversion import MeanReversionStrategy
from strategy.scoring import ScoringEngine
from strategy.technical import TechnicalStrategy


def _build_technical(config: StrategyConfig) -> Strategy:
    engine = ScoringEngine(config.weights, config.buy_threshold, config.sell_threshold)
    return TechnicalStrategy(config, engine)


def _build_mean_reversion(config: StrategyConfig) -> Strategy:
    engine = ScoringEngine(config.weights, config.buy_threshold, config.sell_threshold)
    return MeanReversionStrategy(config, engine)


STRATEGY_BUILDERS: Dict[str, Callable[[StrategyConfig], Strategy]] = {
    "technical": _build_technical,
    "mean_reversion": _build_mean_reversion,
}


def build_strategy_by_name(name: str, config: StrategyConfig) -> Strategy:
    if name not in STRATEGY_BUILDERS:
        raise ValueError(f"unknown strategy: {name!r}. Available: {sorted(STRATEGY_BUILDERS)}")
    return STRATEGY_BUILDERS[name](config)
